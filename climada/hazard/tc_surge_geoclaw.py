"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Inundation from TC storm surges, modeled using the library GeoClaw

This module requires a Fortran compiler (such as gfortran) to run!
"""

import __main__
import contextlib
import datetime as dt
import importlib
import logging
import re
import os
import site
import subprocess
import sys
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

from climada.hazard import Centroids, Hazard, Tag as TagHazard, TropCyclone
from climada.hazard.tc_tracks import estimate_rmw, estimate_roci
from climada.util import DATA_DIR, ureg
import climada.util.coordinates as coord_util

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TCSurgeGeoClaw'
"""Hazard type acronym for this module."""

CLAWPACK_GIT_URL = "https://github.com/clawpack/clawpack.git"
"""URL of the official Clawpack git repository."""

CLAWPACK_VERSION = "v5.7.1"
"""Version or git decorator (tag, branch) of Clawpack to use."""

CLAWPACK_SRC_DIR = os.path.join(DATA_DIR, "geoclaw", "src")
"""Directory for Clawpack source checkouts (if it doesn't exist)"""

GEOCLAW_WORK_DIR = os.path.join(DATA_DIR, "geoclaw", "runs")
"""Base directory for GeoClaw run data."""

INLAND_MAX_DIST_KM = 50
"""Maximum inland distance of the centroids in km."""

OFFSHORE_MAX_DIST_KM = 10
"""Maximum offshore distance of the centroids in km."""

MAX_LATITUDE = 61
"""Maximum latitude of potentially affected centroids."""

CENTR_NODE_MAX_DIST_DEG = 5.5
"""Maximum distance between centroid and TC track node in degrees."""

KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude
MBAR_TO_PA = (1.0 * ureg.mbar).to(ureg.pascal).magnitude
KM_TO_DEG = 1.0 / (60 * NM_TO_KM)
"""Unit conversion factors."""


class TCSurgeGeoClaw(Hazard):
    """TC storm surge heights in m, modeled using GeoClaw.

    Attributes
    ----------
    gauge_data : list of lists of dicts
        For each storm and each gauge, a dict containing the `location` of the gauge, and
        (for each surge event) `base_sea_level`, `topo_height`, `time` and `height_above_geoid`
        information.
    """
    def __init__(self):
        Hazard.__init__(self, HAZ_TYPE)


    @staticmethod
    def from_tc_tracks(tracks, zos_path, topo_path, centroids=None, description='', gauges=None):
        """New instance with surge inundation from TCTracks objects.

        Parameters
        ----------
        tracks : TCTracks
            tracks of events
        zos_path : str
            Path to NetCDF file containing gridded monthly sea level data.
        topo_path : str
            Path to raster file containing gridded elevation data.
        centroids : Centroids, optional
            Centroids where to model TC. Default: global centroids.
        description : str, optional
            description of the events
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes.
        """
        if tracks.size == 0:
            raise ValueError("The given TCTracks object doesn't contain any tracks.")
        setup_clawpack()

        if centroids is None:
            # cell-centered grid within padded bounds at 30 arc-seconds resolution
            res_deg = 30 / (60 * 60)
            bounds = tracks.get_bounds(deg_buffer=CENTR_NODE_MAX_DIST_DEG)
            lat = np.arange(bounds[1] + 0.5 * res_deg, bounds[3], res_deg)
            lon = np.arange(bounds[0] + 0.5 * res_deg, bounds[2], res_deg)
            lon, lat = [ar.ravel() for ar in np.meshgrid(lon, lat)]
            centroids = Centroids()
            centroids.set_lat_lon(lat, lon)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
        if not centroids.dist_coast.size or np.all(centroids.dist_coast >= 0):
            centroids.set_dist_coast(signed=True, precomputed=True)
        coastal_msk = (centroids.dist_coast <= OFFSHORE_MAX_DIST_KM * 1000)
        coastal_msk &= (centroids.dist_coast >= -INLAND_MAX_DIST_KM * 1000)
        coastal_msk &= (np.abs(centroids.lat) <= MAX_LATITUDE)
        coastal_idx = coastal_msk.nonzero()[0]

        LOGGER.info('Computing TC surge of %s tracks on %s centroids.',
                    str(tracks.size), str(coastal_idx.size))
        haz = TCSurgeGeoClaw()
        haz.concatenate(
            [TCSurgeGeoClaw.from_xr_track(t, centroids, coastal_idx, zos_path, topo_path,
                                          gauges=gauges)
             for t in tracks.data])
        TropCyclone.frequency_from_tracks(haz, tracks.data)
        haz.tag.description = description
        return haz


    @staticmethod
    def from_xr_track(track, centroids, coastal_idx, zos_path, topo_path, gauges=None):
        """Generate TC surge hazard from a single xarray track dataset

        Parameters
        ----------
        track : xr.Dataset
            single tropical cyclone track.
        centroids : Centroids
            Centroids instance.
        coastal_idx : np.array
            Indices of centroids close to coast.
        zos_path : str
            Path to NetCDF file containing gridded monthly sea level data.
        topo_path : str
            Path to raster file containing gridded elevation data.
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes.

        Returns
        -------
        haz : TCSurgeGeoClaw object
        """
        coastal_centroids = centroids.coord[coastal_idx]
        intensity = np.zeros(centroids.coord.shape[0])
        intensity[coastal_idx], gauge_data = geoclaw_surge_from_track(track, coastal_centroids,
                                                                      zos_path, topo_path,
                                                                      gauges=gauges)

        new_haz = TCSurgeGeoClaw()
        new_haz.tag = TagHazard(HAZ_TYPE, 'Name: ' + track.name)
        new_haz.intensity = sp.csr_matrix(intensity)
        new_haz.gauge_data = [gauge_data]
        new_haz.units = 'm'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.sid]
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1)
        new_haz.date = np.array([
            dt.datetime(track.time.dt.year[0],
                        track.time.dt.month[0],
                        track.time.dt.day[0]).toordinal()
        ])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        new_haz.basin = [track.basin]
        return new_haz


def geoclaw_surge_from_track(track, centroids, zos_path, topo_path, gauges=None):
    """Compute TC surge height on centroids from a single track dataset

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : 2d np.array
        Points for which to record the maximum height of inundation. Each row is a lat-lon point.
    zos_path : str
        Path to NetCDF file containing gridded monthly sea level data.
    topo_path : str
        Path to raster file containing gridded elevation data.
    gauges : list of pairs (lat, lon), optional
        The locations of tide gauges where to measure temporal changes in sea level height.
        This is used mostly for validation purposes.

    Returns
    -------
    intensity : np.array
        Surge height in meters.
    gauge_data : list of dicts
        For each gauge, a dict containing the `location` of the gauge, and (for each surge event)
        `base_sea_level`, `topo_height`, `time` and `height_above_geoid` information.
    """
    gauges = [] if gauges is None else gauges

    # initialize intensity
    intensity = np.zeros(centroids.shape[0])

    # normalize longitudes of centroids and track
    track_bounds = coord_util.latlon_bounds(track.lat.values, track.lon.values)
    mid_lon = 0.5 * (track_bounds[0] + track_bounds[2])
    track['lon'][:] = coord_util.lon_normalize(track.lon.values, center=mid_lon)
    centroids[:, 1] = coord_util.lon_normalize(centroids[:, 1], center=mid_lon)

    # restrict to centroids in rectangular bounding box around track
    track_bounds_pad = np.array(track_bounds)
    track_bounds_pad[:2] -= CENTR_NODE_MAX_DIST_DEG
    track_bounds_pad[2:] += CENTR_NODE_MAX_DIST_DEG
    track_centr_msk = (track_bounds_pad[1] <= centroids[:, 0])
    track_centr_msk &= (centroids[:, 0] <= track_bounds_pad[3])
    track_centr_msk &= (track_bounds_pad[0] <= centroids[:, 1])
    track_centr_msk &= (centroids[:, 1] <= track_bounds_pad[2])
    track_centr_idx = track_centr_msk.nonzero()[0]

    # exclude centroids at too low/high topographic altitude
    centroids_height = coord_util.read_raster_sample(
        topo_path, centroids[track_centr_msk, 0], centroids[track_centr_msk, 1],
        intermediate_res=0.008)
    track_centr_idx = track_centr_idx[(centroids_height > -10) & (centroids_height < 10)]
    track_centr_msk.fill(False)
    track_centr_msk[track_centr_idx] = True
    track_centr = centroids[track_centr_msk]

    if track_centr.shape[0] == 0:
        LOGGER.info("No centroids within reach of this storm track.")
        return intensity

    # make sure that radius information is available
    if 'radius_oci' not in track.coords:
        track['radius_oci'] = xr.zeros_like(track['radius_max_wind'])
    track['radius_max_wind'][:] = estimate_rmw(track.radius_max_wind.values,
                                               track.central_pressure.values)
    track['radius_oci'][:] = estimate_roci(track.radius_oci.values, track.central_pressure.values)
    track['radius_oci'][:] = np.fmax(track.radius_max_wind.values, track.radius_oci.values)

    # create work directory
    work_dir = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + f"-{track.sid}"
    work_dir = os.path.join(GEOCLAW_WORK_DIR, work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # get landfall events
    LOGGER.info("Determine georegions and temporal periods of landfall events...")
    events = TCSurgeEvents(track, track_centr)
    events.plot_areas(path=os.path.join(work_dir, "event_areas.pdf"))


    gauge_data = [{'location': g, 'base_sea_level': [], 'topo_height': [],
                   'time': [], 'height_above_geoid': []} for g in gauges]
    if len(events) == 0:
        LOGGER.info("This storm doesn't affect any coastal areas.")
    else:
        LOGGER.info("Starting %d runs of GeoClaw...", len(events))
        surge_h = []
        for event in events:
            event_surge_h = np.zeros(track_centr.shape[0])
            event_track = track.sel(time=event['time_mask_buffered'])
            runner = GeoclawRunner(work_dir, event_track, event['period'][0], event,
                                   track_centr[event['centroid_mask']], zos_path, topo_path,
                                   gauges=gauges)
            runner.run()
            event_surge_h[event['centroid_mask']] = runner.surge_h
            surge_h.append(event_surge_h)
            for igauge, new_gauge_data in enumerate(runner.gauge_data):
                if len(new_gauge_data['time']) > 0:
                    for var in ['base_sea_level', 'topo_height', 'time', 'height_above_geoid']:
                        gauge_data[igauge][var].append(new_gauge_data[var])

        # write results to intensity array
        intensity[track_centr_msk] = np.stack(surge_h, axis=0).max(axis=0)

    return intensity, gauge_data


class GeoclawRunner():
    """"Wrapper for work directory setup and running of GeoClaw simulations.

    Attributes
    ----------
    surge_h : np.array
        Maximum height of inundation recorded at given centroids.
    gauge_data : list of dicts
        For each gauge, a dict containing `location`, `base_sea_level`, `topo_height`, `time` and
        `height_above_geoid` information.
    """

    def __init__(self, base_dir, track, time_offset, areas, centroids, zos_path, topo_path,
                 gauges=None):
        """Initialize GeoClaw working directory with ClawPack rundata

        Parameters
        ----------
        base_dir : str
            Location where to create the working directory.
        track : xr.Dataset
            Single tropical cyclone track.
        time_offset : np.datetime64
            Usually, time of landfall
        areas : dict
            Landfall event (single iterator output from TCSurgeEvents).
        centroids : np.array
            Points for which to record the maximum height of inundation.
            Each row is a lat-lon point.
        zos_path : str
            Path to NetCDF file containing gridded monthly sea level data.
        topo_path : str
            Path to raster file containing gridded elevation data.
        gauges : list of pairs (lat, lon), optional
            The locations of tide gauges where to measure temporal changes in sea level height.
            This is used mostly for validation purposes.
        """
        gauges = [] if gauges is None else gauges

        LOGGER.info("Running GeoClaw to determine surge on %d centroids", centroids.shape[0])
        self.track = track
        self.areas = areas
        self.centroids = centroids
        self.time_offset = time_offset
        self.zos_path = zos_path
        self.topo_path = topo_path
        self.gauge_data = [{'location': g, 'base_sea_level': 0, 'topo_height': -32768.0,
                            'time': [], 'height_above_geoid': []} for g in gauges]
        self.surge_h = np.zeros(centroids.shape[0])

        # compute time horizon
        self.time_horizon = tuple([int(t / np.timedelta64(1, 's'))
                                   for t in (self.track.time[0] - self.time_offset,
                                             self.track.time[-1] - self.time_offset)])

        # create work directory
        self.work_dir = os.path.join(
            base_dir, dt64_to_pydt(self.time_offset).strftime("%Y-%m-%d-%H"))
        os.makedirs(self.work_dir, exist_ok=True)
        LOGGER.info("Init GeoClaw working directory in %s", self.work_dir)

        # write Makefile
        with open(os.path.join(self.work_dir, "Makefile"), "w") as file_p:
            file_p.write(f"""\
CLAW = {clawpack_info()[0]}
CLAW_PKG = geoclaw
EXE = xgeoclaw
include $(CLAW)/geoclaw/src/2d/shallow/Makefile.geoclaw
SOURCES = $(CLAW)/riemann/src/rpn2_geoclaw.f \\
          $(CLAW)/riemann/src/rpt2_geoclaw.f \\
          $(CLAW)/riemann/src/geoclaw_riemann_utils.f
include $(CLAW)/clawutil/src/Makefile.common
""")
        with open(os.path.join(self.work_dir, "setrun.py"), "w") as file_p:
            file_p.write("")

        # write rundata
        self.write_rundata()


    def run(self):
        """Run GeoClaw script and set `surge_h` attribute."""
        LOGGER.info("Running GeoClaw...")
        self.stdout = ""
        with subprocess.Popen(["make", ".output"],
                              cwd=self.work_dir,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as proc:
            time_span = self.time_horizon[1] - self.time_horizon[0]
            last_perc = -100
            stopped = False
            for line in proc.stdout:
                line = line.decode()
                self.stdout += line
                line = line.rstrip()
                error_strings = [
                    "ABORTING CALCULATION",
                    "Stopping calculation",
                ]
                if any(err in line for err in error_strings):
                    stopped = True
                re_m = re.match(r".*t = ([-ED0-9\.\+]+)$", line)
                if re_m is not None:
                    time = float(re_m.group(1).replace("D", "E"))
                    perc = 100 * (time - self.time_horizon[0]) / time_span
                    if perc - last_perc >= 10:
                        LOGGER.info("%d%%", perc)
                        last_perc = perc
        if proc.returncode != 0 or stopped:
            self.print_stdout()
            LOGGER.error("GeoClaw run failed (see output above).")
        else:
            LOGGER.info("Reading GeoClaw output...")
            try:
                self.read_fgmax_data()
                self.read_gauge_data()
            except FileNotFoundError:
                self.print_stdout()
                LOGGER.info("Reading GeoClaw output failed (see output above).")


    def print_stdout(self):
        """"Print standard (and error) output of GeoClaw run."""
        LOGGER.info("Output of 'make .output' in GeoClaw work directory:")
        print(self.stdout)


    def read_fgmax_data(self):
        """Read fgmax output data from GeoClaw working directory."""
        # pylint: disable=import-outside-toplevel
        from clawpack.geoclaw import fgmax_tools
        outdir = os.path.join(self.work_dir, "_output")
        fg_path = os.path.join(outdir, "fgmax0001.txt")

        if not os.path.exists(fg_path):
            LOGGER.error("GeoClaw quit without creating fgmax data!")
            raise FileNotFoundError

        fgmax_grid = fgmax_tools.FGmaxGrid()
        fg_fname = os.path.join(self.work_dir, "fgmax_grids.data")
        with contextlib.redirect_stdout(None):
            fgmax_grid.read_fgmax_grids_data(1, fg_fname)
            fgmax_grid.read_output(outdir=outdir)
        assert fgmax_grid.point_style == 0
        self.surge_h[:] = fgmax_grid.h
        self.surge_h[fgmax_grid.arrival_time.mask] = 0


    def read_gauge_data(self):
        """Read gauge output data from GeoClaw working directory."""
        # pylint: disable=import-outside-toplevel
        from clawpack.pyclaw.gauges import GaugeSolution
        outdir = os.path.join(self.work_dir, "_output")
        for i_gauge, gauge in enumerate(self.gauge_data):
            gauge['base_sea_level'] = self.rundata.geo_data.sea_level
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                g = GaugeSolution(gauge_id=i_gauge + 1, path=outdir)
            if g.t is None:
                continue
            gauge['time'] = self.time_offset + g.t * np.timedelta64(1, 's')
            gauge['topo_height'] = np.mean(g.q[1,:] - g.q[0,:])
            gauge['height_above_geoid'] = g.q[1,:]


    def write_rundata(self):
        """Create Makefile and all necessary datasets in working directory."""
        # pylint: disable=import-outside-toplevel
        import clawpack.clawutil.data
        num_dim = 2
        self.rundata = clawpack.clawutil.data.ClawRunData("geoclaw", num_dim)
        self.set_rundata_claw()
        self.set_rundata_amr()
        self.set_rundata_geo()
        self.set_rundata_fgmax()
        self.set_rundata_storm()
        self.set_rundata_gauges()
        with contextlib.redirect_stdout(None), backup_loggers():
            self.rundata.write(out_dir=self.work_dir)


    def set_rundata_claw(self):
        """Set the rundata parameters in the `clawdata` category."""
        clawdata = self.rundata.clawdata
        clawdata.verbosity = 1
        clawdata.num_output_times = 0
        clawdata.lower = self.areas['wind_area'][:2]
        clawdata.upper = self.areas['wind_area'][2:]
        clawdata.num_cells = [int(clawdata.upper[0] - clawdata.lower[0]) * 4,
                              int(clawdata.upper[1] - clawdata.lower[1]) * 4]
        clawdata.num_eqn = 3
        clawdata.num_aux = 3 + 1 + 3
        clawdata.capa_index = 2
        clawdata.t0, clawdata.tfinal = self.time_horizon
        clawdata.dt_initial = 0.8 / max(clawdata.num_cells)
        clawdata.cfl_desired = 0.75
        clawdata.num_waves = 3
        clawdata.limiter = ['mc', 'mc', 'mc']
        clawdata.use_fwaves = True
        clawdata.source_split = 'godunov'
        clawdata.bc_lower = ['extrap', 'extrap']
        clawdata.bc_upper = ['extrap', 'extrap']


    def set_rundata_amr(self):
        """Set AMR-related rundata attributes."""
        clawdata = self.rundata.clawdata
        amrdata = self.rundata.amrdata
        refinedata = self.rundata.refinement_data
        amrdata.refinement_ratios_x = [2, 2, 2, 2, 3]
        amrdata.refinement_ratios_y = amrdata.refinement_ratios_x
        amrdata.refinement_ratios_t = amrdata.refinement_ratios_x
        amrdata.amr_levels_max = len(amrdata.refinement_ratios_x) + 1
        resolutions = [(clawdata.upper[0] - clawdata.lower[0]) / clawdata.num_cells[0]]
        for fact in amrdata.refinement_ratios_x:
            resolutions.append(resolutions[-1] / fact)
        LOGGER.info("GeoClaw resolution in arc-seconds: %s",
                    str(["%.2f" % (r * 60 * 60) for r in resolutions]))
        amrdata.aux_type = ['center', 'capacity', 'yleft', 'center', 'center', 'center', 'center']
        amrdata.regrid_interval = 3
        amrdata.regrid_buffer_width = 2
        amrdata.verbosity_regrid = 0
        regions = self.rundata.regiondata.regions
        t_1, t_2 = self.rundata.clawdata.t0, self.rundata.clawdata.tfinal
        maxlevel = amrdata.amr_levels_max
        x_1, y_1, x_2, y_2 = self.areas['landfall_area']
        regions.append([maxlevel - 2, maxlevel, t_1, t_2, x_1, x_2, y_1, y_2])
        for area in self.areas['surge_areas']:
            x_1, y_1, x_2, y_2 = area
            regions.append([maxlevel, maxlevel, t_1, t_2, x_1, x_2, y_1, y_2])
        refinedata.speed_tolerance = list(np.arange(1.0, maxlevel))
        refinedata.variable_dt_refinement_ratios = True
        refinedata.wave_tolerance = 1.0
        refinedata.deep_depth = 1e3


    def set_rundata_geo(self):
        """Set geo-related rundata attributes."""
        geodata = self.rundata.geo_data
        topodata = self. rundata.topo_data

        # lat-lon coordinate system
        geodata.coordinate_system = 2

        # different friction on land and at sea
        geodata.manning_coefficient = [0.025, 0.050]
        geodata.manning_break = [0.0]
        geodata.dry_tolerance = 1.e-2

        # get sea level information for affected months
        months = np.stack((self.track.time.dt.year, self.track.time.dt.month), axis=-1)
        if self.track.time[0].dt.day < 8:
            if months[0, 1] == 1:
                months[0, 0] -= 1
                months[0, 1] = 12
            else:
                months[0, 1] -= 1
        if self.track.time[-1].dt.day > 22:
            if months[-1, 1] == 12:
                months[-1, 0] += 1
                months[-1, 1] = 1
            else:
                months[-1, 1] += 1
        months = np.unique(months, axis=0)
        geodata.sea_level = mean_max_sea_level(self.zos_path, months, self.areas['wind_area'])

        # load elevation data, resolution depending on area of refinement
        topodata.topofiles = []
        areas = [
            self.areas['wind_area'],
            self.areas['landfall_area']
        ] + self.areas['surge_areas']
        resolutions = [360, 120] + [30 for a in self.areas['surge_areas']]
        dems_for_plot = []
        for res_as, bounds in zip(resolutions, areas):
            bounds, topo = load_topography(self.topo_path, bounds, res_as)
            if 0 in topo.Z.shape:
                LOGGER.warning("Area is ignored because it is too small.")
                continue
            tt3_fname = 'topo_{}s_{}.tt3'.format(res_as, bounds_to_str(bounds))
            tt3_fname = os.path.join(self.work_dir, tt3_fname)
            topo.write(tt3_fname)
            topodata.topofiles.append([3, 1, self.rundata.amrdata.amr_levels_max,
                                       self.rundata.clawdata.t0, self.rundata.clawdata.tfinal,
                                       tt3_fname])
            dems_for_plot.append((bounds, topo.Z))
        plot_dems(dems_for_plot, track=self.track, centroids=self.centroids,
                  path=os.path.join(self.work_dir, "dems.pdf"))


    def set_rundata_fgmax(self):
        """Set monitoring-related rundata attributes."""
        # pylint: disable=import-outside-toplevel
        from clawpack.geoclaw import fgmax_tools

        # monitor max height values on centroids
        self.rundata.fgmax_data.num_fgmax_val = 1
        fgmax_grid = fgmax_tools.FGmaxGrid()
        fgmax_grid.point_style = 0
        fgmax_grid.tstart_max = self.rundata.clawdata.t0
        fgmax_grid.tend_max = self.rundata.clawdata.tfinal
        fgmax_grid.dt_check = 0
        fgmax_grid.min_level_check = self.rundata.amrdata.amr_levels_max
        fgmax_grid.arrival_tol = 1.e-2
        fgmax_grid.npts = self.centroids.shape[0]
        fgmax_grid.X = self.centroids[:, 1]
        fgmax_grid.Y = self.centroids[:, 0]
        self.rundata.fgmax_data.fgmax_grids.append(fgmax_grid)


    def set_rundata_storm(self):
        """Set storm-related rundata attributes."""
        surge_data = self.rundata.surge_data
        surge_data.wind_forcing = True
        surge_data.drag_law = 1
        surge_data.pressure_forcing = True
        surge_data.storm_specification_type = 'holland80'
        surge_data.storm_file = os.path.join(self.work_dir, "track.storm")
        gc_storm = climada_xarray_to_geoclaw_storm(self.track,
                                                   offset=dt64_to_pydt(self.time_offset))
        gc_storm.write(surge_data.storm_file, file_format='geoclaw')


    def set_rundata_gauges(self):
        """Set gauge-related rundata attributes."""
        for i_gauge, gauge in enumerate(self.gauge_data):
            lat, lon = gauge['location']
            self.rundata.gaugedata.gauges.append([i_gauge + 1, lon, lat,
                                                  self.rundata.clawdata.t0,
                                                  self.rundata.clawdata.tfinal])
        # q[0]: height above topography (topo includes added sea_level)
        self.rundata.gaugedata.q_out_fields = [0]


def plot_dems(dems, track=None, path=None, centroids=None):
    """Plot given DEMs as rasters to one worldmap

    Parameters
    ----------
    dems : list of pairs
        pairs (bounds, heights)
    path : str or None
        If given, save plot in this location. Default: None
    track : xr.Dataset
        If given, overlay the tropical cyclone track. Default: None
    centroids : np.array
        If given, overlay as scatter points. Default: None
    """
    matplotlib.rc('axes', linewidth=0.5)
    matplotlib.rc('font', size=7, family='serif')
    matplotlib.rc('xtick', top=True, direction='out')
    matplotlib.rc('xtick.major', size=2.5, width=0.5)
    matplotlib.rc('ytick', right=True, direction='out')
    matplotlib.rc('ytick.major', size=2.5, width=0.5)
    total_bounds = (
        min([bounds[0] for bounds, _ in dems]),
        min([bounds[1] for bounds, _ in dems]),
        max([bounds[2] for bounds, _ in dems]),
        max([bounds[3] for bounds, _ in dems]),
    )
    mid_lon = 0.5 * (total_bounds[0] + total_bounds[2])
    aspect_ratio = 1.124 * ((total_bounds[2] - total_bounds[0])
                            / (total_bounds[3] - total_bounds[1]))
    fig = plt.figure(
        figsize=(10, 10 / aspect_ratio) if aspect_ratio >= 1 else (aspect_ratio * 10, 10),
        dpi=100)
    proj = ccrs.PlateCarree(central_longitude=mid_lon)
    axes = fig.add_subplot(111, projection=proj)
    axes.spines['geo'].set_linewidth(0.5)
    axes.set_xlim(total_bounds[0] - mid_lon, total_bounds[2] - mid_lon)
    axes.set_ylim(total_bounds[1], total_bounds[3])
    cmap, cnorm = colormap_coastal_dem(axes=axes)
    for bounds, heights in dems:
        bounds = (bounds[0] - mid_lon, bounds[1], bounds[2] - mid_lon, bounds[3])
        axes.imshow(heights, transform=proj, cmap=cmap, norm=cnorm,
                    extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
                    vmin=cnorm.values[0], vmax=cnorm.values[-1])
        plot_bounds(axes, bounds, color='k', linewidth=0.5)
    axes.coastlines(resolution='10m', linewidth=0.5)
    if track is not None:
        axes.plot(track.lon - mid_lon, track.lat, color='k', linewidth=0.5)
    if centroids is not None:
        axes.scatter(centroids[:, 1] - mid_lon, centroids[:, 0], s=0.1, alpha=0.5)
    fig.subplots_adjust(left=0.02, bottom=0.01, right=0.89, top=0.99, wspace=0, hspace=0)
    if path is None or not hasattr(__main__, '__file__'):
        plt.show()
    if path is not None:
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(path)
        plt.close(fig)


def colormap_coastal_dem(axes=None):
    """Return colormap and normalization for coastal areas of DEMs

    Parameters
    ----------
    axes : matplotlib.axes.Axes, optional
        If given, add a colorbar to the right of it. Default: None
    """
    cmap_terrain = [
        (0, 0, 0),
        (3, 73, 114),
        (52, 126, 255),
        (146, 197, 222),
        (255, 251, 171),
        (165, 230, 162),
        (27, 149, 29),
        (32, 114, 11),
        (117, 84, 0),
    ]
    cmap_terrain = matplotlib.colors.LinearSegmentedColormap.from_list(
        "coastal_dem", [tuple(c / 255 for c in rgb) for rgb in cmap_terrain])
    cnorm_coastal_dem = LinearSegmentedNormalize([-8000, -1000, -10, -5, 0, 5, 10, 100, 1000])
    if axes:
        cbar_ax = inset_axes(axes, width="5%", height="100%",
                             loc='lower left', bbox_to_anchor=(1.02, 0., 0.5, 1),
                             bbox_transform=axes.transAxes, borderpad=0)
        cbar = plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_terrain), cax=cbar_ax)
        cbar.set_ticks(cnorm_coastal_dem.values)
        cbar.set_ticklabels(cnorm_coastal_dem.vthresh)
    return cmap_terrain, cnorm_coastal_dem


class LinearSegmentedNormalize(matplotlib.colors.Normalize):
    """Piecewise linear color normalization."""
    def __init__(self, vthresh):
        """Initialize normalization

        Parameters
        ----------
        vthresh : list
            Equally distributed to the interval [0,1].
        """
        self.vthresh = vthresh
        self.values = np.linspace(0, 1, len(self.vthresh))
        matplotlib.colors.Normalize.__init__(self, vthresh[0], vthresh[1], False)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(np.interp(value, self.vthresh, self.values))


def climada_xarray_to_geoclaw_storm(track, offset=None):
    """Convert CLIMADA's xarray TC track to GeoClaw storm object

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    offset : datetime
        Time zero for internal use in GeoClaw.

    Returns
    -------
    gc_storm : clawpack.geoclaw.surge.storm.Storm
    """
    # pylint: disable=import-outside-toplevel
    from clawpack.geoclaw.surge.storm import Storm
    gc_storm = Storm()
    gc_storm.t = dt64_to_pydt(track.time.values)
    if offset is not None:
        gc_storm.time_offset = offset
    gc_storm.eye_location = np.stack([track.lon, track.lat], axis=-1)
    gc_storm.max_wind_speed = track.max_sustained_wind.values * KN_TO_MS
    gc_storm.max_wind_radius = track.radius_max_wind.values * NM_TO_KM * 1000
    gc_storm.central_pressure = track.central_pressure.values * MBAR_TO_PA
    gc_storm.storm_radius = track.radius_oci.values * NM_TO_KM * 1000
    return gc_storm


def mean_max_sea_level(path, months, bounds):
    """Mean of maxima over affected area in affected months

    Parameters
    ----------
    path : str
        Path to NetCDF file containing monthly sea level data.
    months : np.array
        each row is a tuple (year, month)
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    zos : float
        Sea level height in meters
    """
    LOGGER.info("Reading sea level data from %s", path)
    var_names = {
        'lon': ('coords', ["longitude", "lon", "x"]),
        'lat': ('coords', ["latitude", "lat", "y"]),
        'time': ('coords', ["time", "date", "datetime"]),
        'zos': ('variables', ["zos", "sla", "ssh", "adt"]),
    }
    with xr.open_dataset(path) as zos_ds:
        for new_name, (var_type, all_names) in var_names.items():
            old_name = [c for c in getattr(zos_ds, var_type) if c.lower() in all_names][0]
            zos_ds = zos_ds.rename({old_name: new_name})
        ds_bounds = (zos_ds.lon.values.min(), zos_ds.lat.values.min(),
                     zos_ds.lon.values.max(), zos_ds.lat.values.max())
        ds_period = (zos_ds.time[0], zos_ds.time[-1])
        LOGGER.info("Sea level data available within bounds %s", ds_bounds)
        LOGGER.info("Sea level data available within period from %04d-%02d till %04d-%02d",
                    ds_period[0].dt.year, ds_period[0].dt.month,
                    ds_period[1].dt.year, ds_period[1].dt.month)
        mask_time = np.any([(zos_ds.time.dt.year == m[0]) & (zos_ds.time.dt.month == m[1])
                           for m in months], axis=0)
        if np.count_nonzero(mask_time) != months.shape[0]:
            LOGGER.error("The sea level data set doesn't contain the required months: %s",
                         ", ".join(f"{m[0]:04d}-{m[1]:02d}" for m in months))
            raise IndexError
        zos_ds = zos_ds.sel(time=mask_time)
        zos_lon = coord_util.lon_normalize(zos_ds.lon.values,
                                           center=0.5 * (bounds[0] + bounds[2]))
        mask_lat = (bounds[1] <= zos_ds.lat) & (zos_ds.lat <= bounds[3])
        mask_lon = (bounds[0] <= zos_lon) & (zos_lon <= bounds[2]) & np.isfinite(zos_ds.lon)
        mask_bounds = (mask_lat & mask_lon)
        if not np.any(mask_bounds):
            LOGGER.error("The sea level data set doesn't intersect the required bounds: %s",
                         bounds)
            raise IndexError
        zos_ds = zos_ds.where(mask_bounds, drop=True)
        zos = zos_ds.zos.values[:]
    return np.nanmean(np.nanmax(zos, axis=(1, 2)))


def load_topography(path, bounds, res_as):
    """Load topographical elevation data in specified bounds and resolution

    The bounds of the returned topodata are always larger than the requested bounds to make sure
    that the pixel centers still cover the requested region.

    Parameters
    ----------
    path : str
        Path to raster file containing elevation data above reference geoid.
    bounds : tuple
        Bounds (lon_min, lat_min, lon_max, lat_max) of region of interest.
    res_as : float
        Resolution in arc-seconds.

    Returns
    -------
    bounds : tuple
        Bounds (lon_min, lat_min, lon_max, lat_max) actually covered by the returned topodata.
    topo : clawpack.geoclaw.topotools.Topography object
        The object's x, y and Z attributes contain the loaded topodata.
    """
    # pylint: disable=import-outside-toplevel
    from clawpack.geoclaw import topotools

    LOGGER.info("Load elevation data [%s, %s] from %s", res_as, bounds, path)
    res = res_as / (60 * 60)
    zvalues, transform = coord_util.read_raster_bounds(path, bounds, res=res, bands=[1])
    zvalues = zvalues[0]
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax, ymax = xmin + zvalues.shape[1] * xres, ymin + zvalues.shape[0] * yres
    if xres < 0:
        zvalues = np.flip(zvalues, axis=1)
        xres, xmin, xmax = -xres, xmax, xmin
    if yres < 0:
        zvalues = np.flip(zvalues, axis=0)
        yres, ymin, ymax = -yres, ymax, ymin
    xmin, xmax = coord_util.lon_normalize(np.array([xmin, xmax]),
                                          center=0.5 * (bounds[0] + bounds[2]))
    bounds = (xmin, ymin, xmax, ymax)
    xcoords = np.arange(xmin + xres / 2, xmax, xres)
    ycoords = np.arange(ymin + yres / 2, ymax, yres)

    topo = topotools.Topography()
    topo.set_xyZ(xcoords, ycoords, zvalues.astype(np.float64))
    return bounds, topo


class TCSurgeEvents():
    """Periods and areas along TC track where centroids are reachable by surge

    When iterating over this object, it will return single events represented
    by dictionaries of this form:
        { 'period', 'time_mask', 'time_mask_buffered', 'wind_area',
          'landfall_area', 'surge_areas', 'centroid_mask' }

    Attributes
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : 2d np.array
        Each row is a centroid [lat, lon]. These are supposed to be coastal points of interest.
    d_centroids : 2d np.array
        For each eye position, distances to centroids.
    nevents : int
        Number of landfall events.
    period : list of tuples
        For each event, a pair of datetime objects indicating beginnig and end
        of landfall event period.
    time_mask : list of np.array
        For each event, a mask along `track.time` indicating the landfall event period.
    time_mask_buffered : list of np.array
        For each event, a mask along `track.time` indicating the landfall event period
        with added buffer for storm form-up.
    wind_area : list of tuples
        For each event, a rectangular box around the geographical area that is affected
        by storm winds during the (buffered) landfall event.
    landfall_area : list of tuples
        For each event, a rectangular box around the geographical area that is affected
        by storm surge during the landfall event.
    surge_areas : list of list of tuples
        For each event, a list of tight rectangular boxes around the centroids that will
        be affected by storm surge during the landfall event.
    centroid_mask : list of np.array
        For each event, a mask along first axis of `centroids` indicating which centroids are
        reachable by surge during this landfall event.
    """
    keys = ['period', 'time_mask', 'time_mask_buffered', 'wind_area',
            'landfall_area', 'surge_areas', 'centroid_mask']
    maxlen = 36
    maxbreak = 12

    def __init__(self, track, centroids):
        """Determine temporal periods and geographical regions where the storm
        affects the centroids

        Parameters
        ----------
        track : xr.Dataset
            Single tropical cyclone track.
        centroids : 2d np.array
            Each row is a centroid [lat, lon].
        """
        self.track = track
        self.centroids = centroids

        locs = np.stack([self.track.lat, self.track.lon], axis=1)
        self.d_centroids = coord_util.dist_approx(
            locs[None, :, 0], locs[None, :, 1],
            self.centroids[None, :, 0], self.centroids[None, :, 1],
            method="geosphere")[0]

        self._set_periods()
        self.time_mask = [self._period_to_mask(p) for p in self.period]
        self.time_mask_buffered = [self._period_to_mask(p, buffer=(0.3, 0.3))
                                   for p in self.period]
        self._set_areas()
        self._remove_harmless_events()


    def __iter__(self):
        for i_event in range(self.nevents):
            yield {key: getattr(self, key)[i_event] for key in self.keys}


    def __len__(self):
        return self.nevents


    def _remove_harmless_events(self):
        """Remove events without affected areas (surge_areas)"""
        relevant_idx = [i for i in range(self.nevents) if len(self.surge_areas[i]) > 0]
        for key in self.keys:
            setattr(self, key, [getattr(self, key)[i] for i in relevant_idx])
        self.nevents = len(relevant_idx)


    def _set_periods(self):
        """Determine beginning and end of landfall events."""
        radii = np.fmax(0.4 * self.track.radius_oci.values,
                        1.6 * self.track.radius_max_wind.values) * NM_TO_KM
        centr_counts = np.count_nonzero(self.d_centroids < radii[:, None], axis=1)
        # below 35 knots, winds are not strong enough for significant surge
        mask = (centr_counts > 1) & (self.track.max_sustained_wind > 35)

        # convert landfall mask to (clustered) start/end pairs
        period = []
        start = end = None
        for i, date in enumerate(self.track.time):
            if start is not None:
                # periods cover at most 36 hours and a split will be forced
                # at breaks of more than 12 hours.
                exceed_maxbreak = (date - end) / np.timedelta64(1, 'h') > self.maxbreak
                exceed_maxlen = (date - start) / np.timedelta64(1, 'h') > self.maxlen
                if exceed_maxlen or exceed_maxbreak:
                    period.append((start, end))
                    start = end = None
            if mask[i]:
                end = date
                if start is None:
                    start = date
        if start is not None:
            period.append((start, end))
        self.period = [(s.values[()], e.values[()]) for s, e in period]
        self.nevents = len(self.period)


    def _period_to_mask(self, period, buffer=(0.0, 0.0)):
        """Compute buffered 1d-mask over track time series from period

        Parameters
        ----------
        period : pair of datetimes
            start/end of period
        buffer : pair of floats
            buffer to add in days

        Returns
        -------
        mask : np.array
        """
        diff_start = np.array([(t - period[0]) / np.timedelta64(1, 'D')
                               for t in self.track.time])
        diff_end = np.array([(t - period[1]) / np.timedelta64(1, 'D')
                             for t in self.track.time])
        return (diff_start >= -buffer[0]) & (diff_end <= buffer[1])


    def _set_areas(self):
        """For each event, determine areas affected by wind and surge."""
        self.wind_area = []
        self.landfall_area = []
        self.surge_areas = []
        self.centroid_mask = []
        for i_event, mask_buf in enumerate(self.time_mask_buffered):
            track = self.track.sel(time=mask_buf)
            mask = self.time_mask[i_event][mask_buf]
            lf_radii = np.fmax(0.4 * track.radius_oci.values, 1.6 * track.radius_max_wind.values)

            # wind area (maximum bounds to consider)
            pad = 0.9 * track.radius_oci / 60
            self.wind_area.append((
                float((track.lon - pad).min()),
                float((track.lat - pad).min()),
                float((track.lon + pad).max()),
                float((track.lat + pad).max()),
            ))

            # landfall area
            pad = lf_radii / 60
            self.landfall_area.append((
                float((track.lon - pad)[mask].min()),
                float((track.lat - pad)[mask].min()),
                float((track.lon + pad)[mask].max()),
                float((track.lat + pad)[mask].max()),
            ))

            # surge areas
            lf_radii *= NM_TO_KM
            centroids_mask = np.any(
                self.d_centroids[mask_buf][mask] < lf_radii[mask, None], axis=0)
            points = self.centroids[centroids_mask, ::-1]
            surge_areas = []
            if points.shape[0] > 0:
                pt_bounds = list(points.min(axis=0)) + list(points.max(axis=0))
                pt_size = (pt_bounds[2] - pt_bounds[0]) * (pt_bounds[3] - pt_bounds[1])
                if pt_size < (2 * lf_radii.max() * KM_TO_DEG)**2:
                    small_bounds = [pt_bounds]
                else:
                    small_bounds, pt_size = boxcover_points_along_axis(points, 3)
                min_size = 3. / (60. * 60.)
                if pt_size > (2 * min_size)**2:
                    for bounds in small_bounds:
                        bounds[:2] = [v - min_size for v in bounds[:2]]
                        bounds[2:] = [v + min_size for v in bounds[2:]]
                        surge_areas.append(bounds)
            surge_areas = [tuple([float(b) for b in bounds]) for bounds in surge_areas]
            self.surge_areas.append(surge_areas)

            # centroids affected by surge
            centroids_mask = np.zeros(self.centroids.shape[0], dtype=bool)
            for bounds in surge_areas:
                centroids_mask |= ((bounds[0] <= self.centroids[:, 1])
                                   & (bounds[1] <= self.centroids[:, 0])
                                   & (self.centroids[:, 1] <= bounds[2])
                                   & (self.centroids[:, 0] <= bounds[3]))
            self.centroid_mask.append(centroids_mask)


    def plot_areas(self, path=None):
        """Plot areas associated with this track's landfall events

        Parameters
        ----------
        path : str, optional
            If given, save the plots to the given location. Default: None
        """
        t_bounds = (self.track.lon.min(), self.track.lat.min(),
                    self.track.lon.max(), self.track.lat.max())
        mid_lon = 0.5 * float(t_bounds[0] + t_bounds[2])
        proj = ccrs.PlateCarree(central_longitude=mid_lon)
        aspect_ratio = 1.124 * ((t_bounds[2] - t_bounds[0] + 2 * CENTR_NODE_MAX_DIST_DEG)
                                / (t_bounds[3] - t_bounds[1] + 2 * CENTR_NODE_MAX_DIST_DEG))
        fig = plt.figure(
            figsize=(10, 10 / aspect_ratio) if aspect_ratio >= 1 else (aspect_ratio * 10, 10),
            dpi=100)
        axes = fig.add_subplot(111, projection=proj)
        axes.outline_patch.set_linewidth(0.5)
        axes.set_xlim(t_bounds[0] - mid_lon - CENTR_NODE_MAX_DIST_DEG,
                      t_bounds[2] - mid_lon + CENTR_NODE_MAX_DIST_DEG)
        axes.set_ylim(t_bounds[1] - CENTR_NODE_MAX_DIST_DEG,
                      t_bounds[3] + CENTR_NODE_MAX_DIST_DEG)

        # plot coastlines
        axes.add_feature(cfeature.OCEAN.with_scale('50m'), linewidth=0.1)

        # plot TC track with masks
        axes.plot(self.track.lon - mid_lon, self.track.lat, color='k', linewidth=0.5)
        for mask in self.time_mask_buffered:
            axes.plot(self.track.lon[mask] - mid_lon, self.track.lat[mask],
                      color='k', linewidth=1.5)

        # plot rectangular areas
        linestep = max(0.5, 1 - 0.1 * self.nevents)
        linew = 1 + linestep * self.nevents
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i_event, mask in enumerate(self.time_mask):
            axes.plot(self.track.lon[mask] - mid_lon, self.track.lat[mask],
                      color=color_cycle[i_event], linewidth=3)
            linew -= linestep
            areas = [
                self.wind_area[i_event],
                self.landfall_area[i_event],
            ] + self.surge_areas[i_event]
            for bounds in areas:
                bounds = (bounds[0] - mid_lon, bounds[1], bounds[2] - mid_lon, bounds[3])
                plot_bounds(axes, bounds, color=color_cycle[i_event], linewidth=linew)

        # plot track data points
        axes.scatter(self.track.lon - mid_lon, self.track.lat, s=2)
        fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0, hspace=0)
        if path is None or not hasattr(__main__, '__file__'):
            plt.show()
        if path is not None:
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(path)
            plt.close(fig)


def plot_bounds(axes, bounds, **kwargs):
    """Plot given bounds as rectangular boundary lines

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Target Axes to plot to.
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)
    **kwargs :
        Keyword arguments that are passed on to the `plot` function.
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    axes.plot([lon_min, lon_min, lon_max, lon_max, lon_min],
              [lat_min, lat_max, lat_max, lat_min, lat_min], **kwargs)


def boxcover_points_along_axis(points, nsplits):
    """Cover n-dimensional points with grid-aligned boxes

    Parameters
    ----------
    points : np.array
        Each row is an n-dimensional point.
    nsplits : int
        Maximum number of boxes to use.

    Returns
    -------
    boxes : list of tuples
        Bounds of covering boxes.
    boxes_size : float
        Total volume/area of the covering boxes.
    """
    ndim = points.shape[1]
    bounds_min, bounds_max = points.min(axis=0), points.max(axis=0)
    final_boxes = []
    final_boxes_size = 1 + np.prod(bounds_max - bounds_min)
    for axis in range(ndim):
        splits = [((nsplits - i) / nsplits) * bounds_min[axis]
                  + (i / nsplits) * bounds_max[axis]
                  for i in range(1, nsplits)]
        boxes = []
        for i in range(nsplits):
            if i == 0:
                mask = points[:, axis] <= splits[0]
            elif i == nsplits - 1:
                mask = points[:, axis] > splits[-1]
            else:
                mask = (points[:, axis] <= splits[i]) \
                    & (points[:, axis] > splits[i - 1])
            masked_points = points[mask, :]
            if masked_points.shape[0] > 0:
                boxes.append((masked_points.min(axis=0), masked_points.max(axis=0)))
        boxes_size = np.sum([np.prod(bmax - bmin) for bmin, bmax in boxes])
        if boxes_size < final_boxes_size:
            final_boxes = [list(bmin) + list(bmax) for bmin, bmax in boxes]
            final_boxes_size = boxes_size
    return final_boxes, final_boxes_size


def clawpack_info():
    """Information about the available clawpack version

    Returns
    -------
    path : str or None
        If the python package clawpack is not available, None is returned.
        Otherwise, the CLAW source path is returned.
    decorators : tuple of str
        Strings describing the available version of clawpack. If it's a git
        checkout, the first string will be the full commit hash and the
        following strings will be git decorators such as tags or branch names
        that point to this checkout.
    """
    git_cmd = ["git", "log", "--pretty=format:%H%D", "-1"]
    try:
        # pylint: disable=import-outside-toplevel
        import clawpack
    except ImportError:
        return None, ()

    ver = clawpack.__version__
    path = os.path.dirname(os.path.dirname(clawpack.__file__))
    LOGGER.info("Found Clawpack version %s in %s", ver, path)

    proc = subprocess.Popen(git_cmd, stdout=subprocess.PIPE, cwd=path)
    out = proc.communicate()[0].decode()
    if proc.returncode != 0:
        return path, (ver,)
    decorators = [out[:40]] + out[40:].split(", ")
    decorators = [d.replace("tag: ", "") for d in decorators]
    decorators = [d.replace("HEAD -> ", "") for d in decorators]
    return path, decorators


def setup_clawpack(version=CLAWPACK_VERSION):
    """Install the specified version of clawpack if not already present

    Parameters
    ----------
    version : str, optional
        A git (short or long) hash, branch name or tag.
    """
    path, git_ver = clawpack_info()
    if path is None or version not in git_ver and version not in git_ver[0]:
        LOGGER.info("Installing Clawpack version %s", version)
        src_path = CLAWPACK_SRC_DIR
        pkg = f"git+{CLAWPACK_GIT_URL}@{version}#egg=clawpack-{version}"
        cmd = [sys.executable, "-m", "pip", "install", "--src", src_path, "-e", pkg]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            logging.error("pip install failed with return code %d. stdout: %s",
                          exc.returncode, exc.output)
            logging.error("Make sure that a Fortran compiler (e.g. gfortran) is available on "
                          "your machine before using tc_surge_geoclaw!")
            raise exc
        importlib.reload(site)
        importlib.invalidate_caches()

    with backup_loggers():
        # pylint: disable=unused-import,import-outside-toplevel
        import clawpack.pyclaw


@contextlib.contextmanager
def backup_loggers():
    # some modules (such as clawpack.pyclaw) use logging.config.fileConfig which disables all
    # registered loggers; the following lines revert this
    try:
        logger_state = {name: logger.disabled
                        for name, logger in logging.root.manager.loggerDict.items()
                        if isinstance(logger, logging.Logger)}
        yield logger_state
    finally:
        for name, logger in logging.root.manager.loggerDict.items():
            if name in logger_state and not logger_state[name]:
                logger.disabled = False


def bounds_to_str(bounds):
    """Convert longitude/latitude bounds to a human-readable string

    Example
    -------
    >>> bounds_to_str((-4.2, 1.0, -3.05, 2.125))
    '1N-2.125N_4.2W-3.05W'

    Parameters
    ----------
    bounds : tuple
        (lon_min, lat_min, lon_max, lat_max)

    Returns
    -------
    string : str
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    return '{:.4g}{}-{:.4g}{}_{:.4g}{}-{:.4g}{}'.format(
        abs(lat_min), 'N' if lat_min >= 0 else 'S',
        abs(lat_max), 'N' if lat_max >= 0 else 'S',
        abs(lon_min), 'E' if lon_min >= 0 else 'W',
        abs(lon_max), 'E' if lon_max >= 0 else 'W')


def dt64_to_pydt(date):
    """Convert datetime64 value or array to python datetime object or list

    Parameters
    ----------
    date : datetime64 value or array

    Returns
    -------
    dt : datetime or list of datetime objects
    """
    result = pd.Series(date).dt.to_pydatetime()
    if isinstance(date, np.datetime64):
        return result[0]
    return list(result)
