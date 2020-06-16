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

Define TCTracks: IBTracs reader and tracks manager.
"""

__all__ = ['SAFFIR_SIM_CAT', 'TCTracks', 'set_category']

import os
import glob
import shutil
import logging
import datetime as dt
import array
import itertools
import numpy as np
import matplotlib.cm as cm_mp
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
from sklearn.neighbors import DistanceMetric
import netCDF4 as nc
from numba import jit
import scipy.io.matlab as matlab

from climada.util import ureg
from climada.util.config import CONFIG
import climada.util.coordinates as coord_util
from climada.util.constants import EARTH_RADIUS_KM, SYSTEM_DIR
from climada.util.files_handler import get_file_names, download_ftp
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 137, 1000]
""" Saffir-Simpson Hurricane Wind Scale in kn based on NOAA"""

CAT_NAMES = {1: 'Tropical Depression', 2: 'Tropical Storm',
             3: 'Hurrican Cat. 1', 4: 'Hurrican Cat. 2',
             5: 'Hurrican Cat. 3', 6: 'Hurrican Cat. 4', 7: 'Hurrican Cat. 5'}
""" Saffir-Simpson category names. """

CAT_COLORS = cm_mp.rainbow(np.linspace(0, 1, len(SAFFIR_SIM_CAT)))
""" Color scale to plot the Saffir-Simpson scale."""

IBTRACS_URL = 'ftp://eclipse.ncdc.noaa.gov/pub/ibtracs//v04r00/provisional/netcdf/'
""" FTP of IBTrACS netcdf file containing all tracks v4.0 """

IBTRACS_FILE = 'IBTrACS.ALL.v04r00.nc'
""" IBTrACS v4.0 file all """

DEF_ENV_PRESSURE = 1010
""" Default environmental pressure """

EMANUEL_RMW_CORR_FILES = [
    'temp_ccsm420thcal.mat', 'temp_ccsm4rcp85_full.mat',
    'temp_gfdl520thcal.mat', 'temp_gfdl5rcp85cal_full.mat',
    'temp_hadgem20thcal.mat', 'temp_hadgemrcp85cal_full.mat',
    'temp_miroc20thcal.mat', 'temp_mirocrcp85cal_full.mat',
    'temp_mpi20thcal.mat', 'temp_mpircp85cal_full.mat',
    'temp_mri20thcal.mat', 'temp_mrircp85cal_full.mat',
]
EMANUEL_RMW_CORR_FACTOR = 2.0
""" Kerry Emanuel track files in this list require a correction: The radius of
    maximum wind (rmstore) needs to be multiplied by factor 2. """

class TCTracks():
    """Contains tropical cyclone tracks.

    Attributes:
        data (list(xarray.Dataset)): list of tropical cyclone tracks. Each
            track contains following attributes:
                - time (coords)
                - lat (coords)
                - lon (coords)
                - time_step
                - radius_max_wind
                - max_sustained_wind
                - central_pressure
                - environmental_pressure
                - max_sustained_wind_unit (attrs)
                - central_pressure_unit (attrs)
                - name (attrs)
                - sid (attrs)
                - orig_event_flag (attrs)
                - data_provider (attrs)
                - basin (attrs)
                - id_no (attrs)
                - category (attrs)
            computed during processing:
                - on_land
                - dist_since_lf
    """
    def __init__(self, pool=None):
        """Empty constructor. Read csv IBTrACS files if provided. """
        self.data = list()
        if pool:
            self.pool = pool
            LOGGER.debug('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def append(self, tracks):
        """Append tracks to current.

        Parameters:
            tracks (xarray.Dataset or list(xarray.Dataset)): tracks to append.
        """
        if not isinstance(tracks, list):
            tracks = [tracks]
        self.data.extend(tracks)

    def get_track(self, track_name=None):
        """Get track with provided name. Return all tracks if no name provided.

        Parameters:
            track_name (str, optional): name or sid (ibtracsID for IBTrACS)
                of track

        Returns:
            xarray.Dataset or [xarray.Dataset]
        """
        if track_name is None:
            if len(self.data) == 1:
                return self.data[0]
            return self.data

        for track in self.data:
            if track.name == track_name:
                return track
            if hasattr(track, 'sid') and track.sid == track_name:
                return track

        LOGGER.info('No track with name or sid %s found.', track_name)
        return []

    def read_ibtracs_netcdf(self, provider='usa', storm_id=None,
                            year_range=(1980, 2018), basin=None,
                            file_name='IBTrACS.ALL.v04r00.nc', correct_pres=True):
        """Fill from raw ibtracs v04. Removes nans in coordinates, central
        pressure and removes repeated times data. Fills nans of environmental_pressure
        and radius_max_wind. Checks environmental_pressure > central_pressure.

        Parameters:
            provider (str): data provider. e.g. usa, newdelhi, bom, cma, tokyo
            storm_id (str or list(str), optional): ibtracs if of the storm,
                e.g. 1988234N13299, [1988234N13299, 1989260N11316]
            year_range(tuple, optional): (min_year, max_year). Default: (1980, 2018)
            basin (str, optional): e.g. US, SA, NI, SI, SP, WP, EP, NA. if not
                provided, consider all basins.
            file_name (str, optional): name of netcdf file to be dowloaded or located
                at climada/data/system. Default: 'IBTrACS.ALL.v04r00.nc'.
            correct_pres (bool, optional): correct central pressure if missing
                values. Default: False
        """
        self.data = list()
        fn_nc = os.path.join(os.path.abspath(SYSTEM_DIR), file_name)
        if not glob.glob(fn_nc):
            try:
                download_ftp(os.path.join(IBTRACS_URL, IBTRACS_FILE), IBTRACS_FILE)
                shutil.move(IBTRACS_FILE, fn_nc)
            except ValueError as err:
                LOGGER.error('Error while downloading %s. Try to download it '+
                             'manually and put the file in ' +
                             'climada_python/data/system/', IBTRACS_URL)
                raise err

        sel_tracks = self._filter_ibtracs(fn_nc, storm_id, year_range, basin)
        nc_data = nc.Dataset(fn_nc)
        all_tracks = []
        for i_track in sel_tracks:
            all_tracks.append(self._read_one_raw(nc_data, i_track, provider,
                                                 correct_pres))
        self.data = [track for track in all_tracks if track is not None]

    def read_processed_ibtracs_csv(self, file_names):
        """Fill from processed ibtracs csv file.

        Parameters:
            file_names (str or list(str)): absolute file name(s) or
                folder name containing the files to read.
        """
        self.data = list()
        all_file = get_file_names(file_names)
        for file in all_file:
            self._read_one_csv(file)

    def read_simulations_emanuel(self, file_names, hemisphere='S'):
        """Fill from Kerry Emanuel tracks.

        Parameters:
            file_names (str or list(str)): absolute file name(s) or
                folder name containing the files to read.
            hemisphere (str, optional): 'S', 'N' or 'both'. Default: 'S'
        """
        self.data = []
        for path in get_file_names(file_names):
            rmw_corr = os.path.basename(path) in EMANUEL_RMW_CORR_FILES
            self._read_file_emanuel(path, hemisphere=hemisphere,
                                          rmw_corr=rmw_corr)

    def _read_file_emanuel(self, path, hemisphere='S', rmw_corr=False):
        """Append tracks from file containing Kerry Emanuel simulations.

        Parameters:
            path (str): absolute path of file to read.
            hemisphere (str, optional): 'S', 'N' or 'both'. Default: 'S'
            rmw_corr (str, optional): If True, multiply the radius of
                maximum wind by factor 2. Default: False.
        """
        if hemisphere == 'S':
            hem_min, hem_max = -90, 0
        elif hemisphere == 'N':
            hem_min, hem_max = 0, 90
        else:
            hem_min, hem_max = -90, 90

        LOGGER.info('Reading %s.', path)
        data_mat = matlab.loadmat(path)
        lat = data_mat['latstore']
        ntracks, nnodes = lat.shape
        years_uniq = np.unique(data_mat['yearstore'])
        LOGGER.info(f"File contains {ntracks} tracks "
                    f"(at most {nnodes} nodes each), "
                    f"representing {years_uniq.size} years "
                    f"({years_uniq[0]}-{years_uniq[-1]}).")

        # filter according to chosen hemisphere
        hem_mask = (lat >= hem_min) & (lat <= hem_max) | (lat == 0)
        hem_idx = np.all(hem_mask, axis=1).nonzero()[0]
        data_hem = lambda keys: [data_mat[f'{k}store'][hem_idx] for k in keys]

        lat, lon = data_hem(['lat', 'long'])
        months, days, hours = data_hem(['month', 'day', 'hour'])
        months, days, hours = [np.int8(ar) for ar in [months, days, hours]]
        tc_rmw, tc_maxwind, tc_pressure = data_hem(['rm', 'v', 'p'])
        years = data_mat['yearstore'][0,hem_idx]

        ntracks, nnodes = lat.shape
        LOGGER.info(f"Loading {ntracks} tracks on {hemisphere} hemisphere.")

        # change lon format to -180 to 180
        lon[lon > 180] = lon[lon > 180] - 360

        # change units from kilometers to nautical miles
        tc_rmw = (tc_rmw * ureg.kilometer).to(ureg.nautical_mile).magnitude
        if rmw_corr:
            LOGGER.info("Applying RMW correction.")
            tc_rmw *= EMANUEL_RMW_CORR_FACTOR

        for i_track in range(lat.shape[0]):
            valid_idx = (lat[i_track,:] != 0).nonzero()[0]
            nnodes = valid_idx.size
            time_step = np.abs(np.diff(hours[i_track,valid_idx])).min()

            # deal with change of year
            year = np.full(valid_idx.size, years[i_track])
            year_change = (np.diff(months[i_track,valid_idx]) < 0)
            year_change = year_change.nonzero()[0]
            if year_change.size > 0:
                year[year_change[0] + 1:] += 1

            try:
                datetimes = map(dt.datetime, year,
                                months[i_track, valid_idx],
                                days[i_track, valid_idx],
                                hours[i_track, valid_idx])
                datetimes = list(datetimes)
            except ValueError as e:
                # dates are known to contain invalid February 30
                date_feb = (months[i_track, valid_idx] == 2) \
                         & (days[i_track, valid_idx] > 28)
                if np.count_nonzero(date_feb) == 0:
                    # unknown invalid date issue
                    raise e
                step = time_step if not date_feb[0] else -time_step
                reference_idx = 0 if not date_feb[0] else -1
                reference_date = dt.datetime(
                    year[reference_idx],
                    months[i_track,valid_idx[reference_idx]],
                    days[i_track,valid_idx[reference_idx]],
                    hours[i_track,valid_idx[reference_idx]],)
                datetimes = [reference_date + dt.timedelta(hours=int(step*i))
                             for i in range(nnodes)]
            datetimes = np.array(datetimes)

            max_sustained_wind = tc_maxwind[i_track,valid_idx]
            max_sustained_wind_unit = 'kn'
            env_pressure = np.full(nnodes, DEF_ENV_PRESSURE)
            category = set_category(max_sustained_wind,
                                    max_sustained_wind_unit,
                                    SAFFIR_SIM_CAT)
            tr_ds = xr.Dataset({
                'time_step': ('time', np.full(nnodes, time_step)),
                'radius_max_wind': ('time', tc_rmw[i_track,valid_idx]),
                'max_sustained_wind': ('time', max_sustained_wind),
                'central_pressure': ('time', tc_pressure[i_track, valid_idx]),
                'environmental_pressure': ('time', env_pressure),
            }, coords={
                'time': datetimes,
                'lat': ('time', lat[i_track, valid_idx]),
                'lon': ('time', lon[i_track, valid_idx]),
            }, attrs={
                'max_sustained_wind_unit': max_sustained_wind_unit,
                'central_pressure_unit': 'mb',
                'name': str(hem_idx[i_track]),
                'sid': str(hem_idx[i_track]),
                'orig_event_flag': True,
                'data_provider': 'Emanuel',
                'basin': hemisphere,
                'id_no': hem_idx[i_track],
                'category': category,
            })
            self.data.append(tr_ds)

    def read_one_gettelman(self, nc_data, i_track):
        """Fill from Andrew Gettelman tracks.

        Parameters:
        nc_data (str): netCDF4.Dataset Objekt
        i_tracks (int): track number
        """
        scale_to_10m = (10./60.)**.11
        mps2kts = 1.94384
        basin_dict = {0: 'NA - North Atlantic',
                      1: 'SA - South Atlantic',
                      2: 'WP - West Pacific',
                      3: 'EP - East Pacific',
                      4: 'SP - South Pacific',
                      5: 'NI - North Indian',
                      6: 'SI - South Indian',
                      7: 'AS - Arabian Sea',
                      8: 'BB - Bay of Bengal',
                      9: 'EA - Eastern Australia',
                      10: 'WA - Western Australia',
                      11: 'CP - Central Pacific',
                      12: 'CS - Carribbean Sea',
                      13: 'GM - Gulf of Mexico',
                      14: 'MM - Missing'}

        val_len = nc_data.variables['numObs'][i_track]
        sid = str(i_track)
        times = nc_data.variables['source_time'][i_track, :][:val_len]

        datetimes = list()
        for t in times:
            try:
                datetimes.append(dt.datetime.strptime(str(nc.num2date(t,
                                 'days since {}'.format('1858-11-17'),
                                 calendar='standard')), '%Y-%m-%d %H:%M:%S'))
            except ValueError:
                # If wrong t, set t to previous t plus 3 hours
                if datetimes:
                    datetimes.append(datetimes[-1] + dt.timedelta(hours=3))
                else:
                    pos = list(times).index(t)
                    t = times[pos+1] - 1/24*3
                    datetimes.append(dt.datetime.strptime(str(nc.num2date(t,
                                     'days since {}'.format('1858-11-17'),
                                     calendar='standard')), '%Y-%m-%d %H:%M:%S'))
        time_step = []
        for i_time, time in enumerate(datetimes[1:], 1):
            time_step.append((time - datetimes[i_time-1]).total_seconds()/3600)
        time_step.append(time_step[-1])

        basin = list()
        for bs in nc_data.variables['basin'][i_track, :][:val_len]:
            try:
                basin.extend([basin_dict[bs]])
            except KeyError:
                basin.extend([np.nan])

        lon = nc_data.variables['lon'][i_track, :][:val_len]
        lon[lon>180]=lon[lon>180]-360 # change lon format to -180 to 180
        lat = nc_data.variables['lat'][i_track, :][:val_len]
        cen_pres = nc_data.variables['pres'][i_track, :][:val_len]
        av_prec = nc_data.variables['precavg'][i_track, :][:val_len]
        max_prec = nc_data.variables['precmax'][i_track, :][:val_len]

        wind = nc_data.variables['wind'][i_track, :][:val_len]*mps2kts*scale_to_10m  # m/s to kn
        if not all(wind.data):  # if wind is empty
            wind = np.ones(wind.size)*-999.9

        tr_df = pd.DataFrame({'time': datetimes, 'lat': lat, 'lon': lon,
                              'max_sustained_wind': wind,
                              'central_pressure': cen_pres,
                              'environmental_pressure': np.ones(lat.size)*1015.,
                              'radius_max_wind': np.ones(lat.size)*65.,
                              'maximum_precipitation': max_prec,
                              'average_precipitation': av_prec,
                              'basins': basin,
                              'time_step': time_step})

        # construct xarray
        tr_ds = xr.Dataset.from_dataframe(tr_df.set_index('time'))
        tr_ds.coords['lat'] = ('time', tr_ds.lat)
        tr_ds.coords['lon'] = ('time', tr_ds.lon)
        tr_ds.attrs = {'max_sustained_wind_unit': 'kn',
                       'central_pressure_unit': 'mb',
                       'sid': sid,
                       'name': sid, 'orig_event_flag': False,
                       'basin': basin[0],
                       'category': set_category(wind, 'kn')}
        self.data.append(tr_ds)

    def equal_timestep(self, time_step_h=1, land_params=False):
        """ Generate interpolated track values to time steps of min_time_step.
        Parameters:
            time_step_h (float, optional): time step in hours to which to
                interpolate. Default: 1.
            land_params (bool, optional): compute on_land and dist_since_lf at
                each node. Default: False.
        """
        LOGGER.info('Interpolating %s tracks to %sh time steps.', self.size,
                    time_step_h)

        if land_params:
            land_geom = _calc_land_geom(self.data)
        else:
            land_geom = None

        if self.pool:
            chunksize = min(self.size//self.pool.ncpus, 1000)
            self.data = self.pool.map(self._one_interp_data, self.data,
                                      itertools.repeat(time_step_h, self.size),
                                      itertools.repeat(land_geom, self.size),
                                      chunksize=chunksize)
        else:
            new_data = list()
            for track in self.data:
                new_data.append(self._one_interp_data(track, time_step_h,
                                                      land_geom))
            self.data = new_data

    def calc_random_walk(self, ens_size=9, ens_amp0=1.5, ens_amp=0.1, \
        max_angle=np.pi/10, seed=CONFIG['trop_cyclone']['random_seed'], decay=True):
        """
        Generate synthetic tracks based on directed random walk. An ensemble of
        tracks is computed for every track contained.
        Please note that there is a bias towards higher latitudes in the random
        wiggle. The wiggles are applied for each timestep. Please consider using
        equal_timestep() for unification before generating synthetic tracks.
        Be careful when changing ens_amp and max_angle and test changes of the
        parameter values before application.

        Parameters:
            ens_size (int, optional): number of ensemble members per track.
                Default 9.
            ens_amp0 (float, optional): amplitude of max random starting point
                shift in decimal degree (longitude and latitude). Default: 1.5
            ens_amp (float, optional): amplitude of random walk wiggles in
                decimal degree (longitude and latitude). Default: 0.1
            max_angle (float, optional): maximum angle of variation. Default: pi/10.
                - max_angle=pi results in undirected random change with
                    no change in direction;
                - max_angle=0 (or very close to 0) is not recommended. It results
                    in non-random synthetic tracks with constant shift to higher latitudes;
                - for 0<max_angle<pi/2, the change in latitude is always toward
                    higher latitudes, i.e. poleward,
                - max_angle=pi/4 results in random angles within one quadrant,
                    also with a poleward bias;
                - decreasing max_angle starting from pi will gradually increase
                    the spread of the synthetic tracks until they start converging
                    towards the result of max_angle=0 at a certain value (depending
                    on length of timesteps and ens_amp).
            seed (int, optional): random number generator seed for replicability
                of random walk. Put negative value if you don't want to use it.
                Default: configuration file
            decay (bool, optional): compute land decay in probabilistic tracks.
                Default: True
        """
        LOGGER.info('Computing %s synthetic tracks.', ens_size*self.size)

        if max_angle==0:
            LOGGER.warning('max_angle=0 is not recommended. It results in non-random \
                         synthetic tracks with a constant shift to higher latitudes.')
        if seed >= 0:
            np.random.seed(seed)

        random_vec = list()
        for track in self.data:
            random_vec.append(np.random.uniform(size=ens_size*(2+track.time.size)))

        num_tracks = self.size
        new_ens = list()
        if self.pool:
            chunksize = min(num_tracks//self.pool.ncpus, 1000)
            new_ens = self.pool.map(self._one_rnd_walk, self.data,
                                    itertools.repeat(ens_size, num_tracks),
                                    itertools.repeat(ens_amp0, num_tracks),
                                    itertools.repeat(ens_amp, num_tracks),
                                    itertools.repeat(max_angle, num_tracks),
                                    random_vec, chunksize=chunksize)
        else:
            for i_track, track in enumerate(self.data):
                new_ens.append(self._one_rnd_walk(track, ens_size, ens_amp0, \
                               ens_amp, max_angle, random_vec[i_track]))
        self.data = list()
        for ens_track in new_ens:
            self.data.extend(ens_track)

        if decay:
            try:
                land_geom = _calc_land_geom(self.data)
                v_rel, p_rel = self._calc_land_decay(land_geom)
                self._apply_land_decay(v_rel, p_rel, land_geom)
            except ValueError as err:
                LOGGER.info('No land decay coefficients could be applied. %s',
                            str(err))

    @property
    def size(self):
        """ Get longitude from coord array """
        return len(self.data)

    def plot(self, axis=None, **kwargs):
        """Track over earth. Historical events are blue, probabilistic black.

        Parameters:
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for LineCollection matplotlib, e.g. alpha=0.5

        Returns:
            matplotlib.axes._subplots.AxesSubplot
        """
        if 'lw' not in kwargs:
            kwargs['lw'] = 2
        if 'transform' not in kwargs:
            kwargs['transform'] = ccrs.PlateCarree()

        if not self.size:
            LOGGER.info('No tracks to plot')
            return None

        deg_border = 0.5
        if not axis:
            _, axis = u_plot.make_map()
        min_lat, max_lat = 10000, -10000
        min_lon, max_lon = 10000, -10000
        for track in self.data:
            min_lat, max_lat = min(min_lat, np.min(track.lat.values)), \
                               max(max_lat, np.max(track.lat.values))
            min_lon, max_lon = min(min_lon, np.min(track.lon.values)), \
                               max(max_lon, np.max(track.lon.values))
        min_lon, max_lon = min_lon-deg_border, max_lon+deg_border
        min_lat, max_lat = min_lat-deg_border, max_lat+deg_border
        if abs(min_lon - max_lon) > 360:
            min_lon, max_lon = -180, 180
        axis.set_extent(([min_lon, max_lon, min_lat, max_lat]), crs=kwargs['transform'])
        u_plot.add_shapes(axis)

        synth_flag = False
        cmap = ListedColormap(colors=CAT_COLORS)
        norm = BoundaryNorm([0] + SAFFIR_SIM_CAT, len(SAFFIR_SIM_CAT))
        for track in self.data:
            points = np.array([track.lon.values,
                               track.lat.values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            try:
                segments = np.delete(segments, np.argwhere(segments[:, 0, 0] * \
                                     segments[:, 1, 0] < 0).reshape(-1), 0)
            except IndexError:
                pass
            if track.orig_event_flag:
                track_lc = LineCollection(segments, cmap=cmap, norm=norm, \
                    linestyle='solid', **kwargs)
            else:
                synth_flag = True
                track_lc = LineCollection(segments, cmap=cmap, norm=norm, \
                    linestyle=':', **kwargs)
            track_lc.set_array(track.max_sustained_wind.values)
            axis.add_collection(track_lc)

        leg_lines = [Line2D([0], [0], color=CAT_COLORS[i_col], lw=2)
                     for i_col in range(len(SAFFIR_SIM_CAT))]
        leg_names = [CAT_NAMES[i_col] for i_col
                     in range(1, len(SAFFIR_SIM_CAT)+1)]
        if synth_flag:
            leg_lines.append(Line2D([0], [0], color='grey', lw=2, ls='solid'))
            leg_lines.append(Line2D([0], [0], color='grey', lw=2, ls=':'))
            leg_names.append('Historical')
            leg_names.append('Synthetic')

        axis.legend(leg_lines, leg_names, loc=0)
        return axis

    def write_netcdf(self, folder_name):
        """ Write a netcdf file per track with track.sid name in given folder.

        Parameter:
            folder_name (str): folder name where to write files
        """
        list_path = [os.path.join(folder_name, track.sid+'.nc') for track in self.data]
        LOGGER.info('Writting %s files.', self.size)
        for track in self.data:
            track.attrs['orig_event_flag'] = int(track.orig_event_flag)
        xr.save_mfdataset(self.data, list_path)

    def read_netcdf(self, folder_name):
        """ Read all netcdf files contained in folder and fill a track per file.

        Parameters:
            folder_name (str): folder name where to write files
        """
        file_tr = get_file_names(folder_name)
        LOGGER.info('Reading %s files.', len(file_tr))
        self.data = list()
        for file in file_tr:
            if not os.path.splitext(file)[1] == '.nc':
                continue
            track = xr.open_dataset(file)
            track.attrs['orig_event_flag'] = bool(track.orig_event_flag)
            self.data.append(track)

    @staticmethod
    @jit(parallel=True)
    def _one_rnd_walk(track, ens_size, ens_amp0, ens_amp, max_angle, rnd_vec):
        """ Interpolate values of one track.

        Parameters:
            track (xr.Dataset): track data

        Returns:
            list(xr.Dataset)
        """
        ens_track = list()
        n_dat = track.time.size
        rand_unif_ini = rnd_vec[:2*ens_size].reshape((2, ens_size))
        rand_unif_ang = rnd_vec[2*ens_size:]

        xy_ini = ens_amp0 * (rand_unif_ini - 0.5)
        tmp_ang = np.cumsum(2 * max_angle * rand_unif_ang - max_angle)
        coord_xy = np.empty((2, ens_size * n_dat))
        coord_xy[0] = np.cumsum(ens_amp * np.sin(tmp_ang))
        coord_xy[1] = np.cumsum(ens_amp * np.cos(tmp_ang))

        ens_track.append(track)
        for i_ens in range(ens_size):
            i_track = track.copy(True)

            d_xy = coord_xy[:, i_ens * n_dat: (i_ens + 1) * n_dat] - \
                np.expand_dims(coord_xy[:, i_ens * n_dat], axis=1)
            # change sign of latitude change for southern hemishpere:
            d_xy = np.sign(track.lat.values[0]) * d_xy

            d_lat_lon = d_xy + np.expand_dims(xy_ini[:, i_ens], axis=1)

            i_track.lon.values = i_track.lon.values + d_lat_lon[0, :]
            i_track.lat.values = i_track.lat.values + d_lat_lon[1, :]
            i_track.attrs['orig_event_flag'] = False
            i_track.attrs['name'] = i_track.attrs['name'] + '_gen' + str(i_ens+1)
            i_track.attrs['sid'] = i_track.attrs['sid'] + '_gen' + str(i_ens+1)
            i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens+1)/100

            ens_track.append(i_track)

        return ens_track

    @staticmethod
    @jit(parallel=True, forceobj=True)
    def _one_interp_data(track, time_step_h, land_geom=None):
        """ Interpolate values of one track.

        Parameters:
            track (xr.Dataset): track data

        Returns:
            xr.Dataset
        """
        if track.time.size > 3:
            time_step = '{}H'.format(time_step_h)
            track_int = track.resample(time=time_step).interpolate('linear')
            track_int['time_step'] = ('time', track_int.time.size * [time_step_h])
            # handle change of sign in longitude
            pos_lon = track.coords['lon'].values > 0
            neg_lon = track.coords['lon'].values <= 0
            if neg_lon.any() and pos_lon.any() and \
            np.any(abs(track.coords['lon'].values[pos_lon]) > 170):
                if neg_lon[0]:
                    track.coords['lon'].values[pos_lon] -= 360
                    track_int.coords['lon'] = track.lon.resample(time=time_step).\
                    interpolate('cubic')
                    track_int.coords['lon'][track_int.coords['lon'] < -180] += 360
                else:
                    track.coords['lon'].values[neg_lon] += 360
                    track_int.coords['lon'] = track.lon.resample(time=time_step).\
                    interpolate('cubic')
                    track_int.coords['lon'][track_int.coords['lon'] > 180] -= 360
            else:
                track_int.coords['lon'] = track.lon.resample(time=time_step).\
                    interpolate('cubic')
            track_int.coords['lat'] = track.lat.resample(time=time_step).\
                                      interpolate('cubic')
            track_int.attrs = track.attrs
            track_int.attrs['category'] = set_category( \
                track.max_sustained_wind.values, \
                track.max_sustained_wind_unit)
        else:
            LOGGER.warning('Track interpolation not done. ' \
                           'Not enough elements for %s', track.name)
            track_int = track

        if land_geom:
            _track_land_params(track_int, land_geom)
        return track_int

    def _calc_land_decay(self, land_geom, s_rel=True, check_plot=False):
        """Compute wind and pressure decay coefficients for every TC category
        from the historical events according to the formulas:
            - wind decay = exp(-x*A)
            - pressure decay = S-(S-1)*exp(-x*B)

        Parameters:
            land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
            s_rel (bool, optional): use environmental presure to calc S value
                (true) or central presure (false)
            check_plot (bool, optional): visualize computed coefficients.
                Default: False

        Returns:
            v_rel (dict(category: A)), p_rel (dict(category: (S, B)))
        """
        hist_tracks = [track for track in self.data if track.orig_event_flag]
        if not hist_tracks:
            LOGGER.error('No historical tracks contained. Historical tracks' \
                         ' are needed.')
            raise ValueError

        # Key is Saffir-Simpson scale
        # values are lists of wind/wind at landfall
        v_lf = dict()
        # values are tuples with first value the S parameter, second value
        # list of central pressure/central pressure at landfall
        p_lf = dict()
        # x-scale values to compute landfall decay
        x_val = dict()

        dec_val = list()
        if self.pool:
            chunksize = min(len(hist_tracks)//self.pool.ncpus, 1000)
            dec_val = self.pool.map(_decay_values, hist_tracks, itertools.repeat(land_geom),
                                    itertools.repeat(s_rel), chunksize=chunksize)
        else:
            for track in hist_tracks:
                dec_val.append(_decay_values(track, land_geom, s_rel))

        for (tv_lf, tp_lf, tx_val) in dec_val:
            for key in tv_lf.keys():
                v_lf.setdefault(key, []).extend(tv_lf[key])
                p_lf.setdefault(key, ([], []))
                p_lf[key][0].extend(tp_lf[key][0])
                p_lf[key][1].extend(tp_lf[key][1])
                x_val.setdefault(key, []).extend(tx_val[key])

        v_rel, p_rel = _decay_calc_coeff(x_val, v_lf, p_lf)
        if check_plot:
            _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel)

        return v_rel, p_rel

    def _apply_land_decay(self, v_rel, p_rel, land_geom, s_rel=True,
                          check_plot=False):
        """Compute wind and pressure decay due to landfall in synthetic tracks.

        Parameters:
            v_rel (dict): {category: A}, where wind decay = exp(-x*A)
            p_rel (dict): (category: (S, B)}, where pressure decay
                = S-(S-1)*exp(-x*B)
            land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
            s_rel (bool, optional): use environmental presure to calc S value
                (true) or central presure (false)
            check_plot (bool, optional): visualize computed changes
        """
        sy_tracks = [track for track in self.data if not track.orig_event_flag]
        if not sy_tracks:
            LOGGER.error('No synthetic tracks contained. Synthetic tracks' \
                         ' are needed.')
            raise ValueError

        if not v_rel or not p_rel:
            LOGGER.info('No decay coefficients.')
            return

        if check_plot:
            orig_wind, orig_pres = [], []
            for track in sy_tracks:
                orig_wind.append(np.copy(track.max_sustained_wind.values))
                orig_pres.append(np.copy(track.central_pressure.values))

        if self.pool:
            chunksize = min(self.size//self.pool.ncpus, 1000)
            self.data = self.pool.map(_apply_decay_coeffs, self.data,
                                      itertools.repeat(v_rel), itertools.repeat(p_rel),
                                      itertools.repeat(land_geom), itertools.repeat(s_rel),
                                      chunksize=chunksize)
        else:
            new_data = list()
            for track in self.data:
                new_data.append(_apply_decay_coeffs(track, v_rel, p_rel, \
                    land_geom, s_rel))
            self.data = new_data

        if check_plot:
            _check_apply_decay_plot(self.data, orig_wind, orig_pres)

    def _read_one_csv(self, file_name):
        """Read IBTrACS track file.

            Parameters:
                file_name (str): file name containing one IBTrACS track to read
        """
        LOGGER.info('Reading %s', file_name)
        dfr = pd.read_csv(file_name)
        name = dfr['ibtracsID'].values[0]

        datetimes = list()
        for time in dfr['isotime'].values:
            year = np.fix(time/1e6)
            time = time - year*1e6
            month = np.fix(time/1e4)
            time = time - month*1e4
            day = np.fix(time/1e2)
            hour = time - day*1e2
            datetimes.append(dt.datetime(int(year), int(month), int(day), \
                                         int(hour)))

        lat = dfr['cgps_lat'].values.astype('float')
        lon = dfr['cgps_lon'].values.astype('float')
        cen_pres = dfr['pcen'].values.astype('float')
        max_sus_wind = dfr['vmax'].values.astype('float')
        max_sus_wind_unit = 'kn'
        cen_pres = _missing_pressure(cen_pres, max_sus_wind, lat, lon)

        tr_ds = xr.Dataset()
        tr_ds.coords['time'] = ('time', datetimes)
        tr_ds.coords['lat'] = ('time', lat)
        tr_ds.coords['lon'] = ('time', lon)
        tr_ds['time_step'] = ('time', dfr['tint'].values)
        tr_ds['radius_max_wind'] = ('time', dfr['rmax'].values.astype('float'))
        tr_ds['max_sustained_wind'] = ('time', max_sus_wind)
        tr_ds['central_pressure'] = ('time', cen_pres)
        tr_ds['environmental_pressure'] = ('time', \
                                           dfr['penv'].values.astype('float'))
        tr_ds.attrs['max_sustained_wind_unit'] = max_sus_wind_unit
        tr_ds.attrs['central_pressure_unit'] = 'mb'
        tr_ds.attrs['name'] = name
        tr_ds.attrs['sid'] = name
        tr_ds.attrs['orig_event_flag'] = bool(dfr['original_data']. values[0])
        tr_ds.attrs['data_provider'] = dfr['data_provider'].values[0]
        tr_ds.attrs['basin'] = dfr['gen_basin'].values[0]
        try:
            tr_ds.attrs['id_no'] = float(name.replace('N', '0'). \
                                         replace('S', '1'))
        except ValueError:
            tr_ds.attrs['id_no'] = float(str(datetimes[0].date()). \
                                         replace('-', ''))
        tr_ds.attrs['category'] = set_category(max_sus_wind, \
                   max_sus_wind_unit)

        self.data.append(tr_ds)

    @staticmethod
    def _filter_ibtracs(fn_nc, storm_id, year_range, basin):
        """ Select tracks from input conditions.

        Parameters:
            fn_nc (str): ibtracs netcdf data file name
            storm_id (str os list): ibtrac id of the storm
            year_range(tuple): (min_year, max_year)
            basin (str): e.g. US, SA, NI, SI, SP, WP, EP, NA

        Returns:
            np.array
        """
        nc_data = nc.Dataset(fn_nc)
        storm_ids = [''.join(name.astype(str))
                     for name in nc_data.variables['sid']]
        sel_tracks = []
        # filter name
        if storm_id:
            if not isinstance(storm_id, list):
                storm_id = [storm_id]
            for storm in storm_id:
                sel_tracks.append(storm_ids.index(storm))
            sel_tracks = np.array(sel_tracks)
        else:
            # filter years
            years = np.array([int(iso_name[:4]) for iso_name in storm_ids])
            sel_tracks = np.argwhere(np.logical_and(years >= year_range[0], \
                years <= year_range[1])).reshape(-1)
            if not sel_tracks.size:
                LOGGER.info('No tracks in time range (%s, %s).', year_range[0],
                            year_range[1])
                return sel_tracks
            # filter basin
            if basin:
                basin0 = np.array([''.join(bas.astype(str)) \
                    for bas in nc_data.variables['basin'][:, 0, :]])[sel_tracks]
                sel_bas = np.argwhere(basin0 == basin).reshape(-1)
                if not sel_tracks.size:
                    LOGGER.info('No tracks in basin %s.', basin)
                    return sel_tracks
                sel_tracks = sel_tracks[sel_bas]
        return sel_tracks

    def _read_one_raw(self, nc_data, i_track, provider, correct_pres=False):
        """Fill given track.

            Parameters:
            nc_data (Dataset): netcdf data set
            i_track (int): track position in netcdf data
            provider (str): data provider. e.g. usa, newdelhi, bom, cma, tokyo
        """
        name = ''.join(nc_data.variables['name'][i_track] \
            [nc_data.variables['name'][i_track].mask == False].data.astype(str))
        sid = ''.join(nc_data.variables['sid'][i_track].astype(str))
        basin = ''.join(nc_data.variables['basin'][i_track, 0, :].astype(str))
        LOGGER.info('Reading %s: %s', sid, name)

        isot = nc_data.variables['iso_time'][i_track, :, :]
        val_len = isot.mask[isot.mask == False].shape[0]//isot.shape[1]
        datetimes = list()
        for date_time in isot[:val_len]:
            datetimes.append(dt.datetime.strptime(''.join(date_time.astype(str)),
                                                  '%Y-%m-%d %H:%M:%S'))

        id_no = float(sid.replace('N', '0').replace('S', '1'))
        lat = nc_data.variables[provider + '_lat'][i_track, :][:val_len]
        lon = nc_data.variables[provider + '_lon'][i_track, :][:val_len]

        max_sus_wind = nc_data.variables[provider + '_wind'][i_track, :]. \
            data[:val_len].astype(float)
        cen_pres = nc_data.variables[provider + '_pres'][i_track, :]. \
            data[:val_len].astype(float)

        if correct_pres:
            cen_pres = _missing_pressure(cen_pres, max_sus_wind, lat, lon)

        if np.all(lon == nc_data.variables[provider + '_lon']._FillValue) or \
        (np.any(lon == nc_data.variables[provider + '_lon']._FillValue) and \
        np.all(max_sus_wind == nc_data.variables[provider + '_wind']._FillValue) \
        and np.all(cen_pres == nc_data.variables[provider + '_pres']._FillValue)):
            LOGGER.warning('Skipping %s. It does not contain valid values. ' +\
                           'Try another provider.', sid)
            return None

        try:
            rmax = nc_data.variables[provider + '_rmw'][i_track, :][:val_len]
        except KeyError:
            LOGGER.info('%s: No rmax for given provider %s. Set to default.',
                        sid, provider)
            rmax = np.zeros(lat.size)
        try:
            penv = nc_data.variables[provider + '_poci'][i_track, :][:val_len]
        except KeyError:
            LOGGER.info('%s: No penv for given provider %s. Set to default.',
                        sid, provider)
            penv = np.ones(lat.size)*self._set_penv(basin)

        tr_ds = pd.DataFrame({'time': datetimes, 'lat': lat, 'lon':lon, \
            'radius_max_wind': rmax.astype('float'), 'max_sustained_wind': max_sus_wind, \
            'central_pressure': cen_pres, 'environmental_pressure': penv.astype('float')})

        # deal with nans
        tr_ds = self._deal_nans(tr_ds, nc_data, provider, datetimes, basin)
        if not tr_ds.shape[0]:
            LOGGER.warning('Skipping %s. No usable data.', sid)
            return None
        # ensure environmental pressure > central pressure
        chg_pres = (tr_ds.central_pressure > tr_ds.environmental_pressure).values
        tr_ds.environmental_pressure.values[chg_pres] = tr_ds.central_pressure.values[chg_pres]

        # construct xarray
        tr_ds = xr.Dataset.from_dataframe(tr_ds.set_index('time'))
        tr_ds.coords['lat'] = ('time', tr_ds.lat)
        tr_ds.coords['lon'] = ('time', tr_ds.lon)
        tr_ds.attrs = {'max_sustained_wind_unit': 'kn', 'central_pressure_unit': 'mb', \
            'name': name, 'sid': sid, 'orig_event_flag': True, 'data_provider': provider, \
            'basin': basin, 'id_no': id_no, 'category': set_category(max_sus_wind, 'kn')}
        return tr_ds

    def _deal_nans(self, tr_ds, nc_data, provider, datetimes, basin):
        """ Remove or substitute fill values of netcdf variables. """
        # remove nan coordinates
        tr_ds.drop(tr_ds[tr_ds.lat == nc_data.variables[provider + '_lat']. \
            _FillValue].index, inplace=True)
        tr_ds.drop(tr_ds[np.isnan(tr_ds.lat.values)].index, inplace=True)
        tr_ds.drop(tr_ds[tr_ds.lon == nc_data.variables[provider + '_lon']. \
            _FillValue].index, inplace=True)
        tr_ds.drop(tr_ds[np.isnan(tr_ds.lon.values)].index, inplace=True)
        # remove nan central pressures
        tr_ds.drop(tr_ds[tr_ds.central_pressure == nc_data.variables[provider + '_pres']. \
            _FillValue].index, inplace=True)
        # remove repeated dates
        tr_ds.drop_duplicates('time', inplace=True)
        # fill nans of environmental_pressure and radius_max_wind
        try:
            tr_ds.environmental_pressure.values[tr_ds.environmental_pressure == \
                nc_data.variables[provider + '_poci']._FillValue] = np.nan
            tr_ds.environmental_pressure = tr_ds.environmental_pressure.ffill(limit=4). \
                bfill(limit=4).fillna(self._set_penv(basin))
        except KeyError:
            pass
        try:
            tr_ds.radius_max_wind.values[tr_ds.radius_max_wind == \
                nc_data.variables[provider + '_rmw']._FillValue] = np.nan
            tr_ds['radius_max_wind'] = tr_ds.radius_max_wind.ffill(limit=1).bfill(limit=1).fillna(0)
        except KeyError:
            pass
        # set time steps
        tr_ds['time_step'] = np.zeros(tr_ds.shape[0])
        for i_time, time in enumerate(tr_ds.time[1:], 1):
            tr_ds.time_step.values[i_time] = (time - datetimes[i_time-1]).total_seconds()/3600
        if tr_ds.shape[0]:
            tr_ds.time_step.values[0] = tr_ds.time_step.values[-1]

        return tr_ds

    @staticmethod
    def _set_penv(basin):
        """ Set environmental pressure depending on basin """
        penv = 1010
        if basin in ('NI', 'SI', 'WP'):
            penv = 1005
        elif basin == 'SP':
            penv = 1004
        return penv


def _calc_land_geom(ens_track):
    """Compute land geometry used for land distance computations.

    Returns:
        shapely.geometry.multipolygon.MultiPolygon
    """
    deg_buffer = 0.1
    min_lat = np.min([np.min(track.lat.values) for track in ens_track])
    min_lat = max(min_lat-deg_buffer, -90)

    max_lat = np.max([np.max(track.lat.values) for track in ens_track])
    max_lat = min(max_lat+deg_buffer, 90)

    min_lon = np.min([np.min(track.lon.values) for track in ens_track])
    min_lon = max(min_lon-deg_buffer, -180)

    max_lon = np.max([np.max(track.lon.values) for track in ens_track])
    max_lon = min(max_lon+deg_buffer, 180)

    return coord_util.get_land_geometry(extent=(min_lon, max_lon, \
        min_lat, max_lat), resolution=10)

def _track_land_params(track, land_geom):
    """ Compute parameters of land for one track.

    Parameters:
        track (xr.Dataset): track values
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
    """
    track['on_land'] = ('time', coord_util.coord_on_land(track.lat.values, \
         track.lon.values, land_geom))
    track['dist_since_lf'] = ('time', _dist_since_lf(track))

def _dist_since_lf(track):
    """ Compute the distance to landfall in km point for every point on land.
    Points on water get nan values.

    Parameters:
        track (xr.Dataset): tropical cyclone track

    Returns:
        np.arrray
    """
    dist_since_lf = np.zeros(track.time.values.shape)

    # Index in sea that follows a land index
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0]
    if not sea_land_idx.size:
        return (dist_since_lf+1)*np.nan

    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    orig_lf = _calc_orig_lf(track, sea_land_idx)

    dist = DistanceMetric.get_metric('haversine')
    nodes1 = np.radians(np.array([track.lat.values[1:],
                                  track.lon.values[1:]]).transpose())
    nodes0 = np.radians(np.array([track.lat.values[:-1],
                                  track.lon.values[:-1]]).transpose())
    dist_since_lf[1:] = dist.pairwise(nodes1, nodes0).diagonal()
    dist_since_lf[np.logical_not(track.on_land.values)] = 0.0
    nodes1 = np.array([track.lat.values[sea_land_idx+1],
                       track.lon.values[sea_land_idx+1]]).transpose()/180*np.pi
    dist_since_lf[sea_land_idx+1] = \
        dist.pairwise(nodes1, orig_lf/180*np.pi).diagonal()
    for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
        dist_since_lf[sea_land+1:land_sea] = \
            np.cumsum(dist_since_lf[sea_land+1:land_sea])

    dist_since_lf *= EARTH_RADIUS_KM
    dist_since_lf[np.logical_not(track.on_land.values)] = np.nan

    return dist_since_lf

def _calc_orig_lf(track, sea_land_idx):
    """ Approximate coast coordinates in landfall as the middle point
    before landfall and after.

    Parameters:
        track (xr.Dataset): TC track
        sea_land_idx (np.array): array position of sea before landfall

    Returns:
        np.array (first column lat and second lon of each landfall coord)
    """
    # TODO change to pos where landfall (v_landfall)??
    orig_lf = np.empty((sea_land_idx.size, 2))
    for i_lf, lf_point in enumerate(sea_land_idx):
        orig_lf[i_lf][0] = track.lat[lf_point] + \
            (track.lat[lf_point+1] - track.lat[lf_point])/2
        orig_lf[i_lf][1] = track.lon[lf_point] + \
            (track.lon[lf_point+1] - track.lon[lf_point])/2
    return orig_lf

def _decay_v_function(a_coef, x_val):
    """Decay function used for wind after landfall."""
    return np.exp(-a_coef * x_val)

def _solve_decay_v_function(v_y, x_val):
    """Solve decay function used for wind after landfall. Get A coefficient."""
    return -np.log(v_y) / x_val

def _decay_p_function(s_coef, b_coef, x_val):
    """Decay function used for pressure after landfall."""
    return s_coef - (s_coef - 1) * np.exp(-b_coef*x_val)

def _solve_decay_p_function(ps_y, p_y, x_val):
    """Solve decay function used for pressure after landfall.
    Get B coefficient."""
    return -np.log((ps_y - p_y)/(ps_y - 1.0)) / x_val

def _calc_decay_ps_value(track, p_landfall, pos, s_rel):
    if s_rel:
        p_land_s = track.environmental_pressure[pos].values
    else:
        p_land_s = track.central_pressure[pos].values
    return float(p_land_s / p_landfall)

def _decay_values(track, land_geom, s_rel):
    """ Compute wind and pressure relative to landafall values.

    Parameters:
        track (xr.Dataset): track
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool): use environmental presure for S value (true) or
            central presure (false)

    Returns:
        v_lf (dict): key is Saffir-Simpson scale, values are arrays of
            wind/wind at landfall
        p_lf (dict): key is Saffir-Simpson scale, values are tuples with
            first value array of S parameter, second value array of central
            pressure/central pressure at landfall
        x_val (dict): key is Saffir-Simpson scale, values are arrays with
            the values used as "x" in the coefficient fitting, the
            distance since landfall
    """
    v_lf = dict()
    p_lf = dict()
    x_val = dict()

    _track_land_params(track, land_geom)
    # Index in land that comes from previous sea index
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
        for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
            v_landfall = track.max_sustained_wind[sea_land-1].values
            ss_scale_idx = np.where(v_landfall < SAFFIR_SIM_CAT)[0][0]+1

            v_land = track.max_sustained_wind[sea_land-1:land_sea].values
            if v_land[0] > 0:
                v_land = (v_land[1:]/v_land[0]).tolist()
            else:
                v_land = v_land[1:].tolist()

            p_landfall = float(track.central_pressure[sea_land-1].values)
            p_land = track.central_pressure[sea_land-1:land_sea].values
            p_land = (p_land[1:]/p_land[0]).tolist()

            p_land_s = _calc_decay_ps_value(track, p_landfall, land_sea-1, s_rel)
            p_land_s = len(p_land)*[p_land_s]

            if ss_scale_idx not in v_lf:
                v_lf[ss_scale_idx] = array.array('f', v_land)
                p_lf[ss_scale_idx] = (array.array('f', p_land_s),
                                      array.array('f', p_land))
                x_val[ss_scale_idx] = array.array('f', \
                                      track.dist_since_lf[sea_land:land_sea])
            else:
                v_lf[ss_scale_idx].extend(v_land)
                p_lf[ss_scale_idx][0].extend(p_land_s)
                p_lf[ss_scale_idx][1].extend(p_land)
                x_val[ss_scale_idx].extend(track.dist_since_lf[ \
                                           sea_land:land_sea])
    return v_lf, p_lf, x_val

def _decay_calc_coeff(x_val, v_lf, p_lf):
    """ From track's relative velocity and pressure, compute the decay
    coefficients.
    - wind decay = exp(-x*A)
    - pressure decay = S-(S-1)*exp(-x*A)

    Parameters:
        x_val (dict): key is Saffir-Simpson scale, values are lists with
            the values used as "x" in the coefficient fitting, the
            distance since landfall
        v_lf (dict): key is Saffir-Simpson scale, values are lists of
            wind/wind at landfall
        p_lf (dict): key is Saffir-Simpson scale, values are tuples with
            first value the S parameter, second value list of central
            pressure/central pressure at landfall

    Returns:
        v_rel (dict()), p_rel (dict())
    """
    np.warnings.filterwarnings('ignore')
    v_rel = dict()
    p_rel = dict()
    for ss_scale, val_lf in v_lf.items():
        x_val_ss = np.array(x_val[ss_scale])

        y_val = np.array(val_lf)
        v_coef = _solve_decay_v_function(y_val, x_val_ss)
        v_coef = v_coef[np.isfinite(v_coef)]
        v_coef = np.mean(v_coef)

        ps_y_val = np.array(p_lf[ss_scale][0])
        y_val = np.array(p_lf[ss_scale][1])
        y_val[ps_y_val <= y_val] = np.nan
        y_val[ps_y_val <= 1] = np.nan
        valid_p = np.isfinite(y_val)
        ps_y_val = ps_y_val[valid_p]
        y_val = y_val[valid_p]
        p_coef = _solve_decay_p_function(ps_y_val, y_val, x_val_ss[valid_p])
        ps_y_val = np.mean(ps_y_val)
        p_coef = np.mean(p_coef)

        if np.isfinite(v_coef) and np.isfinite(ps_y_val) and np.isfinite(ps_y_val):
            v_rel[ss_scale] = v_coef
            p_rel[ss_scale] = (ps_y_val, p_coef)

    scale_fill = np.array(list(p_rel.keys()))
    if not scale_fill.size:
        LOGGER.info('No historical track with landfall.')
        return v_rel, p_rel
    for ss_scale in range(1, len(SAFFIR_SIM_CAT)+1):
        if ss_scale not in p_rel:
            close_scale = scale_fill[np.argmin(np.abs(scale_fill-ss_scale))]
            LOGGER.debug('No historical track of category %s with landfall. ' \
                         'Decay parameters from category %s taken.',
                         CAT_NAMES[ss_scale], CAT_NAMES[close_scale])
            v_rel[ss_scale] = v_rel[close_scale]
            p_rel[ss_scale] = p_rel[close_scale]

    return v_rel, p_rel

def _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel):
    """ Generate one graph with wind decay and an other with central pressure
    decay, true and approximated."""
    # One graph per TC category
    for track_cat, color in zip(v_lf.keys(),
                                cm_mp.rainbow(np.linspace(0, 1, len(v_lf)))):
        _, axes = plt.subplots(2, 1)
        x_eval = np.linspace(0, np.max(x_val[track_cat]), 20)

        axes[0].set_xlabel('Distance from landfall (km)')
        axes[0].set_ylabel('Max sustained wind relative to landfall')
        axes[0].set_title('Wind')
        axes[0].plot(x_val[track_cat], v_lf[track_cat], '*', c=color,
                     label=CAT_NAMES[track_cat])
        axes[0].plot(x_eval, _decay_v_function(v_rel[track_cat], x_eval),
                     '-', c=color)

        axes[1].set_xlabel('Distance from landfall (km)')
        axes[1].set_ylabel('Central pressure relative to landfall')
        axes[1].set_title('Pressure')
        axes[1].plot(x_val[track_cat], p_lf[track_cat][1], '*', c=color,
                     label=CAT_NAMES[track_cat])
        axes[1].plot(x_eval, _decay_p_function(p_rel[track_cat][0], \
                     p_rel[track_cat][1], x_eval), '-', c=color)

def _apply_decay_coeffs(track, v_rel, p_rel, land_geom, s_rel):
    """ Change track's max sustained wind and central pressure using the land
    decay coefficients.

    Parameters:
        track (xr.Dataset): TC track
        v_rel (dict): {category: A}, where wind decay = exp(-x*A)
        p_rel (dict): (category: (S, B)},
            where pressure decay = S-(S-1)*exp(-x*B)
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
        s_rel (bool): use environmental presure for S value (true) or
            central presure (false)

    Returns:
        xr.Dataset
    """
    # return if historical track
    if track.orig_event_flag:
        return track

    _track_land_params(track, land_geom)
    # Index in land that comes from previous sea index
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0] + 1
    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    if not sea_land_idx.size or land_sea_idx.size > sea_land_idx.size:
        return track
    for idx, (sea_land, land_sea) \
    in enumerate(zip(sea_land_idx, land_sea_idx)):
        v_landfall = track.max_sustained_wind[sea_land-1].values
        p_landfall = float(track.central_pressure[sea_land-1].values)
        try:
            ss_scale_idx = np.where(v_landfall < SAFFIR_SIM_CAT)[0][0]+1
        except IndexError:
            continue
        if land_sea - sea_land == 1:
            continue
        p_decay = _calc_decay_ps_value(track, p_landfall, land_sea-1, s_rel)
        p_decay = _decay_p_function(p_decay, p_rel[ss_scale_idx][1], \
            track.dist_since_lf[sea_land:land_sea].values)
        # dont applay decay if it would decrease central pressure
        p_decay[p_decay < 1] = track.central_pressure[sea_land:land_sea][p_decay < 1]/p_landfall
        track.central_pressure[sea_land:land_sea] = p_landfall * p_decay

        v_decay = _decay_v_function(v_rel[ss_scale_idx], \
            track.dist_since_lf[sea_land:land_sea].values)
        # dont applay decay if it would increas wind speeds
        v_decay[v_decay > 1] = track.max_sustained_wind[sea_land:land_sea][v_decay > 1]/v_landfall
        track.max_sustained_wind[sea_land:land_sea] = v_landfall * v_decay

        # correct values of sea between two landfalls
        if land_sea < track.time.size and idx+1 < sea_land_idx.size:
            rndn = 0.1 * float(np.abs(np.random.normal(size=1)*5)+6)
            r_diff = track.central_pressure[land_sea].values - \
                track.central_pressure[land_sea-1].values + rndn
            track.central_pressure[land_sea:sea_land_idx[idx+1]] += - r_diff

            rndn = rndn * 10 # mean value 10
            r_diff = track.max_sustained_wind[land_sea].values - \
                track.max_sustained_wind[land_sea-1].values - rndn
            track.max_sustained_wind[land_sea:sea_land_idx[idx+1]] += - r_diff

    # correct limits
    np.warnings.filterwarnings('ignore')
    cor_p = track.central_pressure.values > track.environmental_pressure.values
    track.central_pressure[cor_p] = track.environmental_pressure[cor_p]
    track.max_sustained_wind[track.max_sustained_wind < 0] = 0
    track.attrs['category'] = set_category(track.max_sustained_wind.values,
                                           track.max_sustained_wind_unit)
    return track

def _check_apply_decay_plot(all_tracks, syn_orig_wind, syn_orig_pres):
    """ Plot wind and presure before and after correction for synthetic tracks.
    Plot wind and presure for unchanged historical tracks."""
    # Plot synthetic tracks
    sy_tracks = [track for track in all_tracks if not track.orig_event_flag]
    graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a = \
    _check_apply_decay_syn_plot(sy_tracks, syn_orig_wind,
                                syn_orig_pres)

    # Plot historic tracks
    hist_tracks = [track for track in all_tracks if track.orig_event_flag]
    graph_hv, graph_hp, graph_hpd_a, graph_hped_a = \
    _check_apply_decay_hist_plot(hist_tracks)

    # Put legend and fix size
    leg_lines = [Line2D([0], [0], color=CAT_COLORS[i_col], lw=2)
                 for i_col in range(len(SAFFIR_SIM_CAT))]
    leg_lines.append(Line2D([0], [0], color='k', lw=2))
    leg_names = [CAT_NAMES[i_col] for i_col in range(1, len(SAFFIR_SIM_CAT)+1)]
    leg_names.append('Sea')
    all_gr = [graph_v_a, graph_v_b, graph_p_a, graph_p_b, graph_ped_a,
              graph_pd_a, graph_hv, graph_hp, graph_hpd_a, graph_hped_a]
    for graph in all_gr:
        graph.axs[0].legend(leg_lines, leg_names)
        fig, _ = graph.get_elems()
        fig.set_size_inches(18.5, 10.5)

def _check_apply_decay_syn_plot(sy_tracks, syn_orig_wind,
                                syn_orig_pres):
    """Plot winds and pressures of synthetic tracks before and after
    correction."""
    _, graph_v_b = plt.subplots()
    graph_v_b.set_title('Wind before land decay correction')
    graph_v_b.set_xlabel('Node number')
    graph_v_b.set_ylabel('Max sustained wind (kn)')

    _, graph_v_a = plt.subplots()
    graph_v_a.set_title('Wind after land decay correction')
    graph_v_a.set_xlabel('Node number')
    graph_v_a.set_ylabel('Max sustained wind (kn)')

    _, graph_p_b = plt.subplots()
    graph_p_b.set_title('Pressure before land decay correctionn')
    graph_p_b.set_xlabel('Node number')
    graph_p_b.set_ylabel('Central pressure (mb)')

    _, graph_p_a = plt.subplots()
    graph_p_a.set_title('Pressure after land decay correctionn')
    graph_p_a.set_xlabel('Node number')
    graph_p_a.set_ylabel('Central pressure (mb)')

    _, graph_pd_a = plt.subplots()
    graph_pd_a.set_title('Relative pressure after land decay correction')
    graph_pd_a.set_xlabel('Distance from landfall (km)')
    graph_pd_a.set_ylabel('Central pressure relative to landfall')

    _, graph_ped_a = plt.subplots()
    graph_ped_a.set_title('Environmental - central pressure after land decay correction')
    graph_ped_a.set_xlabel('Distance from landfall (km)')
    graph_ped_a.set_ylabel('Environmental pressure - Central pressure (mb)')

    for track, orig_wind, orig_pres in \
    zip(sy_tracks, syn_orig_wind, syn_orig_pres):
        # Index in land that comes from previous sea index
        sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0]+1
        # Index in sea that comes from previous land index
        land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0]+1
        if track.on_land[-1]:
            land_sea_idx = np.append(land_sea_idx, track.time.size)
        if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                v_lf = track.max_sustained_wind[sea_land-1].values
                p_lf = track.central_pressure[sea_land-1].values
                ss_scale = np.where(v_lf < SAFFIR_SIM_CAT)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_v_a.plot(on_land, track.max_sustained_wind[on_land],
                               'o', c=CAT_COLORS[ss_scale])
                graph_v_b.plot(on_land, orig_wind[on_land],
                               'o', c=CAT_COLORS[ss_scale])
                graph_p_a.plot(on_land, track.central_pressure[on_land],
                               'o', c=CAT_COLORS[ss_scale])
                graph_p_b.plot(on_land, orig_pres[on_land],
                               'o', c=CAT_COLORS[ss_scale])
                graph_pd_a.plot(track.dist_since_lf[on_land],
                                track.central_pressure[on_land]/p_lf,
                                'o', c=CAT_COLORS[ss_scale])
                graph_ped_a.plot(track.dist_since_lf[on_land],
                                 track.environmental_pressure[on_land]-
                                 track.central_pressure[on_land],
                                 'o', c=CAT_COLORS[ss_scale])

            on_sea = np.arange(track.time.size)[np.logical_not(track.on_land)]
            graph_v_a.plot(on_sea, track.max_sustained_wind[on_sea],
                           'o', c='k', markersize=5)
            graph_v_b.plot(on_sea, orig_wind[on_sea],
                           'o', c='k', markersize=5)
            graph_p_a.plot(on_sea, track.central_pressure[on_sea],
                           'o', c='k', markersize=5)
            graph_p_b.plot(on_sea, orig_pres[on_sea],
                           'o', c='k', markersize=5)

    return graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a

def _check_apply_decay_hist_plot(hist_tracks):
    """Plot winds and pressures of historical tracks."""
    _, graph_hv = plt.subplots()
    graph_hv.set_title('Historical wind')
    graph_hv.set_xlabel('Node number')
    graph_hv.set_ylabel('Max sustained wind (kn)')

    _, graph_hp = plt.subplots()
    graph_hp.set_title('Historical pressure')
    graph_hp.set_xlabel('Node number')
    graph_hp.set_ylabel('Central pressure (mb)')

    _, graph_hpd_a = plt.subplots()
    graph_hpd_a.set_title('Historical relative pressure')
    graph_hpd_a.set_xlabel('Distance from landfall (km)')
    graph_hpd_a.set_ylabel('Central pressure relative to landfall')

    _, graph_hped_a = plt.subplots()
    graph_hped_a.set_title('Historical environmental - central pressure')
    graph_hped_a.set_xlabel('Distance from landfall (km)')
    graph_hped_a.set_ylabel('Environmental pressure - Central pressure (mb)')

    for track in hist_tracks:
        # Index in land that comes from previous sea index
        sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0]+1
        # Index in sea that comes from previous land index
        land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0]+1
        if track.on_land[-1]:
            land_sea_idx = np.append(land_sea_idx, track.time.size)
        if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
            for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
                p_lf = track.central_pressure[sea_land-1].values
                scale = np.where(track.max_sustained_wind[sea_land-1].values <
                                 SAFFIR_SIM_CAT)[0][0]
                on_land = np.arange(track.time.size)[sea_land:land_sea]

                graph_hv.add_curve(on_land, track.max_sustained_wind[on_land],
                                   'o', c=CAT_COLORS[scale])
                graph_hp.add_curve(on_land, track.central_pressure[on_land],
                                   'o', c=CAT_COLORS[scale])
                graph_hpd_a.plot(track.dist_since_lf[on_land],
                                 track.central_pressure[on_land]/p_lf,
                                 'o', c=CAT_COLORS[scale])
                graph_hped_a.plot(track.dist_since_lf[on_land],
                                  track.environmental_pressure[on_land]-
                                  track.central_pressure[on_land],
                                  'o', c=CAT_COLORS[scale])

            on_sea = np.arange(track.time.size)[np.logical_not(track.on_land)]
            graph_hp.plot(on_sea, track.central_pressure[on_sea],
                          'o', c='k', markersize=5)
            graph_hv.plot(on_sea, track.max_sustained_wind[on_sea],
                          'o', c='k', markersize=5)

    return graph_hv, graph_hp, graph_hpd_a, graph_hped_a

def _missing_pressure(cen_pres, v_max, lat, lon):
    """Deal with missing central pressures."""
    if np.argwhere(cen_pres <= 0).size > 0:
        cen_pres = 1024.388 + 0.047*lat - 0.029*lon - 0.818*v_max # ibtracs 1980 -2013 (r2=0.91)
#        cen_pres = 1024.688+0.055*lat-0.028*lon-0.815*v_max      # peduzzi
    return cen_pres

def _change_max_wind_unit(wind, unit_orig, unit_dest):
    """ Compute maximum wind speed in unit_dest

    Parameters:
        wind (np.array): wind
        unit_orig (str): units of wind
        unit_dest (str): units to change wind

    Returns:
        double
    """
    if unit_orig in ('kn', 'kt'):
        ur_orig = ureg.knot
    elif unit_orig == 'mph':
        ur_orig = ureg.mile / ureg.hour
    elif unit_orig == 'm/s':
        ur_orig = ureg.meter / ureg.second
    elif unit_orig == 'km/h':
        ur_orig = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Unit not recognised %s.', unit_orig)
        raise ValueError
    if unit_dest in ('kn', 'kt'):
        ur_dest = ureg.knot
    elif unit_dest == 'mph':
        ur_dest = ureg.mile / ureg.hour
    elif unit_dest == 'm/s':
        ur_dest = ureg.meter / ureg.second
    elif unit_dest == 'km/h':
        ur_dest = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Unit not recognised %s.', unit_dest)
        raise ValueError
    return (np.nanmax(wind) * ur_orig).to(ur_dest).magnitude

def set_category(max_sus_wind, max_sus_wind_unit, saffir_scale=None):
    """Add storm category according to saffir-simpson hurricane scale

      - -1 tropical depression
      - 0 tropical storm
      - 1 Hurrican category 1
      - 2 Hurrican category 2
      - 3 Hurrican category 3
      - 4 Hurrican category 4
      - 5 Hurrican category 5

    Parameters:
        max_sus_wind (np.array): max sustained wind
        max_sus_wind_unit (str): units of max sustained wind
        saffir_scale (list, optional): Saffir-Simpson scale in same units as wind

    Returns:
        double
    """
    if saffir_scale:
        max_wind = np.nanmax(max_sus_wind)
    elif max_sus_wind_unit != 'kn':
        max_wind = _change_max_wind_unit(max_sus_wind, max_sus_wind_unit, 'kn')
        saffir_scale = SAFFIR_SIM_CAT
    else:
        saffir_scale = SAFFIR_SIM_CAT
        max_wind = np.nanmax(max_sus_wind)
    try:
        return (np.argwhere(max_wind < saffir_scale) - 1)[0][0]
    except IndexError:
        return -1
