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

__all__ = ['CAT_NAMES', 'SAFFIR_SIM_CAT', 'TCTracks', 'set_category']

import os
import glob
import shutil
import logging
import warnings
import datetime as dt
import itertools
import numpy as np
import matplotlib.cm as cm_mp
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
from sklearn.neighbors import DistanceMetric
import netCDF4 as nc
from numba import jit
import scipy.io.matlab as matlab
import statsmodels.api as sm

from climada.util import ureg
import climada.util.coordinates as coord_util
from climada.util.constants import EARTH_RADIUS_KM, SYSTEM_DIR
from climada.util.files_handler import get_file_names, download_ftp
import climada.util.plot as u_plot
import climada.hazard.tc_tracks_synth

LOGGER = logging.getLogger(__name__)

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 137, 1000]
"""Saffir-Simpson Hurricane Wind Scale in kn based on NOAA"""

CAT_NAMES = {
    -1: 'Tropical Depression',
    0: 'Tropical Storm',
    1: 'Hurricane Cat. 1',
    2: 'Hurricane Cat. 2',
    3: 'Hurricane Cat. 3',
    4: 'Hurricane Cat. 4',
    5: 'Hurricane Cat. 5',
}
"""Saffir-Simpson category names."""

CAT_COLORS = cm_mp.rainbow(np.linspace(0, 1, len(SAFFIR_SIM_CAT)))
"""Color scale to plot the Saffir-Simpson scale."""

IBTRACS_URL = ('https://www.ncei.noaa.gov/data/'
               'international-best-track-archive-for-climate-stewardship-ibtracs/'
               'v04r00/access/netcdf')
"""Site of IBTrACS netcdf file containing all tracks v4.0,
s. https://www.ncdc.noaa.gov/ibtracs/index.php?name=ib-v4-access"""

IBTRACS_FILE = 'IBTrACS.ALL.v04r00.nc'
"""IBTrACS v4.0 file all"""

IBTRACS_AGENCIES = [
    'wmo', 'usa', 'tokyo', 'newdelhi', 'reunion', 'bom', 'nadi', 'wellington',
    'cma', 'hko', 'ds824', 'td9636', 'td9635', 'neumann', 'mlc',
]
"""Names/IDs of agencies in IBTrACS v4.0"""

IBTRACS_USA_AGENCIES = [
    'atcf', 'cphc', 'hurdat_atl', 'hurdat_epa', 'jtwc_cp', 'jtwc_ep', 'jtwc_io',
    'jtwc_sh', 'jtwc_wp', 'nhc_working_bt', 'tcvightals', 'tcvitals'
]
"""Names/IDs of agencies in IBTrACS that correspond to 'usa_*' variables"""

DEF_ENV_PRESSURE = 1010
"""Default environmental pressure"""

BASIN_ENV_PRESSURE = {
    '': DEF_ENV_PRESSURE,
    'EP': 1010, 'NA': 1010, 'SA': 1010,
    'NI': 1005, 'SI': 1005, 'WP': 1005,
    'SP': 1004,
}
"""Basin-specific default environmental pressure"""

EMANUEL_RMW_CORR_FILES = [
    'temp_ccsm420thcal.mat', 'temp_ccsm4rcp85_full.mat',
    'temp_gfdl520thcal.mat', 'temp_gfdl5rcp85cal_full.mat',
    'temp_hadgem20thcal.mat', 'temp_hadgemrcp85cal_full.mat',
    'temp_miroc20thcal.mat', 'temp_mirocrcp85cal_full.mat',
    'temp_mpi20thcal.mat', 'temp_mpircp85cal_full.mat',
    'temp_mri20thcal.mat', 'temp_mrircp85cal_full.mat',
]
EMANUEL_RMW_CORR_FACTOR = 2.0
"""Kerry Emanuel track files in this list require a correction: The radius of
    maximum wind (rmstore) needs to be multiplied by factor 2."""

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
        """Empty constructor. Read csv IBTrACS files if provided."""
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
        Returns the first matching track based on the assumption that no other
        track with the same name or sid exists in the set.

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

    def subset(self, filterdict):
        """Subset tracks based on attributes. Currently only uses exact matches.
        Returns a new instance.

        Parameters:
            filterdict (dict): Of the form {'sid': 'pattern', ...}. Although
                this is not an ordered dict, presumably the filter of greatest
                magnitude should come first.
        """
        out = self.__class__(self.pool)
        out.data = self.data

        for key, pattern in filterdict.items():
            out.data = [ds for ds in out.data if ds.attrs[key] == pattern]

        return out

    def read_ibtracs_netcdf(self, provider=None, storm_id=None,
                            year_range=None, basin=None, estimate_missing=False,
                            correct_pres=False,
                            file_name='IBTrACS.ALL.v04r00.nc'):
        """Fill from raw ibtracs v04. Removes nans in coordinates, central
        pressure and removes repeated times data. Fills nans of environmental_pressure
        and radius_max_wind. Checks environmental_pressure > central_pressure.

        Parameters:
            provider (str, optional): If specified, enforce use of specific
                agency, such as "usa", "newdelhi", "bom", "cma", "tokyo".
                Default: None (and automatic choice).
            storm_id (str or list(str), optional): IBTrACS ID of the storm,
                e.g. 1988234N13299, [1988234N13299, 1989260N11316]
            year_range(tuple, optional): (min_year, max_year). Default: (1980, 2018)
            basin (str, optional): e.g. US, SA, NI, SI, SP, WP, EP, NA. if not
                provided, consider all basins.
            estimate_missing (bool, optional): estimate missing central pressure
                wind speed and radius values using other available values.
                Default: False
            correct_pres (bool, optional): For backwards compatibility, alias
                for `estimate_missing`. This is deprecated, use
                `estimate_missing` instead!
            file_name (str, optional): name of netcdf file to be dowloaded or located
                at climada/data/system. Default: 'IBTrACS.ALL.v04r00.nc'.
        """
        if correct_pres:
            LOGGER.warning("`correct_pres` is deprecated. "
                           "Use `estimate_missing` instead.")
            estimate_missing = True
        self.data = list()
        fn_nc = os.path.join(os.path.abspath(SYSTEM_DIR), file_name)
        if not glob.glob(fn_nc):
            try:
                download_ftp(f'{IBTRACS_URL}/{IBTRACS_FILE}', IBTRACS_FILE)
                shutil.move(IBTRACS_FILE, fn_nc)
            except ValueError as err:
                LOGGER.error('Error while downloading %s. Try to download it '
                             'manually and put the file in '
                             'climada_python/data/system/', IBTRACS_URL)
                raise err

        ibtracs_ds = xr.open_dataset(fn_nc)
        match = np.ones(ibtracs_ds.sid.shape[0], dtype=bool)
        if storm_id:
            if not isinstance(storm_id, list):
                storm_id = [storm_id]
            match &= ibtracs_ds.sid.isin([i.encode() for i in storm_id])
            if np.count_nonzero(match) == 0:
                LOGGER.info('No tracks with given IDs %s.', storm_id)
        else:
            year_range = year_range if year_range else (1980, 2018)
        if year_range:
            years = ibtracs_ds.sid.str.slice(0, 4).astype(int)
            match &= (years >= year_range[0]) & (years <= year_range[1])
            if np.count_nonzero(match) == 0:
                LOGGER.info('No tracks in time range (%s, %s).', *year_range)
        if basin:
            match &= (ibtracs_ds.basin == basin.encode()).any(dim='date_time')
            if np.count_nonzero(match) == 0:
                LOGGER.info('No tracks in basin %s.', basin)

        if np.count_nonzero(match) == 0:
            LOGGER.info('There are no tracks matching the specified requirements.')
            self.data = []
            return

        ibtracs_ds = ibtracs_ds.sel(storm=match)
        ibtracs_ds['valid_t'] = ibtracs_ds.time.notnull()
        valid_st = ibtracs_ds.valid_t.any(dim="date_time")
        invalid_st = np.nonzero(~valid_st.data)[0]
        if invalid_st.size > 0:
            st_ids = ', '.join(ibtracs_ds.sid.sel(storm=invalid_st).astype(str).data)
            LOGGER.warning('No valid timestamps found for %s.', st_ids)
            ibtracs_ds = ibtracs_ds.sel(storm=valid_st)

        if not provider:
            agency_pref, track_agency_ix = ibtracs_track_agency(ibtracs_ds)

        for var in ['wind', 'pres', 'rmw', 'poci', 'roci']:
            if provider:
                # enforce use of specified provider's data points
                ibtracs_ds[var] = ibtracs_ds[f'{provider}_{var}']
            else:
                # array of values in order of preference
                cols = [f'{a}_{var}' for a in agency_pref]
                cols = [col for col in cols if col in ibtracs_ds.data_vars.keys()]
                all_vals = ibtracs_ds[cols].to_array(dim='agency')
                preferred_ix = all_vals.notnull().argmax(dim='agency')

                if var in ['wind', 'pres']:
                    # choice: wmo -> wmo_agency/usa_agency -> preferred
                    ibtracs_ds[var] = ibtracs_ds['wmo_' + var] \
                        .fillna(all_vals.isel(agency=track_agency_ix)) \
                        .fillna(all_vals.isel(agency=preferred_ix))
                else:
                    ibtracs_ds[var] = all_vals.isel(agency=preferred_ix)
        ibtracs_ds = ibtracs_ds[['sid', 'name', 'basin', 'lat', 'lon', 'time', 'valid_t',
                                 'wind', 'pres', 'rmw', 'roci', 'poci']]

        if estimate_missing:
            ibtracs_ds['pres'][:] = _estimate_pressure(ibtracs_ds.pres,
                                                       ibtracs_ds.lat, ibtracs_ds.lon,
                                                       ibtracs_ds.wind)
            ibtracs_ds['wind'][:] = _estimate_vmax(ibtracs_ds.wind,
                                                   ibtracs_ds.lat, ibtracs_ds.lon,
                                                   ibtracs_ds.pres)

        ibtracs_ds['valid_t'] &= ibtracs_ds.wind.notnull() & ibtracs_ds.pres.notnull()
        valid_st = ibtracs_ds.valid_t.any(dim="date_time")
        invalid_st = np.nonzero(~valid_st.data)[0]
        if invalid_st.size > 0:
            st_ids = ', '.join(ibtracs_ds.sid.sel(storm=invalid_st).astype(str).data)
            LOGGER.warning('No valid wind/pressure values found for %s.', st_ids)
            ibtracs_ds = ibtracs_ds.sel(storm=valid_st)

        max_wind = ibtracs_ds.wind.max(dim="date_time").data.ravel()
        category_test = (max_wind[:, None] < np.array(SAFFIR_SIM_CAT)[None])
        category = np.argmax(category_test, axis=1) - 1
        basin_map = {b.encode("utf-8"): v for b, v in BASIN_ENV_PRESSURE.items()}
        basin_fun = lambda b: basin_map[b]

        ibtracs_ds['id_no'] = (ibtracs_ds.sid.str.replace(b'N', b'0')
                               .str.replace(b'S', b'1')
                               .astype(float))
        ibtracs_ds['time_step'] = xr.zeros_like(ibtracs_ds.time, dtype=float)
        ibtracs_ds['time_step'][:, 1:] = (ibtracs_ds.time.diff(dim="date_time")
                                          / np.timedelta64(1, 's'))
        ibtracs_ds['time_step'][:, 0] = ibtracs_ds.time_step[:, 1]
        provider = provider if provider else 'ibtracs'

        last_perc = 0
        all_tracks = []
        for i_track, t_msk in enumerate(ibtracs_ds.valid_t.data):
            perc = 100 * len(all_tracks) / ibtracs_ds.sid.size
            if perc - last_perc >= 10:
                LOGGER.info("Progress: %d%%", perc)
                last_perc = perc
            track_ds = ibtracs_ds.sel(storm=i_track, date_time=t_msk)
            st_penv = xr.apply_ufunc(basin_fun, track_ds.basin, vectorize=True)
            track_ds['time'][:1] = track_ds.time[:1].dt.floor('H')
            if track_ds.time.size > 1:
                track_ds['time_step'][0] = (track_ds.time[1] - track_ds.time[0]) \
                                      / np.timedelta64(1, 's')

            with warnings.catch_warnings():
                # See https://github.com/pydata/xarray/issues/4167
                warnings.simplefilter(action="ignore", category=FutureWarning)

                track_ds['rmw'] = track_ds.rmw \
                    .ffill(dim='date_time', limit=1) \
                    .bfill(dim='date_time', limit=1) \
                    .fillna(0)
                track_ds['roci'] = track_ds.roci \
                    .ffill(dim='date_time', limit=1) \
                    .bfill(dim='date_time', limit=1) \
                    .fillna(0)
                track_ds['poci'] = track_ds.poci \
                    .ffill(dim='date_time', limit=4) \
                    .bfill(dim='date_time', limit=4)
                # this is the most time consuming line in the processing:
                track_ds['poci'] = track_ds.poci.fillna(st_penv)

            if estimate_missing:
                track_ds['rmw'][:] = estimate_rmw(track_ds.rmw.values, track_ds.pres.values)
                track_ds['roci'][:] = estimate_roci(track_ds.roci.values, track_ds.rmw.values)
                track_ds['roci'][:] = np.fmax(track_ds.rmw.values, track_ds.roci.values)

            # ensure environmental pressure >= central pressure
            # this is the second most time consuming line in the processing:
            track_ds['poci'][:] = np.fmax(track_ds.poci, track_ds.pres)

            all_tracks.append(xr.Dataset({
                'time_step': ('time', track_ds.time_step),
                'radius_max_wind': ('time', track_ds.rmw.data),
                'radius_oci': ('time', track_ds.roci.data),
                'max_sustained_wind': ('time', track_ds.wind.data),
                'central_pressure': ('time', track_ds.pres.data),
                'environmental_pressure': ('time', track_ds.poci.data),
            }, coords={
                'time': track_ds.time.dt.round('s').data,
                'lat': ('time', track_ds.lat.data),
                'lon': ('time', track_ds.lon.data),
            }, attrs={
                'max_sustained_wind_unit': 'kn',
                'central_pressure_unit': 'mb',
                'name': track_ds.name.astype(str).item(),
                'sid': track_ds.sid.astype(str).item(),
                'orig_event_flag': True,
                'data_provider': provider,
                'basin': track_ds.basin.values[0].astype(str).item(),
                'id_no': track_ds.id_no.item(),
                'category': category[i_track],
            }))
        self.data = all_tracks

    def read_processed_ibtracs_csv(self, file_names):
        """Fill from processed ibtracs csv file(s).

        Parameters:
            file_names (str or list(str)): absolute file name(s) or
                folder name containing the files to read.
        """
        self.data = list()
        all_file = get_file_names(file_names)
        for file in all_file:
            self._read_ibtracs_csv_single(file)

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
        LOGGER.info("File contains %s tracks (at most %s nodes each), "
                    "representing %s years (%s-%s).", ntracks, nnodes,
                    years_uniq.size, years_uniq[0], years_uniq[-1])

        # filter according to chosen hemisphere
        hem_mask = (lat >= hem_min) & (lat <= hem_max) | (lat == 0)
        hem_idx = np.all(hem_mask, axis=1).nonzero()[0]
        data_hem = lambda keys: [data_mat[f'{k}store'][hem_idx] for k in keys]

        lat, lon = data_hem(['lat', 'long'])
        months, days, hours = data_hem(['month', 'day', 'hour'])
        months, days, hours = [np.int8(ar) for ar in [months, days, hours]]
        tc_rmw, tc_maxwind, tc_pressure = data_hem(['rm', 'v', 'p'])
        years = data_mat['yearstore'][0, hem_idx]

        ntracks, nnodes = lat.shape
        LOGGER.info("Loading %s tracks on %s hemisphere.", ntracks, hemisphere)

        # change lon format to -180 to 180
        lon[lon > 180] = lon[lon > 180] - 360

        # change units from kilometers to nautical miles
        tc_rmw = (tc_rmw * ureg.kilometer).to(ureg.nautical_mile).magnitude
        if rmw_corr:
            LOGGER.info("Applying RMW correction.")
            tc_rmw *= EMANUEL_RMW_CORR_FACTOR

        for i_track in range(lat.shape[0]):
            valid_idx = (lat[i_track, :] != 0).nonzero()[0]
            nnodes = valid_idx.size
            time_step = np.abs(np.diff(hours[i_track, valid_idx])).min()

            # deal with change of year
            year = np.full(valid_idx.size, years[i_track])
            year_change = (np.diff(months[i_track, valid_idx]) < 0)
            year_change = year_change.nonzero()[0]
            if year_change.size > 0:
                year[year_change[0] + 1:] += 1

            try:
                datetimes = map(dt.datetime, year,
                                months[i_track, valid_idx],
                                days[i_track, valid_idx],
                                hours[i_track, valid_idx])
                datetimes = list(datetimes)
            except ValueError as err:
                # dates are known to contain invalid February 30
                date_feb = (months[i_track, valid_idx] == 2) \
                         & (days[i_track, valid_idx] > 28)
                if np.count_nonzero(date_feb) == 0:
                    # unknown invalid date issue
                    raise err
                step = time_step if not date_feb[0] else -time_step
                reference_idx = 0 if not date_feb[0] else -1
                reference_date = dt.datetime(
                    year[reference_idx],
                    months[i_track, valid_idx[reference_idx]],
                    days[i_track, valid_idx[reference_idx]],
                    hours[i_track, valid_idx[reference_idx]],)
                datetimes = [reference_date + dt.timedelta(hours=int(step * i))
                             for i in range(nnodes)]
            datetimes = np.array(datetimes)

            max_sustained_wind = tc_maxwind[i_track, valid_idx]
            max_sustained_wind_unit = 'kn'
            env_pressure = np.full(nnodes, DEF_ENV_PRESSURE)
            category = set_category(max_sustained_wind,
                                    max_sustained_wind_unit,
                                    SAFFIR_SIM_CAT)
            tr_ds = xr.Dataset({
                'time_step': ('time', np.full(nnodes, time_step)),
                'radius_max_wind': ('time', tc_rmw[i_track, valid_idx]),
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
        scale_to_10m = (10. / 60.)**.11
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
        for time in times:
            try:
                datetimes.append(
                    dt.datetime.strptime(
                        str(nc.num2date(time, 'days since {}'.format('1858-11-17'),
                                        calendar='standard')),
                        '%Y-%m-%d %H:%M:%S'))
            except ValueError:
                # If wrong t, set t to previous t plus 3 hours
                if datetimes:
                    datetimes.append(datetimes[-1] + dt.timedelta(hours=3))
                else:
                    pos = list(times).index(time)
                    time = times[pos + 1] - 1 / 24 * 3
                    datetimes.append(
                        dt.datetime.strptime(
                            str(nc.num2date(time, 'days since {}'.format('1858-11-17'),
                                            calendar='standard')),
                            '%Y-%m-%d %H:%M:%S'))
        time_step = []
        for i_time, time in enumerate(datetimes[1:], 1):
            time_step.append((time - datetimes[i_time - 1]).total_seconds() / 3600)
        time_step.append(time_step[-1])

        basins = list()
        for basin in nc_data.variables['basin'][i_track, :][:val_len]:
            try:
                basins.extend([basin_dict[basin]])
            except KeyError:
                basins.extend([np.nan])

        lon = nc_data.variables['lon'][i_track, :][:val_len]
        lon[lon > 180] = lon[lon > 180] - 360  # change lon format to -180 to 180
        lat = nc_data.variables['lat'][i_track, :][:val_len]
        cen_pres = nc_data.variables['pres'][i_track, :][:val_len]
        av_prec = nc_data.variables['precavg'][i_track, :][:val_len]
        max_prec = nc_data.variables['precmax'][i_track, :][:val_len]

        # m/s to kn
        wind = nc_data.variables['wind'][i_track, :][:val_len] * mps2kts * scale_to_10m
        if not all(wind.data):  # if wind is empty
            wind = np.ones(wind.size) * -999.9

        tr_df = pd.DataFrame({'time': datetimes, 'lat': lat, 'lon': lon,
                              'max_sustained_wind': wind,
                              'central_pressure': cen_pres,
                              'environmental_pressure': np.ones(lat.size) * 1015.,
                              'radius_max_wind': np.ones(lat.size) * 65.,
                              'maximum_precipitation': max_prec,
                              'average_precipitation': av_prec,
                              'basins': basins,
                              'time_step': time_step})

        # construct xarray
        tr_ds = xr.Dataset.from_dataframe(tr_df.set_index('time'))
        tr_ds.coords['lat'] = ('time', tr_ds.lat)
        tr_ds.coords['lon'] = ('time', tr_ds.lon)
        tr_ds.attrs = {'max_sustained_wind_unit': 'kn',
                       'central_pressure_unit': 'mb',
                       'sid': sid,
                       'name': sid, 'orig_event_flag': False,
                       'basin': basins[0],
                       'id_no': i_track,
                       'category': set_category(wind, 'kn')}
        self.data.append(tr_ds)

    def equal_timestep(self, time_step_h=1, land_params=False):
        """Generate interpolated track values to time steps of min_time_step.
        Parameters:
            time_step_h (float, optional): time step in hours to which to
                interpolate. Default: 1.
            land_params (bool, optional): compute on_land and dist_since_lf at
                each node. Default: False.
        """
        LOGGER.info('Interpolating %s tracks to %sh time steps.', self.size,
                    time_step_h)

        if land_params:
            extent = self.get_extent()
            land_geom = coord_util.get_land_geometry(extent, resolution=10)
        else:
            land_geom = None

        if self.pool:
            chunksize = min(self.size // self.pool.ncpus, 1000)
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

    def calc_random_walk(self, **kwargs):
        """See function in `climada.hazard.tc_tracks_synth`"""
        climada.hazard.tc_tracks_synth.calc_random_walk(self, **kwargs)

    @property
    def size(self):
        """Get longitude from coord array"""
        return len(self.data)

    def get_bounds(self, deg_buffer=0.1):
        """Get bounds as (lon_min, lat_min, lon_max, lat_max) tuple.

        Parameters:
            deg_buffer (float): A buffer to add around the bounding box

        Returns:
            tuple (lon_min, lat_min, lon_max, lat_max)
        """
        bounds = coord_util.latlon_bounds(
            np.concatenate([t.lat.values for t in self.data]),
            np.concatenate([t.lon.values for t in self.data]),
            buffer=deg_buffer)
        return bounds

    @property
    def bounds(self):
        """Exact bounds of trackset as tuple, no buffer."""
        return self.get_bounds(deg_buffer=0.0)

    def get_extent(self, deg_buffer=0.1):
        """Get extent as (lon_min, lon_max, lat_min, lat_max) tuple.

        Parameters:
            deg_buffer (float): A buffer to add around the bounding box

        Returns:
            tuple (lon_min, lon_max, lat_min, lat_max)
        """
        bounds = self.get_bounds(deg_buffer=deg_buffer)
        return (bounds[0], bounds[2], bounds[1], bounds[3])

    @property
    def extent(self):
        """Exact extent of trackset as tuple, no buffer."""
        return self.get_extent(deg_buffer=0.0)

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

        extent = self.get_extent(deg_buffer=1)
        mid_lon = 0.5 * (extent[1] + extent[0])

        if not axis:
            proj = ccrs.PlateCarree(central_longitude=mid_lon)
            _, axis = u_plot.make_map(proj=proj)
        axis.set_extent(extent, crs=kwargs['transform'])
        u_plot.add_shapes(axis)

        synth_flag = False
        cmap = ListedColormap(colors=CAT_COLORS)
        norm = BoundaryNorm([0] + SAFFIR_SIM_CAT, len(SAFFIR_SIM_CAT))
        for track in self.data:
            lonlat = np.stack([track.lon.values, track.lat.values], axis=-1)
            lonlat[:, 0] = coord_util.lon_normalize(lonlat[:, 0], center=mid_lon)
            segments = np.stack([lonlat[:-1], lonlat[1:]], axis=1)
            # remove segments which cross 180 degree longitude boundary
            segments = segments[segments[:, 0, 0] * segments[:, 1, 0] >= 0, :, :]
            if track.orig_event_flag:
                track_lc = LineCollection(segments, cmap=cmap, norm=norm,
                                          linestyle='solid', **kwargs)
            else:
                synth_flag = True
                track_lc = LineCollection(segments, cmap=cmap, norm=norm,
                                          linestyle=':', **kwargs)
            track_lc.set_array(track.max_sustained_wind.values)
            axis.add_collection(track_lc)

        leg_lines = [Line2D([0], [0], color=CAT_COLORS[i_col], lw=2)
                     for i_col in range(len(SAFFIR_SIM_CAT))]
        leg_names = [CAT_NAMES[i_col] for i_col in sorted(CAT_NAMES.keys())]
        if synth_flag:
            leg_lines.append(Line2D([0], [0], color='grey', lw=2, ls='solid'))
            leg_lines.append(Line2D([0], [0], color='grey', lw=2, ls=':'))
            leg_names.append('Historical')
            leg_names.append('Synthetic')

        axis.legend(leg_lines, leg_names, loc=0)
        return axis

    def write_netcdf(self, folder_name):
        """Write a netcdf file per track with track.sid name in given folder.

        Parameters:
            folder_name (str): folder name where to write files
        """
        list_path = [os.path.join(folder_name, track.sid + '.nc') for track in self.data]
        LOGGER.info('Writting %s files.', self.size)
        for track in self.data:
            track.attrs['orig_event_flag'] = int(track.orig_event_flag)
        xr.save_mfdataset(self.data, list_path)

    def read_netcdf(self, folder_name):
        """Read all netcdf files contained in folder and fill a track per file.

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
    @jit(parallel=True, forceobj=True)
    def _one_interp_data(track, time_step_h, land_geom=None):
        """Interpolate values of one track.

        Parameters:
            track (xr.Dataset): track data

        Returns:
            xr.Dataset
        """
        if track.time.size >= 2:
            method = ['linear', 'quadratic', 'cubic'][min(2, track.time.size - 2)]

            # handle change of sign in longitude
            lon = track.lon.copy()
            if (lon < -170).any() and (lon > 170).any():
                # crosses 180 degrees east/west -> use positive degrees east
                lon[lon < 0] += 360

            time_step = '{}H'.format(time_step_h)
            track_int = track.resample(time=time_step, keep_attrs=True, skipna=True)\
                             .interpolate('linear')
            track_int['time_step'][:] = time_step_h
            lon_int = lon.resample(time=time_step).interpolate(method)
            lon_int[lon_int > 180] -= 360
            track_int.coords['lon'] = lon_int
            track_int.coords['lat'] = track.lat.resample(time=time_step)\
                                               .interpolate(method)
            track_int.attrs['category'] = set_category(
                track_int.max_sustained_wind.values,
                track_int.max_sustained_wind_unit)
        else:
            LOGGER.warning('Track interpolation not done. '
                           'Not enough elements for %s', track.name)
            track_int = track

        if land_geom:
            track_land_params(track_int, land_geom)
        return track_int

    def _read_ibtracs_csv_single(self, file_name):
        """Read IBTrACS track file in CSV format.

            Parameters:
                file_name (str): File name of CSV file.
        """
        LOGGER.info('Reading %s', file_name)
        dfr = pd.read_csv(file_name)
        name = dfr['ibtracsID'].values[0]

        datetimes = list()
        for time in dfr['isotime'].values:
            year = np.fix(time / 1e6)
            time = time - year * 1e6
            month = np.fix(time / 1e4)
            time = time - month * 1e4
            day = np.fix(time / 1e2)
            hour = time - day * 1e2
            datetimes.append(dt.datetime(int(year), int(month), int(day),
                                         int(hour)))

        lat = dfr['cgps_lat'].values.astype('float')
        lon = dfr['cgps_lon'].values.astype('float')
        cen_pres = dfr['pcen'].values.astype('float')
        max_sus_wind = dfr['vmax'].values.astype('float')
        max_sus_wind_unit = 'kn'
        if np.any(cen_pres <= 0):
            # Warning: If any pressure value is invalid, this enforces to use
            # estimated pressure values everywhere!
            cen_pres[:] = -999
            cen_pres = _estimate_pressure(cen_pres, lat, lon, max_sus_wind)

        tr_ds = xr.Dataset()
        tr_ds.coords['time'] = ('time', datetimes)
        tr_ds.coords['lat'] = ('time', lat)
        tr_ds.coords['lon'] = ('time', lon)
        tr_ds['time_step'] = ('time', dfr['tint'].values)
        tr_ds['radius_max_wind'] = ('time', dfr['rmax'].values.astype('float'))
        tr_ds['max_sustained_wind'] = ('time', max_sus_wind)
        tr_ds['central_pressure'] = ('time', cen_pres)
        tr_ds['environmental_pressure'] = ('time',
                                           dfr['penv'].values.astype('float'))
        tr_ds.attrs['max_sustained_wind_unit'] = max_sus_wind_unit
        tr_ds.attrs['central_pressure_unit'] = 'mb'
        tr_ds.attrs['name'] = name
        tr_ds.attrs['sid'] = name
        tr_ds.attrs['orig_event_flag'] = bool(dfr['original_data']. values[0])
        tr_ds.attrs['data_provider'] = dfr['data_provider'].values[0]
        tr_ds.attrs['basin'] = dfr['gen_basin'].values[0]
        try:
            tr_ds.attrs['id_no'] = float(name.replace('N', '0').
                                         replace('S', '1'))
        except ValueError:
            tr_ds.attrs['id_no'] = float(str(datetimes[0].date()).
                                         replace('-', ''))
        tr_ds.attrs['category'] = set_category(max_sus_wind, max_sus_wind_unit)

        self.data.append(tr_ds)


def track_land_params(track, land_geom):
    """Compute parameters of land for one track.

    Parameters:
        track (xr.Dataset): tropical cyclone track
        land_geom (shapely.geometry.multipolygon.MultiPolygon): land geometry
    """
    track['on_land'] = ('time',
                        coord_util.coord_on_land(track.lat.values, track.lon.values, land_geom))
    track['dist_since_lf'] = ('time', _dist_since_lf(track))

def _dist_since_lf(track):
    """Compute the distance to landfall in km point for every point on land.
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
        return (dist_since_lf + 1) * np.nan

    # Index in sea that comes from previous land index
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1)[0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    orig_lf = np.empty((sea_land_idx.size, 2))
    for i_lf, lf_point in enumerate(sea_land_idx):
        orig_lf[i_lf][0] = track.lat[lf_point] + \
            (track.lat[lf_point + 1] - track.lat[lf_point]) / 2
        orig_lf[i_lf][1] = track.lon[lf_point] + \
            (track.lon[lf_point + 1] - track.lon[lf_point]) / 2

    dist = DistanceMetric.get_metric('haversine')
    nodes1 = np.radians(np.array([track.lat.values[1:],
                                  track.lon.values[1:]]).transpose())
    nodes0 = np.radians(np.array([track.lat.values[:-1],
                                  track.lon.values[:-1]]).transpose())
    dist_since_lf[1:] = dist.pairwise(nodes1, nodes0).diagonal()
    dist_since_lf[~track.on_land.values] = 0.0
    nodes1 = np.array([track.lat.values[sea_land_idx + 1],
                       track.lon.values[sea_land_idx + 1]]).transpose() / 180 * np.pi
    dist_since_lf[sea_land_idx + 1] = \
        dist.pairwise(nodes1, orig_lf / 180 * np.pi).diagonal()
    for sea_land, land_sea in zip(sea_land_idx, land_sea_idx):
        dist_since_lf[sea_land + 1:land_sea] = \
            np.cumsum(dist_since_lf[sea_land + 1:land_sea])

    dist_since_lf *= EARTH_RADIUS_KM
    dist_since_lf[~track.on_land.values] = np.nan

    return dist_since_lf

def _estimate_pressure(cen_pres, lat, lon, v_max):
    """Replace missing pressure values with statistical estimate.

    In addition to NaNs, negative values and zeros in `cen_pres` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('pres', ['lat', 'lon', 'wind'], year_range=(1980, 2019))
    >>> r^2: 0.8746154487335112

    Parameters
    ----------
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).
    lat : array-like
        Latitudinal coordinates of eye location.
    lon : array-like
        Longitudinal coordinates of eye location.
    v_max : array-like
        Maximum wind speed along track in knots.

    Returns
    -------
    cen_pres_estimated : np.array
        Estimated central pressure values in hPa (mbar).
    """
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    v_max = np.where(np.isnan(v_max), -1, v_max)
    lat, lon = [np.where(np.isnan(ar), -999, ar) for ar in [lat, lon]]
    msk = (cen_pres <= 0) & (v_max > 0) & (lat > -999) & (lon > -999)
    c_const, c_lat, c_lon, c_vmax = 1024.392, 0.0620, -0.0335, -0.737
    cen_pres[msk] = c_const + c_lat * lat[msk] \
                            + c_lon * lon[msk] \
                            + c_vmax * v_max[msk]
    return np.where(cen_pres <= 0, np.nan, cen_pres)

def _estimate_vmax(v_max, lat, lon, cen_pres):
    """Replace missing wind speed values with a statistical estimate.

    In addition to NaNs, negative values and zeros in `v_max` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('wind', ['lat', 'lon', 'pres'], year_range=(1980, 2019))
    >>> r^2: 0.8717153945288457

    Parameters
    ----------
    v_max : array-like
        Maximum wind speed along track in knots.
    lat : array-like
        Latitudinal coordinates of eye location.
    lon : array-like
        Longitudinal coordinates of eye location.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    v_max_estimated : np.array
        Estimated maximum wind speed values in knots.
    """
    v_max = np.where(np.isnan(v_max), -1, v_max)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    lat, lon = [np.where(np.isnan(ar), -999, ar) for ar in [lat, lon]]
    msk = (v_max <= 0) & (cen_pres > 0) & (lat > -999) & (lon > -999)
    c_const, c_lat, c_lon, c_pres = 1216.823, 0.0852, -0.0398, -1.182
    v_max[msk] = c_const + c_lat * lat[msk] \
                         + c_lon * lon[msk] \
                         + c_pres * cen_pres[msk]
    return np.where(v_max <= 0, np.nan, v_max)

def estimate_roci(roci, cen_pres):
    """Replace missing radius (ROCI) values with statistical estimate.

    In addition to NaNs, negative values and zeros in `roci` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('roci', ['pres'],
    ...                   order=[(872, 950, 985, 1005, 1021)],
    ...                   year_range=(1980, 2019))
    >>> r^2: 0.9148320406675339

    Parameters
    ----------
    roci : array-like
        ROCI values along track in km.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    roci_estimated : np.array
        Estimated ROCI values in km.
    """
    roci = np.where(np.isnan(roci), -1, roci)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    msk = (roci <= 0) & (cen_pres > 0)
    pres_l = [872, 950, 985, 1005, 1021]
    roci_l = [210.711487, 215.897110, 198.261520, 159.589508, 90.900116]
    roci[msk] = 0
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        roci[msk] += roci_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - cen_pres[msk])
                                             - slope_1 * np.fmax(0, cen_pres[msk] - pres_l_i)))
    return np.where(roci <= 0, np.nan, roci)

def estimate_rmw(rmw, cen_pres):
    """Replace missing radius (RMW) values with statistical estimate.

    In addition to NaNs, negative values and zeros in `rmw` are interpreted as missing values.

    See function `ibtracs_fit_param` for more details about the statistical estimation:

    >>> ibtracs_fit_param('rmw', ['pres'], order=[(872, 940, 980, 1021)], year_range=(1980, 2019))
    >>> r^2: 0.7905970811843872

    Parameters
    ----------
    rmw : array-like
        RMW values along track in km.
    cen_pres : array-like
        Central pressure values along track in hPa (mbar).

    Returns
    -------
    rmw : np.array
        Estimated RMW values in km.
    """
    rmw = np.where(np.isnan(rmw), -1, rmw)
    cen_pres = np.where(np.isnan(cen_pres), -1, cen_pres)
    msk = (rmw <= 0) & (cen_pres > 0)
    pres_l = [872, 940, 980, 1021]
    rmw_l = [14.907318, 15.726927, 25.742142, 56.856522]
    rmw[msk] = 0
    for i, pres_l_i in enumerate(pres_l):
        slope_0 = 1. / (pres_l_i - pres_l[i - 1]) if i > 0 else 0
        slope_1 = 1. / (pres_l[i + 1] - pres_l_i) if i + 1 < len(pres_l) else 0
        rmw[msk] += rmw_l[i] * np.fmax(0, (1 - slope_0 * np.fmax(0, pres_l_i - cen_pres[msk])
                                           - slope_1 * np.fmax(0, cen_pres[msk] - pres_l_i)))
    return np.where(rmw <= 0, np.nan, rmw)

def ibtracs_fit_param(explained, explanatory, year_range=(1980, 2019), order=1):
    """Statistically fit an ibtracs parameter to other ibtracs variables

    A linear ordinary least squares fit is done using the statsmodels package.

    Parameters:
        explained (str): name of explained variable
        explanatory (iterable): names of explanatory variables
        year_range (tuple): first and last year to include in the analysis
        order (int or tuple): the maximal order of the explanatory variables

    Returns:
        OLSResults
    """
    wmo_vars = ['wind', 'pres', 'rmw', 'roci', 'poci']
    all_vars = ['lat', 'lon'] + wmo_vars
    explanatory = list(explanatory)
    variables = explanatory + [explained]
    for var in variables:
        if var not in all_vars:
            LOGGER.error("Unknown ibtracs variable: %s", var)
            raise KeyError

    # load ibtracs dataset
    fn_nc = os.path.join(os.path.abspath(SYSTEM_DIR), 'IBTrACS.ALL.v04r00.nc')
    ibtracs_ds = xr.open_dataset(fn_nc)

    # choose specified year range
    years = ibtracs_ds.sid.str.slice(0, 4).astype(int)
    match = (years >= year_range[0]) & (years <= year_range[1])
    ibtracs_ds = ibtracs_ds.sel(storm=match)

    # fill values
    agency_pref, track_agency_ix = ibtracs_track_agency(ibtracs_ds)
    for var in wmo_vars:
        if var not in variables:
            continue
        # array of values in order of preference
        cols = [f'{a}_{var}' for a in agency_pref]
        cols = [col for col in cols if col in ibtracs_ds.data_vars.keys()]
        all_vals = ibtracs_ds[cols].to_array(dim='agency')
        preferred_ix = all_vals.notnull().argmax(dim='agency')
        if var in ['wind', 'pres']:
            # choice: wmo -> wmo_agency/usa_agency -> preferred
            ibtracs_ds[var] = ibtracs_ds['wmo_' + var] \
                .fillna(all_vals.isel(agency=track_agency_ix)) \
                .fillna(all_vals.isel(agency=preferred_ix))
        else:
            ibtracs_ds[var] = all_vals.isel(agency=preferred_ix)
    fit_df = pd.DataFrame({var: ibtracs_ds[var].values.ravel() for var in variables})
    fit_df = fit_df.dropna(axis=0, how='any').reset_index(drop=True)
    if 'lat' in explanatory:
        fit_df['lat'] = fit_df['lat'].abs()

    # prepare explanatory variables
    d_explanatory = fit_df[explanatory]
    if isinstance(order, int):
        order = (order,) * len(explanatory)
    add_const = False
    for ex, max_o in zip(explanatory, order):
        if isinstance(max_o, tuple):
            if fit_df[ex].min() > max_o[0]:
                print(f"Minimum data value is {fit_df[ex].min()} > {max_o[0]}.")
            if fit_df[ex].max() < max_o[-1]:
                print(f"Maximum data value is {fit_df[ex].max()} < {max_o[-1]}.")
            # piecewise linear with given break points
            d_explanatory = d_explanatory.drop(labels=[ex], axis=1)
            for i, max_o_i in enumerate(max_o):
                col = f'{ex}{max_o_i}'
                slope_0 = 1. / (max_o_i - max_o[i - 1]) if i > 0 else 0
                slope_1 = 1. / (max_o[i + 1] - max_o_i) if i + 1 < len(max_o) else 0
                d_explanatory[col] = np.fmax(0, (1 - slope_0 * np.fmax(0, max_o_i - fit_df[ex])
                                                 - slope_1 * np.fmax(0, fit_df[ex] - max_o_i)))
        elif max_o < 0:
            d_explanatory = d_explanatory.drop(labels=[ex], axis=1)
            for order in range(1, abs(max_o) + 1):
                d_explanatory[f'{ex}^{-order}'] = fit_df[ex]**(-order)
            add_const = True
        else:
            for order in range(2, max_o + 1):
                d_explanatory[f'{ex}^{order}'] = fit_df[ex]**order
            add_const = True
    d_explained = fit_df[[explained]]
    if add_const:
        d_explanatory['const'] = 1.0

    # run statistical fit
    sm_results = sm.OLS(d_explained, d_explanatory).fit()

    # print results
    print(sm_results.params)
    print("r^2:", sm_results.rsquared)

    return sm_results

def ibtracs_track_agency(ds_sel):
    """Get preferred IBTrACS agency for each entry in the dataset

    Parameters
    ----------
    ds_sel : xarray.Dataset
        Subselection of original IBTrACS NetCDF dataset.

    Returns
    -------
    agency_pref : list of str
        Names of IBTrACS agencies in order of preference.
    track_agency_ix : xarray.DataArray of ints
        For each entry in `ds_sel`, the agency to use, given as an index into `agency_pref`.
    """
    agency_pref = IBTRACS_AGENCIES.copy()
    agency_map = {a.encode('utf-8'): i for i, a in enumerate(agency_pref)}
    agency_map.update({
        a.encode('utf-8'): agency_map[b'usa'] for a in IBTRACS_USA_AGENCIES
    })
    agency_map[b''] = agency_map[b'wmo']
    agency_fun = lambda x: agency_map[x]
    track_agency = ds_sel.wmo_agency.where(ds_sel.wmo_agency != '', ds_sel.usa_agency)
    track_agency_ix = xr.apply_ufunc(agency_fun, track_agency, vectorize=True)
    return agency_pref, track_agency_ix

def _change_max_wind_unit(wind, unit_orig, unit_dest):
    """Compute maximum wind speed in unit_dest

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

def set_category(max_sus_wind, wind_unit, saffir_scale=None):
    """Add storm category according to saffir-simpson hurricane scale

      - -1 tropical depression
      - 0 tropical storm
      - 1 Hurricane category 1
      - 2 Hurricane category 2
      - 3 Hurricane category 3
      - 4 Hurricane category 4
      - 5 Hurricane category 5

    Parameters:
        max_sus_wind (np.array): max sustained wind
        wind_unit (str): units of max sustained wind
        saffir_scale (list, optional): Saffir-Simpson scale in same units as wind

    Returns:
        double
    """
    if saffir_scale is None:
        saffir_scale = SAFFIR_SIM_CAT
        if wind_unit != 'kn':
            max_sus_wind = _change_max_wind_unit(max_sus_wind, wind_unit, 'kn')
    max_wind = np.nanmax(max_sus_wind)
    try:
        return (np.argwhere(max_wind < saffir_scale) - 1)[0][0]
    except IndexError:
        return -1
