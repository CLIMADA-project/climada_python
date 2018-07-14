"""
Define TCTracks: IBTracs reader and tracks manager.
"""

__all__ = ['SAFFIR_SIM_CAT', 'TCTracks']

import logging
import datetime as dt
import array
import numpy as np
import pandas as pd
import xarray as xr
from pint import UnitRegistry

from climada.util.config import CONFIG
import climada.util.coordinates as coord_util
from climada.util.files_handler import get_file_names
import climada.util.plot as plot
from climada.util.constants import ONE_LAT_KM
from climada.util.interpolation import dist_sqr_approx

LOGGER = logging.getLogger(__name__)

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 135, 1000]
""" Saffir-Simpson Hurricane Wind Scale """

class TCTracks(object):
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
                - orig_event_flag (attrs)
                - data_provider (attrs)
                - basin (attrs)
                - id_no (attrs)
                - category (attrs)
    """
    def __init__(self):
        """Empty constructor. Read csv IBTrACS files if provided. """
        self.data = list()

    def append(self, tracks):
        """Append tracks to current.

        Parameters:
            tracks (xarray.Dataset or list(xarray.Dataset)): tracks to append.
        """
        if not isinstance(tracks, list):
            tracks = [tracks]
        self.data.extend(tracks)

    def get_track(self, name_ev=None, date_ev=None):
        """Get all tracks with provided name that happened at provided date.

        Parameters:
            name_ev (str, optional): name of event
            date_ev (str, optional): date of the event in format

        Returns:
            xarray.Dataset
        """
        if not name_ev and not date_ev:
            return self.data[0]
        else:
            raise NotImplementedError

    def read_ibtracs_csv(self, file_names):
        """Clear and model tropical cyclone from input csv IBTrACS file.
        Parallel process.

        Parameters:
            file_names (str or list(str)): absolute file name(s) or
                folder name containing the files to read.
        """
        all_file = get_file_names(file_names)
        for file in all_file:
            self._read_one_csv(file)

    def retrieve_ibtracs(self, name_ev, date_ev):
        """ Download from IBTrACS repository a specific track.

        Parameters:
            name_ev (str, optional): name of event
            date_ev (str, optional): date of the event in format
        """
        raise NotImplementedError

    def equal_timestep(self, time_step_h=
                       CONFIG['trop_cyclone']['time_step_h']):
        """ Generate interpolated track values to time steps of min_time_step.

        Parameters:
            time_step_h (float): time step in hours to which to interpolate
        """
        new_list = list()
        for track in self.data:
            if track.time.size > 3:
                time_step = str(time_step_h) + 'H'
                track_int = track.resample(time=time_step). \
                            interpolate('linear')
                track_int['time_step'] = ('time', track_int.time.size *
                                          [time_step_h])
                track_int.coords['lat'] = track.lat.resample(time=time_step).\
                                          interpolate('cubic')
                track_int.coords['lon'] = track.lon.resample(time=time_step).\
                                          interpolate('cubic')
                track_int.attrs = track.attrs
            else:
                LOGGER.warning('Track interpolation not done. ' +
                               'Not enough elements for %s', track.name)
                track_int = track
            new_list.append(track_int)

        self.data = new_list

    def calc_random_walk(self, ens_size=9, ens_amp0=1.5, max_angle=np.pi/10,
                         ens_amp=0.1, rand_unif_ini=None,
                         rand_unif_ang=None):
        """ Generate random tracks for every track contained.

        Parameters:
            ens_size (int, optional): number of created tracks per original
                track. Default 9.
            rand_unif_ini (np.array, optional): array of uniform [0,1) random
                numbers of size 2
            rand_unif_ang (np.array, optional): array of uniform [0,1) random
                numbers of size ens_size x size track
            ens_amp0 (float, optional): amplitude of max random starting point
                shift degree longitude. Default: 1.5
            max_angle (float, optional): maximum angle of variation, =pi is
                like undirected, pi/4 means one quadrant. Default: pi/10
            ens_amp (float, optional): amplitude of random walk wiggles in
                degree longitude for 'directed'. Default: 0.1
        """
        ens_track = list()
        for track in self.data:
            n_dat = track.time.size
            if rand_unif_ini is None or rand_unif_ini.shape != (2, ens_size):
                rand_unif_ini = np.random.uniform(size=(2, ens_size))
            if rand_unif_ang is None or rand_unif_ang.size != ens_size*n_dat:
                rand_unif_ang = np.random.uniform(size=ens_size*n_dat)

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

                d_lat_lon = d_xy + np.expand_dims(xy_ini[:, i_ens], axis=1)

                i_track.lon.values = i_track.lon.values + d_lat_lon[0, :]
                i_track.lat.values = i_track.lat.values + d_lat_lon[1, :]
                i_track.attrs['orig_event_flag'] = False
                i_track.attrs['name'] = i_track.attrs['name'] + '_gen' + \
                                        str(i_ens+1)
                i_track.attrs['id_no'] = i_track.attrs['id_no'] + (i_ens+1)/100

                ens_track.append(i_track)

        self.data = ens_track

    def plot(self, title=None):
        """Track over earth. Historical events are blue, probabilistic black.

        Parameters:
            title (str, optional): plot title

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        deg_border = 0.5
        fig, axis = plot.make_map()
        axis = axis[0][0]
        min_lat, max_lat = 10000, -10000
        min_lon, max_lon = 10000, -10000
        for track in self.data:
            min_lat, max_lat = min(min_lat, np.min(track.lat.values)), \
                               max(max_lat, np.max(track.lat.values))
            min_lon, max_lon = min(min_lon, np.min(track.lon.values)), \
                               max(max_lon, np.max(track.lon.values))
        axis.set_extent(([min_lon-deg_border, max_lon+deg_border,
                          min_lat-deg_border, max_lat+deg_border]))
        plot.add_shapes(axis)
        if title:
            axis.set_title(title)
        for track in self.data:
            if track.orig_event_flag:
                color = 'b'
            else:
                color = 'k'
            axis.plot(track.lon.values, track.lat.values, c=color)
        return fig, axis

    def calc_land_decay(self, s_rel=True):
        """Compute wind and pressure decay coefficients for every TC category
        from the historical events according to the formulas:
            - wind decay = exp(-x*A)
            - pressure decay = S-(S-1)*exp(-x*B)

        Parameters:
            s_rel (bool, optional): use environmental presure to calc S value
                (true) or central presure (false)

        Returns:
            v_rel (dict(category: A)), p_rel (dict(category: (S, B)))
        """
        hist_tracks = [track for track in self.data if track.orig_event_flag]

        if not hist_tracks:
            LOGGER.error('No historical tracks contained. Historical tracks' +
                         'of different TC categories are needed.')
            raise ValueError

        # Key is Saffir-Simpson scale
        # values are lists of wind/wind at landfall
        v_lf = dict()
        # values are tuples with first value the S parameter, second value
        # list of central pressure/central pressure at landfall
        p_lf = dict()
        # x-scale values to compute landfall decay
        x_val = dict()
        for track in hist_tracks:
            track['on_land'] = ('time', coord_util.coord_on_land(
                track.lat.values, track.lon.values))
            track['dist_since_lf'] = ('time', _dist_since_lf(track))
            _decay_values(s_rel, track, v_lf, p_lf, x_val)

        v_rel, p_rel = _decay_calc_coeff(x_val, v_lf, p_lf)

        return v_rel, p_rel

    def apply_land_decay(self, v_rel, p_rel):
        """Compute wind and pressure decay due to landfall in synthetic tracks.

        Parameters:
            v_rel (dict): {category: A}, where wind decay = exp(-x*A)
            p_rel (dict): (category: (S, B)},
                where pressure decay = S-(S-1)*exp(-x*B)
        """
        raise NotImplementedError

    @property
    def size(self):
        """ Get longitude from coord array """
        return len(self.data)

    def _read_one_csv(self, file_name):
        """Read IBTrACS track file.

            Parameters:
                file_name (str): file name containing one IBTrACS track to read
        """
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

        lat = dfr['cgps_lat'].values
        lon = dfr['cgps_lon'].values
        cen_pres = dfr['pcen'].values
        max_sus_wind = dfr['vmax'].values
        max_sus_wind_unit = 'kn'
        cen_pres = _missing_pressure(cen_pres, max_sus_wind, lat, lon)

        tr_ds = xr.Dataset()
        tr_ds.coords['time'] = ('time', datetimes)
        tr_ds.coords['lat'] = ('time', lat)
        tr_ds.coords['lon'] = ('time', lon)
        tr_ds['time_step'] = ('time', dfr['tint'].values)
        tr_ds['radius_max_wind'] = ('time', dfr['rmax'].values)
        tr_ds['max_sustained_wind'] = ('time', max_sus_wind)
        tr_ds['central_pressure'] = ('time', cen_pres)
        tr_ds['environmental_pressure'] = ('time', dfr['penv'].values)
        tr_ds.attrs['max_sustained_wind_unit'] = max_sus_wind_unit
        tr_ds.attrs['central_pressure_unit'] = 'mb'
        tr_ds.attrs['name'] = name
        tr_ds.attrs['orig_event_flag'] = bool(dfr['original_data']. values[0])
        tr_ds.attrs['data_provider'] = dfr['data_provider'].values[0]
        tr_ds.attrs['basin'] = dfr['gen_basin'].values[0]
        tr_ds.attrs['id_no'] = float(name.replace('N', '0'). replace('S', '1'))
        tr_ds.attrs['category'] = _set_category(max_sus_wind, \
                   max_sus_wind_unit)

        self.data.append(tr_ds)

def _dist_since_lf(track):
    """ Compute the distance to landfall point for every point on land.
    Points on water get nan values.

    Parameters:
        track (xr.Dataset): tropical cyclone track

    Returns:
        np.arrray
    """
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1)[0]
    orig_lf = _calc_orig_lf(track, sea_land_idx)

    dist_since_lf = np.empty(track.time.values.shape)
    for idx in np.argwhere(track.on_land.values)[:, 0]:
        try:
            orig_idx = np.argwhere(idx > sea_land_idx)[-1]
        except IndexError:
            track.on_land[idx] = False
            continue
        if idx == sea_land_idx[orig_idx] + 1:
            orig_coord = orig_lf[orig_idx].squeeze()
            dist_since_lf[idx] = np.sqrt(dist_sqr_approx(
                track.lat[idx], track.lon[idx],
                np.cos(track.lat[idx] / 180 * np.pi),
                orig_coord[0], orig_coord[1]))*ONE_LAT_KM
        elif idx > sea_land_idx[orig_idx]:
            dist_since_lf[idx] = dist_since_lf[idx-1] + \
                np.sqrt(dist_sqr_approx(track.lat[idx], track.lon[idx], \
                np.cos(track.lat[idx] / 180 * np.pi), track.lat[idx-1], \
                track.lon[idx-1]))*ONE_LAT_KM

    dist_since_lf[np.logical_not(track.on_land)] = np.nan

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

def _decay_values(s_rel, track, v_lf, p_lf, x_val):
    """ Compute wind and pressure relative to landafall values.

    Parameters:
        s_rel (bool): use environmental presure for S value (true) or
            central presure (false)
        track (xr.Dataset): track
        v_lf (dict): key is Saffir-Simpson scale, values are arrays of
            wind/wind at landfall
        p_lf (dict): key is Saffir-Simpson scale, values are tuples with
            first value the S parameter, second value array of central
            pressure/central pressure at landfall
        x_val (dict): key is Saffir-Simpson scale, values are arrays with
            the values used as "x" in the coefficient fitting, the
            distance since
    """
    sea_land_idx = np.where(np.diff(track.on_land.astype(int)) == 1) \
                   [0] + 1
    land_sea_idx = np.where(np.diff(track.on_land.astype(int)) == -1) \
                   [0] + 1
    if track.on_land[-1]:
        land_sea_idx = np.append(land_sea_idx, track.time.size)
    if sea_land_idx.size and land_sea_idx.size <= sea_land_idx.size:
        onland_time = land_sea_idx - sea_land_idx[0:land_sea_idx.size]
        for i_time in range(onland_time.size):
            v_landfall = track.max_sustained_wind \
                               [sea_land_idx[i_time]-1].values
            ss_scale_idx = np.where(v_landfall < SAFFIR_SIM_CAT)[0][0]+1

            v_land = track.max_sustained_wind[sea_land_idx[i_time]-1: \
                sea_land_idx[i_time]+onland_time[i_time]].values
            v_land = (v_land[1:]/v_land[0]).tolist()

            p_landfall = float(track.central_pressure[
                sea_land_idx[i_time]-1].values)
            p_land = track.central_pressure[sea_land_idx[i_time]-1: \
                sea_land_idx[i_time]+onland_time[i_time]].values
            p_land = (p_land[1:]/p_land[0]).tolist()

            if s_rel:
                p_land_s = track.environmental_pressure[np.argwhere(
                    track.on_land.values).squeeze()[-1]].values
            else:
                p_land_s = track.central_pressure[np.argwhere(
                    track.on_land.values).squeeze()[-1]].values
            p_land_s = len(p_land)*[float(p_land_s / p_landfall)]

            if ss_scale_idx not in v_lf:
                v_lf[ss_scale_idx] = array.array('f', v_land)
                p_lf[ss_scale_idx] = (array.array('f', p_land_s),
                                      array.array('f', p_land))
            else:
                v_lf[ss_scale_idx].extend(v_land)
                p_lf[ss_scale_idx][0].extend(p_land_s)
                p_lf[ss_scale_idx][1].extend(p_land)

        x_val[ss_scale_idx] = track.dist_since_lf[
            np.isfinite(track.dist_since_lf)].values

def _decay_calc_coeff(x_val, v_lf, p_lf):
    """ From track's relative velocity and pressure, compute the decay
    coefficients.
    - wind decay = exp(-x*A)
    - pressure decay = S-(S-1)*exp(-x*A)

    Parameters:
        x_val (dict): key is Saffir-Simpson scale, values are lists with
            the values used as "x" in the coefficient fitting, the
            distance since
        v_lf (dict): key is Saffir-Simpson scale, values are lists of
            wind/wind at landfall
        p_lf (dict): key is Saffir-Simpson scale, values are tuples with
            first value the S parameter, second value list of central
            pressure/central pressure at landfall

    Returns:
        v_rel (dict()), p_rel (dict())
    """
    v_rel = dict()
    p_rel = dict()
    for ss_scale, val_lf in v_lf.items():
        v_y_val = np.array(val_lf)
        v_coef = -np.log(v_y_val) / x_val[ss_scale]

        ps_y_val = np.array(p_lf[ss_scale][0])
        p_y_val = np.array(p_lf[ss_scale][1])
        p_y_val[ps_y_val <= p_y_val] = np.nan
        p_y_val[ps_y_val <= 1] = np.nan
        valid_p = np.isfinite(p_y_val)
        ps_y_val = ps_y_val[valid_p]
        p_y_val = p_y_val[valid_p]

        p_coef = -np.log((ps_y_val - p_y_val)/(ps_y_val - 1.0)) / \
                         x_val[ss_scale][valid_p]

        v_rel[ss_scale] = np.mean(v_coef)
        p_rel[ss_scale] = (np.mean(ps_y_val), np.mean(p_coef))

    scale_full = np.array(list(p_rel.keys()))
    for ss_scale in range(1, len(SAFFIR_SIM_CAT)+1):
        if ss_scale not in p_rel:
            close_scale = scale_full[np.argmin(np.abs(
                scale_full-ss_scale))]
            LOGGER.info('No historical track of category %s. Decay ' +
                        'parameters from category %s taken.', ss_scale,
                        close_scale)
            v_rel[ss_scale] = v_rel[close_scale]
            p_rel[ss_scale] = p_rel[close_scale]

    return v_rel, p_rel

def _missing_pressure(cen_pres, v_max, lat, lon):
    """Deal with missing central pressures."""
    if np.argwhere(cen_pres < 0).size > 0:
        cen_pres = 1024.388 + 0.047*lat - 0.029*lon - 0.818*v_max
    return cen_pres

def _set_category(max_sus_wind, max_sus_wind_unit):
    """Add storm category according to saffir-simpson hurricane scale
   -1 tropical depression
    0 tropical storm
    1 Hurrican category 1
    2 Hurrican category 2
    3 Hurrican category 3
    4 Hurrican category 4
    5 Hurrican category 5
    """
    ureg = UnitRegistry()
    if (max_sus_wind_unit == 'kn') or (max_sus_wind_unit == 'kt'):
        unit = ureg.knot
    elif max_sus_wind_unit == 'mph':
        unit = ureg.mile / ureg.hour
    elif max_sus_wind_unit == 'm/s':
        unit = ureg.meter / ureg.second
    elif max_sus_wind_unit == 'km/h':
        unit = ureg.kilometer / ureg.hour
    else:
        LOGGER.error('Wind not recorded in kn, conversion to kn needed.')
        raise ValueError
    max_wind_kn = (np.max(max_sus_wind) * unit).to(ureg.knot).magnitude

    return (np.argwhere(max_wind_kn < SAFFIR_SIM_CAT) - 1)[0][0]
