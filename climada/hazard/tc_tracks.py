"""
Define TCTracks: IBTracs reader and tracks manager.
"""

__all__ = ['SAFFIR_SIM_CAT', 'TCTracks', 'set_category']

import logging
import datetime as dt
import array
import itertools
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import pandas as pd
import xarray as xr
from sklearn.neighbors import DistanceMetric
from pint import UnitRegistry
from pathos.multiprocessing import ProcessingPool as Pool

from climada.util.config import CONFIG
import climada.util.coordinates as coord_util
from climada.util.constants import EARTH_RADIUS_KM
from climada.util.files_handler import get_file_names
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

SAFFIR_SIM_CAT = [34, 64, 83, 96, 113, 135, 1000]
""" Saffir-Simpson Hurricane Wind Scale """

CAT_NAMES = {1: 'Tropical Depression', 2: 'Tropical Storm',
             3: 'Hurrican Cat. 1', 4: 'Hurrican Cat. 2',
             5: 'Hurrican Cat. 3', 6: 'Hurrican Cat. 4', 7: 'Hurrican Cat. 5'}
""" Saffir-Simpson category names. """

CAT_COLORS = cm.rainbow(np.linspace(0, 1, len(SAFFIR_SIM_CAT)))
""" Color scale to plot the Saffir-Simpson scale."""

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
                - orig_event_flag (attrs)
                - data_provider (attrs)
                - basin (attrs)
                - id_no (attrs)
                - category (attrs)
            computed during processing:
                - on_land
                - dist_since_lf
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

    def get_track(self, track_name=None):
        """Get track with provided name. Return all tracks if no name provided.

        Parameters:
            track_name (str, optional): name of track (ibtracsID for IBTrACS)

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

        LOGGER.info('No track with name %s found.', track_name)
        return []

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

    def equal_timestep(self, time_step_h=CONFIG['trop_cyclone']['time_step_h'],
                       land_params=True):
        """ Generate interpolated track values to time steps of min_time_step.

        Parameters:
            time_step_h (float): time step in hours to which to interpolate
            land_params (bool, optional): compute on_land and dist_since_lf at
                each node. Default: False.
        """
        LOGGER.info('Interpolating %s tracks to %sh time steps.', self.size,
                    time_step_h)

        if land_params:
            land_geom = _calc_land_geom(self.data)
        else:
            land_geom = None

        chunksize = min(self.size, 500)
        self.data = Pool().map(self._one_interp_data, self.data,
                               itertools.repeat(time_step_h, self.size),
                               itertools.repeat(land_geom, self.size),
                               chunksize=chunksize)

    def calc_random_walk(self, ens_size=9, ens_amp0=1.5, max_angle=np.pi/10, \
        ens_amp=0.1, seed=CONFIG['trop_cyclone']['random_seed'], decay=True):
        """ Generate synthetic tracks. An ensamble of tracks is computed for
        every track contained.

        Parameters:
            ens_size (int, optional): number of ensamble per original track.
                Default 9.
            ens_amp0 (float, optional): amplitude of max random starting point
                shift degree longitude. Default: 1.5
            max_angle (float, optional): maximum angle of variation, =pi is
                like undirected, pi/4 means one quadrant. Default: pi/10
            ens_amp (float, optional): amplitude of random walk wiggles in
                degree longitude for 'directed'. Default: 0.1
            seed (int, optional): random number generator seed. Put negative
                value if you don't want to use it. Default: configuration file
            decay (bool, optional): compute land decay in probabilistic tracks.
                Default: True
        """
        LOGGER.info('Computing %s synthetic tracks.', ens_size*self.size)

        if seed >= 0:
            np.random.seed(seed)

        # problem random num generator in multiprocessing. python 3.7?
        new_ens = list()
        for track in self.data:
            new_ens.extend(self._one_rnd_walk(track, ens_size, ens_amp0,
                                              ens_amp, max_angle))
        self.data = new_ens

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

    def plot(self, title=None):
        """Track over earth. Historical events are blue, probabilistic black.

        Parameters:
            title (str, optional): plot title

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if not self.size:
            LOGGER.info('No tracks to plot')
            return None

        deg_border = 0.5
        fig, axis = u_plot.make_map()
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
        u_plot.add_shapes(axis)

        synth_flag = False
        cmap = ListedColormap(colors=CAT_COLORS)
        norm = BoundaryNorm([0] + SAFFIR_SIM_CAT, len(SAFFIR_SIM_CAT))
        if title:
            axis.set_title(title)
        for track in self.data:
            points = np.array([track.lon.values,
                               track.lat.values]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            if track.orig_event_flag:
                track_lc = LineCollection(segments, cmap=cmap, norm=norm, \
                    linestyle='solid', transform=ccrs.PlateCarree(), lw=2)
            else:
                synth_flag = True
                track_lc = LineCollection(segments, cmap=cmap, norm=norm, \
                    linestyle=':', transform=ccrs.PlateCarree(), lw=2)
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

        axis.legend(leg_lines, leg_names)
        return fig, axis

    @staticmethod
    def _one_rnd_walk(track, ens_size, ens_amp0, ens_amp, max_angle):
        """ Interpolate values of one track.

        Parameters:
            track (xr.Dataset): track data

        Returns:
            list(xr.Dataset)
        """
        ens_track = list()
        n_dat = track.time.size
        rand_unif_ini = np.random.uniform(size=(2, ens_size))
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

        return ens_track

    @staticmethod
    def _one_interp_data(track, time_step_h, land_geom=None):
        """ Interpolate values of one track.

        Parameters:
            track (xr.Dataset): track data

        Returns:
            xr.Dataset
        """
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
            track.attrs['category'] = set_category( \
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
        chunksize = min(len(hist_tracks), 500)
        for (tv_lf, tp_lf, tx_val) in Pool().map(_decay_values,
                                                 hist_tracks,
                                                 itertools.repeat(land_geom),
                                                 itertools.repeat(s_rel),
                                                 chunksize=chunksize):
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

        chunksize = min(self.size, 500)
        self.data = Pool().map(_apply_decay_coeffs, self.data,
                               itertools.repeat(v_rel),
                               itertools.repeat(p_rel),
                               itertools.repeat(land_geom),
                               itertools.repeat(s_rel),
                               chunksize=chunksize)

        if check_plot:
            _check_apply_decay_plot(self.data, orig_wind, orig_pres)

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

    return coord_util.get_land_geometry(border=(min_lon, max_lon, \
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
    """ Compute the distance to landfall point for every point on land.
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

def _calc_decay_ps_value(track, p_landfall, s_rel):
    if s_rel:
        p_land_s = track.environmental_pressure[np.argwhere(
            track.on_land.values).squeeze((-1,))[-1]].values
    else:
        p_land_s = track.central_pressure[np.argwhere(
            track.on_land.values).squeeze((-1,))[-1]].values
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
            v_land = (v_land[1:]/v_land[0]).tolist()

            p_landfall = float(track.central_pressure[sea_land-1].values)
            p_land = track.central_pressure[sea_land-1:land_sea].values
            p_land = (p_land[1:]/p_land[0]).tolist()

            p_land_s = _calc_decay_ps_value(track, p_landfall, s_rel)
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
        x_val_ss = np.array(x_val[ss_scale])

        y_val = np.array(val_lf)
        v_coef = _solve_decay_v_function(y_val, x_val_ss)

        ps_y_val = np.array(p_lf[ss_scale][0])
        y_val = np.array(p_lf[ss_scale][1])
        y_val[ps_y_val <= y_val] = np.nan
        y_val[ps_y_val <= 1] = np.nan
        valid_p = np.isfinite(y_val)
        ps_y_val = ps_y_val[valid_p]
        y_val = y_val[valid_p]
        p_coef = _solve_decay_p_function(ps_y_val, y_val, x_val_ss[valid_p])

        v_rel[ss_scale] = np.mean(v_coef)
        p_rel[ss_scale] = (np.mean(ps_y_val), np.mean(p_coef))

    scale_fill = np.array(list(p_rel.keys()))
    if not scale_fill.size:
        LOGGER.info('No historical track with landfall.')
        return v_rel, p_rel
    for ss_scale in range(1, len(SAFFIR_SIM_CAT)+1):
        if ss_scale not in p_rel:
            close_scale = scale_fill[np.argmin(np.abs(scale_fill-ss_scale))]
            LOGGER.debug('No historical track of category %s. Decay ' \
                         'parameters from category %s taken.',
                         CAT_NAMES[ss_scale], CAT_NAMES[close_scale])
            v_rel[ss_scale] = v_rel[close_scale]
            p_rel[ss_scale] = p_rel[close_scale]

    return v_rel, p_rel

def _check_decay_values_plot(x_val, v_lf, p_lf, v_rel, p_rel):
    """ Generate one graph with wind decay and an other with central pressure
    decay, true and approximated."""
    # One graph per TC category
    for track_cat, color in zip(v_lf.keys(),
                                cm.rainbow(np.linspace(0, 1, len(v_lf)))):
        graph = u_plot.Graph2D('', 2)
        x_eval = np.linspace(0, np.max(x_val[track_cat]), 20)

        graph.add_subplot('Distance from landfall (km)', \
            'Max sustained wind relative to landfall', 'Wind')
        graph.add_curve(x_val[track_cat], v_lf[track_cat], '*', c=color,
                        label=CAT_NAMES[track_cat])
        graph.add_curve(x_eval, _decay_v_function(v_rel[track_cat], x_eval),
                        '-', c=color)

        graph.add_subplot('Distance from landfall (km)', \
            'Central pressure relative to landfall', 'Pressure')
        graph.add_curve(x_val[track_cat], p_lf[track_cat][1], '*', c=color,
                        label=CAT_NAMES[track_cat])
        graph.add_curve(x_eval, _decay_p_function(p_rel[track_cat][0], \
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
        ss_scale_idx = np.where(v_landfall < SAFFIR_SIM_CAT)[0][0]+1
        if land_sea - sea_land == 1:
            continue
        p_decay = _calc_decay_ps_value(track, p_landfall, s_rel)
        p_decay = _decay_p_function(p_decay, p_rel[ss_scale_idx][1], \
            track.dist_since_lf[sea_land:land_sea].values)
        track.central_pressure[sea_land:land_sea] = p_landfall * p_decay

        v_decay = _decay_v_function(v_rel[ss_scale_idx], \
            track.dist_since_lf[sea_land:land_sea].values)
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
    graph_v_b = u_plot.Graph2D('Wind before land decay correction')
    graph_v_b.add_subplot('Node number', 'Max sustained wind (kn)')
    graph_v_a = u_plot.Graph2D('Wind after land decay correction')
    graph_v_a.add_subplot('Node number', 'Max sustained wind (kn)')

    graph_p_b = u_plot.Graph2D('Pressure before land decay correction')
    graph_p_b.add_subplot('Node number', 'Central pressure (mb)')
    graph_p_a = u_plot.Graph2D('Pressure after land decay correction')
    graph_p_a.add_subplot('Node number', 'Central pressure (mb)')

    graph_pd_a = u_plot.Graph2D('Relative pressure after land decay correction')
    graph_pd_a.add_subplot('Distance from landfall (km)',
                           'Central pressure relative to landfall')
    graph_ped_a = u_plot.Graph2D('Environmental - central pressure after land ' +
                                 'decay correction')
    graph_ped_a.add_subplot('Distance from landfall (km)',
                            'Environmental pressure - Central pressure (mb)')

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

                graph_v_a.add_curve(on_land, track.max_sustained_wind[on_land],
                                    'o', c=CAT_COLORS[ss_scale])
                graph_v_b.add_curve(on_land, orig_wind[on_land],
                                    'o', c=CAT_COLORS[ss_scale])
                graph_p_a.add_curve(on_land, track.central_pressure[on_land],
                                    'o', c=CAT_COLORS[ss_scale])
                graph_p_b.add_curve(on_land, orig_pres[on_land],
                                    'o', c=CAT_COLORS[ss_scale])
                graph_pd_a.add_curve(track.dist_since_lf[on_land],
                                     track.central_pressure[on_land]/p_lf,
                                     'o', c=CAT_COLORS[ss_scale])
                graph_ped_a.add_curve(track.dist_since_lf[on_land],
                                      track.environmental_pressure[on_land]-
                                      track.central_pressure[on_land],
                                      'o', c=CAT_COLORS[ss_scale])

            on_sea = np.arange(track.time.size)[np.logical_not(track.on_land)]
            graph_v_a.add_curve(on_sea, track.max_sustained_wind[on_sea],
                                'o', c='k', markersize=5)
            graph_v_b.add_curve(on_sea, orig_wind[on_sea],
                                'o', c='k', markersize=5)
            graph_p_a.add_curve(on_sea, track.central_pressure[on_sea],
                                'o', c='k', markersize=5)
            graph_p_b.add_curve(on_sea, orig_pres[on_sea],
                                'o', c='k', markersize=5)

    return graph_v_b, graph_v_a, graph_p_b, graph_p_a, graph_pd_a, graph_ped_a

def _check_apply_decay_hist_plot(hist_tracks):
    """Plot winds and pressures of historical tracks."""
    graph_hv = u_plot.Graph2D('Historical wind')
    graph_hv.add_subplot('Node number', 'Max sustained wind (kn)')

    graph_hp = u_plot.Graph2D('Historical pressure')
    graph_hp.add_subplot('Node number', 'Central pressure (mb)')

    graph_hpd_a = u_plot.Graph2D('Historical relative pressure')
    graph_hpd_a.add_subplot('Distance from landfall (km)',
                            'Central pressure relative to landfall')
    graph_hped_a = u_plot.Graph2D('Historical environmental - central pressure')
    graph_hped_a.add_subplot('Distance from landfall (km)',
                             'Environmental pressure - Central pressure (mb)')
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
                graph_hpd_a.add_curve(track.dist_since_lf[on_land],
                                      track.central_pressure[on_land]/p_lf,
                                      'o', c=CAT_COLORS[scale])
                graph_hped_a.add_curve(track.dist_since_lf[on_land],
                                       track.environmental_pressure[on_land]-
                                       track.central_pressure[on_land],
                                       'o', c=CAT_COLORS[scale])

            on_sea = np.arange(track.time.size)[np.logical_not(track.on_land)]
            graph_hp.add_curve(on_sea, track.central_pressure[on_sea],
                               'o', c='k', markersize=5)
            graph_hv.add_curve(on_sea, track.max_sustained_wind[on_sea],
                               'o', c='k', markersize=5)

    return graph_hv, graph_hp, graph_hpd_a, graph_hped_a

def _missing_pressure(cen_pres, v_max, lat, lon):
    """Deal with missing central pressures."""
    if np.argwhere(cen_pres <= 0).size > 0:
        cen_pres = 1024.388 + 0.047*lat - 0.029*lon - 0.818*v_max # ibtracs 1980 -2013 (r2=0.91)
#        cen_pres = 1024.688+0.055*lat-0.028*lon-0.815*v_max      # peduzzi
    return cen_pres

def set_category(max_sus_wind, max_sus_wind_unit):
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

    Returns:
        double
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
