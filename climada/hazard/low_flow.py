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

Define LowFlow (LF) class.
WORK IN PROGRESS
"""

__all__ = ['LowFlow']

import logging
import os
import copy
import numba
import cftime
import itertools
import datetime as dt
import xarray as xr
import geopandas as gpd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from shapely.geometry import Point
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids import Centroids
from climada.util.coordinates import get_resolution

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'LF'
""" Hazard type acronym for Low Flow / Water Scarcity """

FILENAME_NC = '%s_%s_ewembi_%s_%s_%s_%s.nc'  #
# %(gh_model, cl_model, scenario, soc, fn_str_var, yearrange)

GH_MODEL = ['H08',
            'CLM45',
            'ORCHIDEE',
            'LPJmL',
            'WaterGAP2',
            'JULES-W1',
            'MATSIRO'
            ]
CL_MODEL = ['gfdl-esm2m',
            'hadgem2-es',
            'ipsl-cm5a-lr',
            'miroc5',
            ]
SCENARIO = ['historical',
            'rcp26',
            'rcp60'
            ]

SOC = ['histsoc',
       '2005soc',
       'rcp26soc',
       'rcp60soc']

FN_STR_VAR = 'co2_dis_global_daily'  # FileName STRing depending on VARiable
# (according to ISIMIP filenaming)


# list of year chunks: multiple files are combined
YEARCHUNKS = dict()
# historical:
YEARCHUNKS[SCENARIO[0]] = list()
for i in np.arange(1860, 2000, 10):
    YEARCHUNKS[SCENARIO[0]].append('%i_%i' % (i + 1, i + 10))
YEARCHUNKS[SCENARIO[0]].append('2001_2005')

# rcp26:
YEARCHUNKS[SCENARIO[1]] = ['2006_2010']
for i in np.arange(2010, 2090, 10):
    YEARCHUNKS[SCENARIO[1]].append('%i_%i' % (i + 1, i + 10))
YEARCHUNKS[SCENARIO[1]].append('2091_2099')
# rcp60:
YEARCHUNKS[SCENARIO[2]] = YEARCHUNKS[SCENARIO[1]]

REFERENCE_YEARRANGE = [1971, 2005]

TARGET_YEARRANGE = [2001, 2005]

BBOX = [-180, -85, 180, 85]  # [Lon min, lat min, lon max, lat max]


class LowFlow(Hazard):
    """Contains water scarcity events.

    Attributes:
        ...
    """
    clus_thresh_t = 1
    clus_thres_xy = 2  # 4
    min_samples = 1
    resolution = .5

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_from_nc(self, input_dir=None, centroids=None, countries=[], reg=None,
                    bbox=None, percentile=2.5, min_intensity=1, min_number_cells=1,
                    min_days_per_month=1,
                    yearrange=TARGET_YEARRANGE, yearrange_ref=REFERENCE_YEARRANGE,
                    gh_model=GH_MODEL[0], cl_model=CL_MODEL[0],
                    scenario=SCENARIO[0], scenario_ref=SCENARIO[0], soc=SOC[0], \
                    soc_ref=SOC[0], fn_str_var=FN_STR_VAR, keep_dis_data=False):
        """Wrapper to fill hazard from nc_dis file from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            centroids (Centroids): centroids
                (area that is considered, reg and country must be None)
            countries (list of countries ISO3) selection of countries
                (reg must be None!) [not yet implemented]
            reg (list of regions): can be set with region code if whole areas
                are considered (if not None, countries and centroids
                are ignored) [not yet implemented]
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            percentile (float): percentile used to compute threshold,
                0.0 < percentile < 100.0
            min_intensity (int): minimum intensity (nr of days) in an event event;
                events with lower max. intensity are dropped
            min_number_cells (int): minimum spatial extent (nr of grid cells)
                in an event event;
                events with lower geographical extent are dropped
            min_days_per_month (int): minimum nr of days below threshold in a month;
                months with lower nr of days below threshold are not considered
                for the event creation (clustering)
            yearrange (int tuple): year range for hazard set, f.i. (2001, 2005)
            yearrange_ref (int tuple): year range for reference (threshold),
                f.i. (1971, 2000)
            gh_model (str): abbrev. hydrological model (only when input_dir is selected)
                f.i. 'h08' etc.
            cl_model (str): abbrev. climate model (only when input_dir is selected)
                f.i. 'gfdl-esm2m' etc.
            scenario (str): climate change scenario (only when input_dir is selected)
                f.i. 'historical', 'rcp26', or 'rcp60'
            scenario_ref (str): climate change scenario for reference
                (only when input_dir is selected)
            soc (str): socio-economic trajectory (only when input_dir is selected)
                f.i. 'histsoc', '2005soc', 'rcp26soc', or 'rcp60soc'
            soc_ref (str): csocio-economic trajectory for reference
                (only when input_dir is selected)
            fn_str_var (str): FileName STRing depending on VARiable and
                ISIMIP simuation round
            keep_dis_data (boolean): keep monthly data (variable ndays = days below threshold)
                as dataframe (attribute "data") and save additional field 'relative_dis'?
        raises:
            NameError
        """
        print('GETTING STARTED!')
        if input_dir is not None:
            if not os.path.exists(input_dir):
                LOGGER.warning('Input directory %s does not exist', input_dir)
                raise NameError
        else:
            LOGGER.warning('Input directory %s not set', input_dir)
            raise NameError

        if centroids is not None:
            centr_handling = 'align'
        elif countries or reg:
            LOGGER.warning('country or reg ignored: not yet implemented')
            centr_handling = 'full_hazard'
        else:
            centr_handling = 'full_hazard'

        # read data and call preprocessing routine:
        self.data, centroids_import = _data_preprocessing_percentile(percentile, \
                                      yearrange, yearrange_ref, \
                                      input_dir, gh_model, cl_model, scenario, \
                                      scenario_ref, \
                                      soc, soc_ref, fn_str_var, bbox, min_days_per_month, \
                                      keep_dis_data)

        if centr_handling == 'full_hazard':
            centroids = centroids_import
        self.identify_clusters()

        # sum "dis" (days per month below threshold) per pixel and cluster_id
        # and write to hazard.intensiy
        self.events_from_clusters(centroids)
        if min_intensity > 1 or min_number_cells > 1:
            haz_tmp = self.filter_events(min_intensity=min_intensity, \
                                         min_number_cells=min_number_cells)
            LOGGER.info('Filtering events: %i events remaining', haz_tmp.size)
            self.event_id = haz_tmp.event_id
            self.event_name = list(map(str, self.event_id))
            self.date = haz_tmp.date
            self.date_end = haz_tmp.date_end
            self.orig = haz_tmp.orig
            self.frequency = haz_tmp.frequency
            self.intensity = haz_tmp.intensity
            self.fraction = haz_tmp.fraction
            del haz_tmp
        if not keep_dis_data:
            del self.data

    def events_from_clusters(self, centroids):
        """init hazard events from clusters"""
        # intensity = list()

        uni_ev = np.unique(self.data['cluster_id'].values)
        num_ev = uni_ev.size
        num_centr = centroids.size
        res_centr = self._centroids_resolution(centroids)

        # For one event, if more than one points of data dataframe have the
        # same coordinates, take the sum of days below threshold
        # of these points (duration as accumulated intensity).
        tree_centr = BallTree(centroids.coord, metric='chebyshev')
        if self.pool:
            chunksize = min(num_ev // self.pool.ncpus, 1000)
            intensity_list = self.pool.map(self._intensity_one_cluster,
                                           itertools.repeat(self.data, num_ev),
                                           itertools.repeat(tree_centr, num_ev),
                                           uni_ev, itertools.repeat(res_centr),
                                           itertools.repeat(num_centr),
                                           chunksize=chunksize)
        else:
            intensity_list = []
            for cl_id in uni_ev:
                intensity_list.append(self._intensity_one_cluster(self.data, \
                                      tree_centr, cl_id, res_centr, num_centr))

        self.tag = TagHazard(HAZ_TYPE)
        self.units = 'days'  # days below threshold
        self.centroids = centroids

        # Following values are defined for each event
        self.event_id = np.sort(self.data.cluster_id.unique())
        self.event_id = self.event_id[self.event_id>0]
        self.event_name = list(map(str, self.event_id))
        self.date = np.zeros(num_ev, int)
        self.date_end = np.zeros(num_ev, int)

        for ev_idx, ev_id in enumerate(uni_ev):
            self.date[ev_idx] = self.data[self.data.cluster_id == ev_id].dtime.min()
            self.date_end[ev_idx] = self.data[self.data.cluster_id == ev_id].dtime.max()
        self.orig = np.ones(num_ev, bool)
        self._set_frequency()

        # Following values are defined for each event and centroid
        self.intensity = sparse.lil_matrix(np.zeros((num_ev, num_centr)))
        for idx, ev_intensity in enumerate(intensity_list):
            self.intensity[idx] = ev_intensity
        self.intensity = self.intensity.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    def identify_clusters(self, clus_thres_xy=None, clus_thresh_t=None, min_samples=None):
        """call clustering and set events"""
        if min_samples:
            self.min_samples = min_samples
        if clus_thres_xy:
            self.clus_thres_xy = clus_thres_xy
        if clus_thresh_t:
            self.clus_thresh_t = clus_thresh_t

        self.data['cluster_id'] = np.zeros(len(self.data), dtype=int)

        # Compute clus_id: cluster identifier inside cons_id
        for cluster_vars in [['lat', 'lon'], ['lat', 'dt_month'], ['lon', 'dt_month']]:
            self.data = self._df_clustering(self.data, cluster_vars, \
                                            self.resolution, self.clus_thres_xy, \
                                            self.clus_thresh_t, self.min_samples)

        self.data = unique_clusters(self.data)
        return self.data


    def _set_frequency(self):
        """Set hazard frequency from intensity matrix. """
        delta_time = dt.datetime.fromordinal(int(np.max(self.date))).year - \
                     dt.datetime.fromordinal(int(np.min(self.date))).year + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    @staticmethod
    def _df_clustering(data, cluster_vars, res_data, clus_thres_xy, \
                       clus_thres_t, min_samples):
        """Compute 2D clusters and sort data with ascending clus_id
        for each combination of the 3 dimensions (lat, lon, dt_month).

        Parameters:
            data (dataframe): dataset obtained from ISIMIP  data
            res_data (float): input data grid resolution in degrees

        Returns:
            data
        """
        # set iter_var (dimension not used for clustering)
        if not 'lat' in cluster_vars:
            iter_var = 'lat'
        elif not 'lon' in cluster_vars:
            iter_var = 'lon'
        else:
            iter_var = 'dt_month'

        LOGGER.debug('Computing 3D clusters.')

        clus_id_var = 'c_%s_%s' % (cluster_vars[0], cluster_vars[1])
        data[clus_id_var] = np.zeros(len(data), dtype=int) - 1

        data_iter = data[data['iter_ev']][[iter_var, cluster_vars[0], cluster_vars[1], \
                                           'cons_id', clus_id_var]]

        if 'dt_month' in clus_id_var:
            # transform month count in accordance with spatial resolution
            # to achieve same distance between consecutive and geographically
            # neighboring points:
            data_iter.dt_month = data_iter.dt_month * res_data * clus_thres_xy / clus_thres_t

        # Loop over slices: For each slice, perform 2D clustering with DBSCAN
        for i_var in data_iter[iter_var].unique():
            temp = np.argwhere(data_iter[iter_var] == i_var).reshape(-1, )  # slice
            x_y = data_iter.iloc[temp][[cluster_vars[0], cluster_vars[1]]].values
            x_y_uni, x_y_cpy = np.unique(x_y, return_inverse=True, axis=0)
            cluster_id = DBSCAN(eps=res_data * clus_thres_xy, min_samples=min_samples). \
                fit(x_y_uni).labels_
            cluster_id = cluster_id[x_y_cpy]
            data_iter[clus_id_var].values[temp] = cluster_id + data_iter[clus_id_var].max() + 1

        data[clus_id_var].values[data['iter_ev'].values] = data_iter[clus_id_var].values

        return data

    def filter_events(self, min_intensity=1, min_number_cells=1):
        """remove events with max intensity below min_intensity or spatial extend
        below min_number_cells

        Parameters:
            min_intensity (int or float): Minimum criterion for intensity
            min_number_cells (int or float): Minimum crietrion for number of grid cell

        Returns:
            Hazard
        """
        haz_tmp = copy.deepcopy(self)
        haz_tmp.date_tmp = haz_tmp.date
        for i_event, _ in enumerate(haz_tmp.event_id):
            if np.sum(haz_tmp.intensity[i_event] > 0) < min_number_cells or \
                    np.max(haz_tmp.intensity[i_event]) < min_intensity:
                haz_tmp.date[i_event] = 2100001
        haz_tmp = haz_tmp.select(date=(1, 2000000))
        haz_tmp.date = haz_tmp.date_tmp
        del haz_tmp.date_tmp
        haz_tmp.event_id = np.arange(1, len(haz_tmp.event_id) + 1).astype(int)
        return haz_tmp

    @staticmethod
    def _centroids_resolution(centroids):
        """ Return resolution of the centroids in their units

        Parameters:
            centroids (Centroids): centroids instance

        Returns:
            float
        """
        if centroids.meta:
            res_centr = abs(centroids.meta['transform'][4]), \
                        centroids.meta['transform'][0]
        else:
            res_centr = get_resolution(centroids.lat, centroids.lon)
        if abs(abs(res_centr[0]) - abs(res_centr[1])) > 1.0e-6:
            LOGGER.warning('Centroids do not represent regular pixels %s.', str(res_centr))
            return (res_centr[0] + res_centr[1]) / 2
        return res_centr[0]

    @staticmethod
    def _intensity_one_cluster(data, tree_centr, cluster_id, res_centr, num_centr):
        """ For a given cluster, fill in an intensity np.array with the summed intensity
        at each centroid.

        Parameters:
            data (DataFrame)
            centroids (Centroids): centroids for the dataset
            cluster_id (int): id of the selected cluster
            res_centr (float): resolution of centroids in degree
            num_centr (int): number of centroids

        Returns:
            intensity_cl (np.array): summed intensity of cluster at each centroids

        """
        LOGGER.debug('Number of days below threshold corresponding to event %s.', str(cluster_id))
        temp_data = data.reindex(index=(np.argwhere(data['cluster_id'] == \
                                 cluster_id).reshape(-1, )),
                                 columns=['lat', 'lon', 'ndays'])

        # Identifies the unique (lat,lon) points of the firms dataframe -> lat_lon_uni
        # Set the same index value for each duplicate (lat,lon) points -> lat_lon_cpy
        lat_lon_uni, lat_lon_cpy = np.unique(temp_data[['lat', 'lon']].values,
                                             return_inverse=True, axis=0)
        index_uni = np.unique(lat_lon_cpy, axis=0)

        # Search closest centroid for each firms point
        ind, _ = tree_centr.query_radius(lat_lon_uni, r=res_centr / 2, count_only=False,
                                         return_distance=True, sort_results=True)
        ind = np.array([ind_i[0] if ind_i.size else -1 for ind_i in ind])
        intensity_cl = _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy,
                                       temp_data['ndays'].values)
        return sparse.lil_matrix(intensity_cl)


def _init_centroids(data_x, centr_res_factor=1):
    """ Get centroids from the firms dataset and refactor them.

    Parameters:
        data_x (xarray): dataset obtained from ISIMIP netcdf
        centr_res_factor (float): the factor applied to voluntarly decrease/increase
            the centroids resolution

    Returns:
        centroids (Centroids)
    """
    res_data = np.min(np.abs([np.diff(data_x.lon.values).min(), np.diff(data_x.lat.values).min()]))
    centroids = Centroids()
    centroids.set_raster_from_pnt_bounds((data_x.lon.values.min(), \
                                          data_x.lat.values.min(), data_x.lon.values.max(), \
                                          data_x.lat.values.max()), res=res_data / centr_res_factor)
    centroids.set_meta_to_lat_lon()
    centroids.set_area_approx()
    centroids.set_on_land()
    centroids.empty_geometry_points()
    return centroids


def unique_clusters(data):
    """identify unqiue clustes based on clusters in 3 dimensions and set unique
    cluster_id"""
    data.cluster_id = np.zeros(len(data.c_lat_lon)) - 1

    data.loc[data.c_lat_lon == -1, 'c_lat_lon'] = np.nan
    data.loc[data.c_lat_dt_month == -1, 'c_lat_dt_month'] = np.nan
    data.loc[data.c_lon_dt_month == -1, 'c_lon_dt_month'] = np.nan

    cc = 0  # event id counter
    current = 0
    for c_lat_lon in data.c_lat_lon.unique():
        if np.isnan(c_lat_lon):
            data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'] = -1
        else:
            if len(data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'].unique()) == 1 and -1 in data.loc[
                    data.c_lat_lon == c_lat_lon, 'cluster_id'].unique():
                cc += 1
                current = cc
            else:
                current = max(data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'].unique())
            data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'] = current

            for c_lat_dt_month in data.loc[data.c_lat_lon == c_lat_lon, 'c_lat_dt_month'].unique():
                if not np.isnan(c_lat_dt_month):
                    data.loc[data.c_lat_dt_month == c_lat_dt_month, 'cluster_id'] = current
            for c_lon_dt_month in data.loc[data.c_lat_lon == c_lat_lon, 'c_lon_dt_month'].unique():
                if not np.isnan(c_lon_dt_month):
                    data.loc[data.c_lon_dt_month == c_lon_dt_month, 'cluster_id'] = current

    data.loc[np.isnan(data.c_lon_dt_month), 'cluster_id'] = -1
    data.loc[np.isnan(data.c_lat_dt_month), 'cluster_id'] = -1
    data.cluster_id = data.cluster_id.astype(int)
    return data


def _data_preprocessing_percentile(percentile, yearrange, yearrange_ref, \
                                   input_dir, gh_model, cl_model, scenario, \
                                   scenario_ref, soc, soc_ref, fn_str_var, bbox, \
                                   min_days_per_month, keep_dis_data):
    """load data and reference data and calculate monthly percentiles
    then extract intensity based on days below threshold
    returns geopandas dataframe

    returns:
        df (DataFrame): preprocessed data with days below threshold per grid cell and month
        centroids (Centroids instance): regular grid centroid with same resolution as input data
    """
    data = _read_and_combine_nc(yearrange, input_dir, gh_model, cl_model, \
                                scenario, soc, fn_str_var, bbox)
    threshold_grid = _compute_threshold_grid(percentile, yearrange_ref, \
                                             input_dir, gh_model, cl_model, \
                                             scenario_ref, soc_ref, fn_str_var, bbox)
    data = _days_below_threshold_per_month(data, threshold_grid, min_days_per_month, keep_dis_data)
    df = _xarray_to_geopandas(data)
    df = df.sort_values(['lat', 'lon', 'dtime'], ascending=[True, True, True])
    centroids = _init_centroids(data, centr_res_factor=1)
    return df.reset_index(drop=True), centroids


def _read_and_combine_nc(yearrange, input_dir, gh_model, cl_model, \
                         scenario, soc, fn_str_var, bbox, fn=FILENAME_NC):
    """import and combine data from nc files, return as xarray"""

    first_file = True
    for _, yearchunk in enumerate(YEARCHUNKS[scenario]):
        # skip if file is not required, i.e. not in yearrange:
        if int(yearchunk[0:4]) > yearrange[1] or int(yearchunk[-4:]) < yearrange[0]:
            continue
        filename = os.path.join(input_dir, fn % (gh_model, cl_model, scenario, \
                                                 soc, fn_str_var, yearchunk))
        if not os.path.isfile(filename):
            LOGGER.error('Netcdf file not found: %s', filename)
            FileNotFoundError
        if first_file:
            data = _read_single_nc(filename, yearrange, bbox)
            first_file = False
        else:
            data = data.combine_first(_read_single_nc(filename, yearrange, bbox))

    # set negative discharge values to zero (debugging of input data):
    data.dis.values[data.dis.values < 0] = 0
    return data


def _read_single_nc(filename, yearrange, bbox):
    """import data from single nc file, return as xarray"""
    data = xr.open_dataset(filename)
    try:
        if not bbox:
            return data.sel(time=slice(dt.datetime(yearrange[0], 1, 1), \
                                       dt.datetime(yearrange[-1], 12, 31)))
        return data.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]), \
                        time=slice(dt.datetime(yearrange[0], 1, 1), \
                                   dt.datetime(yearrange[-1], 12, 31)))
    except TypeError:
        # fix date format if not datetime
        if not bbox:
            data = data.sel(time=slice(cftime.DatetimeNoLeap(yearrange[0], 1, 1), \
                                       cftime.DatetimeNoLeap(yearrange[-1], 12, 31)))
        else:
            data = data.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]), \
                            time=slice(cftime.DatetimeNoLeap(yearrange[0], 1, 1), \
                                       cftime.DatetimeNoLeap(yearrange[-1], 12, 31)))
        datetimeindex = data.indexes['time'].to_datetimeindex()
        data['time'] = datetimeindex
    return data


def _compute_threshold_grid(percentile, yearrange_ref, input_dir, gh_model, cl_model, \
                            scenario, soc, fn_str_var, bbox):
    """returns the x-th percentile for every pixel over a given
    time horizon (based on daily data) [all-year round percentiles!]"""
    LOGGER.info('Computing threshold value per grid cell for Q%i, %i-%i', \
                percentile, yearrange_ref[0], yearrange_ref[1])
    data = _read_and_combine_nc(yearrange_ref, input_dir, gh_model, cl_model, \
                                scenario, soc, fn_str_var, bbox)
    return data.reduce(np.nanpercentile, dim='time', q=percentile)


def _compute_threshold_grid_per_month(percentile, yearrange_ref, input_dir, \
                                      gh_model, cl_model, scenario, soc, fn_str_var, bbox):
    """returns the x-th percentile for every pixel over a given
    time horizon per month (based on daily data)
    OUTDATED"""
    LOGGER.info('Computing threshold value per grid cell for Q%i, %i-%i', \
                 percentile, yearrange_ref[0], yearrange_ref[1])
    data = _read_and_combine_nc(yearrange_ref, input_dir, gh_model, cl_model, \
                                scenario, soc, fn_str_var, bbox)
    return data.groupby('time.month').reduce(np.nanpercentile, dim='time', q=percentile)


def _days_below_threshold_per_month(data, threshold_grid, min_days_per_month, keep_dis_data):
    """returns sum of days below threshold per month (as xarray with monthly data)

    if keep_dis_data is True, a DataFrame called 'data' with additional data is saved within
    the hazard object. It provides data per event, grid cell, and month
    data comes with the following columns: ['lat', 'lon', 'time', 'ndays',
       'relative_dis', 'iter_ev', 'cons_id',
       'dtime', 'dt_month', 'geometry', 'cluster_id', 'c_lat_lon',
       'c_lat_dt_month', 'c_lon_dt_month']
    Note: cluster_id corresponds 1:1 with associated event_id.

    """
    # data = data.groupby('time.month')-threshold_grid # outdated
    data_threshold = data - threshold_grid
    if keep_dis_data:
        data_low = data.where(data_threshold < 0) / data.mean(dim='time')
        data_low = data_low.resample(time='1M').mean()
    data_threshold.dis.values[data_threshold.dis.values >= 0] = 0
    data_threshold.dis.values[data_threshold.dis.values < 0] = 1
    data_threshold = data_threshold.resample(time='1M').sum()
    data_threshold.dis.values[data_threshold.dis.values < min_days_per_month] = 0
    data_threshold = data_threshold.rename({'dis': 'ndays'})
    if keep_dis_data:
        data_threshold['relative_dis'] = data_low['dis']
    return data_threshold.where(data_threshold['ndays'] > 0)


def _xarray_to_geopandas(data):
    """returns prooessed geopanda dataframe with NaN dropped"""
    df = data.to_dataframe()
    df.reset_index(inplace=True)
    df = df.dropna()
    df['iter_ev'] = np.ones(len(df), bool)
    df['cons_id'] = np.zeros(len(df), int) - 1
    # df['cluster_id'] = np.zeros(len(df), int)
    # df['clus_id'] = np.zeros(len(df), int) - 1
    df['dtime'] = df['time'].apply(lambda x: x.toordinal())
    df['dt_month'] = df['time'].apply(lambda x: x.year * 12 + x.month)
    return gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in zip(df['lon'], df['lat'])])

@numba.njit
def _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy, intensity_raw):
    intensity_cl = np.zeros((1, num_centr), dtype=numba.float64)
    for idx in range(index_uni.size):
        if ind[idx] != -1:
            intensity_cl[0, ind[idx]] = \
                np.sum(intensity_raw[lat_lon_cpy == index_uni[idx]])  # ToDo: TEST!
    return intensity_cl
