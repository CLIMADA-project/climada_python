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
import itertools
import datetime as dt
import cftime
import xarray as xr
import geopandas as gpd
import numpy as np
import numba

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
"""Hazard type acronym for Low Flow / Water Scarcity"""

FILENAME_NC = '%s_%s_%s_%s_%s_%s_%s.nc'
"""structure of ISIMIP discharge output data taking the following strings:
    %(gh_model, cl_model, scenario, soc, fn_str_var, yearrange)"""

GH_MODEL = ['H08',
            'CLM45',
            'ORCHIDEE',
            'LPJmL',
            'WaterGAP2',
            'JULES-W1',
            'MATSIRO'
            ]
"""available gridded hydrological models"""

CL_MODEL = ['gfdl-esm2m',
            'hadgem2-es',
            'ipsl-cm5a-lr',
            'miroc5',
            'gswp3',
            'wfdei',
            'princeton',
            'watch',
            ]
"""climate forcings: available global gridded climate models / re-analysis data"""

SCENARIO = ['historical',
            'rcp26',
            'rcp60',
            'hist']
"""climate scenarios"""

SOC = ['histsoc', # historical
       '2005soc', # constant at 2005 level
       'rcp26soc',
       'rcp60soc',
       'pressoc']
"""socio-economic parameter sets of model runs"""

FN_STR_VAR = 'co2_dis_global_daily'  # FileName STRing depending on VARiable
"""constant part of discharge output file (according to ISIMIP filenaming)"""

YEARCHUNKS = dict()
"""list of year chunks: multiple files are combined"""

YEARCHUNKS[SCENARIO[0]] = list()
"""historical year chunks ISIMIP 2b"""
for i in np.arange(1860, 2000, 10):
    YEARCHUNKS[SCENARIO[0]].append('%i_%i' % (i + 1, i + 10))
YEARCHUNKS[SCENARIO[0]].append('2001_2005')

YEARCHUNKS[SCENARIO[3]] = list()
"""historical year chunks ISIMIP 2a"""
for i in np.arange(1970, 2010, 10):
    YEARCHUNKS[SCENARIO[3]].append('%i_%i' % (i + 1, i + 10))

YEARCHUNKS[SCENARIO[1]] = ['2006_2010']
for i in np.arange(2010, 2090, 10):
    YEARCHUNKS[SCENARIO[1]].append('%i_%i' % (i + 1, i + 10))
YEARCHUNKS[SCENARIO[1]].append('2091_2099')
YEARCHUNKS[SCENARIO[2]] = YEARCHUNKS[SCENARIO[1]]
"""future year chunks"""

REFERENCE_YEARRANGE = (1971, 2005)
"""default year range used to compute threshold (base line reference)"""

TARGET_YEARRANGE = (2001, 2005)
"""default year range of hazard"""

BBOX = [-180, -85, 180, 85]
"""default geographical bounding box: [lon_min, lat_min, lon_max, lat_max]"""

# reducing these two parameters decreases memory load but increases computation time:
BBOX_WIDTH = 75
"""default width and height of geographical bounding boxes for loop in degree lat/lon.
i.e. the bounding box is split into square boxes with maximum size BBOX_WIDTH*BBOX_WIDTH
(avoid memory usage spike)"""
INTENSITY_STEP = 300
"""max. number of events to be written to hazard.intensity matrix at once
(avoid memory usage spike)"""

class LowFlow(Hazard):
    """Contains water scarcity events.

    Attributes:
        date_start (np.array(int)): for every event, the starting date (ordinal)
            (the Hazard attribute 'date' contains the date of maximum event intensity)
        date_end (np.array(int)): for every event, the starting date (ordinal)
    """

    clus_thresh_t = 1
    """Default maximum time difference in months
    to be counted as connected points during clustering"""
    clus_thres_xy = 2  # 4
    """Default maximum grid cell distance (number of grid cells)
    to be counted as connected points during clustering"""
    min_samples = 1
    """Default minimum amount of data points in one cluster to retain the cluster"""
    resolution = .5
    """Default spatial resoultion of input data in degree lat/lon"""

    def __init__(self, pool=None):
        """Empty constructor."""
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_from_nc(self, input_dir=None, centroids=None, countries=None, reg=None,
                    bbox=None, percentile=2.5, min_intensity=1, min_number_cells=1,
                    min_days_per_month=1, yearrange=TARGET_YEARRANGE,
                    yearrange_ref=REFERENCE_YEARRANGE, gh_model=GH_MODEL[0], cl_model=CL_MODEL[0],
                    scenario=SCENARIO[0], scenario_ref=SCENARIO[0], soc=SOC[0],
                    soc_ref=SOC[0], fn_str_var=FN_STR_VAR, keep_dis_data=False,
                    yearchunks='default', mask_threshold=('mean', 1)):
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
                as dataframe (attribute "data") and save additional field 'relative_dis'
                (relative discharge compared to the long term)
            yearchunks: list of year chunks corresponding to each nc flow file. If set to
                'default', uses the chunking corresponding to the scenario.
            mask_threshold: tuple with threshold value [1] for criterion [0] for mask:
                Threshold below which the grid is masked out. e.g.:
                ('mean', 1.) --> grid cells with a mean discharge below 1 are ignored
                ('percentile', .3) --> grid cells with a value of the computed percentile discharge
                values below 0.3 are ignored. default: ('mean', 1}). Set to None for
                no threshold.
                Provide a list of tuples for multiple thresholds.
        raises:
            NameError
        """
        print('GETTING STARTED!')
        if input_dir:
            if not os.path.exists(input_dir):
                LOGGER.warning('Input directory %s does not exist', input_dir)
                raise NameError
        else:
            LOGGER.warning('Input directory %s not set', input_dir)
            raise NameError

        if centroids:
            centr_handling = 'align'
        elif countries or reg:
            LOGGER.warning('country or reg ignored: not yet implemented')
            centr_handling = 'full_hazard'
        else:
            centr_handling = 'full_hazard'

        # read data and call preprocessing routine:
        self.data, centroids_import = _data_preprocessing_percentile(
            percentile, yearrange, yearrange_ref, input_dir, gh_model, cl_model,
            scenario, scenario_ref, soc, soc_ref, fn_str_var, bbox, min_days_per_month,
            keep_dis_data, yearchunks, mask_threshold)

        if centr_handling == 'full_hazard':
            centroids = centroids_import
        self.identify_clusters()

        # sum "dis" (days per month below threshold) per pixel and cluster_id
        # and write to hazard.intensiy
        self.events_from_clusters(centroids)

        if min_intensity > 1 or min_number_cells > 1:
            haz_tmp = self.filter_events(min_intensity=min_intensity,
                                         min_number_cells=min_number_cells)
            LOGGER.info('Filtering events: %i events remaining', haz_tmp.size)
            self.event_id = haz_tmp.event_id
            self.event_name = list(map(str, self.event_id))
            self.date = haz_tmp.date
            self.date_start = haz_tmp.date_start
            self.date_end = haz_tmp.date_end
            self.orig = haz_tmp.orig
            self.frequency = haz_tmp.frequency
            self.intensity = haz_tmp.intensity
            self.fraction = haz_tmp.fraction
            del haz_tmp
        if not keep_dis_data:
            self.data = None
        self.set_frequency(yearrange=yearrange)
        self.tag = TagHazard(haz_type=HAZ_TYPE, file_name=\
                            FILENAME_NC % (gh_model, cl_model, "*", scenario, soc, \
                                            fn_str_var, "*_*.nc"), \
                             description='yearrange: %i-%i (%s, %s), reference: %i-%i (%s, %s)' \
                                 %(yearrange[0], yearrange[-1], scenario, soc, \
                                   yearrange_ref[0], yearrange_ref[-1], \
                                   scenario_ref, soc_ref)
                             )

    def _intensity_loop(self, uni_ev, coord, res_centr, num_centr):
        """Compute and intensity matrix. For each event, if more than one points of
        data have the same coordinates, take the sum of days below threshold
        of these points (duration as accumulated intensity).

        Parameters:
            uni_ev (list): list of unique cluster IDs
            coord (list): Coordinates as in Centroids.coord
            res_centr (float): Geographical resolution of centroids
            num_centroids (int): Number of centroids

        Returns:
            intensity_mat (sparse.lilmatrix): intensity values as sparse matrix
        """
        tree_centr = BallTree(coord, metric='chebyshev')
        if self.pool:
            chunksize = min(uni_ev.size // self.pool.ncpus, 1000)
            intensity_list = self.pool.map(self._intensity_one_cluster_pool,
                                           itertools.repeat(self.data, uni_ev.size),
                                           itertools.repeat(tree_centr, uni_ev.size),
                                           uni_ev, itertools.repeat(res_centr),
                                           itertools.repeat(num_centr),
                                           chunksize=chunksize)
        else:
            intensity_list = []
            for cl_id in uni_ev:
                intensity_list.append(
                    self._intensity_one_cluster(tree_centr, cl_id,
                                                res_centr, num_centr))
        stps = list(np.arange(0, len(intensity_list)-1, INTENSITY_STEP)) + [len(intensity_list)]
        if len(stps) == 1:
            return sparse.lil_matrix(intensity_list)
        for idx, stp in enumerate(stps[0:-1]):
            if not idx:
                intensity_mat = sparse.lil_matrix(intensity_list[0:stps[1]])
            else:
                intensity_mat = sparse.vstack((intensity_mat,
                                               sparse.lil_matrix(intensity_list[stp:stps[idx+1]])))
        return intensity_mat

    def _set_dates(self, uni_ev):
        """Set dates of maximum intensity (date) as well as start and end dates
        per event

        Parameters:
            uni_ev (list): list of unique cluster IDs
        """
        self.date = np.zeros(uni_ev.size, int)
        self.date_start = np.zeros(uni_ev.size, int)
        self.date_end = np.zeros(uni_ev.size, int)
        for ev_idx, ev_id in enumerate(uni_ev):
            # set event date to date of maximum intensity (ndays)
            self.date[ev_idx] = self.data[self.data.cluster_id == ev_id]\
                .groupby('dtime')['ndays'].sum().idxmax()
            self.date_start[ev_idx] = self.data[self.data.cluster_id == ev_id].dtime.min()
            self.date_end[ev_idx] = self.data[self.data.cluster_id == ev_id].dtime.max()

    def events_from_clusters(self, centroids):
        """Initiate hazard events from connected clusters found in self.data

        Parameters:
            centroids (Centroids)"""
        # intensity = list()

        uni_ev = np.unique(self.data['cluster_id'].values)
        num_centr = centroids.size
        res_centr = self._centroids_resolution(centroids)

        self.tag = TagHazard(HAZ_TYPE)
        self.units = 'days'  # days below threshold
        self.centroids = centroids

        # Following values are defined for each event
        self.event_id = np.sort(self.data.cluster_id.unique())
        self.event_id = self.event_id[self.event_id > 0]
        self.event_name = list(map(str, self.event_id))

        self._set_dates(uni_ev)

        self.orig = np.ones(uni_ev.size)
        self.set_frequency()

        self.intensity = self._intensity_loop(uni_ev, centroids.coord, res_centr, num_centr)

        # Following values are defined for each event and centroid
        self.intensity = self.intensity.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    def identify_clusters(self, clus_thres_xy=None, clus_thresh_t=None, min_samples=None):
        """call clustering functions and set events in hazard

        Optional parameters:
            clus_thres_xy (int): new value of maximum grid cell distance
                (number of grid cells) to be counted as connected points during clustering
            clus_thresh_t (int): new value of maximum timse step difference (months)
                to be counted as connected points during clustering
            min_samples (int): new value or minimum amount of data points in one
                cluster to retain the cluster as an event, smaller clusters will be ignored
        Returns
            pandas.DataFrame
        """
        if min_samples:
            self.min_samples = min_samples
        if clus_thres_xy:
            self.clus_thres_xy = clus_thres_xy
        if clus_thresh_t:
            self.clus_thresh_t = clus_thresh_t

        self.data['cluster_id'] = np.zeros(len(self.data), dtype=int)
        LOGGER.debug('Computing 3D clusters.')
        # Compute clus_id: cluster identifier inside cons_id
        for cluster_vars in [('lat', 'lon'), ('lat', 'dt_month'), ('lon', 'dt_month')]:
            self.data = self._df_clustering(self.data, cluster_vars,
                                            self.resolution, self.clus_thres_xy,
                                            self.clus_thresh_t, self.min_samples)

        self.data = unique_clusters(self.data)
        return self.data

    @staticmethod
    def _df_clustering(data, cluster_vars, res_data, clus_thres_xy,
                       clus_thres_t, min_samples):
        """Compute 2D clusters and sort data with ascending clus_id
        for each combination of the 3 dimensions (lat, lon, dt_month).

        Parameters:
            data (dataframe): dataset obtained from ISIMIP  data
            cluster_vars (tuple): pair of dimensions for 2D clustering,
                e.g. ('lat', 'dt_month')
            res_data (float): input data grid resolution in degrees
            clus_thres_xy (int): clustering distance threshold in space
            clus_thresh_t (int): clustering distance threshold in time
            min_samples (int): clustering min. number

        Returns:
            pandas.DataFrame
        """
        # set iter_var (dimension not used for clustering)
        if 'lat' not in cluster_vars:
            iter_var = 'lat'
        elif 'lon' not in cluster_vars:
            iter_var = 'lon'
        else:
            iter_var = 'dt_month'

        clus_id_var = 'c_%s_%s' % (cluster_vars[0], cluster_vars[1])
        data[clus_id_var] = np.zeros(len(data), dtype=int) - 1

        data_iter = data[data['iter_ev']][[iter_var, cluster_vars[0], cluster_vars[1],
                                           'cons_id', clus_id_var]]

        if 'dt_month' in clus_id_var:
            # transform month count in accordance with spatial resolution
            # to achieve same distance between consecutive and geographically
            # neighboring points:
            data_iter.dt_month = data_iter.dt_month * res_data * clus_thres_xy / clus_thres_t

        # Loop over slices: For each slice, perform 2D clustering with DBSCAN
        for i_var in data_iter[iter_var].unique():
            temp = np.argwhere(np.array(data_iter[iter_var] == i_var)).reshape(-1, )  # slice
            x_y = data_iter.iloc[temp][[cluster_vars[0], cluster_vars[1]]].values
            x_y_uni, x_y_cpy = np.unique(x_y, return_inverse=True, axis=0)
            cluster_id = DBSCAN(eps=res_data * clus_thres_xy, min_samples=min_samples). \
                fit(x_y_uni).labels_
            cluster_id = cluster_id[x_y_cpy]
            data_iter[clus_id_var].values[temp] = cluster_id + data_iter[clus_id_var].max() + 1

        data[clus_id_var].values[data['iter_ev'].values] = data_iter[clus_id_var].values
        return data

    def filter_events(self, min_intensity=1, min_number_cells=1):
        """Remove events with max intensity below min_intensity or spatial extend
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
        """Return resolution of the centroids in their units

        Parameters:
            centroids (Centroids): centroids instance

        Returns:
            float
        """
        if centroids.meta:
            res_centr = abs(centroids.meta['transform'][4]), \
                        centroids.meta['transform'][0]
        else:
            res_centr = np.abs(get_resolution(centroids.lat, centroids.lon))
        if np.abs(res_centr[0] - res_centr[1]) > 1.0e-6:
            LOGGER.warning('Centroids do not represent regular pixels %s.', str(res_centr))
            return (res_centr[0] + res_centr[1]) / 2
        return res_centr[0]

    def _intensity_one_cluster(self, tree_centr, cluster_id, res_centr, num_centr):
        """For a given cluster, fill in an intensity np.array with the summed intensity
        at each centroid.

        Parameters:
            cluster_id (int): id of the selected cluster
            res_centr (float): resolution of centroids in degree
            num_centr (int): number of centroids

        Returns:
            intensity_cl (np.array): summed intensity of cluster at each centroids

        """
        LOGGER.debug('Number of days below threshold corresponding to event %s.', str(cluster_id))
        temp_data = self.data.reindex(
            index=np.argwhere(np.array(self.data['cluster_id'] == cluster_id)).reshape(-1),
            columns=['lat', 'lon', 'ndays'])
        # Identifies the unique (lat,lon) points of the data dataframe -> lat_lon_uni
        # Set the same index value for each duplicate (lat,lon) points -> lat_lon_cpy
        lat_lon_uni, lat_lon_cpy = np.unique(temp_data[['lat', 'lon']].values,
                                             return_inverse=True, axis=0)
        index_uni = np.unique(lat_lon_cpy, axis=0)
        # Search closest centroid for each point
        ind, _ = tree_centr.query_radius(lat_lon_uni, r=res_centr / 2, count_only=False,
                                         return_distance=True, sort_results=True)
        ind = np.array([ind_i[0] if ind_i.size else -1 for ind_i in ind])
        intensity_cl = _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy,
                                       temp_data['ndays'].values)
        return intensity_cl

    @staticmethod
    def _intensity_one_cluster_pool(data, tree_centr, cluster_id, res_centr, num_centr):
        """For a given cluster, fill in an intensity np.array with the summed intensity
        at each centroid. Version for self.pool = True

        Parameters:
            data (DataFrame)
            cluster_id (int): id of the selected cluster
            res_centr (float): resolution of centroids in degree
            num_centr (int): number of centroids

        Returns:
            intensity_cl (np.array): summed intensity of cluster at each centroids

        """
        LOGGER.debug('Number of days below threshold corresponding to event %s.', str(cluster_id))
        temp_data = data.reindex(
            index=np.argwhere(np.array(data['cluster_id'] == cluster_id)).reshape(-1),
            columns=['lat', 'lon', 'ndays'])

        lat_lon_uni, lat_lon_cpy = np.unique(temp_data[['lat', 'lon']].values,
                                             return_inverse=True, axis=0)
        index_uni = np.unique(lat_lon_cpy, axis=0)
        ind, _ = tree_centr.query_radius(lat_lon_uni, r=res_centr / 2, count_only=False,
                                         return_distance=True, sort_results=True)
        ind = np.array([ind_i[0] if ind_i.size else -1 for ind_i in ind])
        intensity_cl = _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy,
                                       temp_data['ndays'].values)
        return intensity_cl

def _init_centroids(data_x, centr_res_factor=1):
    """Get centroids from the firms dataset and refactor them.

    Parameters:
        data_x (xarray): dataset obtained from ISIMIP netcdf
        centr_res_factor (float): the factor applied to voluntarly decrease/increase
            the centroids resolution

    Returns:
        centroids (Centroids)
    """
    res_data = np.min(np.abs([np.diff(data_x.lon.values).min(), np.diff(data_x.lat.values).min()]))
    centroids = Centroids()
    centroids.set_raster_from_pnt_bounds((data_x.lon.values.min(),
                                          data_x.lat.values.min(),
                                          data_x.lon.values.max(),
                                          data_x.lat.values.max()),
                                         res=res_data / centr_res_factor)
    centroids.set_meta_to_lat_lon()
    centroids.set_area_approx()
    centroids.set_on_land()
    centroids.empty_geometry_points()
    return centroids


def unique_clusters(data):
    """identify unqiue clustes based on clusters in 3 dimensions and set unique
    cluster_id

    Parameters:
        data (pandas.DataFrame): contains monthly gridded data of days below threshold

    Returns:
        pandas.DataFrame
    """
    data.cluster_id = np.zeros(len(data.c_lat_lon)) - 1

    data.loc[data.c_lat_lon == -1, 'c_lat_lon'] = np.nan
    data.loc[data.c_lat_dt_month == -1, 'c_lat_dt_month'] = np.nan
    data.loc[data.c_lon_dt_month == -1, 'c_lon_dt_month'] = np.nan

    idc = 0  # event id counter
    current = 0
    for c_lat_lon in data.c_lat_lon.unique():
        if np.isnan(c_lat_lon):
            data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'] = -1
        else:
            if len(data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'].unique()) == 1 \
                    and -1 in data.loc[data.c_lat_lon == c_lat_lon, 'cluster_id'].unique():
                idc += 1
                current = idc
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

def _data_preprocessing_percentile(percentile, yearrange, yearrange_ref,
                                   input_dir, gh_model, cl_model, scenario,
                                   scenario_ref, soc, soc_ref, fn_str_var, bbox,
                                   min_days_per_month, keep_dis_data, yearchunks,
                                   mask_threshold):
    """load data and reference data and calculate monthly percentiles
    then extract intensity based on days below threshold
    returns geopandas dataframe

    Parameters:
        c.f. parameters in LowFlow.set_from_nc()

    Returns:
        data (pandas.DataFrame) preprocessed data with days below threshold
            per grid cell and month
        centroids (Centroids): regular grid centroid with same resolution as input data
    """

    threshold_grid, mean_ref = _compute_threshold_grid(percentile, yearrange_ref,
                                                       input_dir, gh_model, cl_model,
                                                       scenario_ref, soc_ref,
                                                       fn_str_var, bbox,
                                                       yearchunks,
                                                       mask_threshold=mask_threshold,
                                                       keep_dis_data=keep_dis_data)
    first_file = True
    if yearchunks == 'default':
        yearchunks = YEARCHUNKS[scenario]
    # loop over yearchunks
    # (for memory reasons: only loading one file with daily data per step,
    # combining data after conversion to monthly data )
    for yearchunk in yearchunks:
        # skip if file is not required, i.e. not in yearrange:
        if int(yearchunk[0:4]) <= yearrange[1] and int(yearchunk[-4:]) >= yearrange[0]:
            data_chunk = _read_and_combine_nc(
                (max(yearrange[0], int(yearchunk[0:4])),
                 min(yearrange[-1], int(yearchunk[-4:]))),
                input_dir, gh_model, cl_model,
                scenario, soc, fn_str_var, bbox, [yearchunk])
            data_chunk = _days_below_threshold_per_month(data_chunk, threshold_grid, mean_ref,
                                                         min_days_per_month, keep_dis_data)
            if first_file:
                centroids = _init_centroids(data_chunk, centr_res_factor=1)
                dataf = _xarray_to_geopandas(data_chunk)
                first_file = False
            else:
                dataf = dataf.append(_xarray_to_geopandas(data_chunk))
    del data_chunk
    dataf = dataf.sort_values(['lat', 'lon', 'dtime'], ascending=[True, True, True])
    return dataf.reset_index(drop=True), centroids

def _read_and_combine_nc(yearrange, input_dir, gh_model, cl_model, scenario,
                         soc, fn_str_var, bbox, yearchunks, fname_nc=FILENAME_NC):
    """Import and combine data from nc files

    Parameters:
        c.f. parameters in LowFlow.set_from_nc()

    Returns:
        xarray
    """
    first_file = True
    if yearchunks == 'default':
        yearchunks = YEARCHUNKS[scenario]
    for yearchunk in yearchunks:
        # skip if file is not required, i.e. not in yearrange:
        if int(yearchunk[0:4]) > yearrange[1] or int(yearchunk[-4:]) < yearrange[0]:
            continue
        if scenario == 'hist':
            bias_correction = 'nobc'
        else:
            bias_correction = 'ewembi'

        filename = os.path.join(input_dir, fname_nc % (gh_model, cl_model,
                                                       bias_correction, scenario,
                                                       soc, fn_str_var, yearchunk))
        if not os.path.isfile(filename):
            LOGGER.error('Netcdf file not found: %s', filename)
        if first_file:
            data = _read_single_nc(filename, yearrange, bbox)
            first_file = False
        else:
            data = data.combine_first(_read_single_nc(filename, yearrange, bbox))

    # set negative discharge values to zero (debugging of input data):
    data.dis.values[data.dis.values < 0] = 0
    return data

def _read_single_nc(filename, yearrange, bbox):
    """Import data from single nc file, return as xarray

    Parameters:
        filename (str or Path): full path of input netcdf file
        yearrange: (tuple): year range to be extracted from file
        bbox (list): geographical bounding box in the form:
            [lon_min, lat_min, lon_max, lat_max]

    Returns:
        xarray
    """
    data = xr.open_dataset(filename)
    try:
        if not bbox:
            return data.sel(time=slice(dt.datetime(yearrange[0], 1, 1),
                                       dt.datetime(yearrange[-1], 12, 31)))
        return data.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]),
                        time=slice(dt.datetime(yearrange[0], 1, 1),
                                   dt.datetime(yearrange[-1], 12, 31)))
    except TypeError:
        # fix date format if not datetime
        if not bbox:
            data = data.sel(time=slice(cftime.DatetimeNoLeap(yearrange[0], 1, 1),
                                       cftime.DatetimeNoLeap(yearrange[-1], 12, 31)))
        else:
            data = data.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]),
                            time=slice(cftime.DatetimeNoLeap(yearrange[0], 1, 1),
                                       cftime.DatetimeNoLeap(yearrange[-1], 12, 31)))
        datetimeindex = data.indexes['time'].to_datetimeindex()
        data['time'] = datetimeindex
    return data


def _xarray_reduce(data, fun=None, percentile=None):
    """reduce xarray

    Parameters:
        data (xarray)

    Optional Parameters:
        fun (str): function to be applied, either "mean" or "percentile"
        percentile (num): percentile to be extracted, e.g. 5 for 5th percentile
            (only if fun=='percentile')

    Returns:
        xarray
    """
    if fun == 'mean':
        return data.mean(dim='time')
    if fun[0] == 'p':
        return data.reduce(np.nanpercentile, dim='time', q=percentile)
    return None

def _split_bbox(bbox, width=BBOX_WIDTH):
    """split bounding box into squares, return new set of bounding boxes

    Parameters:
        bbox (list): geographical bounding box in the form:
            [lon_min, lat_min, lon_max, lat_max]

    Optional Parameters:
        width (float): width and height of geographical bounding boxes for loop in degree lat/lon.
        i.e. the bounding box is split into square boxes with maximum size BBOX_WIDTH*BBOX_WIDTH

    Returns:
        bbox_list (list): list of bounding boxes of the same format as bbox
    """
    if not bbox:
        bbox = [-180, -85, 180, 85]
    lons = [bbox[0]] + \
        [int(idc) for idc in np.arange(np.ceil(bbox[0]+width-1),
                                       np.floor(bbox[2]-width+1), width)] + [bbox[2]]
    lats = [bbox[1]] + \
        [int(idc) for idc in np.arange(np.ceil(bbox[1]+width-1),
                                       np.floor(bbox[3]-width+1), width)] + [bbox[3]]
    bbox_list = list()
    for ilon, _ in enumerate(lons[:-1]):
        for ilat, _ in enumerate(lats[:-1]):
            bbox_list.append([lons[ilon], lats[ilat], lons[ilon+1], lats[ilat+1]])
    return bbox_list

def _compute_threshold_grid(percentile, yearrange_ref, input_dir, gh_model, cl_model,
                            scenario, soc, fn_str_var, bbox, yearchunks,
                            mask_threshold=None, keep_dis_data=False):
    """given model run and year range specification, this function
    returns the x-th percentile for every pixel over a given
    time horizon (based on daily data) [all-year round percentiles!],
    as well as the mean at each grid cell.

    Parameters:
        c.f. parameters in LowFlow.set_from_nc()

    Optional parameters:
        mask_threshold (tuple or list), Threshold(s) of below which the
            grid is masked out. e.g. ('mean', 1.)

    Returns:
        p_grid (xarray): grid with dis of given percentile (1-timestep)
        mean_grid (xarray): grid with mean(dis)
        """
    LOGGER.info('Computing threshold value per grid cell for Q%i, %i-%i',
                percentile, yearrange_ref[0], yearrange_ref[1])
    if isinstance(mask_threshold, tuple):
        mask_threshold = [mask_threshold]
    bbox = _split_bbox(bbox)
    p_grid = []
    mean_grid = []
    # loop over coordinate bounding boxes to save memory:
    for box in bbox:
        data = _read_and_combine_nc(yearrange_ref, input_dir, gh_model, cl_model,
                                    scenario, soc, fn_str_var, box, yearchunks)
        if data.dis.data.size: # only if data is not empty
            p_grid += [_xarray_reduce(data, fun='p', percentile=percentile)]
            # only compute mean_grid if required by user or mask_threshold:
            if keep_dis_data or (mask_threshold and True in ['mean' in x for x in mask_threshold]):
                mean_grid += [_xarray_reduce(data, fun='mean')]

    del data
    p_grid = xr.combine_by_coords(p_grid)
    if mean_grid:
        mean_grid = xr.combine_by_coords(mean_grid)

    if isinstance(mask_threshold, list):
        for crit in mask_threshold:
            if 'mean' in crit[0]:
                p_grid.dis.values[mean_grid.dis.values < crit[1]] = 0
                mean_grid.dis.values[mean_grid.dis.values < crit[1]] = 0
            if 'percentile' in crit[0]:
                p_grid.dis.values[p_grid.dis.values < crit[1]] = 0
                mean_grid.dis.values[p_grid.dis.values < crit[1]] = 0
    if keep_dis_data:
        return p_grid, mean_grid
    return p_grid, None

def _days_below_threshold_per_month(data, threshold_grid, mean_ref,
                                    min_days_per_month, keep_dis_data):
    """returns sum of days below threshold per month (as xarray with monthly data)

    if keep_dis_data is True, a DataFrame called 'data' with additional data is saved within
    the hazard object. It provides data per event, grid cell, and month
    data comes with the following columns: ['lat', 'lon', 'time', 'ndays',
       'relative_dis', 'iter_ev', 'cons_id',
       'dtime', 'dt_month', 'geometry', 'cluster_id', 'c_lat_lon',
       'c_lat_dt_month', 'c_lon_dt_month']
    Note: cluster_id corresponds 1:1 with associated event_id.

    Parameters:
        c.f. parameters in LowFlow.set_from_nc()

    Returns:
        xarray

    """
    # data = data.groupby('time.month')-threshold_grid # outdated
    data_threshold = data - threshold_grid
    if keep_dis_data:
        data_low = data.where(data_threshold < 0) / mean_ref
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
    dataf = data.to_dataframe()
    dataf.reset_index(inplace=True)
    dataf = dataf.dropna()
    dataf['iter_ev'] = np.ones(len(dataf), bool)
    dataf['cons_id'] = np.zeros(len(dataf), int) - 1
    # dataf['cluster_id'] = np.zeros(len(dataf), int)
    # dataf['clus_id'] = np.zeros(len(dataf), int) - 1
    dataf['dtime'] = dataf['time'].apply(lambda x: x.toordinal())
    dataf['dt_month'] = dataf['time'].apply(lambda x: x.year * 12 + x.month)
    return gpd.GeoDataFrame(dataf, geometry=[Point(x, y) for x, y in zip(dataf['lon'],
                                                                         dataf['lat'])])

@numba.njit
def _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy, intensity_raw):
    """fill intensity list for a single cluster

    Parameters:
        num_centr (int): total number of centroids
        ind (list): list of centroid indices  in cluster
        lat_lon_cpy (array of int): index according to (lat, lon) in intensity_raw
        intensity_raw (array of int): array of ndays values at each data point in cluster

    Returns:
        list with summed ndays (=intensity) per geographical point (lat, lon)
    """

    intensity_cl = np.zeros((1, num_centr), dtype=numba.float64)
    for idx in range(index_uni.size):
        if ind[idx] != -1:
            intensity_cl[0, ind[idx]] = \
                np.sum(intensity_raw[lat_lon_cpy == index_uni[idx]])
    return intensity_cl[0]
