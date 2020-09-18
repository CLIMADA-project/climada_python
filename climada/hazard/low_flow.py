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

FN_STR_VAR = 'co2_dis_global_daily'  # FileName STRing depending on VARiable
"""constant part of discharge output file (according to ISIMIP filenaming)"""

YEARCHUNKS = dict()
"""list of year chunks: multiple files are combined"""

YEARCHUNKS['historical'] = list()
"""historical year chunks ISIMIP 2b"""
for i in np.arange(1860, 2000, 10):
    YEARCHUNKS['historical'].append(f'{i+1}_{i+10}')
YEARCHUNKS['historical'].append('2001_2005')

YEARCHUNKS['hist'] = list()
"""historical year chunks ISIMIP 2a"""
for i in np.arange(1970, 2010, 10):
    YEARCHUNKS['hist'].append(f'{i+1}_{i+10}')

YEARCHUNKS['rcp26'] = ['2006_2010']
for i in np.arange(2010, 2090, 10):
    YEARCHUNKS['rcp26'].append(f'{i+1}_{i+10}')
YEARCHUNKS['rcp26'].append('2091_2099')
YEARCHUNKS['rcp60'] = YEARCHUNKS['rcp26']
"""future year chunks"""

REFERENCE_YEARRANGE = (1971, 2005)
"""default year range used to compute threshold (base line reference)"""

TARGET_YEARRANGE = (2001, 2005)
"""arbitrary default, i.e. default year range of historical low flow hazard 2001-2005"""

BBOX = (-180, -85, 180, 85)
"""default quasi-global geographical bounding box: [lon_min, lat_min, lon_max, lat_max]"""

# reducing these two parameters decreases memory load but increases computation time:
BBOX_WIDTH = 75
"""default width and height of geographical bounding boxes for loop in degree lat/lon.
i.e., the bounding box is split into square boxes with maximum size BBOX_WIDTH*BBOX_WIDTH
(avoid memory usage spike)"""
INTENSITY_STEP = 300
"""max. number of events to be written to hazard.intensity matrix at once
(avoid memory usage spike)"""

class LowFlow(Hazard):
    """Contains river low flow events (surface water scarcity).
    The intensity of the hazard is number of days below a threshold (defined as
    percentile in reference data). The method set_from_nc can be used to create
    a LowFlow hazard set populated with data based on gridded hydrological model runs
    as provided by the ISIMIP project (https://www.isimip.org/), e.g. ISIMIP2a/b.
    grid cells with a minimum number of days below threshold per month are clustered
    in space (lat/lon) and time (monthly) to identify and set connected events.

    Attributes:
        clus_thresh_t (int): maximum time difference in months to be counted as$
            connected points during clustering, default = 1
        clus_thresh_xy (int): maximum spatial grid cell distance in number of cells
            to be counted as connected points during clustering, default = 2
        min_samples (1): Minimum amount of data points in one cluster to consider as event,
            default = 1.
        date_start (np.array(int)): for each event, the date of the first month
            of the event (ordinal)
            Note: Hazard attribute 'date' contains the date of maximum event intensity.
        date_end (np.array(int)): for each event, the date of the last month of
            the event (ordinal)
        resolution (float): spatial resoultion of gridded discharge input data in degree lat/lon,
            default = 0.5°
    """

    clus_thresh_t = 1 # Default = 1: months with intensity<min_intensity interrupt event
    clus_thresh_xy = 2 # Default = 2: allows 1 cell gap and diagonal connection
    min_samples = 1 # Default = 1: no filtering of small events with this default
    resolution = .5 # Default = .5: in agreement with resolution of data from ISIMIP 1-3


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
                    yearrange_ref=REFERENCE_YEARRANGE, gh_model=None, cl_model=None,
                    scenario='historical', scenario_ref='historical', soc='histsoc',
                    soc_ref='histsoc', fn_str_var=FN_STR_VAR, keep_dis_data=False,
                    yearchunks='default', mask_threshold=('mean', 1)):
        """Wrapper to fill hazard from NetCDF file containing variable dis (daily),
        e.g. as provided from from ISIMIP Water Sectior (Global):
            https://esg.pik-potsdam.de/search/isimip/

        Parameters:
            input_dir (string): path to input data directory. In this folder,
                netCDF files with gridded hydrological model output are required,
                containing the variable dis (discharge) on a daily temporal resolution
                as f.i. provided by the ISIMIP project (https://www.isimip.org/)
            centroids (Centroids): centroids
                (area that is considered, reg and country must be None)
            countries (list of countries ISO3) selection of countries
                (reg must be None!) [not yet implemented]
            reg (list of regions): can be set with region code if whole areas
                are considered (if not None, countries and centroids
                are ignored) [not yet implemented]
            bbox (tuple of four floats): bounding box:
                (lon min, lat min, lon max, lat max)
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
                f.i. 'H08', 'CLM45', 'ORCHIDEE', 'LPJmL', 'WaterGAP2', 'JULES-W1', 'MATSIRO'
            cl_model (str): abbrev. climate model (only when input_dir is selected)
                f.i. 'gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr', 'miroc5', 'gswp3',
                'wfdei', 'princeton', 'watch'
            scenario (str): climate change scenario (only when input_dir is selected)
                f.i. 'historical', 'rcp26', 'rcp60', 'hist'
            scenario_ref (str): climate change scenario for reference
                (only when input_dir is selected)
            soc (str): socio-economic trajectory (only when input_dir is selected)
                f.i. 'histsoc',  # historical trajectory
                     '2005soc',  # constant at 2005 level
                     'rcp26soc', # RCP6.0 trajectory
                     'rcp60soc', # RCP6.0 trajectory
                     'pressoc' # constant at pre-industrial socio-economic level
            soc_ref (str): csocio-economic trajectory for reference, like soc.
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
        self.lowflow_df, centroids_import = data_preprocessing_percentile(
            percentile, yearrange, yearrange_ref, input_dir, gh_model, cl_model,
            scenario, scenario_ref, soc, soc_ref, fn_str_var, bbox, min_days_per_month,
            keep_dis_data, yearchunks, mask_threshold)

        if centr_handling == 'full_hazard':
            centroids = centroids_import
        self.identify_clusters()
        self.set_intensity_from_clusters(centroids, min_intensity, min_number_cells,
                                 yearrange, yearrange_ref, gh_model, cl_model,
                                 scenario, scenario_ref, soc, soc_ref, fn_str_var, keep_dis_data)

    def set_intensity_from_clusters(self, centroids=None, min_intensity=1, min_number_cells=1,
                            yearrange=TARGET_YEARRANGE, yearrange_ref=REFERENCE_YEARRANGE,
                            gh_model=None, cl_model=None,
                            scenario='historical', scenario_ref='historical', soc='histsoc',
                            soc_ref='histsoc', fn_str_var=FN_STR_VAR, keep_dis_data=False):
        """ Build low flow hazards with events from clustering and centroids and add attributes.
        """
        # sum "dis" (days per month below threshold) per pixel and
        # cluster_id and write to hazard.intensity
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
            self.lowflow_df = None
        self.set_frequency(yearrange=yearrange)
        self.tag = TagHazard(haz_type=HAZ_TYPE, file_name=\
                             f'{gh_model}_{cl_model}_*_{scenario}_{soc}_{fn_str_var}_*.nc', \
                             description= f'yearrange: {yearrange[0]}-{yearrange[0]} ' +\
                                          f'({scenario}, {soc}), ' +\
                                          f'reference: {yearrange_ref[0]}-{yearrange_ref[0]} ' +\
                                          f'({scenario_ref}, {soc_ref})'
                             )

    def _intensity_loop(self, uniq_ev, coord, res_centr, num_centr):
        """Compute intensity and populate intensity matrix.
        For each event, if more than one points of
        data have the same coordinates, take the sum of days below threshold
        of these points (duration as accumulated intensity).

        Parameters:
            uniq_ev (list of str): list of unique cluster IDs
            coord (list): Coordinates as in Centroids.coord
            res_centr (float): Geographical resolution of centroids
            num_centroids (int): Number of centroids

        Returns:
            intensity_mat (sparse.lilmatrix): intensity values as sparse matrix
        """
        tree_centr = BallTree(coord, metric='chebyshev')
        # steps: list of steps to be written to intensity matrix at once:
        steps = list(np.arange(0, len(uniq_ev) - 1, INTENSITY_STEP)) + [len(uniq_ev)]
        if len(steps) == 1:
            intensity_list = [self._intensity_one_cluster(tree_centr, cl_id, res_centr, num_centr)
                  for cl_id in uniq_ev]
            return sparse.csr_matrix(intensity_list)
        # step_range: list of tuples containing the unique IDs to be written to
        # the intensity matrix in one step
        step_range = [tuple(uniq_ev[stp:steps[idx+1]]) for idx, stp in enumerate(steps[0:-1])]
        for idx, stp in enumerate(step_range):
            intensity_list = []
            for cl_id in stp:
                intensity_list.append(
                 self._intensity_one_cluster(tree_centr, cl_id, res_centr, num_centr))
            if not idx:
                intensity_mat = sparse.lil_matrix(intensity_list)
            else:
                intensity_mat = sparse.vstack((intensity_mat,
                                               sparse.csr_matrix(intensity_list)))
        return intensity_mat

    def _set_dates(self, uniq_ev):
        """Set dates of maximum intensity (date) as well as start and end dates
        per event

        Parameters:
            uniq_ev (list): list of unique cluster IDs
        """
        self.date = np.zeros(uniq_ev.size, int)
        self.date_start = np.zeros(uniq_ev.size, int)
        self.date_end = np.zeros(uniq_ev.size, int)
        for ev_idx, ev_id in enumerate(uniq_ev):
            # set event date to date of maximum intensity (ndays)
            self.date[ev_idx] = self.lowflow_df[self.lowflow_df.cluster_id == ev_id]\
                .groupby('dtime')['ndays'].sum().idxmax()
            self.date_start[ev_idx] = self.lowflow_df[self.lowflow_df.cluster_id == ev_id].dtime.min()
            self.date_end[ev_idx] = self.lowflow_df[self.lowflow_df.cluster_id == ev_id].dtime.max()

    def events_from_clusters(self, centroids):
        """Initiate hazard events from connected clusters found in self.lowflow_df

        Parameters:
            centroids (Centroids)"""
        # intensity = list()

        uniq_ev = np.unique(self.lowflow_df['cluster_id'].values)
        num_centr = centroids.size
        res_centr = self._centroids_resolution(centroids)

        self.tag = TagHazard(HAZ_TYPE)
        self.units = 'days'  # days below threshold
        self.centroids = centroids

        # Following values are defined for each event
        self.event_id = np.sort(uniq_ev)
        self.event_id = self.event_id[self.event_id > 0]
        self.event_name = list(map(str, self.event_id))

        self._set_dates(uniq_ev)

        self.orig = np.ones(uniq_ev.size)
        self.set_frequency()

        self.intensity = self._intensity_loop(uniq_ev, centroids.coord, res_centr, num_centr)

        # Following values are defined for each event and centroid
        self.intensity = self.intensity.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    def identify_clusters(self, clus_thresh_xy=None, clus_thresh_t=None, min_samples=None):
        """call clustering functions to identify the clusters inside the dataframe

        Optional parameters:
            clus_thresh_xy (int): new value of maximum grid cell distance
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
        if clus_thresh_xy:
            self.clus_thresh_xy = clus_thresh_xy
        if clus_thresh_t:
            self.clus_thresh_t = clus_thresh_t

        self.lowflow_df['cluster_id'] = np.zeros(len(self.lowflow_df), dtype=int)
        LOGGER.debug('Computing 3D clusters.')
        # Compute clus_id: cluster identifier inside cons_id
        for cluster_vars in [('lat', 'lon'), ('lat', 'dt_month'), ('lon', 'dt_month')]:
            self.lowflow_df = self._df_clustering(self.lowflow_df, cluster_vars,
                                            self.resolution, self.clus_thresh_xy,
                                            self.clus_thresh_t, self.min_samples)

        self.lowflow_df = unique_clusters(self.lowflow_df)
        return self.lowflow_df

    @staticmethod
    def _df_clustering(lowflow_df, cluster_vars, res_data, clus_thresh_xy,
                       clus_thres_t, min_samples):
        """Compute 2D clusters and sort lowflow_df with ascending clus_id
        for each combination of the 3 dimensions (lat, lon, dt_month).

        Parameters:
            lowflow_df (dataframe): dataset obtained from ISIMIP  data
            cluster_vars (tuple pf str): pair of dimensions for 2D clustering,
                e.g. ('lat', 'dt_month')
            res_data (float): input data grid resolution in degrees
            clus_thresh_xy (int): clustering distance threshold in space
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
        lowflow_df[clus_id_var] = np.zeros(len(lowflow_df), dtype=int) - 1

        data_iter = lowflow_df[lowflow_df['iter_ev']][[iter_var, cluster_vars[0], cluster_vars[1],
                                           'cons_id', clus_id_var]]

        if 'dt_month' in clus_id_var:
            # transform month count in accordance with spatial resolution
            # to achieve same distance between consecutive and geographically
            # neighboring points:
            data_iter.dt_month = data_iter.dt_month * res_data * clus_thresh_xy / clus_thres_t

        # Loop over slices: For each slice, perform 2D clustering with DBSCAN
        for i_var in data_iter[iter_var].unique():
            temp = np.argwhere(np.array(data_iter[iter_var] == i_var)).reshape(-1, )  # slice
            x_y = data_iter.iloc[temp][[cluster_vars[0], cluster_vars[1]]].values
            x_y_uni, x_y_cpy = np.unique(x_y, return_inverse=True, axis=0)
            cluster_id = DBSCAN(eps=res_data * clus_thresh_xy, min_samples=min_samples). \
                fit(x_y_uni).labels_
            cluster_id = cluster_id[x_y_cpy]
            data_iter[clus_id_var].values[temp] = cluster_id + data_iter[clus_id_var].max() + 1

        lowflow_df[clus_id_var].values[lowflow_df['iter_ev'].values] = data_iter[clus_id_var].values
        return lowflow_df

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
        haz_tmp.orig_tmp = copy.deepcopy(self.orig)
        haz_tmp.orig = np.array(np.ones(self.orig.size))
        # identify events to be filtered out and set haz_tmp.orig to 0.
        for i_event, _ in enumerate(haz_tmp.event_id):
            if np.sum(haz_tmp.intensity[i_event] > 0) < min_number_cells or \
                    np.max(haz_tmp.intensity[i_event]) < min_intensity:
                haz_tmp.orig[i_event] = 0.
        haz_tmp = haz_tmp.select(orig=True) # select events with orig == 1
        haz_tmp.orig = haz_tmp.orig_tmp # reset orig
        del haz_tmp.orig_tmp
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
            tree_centr (object) BallTree instance created from centroids' coordinates
            res_centr (float): resolution of centroids in degree
            num_centr (int): number of centroids

        Returns:
            intensity_cl (np.array): summed intensity of cluster at each centroids
        """
        LOGGER.debug('Number of days below threshold corresponding to event %s.', str(cluster_id))
        temp_data = self.lowflow_df.reindex(
            index=np.argwhere(np.array(self.lowflow_df['cluster_id'] == cluster_id)).reshape(-1),
            columns=['lat', 'lon', 'ndays'])
        # Identifies the unique (lat,lon) points of the lowflow_df dataframe -> lat_lon_uni
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
    def _intensity_one_cluster_pool(lowflow_df, tree_centr, cluster_id, res_centr, num_centr):
        """For a given cluster, fill in an intensity np.array with the summed intensity
        at each centroid. Version for self.pool = True

        Parameters:
            lowflow_df (DataFrame)
            tree_centr (object): BallTree instance created from centroids' coordinates
            cluster_id (int): id of the selected cluster
            res_centr (float): resolution of centroids in degree
            num_centr (int): number of centroids

        Returns:
            intensity_cl (np.array): summed intensity of cluster at each centroids

        """
        LOGGER.debug('Number of days below threshold corresponding to event %s.', str(cluster_id))
        temp_data = lowflow_df.reindex(
            index=np.argwhere(np.array(lowflow_df['cluster_id'] == cluster_id)).reshape(-1),
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

def _init_centroids(dis_xarray, centr_res_factor=1):
    """Get centroids from the firms dataset and refactor them.

    Parameters:
        dis_xarray (xarray): dataset obtained from ISIMIP netcdf

    Optional Parameters:
        centr_res_factor (float): the factor applied to voluntarly decrease/increase
            the centroids resolution

    Returns:
        centroids (Centroids)
    """
    res_data = np.min(np.abs([np.diff(dis_xarray.lon.values).min(),
                              np.diff(dis_xarray.lat.values).min()]))
    centroids = Centroids()
    centroids.set_raster_from_pnt_bounds((dis_xarray.lon.values.min(),
                                          dis_xarray.lat.values.min(),
                                          dis_xarray.lon.values.max(),
                                          dis_xarray.lat.values.max()),
                                         res=res_data / centr_res_factor)
    centroids.set_meta_to_lat_lon()
    centroids.set_area_approx()
    centroids.set_on_land()
    centroids.empty_geometry_points()
    return centroids


def unique_clusters(lowflow_df):
    """identify unqiue clustes based on clusters in 3 dimensions and set unique
    cluster_id

    Parameters:
        lowflow_df (pandas.DataFrame): contains monthly gridded data of days below threshold

    Returns:
        lowflow_df (pandas.DataFrame): As input with new values in column cluster_id
    """
    lowflow_df.cluster_id = np.zeros(len(lowflow_df.c_lat_lon)) - 1

    lowflow_df.loc[lowflow_df.c_lat_lon == -1, 'c_lat_lon'] = np.nan
    lowflow_df.loc[lowflow_df.c_lat_dt_month == -1, 'c_lat_dt_month'] = np.nan
    lowflow_df.loc[lowflow_df.c_lon_dt_month == -1, 'c_lon_dt_month'] = np.nan

    idc = 0  # event id counter
    current = 0
    for c_lat_lon in lowflow_df.c_lat_lon.unique():
        if np.isnan(c_lat_lon):
            lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'cluster_id'] = -1
        else:
            if len(lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'cluster_id'].unique()) == 1 \
                    and -1 in lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'cluster_id'].unique():
                idc += 1
                current = idc
            else:
                current = max(lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'cluster_id'].unique())
            lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'cluster_id'] = current

            for c_lat_dt_month in lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'c_lat_dt_month'].unique():
                if not np.isnan(c_lat_dt_month):
                    lowflow_df.loc[lowflow_df.c_lat_dt_month == c_lat_dt_month, 'cluster_id'] = current
            for c_lon_dt_month in lowflow_df.loc[lowflow_df.c_lat_lon == c_lat_lon, 'c_lon_dt_month'].unique():
                if not np.isnan(c_lon_dt_month):
                    lowflow_df.loc[lowflow_df.c_lon_dt_month == c_lon_dt_month, 'cluster_id'] = current

    lowflow_df.loc[np.isnan(lowflow_df.c_lon_dt_month), 'cluster_id'] = -1
    lowflow_df.loc[np.isnan(lowflow_df.c_lat_dt_month), 'cluster_id'] = -1
    lowflow_df.cluster_id = lowflow_df.cluster_id.astype(int)
    return lowflow_df

def data_preprocessing_percentile(percentile, yearrange, yearrange_ref,
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
        lowflow_df (pandas.DataFrame) preprocessed data with days below threshold
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
        # skip if file is not required, i.e., not in yearrange:
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
                         soc, fn_str_var, bbox, yearchunks):
    """Import and combine data from nc files

    Parameters:
        c.f. parameters in LowFlow.set_from_nc()

    Returns:
        dis_xarray (xarray)
    """
    first_file = True
    if yearchunks == 'default':
        yearchunks = YEARCHUNKS[scenario]
    for yearchunk in yearchunks:
        # skip if file is not required, i.e., not in yearrange:
        if int(yearchunk[0:4]) > yearrange[1] or int(yearchunk[-4:]) < yearrange[0]:
            continue
        if scenario == 'hist':
            bias_corr = 'nobc'
        else:
            bias_corr = 'ewembi'

        filename = os.path.join(input_dir, \
                                f'{gh_model}_{cl_model}_{bias_corr}_{scenario}_{soc}_{fn_str_var}_{yearchunk}.nc'
                                )
        if not os.path.isfile(filename):
            LOGGER.error('Netcdf file not found: %s', filename)
        if first_file:
            dis_xarray = _read_single_nc(filename, yearrange, bbox)
            first_file = False
        else:
            dis_xarray = dis_xarray.combine_first(_read_single_nc(filename, yearrange, bbox))

    # set negative discharge values to zero (debugging of input data):
    dis_xarray.dis.values[dis_xarray.dis.values < 0] = 0
    return dis_xarray

def _read_single_nc(filename, yearrange, bbox):
    """Import data from single nc file, return as xarray

    Parameters:
        filename (str or pathlib.Path): full path of input netcdf file
        yearrange: (tuple): year range to be extracted from file
        bbox (tuple of float): geographical bounding box in the form:
            (lon_min, lat_min, lon_max, lat_max)

    Returns:
        dis_xarray (xarray)
    """
    dis_xarray = xr.open_dataset(filename)
    try:
        if not bbox:
            return dis_xarray.sel(time=slice(dt.datetime(yearrange[0], 1, 1),
                                       dt.datetime(yearrange[-1], 12, 31)))
        lon_min, lat_min, lon_max, lat_max = bbox
        return dis_xarray.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min),
                        time=slice(dt.datetime(yearrange[0], 1, 1),
                                   dt.datetime(yearrange[-1], 12, 31)))
    except TypeError:
        # fix date format if not datetime
        if not bbox:
            dis_xarray = dis_xarray.sel(time=slice(cftime.DatetimeNoLeap(yearrange[0], 1, 1),
                                       cftime.DatetimeNoLeap(yearrange[-1], 12, 31)))
        else:
            lon_min, lat_min, lon_max, lat_max = bbox
            dis_xarray = dis_xarray.sel(lon=slice(lon_min, lon_max), lat=slice(lat_max, lat_min),
                            time=slice(cftime.DatetimeNoLeap(yearrange[0], 1, 1),
                                       cftime.DatetimeNoLeap(yearrange[-1], 12, 31)))
        datetimeindex = dis_xarray.indexes['time'].to_datetimeindex()
        dis_xarray['time'] = datetimeindex
    return dis_xarray


def _xarray_reduce(dis_xarray, fun=None, percentile=None):
    """wrapper function to reduce xarray along time axis

    Parameters:
        dis_xarray (xarray)

    Optional Parameters:
        fun (str): function to be applied, either "mean" or "percentile"
        percentile (num): percentile to be extracted, e.g. 5 for 5th percentile
            (only if fun=='percentile')

    Returns:
        xarray
    """
    if fun == 'mean':
        return dis_xarray.mean(dim='time')
    if fun[0] == 'p':
        return dis_xarray.reduce(np.nanpercentile, dim='time', q=percentile)
    return None

def _split_bbox(bbox, width=BBOX_WIDTH):
    """split bounding box into squares, return new set of bounding boxes
    Note: Could this function be a candidate for climada.util in the future?
    
    Parameters:
        bbox (tuple of float): geographical bounding box in the form:
            (lon_min, lat_min, lon_max, lat_max)

    Optional Parameters:
        width (float): width and height of geographical bounding boxes for loop in degree lat/lon.
        i.e., the bounding box is split into square boxes with maximum size BBOX_WIDTH*BBOX_WIDTH

    Returns:
        bbox_list (list): list of bounding boxes of the same format as bbox
    """
    if not bbox:
        lon_min, lat_min, lon_max, lat_max = (-180, -85, 180, 85)
    else:
        lon_min, lat_min, lon_max, lat_max = bbox
    lons = [lon_min] + \
        [int(idc) for idc in np.arange(np.ceil(lon_min+width-1),
                                       np.floor(lon_max-width+1), width)] + [lon_max]
    lats = [lat_min] + \
        [int(idc) for idc in np.arange(np.ceil(lat_min+width-1),
                                       np.floor(lat_max-width+1), width)] + [lat_max]
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
        dis_xarray = _read_and_combine_nc(yearrange_ref, input_dir, gh_model, cl_model,
                                    scenario, soc, fn_str_var, box, yearchunks)
        if dis_xarray.dis.data.size: # only if data is not empty
            p_grid += [_xarray_reduce(dis_xarray, fun='p', percentile=percentile)]
            # only compute mean_grid if required by user or mask_threshold:
            if keep_dis_data or (mask_threshold and True in ['mean' in x for x in mask_threshold]):
                mean_grid += [_xarray_reduce(dis_xarray, fun='mean')]

    del dis_xarray
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

def _days_below_threshold_per_month(dis_xarray, threshold_grid, mean_ref,
                                    min_days_per_month, keep_dis_data):
    """returns sum of days below threshold per month (as xarray with monthly data)

    if keep_dis_data is True, a DataFrame called 'lowflow_df' with additional data
    is saved within the hazard object.
    It provides data per event, grid cell, and month
    lowflow_df comes with the following columns: ['lat', 'lon', 'time', 'ndays',
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
    data_threshold = dis_xarray - threshold_grid
    if keep_dis_data:
        data_low = dis_xarray.where(data_threshold < 0) / mean_ref
        data_low = data_low.resample(time='1M').mean()
    data_threshold.dis.values[data_threshold.dis.values >= 0] = 0
    data_threshold.dis.values[data_threshold.dis.values < 0] = 1
    data_threshold = data_threshold.resample(time='1M').sum()
    data_threshold.dis.values[data_threshold.dis.values < min_days_per_month] = 0
    data_threshold = data_threshold.rename({'dis': 'ndays'})
    if keep_dis_data:
        data_threshold['relative_dis'] = data_low['dis']
    return data_threshold.where(data_threshold['ndays'] > 0)

def _xarray_to_geopandas(dis_xarray):
    """create GeoDataFrame from xarray with NaN values dropped
    Note: Could this function be a candidate for climada.util in the future?

    Parameters:
        dis_xarray (xarray): data as xarray object

    Returns:
        lowflow_df (GeoDataFrame)."""

    dataf = dis_xarray.to_dataframe()
    dataf.reset_index(inplace=True)
    dataf = dataf.dropna()
    dataf['iter_ev'] = np.ones(len(dataf), bool)
    dataf['cons_id'] = np.zeros(len(dataf), int) - 1
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
