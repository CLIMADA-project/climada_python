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

Define BushFire class.
"""

__all__ = ['BushFire']

import logging
import warnings
import itertools
from datetime import date
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
import numba
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from climada.hazard.centroids.centr import Centroids
from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.util.constants import ONE_LAT_KM, DEF_CRS
from climada.util.coordinates import get_resolution
from climada.util.dates_times import str_to_date

from climada.util.alpha_shape import alpha_shape, plot_polygon
warnings.simplefilter(action='ignore', category=FutureWarning)

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'BF'
""" Hazard type acronym for Bush Fire """

CLEAN_THRESH = 30
""" Minimal confidence value for the data from MODIS instrument to be use as input"""

RES_DATA = 1.0
""" Resolution of the data if no data origin provided (km) """

WIGGLE = 9
""" Maximum number of cells where fire can be ignited """

PROP_PROPA = 0.25
"""Probability of fire propagation """

class BushFire(Hazard):
    """Contains bush fire events.

    Attributes:
        date_end (np.array(int)): ordinal date of last occurence of event
    """

    days_thres = 2
    """ Minimum number of days to consider different events """

    clus_thres = 15
    """ Clustering factor which multiplies instrument resolution """

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_hist_events(self, csv_firms, centr_res_factor=1, centroids=None):
        """ Parse FIRMS data and generate historical events by temporal and spatial
        clustering.

        Parameters:
            csv_firms: csv file of the FIRMS data (https://firms.modaps.eosdis.nasa.gov/download/)
            centr_res_factor (int, optional): resolution factor with respect to
                the satellite data to use for centroids creation. Default: 1
            centroids (Centroids, optional): centroids in degrees to map data
        """
        LOGGER.info('Setting up historical events.')
        self.clear()

        # read and initialize data
        firms = self._clean_firms_csv(csv_firms)
        # compute centroids
        res_data = self._firms_resolution(firms)
        if not centroids:
            centroids = self._centroids_creation(firms, res_data, centr_res_factor)
        else:
            if not centroids.coord.size:
                centroids.set_meta_to_lat_lon()
        res_centr = self._centroids_resolution(centroids)

        # event identification
        while firms.iter_ev.any():
            # Compute cons_id: consecutive events in current iteration
            self._firms_cons_days(firms)
            # Compute clus_id: cluster identifier inside cons_id
            self._firms_clustering(firms, res_data, self.clus_thres)
            # compute event_id
            self._firms_event(self.days_thres, firms.cons_id.values,
                              firms.clus_id.values, firms.event_id.values,
                              firms.iter_ev.values, firms.datenum.values)
            LOGGER.info('Remaining events to identify: %s.', str(np.argwhere(\
            firms.iter_ev).size))

        # compute brightness and fill class attributes
        LOGGER.info('Computing intensity of %s events.', np.unique(firms.event_id).size)
        self._calc_brightness(firms, centroids, res_centr)

    def set_proba_events(self, ens_size=9, seed=8):
        """ Generate a set of probabilistic events for each historical event.
        Execute after set_hist_events. Centroids must be in a grid.

        Parameters:
            ens_size (int): number of probabilistic events to generate per
                historical event
            seed (int, optional): random number generator seed. Default: 8.

        Raises:
            ValueError
        """
        LOGGER.info('Setting up probabilistic events.')
        if self.pool:
            chunksize = min(self.size//self.pool.ncpus, 1000)
            haz_syn = self.pool.map(self._set_proba_one_event, range(self.size),
                                    itertools.repeat(ens_size, self.size),
                                    np.arange(seed, seed + self.size),
                                    chunksize=chunksize)
            haz = []
            for haz_i in haz_syn:
                haz.extend(haz_i)
        else:
            haz = []
            for ev_idx in range(self.size):
                haz.extend(self._set_proba_one_event(ev_idx, ens_size,
                                                     seed+ev_idx))

        # problem random num generator in multiprocessing. python 3.7?
        LOGGER.debug('Append events.')
        prob_haz = BushFire()
        prob_haz._append_all(haz)
        self.append(prob_haz)

        # Update the frequency adter adding the probabilistic events
        self._set_frequency()

    def hull_burned_area(self, ev_id, alpha=100.87):
        """Compute the burned area for a given event.

        Parameters:
            ev_id: id of the selected event
            alpha (float, optional): parameter used to compute the concave hull

        Returns:
            float
        """
        ev_idx = np.argwhere(self.event_id == ev_id).reshape(-1)[0]
        if not ev_idx.size:
            LOGGER.error('No event with id %s', str(ev_id))
            raise ValueError

        if not self.centroids.lat.size:
            self.centroids.set_meta_to_lat_lon()

        # Extract coordinates where intensity > 0 (for a given event)
        fire_lat = self.centroids.lat[self.intensity[ev_idx, :].nonzero()[1]]
        fire_lon = self.centroids.lon[self.intensity[ev_idx, :].nonzero()[1]]

        # Creation of the geodataframe
        fire = gpd.GeoDataFrame(crs=DEF_CRS)
        fire['geometry'] = list(zip(fire_lon, fire_lat))
        fire['geometry'] = fire['geometry'].apply(Point)
        points = fire.geometry.values

        # Compute concave hull
        concave_hull, _ = alpha_shape(points, alpha=alpha)

        # Compute area concave hull in right projection
        project = partial(
            pyproj.transform,
            pyproj.Proj(init=DEF_CRS['init']), # source coordinate system
            pyproj.Proj(init='epsg:3310')) # destination coordinate system: albers california

        concave_hull_m = transform(project, concave_hull)  # apply projection
        area_hull_one_event = concave_hull_m.area/10000
        LOGGER.info('area: %s ha', area_hull_one_event)

        # Plot the polygone around the fire
        plot_polygon(concave_hull)
        plt.plot(fire_lon, fire_lat, 'o', color='red', markersize=0.5)
        return area_hull_one_event

    @staticmethod
    def _clean_firms_csv(csv_firms):
        """Read and remove low confidence data from firms:
            - MODIS: remove data where confidence values are lower than CLEAN_THRESH
            - VIIRS: remove data where confidence values are set to low (keep
                nominal and high values)

        Parameters:
            csv_firms: csv file of the FIRMS data

        Returns:
            pd.DataFrame
        """
        firms = pd.read_csv(csv_firms)
        # Check for the type of instrument (MODIS vs VIIRS)
        # Remove data with low confidence interval
        # Uniformize the name of the birghtness columns between VIIRS and MODIS
        temp = pd.DataFrame()
        if 'instrument' in firms.columns:
            if firms.instrument.any() == 'MODIS' or firms.instrument.any() == 'VIIRS':
                firms_modis = firms.drop(firms[firms.instrument == 'VIIRS'].index)
                firms_modis.confidence = np.array(
                    list(map(int, firms_modis.confidence.values.tolist())))
                firms_modis = firms_modis.drop(firms_modis[ \
                    firms_modis.confidence < CLEAN_THRESH].index)
                temp = firms_modis
                firms_viirs = firms.drop(firms[firms.instrument == 'MODIS'].index)
                if firms_viirs.size:
                    firms_viirs = firms_viirs.drop(firms_viirs[firms_viirs.confidence == 'l'].index)
                    firms_viirs = firms_viirs.rename(columns={'bright_ti4':'brightness'})
                    temp = temp.append(firms_viirs, sort=True)
                    temp = temp.drop(columns=['bright_ti4'])

            firms = temp
            firms = firms.reset_index()
            firms = firms.drop(columns=['index'])

        firms['iter_ev'] = np.ones(len(firms), bool)
        firms['cons_id'] = np.zeros(len(firms), int) - 1
        firms['event_id'] = np.zeros(len(firms), int)
        firms['clus_id'] = np.zeros(len(firms), int) - 1
        firms['datenum'] = np.array(str_to_date(firms['acq_date'].values))
        return firms

    @staticmethod
    def _firms_resolution(firms):
        """ Returns resolution of satellite used in FIRMS in degrees

        Parameters:
            firms (pd.DataFrame): FIRMS data

        Returns:
            float
        """
        # Resolution in km of the centroids depends on the data origin.
        if 'instrument' in firms.columns:
            if firms['instrument'].any() == 'MODIS':
                res_data = 1.0
            else:
                res_data = 0.375 # For VIIRS data
        else:
            res_data = RES_DATA # For undefined data origin, defined by user
        return res_data/ONE_LAT_KM

    @staticmethod
    def _centroids_creation(firms, res_data, centr_res_factor):
        """ Get centroids from the firms dataset and refactor them.

        Parameters:
            firms (DataFrame): dataset obtained from FIRMS data
            res_data (float): FIRMS instrument resolution in degrees
            centr_res_factor (float): the factor applied to voluntarly decrease/increase
                the centroids resolution

        Returns:
            centroids (Centroids)
        """
        centroids = Centroids()
        centroids.set_raster_from_pnt_bounds((firms['longitude'].min(), \
            firms['latitude'].min(), firms['longitude'].max(), \
            firms['latitude'].max()), res=res_data/centr_res_factor)
        centroids.set_meta_to_lat_lon()
        centroids.set_area_approx()
        centroids.set_on_land()
        centroids.empty_geometry_points()

        return centroids

    def _firms_cons_days(self, firms):
        """ Compute clusters of consecutive days (temporal clusters).
            An interruption of days_thresh is necessary to be set in two
            different temporal clusters.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """
        LOGGER.debug('Computing clusters of consecutive days.')
        firms_iter = firms[firms['iter_ev']][['datenum', 'cons_id', 'event_id']]
        max_cons_id = firms.cons_id.max() + 1
        for event_id in np.unique(firms_iter.event_id.values):

            firms_cons = firms_iter[firms_iter.event_id == event_id].reset_index()

            # Order firms dataframe per ascending acq_date order
            firms_cons = firms_cons.sort_values('datenum')
            sort_idx = firms_cons.index
            firms_cons = firms_cons.reset_index()
            firms_cons = firms_cons.drop(columns=['index'])

            # Check if there is more than 1 day interruption between data
            firms_cons.at[0, 'cons_id'] = max_cons_id
            max_cons_id += 1
            for index in range(1, len(firms_cons)):
                if abs((firms_cons.at[index, 'datenum'] - firms_cons.at[index-1, 'datenum'])) \
                >= self.days_thres:
                    firms_cons.at[index, 'cons_id'] = max_cons_id
                    max_cons_id += 1
                else:
                    firms_cons.at[index, 'cons_id'] = firms_cons.at[(index-1), 'cons_id']

            re_order = np.zeros(len(firms_cons), int)
            for data, order in zip(firms_cons.cons_id.values, sort_idx):
                re_order[order] = data
            firms_iter.cons_id.values[firms_iter.event_id == event_id] = re_order

        firms.cons_id.values[firms['iter_ev'].values] = firms_iter.cons_id.values
        return firms

    @staticmethod
    def _firms_clustering(firms, res_data, clus_thres):
        """Compute geographic clusters and sort firms with ascending clus_id
        for each cons_id.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            res_data (float): FIRMS instrument resolution in degrees

        Returns:
            firms
        """

        LOGGER.debug('Computing geographic clusters in consecutive events.')
        firms_iter = firms[firms['iter_ev']][['latitude', 'longitude', 'cons_id',
                                              'clus_id', 'event_id']]

        for event_id in np.unique(firms_iter.event_id.values):

            firms_cons = firms_iter[firms_iter.event_id == event_id]

            # Creation of an identifier for geographical clustering
            # For each temporal cluster, perform geographical clustering with DBSCAN algo
            for cons_id in np.unique(firms_cons['cons_id'].values):
                temp = np.argwhere(firms_cons['cons_id'].values == cons_id).reshape(-1,)
                lat_lon = firms_cons.iloc[temp][['latitude', 'longitude']].values
                lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
                cluster_id = DBSCAN(eps=res_data*clus_thres, min_samples=1).\
                                    fit(lat_lon_uni).labels_
                cluster_id = cluster_id[lat_lon_cpy]
                firms_cons.clus_id.values[temp] = cluster_id
                firms_iter.clus_id.values[firms_iter.event_id == event_id] = \
                    firms_cons.clus_id.values

        firms.clus_id.values[firms['iter_ev'].values] = firms_iter.clus_id.values

        return firms

    @staticmethod
    @numba.njit(parallel=True)
    def _firms_event(days_thres, fir_cons_id, fir_clus_id, fir_ev_id, fir_iter,
                     fir_date):
        """Creation of event_id for each dataset point.
        An event is characterized by a unique combination of 'cons_id' and 'clus_id'.

        Parameters:
            firms (dataframe)

        Returns:
            firms
        """
        ev_id = 0
        for cons_id in np.unique(fir_cons_id):
            firms_cons = fir_clus_id[fir_cons_id == cons_id]
            for clus_id in np.unique(firms_cons):
                fir_ev_id[np.logical_and(fir_cons_id == cons_id, \
                fir_clus_id == clus_id)] = ev_id
                ev_id += 1

        for ev_id in np.unique(fir_ev_id):
            date_ord = np.sort(fir_date[fir_ev_id == ev_id])
            if (np.diff(date_ord) < days_thres).all():
                fir_iter[fir_ev_id == ev_id] = False
            else:
                fir_iter[fir_ev_id == ev_id] = True

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
            return (res_centr[0] + res_centr[1])/2
        return res_centr[0]

    def _calc_brightness(self, firms, centroids, res_centr):
        """ Compute intensity matrix per event with the maximum brightness at
        each centroid and al other hazard attributes.

        Parameters:
            firms (dataframe)
            centroids (Centroids): centroids for the dataset
            res_centr (float): centroids resolution in centroids unit

        Returns:
            brightness (Hazard)
        """
        uni_ev = np.unique(firms['event_id'].values)
        num_ev = uni_ev.size
        num_centr = centroids.size

        # For one event, if more than one points of firms dataframe have the
        # same coordinates, take the maximum brightness value
        # of these points (maximal damages).
        tree_centr = BallTree(centroids.coord, metric='chebyshev')
        if self.pool:
            chunksize = min(num_ev//self.pool.ncpus, 1000)
            bright_list = self.pool.map(self._brightness_one_event,
                                        itertools.repeat(firms, num_ev),
                                        itertools.repeat(tree_centr, num_ev),
                                        uni_ev, itertools.repeat(res_centr),
                                        itertools.repeat(num_centr),
                                        chunksize=chunksize)
        else:
            bright_list = []
            for ev_id in uni_ev:
                bright_list.append(self._brightness_one_event(firms, tree_centr, \
                ev_id, res_centr, num_centr))

        self.tag = TagHazard(HAZ_TYPE)
        self.units = 'K' # Kelvin units brightness
        self.centroids = centroids

        # Following values are defined for each event
        self.event_id = np.arange(1, num_ev+1).astype(int)
        self.event_name = list(map(str, self.event_id))
        self.date = np.zeros(num_ev, int)
        self.date_end = np.zeros(num_ev, int)
        for ev_idx, ev_id in enumerate(uni_ev):
            self.date[ev_idx] = firms[firms.event_id == ev_id].datenum.min()
            self.date_end[ev_idx] = firms[firms.event_id == ev_id].datenum.max()
        self.orig = np.ones(num_ev, bool)
        self._set_frequency()

        # Following values are defined for each event and centroid
        self.intensity = sparse.lil_matrix(np.zeros((num_ev, num_centr)))
        for idx, ev_bright in enumerate(bright_list):
            self.intensity[idx] = ev_bright
        self.intensity = self.intensity.tocsr()
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    @staticmethod
    def _brightness_one_event(firms, tree_centr, ev_id, res_centr, num_centr):
        """ For a given event, fill in an intensity np.array with the maximum brightness
        at each centroid.

        Parameters:
            firms (dataframe)
            centroids (Centroids): centroids for the dataset
            ev_id (int): id of the selected event

        Returns:
            brightness_ev (np.array): maximum brightness at each centroids

        """
        LOGGER.debug('Brightness corresponding to FIRMS event %s.', str(ev_id))
        temp_firms = firms.reindex(index=(np.argwhere(firms['event_id'] == ev_id).reshape(-1,)),
                                   columns=['latitude', 'longitude', 'brightness'])

        # Identifies the unique (lat,lon) points of the firms dataframe -> lat_lon_uni
        # Set the same index value for each duplicate (lat,lon) points -> lat_lon_cpy
        lat_lon_uni, lat_lon_cpy = np.unique(temp_firms[['latitude', 'longitude']].values,
                                             return_inverse=True, axis=0)
        index_uni = np.unique(lat_lon_cpy, axis=0)

        # Search closest centroid for each firms point
        ind, _ = tree_centr.query_radius(lat_lon_uni, r=res_centr/2, count_only=False,
                                         return_distance=True, sort_results=True)
        ind = np.array([ind_i[0] if ind_i.size else -1 for ind_i in ind])
        brightness_ev = _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy,
                                        temp_firms['brightness'].values)
        return sparse.lil_matrix(brightness_ev)

    def _set_proba_one_event(self, ev_idx, ens_size, seed):
        """ Generate a set of probabilistic events for a given historical event.

        Parameters:
            ev_idx (int): the selected historical event
            ens_size (int): number of probabilistic events to generate for a
                given historical event
            seed (int): random number generator seed

        Raises:
            bf_haz (list): list of hazard corresponding to new probabilistic events

        """
        np.random.seed(seed)
        bf_haz = []
        for i_ens in range(ens_size):
            bf_haz.append(self._random_bushfire_one_event(ev_idx, i_ens))
        return bf_haz

    def _random_bushfire_one_event(self, ev_idx, i_ens):
        """ For a given historical event, select randomly a centroid as
            ignition starting point (plus or minus a wiggle) and check
            that the new fire starting point is located on land.
            Propagate the fire from the ignition point with a cellular automat.
            Propagation rules:
                1. an empty centroid becomes a burning centroid with a probability
                    PROP_PROBA if any of its eight neighbouring cells are burning
                2. an already burning centroid becomes an ember centroid
                    (do not propagate fire anymore but still increases the damage)
                3. an ember centroid stays an ember centroid
            Properties from centr_burned:
                0 = unburned centroid
                1 = burned centroid
                >1 = ember centroid
            Stop criteria: the propagation stops when the burnt area is equal or
            bigger than the burnt area of the given historical event.
            Inspired from https://scipython.com/blog/the-forest-fire-model/

        Parameters:
            ev_idx (int): the selected historical event
            i_ens (int): number of the generated probabilistic event

        Returns:
            new_haz: Hazard
        """
        # Calculate area of historical event in m2
        pos_centr = np.argwhere(self.intensity[ev_idx, :] > 0)[:, 1]
        area_hist_event = self.centroids.area_pixel[pos_centr].sum()

        LOGGER.debug('Start ignition.')
        # Random selection of ignition centroid
        for _ in range(self.centroids.size):
            centr = np.random.choice(pos_centr)
            centr_ix = int(centr/self.centroids.shape[1])
            centr_iy = centr%self.centroids.shape[1]
            centr_ix += np.random.randint(-WIGGLE, WIGGLE+1)
            centr_ix = max(0, centr_ix)
            centr_ix = min(self.centroids.shape[0]-1, centr_ix)
            centr_iy += np.random.randint(-WIGGLE, WIGGLE+1)
            centr_iy = max(0, centr_iy)
            centr_iy = min(self.centroids.shape[1]-1, centr_iy)
            centr = centr_ix*self.centroids.shape[1] + centr_iy
            if self.centroids.on_land[centr] and \
            1 <= centr_ix < self.centroids.shape[0] - 1 and \
            1 <= centr_iy < self.centroids.shape[1] - 1:
                break

        LOGGER.debug('Propagate fire.')
        centr_burned = np.zeros((self.centroids.shape), int)
        centr_burned[centr_ix, centr_iy] = 1
        area_prob_event = 0
        # Iterate the fire according to the propagation rules
        count_it = 0
        while area_prob_event - area_hist_event < 0:
            count_it += 1
            # Select randomly one of the already burned centroids
            # and propagate throught its neighborhood
            burned = np.argwhere(centr_burned == 1)
            if len(burned) > 1:
                centr_ix, centr_iy = burned[np.random.randint(0, len(burned))]
            if not count_it % 200000:
                LOGGER.warning('Fire propagation not converging at iteration %s.' +
                               ' Selecting new ignition point.', count_it)
                centr = np.random.choice(pos_centr)
                centr_ix = int(centr/self.centroids.shape[1])
                centr_iy = centr%self.centroids.shape[1]
                centr_burned[centr_ix, centr_iy] = 1
            if 1 <= centr_ix < self.centroids.shape[0]-1 and \
            1 <= centr_iy < self.centroids.shape[1]-1 and \
            self.centroids.on_land[(centr_ix*self.centroids.shape[1] + centr_iy)]:
                area_prob_event = self._fire_propagation(self.centroids.shape, \
                    self.centroids.on_land.reshape(self.centroids.shape).astype(int), \
                    self.centroids.area_pixel, centr_ix, centr_iy, centr_burned, \
                    np.random.random(500))

        return self._event_probabilistic(ev_idx, i_ens, centr_burned)

    @staticmethod
    @numba.njit
    def _fire_propagation(centr_shape, centr_land, centr_area, centr_ix,
                          centr_iy, centr_burned, prob_array):
        """ Propagation of the fire in the 8 neighbouring cells around
        (centr_ix, centr_iy) according to propagation rules.

        Parameters:
            centr_ix (int): x coordinates of the starting centroid in the
                NB_CENTR x NB_CENTR centroids matrix
            centr_iy (int): y coordinates of the starting centroid in the
                NB_CENTR x NB_CENTR centroids matrix
            centr_burned (np.array): array containing burned centroids

        Returns:
            float (burned area of probabilistic event in m2)
        """
        # Neighbourhood
        hood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        # Creation of the temporary centroids grids
        burned_one_step = np.zeros(centr_shape, dtype=numba.int32)
        burned_one_step[centr_ix, centr_iy] = 1

        # Displacements from a centroid to its eight nearest neighbours
        # with a PROP_PROPA and if centroids not already burned
        for i_neig, (delta_x, delta_y) in enumerate(hood):
            if prob_array[i_neig] <= PROP_PROPA and\
            centr_burned[centr_ix+delta_x, centr_iy+delta_y] == 0:
                burned_one_step[centr_ix+delta_x, centr_iy+delta_y] = 1

        # Check for special case where no new burned centroids is added
        # Try again the propagation in the 8 neighbouring centroids.
        while len(burned_one_step.nonzero()[0]) <= 1:
            for delta_x, delta_y in hood:
                i_neig += 1
                if prob_array[i_neig+len(hood)] <= PROP_PROPA:
                    burned_one_step[centr_ix+delta_x, centr_iy+delta_y] = 1

        # Calculate burned area
        centr_burned += burned_one_step * centr_land
        centr = centr_burned.nonzero()[0] * centr_shape[1] + centr_burned.nonzero()[1]
        return centr_area[centr].sum()

    def _event_probabilistic(self, ev_idx, i_ens, centr_burned):
        """ Define synthetic hazard from randomly burned centroids.

        Parameters:
            ev_idx (int): the selected historical event
            i_ens (int): number of the generated probabilistic event
            centr_burned (np.array): array containing burned centroids

        Returns:
            new_haz (Hazard)
        """
        LOGGER.debug('Brightness probabilistic event.')

        # The brightness values are chosen randomly at every burned centroids
        # from the brightness values of the historical event
        ev_proba_uni = centr_burned.nonzero()[0] * self.centroids.shape[1] + \
            centr_burned.nonzero()[1]
        new_haz = Hazard(HAZ_TYPE)
        new_haz.intensity = sparse.lil_matrix(np.zeros((1, self.centroids.size)))
        for ev_prob in ev_proba_uni:
            new_haz.intensity[0, ev_prob] = np.random.choice( \
                self.intensity[ev_idx, :].data)
        new_haz.intensity = new_haz.intensity.tocsr()

        # Hazard
        new_haz.tag = TagHazard(HAZ_TYPE)
        new_haz.units = 'K' # Kelvin units brightness
        new_haz.centroids = self.centroids
        new_haz.event_id = np.ones(1, int)
        new_haz.frequency = np.ones(1, float)
        new_haz.event_name = [str(ev_idx+1) + '_gen' + str(i_ens+1)]
        new_haz.date = np.array([self.date[ev_idx]], int)
        new_haz.orig = np.zeros(1, bool)

        # Following values are defined for each event and centroid
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1.0)

        return new_haz

    def _set_frequency(self):
        """Set hazard frequency from intensity matrix. """
        delta_time = date.fromordinal(int(np.max(self.date))).year - \
            date.fromordinal(int(np.min(self.date))).year + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

@numba.njit
def _fill_intensity(num_centr, ind, index_uni, lat_lon_cpy, fir_bright):
    brightness_ev = np.zeros((1, num_centr), dtype=numba.float64)
    for idx in range(index_uni.size):
        if ind[idx] != -1:
            brightness_ev[0, ind[idx]] = max(brightness_ev[0, ind[idx]], \
                         np.max(fir_bright[lat_lon_cpy == index_uni[idx]]))
    return brightness_ev
