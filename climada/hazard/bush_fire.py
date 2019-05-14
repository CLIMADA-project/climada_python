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

Define BushFire class."""

__all__ = ['BushFire']

import itertools
import logging
from datetime import date, datetime

from functools import partial
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN
from pathos.multiprocessing import ProcessingPool as Pool
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import transform
import pyproj

from climada.hazard.centroids.centr import Centroids
from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.util.constants import ONE_LAT_KM
from climada.util.constants import EARTH_RADIUS_KM

from climada.util.alpha_shape import alpha_shape


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'BF'
""" Hazard type acronym for Bush Fire """

CLEAN_THRESH = 30
""" Minimal confidence value for the data from MODIS instrument to be use as input"""

THRESH_INTERP = 0.375 #If no information about the instrument used
THRESH_INTERP_MODIS = 1.0
THRESH_INTERP_VIIRS = 0.5
""" Distance threshold in km over which no neighbor will be found """

RES_DATA = 0.375
""" Resolution of the data if no data origin provided (km)"""

MIN_SAMPLES = 1
""" Number of samples (or total weight) in a neighborhood for a point to
be considered as a core point. This includes the point itself."""

WIGGLE = 0.05
"""max distance by which onset of fire can be wiggled. 5 km"""

PROP_PROPA = 0.25
"""Probability of fire propagation.
    For now, arbitrary value"""

class BushFire(Hazard):
    """Contains bush fire events."""

    def __init__(self):
        """Empty constructor. """

        Hazard.__init__(self, HAZ_TYPE)

    def set_bush_fire(self, csv_firms, centr_res_factor, seed, description=''):
        # Identify events: 1 event = 1 fire up to its maximal extension
        # Firms data clustered by consecutive dates
        # Then data clustered geographically for each time cluster
        """ Fill in the hazard file

        Parameters:
            csv_firms: csv file of the FIRMS data
                (obtained here:https://firms.modaps.eosdis.nasa.gov/download/)
            centroids (Centroids, optional): Centroids where to model BF.
                Default: global centroids.
            seed (int, optional): random number generator seed. Put negative
                value if you don't want to use it.
            description (str, optional): description of the events.

        Raises:
            ValueError

        """
        LOGGER.info('Setting up historical event.')

        if seed >= 0:
            np.random.seed(seed)

        firms, description = self._read_firms_csv(csv_firms, description)
        firms = self._clean_firms_csv(firms)
        # Add cons_id
        firms = self._firms_cons_days(firms)
        # Add clus_id and event_id
        centroids, res_data = self._centroids_creation(firms, centr_res_factor)
        firms = self._firms_clustering(firms, res_data)
        firms = self._firms_event(firms)
        # compute brightness and fill class attributes
        self.clear()
        self._calc_brightness(firms, centroids)

    def set_proba_all_event(self, ens_size):
        """ Generate a set of probabilistic events for all historical event.
        Parallel process

        Parameters:
            ens_size (int): number of probabilistic events to generate per
                historical event

        Raises:
            ValueError

        """
        haz = []
        # Next two lines, to keep for debugging
        for _, ev_id in enumerate(self.event_id):
            haz.extend(self._set_proba_one_event(ev_id, ens_size))

        # problem random num generator in multiprocessing. python 3.7?
#        chunksize = min(ens_size, 1000)
#        haz = Pool().map(self._set_proba_one_event, self.event_id,
#                          itertools.repeat(ens_size, self.event_id.size),
#                          chunksize=chunksize)

        LOGGER.debug('Append events.')
        prob_haz = BushFire()
        prob_haz._append_all(haz)
        self.append(prob_haz)

        # Update the frequency adter adding the probabilistic events
        self._set_frequency(ens_size)

    # Historical event (from FIRMS data)
    @staticmethod
    def _read_firms_csv(csv_firms, description=''):
        """Read csv files from FIRMS data.

        Parameters:
            csv_firms: csv file of the FIRMS data
            description (str, optional): description of the events

        Returns:
            firms, description
        """
        # Open and read the file
        firms = pd.read_csv(csv_firms)

        # Create "datenum" column in firms dataframe
        for index, acq_date in enumerate(firms['acq_date'].values):
            datenum = datetime.strptime(acq_date, '%Y-%M-%d').toordinal()
            firms.at[index, 'datenum'] = datenum
        return firms, description

    @staticmethod
    def _clean_firms_csv(firms):
        """Optional - Remove low confidence data from firms:
            1 - MODIS: remove data where confidence values are lower than CLEAN_THRESH
            2 - VIIRS: remove data where confidence values are set to low
            (keeps nominal and high values)

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """
        # Check for the type of instrument (MODIS vs VIIRS)
        # Remove data with low confidence interval
        # Uniformize the name of the birghtness columns between VIIRS and MODIS
        temp = pd.DataFrame()
        if 'instrument' in firms.columns:
            if firms.instrument.any() == 'MODIS' or firms.instrument.any() == 'VIIRS':
                firms_modis = firms.drop(firms[firms.instrument == 'VIIRS'].index)
                firms_modis.confidence = np.array(
                    list(map(int, firms_modis.confidence.values.tolist())))
                firms_modis = firms_modis.drop\
                (firms_modis[firms_modis.confidence < CLEAN_THRESH].index)
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
        return firms

    @staticmethod
    def _firms_cons_days(firms):
        """Compute clusters of consecutive days (temporal clusters).
            An interruption of at least two days is necessary to be set in two
            different temporal clusters.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Computing clusters of consecutive days.')
        # Order firms dataframe per ascending acq_date order
        firms = firms.sort_values('acq_date')
        firms = firms.reset_index()
        firms = firms.drop(columns=['index'])
        cons_id = np.zeros(len(firms['acq_date'].values))
        firms['cons_id'] = pd.Series(cons_id)
        # Check if there is more than 1 day interruption between data
        for index, _ in list(map(tuple, firms['acq_date'].items()))[1:]:
            day_2 = datetime.strptime(firms.at[index, 'acq_date'], "%Y-%m-%d")
            day_1 = datetime.strptime(firms.at[(index-1), 'acq_date'], "%Y-%m-%d")
            if abs((day_2 - day_1).days) > 2:
                firms.at[index, 'cons_id'] = firms.at[(index-1), 'cons_id']+1
            else:
                firms.at[index, 'cons_id'] = firms.at[(index-1), 'cons_id']
        return firms

    def _firms_clustering(self, firms, res_data):
        """Compute geographic clusters and sort firms with ascending clus_id
        for each cons_id.
        Parallel process.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            res_data (float): Data resolution (depends of the data origin)

        Returns:
            firms
        """

        LOGGER.info('Computing geographic clusters in consecutive events.')
        # Creation of an identifier for geographical clustering
        cluster_id = np.zeros((firms.index.size,))-1
        # For each temporal cluster, perform geographical clustering with DBSCAN algo
        for cons_id in np.unique(firms['cons_id'].values):
            temp = np.argwhere(firms['cons_id'].values == cons_id).reshape(-1,)
            lat_lon = firms.reindex(index=temp,
                                    columns=['latitude', 'longitude'])
            lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
            cluster_uni = DBSCAN(eps=res_data/2, min_samples=MIN_SAMPLES).fit(lat_lon_uni).labels_
            cluster_cpy = cluster_uni[lat_lon_cpy]
            cluster_id[temp] = cluster_cpy
        firms['clus_id'] = pd.Series(cluster_id)

        LOGGER.info('Sorting of firms data')
        # Re-order FIRMS file to have clus_id in ascending order inside each cons_id
        firms_sort = pd.DataFrame()
        cons_id_uni = np.unique(firms['cons_id'].values)
        for _, cons_id in enumerate(cons_id_uni):
            firms_sort = firms_sort.append(self._firms_clus(firms, cons_id))

        firms = firms_sort
        firms = firms.reset_index()
        firms = firms.drop(columns=['index'])
        return firms

    @staticmethod
    def _firms_clus(firms, cons_id):
        """ For a given temporal cluster ('cons_id') sort resulting firms data
        with 'clus_id' in ascending order.

        Parameters:
            firms (dataframe)
            cons_id (int): id of temporal cluster

        Returns:
            firms_clus (dataframe): dataframe for the given cons_id, with one
            additional column (clus_id) and sorted with clus_id in ascending order
        """
        firms_clus = firms.reindex(index=(np.argwhere(
            firms['cons_id'].values == cons_id).reshape(-1,)))
        firms_clus = firms_clus.sort_values('clus_id')
        firms_clus = firms_clus.reset_index()
        firms_clus = firms_clus.drop(columns=['index'])
        return firms_clus

    @staticmethod
    def _firms_event(firms):
        """Creation of event_id for each dataset point.
        An event is characterized by a unique combination of 'cons_id' and 'clus_id'.

        Parameters:
            firms (dataframe)

        Returns:
            firms
        """

        LOGGER.info('Creation of event_id.')
        event_id = np.zeros((firms.index.size,))+1
        firms['event_id'] = pd.Series(event_id)
        # New event = change in cons_id and clus_id
        for index, _ in list(map(tuple, firms['clus_id'].items()))[1:]:
            if ((firms.at[index, 'clus_id'] - firms.at[index-1, 'clus_id']) == 0) \
            & ((firms.at[index, 'cons_id'] - firms.at[index-1, 'cons_id']) == 0):
                firms.at[index, 'event_id'] = int(firms.at[index-1, 'event_id'])
            else:
                firms.at[index, 'event_id'] = int(firms.at[index-1, 'event_id']+1)
        print('Nb of event', max(firms['event_id'].values))
        return firms

    @staticmethod
    def _event_per_year(firms, year):
        """Compute the number of events per (calendar) year.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            year (int): selected year

        Returns:
            event_nb (int)
        """
        # Add column with year of event
        firms['year'] = pd.DatetimeIndex(firms['acq_date']).year

        # Create a subdataframe with data from the selected year only
        firms_temp = firms.reindex(index=(np.argwhere(firms['year'].values == year).reshape(-1,)))
        event_nb = np.max(firms_temp['event_id'].values) - np.min(firms_temp['event_id'].values) +1
        print('year:', year, 'nb of event:', event_nb)
        return event_nb

    @staticmethod
    def _centroids_creation(firms, centr_res_factor):
        """ Compute centroids for the firms dataset.
            The number of centroids is defined according to the data resolution.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            centr_res_factor (float): the factor applied to voluntarly decrease/increase
                the centroids resolution

        Returns:
            centroids (Centroids)
            res_data (float): data resolution (km)
        """

        LOGGER.info('Defining the resolution of the centroids.')
        # Resolution of the centroids depends on the data origin.
        # Resolution in km.
        if 'instrument' in firms.columns:
            if firms['instrument'].any() == 'MODIS':
                res_data = 1.0
            else:
                res_data = 0.375 # For VIIRS data
        else:
            res_data = RES_DATA # For undefined data origin, defined by user

        LOGGER.info('Computing centroids.')
        centroids = Centroids()
        dlat_km = abs(firms['latitude'].min() - firms['latitude'].max()) * ONE_LAT_KM
        dlon_km = abs(firms['longitude'].min() - firms['longitude'].max()) * ONE_LAT_KM* \
            np.cos(np.radians((abs(firms['latitude'].min() - firms['latitude'].max()))/2))
        nb_centr_lat = int(dlat_km/res_data * centr_res_factor)
        nb_centr_lon = int(dlon_km/res_data * centr_res_factor)
        coord = (np.mgrid[firms['latitude'].min() : firms['latitude'].max() : \
            complex(0, nb_centr_lat), firms['longitude'].min() : firms['longitude'].max() : \
            complex(0, nb_centr_lon)]).reshape(2, nb_centr_lat*nb_centr_lon).transpose()
        centroids.set_lat_lon(coord[:, 0], coord[:, 1])

#        centroids.set_raster_from_pnt_bounds((firms['longitude'].min(),
#        firms['latitude'].min(), firms['longitude'].max(), firms['latitude'].max()),
#        res=res_data/centr_res_factor)    ---> propagation works?

        # Calculate the area attributed to each centroid
        centroids.set_area_approx()
        # Calculate if the centroids is on land or not
        centroids.set_on_land()
        # Calculate to distance to coast
        centroids.set_dist_coast()
        # Create on land grid
        centroids.land = centroids.on_land.reshape((nb_centr_lat, nb_centr_lon)).astype(int)
        centroids.nb_centr_lat = nb_centr_lat
        centroids.nb_centr_lon = nb_centr_lon
        centroids.empty_geometry_points()
        centroids.set_lat_lon_to_meta()
        return centroids, res_data

    def _calc_brightness(self, firms, centroids):
        """ Fill in the intensity matrix with, for each event,
        the maximum brightness at each centroid.
        Parallel process.

        Parameters:
            firms (dataframe)
            centroids (Centroids): centroids for the dataset

        Returns:
            brightness (Hazard)
        """
        num_ev = int(max(firms['event_id'].values))
        num_centr = centroids.size

        LOGGER.debug('Filling up the matrix.')
        # Matrix intensity: events x centroids
        # For one event, if more than one points of firms dataframe have the
        # same coordinates, take the maximum brightness value
        # of these points (maximal damages).
        # Fill the matrix

        bright_list = []
        # Next two lines, to keep for debugging (replace parallel process)
#        for _, ev_id in enumerate(np.unique(firms['event_id'].values)):
#            bright_list.append(self._brightness_one_event(
#                firms, centroids, ev_id, num_centr))

        chunksize = min(num_ev, 1000)
        bright_list = Pool().map(self._brightness_one_event,
                                 itertools.repeat(firms, num_ev),
                                 itertools.repeat(centroids, num_ev),
                                 np.unique(firms['event_id'].values),
                                 itertools.repeat(num_centr, num_ev),
                                 chunksize=chunksize)

        # Fill in the brightness matrix, event after event
        self.intensity = sparse.lil_matrix(np.zeros((num_ev, num_centr)))
        for idx, _ in enumerate(bright_list):
            self.intensity[idx] = bright_list[idx]
        self.intensity = self.intensity.tocsr()

        # Fill in Hazard file
        self.tag = TagHazard(HAZ_TYPE)
        self.units = 'K' # Kelvin units brightness
        self.centroids = centroids
        # Following values are defined for each event
        self.event_id = np.array(np.unique(firms['event_id'].values))
        self.event_name = list(map(str, self.event_id))
        event_uni, _ = np.unique(firms['event_id'].values, return_inverse=True, axis=0)
        date_tmp = np.zeros(len(np.unique(firms['event_id'].values)))
        for ev_id, event in enumerate(event_uni):
            temp = firms.loc[firms['event_id'] == event, 'datenum']
            temp = temp.reset_index()
            date_tmp[ev_id] = temp.at[0, 'datenum']
        self.date = date_tmp
        self.frequency = np.ones(self.event_id.size)
        self.orig = np.tile(True, len(np.unique(firms['event_id'])))

        # Following values are defined for each event and centroid
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    @staticmethod
    def _brightness_one_event(firms, centroids, ev_id, num_centr):
        """ For a given event, fill in an intensity np.array with the maximum brightness
        at each centroid.

        Parameters:
            firms (dataframe)
            centroids (Centroids): centroids for the dataset
            ev_id (int): id of the selected event
            num_centr (int): total number of centroids

        Returns:
            brightness_ev (np.array): maximum brightness at each centroids

        """
        LOGGER.debug('Sub-dataframe corresponding to event')
        temp_firms = firms.reindex(index=(np.argwhere(firms['event_id'] == ev_id).reshape(-1,)),
                                   columns=['index', 'latitude', 'longitude', 'brightness'])


        LOGGER.info('Identifying closest (lat,lon) points from firms dataframe for each centroid.')
        # Identifies the unique (lat,lon) points of the firms dataframe -> lat_lon_uni
        # Set the same index value for each duplicate (lat,lon) points -> lat_lon_cpy
        # temp_firms['index'] = lat_lon_cpy is used after to fill up the intensity matrix
        lat_lon = np.array(temp_firms[['latitude', 'longitude']].values)
        lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
        temp_firms['index'] = lat_lon_cpy

        index_uni, index_cpy = np.unique(temp_firms['index'].values,
                                         return_inverse=True, axis=0)

        LOGGER.info('Computing maximum brightness for each (lat,lon) point.')
        bright_one_event = np.zeros(index_uni.size)
        for index_idx, index in enumerate(index_uni):
            bright_one_event[index_idx] = np.max(temp_firms['brightness']\
                            [index_uni[index_cpy] == index])

        LOGGER.debug('BallTree')
        # Index of the closest (lat,lon) points of firms dataframe for each centroids
        # If latlon = -1 -> no (lat,lon) points close enought from the centroid
        # (compare to threshold)
        # Threshold for interpolation, depends on data resolution (and then on data origin)
        tree = BallTree(np.radians(centroids.coord), metric='haversine')
        ind, _ = tree.query_radius(np.radians(lat_lon_uni),
                                   r=1/EARTH_RADIUS_KM, count_only=False,
                                   return_distance=True, sort_results=True)
        LOGGER.debug('Brightness np.array for the event')
        brightness_ev = sparse.lil_matrix(np.zeros((num_centr)))
        for idx, _ in enumerate(index_uni):
            brightness_ev[0, ind[idx][0]] = bright_one_event[idx]

        return brightness_ev

    def _area_one_year(self, year):
        """ Calculate the area burn over one (calendar) year.
        This area does NOT correspond to the real area (due to projection issue).

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            year: year for which to calculate the area

        Returns:
            area_one_year (km2): burned area over one calendar year
        """
        year_list = []
        for _, datenum in enumerate(self.date):
            year_list.append(date.fromordinal(int(datenum)).year)

        ev_id_selec = []
        for idx, _ in enumerate(year_list):
            if year_list[idx] == year:
                selec = self.event_id[idx]
                ev_id_selec.append(selec)

        if not ev_id_selec:
            LOGGER.error('No event for this year')
            raise ValueError

        # Area calculation
        area_one_year = 0
        area_one_event = []
        for _, ev_id in enumerate(ev_id_selec):
            area_one_event.append(self._area_one_event(ev_id))
        area_one_year = area_one_year + sum(area_one_event)
        return area_one_year

    def _area_one_event(self, ev_id):
        """ Calculate the area of one event by summing the area of each centroids
            with a non-zero brightness value.
            This area is an approximation and does NOT correspond to
            the real area (due to projection issue).

        Parameters:
            centroids (Centroids): Centroids instance.
            ev_id: selected event id

        Returns:
            area_one_event (km2): burned area of one event
        """

        # Identifies the centroids with non-zero brightness value
        col = list(self.intensity[ev_id-1].indices)
        # Sum the are of each of these centroids
        area_one_event = 0
        for _, colv in enumerate(col):
            area_temp = self.centroids.area_pixel[colv]
            area_one_event = area_one_event + area_temp
        return area_one_event


    # Probabilistic event
    def _random_bushfire_one_event(self, ev_id, i_ens):
        """ For a given historical event, select randomly a centroid as
            ignition starting point (plus or minus a wiggle) and check
            that the new fire starting point is located on land.
            Propagate the fire from the ignition point with a cellular automat.
            Propagation rules:
                1. an empty centroid becomes a burning centroid with a probability PROP_PROBA
                    if any of its eight neighbouring cells are burning
                2. an already burning centroid becomes an ember centroid
                    (do not propagate fire anymore but still increases the damage)
                3. an ember centroid stays an ember centroid
            Properties from centroids.burned:
                0 = unburned centroid
                1 = burned centroid
                >1 = ember centroid
            Stop criteria: the propagation stops when the burnt area is equal or
            bigger than the burnt area of the given historical event.
            Inspired from https://scipython.com/blog/the-forest-fire-model/

        Parameters:
            ev_id (int): id of the selected historical event from where to start
            i_ens (int): number of the generated probabilistic event

        Returns:
            new_haz: Hazard
        """
        # Calculate area of historical event
        area_hist_event = self._area_one_event(ev_id)

        LOGGER.debug('Start ignition.')
        # Identification of the possible ignition centroids
        pos_centr = np.argwhere(self.intensity[ev_id-1] > 0)
        for _ in range(self.centroids.size):
            centr = np.random.RandomState(8).choice(pos_centr[:, 1])
            # Wiggling
            _, _, centr = self.centroids.get_closest_point( \
                self.centroids.lon[centr] + 2*(np.random.random()-0.5)*WIGGLE,
                self.centroids.lat[centr] + 2*(np.random.random()-0.5)*WIGGLE)
            if self.centroids.on_land[centr]:
                centr_ix = int(centr/self.centroids.nb_centr_lon)
                centr_iy = centr%self.centroids.nb_centr_lon
                # Check that the ignition point is not located on the edge
                # Would be problematic for propagation
                if centr_ix >= 1 and centr_ix < self.centroids.nb_centr_lat - 1:
                    if centr_iy >= 1 and centr_iy < self.centroids.nb_centr_lon - 1:
                        break

        LOGGER.debug('Initialize the burned grid.')
        self.centroids.burned = np.zeros((self.centroids.nb_centr_lat, self.centroids.nb_centr_lon))
        self.centroids.burned[centr_ix, centr_iy] = 1
        area_prob_event = 0

        # Iterate the fire according to the propagation rules
        LOGGER.debug('Propagate fire.')
        for _ in range(4000000):
            if area_prob_event - area_hist_event > 0.01:
                break
            else:
                # For each timestep:
                # Select randomly one of the already burned centroids
                # and propagate the fire starting from this centroid
                burned = np.argwhere(self.centroids.burned == 1)
                rand = 0
                if len(burned) > 1:
                    rand = np.random.randint(0, len(burned))
                    centr_ix = burned[rand][0]
                    centr_iy = burned[rand][1]
                if centr_ix >= 1 and centr_ix < self.centroids.nb_centr_lat-1 and \
                centr_iy >= 1 and centr_iy < self.centroids.nb_centr_lon-1 and \
                self.centroids.on_land[(centr_ix*self.centroids.nb_centr_lon + centr_iy)]:
                    area_prob_event = self._fire_propagation(area_prob_event, centr_ix, centr_iy)

        # Add a row to the intensity matrix for the new (probabilistic) event
        new_haz = self._event_probabilistic(ev_id, i_ens)
        return new_haz

    def _fire_propagation(self, area_prob_event, centr_ix, centr_iy):
        """ Propagation of the fire in the 8 neighbouring cells around
        (centr_ix, centr_iy) according to propagation rules.
        Check after each new burned cell if the area of the probabilistic event
        is bigger than the historical event.

        Parameters:
            area_prob_event (km2): burned area of probabilistic event (change at each timestep)
            centr_ix, centr_iy: coordinates of the starting centroids in the
                NB_CENTR x NB_CENTR centroids "matrix"

        Returns:
            area_prob_event (km2): burned area of probabilistic event
        """
        # Neighbourhood
        hood = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

        # Creation of the temporary centroids grids
        self.centroids.burned_one_step = np.zeros((self.centroids.nb_centr_lat,
                                                   self.centroids.nb_centr_lon))
        self.centroids.burned_one_step[centr_ix, centr_iy] = 1

        # Displacements from a centroid to its eight nearest neighbours
        # with a PROP_PROPA and if centroids not already burned
        for delta_x, delta_y in hood:
            if np.random.random() <= PROP_PROPA and\
            self.centroids.burned[centr_ix+delta_x, centr_iy+delta_y] == 0:
                self.centroids.burned_one_step[centr_ix+delta_x, centr_iy+delta_y] = 1

        # Check for special case where no new burned centroids is added
        # Try again the propagation in the 8 neighbouring centroids as long as in special case.
        while len(self.centroids.burned_one_step.nonzero()[0]) <= 1:
            for delta_x, delta_y in hood:
                if np.random.random() <= PROP_PROPA:
                    self.centroids.burned_one_step[centr_ix+delta_x, centr_iy+delta_y] = 1

        self.centroids.burned = self.centroids.burned + \
            self.centroids.burned_one_step * self.centroids.land

        # Calculate burned area
        centr = np.array(self.centroids.burned.nonzero()[0] *
                         self.centroids.nb_centr_lon + self.centroids.burned.nonzero()[1])
        area_prob_event = 0
        for _, centr_id in enumerate(centr):
            area_prob_event = area_prob_event + self.centroids.area_pixel[centr_id]

        return area_prob_event

    def _event_probabilistic(self, ev_id, i_ens):
        """ Append a row in the intensity matrix for the probabilistic event.
        The brightness value of the probabilistic event are randomly selected
        in the intensity matrix corresponding to the historical event.
        Fill in the hazard file for the probabilistic event.

        Parameters:
            ev_id: id of the selected historical event
            i_ens (int): number of the generated probabilistic event

        Returns:
            new_haz (Hazard): new hazard corresponding to one probabilistic event
        """
        LOGGER.debug('Brightness probabilistic event.')

        # Probabilistic event
        ev_proba = np.argwhere(self.centroids.burned >= 1)
        ev_proba_uni = (ev_proba[:, 0]-1) * self.centroids.nb_centr_lon + ev_proba[:, 1]

        # Append a row to the intensity matrix with the brightness values of the
        # probabilistic event
        # The brightness value is chosen randomly from the brightness values of
        # the historical event.
        new_haz = Hazard(HAZ_TYPE)
        new_haz.intensity = sparse.lil_matrix(np.zeros(self.centroids.size))
        for _, ev_prob in enumerate(ev_proba_uni):
            bright_proba = np.random.choice(self.intensity[ev_id -1].data)
            new_haz.intensity[0, ev_prob] = bright_proba
        new_haz.intensity = new_haz.intensity.tocsr()

        # Hazard
        new_haz.tag = TagHazard(HAZ_TYPE)
        new_haz.units = 'K' # Kelvin units brightness
        new_haz.centroids = self.centroids
        new_haz.event_id = np.ones(1, int)
        new_haz.frequency = np.ones(1, float)
        new_haz.event_name = [str(ev_id) + '_gen' + str(i_ens)]
        new_haz.date = np.array([self.date[int(ev_id-1)]])
        new_haz.orig = np.zeros(1, bool)

        # Following values are defined for each event and centroid
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1.0)

        return new_haz

    def _set_proba_one_event(self, ev_id, ens_size):
        """ Generate a set of probabilistic events for a given historical event.

        Parameters:
            ev_id (int): id of the selected historical event
            ens_size (int): number of probabilistic events to generate for a
                given historical event

        Raises:
            bf_haz (list): list of hazard corresponding to new probabilistic events

        """
        bf_haz = []
        for i_ens in range(ens_size):
            bf_haz.append(self._random_bushfire_one_event(ev_id, i_ens))
        return bf_haz

    def _set_frequency(self, ens_size):
        """Set hazard frequency from intensity matrix.
        Parameters:
            ens_size (int): number of probabilistic events to generate per
                historical event
        """
        delta_time = date.fromordinal(int(np.max(self.date))).year - \
            date.fromordinal(int(np.min(self.date))).year + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    def _hull_burned_area(self, ev_id):
        """Compute the burned area for a given event.
        Parameters:
            ev_id: id of the selected event
        """
        # Extract coordinates where intensity > 0 (for a given event)
        fire_lat = []
        fire_lon = []
        for _, cent_id in enumerate(self.intensity[ev_id-1].nonzero()[1]):
            fire_lat.append(self.centroids.lat[cent_id])
            fire_lon.append(self.centroids.lon[cent_id])

        # Creation of the geodataframe
        orig_proj = 'epsg:4326'
        fire = gpd.GeoDataFrame()
        fire['geometry'] = list(zip(fire_lon, fire_lat))
        fire['geometry'] = fire['geometry'].apply(Point)

        fire.crs = {'init': orig_proj}
        points = fire.geometry.values

        # Compute concave hull
        concave_hull, _ = alpha_shape(points, alpha=100.87)

        # Compute area concave hull in right projection
        project = partial(
            pyproj.transform,
            pyproj.Proj(init=orig_proj), # source coordinate system
            pyproj.Proj(init='epsg:3310')) # destination coordinate system: albers california

        concave_hull_m = transform(project, concave_hull)  # apply projection
        area_hull_one_event = concave_hull_m.area/10000
        LOGGER.info('area: %s ha', area_hull_one_event)

        # Plot the polygone around the fire
#        from climada.util.alpha_shape import plot_polygon
#        _ = plot_polygon(concave_hull)
#        _ = pl.plot(fire_lon, fire_lat, 'o', color='red', markersize=0.5)

        return area_hull_one_event
