"""Define BushFire class."""

__all__ = ['BushFire']

import itertools
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import DBSCAN
from pathos.multiprocessing import ProcessingPool as Pool

from climada.hazard.centroids.base import Centroids
from climada.util.interpolation import interpol_index
from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'BF'
""" Hazard type acronym for Bush Fire """

CLEAN_THRES = 50
""" Minimal confidence value for the data to be use as input"""

THRESH_INTERP = 2 # Value used for the unittest
THRESH_INTERP_MODIS = 1.0
THRESH_INTERP_VIIRS = 0.375
""" Distance threshold in km over which no neighbor will be found """

EPS = 0.10
""" Maximum distance (in °) between two samples for them to be considered
as in the same neighborhood"""

MIN_SAMPLES = 1
""" Number of samples (or total weight) in a neighborhood for a point to
be considered as a core point. This includes the point itself."""

NB_CENTR = 500
""" Number of centroids in x and y axes. """

WIGGLE = 0.05
"""max distance by which onset of fire can be wiggled. 5 km"""

PROP_PROPA = 0.50
"""Probability of fire propagation.
    For now, arbitrary value"""

class BushFire(Hazard):
    """Contains bush fire events."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)

    def set_bush_fire(self, csv_firms, centroids, description=''):
        # Identify separate events:
        # 1 event = 1 fire up to its maximal extension
        # Firms data clustered by consecutive dates
        # Then pixel clustered geographically
        # Then event_id defined for each pixel
        """ Fill in the hazard file

        Parameters:
            csv_firms: csv file of the FIRMS data
                (obtained here:https://firms.modaps.eosdis.nasa.gov/download/)
            centroids (Centroids, optional): Centroids where to model BF.
                Default: global centroids.
            description (str, optional): description of the events

        Raises:
            ValueError

        """
        LOGGER.info('Setting up Hazard file.')

        firms, description = self._read_firms_csv(csv_firms)
        # Remove low confidence data
        firms = self._clean_firms_csv(firms)
        # Add cons_id
        firms = self._firms_cons_days(firms)
        # Add clus_id and event_id
        firms = self._firms_clustering(firms)
        firms = self._firms_event(firms)
        # Create centroids
        centroids = self._centroids_creation(firms)
        # Fill in brightness matrix
        brightness, _, _ = self._calc_brightness(firms, centroids)

        self.tag = TagHazard(HAZ_TYPE, csv_firms, description)
        self.units = 'K' # Kelvin units brightness
        self.centroids = centroids
        # Following values are defined for each event
        self.event_id = np.array(np.unique(firms['event_id'].values))
        ymax = datetime.strptime(firms.at[firms.index[-1], 'acq_date'], "%Y-%m-%d")
        ymin = datetime.strptime(firms.at[firms.index[0], 'acq_date'], "%Y-%m-%d")
        nb_years = abs((ymax - ymin).days)/365
        freq = nb_years/(max(firms['event_id'].values))
        self.frequency = np.tile(freq, len(np.unique(firms['event_id'])))
        self.event_name = np.array(np.unique(firms['event_id']))
        event_uni, _ = np.unique(firms['event_id'].values, return_inverse=True, axis=0)
        date = np.zeros(len(np.unique(firms['event_id'].values)))
        for ev_id, event in enumerate(event_uni):
            temp = firms.loc[firms['event_id'] == event, 'datenum']
            temp = temp.reset_index()
            date[ev_id] = temp.at[0, 'datenum']
        self.date = date
        self.orig = np.tile(True, len(np.unique(firms['event_id'])))

        # Following values are defined for each event and centroid
        self.intensity = brightness
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    # Historical event (from FIRMS data)
    @staticmethod
    def _read_firms_csv(csv_firms, description=''):
        """Read csv files from FIRMS data.

        Parameters:
            csv_firms: csv file of the FIRMS data
            description (str, optional): description of the events

        Returns:
            firms, csv_firms, description
        """
        # Open and read the file
        firms = pd.read_csv(csv_firms)

        for index, acq_date in enumerate(firms['acq_date'].values):
            datenum = datetime.strptime(acq_date, '%Y-%M-%d').toordinal()
            firms.at[index, 'datenum'] = datenum
        return firms, description

    @staticmethod
    def _clean_firms_csv(firms):
        """Optional - Remove low confidence data from firms:
            1 - MODIS: remove data where confidence values are lower than CLEAN_THRES
            2 - VIIRS: remove data where confidence values are set to low 'l'
            (keeps nominal and high values)

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """
        # Check for the type of instrument (MODIS vs VIIRS)
        # Remove data with low confidence interval
        temp = pd.DataFrame()
        if 'instrument' in firms.columns:

            if firms.instrument.any() == 'MODIS' or firms.instrument.any() == 'VIIRS':
                firms_modis = firms.drop(firms[firms.instrument == 'VIIRS'].index)
                firms_modis.confidence = np.array(
                    list(map(int, firms_modis.confidence.values.tolist())))
                firms_modis = firms_modis.drop(firms_modis[firms_modis.confidence < CLEAN_THRES].index)
                temp = firms_modis
                firms_viirs = firms.drop(firms[firms.instrument == 'MODIS'].index)
                if firms_viirs.size:
                    firms_viirs = firms_viirs.drop(firms_viirs[firms_viirs.confidence == 'l'].index)
                    firms_viirs = firms_viirs.rename(columns={'bright_ti4':'brightness'})
                    temp = temp.append(firms_viirs)

            firms = temp

        return firms

    @staticmethod
    def _firms_cons_days(firms):
        """Compute clusters of consecutive days.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Computing clusters of consecutive days.')
        # Clustering by consecutive dates
        # Order FIRMS data per ascending acq_date order
        firms = firms.sort_values('acq_date')
        firms = firms.reset_index()
        cons_id = np.zeros(len(firms['acq_date'].values))
        firms['cons_id'] = pd.Series(cons_id)
        for index, _ in list(map(tuple, firms['acq_date'].items()))[1:]:
            day_2 = datetime.strptime(firms.at[index, 'acq_date'], "%Y-%m-%d")
            day_1 = datetime.strptime(firms.at[(index-1), 'acq_date'], "%Y-%m-%d")
            if abs((day_2 - day_1).days) > 1:
                firms.at[index, 'cons_id'] = firms.at[(index-1), 'cons_id']+1
            else:
                firms.at[index, 'cons_id'] = firms.at[(index-1), 'cons_id']
        print(max(firms['cons_id'].values))
        return firms


#    @staticmethod
    def _firms_clustering(self, firms):
        """Compute geographic clusters and sort firms with ascending clus_id
        for each cons_id.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Computing geographic clusters in consecutive events.')
        # Creation of an identifier for geographical clustering
        cluster_id = np.zeros((firms.index.size,))-1
        for cons_id in np.unique(firms['cons_id'].values):
            temp = np.argwhere(firms['cons_id'].values == cons_id).reshape(-1,)
            lat_lon = firms.loc[temp, ['latitude', 'longitude']].values
            lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
            cluster_uni = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(lat_lon_uni).labels_
            cluster_cpy = cluster_uni[lat_lon_cpy]
            cluster_id[temp] = cluster_cpy
        firms['clus_id'] = pd.Series(cluster_id)
        print(max(firms['clus_id'].values))

        LOGGER.info('Sorting of firms data')
        # Re-order FIRMS file to have clus_id in ascending order inside each cons_id
        firms_sort = pd.DataFrame()
        cons_id_uni = np.unique(firms['cons_id'].values)
        for _, cons_id in enumerate(cons_id_uni):
            firms_sort = firms_sort.append(self._firms_clus(firms, cons_id))

#        cons_ev = int(max(firms['cons_id'].values))
#        chunksize = min(cons_ev, 10000)
#        firms_clus = Pool().map(self._firms_clus,
#                                 itertools.repeat(firms, cons_ev),
#                                 np.unique(firms['cons_id'].values),
#                                 chunksize=chunksize)
#        firms_sort = pd.DataFrame()
#        for firm in firms_clus:
#            firms_sort = firms_sort.append(firms_clus)

        firms = firms_sort
        firms = firms.reset_index()
        return firms

    @staticmethod
    def _firms_clus(firms, cons_id):
        """ For a given 'cons_id', perform geographical clustering
        and sort resulting firms data with 'clus_id' in ascending order.

        Parameters:
            firms (dataframe)
            cons_id (int): id of clusters of consecutive days

        Returns:
            firms_clus (dataframe): dataframe for the given cons_id, with one
            additional column (clus_id) and sorted with clus_id in ascending order
        """
#        firms_clus = []
        temp = np.argwhere(firms['cons_id'].values == cons_id).reshape(-1,)
        firms_clus = firms.loc[temp, :]
#        cluster_id = np.zeros((firms.index.size,))-1
#        lat_lon = firms_clus.loc[temp, ['latitude', 'longitude']].values
#        lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
#        cluster_uni = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(lat_lon_uni).labels_
#        cluster_cpy = cluster_uni[lat_lon_cpy]
#        cluster_id[temp] = cluster_cpy
#        firms_clus['clus_id'] = pd.Series(cluster_id)
        firms_clus = firms_clus.sort_values('clus_id')
#        firms_clus = firms_clus.append(firms_cons_id)

        return firms_clus

    @staticmethod
    def _firms_event(firms):
        """Creation of event_id for each dataset point.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Creation of event_id.')
        # New event = change in cons_id and clus_id
        event_id = np.zeros((firms.index.size,))+1
        firms['event_id'] = pd.Series(event_id)
        for index, _ in list(map(tuple, firms['clus_id'].items()))[1:]:
            if ((firms.at[index, 'clus_id'] - firms.at[index-1, 'clus_id']) == 0) \
            & ((firms.at[index, 'cons_id'] - firms.at[index-1, 'cons_id']) == 0):
                firms.at[index, 'event_id'] = int(firms.at[index-1, 'event_id'])
            else:
                firms.at[index, 'event_id'] = int(firms.at[index-1, 'event_id']+1)
        print(max(firms['event_id'].values))
        return firms

    @staticmethod
    def _event_per_year(firms, year):
        """Compute the number of fire events per (calendar) year.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            year (int): selected year

        Returns:
            event_nb (int)
        """
        firms['year'] = pd.DatetimeIndex(firms['acq_date']).year

        temp = 0
        temp = np.argwhere(firms['year'].values == year).reshape(-1,)
        firms_temp = firms.loc[temp, :]
        event_nb = np.max(firms_temp['event_id'].values) - np.min(firms_temp['event_id'].values) +1
        print(year, event_nb)
        return event_nb

    @staticmethod
    def _centroids_creation(firms):
        """ Compute centroids for the firms dataset.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            centroids
        """

        LOGGER.info('Computing centroids.')
        min_lat = firms['latitude'].min()
        max_lat = firms['latitude'].max()
        min_lon = firms['longitude'].min()
        max_lon = firms['longitude'].max()
        centroids = Centroids()
        res_lat = abs(min_lat - max_lat)/NB_CENTR
        res_lon = abs(min_lon - max_lon)/NB_CENTR
        centroids.coord = (np.mgrid[min_lat : max_lat : complex(0, NB_CENTR),
                                    min_lon : max_lon : complex(0, NB_CENTR)]).\
                           reshape(2, NB_CENTR*NB_CENTR).transpose()
#        centroids.coord = (np.mgrid[min_lat + res_lat/2 : max_lat - res_lat/2 : complex(0, NB_CENTR),
#                                    min_lon + res_lon/2 : max_lon - res_lon/2 : complex(0, NB_CENTR)]).\
#                           reshape(2, NB_CENTR*NB_CENTR).transpose()
        centroids.resolution = ((abs(min_lat - max_lat)/NB_CENTR),
                                (abs(min_lon - max_lon)/NB_CENTR))
        print(centroids.resolution)
        centroids.id = np.arange(len(centroids.coord))
        centroids.set_area_per_centroid()
        centroids.set_on_land()
        return centroids

    def _calc_brightness(self, firms, centroids):
        """ Fill in the intensity matrix with, for each event,
        the maximum brightness at each centroid.
        Parallel process.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            centroids (Centroids): centroids for the dataset

        Returns:
            brightness (csr matrix): matrix with, for each event, the maximum recorded
                brightness at each centroids
            num_centr (int): number of centroids
            latlon (np array): index of the closest (lat, lon) points of firms dataframe
                for each centroids
        """

        LOGGER.info('Identifying closest (lat,lon) points from firms dataframe for each centroid.')
        # Identifies the unique (lat,lon) points of the firms dataframe -> lat_lon_uni
        # Set the same index value for each duplicate (lat,lon) points -> lat_lon_cpy
        # This firms['index'] = lat_lon_cpy is used after to fill up the intensity matrix
        lat_lon = np.array(firms[['latitude', 'longitude']].values)
        lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
        firms['index'] = lat_lon_cpy

        # Index of the closest (lat,lon) points of firms dataframe for each centroids
        # If latlon = -1 -> no (lat,lon) points close enought (compare to threshold)
        # from the centroid
        # Thershold for interpolation, depends on instrument used (MODIS or VIIRS)
        if 'instrument' in firms.columns:
            if firms.instrument.all() == 'MODIS':
                latlon = interpol_index(lat_lon_uni, centroids.coord, threshold=THRESH_INTERP_MODIS)
            else:
                latlon = interpol_index(lat_lon_uni, centroids.coord, threshold=THRESH_INTERP_VIIRS)
        else:
            latlon = interpol_index(lat_lon_uni, centroids.coord, threshold=THRESH_INTERP)

        LOGGER.info('Computing brightness matrix.')
        num_ev = int(max(firms['event_id'].values))
        num_centr = len(centroids.id)

        LOGGER.debug('Filling up the matrix.')
        # Matrix intensity: events x centroids
        # Creation a of temporary firms dataframe (temp_firms)
        # containing index and brightness for one event
        # To take into account (lat,lon) points with redudndant coordinates:
        # identification of the unique values of temp_firms['index'] -> index_uni
        # set the same index value for each duplicate (lat,lon) points -> index_cpy
        # For one event, if more than one points of firms dataframe have the same coordinates
        # take the maximum brightness value of these points (maximal damages).
        # Fill the matrix

        bright_list = []
        # Next two lines, to keep for debugging
#        for _, ev_id in enumerate(np.unique(firms['event_id'].values)):
#            bright_list.append(self._brightness_one_event(firms, ev_id, num_centr, latlon))

        chunksize = min(num_ev, 1000)
        bright_list = Pool().map(self._brightness_one_event,
                                 itertools.repeat(firms, num_ev),
                                 np.unique(firms['event_id'].values),
                                 itertools.repeat(num_centr, num_ev),
                                 itertools.repeat(latlon, num_ev),
                                 chunksize=chunksize)

        brightness = sparse.lil_matrix(np.zeros((num_ev, num_centr)))
        for ev_idx, bright_ev in zip(range(num_ev), bright_list):
            brightness[ev_idx, :] = bright_ev[:]
        brightness = brightness.tocsr()

        return brightness, num_centr, latlon


    @staticmethod
    def _brightness_one_event(firms, ev_id, num_centr, latlon):
        """ For a given event,fill in an intensity np.array with the maximum brightness
        at each centroid.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            ev_id (int): id of the selected event
            num_centr (int): total number of centroids
            latlon (np.array): index of the closest (lat,lon) points of firms dataframe
            for each centroids

        Returns:
            brightness_ev (np.array): maximum brightness at each centroids

        """
        temp_firms = firms.loc[firms['event_id'] == ev_id, ['index', 'brightness']]
        index_uni, index_cpy = np.unique(temp_firms['index'].values,
                                         return_inverse=True, axis=0)
        bright_one_event = np.zeros(index_uni.size)
        for index_idx, index in enumerate(index_uni):
            bright_one_event[index_idx] = np.max(temp_firms['brightness']\
                            [index_uni[index_cpy] == index])

        brightness_ev = sparse.lil_matrix(np.zeros((num_centr)))
        for idx, val in enumerate(index_uni):
            centr_idx = np.argwhere(latlon == val)
            brightness_ev[0, centr_idx] = bright_one_event[idx]

        return brightness_ev

    def _area_one_year(self, firms, centroids, brightness, year):
        """ Calculate the area burn over one (calendar) year.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            centroids (Centroids): Centroids instance. Use global
                centroids if not provided.
            brightness (csr matrix): intensity matrix (event vs centroids)
            year: year for which to calculate the area

        Returns:
            area_one_year (km2): area burnt over one year
        """
        # Event_id of the selected year
        firms['year'] = pd.DatetimeIndex(firms['acq_date']).year

        temp = 0
        temp = np.argwhere(firms['year'].values == year).reshape(-1,)
        event_id_year = firms.loc[temp, ['event_id']].values
        event_id_year = np.unique(event_id_year)

        if event_id_year.size == 0:
            LOGGER.error('No event for this year')
            raise ValueError

        # Area calculation
        area_one_year = 0
        area_one_event = []
        for _, event_id in enumerate(event_id_year):
            area_one_event.append(self._area_one_event(centroids, brightness, event_id))
        area_one_year = area_one_year + sum(area_one_event)
        print('area_one_year:', year, area_one_year)
        return area_one_year

    @staticmethod
    def _area_one_event(centroids, brightness, ev_id):
        """ Calculate the area of one event.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            centroids (Centroids): Centroids instance. Use global
                centroids if not provided.
            brightness (csr matrix): intensity matrix (event vs centroids)
            year: year for which to calculate the area


        Returns:
            area_one_event (km2): area burnt in one event
        """

        col = list(brightness[(ev_id-1),].indices)
        area_one_event = 0
        for _, colv in enumerate(col):
            area_temp = centroids.area_per_centroid[colv]
            area_one_event = area_one_event + area_temp
#        print('area_one_event:', event_id, area_one_event)
        return area_one_event
