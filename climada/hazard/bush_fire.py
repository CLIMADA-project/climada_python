"""Define BushFire class."""

__all__ = ['BushFire']

import logging
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime
from sklearn.cluster import DBSCAN

from climada.hazard.centroids.base import Centroids
from climada.util.interpolation import interpol_index
from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from random import randint

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'BF'
""" Hazard type acronym for Bush Fire """

THRESH_INTERP = 2.0
""" Distance threshold in km over which no neighbor will be found """

EPS = 0.5
""" maximum distance (in °) between two samples for them to be considered
as in the same neighborhood"""

MIN_SAMPLES = 1
"""number of samples (or total weight) in a neighborhood for a point to
be considered as a core point. This includes the point itself."""

RESOL_CENTR = 500
""" Number of centroids in x and y axes. """

class BushFire(Hazard):
    """Contains bush fire events."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)

    def set_bush_fire(self, csv_firms, centroids, description=''):
        #Identify separate events:
        #1 event = 1 fire up to its maximal extension
        #Firms data clustered by consecutive dates
        #Then pixel clustered geographically
        #Then event_id defined for each pixel
        """ Fill in the hazard file

        Parameters:
            csv_firms: csv file of the FIRMS data (obtained here:https://firms.modaps.eosdis.nasa.gov/download/)
            centroids (Centroids, optional): Centroids where to model BF.
                Default: global centroids.
            description (str, optional): description of the events

        Raises:
            ValueError

        """
        LOGGER.info('Setting up Hazard file.')

        firms, csv_firms, description = self._read_firms_csv(csv_firms)
        #firms, description = self._read_firms_synth()
        # add cons_id
        firms = self._firms_cons_days(firms)
        # add clus_id and event_id
        firms = self._firms_clustering(firms)
        firms = self._firms_event(firms)
        centroids = self._centroids_creation(firms)
        brightness = self._calc_brightness(firms, centroids)

        self.tag = TagHazard(HAZ_TYPE, csv_firms, description)
        self.units = 'K' # Kelvin units brightness
        self.centroids = centroids
        # following values are defined for each event
        self.event_id = np.array(np.unique(firms['event_id'].values))
        nb_years = (max(firms['datenum'].values) - min(firms['datenum'].values))/365
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

        # following values are defined for each event and centroid
        self.intensity = brightness
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)

    #Historical event (from FIRMS data)
    @staticmethod
    def _read_firms_csv(csv_firms, description=''):
        """Read csv files from FIRMS data.

        Parameters:
            csv_firms: csv file of the FIRMS data
            description (str, optional): description of the events

        Returns:
            firms, csv_firms, description
        """
        # open and read the file
        firms = pd.read_csv(csv_firms)

        for index, acq_date in enumerate(firms['acq_date'].values):
            datenum = datetime.strptime(acq_date, '%Y-%M-%d').toordinal()
            firms.at[index, 'datenum'] = datenum
        return firms, csv_firms, description

    @staticmethod
    def _firms_cons_days(firms):
        """Compute clusters of consecutive days.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Computing clusters of consecutive days.')
        #Clustering by consecutive dates
        cons_id = [0]
        firms['cons_id'] = pd.Series(cons_id)
        for index, val in list(map(tuple, firms['datenum'].items()))[1:]:
            if abs(firms.at[index, 'datenum'] - firms.at[(index-1), 'datenum']) > 1:
                firms.at[index, 'cons_id'] = firms.at[(index-1), 'cons_id']+1
            else:
                firms.at[index, 'cons_id'] = firms.at[(index-1), 'cons_id']
        print(max(firms['cons_id'].values))
        return firms


    @staticmethod
    def _firms_clustering(firms):
        """Compute geographic clusters and sort firms with ascending clus_id for each cons_id.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Computing geographic clusters in consecutive events.')
        #Creation of an identifier for geographical clustering
        cluster_id = np.zeros((firms.index.size,))-1
        for cons_id in np.unique(firms['cons_id'].values):
            cc = np.argwhere(firms['cons_id'].values == cons_id).reshape(-1,)
            lat_lon = firms.loc[cc, ['latitude', 'longitude']].values
            lat_lon_uni, lat_lon_cpy = np.unique(lat_lon, return_inverse=True, axis=0)
            cluster_uni = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(lat_lon_uni).labels_
            cluster_cpy = cluster_uni[lat_lon_cpy]
            cluster_id[cc] = cluster_cpy
        firms['clus_id'] = pd.Series(cluster_id)
        print(max(firms['clus_id'].values))

        LOGGER.info('Sorting of firms data')
        #Re-order FIRMS file to have clus_id in ascending order inside each cons_id
        cons_id_gp = firms.groupby('cons_id')
        firms_sort = cons_id_gp.get_group(0).sort_values('clus_id')
        for val, group in cons_id_gp:
            if val > 0:
                sort_clus = cons_id_gp.get_group(val).sort_values('clus_id')
                firms_sort = firms_sort.append(sort_clus)
        firms = firms_sort
        firms = firms.reset_index()
        return firms

    @staticmethod
    def _firms_event(firms):
        """Creation of event_id for each dataset point.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data

        Returns:
            firms
        """

        LOGGER.info('Creation of event_id.')
        #New event = change in cons_id and clus_id
        event_id = np.zeros((firms.index.size,))+1
        firms['event_id'] = pd.Series(event_id)
        for index, val in list(map(tuple, firms['clus_id'].items()))[1:]:
            if ((firms.at[index, 'clus_id'] - firms.at[index-1, 'clus_id']) == 0) & ((firms.at[index, 'cons_id'] - firms.at[index-1, 'cons_id']) == 0):
                firms.at[index, 'event_id'] = int(firms.at[index-1, 'event_id'])
            else:
                firms.at[index, 'event_id'] = int(firms.at[index-1, 'event_id']+1)
        print(max(firms['event_id'].values))
        return firms

    @staticmethod
    def _centroids_creation(firms):
        """Compute centroids for the firms dataset.

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
        centroids.coord = (np.mgrid[min_lat : max_lat : complex(0, RESOL_CENTR),
                                    min_lon : max_lon : complex(0, RESOL_CENTR)]).\
                           reshape(2, RESOL_CENTR*RESOL_CENTR).transpose()
        centroids.id = np.arange(len(centroids.coord)) + 1
        return centroids

    @staticmethod
    def _calc_brightness(firms, centroids):
        """Fill in the intensity matrix with the maximum brightness at each centroid.

        Parameters:
            firms (dataframe): dataset obtained from FIRMS data
            centroids (Centroids): centroids for the dataset

        Returns:
            brightness (matrix)
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
        latlon = interpol_index(lat_lon_uni, centroids.coord, threshold=THRESH_INTERP)

        LOGGER.info('Computing brightness matrix.')
        num_events = int(max(firms['event_id'].values))
        num_centroids = int(len(centroids.id))
        brightness = sparse.lil_matrix(np.zeros((num_events, num_centroids)))

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

        for ev_idx, ev_id in enumerate(np.unique(firms['event_id'].values)):
            temp_firms = firms.loc[firms['event_id'] == ev_id, ['index', 'brightness']]
            index_uni, index_cpy = np.unique(temp_firms['index'].values, return_inverse=True, axis=0)
            bright = np.zeros(index_uni.size)
            for index_idx, index in enumerate(index_uni):
                bright[index_idx] = np.max(temp_firms['brightness'][index_uni[index_cpy] == index])
            print(temp_firms)
            for idx, val in enumerate(index_uni):
                centr_idx = np.argwhere(latlon == val)
                brightness[ev_idx, centr_idx] = bright[idx]
        brightness = brightness.tocsr()
        return brightness

    #Probabilistic event
    @staticmethod
    def _random_ignition(centroids):
        """Select randomly a centroid as ignition starting point.

        Parameters:
            centroids (Centroids): Centroids instance. Use global
                centroids if not provided.


        Returns: ignition_start (int): centroids.id of the centroid randomly selected
            as ignition starting point.

        """
        ignition_start = randint(0, centroids.id.size)
        ignition_lat = centroids.lat[centroids.id == ignition_start]
        ignition_lon = centroids.lon[centroids.id == ignition_start]
        

        return ignition_start, ignition_lat, ignition_lon

    @staticmethod
    def _propagation (centroids, ignition_start, ignition_lon, ignition_lat):
        """Propagate the fire from the ignition point with a cellular automat.

        Parameters:
            centroids (Centroids): Centroids instance.
            ignition_start (int): id of the centroid randomly selected as
                ignition starting point.

        Returns:


        """

        fire = centroids.id.reshape(-1,1)
        fire = pd.DataFrame.from_dict(fire)
        fire.columns = ['centr_id']
        fire['val'] = np.zeros(centroids.id.size)

        #step 0: change in val only for the ignition starting point
        for raw, centr_id in enumerate(fire['centr_id'].values):
            if fire.at[raw, 'centr_id'] == ignition_start:
                fire.at[raw, 'val'] = 1

        #next steps:
        lat_uni = list(np.unique(centroids.lat))
        dlat = []
        lon_uni = np.unique(centroids.lon)
        dlon = []

        for x in iter(lat_uni):
            d_lat = x - next(iter(lat_uni))
            dlat.append(d_lat)
        for x in iter(lon_uni):
            d_lon = x - next(iter(lon_uni))
            dlon.append(d_lon)
        dlat = dlat[1]
        dlon = dlon[1]






