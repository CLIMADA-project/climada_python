"""
Define Interpolator class.
"""

import warnings
import numpy as np
from climada.util.constants import ONE_LAT_KM, EARTH_RADIUS
from sklearn.neighbors import BallTree

class Interpolator(object):
    """ Interpolator class """

    dist_def = ['approx', 'haversine']
    method = ['NN']

    def __init__(self, threshold=100):
        """ Initialize class.
        INPUT:
            - threshold: threshold used to raise a warning when the minimum
            distance between two points exceeds it. Given in km
        """
        self.threshold = threshold

    def interpol_index(self, centroids, coordinates, method=method[0],
                       distance=dist_def[0]):
        """ Returns for each coordinate the centroids indexes used for
        interpolation """
        if (method == 'NN') & (distance == 'approx'):
            # Compute for each coordinate the closest centroid
            return self.index_nn_aprox(centroids, coordinates)
        elif (method == 'NN') & (distance == 'haversine'):
            # Compute the nearest centroid for each coordinate using the
            # haversine formula. This is done with a Ball tree.
            return self.index_nn_haversine(centroids, coordinates)
        else:
            # Raise error: method or distance not supported
            raise ValueError('Interpolation using ' + method +
                             ' with distance ' + distance +
                             ' is not supported.')

    def index_nn_aprox(self, centroids, coordinates):
        """ Compute the nearest centroid for each coordinate using the
        euclidian distance d = ((dlon)cos(lat))^2+(dlat)^2. For distant points
        (e.g. more than 5km apart) use the haversine distance.
        INPUT:
            - centroids: 2d numpy array. First column contains latitude,
            second column contains longitude. Each row is a geographic point.
            - coordinates: 2d numpy array. First column contains latitude,
            second column contains longitude. Each row is a coordinate.
        """

        # Compute only for the unique coordinates. Copy the results for the
        # not unique coordinates
        _, idx, inv = np.unique(coordinates, axis=0, return_index=True,
                                return_inverse=True)
        n_diff_coord = len(idx)
        # Compute cos(lat) for all centroids
        centr_cos_lat = np.cos(centroids[:, 0]/180*np.pi)
        assigned = np.zeros(coordinates.shape[0])
        for icoord in range(n_diff_coord):
            dist = ((centroids[:, 1] - coordinates[idx[icoord]][1]) * \
                    centr_cos_lat)**2 + \
                    (centroids[:, 0] - coordinates[idx[icoord]][0])**2
            min_idx = dist.argmin()
            # Raise a warning if the minimum distance is greater than the
            # threshold and set an unvalid index -1
            if np.sqrt(dist.min())*ONE_LAT_KM > self.threshold:
                warnings.warn('Distance to closest centroid for coordinate' +\
                             ' (' + str(coordinates[idx[icoord]][0]) + ',' + \
                             str(coordinates[idx[icoord]][1]) + ') is ' + \
                             str(np.sqrt(dist.min())*ONE_LAT_KM))
                min_idx = -1

            # Assign found centroid index to all the same coordinates
            assigned[inv == icoord] = min_idx

        return assigned


    def index_nn_haversine(self, centroids, coordinates):
        """ Compute the neareast centroid for each coordinate using a Ball tree
        with haversine distance
        INPUT:
            - centroids: 2d numpy array. First column contains latitude, second
            column contains longitude. Each row is a geographic point.
            - coordinates: 2d numpy array. First column contains latitude, second
            column contains longitude. Each row is a coordinate.
        """
        # Construct tree from centroids
        tree = BallTree(centroids/180*np.pi, metric='haversine')
        # Select unique exposures coordinates
        _, idx, inv = np.unique(coordinates, axis=0, return_index=True,
                                return_inverse=True)

        # query the k closest points of the n_points using dual tree
        dist, assigned = tree.query(coordinates[idx]/180*np.pi, k=1,
                                    return_distance=True, dualtree=True,
                                    breadth_first=False)

        # Raise a warning if the minimum distance is greater than the
        # threshold and set an unvalid index -1
        num_warn = np.sum(dist*EARTH_RADIUS > self.threshold)
        if num_warn > 0:
            warnings.warn('Distance to closest centroid is greater than ' +\
                          str(self.threshold) + ' for ' + str(num_warn) +\
                          'coordinates.')
            assigned[dist*EARTH_RADIUS > self.threshold] = -1

        # Copy result to all exposures and return value
        return assigned[inv].reshape(assigned[inv].shape[0],)
