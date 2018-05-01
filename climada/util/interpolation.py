"""
Define interpolation functions using different metrics.
"""

__all__ = ['interpol_index',
           'dist_sqr_approx',
           'DIST_DEF',
           'METHOD']

import logging
import numpy as np

from sklearn.neighbors import BallTree
from climada.util.constants import ONE_LAT_KM, EARTH_RADIUS

LOGGER = logging.getLogger(__name__)

DIST_DEF = ['approx', 'haversine']
""" Distances """

METHOD = ['NN']
""" Interpolation methods """

THRESHOLD = 100
""" Distance threshold in km. Nearest neighbors with greater distances are
not considered. """

def dist_sqr_approx(lats1, lons1, cos_lats1, lats2, lons2):
    """ Compute approximated squared distance between two points."""
    return ((lons1 - lons2) * cos_lats1)**2 + (lats1 - lats2)**2

def interpol_index(centroids, coordinates, method=METHOD[0], \
                   distance=DIST_DEF[1]):
    """ Returns for each coordinate the centroids indexes used for
    interpolation

    Parameters:
        centroids (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        coordinates (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        method (str, optional): interpolation method to use. NN default.
        distance (str, optional): distance to use. Haversine default.

    Returns:
        numpy array with so many rows as coordinates containing the
            centroids indexes
    """
    if (method == METHOD[0]) & (distance == DIST_DEF[0]):
        # Compute for each coordinate the closest centroid
        interp = index_nn_aprox(centroids, coordinates)
    elif (method == METHOD[0]) & (distance == DIST_DEF[1]):
        # Compute the nearest centroid for each coordinate using the
        # haversine formula. This is done with a Ball tree.
        interp = index_nn_haversine(centroids, coordinates)
    else:
        LOGGER.error('Interpolation using %s with distance %s is not '\
                     'supported.', method, distance)
        interp = np.array([])
    return interp

def index_nn_aprox(centroids, coordinates):
    """ Compute the nearest centroid for each coordinate using the
    euclidian distance d = ((dlon)cos(lat))^2+(dlat)^2. For distant points
    (e.g. more than 100km apart) use the haversine distance.

    Parameters:
        centroids (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        coordinates (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point

    Returns:
        array with so many rows as coordinates containing the centroids
            indexes
    """

    # Compute only for the unique coordinates. Copy the results for the
    # not unique coordinates
    _, idx, inv = np.unique(coordinates, axis=0, return_index=True,
                            return_inverse=True)
    # Compute cos(lat) for all centroids
    centr_cos_lat = np.cos(centroids[:, 0] / 180 * np.pi)
    assigned = np.zeros(coordinates.shape[0])
    for icoord, iidx in enumerate(idx):
        dist = dist_sqr_approx(centroids[:, 0], centroids[:, 1], \
                                centr_cos_lat, coordinates[iidx, 0], \
                                coordinates[iidx, 1])
        min_idx = dist.argmin()
        # Raise a warning if the minimum distance is greater than the
        # threshold and set an unvalid index -1
        if np.sqrt(dist.min()) * ONE_LAT_KM > THRESHOLD:
            LOGGER.warning('Distance to closest centroid for coordinate ' + \
                '(%s, %s) is %s.', coordinates[iidx][0], coordinates[iidx][1],\
                np.sqrt(dist.min()) * ONE_LAT_KM)
            min_idx = -1

        # Assign found centroid index to all the same coordinates
        assigned[inv == icoord] = min_idx

    return assigned

def index_nn_haversine(centroids, coordinates):
    """ Compute the neareast centroid for each coordinate using a Ball
    tree with haversine distance.

    Parameters:
        centroids (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        coordinates (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point

    Returns:
        array with so many rows as coordinates containing the centroids
            indexes
    """
    # Construct tree from centroids
    tree = BallTree(centroids/180*np.pi, metric='haversine')
    # Select unique exposures coordinates
    _, idx, inv = np.unique(coordinates, axis=0, return_index=True,
                            return_inverse=True)

    # query the k closest points of the n_points using dual tree
    dist, assigned = tree.query(coordinates[idx]/180*np.pi, k=1, \
                                return_distance=True, dualtree=True, \
                                breadth_first=False)

    # Raise a warning if the minimum distance is greater than the
    # threshold and set an unvalid index -1
    num_warn = np.sum(dist*EARTH_RADIUS > THRESHOLD)
    if num_warn > 0:
        LOGGER.warning('Distance to closest centroid is greater than %s' \
            'km for %s coordinates.', THRESHOLD, num_warn)
        assigned[dist*EARTH_RADIUS > THRESHOLD] = -1

    # Copy result to all exposures and return value
    return np.squeeze(assigned[inv])
