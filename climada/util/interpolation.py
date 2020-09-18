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

Define interpolation functions using different metrics.
"""

__all__ = ['interpol_index',
           'dist_sqr_approx',
           'DIST_DEF',
           'METHOD']

import logging
import numpy as np
from numba import jit

from sklearn.neighbors import BallTree
from climada.util.constants import ONE_LAT_KM, EARTH_RADIUS_KM

LOGGER = logging.getLogger(__name__)

DIST_DEF = ['approx', 'haversine']
"""Distances"""

METHOD = ['NN']
"""Interpolation methods"""

THRESHOLD = 100
"""Distance threshold in km. Nearest neighbors with greater distances are
not considered."""

@jit(nopython=True, parallel=True)
def dist_approx(lats1, lons1, cos_lats1, lats2, lons2):
    """Compute equirectangular approximation distance in km."""
    d_lon = lons1 - lons2
    d_lat = lats1 - lats2
    return np.sqrt(d_lon * d_lon * cos_lats1 * cos_lats1 + d_lat * d_lat) * ONE_LAT_KM

@jit(nopython=True, parallel=True)
def dist_sqr_approx(lats1, lons1, cos_lats1, lats2, lons2):
    """Compute squared equirectangular approximation distance. Values need
    to be sqrt and multiplicated by ONE_LAT_KM to obtain distance in km."""
    d_lon = lons1 - lons2
    d_lat = lats1 - lats2
    return d_lon * d_lon * cos_lats1 * cos_lats1 + d_lat * d_lat

def interpol_index(centroids, coordinates, method=METHOD[0],
                   distance=DIST_DEF[1], threshold=THRESHOLD):
    """Returns for each coordinate the centroids indexes used for
    interpolation.

    Parameters:
        centroids (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        coordinates (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        method (str, optional): interpolation method to use. NN default.
        distance (str, optional): distance to use. Haversine default
        threshold (float): distance threshold in km over which no neighbor will
            be found. Those are assigned with a -1 index

    Returns:
        numpy array with so many rows as coordinates containing the
            centroids indexes
    """
    if (method == METHOD[0]) & (distance == DIST_DEF[0]):
        # Compute for each coordinate the closest centroid
        interp = index_nn_aprox(centroids, coordinates, threshold)
    elif (method == METHOD[0]) & (distance == DIST_DEF[1]):
        # Compute the nearest centroid for each coordinate using the
        # haversine formula. This is done with a Ball tree.
        interp = index_nn_haversine(centroids, coordinates, threshold)
    else:
        LOGGER.error('Interpolation using %s with distance %s is not '
                     'supported.', method, distance)
        interp = np.array([])
    return interp

def index_nn_aprox(centroids, coordinates, threshold=THRESHOLD):
    """Compute the nearest centroid for each coordinate using the
    euclidian distance d = ((dlon)cos(lat))^2+(dlat)^2. For distant points
    (e.g. more than 100km apart) use the haversine distance.

    Parameters:
        centroids (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        coordinates (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        threshold (float): distance threshold in km over which no neighbor will
            be found. Those are assigned with a -1 index

    Returns:
        array with so many rows as coordinates containing the centroids
            indexes
    """

    # Compute only for the unique coordinates. Copy the results for the
    # not unique coordinates
    _, idx, inv = np.unique(coordinates, axis=0, return_index=True,
                            return_inverse=True)
    # Compute cos(lat) for all centroids
    centr_cos_lat = np.cos(np.radians(centroids[:, 0]))
    assigned = np.zeros(coordinates.shape[0], int)
    num_warn = 0
    for icoord, iidx in enumerate(idx):
        dist = dist_sqr_approx(centroids[:, 0], centroids[:, 1],
                               centr_cos_lat, coordinates[iidx, 0],
                               coordinates[iidx, 1])
        min_idx = dist.argmin()
        # Raise a warning if the minimum distance is greater than the
        # threshold and set an unvalid index -1
        if np.sqrt(dist.min()) * ONE_LAT_KM > threshold:
            num_warn += 1
            min_idx = -1

        # Assign found centroid index to all the same coordinates
        assigned[inv == icoord] = min_idx

    if num_warn:
        LOGGER.warning('Distance to closest centroid is greater than %s'
                       'km for %s coordinates.', threshold, num_warn)

    return assigned

def index_nn_haversine(centroids, coordinates, threshold=THRESHOLD):
    """Compute the neareast centroid for each coordinate using a Ball
    tree with haversine distance.

    Parameters:
        centroids (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        coordinates (2d array): First column contains latitude, second
            column contains longitude. Each row is a geographic point
        threshold (float): distance threshold in km over which no neighbor will
            be found. Those are assigned with a -1 index

    Returns:
        array with so many rows as coordinates containing the centroids
            indexes
    """
    # Construct tree from centroids
    tree = BallTree(np.radians(centroids), metric='haversine')
    # Select unique exposures coordinates
    _, idx, inv = np.unique(coordinates, axis=0, return_index=True,
                            return_inverse=True)

    # query the k closest points of the n_points using dual tree
    dist, assigned = tree.query(np.radians(coordinates[idx]), k=1,
                                return_distance=True, dualtree=True,
                                breadth_first=False)

    # Raise a warning if the minimum distance is greater than the
    # threshold and set an unvalid index -1
    num_warn = np.sum(dist * EARTH_RADIUS_KM > threshold)
    if num_warn:
        LOGGER.warning('Distance to closest centroid is greater than %s'
                       'km for %s coordinates.', threshold, num_warn)
        assigned[dist * EARTH_RADIUS_KM > threshold] = -1

    # Copy result to all exposures and return value
    return np.squeeze(assigned[inv])
