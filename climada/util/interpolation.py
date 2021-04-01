"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define interpolation functions using different metrics.
"""

__all__ = ['interpol_index',
           'dist_sqr_approx',
           'DIST_DEF',
           'METHOD']

import geopandas as gpd
import logging
import math
import numpy as np
from numba import jit
import pandas as pd

from shapely.geometry import MultiPoint
from sklearn.neighbors import BallTree

from climada.util.constants import ONE_LAT_KM, EARTH_RADIUS_KM
from climada.util.coordinates import dist_great_circle_allgeoms, coord_on_land, metres_to_degrees, pts_to_raster_meta, raster_to_meshgrid

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


def interpolate_lines(gdf_lines, point_dist=5):
    """ Convert a GeoDataframe with LineString geometries to 
    Point geometries, where Points are placed at a specified distance along the
    original LineString 
    
    Parameters
    ----------
    gdf_lines (gpd.GeoDataframe or str) : Geodataframe or filepath from which
        to read the GeoDataframe
    point_dist (float) : Distance in metres apart from which the generated  
        Points should be placed.
    
    Returns
    -------
    gdf_points with individual Point per row, retaining all other column infos
        belonging to its corresponding line (incl. line length of original geom.
        and multi-index referring to original indexing)
        
    See also
    --------
    * coordinates.dist_great_circle_allgeoms()
    * entity.exposure.base.point_exposure_from_lines()
    """

    if not isinstance(gdf_lines, gpd.GeoDataFrame):
        gdf_lines = gpd.read_file(gdf_lines)
        
    gdf_points = gdf_lines.copy()
    gdf_points['length'] = np.ones(len(gdf_points))*point_dist
    gdf_points['length_full'] = dist_great_circle_allgeoms(gdf_points)

    # split line lengths into relative fractions acc to point_dist (e.g. 0, 0.5, 1)
    gdf_points['distance_vector'] = gdf_points.apply(
        lambda row: np.linspace(0, 1, num=int(np.ceil(row.length_full/
                                                      row.length)+1)), axis=1)

    # create MultiPoints along the line for every position in distance_vector
    gdf_points['geometry'] = gdf_points.apply(
        lambda row: MultiPoint(
            [row.geometry.interpolate(dist, normalized=True) 
             for dist in row.distance_vector]), axis=1)
    
    # expand gdf from MultiPoint entries to single Points per row
    return gdf_points.explode().drop(['distance_vector', 'length_full'], axis=1)

def interpolate_polygons(gdf_poly, area_point):
    """For a GeoDataFrame with polygons, get equally distributed lat/lon pairs
    throughout the geometries, at a user-specified area distance
    
    Parameters
    ----------
    gdf_poly : (gpd.GeoDataFrame) with polygons to be interpolated
    area_point : area in m2 which one point should represent
    
    Returns
    -------
    (gpd.GeoDataFrame) with individual Point per row, retaining all other column infos
        belonging to its corresponding polygon
    """
    
    metre_dist = math.sqrt(area_point)
    
    gdf_poly['degree_dist'] = gdf_poly.apply(lambda row: metres_to_degrees(
        row.geometry.representative_point().x, 
        row.geometry.representative_point().y,
        metre_dist), axis=1)
    
    # get params to make an even grid with desired resolution 
    # over bounding boxes of polygons:
    trans = gdf_poly.apply(lambda row: pts_to_raster_meta(
        row.geometry.bounds, (row.degree_dist, -row.degree_dist)), axis=1)
    gdf_poly['trans'] = pd.DataFrame(trans.tolist(), index=trans.index).iloc[:,2]
   
    gdf_poly['width'] = np.floor(abs(gdf_poly.geometry.bounds.minx-
                                     gdf_poly.geometry.bounds.maxx)/
                                 gdf_poly['degree_dist'])
    gdf_poly['height'] = np.floor(abs(gdf_poly.geometry.bounds.miny-
                                      gdf_poly.geometry.bounds.maxy)/
                                  gdf_poly['degree_dist'])
    
    # make grid for each polygon-bbox
    points = gdf_poly.apply(lambda row: raster_to_meshgrid(
        row.trans, row.width, row.height), axis=1)
    points = pd.DataFrame(points.tolist(), columns=['lon', 'lat'])
    for axis in ['lon', 'lat']:
        points[axis] = points.apply(lambda row: row[axis].flatten(), axis=1)
    
    # filter only centroids in actual polygons
    for i, polygon in enumerate(gdf_poly.geometry):
        in_geom = coord_on_land(lat=points['lat'].iloc[i], 
                                lon=points['lon'].iloc[i],
                                land_geom=polygon)
        for axis in ['lon', 'lat']:
            points[axis].iloc[i] = points[axis].iloc[i][in_geom]
    
    points['geometry'] = points.apply(lambda row: 
                                      MultiPoint([(x, y) for x, y 
                                                  in zip(row.lon, row.lat)]),
                                      axis=1)
        
    points = pd.merge(points.drop(['lat', 'lon'], axis=1), 
                      gdf_poly.drop(['geometry', 'trans', 'width', 'height', 
                                     'degree_dist'], axis=1),
                      left_index=True, right_index=True)

    return gpd.GeoDataFrame(points).explode()


