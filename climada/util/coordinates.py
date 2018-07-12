"""
Define GridPoints class
"""

__all__ = ['GridPoints']

import os.path
import logging
import numpy as np
import shapefile
from cartopy.io import shapereader
import shapely.vectorized
import shapely.ops

from climada.util.interpolation import METHOD, DIST_DEF, interpol_index
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

GLOBE_COUNTRIES = "global_country_borders"
""" Name of the earth's country borders shape file generated in function
get_countries_geometry"""

class GridPoints(np.ndarray):
    """Define grid using 2d numpy array. Each row is a point. The first column
    is for latitudes and the second for longitudes (in degrees)."""
    def __new__(cls, input_array=None):
        if input_array is not None:
            obj = np.asarray(input_array).view(cls)
            obj.check()
        else:
            obj = np.empty((0, 2)).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def check(self):
        """Check shape. Repeated points are allowed"""
        if self.shape[1] != 2:
            LOGGER.error("GridPoints with wrong shape: %s != %s",
                         self.shape[1], 2)
            raise ValueError

    def resample(self, coord, method=METHOD[0], distance=DIST_DEF[1]):
        """ Input GridPoints are resampled to current grid by interpolating
        their values.

        Parameters:
            coord (2d array): First column contains latitude, second
                column contains longitude. Each row is a geographic point.
            method (str, optional): interpolation method. Default: nearest
                neighbor.
            distance (str, optional): metric to use. Default: haversine

        Returns:
            np.array
        """
        return interpol_index(self, coord, method, distance)

    def resample_agg_to_lower_res(self, coord):
        """ Input GridPoints are resampled to current grid of lower resolution
        by aggregating the values of the higher resolution grid.

        Parameters:
            coord (2d array): First column contains latitude, second
                column contains longitude. Each row is a geographic point.
        """
        raise NotImplementedError

    def is_regular(self):
        """Return True if grid is regular."""
        regular = False
        _, count_lat = np.unique(self[:, 0], return_counts=True)
        _, count_lon = np.unique(self[:, 1], return_counts=True)
        uni_lat_size = np.unique(count_lat).size
        uni_lon_size = np.unique(count_lon).size
        if uni_lat_size == uni_lon_size and uni_lat_size == 1 \
        and count_lat[0] > 1 and count_lon[0] > 1:
            regular = True
        return regular

    @property
    def lat(self):
        """Get latitude."""
        return self[:, 0]

    @property
    def lon(self):
        """Get longitude."""
        return self[:, 1]

def get_coastlines(border=None, resolution=110):
    """Get latitudes and longitudes of the coast lines inside border. All
    earth if no border.

    Parameters:
        border (tuple, optional): (min_lon, max_lon, min_lat, max_lat)
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            110m, i.e. 1:110.000.000

    Returns:
        lat (np.array), lon(np.array)
    """
    # TODO add writer to load if present
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='physical',
                                         name='coastline')
    shp = shapereader.Reader(shp_file)
    geoms = list(shp.geometries())

    coast_lon = list()
    coast_lat = list()
    for multi_line in geoms:
        coast_lon += multi_line.geoms[0].xy[0]
        coast_lat += multi_line.geoms[0].xy[1]
    coast_lon = np.array(coast_lon)
    coast_lat = np.array(coast_lat)

    if border is None:
        in_point = np.ones(coast_lon.size, dtype=bool)
    else:
        in_lon = np.logical_and(coast_lon >= border[0], coast_lon <= border[1])
        in_lat = np.logical_and(coast_lat >= border[2], coast_lat <= border[3])
        in_point = np.logical_and(in_lon, in_lat)

    return coast_lat[in_point], coast_lon[in_point]

def get_countries_geometry(country_names=None, resolution=110):
    """Get union of all the countries or the provided ones.

    Parameters:
        country_names (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            110m, i.e. 1:110.000.000

    Returns:
        shapely.geometry.multipolygon.MultiPolygon
    """
    resolution = nat_earth_resolution(resolution)
    file_globe = os.path.join(SYSTEM_DIR, GLOBE_COUNTRIES + "_" + resolution +
                              ".shp")
    if not os.path.isfile(file_globe) or country_names is not None:
        shp_file = shapereader.natural_earth(resolution=resolution,
                                             category='cultural',
                                             name='admin_0_countries')
        reader = shapereader.Reader(shp_file)
        countries = list(reader.records())

        if country_names is None:
            cntry_geom = [country.geometry for country in countries]
        else:
            cntry_geom = [country.geometry for country in countries
                          if country.attributes['ISO_A3'] in country_names]

        all_geom = shapely.ops.cascaded_union(cntry_geom)

        if country_names is None:
            LOGGER.info('Writing file %s', file_globe)

            shapewriter = shapefile.Writer()
            shapewriter.field("global_country_borders")
            shapely_to_pyshp(all_geom)
            converted_shape = shapely_to_pyshp(all_geom)
            shapewriter._shapes.append(converted_shape)
            shapewriter.record(["empty record"])
            shapewriter.save(file_globe)

            LOGGER.info('Written file %s', file_globe)
    else:
        reader = shapereader.Reader(file_globe)
        all_geom = list(reader.geometries())[0]

    return all_geom

def coord_on_land(land_geometry, lat, lon):
    """Check if point is on land (True) or water (False) of provided countries.
    All globe considered if no input countries.

    Parameters:
        land_geometry (shapely.geometry.multipolygon.MultiPolygon): profiles of
            land.
        lat (np.array): latitude of points
        lon (np.array): longiture of points

    Returns:
        np.array(bool)
    """
    if lat.size != lon.size:
        LOGGER.error('Wrong size input coordinates: %s != %s.', lat.size,
                     lon.size)
        raise ValueError
    return shapely.vectorized.contains(land_geometry, lon, lat)

def nat_earth_resolution(resolution):
    """Check if resolution is available in Natural Earth. Build string.

    Parameters:
        resolution (int): resolution in millions, 110 == 1:110.000.000.

    Returns:
        str

    Raises:
        ValueError
    """
    avail_res = [10, 50, 110]
    if resolution not in avail_res:
        LOGGER.error('Natural Earth does not accept resolution %s m.',
                     resolution)
        raise ValueError
    return str(resolution) + 'm'

def shapely_to_pyshp(shapely_geom):
    """ Shapely geometry to pyshp. Code from https://gis.stackexchange.com/
    questions/52705/how-to-write-shapely-geometries-to-shapefiles.

    Parameters:
        shapely_geom(shapely.geometry): shapely geometry to convert

    Returns:
        shapefile._Shape

    """
    # first convert shapely to geojson
    geoj = shapely.geometry.mapping(shapely_geom)
    # create empty pyshp shape
    record = shapefile._Shape()
    # set shapetype
    if geoj["type"] == "Null":
        pyshptype = 0
    elif geoj["type"] == "Point":
        pyshptype = 1
    elif geoj["type"] == "LineString":
        pyshptype = 3
    elif geoj["type"] == "Polygon":
        pyshptype = 5
    elif geoj["type"] == "MultiPoint":
        pyshptype = 8
    elif geoj["type"] == "MultiLineString":
        pyshptype = 3
    elif geoj["type"] == "MultiPolygon":
        pyshptype = 5
    record.shapeType = pyshptype
    # set points and parts
    if geoj["type"] == "Point":
        record.points = geoj["coordinates"]
        record.parts = [0]
    elif geoj["type"] in ("MultiPoint", "Linestring"):
        record.points = geoj["coordinates"]
        record.parts = [0]
    elif geoj["type"] == "Polygon":
        index = 0
        points = []
        parts = []
        for eachmulti in geoj["coordinates"]:
            points.extend(eachmulti)
            parts.append(index)
            index += len(eachmulti)
        record.points = points
        record.parts = parts
    elif geoj["type"] in ("MultiPolygon", "MultiLineString"):
        index = 0
        points = []
        parts = []
        for polygon in geoj["coordinates"]:
            for part in polygon:
                points.extend(part)
                parts.append(index)
                index += len(part)
        record.points = points
        record.parts = parts
    return record
