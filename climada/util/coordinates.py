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

Define functions to handle with coordinates
"""

import ast
import copy
import logging
import math
from multiprocessing import cpu_count
from pathlib import Path
import re

import zipfile

from cartopy.io import shapereader
import dask.dataframe as dd
from fiona.crs import from_epsg
import geopandas as gpd
import numpy as np
import pandas as pd
import pycountry
import rasterio
import rasterio.crs
import rasterio.features
import rasterio.mask
import rasterio.warp
import scipy.interpolate
from shapely.geometry import Polygon, MultiPolygon, Point, box
import shapely.ops
import shapely.vectorized
import shapefile

from climada.util.constants import (DEF_CRS, SYSTEM_DIR, ONE_LAT_KM,
                                    NATEARTH_CENTROIDS,
                                    ISIMIP_GPWV3_NATID_150AS,
                                    ISIMIP_NATID_TO_ISO,
                                    NONISO_REGIONS,
                                    RIVER_FLOOD_REGIONS_CSV)
from climada.util.files_handler import download_file
import climada.util.hdf5_handler as u_hdf5
import climada.util.interpolation as u_interp

pd.options.mode.chained_assignment = None

LOGGER = logging.getLogger(__name__)

NE_EPSG = 4326
"""Natural Earth CRS EPSG"""

NE_CRS = from_epsg(NE_EPSG)
"""Natural Earth CRS"""

TMP_ELEVATION_FILE = SYSTEM_DIR.joinpath('tmp_elevation.tif')
"""Path of elevation file written in set_elevation"""

DEM_NODATA = -9999
"""Value to use for no data values in DEM, i.e see points"""

MAX_DEM_TILES_DOWN = 300
"""Maximum DEM tiles to dowload"""

def latlon_to_geosph_vector(lat, lon, rad=False, basis=False):
    """Convert lat/lon coodinates to radial vectors (on geosphere)

    Parameters
    ----------
    lat, lon : ndarrays of floats, same shape
        Latitudes and longitudes of points.
    rad : bool, optional
        If True, latitude and longitude are not given in degrees but in radians.
    basis : bool, optional
        If True, also return an orthonormal basis of the tangent space at the
        given points in lat-lon coordinate system. Default: False.

    Returns
    -------
    vn : ndarray of floats, shape (..., 3)
        Same shape as lat/lon input with additional axis for components.
    vbasis : ndarray of floats, shape (..., 2, 3)
        Only present, if `basis` is True. Same shape as lat/lon input with
        additional axes for components of the two basis vectors.
    """
    if rad:
        rad_lat = lat + 0.5 * np.pi
        rad_lon = lon
    else:
        rad_lat = np.radians(lat + 90)
        rad_lon = np.radians(lon)
    sin_lat, cos_lat = np.sin(rad_lat), np.cos(rad_lat)
    sin_lon, cos_lon = np.sin(rad_lon), np.cos(rad_lon)
    vecn = np.stack((sin_lat * cos_lon, sin_lat * sin_lon, cos_lat), axis=-1)
    if basis:
        vbasis = np.stack((
            cos_lat * cos_lon, cos_lat * sin_lon, -sin_lat,
            -sin_lon, cos_lon, np.zeros_like(cos_lat),
        ), axis=-1).reshape(lat.shape + (2, 3))
        return vecn, vbasis
    return vecn

def lon_normalize(lon, center=0.0):
    """ Normalizes degrees such that always -180 < lon - center <= 180

    The input data is modified in place!

    Parameters
    ----------
    lon : np.array
        Longitudinal coordinates
    center : float, optional
        Central longitude value to use instead of 0. If None, the central longitude is determined
        automatically.

    Returns
    -------
    lon : np.array
        Normalized longitudinal coordinates. Since the input `lon` is modified in place (!), the
        returned array is the same Python object (instead of a copy).
    """
    if center is None:
        center = 0.5 * sum(lon_bounds(lon))
    bounds = (center - 180, center + 180)
    # map to [center - 360, center + 360] using modulo operator
    outside_mask = (lon <= bounds[0]) | (lon > bounds[1])
    lon[outside_mask] = (lon[outside_mask] % 360) + (center - center % 360)
    # map from [center - 360, center + 360] to [center - 180, center + 180], adding ±360
    if center % 360 < 180:
        lon[lon > bounds[1]] -= 360
    else:
        lon[lon <= bounds[0]] += 360
    return lon

def lon_bounds(lon, buffer=0.0):
    """Bounds of a set of degree values, respecting the periodicity in longitude

    The longitudinal upper bound may be 180 or larger to make sure that the upper bound is always
    larger than the lower bound. The lower longitudinal bound will never lie below -180 and it will
    only assume the value -180 if the specified buffering enforces it.

    Note that, as a consequence of this, the returned bounds do not satisfy the inequality
    `lon_min <= lon <= lon_max` in general!

    Usually, an application of this function is followed by a renormalization of longitudinal
    values around the longitudinal middle value:

    >>> bounds = lon_bounds(lon)
    >>> lon_mid = 0.5 * (bounds[0] + bounds[2])
    >>> lon = lon_normalize(lon, center=lon_mid)
    >>> np.all((bounds[0] <= lon) & (lon <= bounds[2]))

    Example
    -------
    >>> lon_bounds(np.array([-179, 175, 178]))
    (175, 181)
    >>> lon_bounds(np.array([-179, 175, 178]), buffer=1)
    (174, 182)

    Parameters
    ----------
    lon : np.array
        Longitudinal coordinates
    buffer : float, optional
        Buffer to add to both sides of the bounding box. Default: 0.0.

    Returns
    -------
    bounds : tuple (lon_min, lon_max)
        Bounding box of the given points.
    """
    lon = lon_normalize(lon.copy())
    lon_uniq = np.unique(lon)
    lon_uniq = np.concatenate([lon_uniq, [360 + lon_uniq[0]]])
    lon_diff = np.diff(lon_uniq)
    gap_max = np.argmax(lon_diff)
    lon_diff_max = lon_diff[gap_max]
    if lon_diff_max < 2:
        # looks like the data covers the whole range [-180, 180] rather evenly
        lon_min = max(lon_uniq[0] - buffer, -180)
        lon_max = min(lon_uniq[-2] + buffer, 180)
    else:
        lon_min = lon_uniq[gap_max + 1]
        lon_max = lon_uniq[gap_max]
        if lon_min > 180:
            lon_min -= 360
        else:
            lon_max += 360
        lon_min -= buffer
        lon_max += buffer
        if lon_min <= -180:
            lon_min += 360
            lon_max += 360
    return (lon_min, lon_max)


def latlon_bounds(lat, lon, buffer=0.0):
    """Bounds of a set of degree values, respecting the periodicity in longitude

    See `lon_bounds` for more information about the handling of longitudinal values crossing the
    antimeridian.

    Example
    -------
    >>> latlon_bounds(np.array([0, -2, 5]), np.array([-179, 175, 178]))
    (175, -2, 181, 5)
    >>> latlon_bounds(np.array([0, -2, 5]), np.array([-179, 175, 178]), buffer=1)
    (174, -3, 182, 6)

    Parameters
    ----------
    lat : np.array
        Latitudinal coordinates
    lon : np.array
        Longitudinal coordinates
    buffer : float, optional
        Buffer to add to all sides of the bounding box. Default: 0.0.

    Returns
    -------
    bounds : tuple (lon_min, lat_min, lon_max, lat_max)
        Bounding box of the given points.
    """
    lon_min, lon_max = lon_bounds(lon, buffer=buffer)
    return (lon_min, max(lat.min() - buffer, -90), lon_max, min(lat.max() + buffer, 90))

def dist_approx(lat1, lon1, lat2, lon2, log=False, normalize=True,
                method="equirect", units='km'):
    """Compute approximation of geodistance in specified units

    Parameters
    ----------
    lat1, lon1 : ndarrays of floats, shape (nbatch, nx)
        Latitudes and longitudes of first points.
    lat2, lon2 : ndarrays of floats, shape (nbatch, ny)
        Latitudes and longitudes of second points.
    log : bool, optional
        If True, return the tangential vectors at the first points pointing to
        the second points (Riemannian logarithm). Default: False.
    normalize : bool, optional
        If False, assume that lon values are already between -180 and 180.
        Default: True
    method : str, optional
        Specify an approximation method to use:
        * "equirect": Distance according to sinusoidal projection. Fast, but inaccurate for large
          distances and high latitudes.
        * "geosphere": Exact spherical distance. Much more accurate at all distances, but slow.
        Note that ellipsoidal distances would be even more accurate, but are currently not
        implemented. Default: "equirect".
    units : str, optional
        Specify a unit for the distance. One of:
        * "km": distance in km.
        * "degree": angular distance in decimal degrees.
        * "radian": angular distance in radians.
        Default: "km".

    Returns
    -------
    dists : ndarray of floats, shape (nbatch, nx, ny)
        Approximate distances in specified units.
    vtan : ndarray of floats, shape (nbatch, nx, ny, 2)
        If `log` is True, tangential vectors at first points in local
        lat-lon coordinate system.
    """
    if units == "km":
        unit_factor = ONE_LAT_KM
    elif units == "radian":
        unit_factor = np.radians(1.0)
    elif units == "degree":
        unit_factor = 1
    else:
        raise KeyError('Unknown distance unit: %s' % units)

    if method == "equirect":
        if normalize:
            mid_lon = 0.5 * sum(lon_bounds(np.concatenate([lon1, lon2])))
            lon_normalize(lon1, center=mid_lon)
            lon_normalize(lon2, center=mid_lon)
        vtan = np.stack([lat2[:, None] - lat1[:, :, None],
                         lon2[:, None] - lon1[:, :, None]], axis=-1)
        fact1 = np.heaviside(vtan[..., 1] - 180, 0)
        fact2 = np.heaviside(-vtan[..., 1] - 180, 0)
        vtan[..., 1] -= (fact1 - fact2) * 360
        vtan[..., 1] *= np.cos(np.radians(lat1[:, :, None]))
        vtan *= unit_factor
        # faster version of `dist = np.linalg.norm(vtan, axis=-1)`
        dist = np.sqrt(np.einsum("...l,...l->...", vtan, vtan))
    elif method == "geosphere":
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = 0.5 * (lat2[:, None] - lat1[:, :, None])
        dlon = 0.5 * (lon2[:, None] - lon1[:, :, None])
        # haversine formula:
        hav = np.sin(dlat)**2 \
            + np.cos(lat1[:, :, None]) * np.cos(lat2[:, None]) * np.sin(dlon)**2
        dist = np.degrees(2 * np.arcsin(np.sqrt(hav))) * unit_factor
        if log:
            vec1, vbasis = latlon_to_geosph_vector(lat1, lon1, rad=True, basis=True)
            vec2 = latlon_to_geosph_vector(lat2, lon2, rad=True)
            scal = 1 - 2 * hav
            fact = dist / np.fmax(np.spacing(1), np.sqrt(1 - scal**2))
            vtan = fact[..., None] * (vec2[:, None] - scal[..., None] * vec1[:, :, None])
            vtan = np.einsum('nkli,nkji->nklj', vtan, vbasis)
    else:
        raise KeyError("Unknown distance approximation method: %s" % method)
    return (dist, vtan) if log else dist

def get_gridcellarea(lat, resolution=0.5, unit='km2'):
    """The area covered by a grid cell is calculated depending on the latitude
        1 degree = ONE_LAT_KM (111.12km at the equator)
        longitudal distance in km = ONE_LAT_KM*resolution*cos(lat)
        latitudal distance in km = ONE_LAT_KM*resolution
        area = longitudal distance * latitudal distance

    Parameters
    ----------
    lat : np.array
        Latitude of the respective grid cell
    resolution: int, optional
        raster resolution in degree (default: 0.5 degree)
    unit: string, optional
        unit of the output area (default: km2, alternative: m2)

    """

    if unit == 'm2':
        area = (ONE_LAT_KM * resolution)**2 * np.cos(np.deg2rad(lat)) * 100 * 1000000
    else:
        area = (ONE_LAT_KM * resolution)**2 * np.cos(np.deg2rad(lat)) * 100

    return area

def grid_is_regular(coord):
    """Return True if grid is regular. If True, returns height and width.

    Parameters
    ----------
    coord : np.array
        Each row is a lat-lon-pair.

    Returns
    -------
    regular : bool
        Whether the grid is regular. Only in this case, the following width and height are
        reliable.
    height : int
        Height of the supposed grid.
    width : int
        Width of the supposed grid.
    """
    regular = False
    _, count_lat = np.unique(coord[:, 0], return_counts=True)
    _, count_lon = np.unique(coord[:, 1], return_counts=True)
    uni_lat_size = np.unique(count_lat).size
    uni_lon_size = np.unique(count_lon).size
    if uni_lat_size == uni_lon_size and uni_lat_size == 1 \
    and count_lat[0] > 1 and count_lon[0] > 1:
        regular = True
    return regular, count_lat[0], count_lon[0]

def get_coastlines(bounds=None, resolution=110):
    """Get Polygones of coast intersecting given bounds

    Parameters
    ----------
    bounds : tuple
        min_lon, min_lat, max_lon, max_lat in EPSG:4326
    resolution : float, optional
        10, 50 or 110. Resolution in m. Default: 110m, i.e. 1:110.000.000

    Returns
    -------
    coastlines : GeoDataFrame
        Polygons of coast intersecting given bounds.
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='physical',
                                         name='coastline')
    coast_df = gpd.read_file(shp_file)
    coast_df.crs = NE_CRS
    if bounds is None:
        return coast_df[['geometry']]
    tot_coast = np.zeros(1)
    while not np.any(tot_coast):
        tot_coast = coast_df.envelope.intersects(box(*bounds))
        bounds = (bounds[0] - 20, bounds[1] - 20,
                  bounds[2] + 20, bounds[3] + 20)
    return coast_df[tot_coast][['geometry']]

def convert_wgs_to_utm(lon, lat):
    """Get EPSG code of UTM projection for input point in EPSG 4326

    Parameters
    ----------
    lon : float
        longitude point in EPSG 4326
    lat : float
        latitude of point (lat, lon) in EPSG 4326

    Returns
    -------
    epsg_code : int
        EPSG code of UTM projection.
    """
    epsg_utm_base = 32601 + (0 if lat >= 0 else 100)
    return epsg_utm_base + (math.floor((lon + 180) / 6) % 60)

def utm_zones(wgs_bounds):
    """Get EPSG code and bounds of UTM zones covering specified region

    Parameters
    ----------
    wgs_bounds : tuple
        lon_min, lat_min, lon_max, lat_max

    Returns
    -------
    zones : list of pairs (zone_epsg, zone_wgs_bounds)
        EPSG code and bounding box in WGS coordinates.
    """
    lon_min, lat_min, lon_max, lat_max = wgs_bounds
    lon_min, lon_max = max(-179.99, lon_min), min(179.99, lon_max)
    utm_min, utm_max = [math.floor((l + 180) / 6) for l in [lon_min, lon_max]]
    zones = []
    for utm in range(utm_min, utm_max + 1):
        epsg = 32601 + utm
        bounds = (-180 + 6 * utm, 0, -180 + 6 * (utm + 1), 90)
        if lat_max >= 0:
            zones.append((epsg, bounds))
        if lat_min < 0:
            bounds = (bounds[0], -90, bounds[2], 0)
            zones.append((epsg + 100, bounds))
    return zones

def dist_to_coast(coord_lat, lon=None, signed=False):
    """Compute (signed) distance to coast from input points in meters.

    Parameters
    ----------
    coord_lat : GeoDataFrame or np.array or float
        One of the following:
        * GeoDataFrame with geometry column in epsg:4326
        * np.array with two columns, first for latitude of each point
          and second with longitude in epsg:4326
        * np.array with one dimension containing latitudes in epsg:4326
        * float with a latitude value in epsg:4326
    lon : np.array or float, optional
        One of the following:
        * np.array with one dimension containing longitudes in epsg:4326
        * float with a longitude value in epsg:4326
    signed : bool
        If True, distance is signed with positive values off shore and negative values on land.
        Default: False

    Returns
    -------
    dist : np.array
        (Signed) distance to coast in meters.
    """
    if isinstance(coord_lat, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if not equal_crs(coord_lat.crs, NE_CRS):
            raise ValueError('Input CRS is not %s' % str(NE_CRS))
        geom = coord_lat
    else:
        if lon is None:
            if isinstance(coord_lat, np.ndarray) and coord_lat.shape[1] == 2:
                lat, lon = coord_lat[:, 0], coord_lat[:, 1]
            else:
                raise ValueError('Missing longitude values.')
        else:
            lat, lon = [np.asarray(v).reshape(-1) for v in [coord_lat, lon]]
            if lat.size != lon.size:
                raise ValueError('Mismatching input coordinates size: %s != %s'
                                 % (lat.size, lon.size))
        geom = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon, lat), crs=NE_CRS)

    pad = 20
    bounds = (geom.total_bounds[0] - pad, geom.total_bounds[1] - pad,
              geom.total_bounds[2] + pad, geom.total_bounds[3] + pad)
    coast = get_coastlines(bounds, 10).geometry
    coast = gpd.GeoDataFrame(geometry=coast, crs=NE_CRS)
    dist = np.empty(geom.shape[0])
    zones = utm_zones(geom.geometry.total_bounds)
    for izone, (epsg, bounds) in enumerate(zones):
        to_crs = from_epsg(epsg)
        zone_mask = (
            (bounds[1] <= geom.geometry.y)
            & (geom.geometry.y <= bounds[3])
            & (bounds[0] <= geom.geometry.x)
            & (geom.geometry.x <= bounds[2])
        )
        if np.count_nonzero(zone_mask) == 0:
            continue
        LOGGER.info("dist_to_coast: UTM %d (%d/%d)",
                    epsg, izone + 1, len(zones))
        bounds = geom[zone_mask].total_bounds
        bounds = (bounds[0] - pad, bounds[1] - pad,
                  bounds[2] + pad, bounds[3] + pad)
        coast_mask = coast.envelope.intersects(box(*bounds))
        utm_coast = coast[coast_mask].geometry.unary_union
        utm_coast = gpd.GeoDataFrame(geometry=[utm_coast], crs=NE_CRS)
        utm_coast = utm_coast.to_crs(to_crs).geometry[0]
        dist[zone_mask] = geom[zone_mask].to_crs(to_crs).distance(utm_coast)
    if signed:
        dist[coord_on_land(geom.geometry.y, geom.geometry.x)] *= -1
    return dist

def dist_to_coast_nasa(lat, lon, highres=False, signed=False):
    """Read interpolated (signed) distance to coast (in m) from NASA data

    Note: The NASA raster file is 300 MB and will be downloaded on first run!

    Parameters
    ----------
    lat : np.array
        latitudes in epsg:4326
    lon : np.array
        longitudes in epsg:4326
    highres : bool, optional
        Use full resolution of NASA data (much slower). Default: False.
    signed : bool
        If True, distance is signed with positive values off shore and negative values on land.
        Default: False

    Returns
    -------
    dist : np.array
        (Signed) distance to coast in meters.
    """
    lat, lon = [np.asarray(ar).ravel() for ar in [lat, lon]]
    lon = lon_normalize(lon.copy())

    # TODO move URL to config
    zipname = "GMT_intermediate_coast_distance_01d.zip"
    tifname = "GMT_intermediate_coast_distance_01d.tif"
    url = "https://oceancolor.gsfc.nasa.gov/docs/distfromcoast/" + zipname
    path = SYSTEM_DIR.joinpath(tifname)
    if not path.is_file():
        path_dwn = download_file(url, download_dir=SYSTEM_DIR)
        zip_ref = zipfile.ZipFile(path_dwn, 'r')
        zip_ref.extractall(SYSTEM_DIR)
        zip_ref.close()

    intermediate_res = None if highres else 0.1
    west_msk = (lon < 0)
    dist = np.zeros_like(lat)
    for msk in [west_msk, ~west_msk]:
        if np.count_nonzero(msk) > 0:
            dist[msk] = read_raster_sample(
                path, lat[msk], lon[msk], intermediate_res=intermediate_res, fill_value=0)
    if not signed:
        dist = np.abs(dist)
    return 1000 * dist

def get_land_geometry(country_names=None, extent=None, resolution=10):
    """Get union of the specified (or all) countries or the points inside the extent.

    Parameters
    ----------
    country_names : list, optional
        list with ISO3 names of countries, e.g ['ZWE', 'GBR', 'VNM', 'UZB']
    extent : tuple, optional
        (min_lon, max_lon, min_lat, max_lat)
    resolution : float, optional
        10, 50 or 110. Resolution in m. Default: 10m, i.e. 1:10.000.000

    Returns
    -------
    geom : shapely.geometry.multipolygon.MultiPolygon
        Polygonal shape of union.
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='cultural',
                                         name='admin_0_countries')
    reader = shapereader.Reader(shp_file)
    if (country_names is None) and (extent is None):
        LOGGER.info("Computing earth's land geometry ...")
        geom = list(reader.geometries())
        geom = shapely.ops.cascaded_union(geom)

    elif country_names:
        countries = list(reader.records())
        geom = [country.geometry for country in countries
                if (country.attributes['ISO_A3'] in country_names) or
                (country.attributes['WB_A3'] in country_names) or
                (country.attributes['ADM0_A3'] in country_names)]
        geom = shapely.ops.cascaded_union(geom)

    else:
        extent_poly = Polygon([(extent[0], extent[2]), (extent[0], extent[3]),
                               (extent[1], extent[3]), (extent[1], extent[2])])
        geom = []
        for cntry_geom in reader.geometries():
            inter_poly = cntry_geom.intersection(extent_poly)
            if not inter_poly.is_empty:
                geom.append(inter_poly)
        geom = shapely.ops.cascaded_union(geom)
    if not isinstance(geom, MultiPolygon):
        geom = MultiPolygon([geom])
    return geom

def coord_on_land(lat, lon, land_geom=None):
    """Check if points are on land.

    Parameters
    ----------
    lat : np.array
        latitude of points in epsg:4326
    lon : np.array
        longitude of points in epsg:4326
    land_geom : shapely.geometry.multipolygon.MultiPolygon, optional
         If given, use these as profiles of land. Otherwise, the global landmass is used.

    Returns
    -------
    on_land : np.array(bool)
        Entries are True if corresponding coordinate is on land and False otherwise.
    """
    if lat.size != lon.size:
        raise ValueError('Wrong size input coordinates: %s != %s.'
                         % (lat.size, lon.size))
    if lat.size == 0:
        return np.empty((0,), dtype=bool)
    delta_deg = 1
    if land_geom is None:
        land_geom = get_land_geometry(
            extent=(np.min(lon) - delta_deg,
                    np.max(lon) + delta_deg,
                    np.min(lat) - delta_deg,
                    np.max(lat) + delta_deg),
            resolution=10)
    return shapely.vectorized.contains(land_geom, lon, lat)

def nat_earth_resolution(resolution):
    """Check if resolution is available in Natural Earth. Build string.

    Parameters
    ----------
    resolution : int
        resolution in millions, 110 == 1:110.000.000.

    Returns
    -------
    res_name : str
        Natural Earth name of resolution (e.g. '110m')

    Raises
    ------
    ValueError
    """
    avail_res = [10, 50, 110]
    if resolution not in avail_res:
        raise ValueError('Natural Earth does not accept resolution %s m.' % resolution)
    return str(resolution) + 'm'

def get_country_geometries(country_names=None, extent=None, resolution=10):
    """Natural Earth country boundaries within given extent

    If no arguments are given, simply returns the whole natural earth dataset.

    Take heed: we assume WGS84 as the CRS unless the Natural Earth download utility from cartopy
    starts including the projection information. (They are saving a whopping 147 bytes by omitting
    it.) Same goes for UTF.

    Parameters
    ----------
    country_names : list, optional
        list with ISO 3166 alpha-3 codes of countries, e.g ['ZWE', 'GBR', 'VNM', 'UZB']
    extent : tuple (min_lon, max_lon, min_lat, max_lat), optional
        Extent, assumed to be in the same CRS as the natural earth data.
    resolution : float, optional
        10, 50 or 110. Resolution in m. Default: 10m

    Returns
    -------
    geom : GeoDataFrame
        Natural Earth multipolygons of the specified countries, resp. the countries that lie
        within the specified extent.
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='cultural',
                                         name='admin_0_countries')
    nat_earth = gpd.read_file(shp_file, encoding='UTF-8')

    if not nat_earth.crs:
        nat_earth.crs = NE_CRS

    # fill gaps in nat_earth
    gap_mask = (nat_earth['ISO_A3'] == '-99')
    nat_earth.loc[gap_mask, 'ISO_A3'] = nat_earth.loc[gap_mask, 'ADM0_A3']

    gap_mask = (nat_earth['ISO_N3'] == '-99')
    for idx, country in nat_earth[gap_mask].iterrows():
        nat_earth.loc[idx, "ISO_N3"] = f"{natearth_country_to_int(country):03d}"

    out = nat_earth
    if country_names:
        if isinstance(country_names, str):
            country_names = [country_names]
        out = out[out.ISO_A3.isin(country_names)]

    if extent:
        bbox = Polygon([
            (extent[0], extent[2]),
            (extent[0], extent[3]),
            (extent[1], extent[3]),
            (extent[1], extent[2])
        ])
        bbox = gpd.GeoSeries(bbox, crs=out.crs)
        bbox = gpd.GeoDataFrame({'geometry': bbox}, crs=out.crs)
        out = gpd.overlay(out, bbox, how="intersection")

    return out

def get_region_gridpoints(countries=None, regions=None, resolution=150,
                          iso=True, rect=False, basemap="natearth"):
    """Get coordinates of gridpoints in specified countries or regions

    Parameters
    ----------
    countries : list, optional
        ISO 3166-1 alpha-3 codes of countries, or internal numeric NatID if `iso` is set to False.
    regions : list, optional
        Region IDs.
    resolution : float, optional
        Resolution in arc-seconds, either 150 (default) or 360.
    iso : bool, optional
        If True, assume that countries are given by their ISO 3166-1 alpha-3 codes (instead of the
        internal NatID). Default: True.
    rect : bool, optional
        If True, a rectangular box around the specified countries/regions is selected.
        Default: False.
    basemap : str, optional
        Choose between different data sources. Currently available: "isimip" and "natearth".
        Default: "natearth".

    Returns
    -------
    lat : np.array
        Latitude of points in epsg:4326.
    lon : np.array
        Longitude of points in epsg:4326.
    """
    if countries is None:
        countries = []
    if regions is None:
        regions = []

    if basemap == "natearth":
        base_file = NATEARTH_CENTROIDS[resolution]
        hdf5_f = u_hdf5.read(base_file)
        meta = hdf5_f['meta']
        grid_shape = (meta['height'][0], meta['width'][0])
        transform = rasterio.Affine(*meta['transform'])
        region_id = hdf5_f['region_id'].reshape(grid_shape)
        lon, lat = raster_to_meshgrid(transform, grid_shape[1], grid_shape[0])
    elif basemap == "isimip":
        hdf5_f = u_hdf5.read(ISIMIP_GPWV3_NATID_150AS)
        dim_lon, dim_lat = hdf5_f['lon'], hdf5_f['lat']
        bounds = dim_lon.min(), dim_lat.min(), dim_lon.max(), dim_lat.max()
        orig_res = get_resolution(dim_lon, dim_lat)
        _, _, transform = pts_to_raster_meta(bounds, orig_res)
        grid_shape = (dim_lat.size, dim_lon.size)
        region_id = hdf5_f['NatIdGrid'].reshape(grid_shape).astype(int)
        region_id[region_id < 0] = 0
        natid2iso_numeric = np.array(country_natid2iso(list(range(231)), "numeric"), dtype=int)
        region_id = natid2iso_numeric[region_id]
        lon, lat = np.meshgrid(dim_lon, dim_lat)
    else:
        raise ValueError(f"Unknown basemap: {basemap}")

    if basemap == "natearth" and resolution not in [150, 360] \
       or basemap == "isimip" and resolution != 150:
        resolution /= 3600
        region_id, transform = refine_raster_data(
            region_id, transform, resolution, method='nearest', fill_value=0)
        grid_shape = region_id.shape
        lon, lat = raster_to_meshgrid(transform, grid_shape[1], grid_shape[0])

    if not iso:
        countries = country_natid2iso(countries)
    countries += region2isos(regions)
    countries = np.unique(country_to_iso(countries, "numeric"))

    if len(countries) > 0:
        msk = np.isin(region_id, countries)
        if rect:
            lat_msk, lon_msk = lat[msk], lon[msk]
            msk = msk.any(axis=0)[None] * msk.any(axis=1)[:, None]
            msk |= (
                (lat >= np.floor(lat_msk.min()))
                & (lon >= np.floor(lon_msk.min()))
                & (lat <= np.ceil(lat_msk.max()))
                & (lon <= np.ceil(lon_msk.max()))
            )
        lat, lon = lat[msk], lon[msk]
    else:
        lat, lon = [ar.ravel() for ar in [lat, lon]]
    return lat, lon

def assign_grid_points(x, y, grid_width, grid_height, grid_transform):
    """To each coordinate in `x` and `y`, assign the closest centroid in the given raster grid

    Make sure that your grid specification is relative to the same coordinate reference system as
    the `x` and `y` coordinates. In case of lon/lat coordinates, make sure that the longitudinal
    values are within the same longitudinal range (such as [-180, 180]).

    If your grid is given by bounds instead of a transform, the functions
    `rasterio.transform.from_bounds` and `pts_to_raster_meta` might be helpful.

    Parameters
    ----------
    x, y : np.array
        x- and y-coordinates of points to assign coordinates to.
    grid_width : int
        Width (number of columns) of the grid.
    grid_height : int
        Height (number of rows) of the grid.
    grid_transform : affine.Affine
        Affine transformation defining the grid raster.

    Returns
    -------
    assigned_idx : np.array of size equal to the size of x and y
        Index into the flattened `grid`. Note that the value `-1` is used to indicate that no
        matching coordinate has been found, even though `-1` is a valid index in NumPy!
    """
    x, y = np.array(x), np.array(y)
    xres, _, xmin, _, yres, ymin = grid_transform[:6]
    xmin, ymin = xmin + 0.5 * xres, ymin + 0.5 * yres
    x_i = np.round((x - xmin) / xres).astype(int)
    y_i = np.round((y - ymin) / yres).astype(int)
    assigned = y_i * grid_width + x_i
    assigned[(x_i < 0) | (x_i >= grid_width)] = -1
    assigned[(y_i < 0) | (y_i >= grid_height)] = -1
    return assigned

def assign_coordinates(coords, coords_to_assign, method="NN", distance="haversine", threshold=100):
    """To each coordinate in `coords`, assign a matching coordinate in `coords_to_assign`

    If there is no exact match for some entry, an attempt is made to assign the geographically
    nearest neighbor. If the distance to the nearest neighbor exceeds `threshold`, the index `-1`
    is assigned.

    Currently, the nearest neighbor matching works with lat/lon coordinates only. However, you can
    disable nearest neighbor matching by setting `threshold` to 0, in which case only exactly
    matching coordinates are assigned to each other.

    Make sure that all coordinates are according to the same coordinate reference system. In case
    of lat/lon coordinates, the "haversine" distance is able to correctly compute the distance
    across the antimeridian. However, when exact matches are enforced with `threshold=0`, lat/lon
    coordinates need to be given in the same longitudinal range (such as (-180, 180)).

    Parameters
    ----------
    coords : np.array with two columns
        Each row is a geographical coordinate pair. The result's size will match this array's
        number of rows.
    coords_to_assign : np.array with two columns
        Each row is a geographical coordinate pair. The result will be an index into the
        rows of this array. Make sure that these coordinates use the same coordinate reference
        system as `coords`.
    method : str, optional
        Interpolation method to use for non-exact matching. Currently, "NN" (nearest neighbor)
        is the only supported value, see `climada.util.interpolation.interpol_index`.
    distance : str, optional
        Distance to use for non-exact matching. Possible values are "haversine" and "approx", see
        `climada.util.interpolation.interpol_index`. Default: "haversine"
    threshold : float, optional
        If the distance to the nearest neighbor exceeds `threshold`, the index `-1` is assigned.
        Set `threshold` to 0 to disable nearest neighbor matching. Default: 100 (km)

    Returns
    -------
    assigned_idx : np.array of size equal to the number of rows in `coords`
        Index into `coords_to_assign`. Note that the value `-1` is used to indicate that no
        matching coordinate has been found, even though `-1` is a valid index in NumPy!
    """

    if not coords.any():
        return np.array([])
    elif not coords_to_assign.any():
        return -np.ones(coords.shape[0]).astype(int)
    coords = coords.astype('float64')
    coords_to_assign = coords_to_assign.astype('float64')
    if np.array_equal(coords, coords_to_assign):
        assigned_idx = np.arange(coords.shape[0])
    else:
        LOGGER.info("No exact centroid match found. Reprojecting coordinates "
                    "to nearest neighbor closer than the threshold = %s",
                    threshold)
        # pairs of floats can be sorted (lexicographically) in NumPy
        coords_view = coords.view(dtype='float64,float64').reshape(-1)
        coords_to_assign_view = coords_to_assign.view(dtype='float64,float64').reshape(-1)

        # assign each hazard coordsinate to an element in coords using searchsorted
        coords_sorter = np.argsort(coords_view)
        sort_assign_idx = np.fmin(coords_sorter.size - 1, np.searchsorted(
            coords_view, coords_to_assign_view, side="left", sorter=coords_sorter))
        sort_assign_idx = coords_sorter[sort_assign_idx]

        # determine which of the assignements match exactly
        exact_assign_idx = (coords_view[sort_assign_idx] == coords_to_assign_view).nonzero()[0]
        assigned_idx = np.full_like(coords_sorter, -1)
        assigned_idx[sort_assign_idx[exact_assign_idx]] = exact_assign_idx

        # assign remaining coordsinates to their geographically nearest neighbor
        if threshold > 0 and exact_assign_idx.size != coords_view.size:
            not_assigned_idx_mask = (assigned_idx == -1)
            assigned_idx[not_assigned_idx_mask] = u_interp.interpol_index(
                coords_to_assign, coords[not_assigned_idx_mask],
                method=method, distance=distance, threshold=threshold)
    return assigned_idx

def region2isos(regions):
    """Convert region names to ISO 3166 alpha-3 codes of countries

    Parameters
    ----------
    regions : str or list of str
        Region name(s).

    Returns
    -------
    isos : list of str
        Sorted list of iso codes of all countries in specified region(s).
    """
    regions = [regions] if isinstance(regions, str) else regions
    reg_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    isos = []
    for region in regions:
        region_msk = (reg_info['Reg_name'] == region)
        if not any(region_msk):
            raise KeyError('Unknown region name: %s' % region)
        isos += list(reg_info['ISO'][region_msk].values)
    return list(set(isos))

def country_to_iso(countries, representation="alpha3", fillvalue=None):
    """Determine ISO 3166 representation of countries

    Example
    -------
    >>> country_to_iso(840)
    'USA'
    >>> country_to_iso("United States", representation="alpha2")
    'US'
    >>> country_to_iso(["United States of America", "SU"], "numeric")
    [840, 810]

    Some geopolitical areas that are not covered by ISO 3166 are added in the "user-assigned"
    range of ISO 3166-compliant values:

    >>> country_to_iso(["XK", "Dhekelia"], "numeric")  # XK for Kosovo
    [983, 907]

    Parameters
    ----------
    countries : one of str, int, list of str, list of int
        Country identifiers: name, official name, alpha-2, alpha-3 or numeric ISO codes.
        Numeric representations may be specified as str or int.
    representation : str (one of "alpha3", "alpha2", "numeric", "name"), optional
        All countries are converted to this representation according to ISO 3166.
        Default: "alpha3".
    fillvalue : str or int or None, optional
        The value to assign if a country is not recognized by the given identifier. By default,
        a LookupError is raised. Default: None

    Returns
    -------
    iso_list : one of str, int, list of str, list of int
        ISO 3166 representation of countries. Will only return a list if the input is a list.
        Numeric representations are returned as integers.
    """
    return_single = np.isscalar(countries)
    countries = [countries] if return_single else countries

    if not re.match(r"(alpha[-_]?[23]|numeric|name)", representation):
        raise ValueError(f"Unknown ISO representation: {representation}")
    representation = re.sub(r"alpha-?([23])", r"alpha_\1", representation)

    iso_list = []
    for country in countries:
        country = country if isinstance(country, str) else f"{int(country):03d}"
        try:
            match = pycountry.countries.lookup(country)
        except LookupError:
            try:
                match = pycountry.historic_countries.lookup(country)
            except LookupError:
                match = next(filter(lambda c: country in c.values(), NONISO_REGIONS), None)
                if match is not None:
                    match = pycountry.db.Data(**match)
                elif fillvalue is not None:
                    match = pycountry.db.Data(**{representation: fillvalue})
                else:
                    raise LookupError(f'Unknown country identifier: {country}') from None
        iso = getattr(match, representation)
        if representation == "numeric":
            iso = int(iso)
        iso_list.append(iso)
    return iso_list[0] if return_single else iso_list

def country_iso_alpha2numeric(iso_alpha):
    """Deprecated: Use `country_to_iso` with `representation="numeric"` instead"""
    LOGGER.warning("country_iso_alpha2numeric is deprecated, use country_to_iso instead.")
    return country_to_iso(iso_alpha, "numeric")

def country_natid2iso(natids, representation="alpha3"):
    """Convert internal NatIDs to ISO 3166-1 alpha-3 codes

    Parameters
    ----------
    natids : int or list of int
        NatIDs of countries (or single ID) as used in ISIMIP's version of the GPWv3
        national identifier grid.
    representation : str, one of "alpha3", "alpha2" or "numeric"
        All countries are converted to this representation according to ISO 3166.
        Default: "alpha3".

    Returns
    -------
    iso_list : one of str, int, list of str, list of int
        ISO 3166 representation of countries. Will only return a list if the input is a list.
        Numeric representations are returned as integers.
    """
    return_str = isinstance(natids, int)
    natids = [natids] if return_str else natids
    iso_list = []
    for natid in natids:
        if natid < 0 or natid >= len(ISIMIP_NATID_TO_ISO):
            raise LookupError('Unknown country NatID: %s' % natid)
        iso_list.append(ISIMIP_NATID_TO_ISO[natid])
    if representation != "alpha3":
        iso_list = country_to_iso(iso_list, representation)
    return iso_list[0] if return_str else iso_list

def country_iso2natid(isos):
    """Convert ISO 3166-1 alpha-3 codes to internal NatIDs

    Parameters
    ----------
    isos : str or list of str
        ISO codes of countries (or single code).

    Returns
    -------
    natids : int or list of int
        Will only return a list if the input is a list.
    """
    return_int = isinstance(isos, str)
    isos = [isos] if return_int else isos
    natids = []
    for iso in isos:
        try:
            natids.append(ISIMIP_NATID_TO_ISO.index(iso))
        except ValueError as ver:
            raise LookupError(f'Unknown country ISO: {iso}') from ver
    return natids[0] if return_int else natids

def natearth_country_to_int(country):
    """Integer representation (ISO 3166, if possible) of Natural Earth GeoPandas country row

    Parameters
    ----------
    country : GeoSeries
        Row from Natural Earth GeoDataFrame.

    Returns
    -------
    iso_numeric : int
        Integer representation of given country.
    """
    if country.ISO_N3 != '-99':
        return int(country.ISO_N3)
    return country_to_iso(str(country.NAME), representation="numeric")

def get_country_code(lat, lon, gridded=False):
    """Provide numeric (ISO 3166) code for every point.

    Oceans get the value zero. Areas that are not in ISO 3166 are given values in the range above
    900 according to NATEARTH_AREA_NONISO_NUMERIC.

    Parameters
    ----------
    lat : np.array
        latitude of points in epsg:4326
    lon : np.array
        longitude of points in epsg:4326
    gridded : bool
        If True, interpolate precomputed gridded data which is usually much faster. Default: False.

    Returns
    -------
    country_codes : np.array(int)
        Numeric code for each point.
    """
    lat, lon = [np.asarray(ar).ravel() for ar in [lat, lon]]
    if lat.size == 0:
        return np.empty((0,), dtype=int)
    LOGGER.info('Setting region_id %s points.', str(lat.size))
    if gridded:
        base_file = u_hdf5.read(NATEARTH_CENTROIDS[150])
        meta, region_id = base_file['meta'], base_file['region_id']
        transform = rasterio.Affine(*meta['transform'])
        region_id = region_id.reshape(meta['height'][0], meta['width'][0])
        region_id = interp_raster_data(region_id, lat, lon, transform,
                                       method='nearest', fill_value=0)
        region_id = region_id.astype(int)
    else:
        extent = (lon.min() - 0.001, lon.max() + 0.001,
                  lat.min() - 0.001, lat.max() + 0.001)
        countries = get_country_geometries(extent=extent)
        countries['area'] = countries.geometry.area
        countries = countries.sort_values(by=['area'], ascending=False)
        region_id = np.full((lon.size,), -1, dtype=int)
        total_land = countries.geometry.unary_union
        ocean_mask = ~shapely.vectorized.contains(total_land, lon, lat)
        region_id[ocean_mask] = 0
        for country in countries.itertuples():
            unset = (region_id == -1).nonzero()[0]
            select = shapely.vectorized.contains(country.geometry,
                                                 lon[unset], lat[unset])
            region_id[unset[select]] = natearth_country_to_int(country)
        region_id[region_id == -1] = 0
    return region_id

def get_admin1_info(country_names):
    """Provide Natural Earth registry info and shape files for admin1 regions

    Parameters
    ----------
    country_names : list
        list with ISO3 names of countries, e.g. ['ZWE', 'GBR', 'VNM', 'UZB']

    Returns
    -------
    admin1_info : dict
        Data according to records in Natural Earth database.
    admin1_shapes : dict
        Shape according to Natural Earth.
    """

    if isinstance(country_names, str):
        country_names = [country_names]
    admin1_file = shapereader.natural_earth(resolution='10m',
                                            category='cultural',
                                            name='admin_1_states_provinces')
    admin1_recs = shapefile.Reader(admin1_file)
    admin1_info = dict()
    admin1_shapes = dict()
    for iso3 in country_names:
        admin1_info[iso3] = list()
        admin1_shapes[iso3] = list()
        for rec, rec_shp in zip(admin1_recs.records(), admin1_recs.shapes()):
            if rec['adm0_a3'] == iso3:
                admin1_info[iso3].append(rec)
                admin1_shapes[iso3].append(rec_shp)
    return admin1_info, admin1_shapes

def get_resolution_1d(coords, min_resol=1.0e-8):
    """Compute resolution of scalar grid

    Parameters
    ----------
    coords : np.array
        scalar coordinates
    min_resol : float, optional
        minimum resolution to consider. Default: 1.0e-8.

    Returns
    -------
    res : float
        Resolution of given grid.
    """
    res = np.diff(np.unique(coords))
    diff = np.diff(coords)
    mask = (res > min_resol) & np.isin(res, np.abs(diff))
    return diff[np.abs(diff) == res[mask].min()][0]


def get_resolution(*coords, min_resol=1.0e-8):
    """Compute resolution of n-d grid points

    Parameters
    ----------
    X, Y, ... : np.array
        Scalar coordinates in each axis
    min_resol : float, optional
        minimum resolution to consider. Default: 1.0e-8.

    Returns
    -------
    resolution : pair of floats
        Resolution in each coordinate direction.
    """
    return tuple([get_resolution_1d(c, min_resol=min_resol) for c in coords])


def pts_to_raster_meta(points_bounds, res):
    """Transform vector data coordinates to raster.

    If a raster of the given resolution doesn't exactly fit the given bounds, the raster might have
    slightly larger (but never smaller) bounds.

    Parameters
    ----------
    points_bounds : tuple
        points total bounds (xmin, ymin, xmax, ymax)
    res : tuple
        resolution of output raster (xres, yres)

    Returns
    -------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    ras_trans : affine.Affine
        Affine transformation defining the raster.
    """
    Affine = rasterio.Affine
    bounds = np.asarray(points_bounds).reshape(2, 2)
    res = np.asarray(res).ravel()
    if res.size == 1:
        res = np.array([res[0], res[0]])
    sizes = bounds[1, :] - bounds[0, :]
    nsteps = np.floor(sizes / np.abs(res)) + 1
    nsteps[np.abs(nsteps * res) < sizes + np.abs(res) / 2] += 1
    bounds[:, res < 0] = bounds[::-1, res < 0]
    origin = bounds[0, :] - res[:] / 2
    ras_trans = Affine.translation(*origin) * Affine.scale(*res)
    return int(nsteps[1]), int(nsteps[0]), ras_trans

def raster_to_meshgrid(transform, width, height):
    """Get coordinates of grid points in raster

    Parameters
    ----------
    transform : affine.Affine
        Affine transform defining the raster.
    width : int
        Number of points in first coordinate axis.
    height : int
        Number of points in second coordinate axis.

    Returns
    -------
    x : np.array
        x-coordinates of grid points.
    y : np.array
        y-coordinates of grid points.
    """
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax = xmin + width * xres
    ymax = ymin + height * yres
    return np.meshgrid(np.arange(xmin + xres / 2, xmax, xres),
                       np.arange(ymin + yres / 2, ymax, yres))


def to_crs_user_input(crs_obj):
    """Returns a crs string or dictionary from a hdf5 file object.

    bytes are decoded to str
    if the string starts with a '{' it is assumed to be a dumped string from a dictionary
    and ast is used to parse it.

    Parameters
    ----------
    crs_obj : int, dict or str or bytes
        the crs object to be converted user input

    Returns
    -------
    str or dict
        to eventually be used as argument of rasterio.crs.CRS.from_user_input
        and pyproj.crs.CRS.from_user_input

    Raises
    ------
    ValueError
        if type(crs_obj) has the wrong type
    """
    if type(crs_obj) in [dict, int]:
        return crs_obj

    crs_string = crs_obj.decode() if isinstance(crs_obj, bytes) else crs_obj

    if not isinstance(crs_string, str):
        raise ValueError(f"crs has unhandled data set type: {type(crs_string)}")

    if crs_string[0] == '{':
        return ast.literal_eval(crs_string)

    return crs_string


def equal_crs(crs_one, crs_two):
    """Compare two crs

    Parameters
    ----------
    crs_one : dict, str or int
        user crs
    crs_two : dict, str or int
        user crs

    Returns
    -------
    equal : bool
        Whether the two specified CRS are equal according tho rasterio.crs.CRS.from_user_input
    """
    if crs_one is None:
        return crs_two is None
    return rasterio.crs.CRS.from_user_input(crs_one) == rasterio.crs.CRS.from_user_input(crs_two)

def _read_raster_reproject(src, src_crs, dst_meta, band=None, geometry=None, dst_crs=None,
                           transform=None, resampling=rasterio.warp.Resampling.nearest):
    """Helper function for `read_raster`."""
    if not band:
        band = [1]
    if not dst_crs:
        dst_crs = src_crs
    if not transform:
        transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)
    else:
        transform, width, height = transform
    dst_meta.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
    })
    kwargs = {}
    if src.meta['nodata']:
        kwargs['src_nodata'] = src.meta['nodata']
        kwargs['dst_nodata'] = src.meta['nodata']

    intensity = np.zeros((len(band), height, width))
    for idx_band, i_band in enumerate(band):
        rasterio.warp.reproject(
            source=src.read(i_band),
            destination=intensity[idx_band, :],
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
            **kwargs)

        if dst_meta['nodata'] and np.isnan(dst_meta['nodata']):
            nodata_mask = np.isnan(intensity[idx_band, :])
        else:
            nodata_mask = (intensity[idx_band, :] == dst_meta['nodata'])
        intensity[idx_band, :][nodata_mask] = 0

    if geometry:
        intensity = intensity.astype('float32')
        # update driver to GTiff as netcdf does not work reliably
        dst_meta.update(driver='GTiff')
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**dst_meta) as dst:
                dst.write(intensity)

            with memfile.open() as dst:
                inten, mask_trans = rasterio.mask.mask(dst, geometry, crop=True, indexes=band)
                dst_meta.update({
                    "height": inten.shape[1],
                    "width": inten.shape[2],
                    "transform": mask_trans,
                })
        intensity = inten[range(len(band)), :]
        intensity = intensity.astype('float64')

        # reset nodata values again as driver Gtiff resets them again
        if dst_meta['nodata'] and np.isnan(dst_meta['nodata']):
            intensity[np.isnan(intensity)] = 0
        else:
            intensity[intensity == dst_meta['nodata']] = 0

    return intensity

def read_raster(file_name, band=None, src_crs=None, window=None, geometry=None,
                dst_crs=None, transform=None, width=None, height=None,
                resampling=rasterio.warp.Resampling.nearest):
    """Read raster of bands and set 0-values to the masked ones.

    Parameters
    ----------
    file_name : str
        name of the file
    band : list(int), optional
        band number to read. Default: 1
    window : rasterio.windows.Window, optional
        window to read
    geometry : shapely.geometry, optional
        consider pixels only in shape
    dst_crs : crs, optional
        reproject to given crs
    transform : rasterio.Affine
        affine transformation to apply
    wdith : float
        number of lons for transform
    height : float
        number of lats for transform
    resampling : rasterio.warp.Resampling optional
        resampling function used for reprojection to dst_crs

    Returns
    -------
    meta : dict
        Raster meta (height, width, transform, crs).
    data : np.array
        Each row corresponds to one band (raster points are flattened, can be
        reshaped to height x width).
    """
    if not band:
        band = [1]
    LOGGER.info('Reading %s', file_name)
    if Path(file_name).suffix == '.gz':
        file_name = '/vsigzip/' + str(file_name)

    with rasterio.Env():
        with rasterio.open(file_name, 'r') as src:
            dst_meta = src.meta.copy()

            if dst_crs or transform:
                LOGGER.debug('Reprojecting ...')

                src_crs = src.crs if src_crs is None else src_crs
                if not src_crs:
                    src_crs = rasterio.crs.CRS.from_user_input(DEF_CRS)
                transform = (transform, width, height) if transform else None
                inten = _read_raster_reproject(src, src_crs, dst_meta, band=band,
                                               geometry=geometry, dst_crs=dst_crs,
                                               transform=transform, resampling=resampling)
            else:
                if geometry:
                    inten, trans = rasterio.mask.mask(src, geometry, crop=True, indexes=band)
                    if dst_meta['nodata'] and np.isnan(dst_meta['nodata']):
                        inten[np.isnan(inten)] = 0
                    else:
                        inten[inten == dst_meta['nodata']] = 0

                else:
                    masked_array = src.read(band, window=window, masked=True)
                    inten = masked_array.data
                    inten[masked_array.mask] = 0

                    if window:
                        trans = rasterio.windows.transform(window, src.transform)
                    else:
                        trans = dst_meta['transform']

                dst_meta.update({
                    "height": inten.shape[1],
                    "width": inten.shape[2],
                    "transform": trans,
                })

    if not dst_meta['crs']:
        dst_meta['crs'] = rasterio.crs.CRS.from_user_input(DEF_CRS)

    intensity = inten[range(len(band)), :]
    dst_shape = (len(band), dst_meta['height'] * dst_meta['width'])

    return dst_meta, intensity.reshape(dst_shape)

def read_raster_bounds(path, bounds, res=None, bands=None):
    """Read raster file within given bounds and refine to given resolution

    Makes sure that the extent of pixel centers covers the specified regions

    Parameters
    ----------
    path : str
        Path to raster file to open with rasterio.
    bounds : tuple
        (xmin, ymin, xmax, ymax)
    res : float, optional
        Resolution of output. Default: Resolution of input raster file.
    bands : list of int, optional
        Bands to read from the input raster file. Default: [1]

    Returns
    -------
    data : 3d np.array
        First dimension is for the selected raster bands. Second dimension is y (lat) and third
        dimension is x (lon).
    transform : rasterio.Affine
        Affine transformation defining the output raster data.
    """
    if Path(path).suffix == '.gz':
        path = '/vsigzip/' + str(path)
    if not bands:
        bands = [1]
    resampling = rasterio.warp.Resampling.bilinear
    with rasterio.open(path, 'r') as src:
        if res:
            if not isinstance(res, tuple):
                res = (res, res)
        else:
            res = (src.transform[0], src.transform[4])
        res = (np.abs(res[0]), np.abs(res[1]))

        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        shape = (int(np.ceil(height / res[1]) + 1),
                 int(np.ceil(width / res[0]) + 1))

        # make sure that the extent of pixel centers covers the specified regions
        extra = (0.5 * ((shape[1] - 1) * res[0] - width),
                 0.5 * ((shape[0] - 1) * res[1] - height))
        bounds = (bounds[0] - extra[0] - 0.5 * res[0], bounds[1] - extra[1] - 0.5 * res[1],
                  bounds[2] + extra[0] + 0.5 * res[0], bounds[3] + extra[1] + 0.5 * res[1])

        data = np.zeros((len(bands),) + shape, dtype=src.dtypes[0])
        res = (np.sign(src.transform[0]) * res[0], np.sign(src.transform[4]) * res[1])
        transform = rasterio.Affine(res[0], 0, bounds[0] if res[0] > 0 else bounds[2],
                                    0, res[1], bounds[1] if res[1] > 0 else bounds[3])
        crs = DEF_CRS if src.crs is None else src.crs
        for iband, band in enumerate(bands):
            rasterio.warp.reproject(
                source=rasterio.band(src, band),
                destination=data[iband],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=resampling)
    return data, transform

def read_raster_sample(path, lat, lon, intermediate_res=None, method='linear', fill_value=None):
    """Read point samples from raster file.

    Parameters
    ----------
    path : str
        path of the raster file
    lat : np.array
        latitudes in file's CRS
    lon : np.array
        longitudes in file's CRS
    intermediate_res : float, optional
        If given, the raster is not read in its original resolution but in the given one. This can
        increase performance for files of very high resolution.
    method : str, optional
        The interpolation method, passed to scipy.interp.interpn. Default: 'linear'.
    fill_value : numeric, optional
        The value used outside of the raster bounds. Default: The raster's nodata value or 0.

    Returns
    -------
    values : np.array of same length as lat
        Interpolated raster values for each given coordinate point.
    """
    if lat.size == 0:
        return np.zeros_like(lat)

    LOGGER.info('Sampling from %s', path)
    if Path(path).suffix == '.gz':
        path = '/vsigzip/' + str(path)

    with rasterio.open(path, "r") as src:
        if intermediate_res is None:
            xres, yres = np.abs(src.transform[0]), np.abs(src.transform[4])
        else:
            xres = yres = intermediate_res
        bounds = (lon.min() - 2 * xres, lat.min() - 2 * yres,
                  lon.max() + 2 * xres, lat.max() + 2 * yres)
        win = src.window(*bounds).round_offsets(op='ceil').round_shape(op='floor')
        win_transform = src.window_transform(win)
        intermediate_shape = None
        if intermediate_res is not None:
            win_bounds = src.window_bounds(win)
            win_width, win_height = win_bounds[2] - win_bounds[0], win_bounds[3] - win_bounds[1]
            intermediate_shape = (int(np.ceil(win_height / intermediate_res)),
                                  int(np.ceil(win_width / intermediate_res)))
        data = src.read(1, out_shape=intermediate_shape, boundless=True, window=win)
        if fill_value is not None:
            data[data == src.meta['nodata']] = fill_value
        else:
            fill_value = src.meta['nodata']


    if intermediate_res is not None:
        xres, yres = win_width / data.shape[1], win_height / data.shape[0]
        xres, yres = np.sign(win_transform[0]) * xres, np.sign(win_transform[4]) * yres
        win_transform = rasterio.Affine(xres, 0, win_transform[2],
                                        0, yres, win_transform[5])
    fill_value = fill_value if fill_value else 0
    return interp_raster_data(data, lat, lon, win_transform, method=method, fill_value=fill_value)

def interp_raster_data(data, interp_y, interp_x, transform, method='linear', fill_value=0):
    """Interpolate raster data, given as array and affine transform

    Parameters
    ----------
    data : np.array
        2d numpy array containing the values
    interp_y : np.array
        y-coordinates of points (corresp. to first axis of data)
    interp_x : np.array
        x-coordinates of points (corresp. to second axis of data)
    transform : affine.Affine
        affine transform defining the raster
    method : str, optional
        The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.
    fill_value : numeric, optional
        The value used outside of the raster
            bounds. Default: 0.

    Returns
    -------
    values : np.array
        Interpolated raster values for each given coordinate point.
    """
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax = xmin + data.shape[1] * xres
    ymax = ymin + data.shape[0] * yres
    data = np.pad(data, 1, mode='edge')

    if yres < 0:
        yres = -yres
        ymax, ymin = ymin, ymax
        data = np.flipud(data)
    if xres < 0:
        xres = -xres
        xmax, xmin = xmin, xmax
        data = np.fliplr(data)
    y_dim = ymin - yres / 2 + yres * np.arange(data.shape[0])
    x_dim = xmin - xres / 2 + xres * np.arange(data.shape[1])

    data = np.array(data, dtype=np.float64)
    data[np.isnan(data)] = fill_value
    return scipy.interpolate.interpn((y_dim, x_dim), data, np.vstack([interp_y, interp_x]).T,
                                     method=method, bounds_error=False, fill_value=fill_value)

def refine_raster_data(data, transform, res, method='linear', fill_value=0):
    """Refine raster data, given as array and affine transform

    Parameters
    ----------
    data : np.array
        2d array containing the values
    transform : affine.Affine
        affine transform defining the raster
    res : float or pair of floats
        new resolution
    method : str, optional
        The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.

    Returns
    -------
    new_data : np.array
        2d array containing the interpolated values.
    new_transform : affine.Affine
        Affine transform defining the refined raster.
    """
    xres, _, xmin, _, yres, ymin = transform[:6]
    xmax = xmin + data.shape[1] * xres
    ymax = ymin + data.shape[0] * yres
    if not isinstance(res, tuple):
        res = (np.sign(xres) * res, np.sign(yres) * res)
    new_dimx = np.arange(xmin + res[0] / 2, xmax, res[0])
    new_dimy = np.arange(ymin + res[1] / 2, ymax, res[1])
    new_shape = (new_dimy.size, new_dimx.size)
    new_x, new_y = [ar.ravel() for ar in np.meshgrid(new_dimx, new_dimy)]
    new_transform = rasterio.Affine(res[0], 0, xmin, 0, res[1], ymin)
    new_data = interp_raster_data(data, new_y, new_x, transform, method=method,
                                  fill_value=fill_value)
    new_data = new_data.reshape(new_shape)
    return new_data, new_transform

def read_vector(file_name, field_name, dst_crs=None):
    """Read vector file format supported by fiona.

    Parameters
    ----------
    file_name : str
        vector file with format supported by fiona and 'geometry' field.
    field_name : list(str)
        list of names of the columns with values.
    dst_crs : crs, optional
        reproject to given crs

    Returns
    -------
    lat : np.array
        Latitudinal coordinates.
    lon : np.array
        Longitudinal coordinates.
    geometry : GeoSeries
        Shape geometries.
    value : np.array
        Values associated to each shape.
    """
    LOGGER.info('Reading %s', file_name)
    data_frame = gpd.read_file(file_name)
    if not data_frame.crs:
        data_frame.crs = DEF_CRS
    if dst_crs is None:
        geometry = data_frame.geometry
    else:
        geometry = data_frame.geometry.to_crs(dst_crs)
    lat, lon = geometry[:].y.values, geometry[:].x.values
    value = np.zeros([len(field_name), lat.size])
    for i_inten, inten in enumerate(field_name):
        value[i_inten, :] = data_frame[inten].values
    return lat, lon, geometry, value

def write_raster(file_name, data_matrix, meta, dtype=np.float32):
    """Write raster in GeoTiff format.

    Parameters
    ----------
    file_name : str
        File name to write.
    data_matrix : np.array
        2d raster data. Either containing one band, or every row is a band and the column
        represents the grid in 1d.
    meta : dict
        rasterio meta dictionary containing raster properties: width, height, crs and transform
        must be present at least. Include `compress="deflate"` for compressed output.
    dtype : numpy dtype, optional
        A numpy dtype. Default: np.float32
    """
    LOGGER.info('Writting %s', file_name)
    if data_matrix.shape != (meta['height'], meta['width']):
        # every row is an event (from hazard intensity or fraction) == band
        shape = (data_matrix.shape[0], meta['height'], meta['width'])
    else:
        shape = (1, meta['height'], meta['width'])
    dst_meta = copy.deepcopy(meta)
    dst_meta.update(driver='GTiff', dtype=dtype, count=shape[0])
    data_matrix = np.asarray(data_matrix, dtype=dtype).reshape(shape)
    with rasterio.open(file_name, 'w', **dst_meta) as dst:
        dst.write(data_matrix, indexes=np.arange(1, shape[0] + 1))

def points_to_raster(points_df, val_names=None, res=0.0, raster_res=0.0, crs=DEF_CRS,
                     scheduler=None):
    """Compute raster (as data and transform) from GeoDataFrame.

    Parameters
    ----------
    points_df : GeoDataFrame
        contains columns latitude, longitude and those listed in the parameter `val_names`.
    val_names : list of str, optional
        The names of columns in `points_df` containing values. The raster will contain one band per
        column. Default: ['value']
    res : float, optional
        resolution of current data in units of latitude and longitude, approximated if not
        provided.
    raster_res : float, optional
        desired resolution of the raster
    crs : object (anything accepted by pyproj.CRS.from_user_input), optional
        If given, overwrites the CRS information given in `points_df`. If no CRS is explicitly
        given and there is no CRS information in `points_df`, the CRS is assumed to be EPSG:4326
        (lat/lon). Default: None
    scheduler : str
        used for dask map_partitions. “threads”, “synchronous” or “processes”

    Returns
    -------
    data : np.array
        2d array containing the raster values.
    transform : affine.Affine
        Affine transform defining the raster coordinates.
    """
    if not val_names:
        val_names = ['value']
    if not res:
        res = np.abs(get_resolution(points_df.latitude.values,
                                    points_df.longitude.values)).min()
    if not raster_res:
        raster_res = res

    def apply_box(df_exp):
        fun = lambda r: Point(r.longitude, r.latitude).buffer(res / 2).envelope
        return df_exp.apply(fun, axis=1)

    LOGGER.info('Raster from resolution %s to %s.', res, raster_res)
    df_poly = gpd.GeoDataFrame(points_df[val_names])
    if not scheduler:
        df_poly['geometry'] = apply_box(points_df)
    else:
        ddata = dd.from_pandas(points_df[['latitude', 'longitude']],
                               npartitions=cpu_count())
        df_poly['geometry'] = ddata.map_partitions(apply_box, meta=Polygon) \
                                   .compute(scheduler=scheduler)
    df_poly.set_crs(crs if crs else points_df.crs if points_df.crs else DEF_CRS, inplace=True)

    # renormalize longitude if necessary
    if df_poly.crs == DEF_CRS:
        xmin, ymin, xmax, ymax = latlon_bounds(points_df.latitude.values,
                                               points_df.longitude.values)
        x_mid = 0.5 * (xmin + xmax)
        df_poly = df_poly.to_crs({"proj": "longlat", "lon_wrap": x_mid})
    else:
        xmin, ymin, xmax, ymax = (points_df.longitude.min(), points_df.latitude.min(),
                                  points_df.longitude.max(), points_df.latitude.max())

    # construct raster
    rows, cols, ras_trans = pts_to_raster_meta((xmin, ymin, xmax, ymax),
                                               (raster_res, -raster_res))
    raster_out = np.zeros((len(val_names), rows, cols))

    # TODO: parallel rasterize
    for i_val, val_name in enumerate(val_names):
        raster_out[i_val, :, :] = rasterio.features.rasterize(
            list(zip(df_poly.geometry, df_poly[val_name])),
            out_shape=(rows, cols),
            transform=ras_trans,
            fill=0,
            all_touched=True,
            dtype=rasterio.float32)

    meta = {
        'crs': df_poly.crs,
        'height': rows,
        'width': cols,
        'transform': ras_trans,
    }
    return raster_out, meta

def subraster_from_bounds(transform, bounds):
    """Compute a subraster definition from a given reference transform and bounds.

    Parameters
    ----------
    transform : rasterio.Affine
        Affine transformation defining the reference grid.
    bounds : tuple of floats (xmin, ymin, xmax, ymax)
        Bounds of the subraster in units and CRS of the reference grid.

    Returns
    -------
    dst_transform : rasterio.Affine
        Subraster affine transformation.
    dst_shape : tuple of ints (height, width)
        Number of pixels of subraster in vertical and horizontal direction.
    """
    window = rasterio.windows.from_bounds(*bounds, transform)

    # align the window bounds to the raster by rounding
    col_min, col_max = np.round(window.col_off), np.round(window.col_off + window.width)
    row_min, row_max = np.round(window.row_off), np.round(window.row_off + window.height)
    window = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)

    dst_transform = rasterio.windows.transform(window, transform)
    dst_shape = (int(window.height), int(window.width))
    return dst_transform, dst_shape

def align_raster_data(source, src_crs, src_transform, dst_crs=None, dst_resolution=None,
                      dst_bounds=None, global_origin=(-180, 90), resampling=None, conserve=None,
                      **kwargs):
    """Reproject 2D np.ndarray to be aligned to a reference grid.

    This function ensures that reprojected data with the same dst_resolution and global_origins are
    aligned to the same global grid, i.e., no offset between destination grid points for different
    source grids that are projected to the same target resolution.

    Note that the origin is required to be in the upper left corner. The result is always oriented
    left to right (west to east) and top to bottom (north to south).

    Parameters
    ----------
    source : np.ndarray
        The source is a 2D ndarray containing the values to be reprojected.
    src_crs : CRS or dict
        Source coordinate reference system, in rasterio dict format.
    src_transform : rasterio.Affine
        Source affine transformation.
    dst_crs : CRS, optional
        Target coordinate reference system, in rasterio dict format. Default: `src_crs`
    dst_resolution : tuple (x_resolution, y_resolution) or float, optional
        Target resolution (positive pixel sizes) in units of the target CRS.
        Default: `(abs(src_transform[0]), abs(src_transform[4]))`
    dst_bounds : tuple of floats (xmin, ymin, xmax, ymax), optional
        Bounds of the target raster in units of the target CRS. By default, the source's bounds
        are reprojected to the target CRS.
    global_origin : tuple (west, north) of floats, optional
        Coordinates of the reference grid's upper left corner. Default: (-180, 90). Make sure to
        change `global_origin` for non-geographical CRS!
    resampling : int, rasterio.enums.Resampling or str, optional
        Resampling method to use. String values like `"nearest"` or `"bilinear"` are resolved to
        attributes of `rasterio.enums.Resampling. Default: `rasterio.enums.Resampling.nearest`
    conserve : str, optional
        If provided, conserve the source array's 'mean' or 'sum' in the transformed data or
        normalize the values of the transformed data ndarray ('norm').
        Default: None (no conservation)
    kwargs : dict, optional
        Additional arguments passed to `rasterio.warp.reproject`.

    Raises
    ------
    ValueError

    Returns
    -------
    destination : np.ndarray with same dtype as `source`
        The transformed 2D ndarray.
    dst_transform : rasterio.Affine
        Destination affine transformation.
    """
    if dst_crs is None:
        dst_crs = src_crs
    if (not dst_crs.is_geographic) and global_origin == (-180, 90):
        LOGGER.warning("Non-geographic destination CRS. Check global_origin!")
    if dst_resolution is None:
        dst_resolution = (np.abs(src_transform[0]), np.abs(src_transform[4]))
    if np.isscalar(dst_resolution):
        dst_resolution = (dst_resolution, dst_resolution)
    if isinstance(resampling, str):
        resampling = getattr(rasterio.warp.Resampling, resampling)

    # determine well-aligned subraster
    global_transform = rasterio.transform.from_origin(*global_origin, *dst_resolution)
    if dst_bounds is None:
        src_bounds = rasterio.transform.array_bounds(*source.shape, src_transform)
        dst_bounds = rasterio.warp.transform_bounds(src_crs, dst_crs, *src_bounds)
    dst_transform, dst_shape = subraster_from_bounds(global_transform, dst_bounds)

    destination = np.zeros(dst_shape, dtype=source.dtype)
    rasterio.warp.reproject(source=source,
                            destination=destination,
                            src_transform=src_transform,
                            src_crs=src_crs,
                            dst_transform=dst_transform,
                            dst_crs=dst_crs,
                            resampling=resampling,
                            **kwargs)

    if conserve == 'mean':
        destination *= source.mean() / destination.mean()
    elif conserve == 'sum':
        destination *= source.sum() / destination.sum()
    elif conserve == 'norm':
        destination *= 1.0 / destination.sum()
    elif conserve is not None:
        raise ValueError(f"Invalid value for conserve: {conserve}")
    return destination, dst_transform

def mask_raster_with_geometry(raster, transform, shapes, nodata=None, **kwargs):
    """
    Change values in `raster` that are outside of given `shapes` to `nodata`.

    This function is a wrapper for rasterio.mask.mask to allow for
    in-memory processing. This is done by first writing data to memfile and then
    reading from it before the function call to rasterio.mask.mask().
    The MemoryFile will be discarded after exiting the with statement.

    Parameters
    ----------
    raster : numpy.ndarray
        raster to be masked with dim: [H, W].
    transform : affine.Affine
         the transform of the raster.
    shapes : GeoJSON-like dict or an object that implements the Python geo
        interface protocol (such as a Shapely Polygon)
        Passed to rasterio.mask.mask
    nodata : int or float, optional
        Passed to rasterio.mask.mask:
        Data points outside `shapes` are set to `nodata`.
    kwargs : optional
        Passed to rasterio.mask.mask.

    Returns
    -------
    masked: numpy.ndarray or numpy.ma.MaskedArray
        raster with dim: [H, W] and points outside shapes set to `nodata`
    """
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=raster.dtype,
            transform=transform,
        ) as dataset:
            dataset.write(raster, 1)
        with memfile.open() as dataset:
            output, _ = rasterio.mask.mask(dataset, shapes, nodata=nodata, **kwargs)
    return output.squeeze(0)

def set_df_geometry_points(df_val, scheduler=None, crs=None):
    """Set given geometry to given dataframe using dask if scheduler.

    Parameters
    ----------
    df_val : GeoDataFrame
        contains latitude and longitude columns
    scheduler : str, optional
        used for dask map_partitions. “threads”, “synchronous” or “processes”
    crs : object (anything readable by pyproj4.CRS.from_user_input), optional
        Coordinate Reference System, if omitted or None: df_val.geometry.crs
    """
    LOGGER.info('Setting geometry points.')

    # keep the original crs if any
    if crs is None:
        try:
            crs = df_val.geometry.crs
        except AttributeError:
            crs = None

    # work in parallel
    if scheduler:
        def apply_point(df_exp):
            return df_exp.apply(lambda row: Point(row.longitude, row.latitude), axis=1)

        ddata = dd.from_pandas(df_val, npartitions=cpu_count())
        df_val['geometry'] = ddata.map_partitions(apply_point, meta=Point) \
                                  .compute(scheduler=scheduler)
    # single process
    else:
        df_val['geometry'] = gpd.GeoSeries(
            gpd.points_from_xy(df_val.longitude, df_val.latitude), index=df_val.index, crs=crs)

    # set crs
    if crs:
        df_val.set_crs(crs, inplace=True)


def fao_code_def():
    """Generates list of FAO country codes and corresponding ISO numeric-3 codes.

    Returns
    -------
    iso_list : list
        list of ISO numeric-3 codes
    faocode_list : list
        list of FAO country codes
    """
    # FAO_FILE2: contains FAO country codes and correstponding ISO3 Code
    #           (http://www.fao.org/faostat/en/#definitions)
    fao_file = pd.read_csv(SYSTEM_DIR.joinpath("FAOSTAT_data_country_codes.csv"))
    fao_code = getattr(fao_file, 'Country Code').values
    fao_iso = (getattr(fao_file, 'ISO3 Code').values).tolist()

    # create a list of ISO3 codes and corresponding fao country codes
    iso_list = list()
    faocode_list = list()
    for idx, iso in enumerate(fao_iso):
        if isinstance(iso, str):
            iso_list.append(country_to_iso(iso, "numeric"))
            faocode_list.append(int(fao_code[idx]))

    return iso_list, faocode_list

def country_faocode2iso(input_fao):
    """Convert FAO country code to ISO numeric-3 codes.

    Parameters
    ----------
    input_fao : int or array
        FAO country codes of countries (or single code)

    Returns
    -------
    output_iso : int or array
        ISO numeric-3 codes of countries (or single code)
    """

    # load relation between ISO numeric-3 code and FAO country code
    iso_list, faocode_list = fao_code_def()

    # determine the fao country code for the input str or list
    output_iso = np.zeros(len(input_fao))
    for item, faocode in enumerate(input_fao):
        idx = np.where(faocode_list == faocode)[0]
        if len(idx) == 1:
            output_iso[item] = iso_list[idx[0]]

    return output_iso

def country_iso2faocode(input_iso):
    """Convert ISO numeric-3 codes to FAO country code.

    Parameters
    ----------
    input_iso : int or array
        ISO numeric-3 codes of countries (or single code)

    Returns
    -------
    output_faocode : int or array
        FAO country codes of countries (or single code)
    """
    # load relation between ISO numeric-3 code and FAO country code
    iso_list, faocode_list = fao_code_def()

    # determine the fao country code for the input str or list
    output_faocode = np.zeros(len(input_iso))
    for item, iso in enumerate(input_iso):
        idx = np.where(iso_list == iso)[0]
        if len(idx) == 1:
            output_faocode[item] = faocode_list[idx[0]]

    return output_faocode
