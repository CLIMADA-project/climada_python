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

Define functions to handle with coordinates
"""

import copy
import logging
import math
from multiprocessing import cpu_count
import os
import zipfile

from cartopy.io import shapereader
import dask.dataframe as dd
from fiona.crs import from_epsg
import geopandas as gpd
from iso3166 import countries as iso_cntry
import numpy as np
import pandas as pd
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
                                    RIVER_FLOOD_REGIONS_CSV)
from climada.util.files_handler import download_file
import climada.util.hdf5_handler as hdf5
from climada.util.constants import DATA_DIR

pd.options.mode.chained_assignment = None

LOGGER = logging.getLogger(__name__)

NE_EPSG = 4326
"""Natural Earth CRS EPSG"""

NE_CRS = from_epsg(NE_EPSG)
"""Natural Earth CRS"""

TMP_ELEVATION_FILE = os.path.join(SYSTEM_DIR, 'tmp_elevation.tif')
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

    The input data is modified in place (!) using the following operations:

        (lon) -> (lon ± 360)

    Parameters:
        lon (np.array): Longitudinal coordinates
        center (float, optional): Central longitude value to use instead of 0.

    Returns:
        np.array (same as input)
    """
    bounds = (center - 180, center + 180)
    maxiter = 10
    i = 0
    while True:
        msk1 = (lon > bounds[1])
        lon[msk1] -= 360
        msk2 = (lon <= bounds[0])
        lon[msk2] += 360
        if msk1.sum() == 0 and msk2.sum() == 0:
            break
        i += 1
        if i > maxiter:
            LOGGER.warning("lon_normalize: killed before finishing")
            break
    return lon

def latlon_bounds(lat, lon, buffer=0.0):
    """Bounds of a set of degree values, respecting the periodicity in longitude

    The longitudinal upper bound may be 180 or larger to make sure that the upper bound is always
    larger than the lower bound. The lower longitudinal bound will never lie below -180 and it will
    only assume the value -180 if the specified buffering enforces it.

    Note that, as a consequence of this, the returned bounds do not satisfy the inequality
    `lon_min <= lon <= lon_max` in general!

    Usually, an application of this function is followed by a renormalization of longitudinal
    values around the longitudinal middle value:

    >>> bounds = latlon_bounds(lat, lon)
    >>> lon_mid = 0.5 * (bounds[0] + bounds[2])
    >>> lon = lon_normalize(lon, center=lon_mid)
    >>> np.all((bounds[0] <= lon) & (lon <= bounds[2]))

    Example:
        >>> latlon_bounds(np.array([0, -2, 5]), np.array([-179, 175, 178]))
        (175, -2, 181, 5)
        >>> latlon_bounds(np.array([0, -2, 5]), np.array([-179, 175, 178]), buffer=1)
        (174, -3, 182, 6)

    Parameters:
        lat (np.array): Latitudinal coordinates
        lon (np.array): Longitudinal coordinates
        buffer (float, optional): Buffer to add to all sides of the bounding box. Default: 0.0.

    Returns:
        tuple (lon_min, lat_min, lon_max, lat_max)
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
    return (lon_min, max(lat.min() - buffer, -90), lon_max, min(lat.max() + buffer, 90))

def dist_approx(lat1, lon1, lat2, lon2, log=False, normalize=True,
                method="equirect"):
    """Compute approximation of geodistance in km

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
        * "equirect": equirectangular; very fast, good only at small distances.
        * "geosphere": spherical approximation, slower, but much higher accuracy.
        Default: "equirect".

    Returns
    -------
    dists : ndarray of floats, shape (nbatch, nx, ny)
        Approximate distances in km.
    vtan : ndarray of floats, shape (nbatch, nx, ny, 2)
        If `log` is True, tangential vectors at first points in local
        lat-lon coordinate system.
    """
    if method == "equirect":
        if normalize:
            lon_normalize(lon1)
            lon_normalize(lon2)
        d_lat = lat2[:, None] - lat1[:, :, None]
        d_lon = lon2[:, None] - lon1[:, :, None]
        fact1 = np.heaviside(d_lon - 180, 0)
        fact2 = np.heaviside(-d_lon - 180, 0)
        d_lon -= (fact1 - fact2) * 360
        d_lon *= np.cos(np.radians(lat1[:, :, None]))
        dist_km = np.sqrt(d_lon**2 + d_lat**2) * ONE_LAT_KM
        if log:
            vtan = np.stack([d_lat, d_lon], axis=-1) * ONE_LAT_KM
    elif method == "geosphere":
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = 0.5 * (lat2[:, None] - lat1[:, :, None])
        dlon = 0.5 * (lon2[:, None] - lon1[:, :, None])
        # haversine formula:
        hav = np.sin(dlat)**2 \
            + np.cos(lat1[:, :, None]) * np.cos(lat2[:, None]) * np.sin(dlon)**2
        dist_km = np.degrees(2 * np.arcsin(np.sqrt(hav))) * ONE_LAT_KM
        if log:
            vec1, vbasis = latlon_to_geosph_vector(lat1, lon1, rad=True, basis=True)
            vec2 = latlon_to_geosph_vector(lat2, lon2, rad=True)
            scal = 1 - 2 * hav
            fact = dist_km / np.fmax(np.spacing(1), np.sqrt(1 - scal**2))
            vtan = fact[..., None] * (vec2[:, None] - scal[..., None] * vec1[:, :, None])
            vtan = np.einsum('nkli,nkji->nklj', vtan, vbasis)
    else:
        LOGGER.error("Unknown distance approximation method: %s", method)
        raise KeyError
    return (dist_km, vtan) if log else dist_km

def grid_is_regular(coord):
    """Return True if grid is regular. If True, returns height and width.

    Parameters
    ----------
    coord : np.array
        Each row is a lat-lon-pair.

    Returns
    -------
    regular : bool
        Whether the grid is regular. Only in this case, the following
        width and height are reliable.
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

    Parameters:
        bounds (tuple): min_lon, min_lat, max_lon, max_lat in EPSG:4326
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            110m, i.e. 1:110.000.000

    Returns:
        GeoDataFrame
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

    Parameters:
        lon (float): longitude point in EPSG 4326
        lat (float): latitude of point (lat, lon) in EPSG 4326

    Return:
        int
    """
    epsg_utm_base = 32601 + (0 if lat >= 0 else 100)
    return epsg_utm_base + (math.floor((lon + 180) / 6) % 60)

def utm_zones(wgs_bounds):
    """Get EPSG code and bounds of UTM zones covering specified region

    Parameters:
        wgs_bounds (tuple): lon_min, lat_min, lon_max, lat_max

    Returns:
        list of pairs (zone_epsg, zone_wgs_bounds)
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

    Parameters:
        coord_lat (GeoDataFrame or np.array or float):
            - GeoDataFrame with geometry column in epsg:4326
            - np.array with two columns, first for latitude of each point and
                second with longitude in epsg:4326
            - np.array with one dimension containing latitudes in epsg:4326
            - float with a latitude value in epsg:4326
        lon (np.array or float, optional):
            - np.array with one dimension containing longitudes in epsg:4326
            - float with a longitude value in epsg:4326
        signed (bool): If True, distance is signed with positive values off shore and negative
            values on land. Default: False

    Returns:
        np.array
    """
    if isinstance(coord_lat, (gpd.GeoDataFrame, gpd.GeoSeries)):
        if not equal_crs(coord_lat.crs, NE_CRS):
            LOGGER.error('Input CRS is not %s', str(NE_CRS))
            raise ValueError
        geom = coord_lat
    else:
        if lon is None:
            if isinstance(coord_lat, np.ndarray) and coord_lat.shape[1] == 2:
                lat, lon = coord_lat[:, 0], coord_lat[:, 1]
            else:
                LOGGER.error('Missing longitude values.')
                raise ValueError
        else:
            lat, lon = [np.asarray(v).reshape(-1) for v in [coord_lat, lon]]
            if lat.size != lon.size:
                LOGGER.error('Mismatching input coordinates size: %s != %s',
                             lat.size, lon.size)
                raise ValueError
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

    Parameters:
        lat (np.array): latitudes in epsg:4326
        lon (np.array): longitudes in epsg:4326
        highres (bool, optional): Use full resolution of NASA data (much
            slower). Default: False.
        signed (bool): If True, distance is signed with positive values off shore and negative
            values on land. Default: False

    Returns:
        np.array
    """
    lat, lon = [np.asarray(ar).ravel() for ar in [lat, lon]]
    lon = lon_normalize(lon.copy())

    # TODO move URL to config
    zipname = "GMT_intermediate_coast_distance_01d.zip"
    tifname = "GMT_intermediate_coast_distance_01d.tif"
    url = "https://oceancolor.gsfc.nasa.gov/docs/distfromcoast/" + zipname
    path = os.path.join(SYSTEM_DIR, tifname)
    if not os.path.isfile(path):
        cwd = os.getcwd()
        os.chdir(SYSTEM_DIR)
        path_dwn = download_file(url)
        zip_ref = zipfile.ZipFile(path_dwn, 'r')
        zip_ref.extractall(SYSTEM_DIR)
        zip_ref.close()
        os.remove(path_dwn)
        os.chdir(cwd)

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
    """Get union of all the countries or the provided ones or the points inside
    the extent.

    Parameters:
        country_names (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        extent (tuple, optional): (min_lon, max_lon, min_lat, max_lat)
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            10m, i.e. 1:10.000.000

    Returns:
        shapely.geometry.multipolygon.MultiPolygon
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
    """Check if point is on land (True) or water (False) of provided coordinates.
    All globe considered if no input countries.

    Parameters:
        lat (np.array): latitude of points in epsg:4326
        lon (np.array): longitude of points in epsg:4326
        land_geom (shapely.geometry.multipolygon.MultiPolygon, optional):
            profiles of land.

    Returns:
        np.array(bool)
    """
    if lat.size != lon.size:
        LOGGER.error('Wrong size input coordinates: %s != %s.', lat.size,
                     lon.size)
        raise ValueError
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

def get_country_geometries(country_names=None, extent=None, resolution=10):
    """Returns a gpd GeoSeries of natural earth multipolygons of the
    specified countries, resp. the countries that lie within the specified
    extent. If no arguments are given, simply returns the whole natural earth
    dataset.
    Take heed: we assume WGS84 as the CRS unless the Natural Earth download
    utility from cartopy starts including the projection information. (They
    are saving a whopping 147 bytes by omitting it.) Same goes for UTF.

    Parameters:
        country_names (list, optional): list with ISO3 names of countries, e.g
            ['ZWE', 'GBR', 'VNM', 'UZB']
        extent (tuple, optional): (min_lon, max_lon, min_lat, max_lat) assumed
            to be in the same CRS as the natural earth data.
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            10m

    Returns:
        GeoDataFrame
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
    for idx in nat_earth[gap_mask].index:
        for col in ['ISO_A3', 'ADM0_A3', 'NAME']:
            try:
                num = iso_cntry.get(nat_earth.loc[idx, col]).numeric
            except KeyError:
                continue
            else:
                nat_earth.loc[idx, 'ISO_N3'] = num
                break

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
        ISO 3166-1 alpha-3 codes of countries, or internal numeric NatID if
        `iso` is set to False.
    regions : list, optional
        Region IDs.
    resolution : float, optional
        Resolution in arc-seconds, either 150 (default) or 360.
    iso : bool, optional
        If True, assume that countries are given by their ISO 3166-1 alpha-3
        codes (instead of the internal NatID). Default: True.
    rect : bool, optional
        If True, a rectangular box around the specified countries/regions is
        selected. Default: False.
    basemap : str, optional
        Choose between different data sources.
        Currently available: "isimip" and "natearth". Default: "natearth".

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
        hdf5_f = hdf5.read(base_file)
        meta = hdf5_f['meta']
        grid_shape = (meta['height'][0], meta['width'][0])
        transform = rasterio.Affine(*meta['transform'])
        region_id = hdf5_f['region_id'].reshape(grid_shape)
        lon, lat = raster_to_meshgrid(transform, grid_shape[1], grid_shape[0])
    elif basemap == "isimip":
        hdf5_f = hdf5.read(ISIMIP_GPWV3_NATID_150AS)
        dim_lon, dim_lat = hdf5_f['lon'], hdf5_f['lat']
        bounds = dim_lon.min(), dim_lat.min(), dim_lon.max(), dim_lat.max()
        orig_res = get_resolution(dim_lon, dim_lat)
        _, _, transform = pts_to_raster_meta(bounds, orig_res)
        grid_shape = (dim_lat.size, dim_lon.size)
        region_id = hdf5_f['NatIdGrid'].reshape(grid_shape).astype(int)
        region_id[region_id < 0] = 0
        natid2iso_alpha = country_natid2iso(list(range(231)))
        natid2iso = country_iso_alpha2numeric(natid2iso_alpha)
        natid2iso = np.array(natid2iso, dtype=int)
        region_id = natid2iso[region_id]
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
    countries = np.unique(country_iso_alpha2numeric(countries))

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
            LOGGER.error('Unknown region name: %s', region)
            raise KeyError
        isos += list(reg_info['ISO'][region_msk].values)
    return list(set(isos))

def country_iso_alpha2numeric(isos):
    """Convert ISO 3166-1 alpha-3 to numeric-3 codes

    Parameters:
        isos (str or list of str): ISO codes of countries (or single code).

    Returns:
        int or list of int
    """
    return_int = isinstance(isos, str)
    isos = [isos] if return_int else isos
    old_iso = {
        '': 0,  # Ocean or fill_value
        "ANT": 530,  # Netherlands Antilles: split up since 2010
        "SCG": 891,  # Serbia and Montenegro: split up since 2006
    }
    nums = []
    for iso in isos:
        if iso in old_iso:
            num = old_iso[iso]
        else:
            num = int(iso_cntry.get(iso).numeric)
        nums.append(num)
    return nums[0] if return_int else nums

def country_natid2iso(natids):
    """Convert internal NatIDs to ISO 3166-1 alpha-3 codes

    Parameters:
        natids (int or list of int): Internal NatIDs of countries (or single ID).

    Returns:
        str or list of str
    """
    return_str = isinstance(natids, int)
    natids = [natids] if return_str else natids
    isos = []
    for natid in natids:
        if natid < 0 or natid >= len(ISIMIP_NATID_TO_ISO):
            LOGGER.error('Unknown country NatID: %s', natid)
            raise KeyError
        isos.append(ISIMIP_NATID_TO_ISO[natid])
    return isos[0] if return_str else isos

def country_iso2natid(isos):
    """Convert ISO 3166-1 alpha-3 codes to internal NatIDs

    Parameters:
        isos (str or list of str): ISO codes of countries (or single code).

    Returns:
        int or list of int
    """
    return_int = isinstance(isos, str)
    isos = [isos] if return_int else isos
    natids = []
    for iso in isos:
        try:
            natids.append(ISIMIP_NATID_TO_ISO.index(iso))
        except ValueError:
            LOGGER.error('Unknown country ISO: %s', iso)
            raise KeyError
    return natids[0] if return_int else natids

NATEARTH_AREA_NONISO_NUMERIC = {
    "Akrotiri": 901,
    "Baikonur": 902,
    "Bajo Nuevo Bank": 903,
    "Clipperton I.": 904,
    "Coral Sea Is.": 905,
    "Cyprus U.N. Buffer Zone": 906,
    "Dhekelia": 907,
    "Indian Ocean Ter.": 908,
    "Kosovo": 983,  # Same as iso3166 package
    "N. Cyprus": 910,
    "Norway": 578,  # Bug in Natural Earth
    "Scarborough Reef": 912,
    "Serranilla Bank": 913,
    "Siachen Glacier": 914,
    "Somaliland": 915,
    "Spratly Is.": 916,
    "USNB Guantanamo Bay": 917,
}

def natearth_country_to_int(country):
    """Integer representation (ISO 3166, if possible) of Natural Earth GeoPandas country row

    Parameters:
        country (GeoSeries): Row from GeoDataFrame.

    Returns:
        int
    """
    if country.ISO_N3 != '-99':
        return int(country.ISO_N3)
    return NATEARTH_AREA_NONISO_NUMERIC[str(country.NAME)]

def get_country_code(lat, lon, gridded=False):
    """Provide numeric (ISO 3166) code for every point.

    Oceans get the value zero. Areas that are not in ISO 3166 are given values
    in the range above 900 according to NATEARTH_AREA_NONISO_NUMERIC.

    Parameters:
        lat (np.array): latitude of points in epsg:4326
        lon (np.array): longitude of points in epsg:4326
        gridded (bool): If True, interpolate precomputed gridded data which
            is usually much faster. Default: False.

    Returns:
        np.array(int)
    """
    lat, lon = [np.asarray(ar).ravel() for ar in [lat, lon]]
    LOGGER.info('Setting region_id %s points.', str(lat.size))
    if gridded:
        base_file = hdf5.read(NATEARTH_CENTROIDS[150])
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
    """Provide registry info and shape files for admin1 regions

    Parameters:
        country_names (list): list with ISO3 names of countries, e.g.
                ['ZWE', 'GBR', 'VNM', 'UZB']

    Returns:
        admin1_info (dict)
        admin1_shapes (dict)
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

    Parameters:
        coords (np.array): scalar coordinates
        min_resol (float, optional): minimum resolution to consider.
            Default: 1.0e-8.

    Returns:
        float
    """
    res = np.diff(np.unique(coords))
    diff = np.diff(coords)
    mask = (res > min_resol) & np.isin(res, np.abs(diff))
    return diff[np.abs(diff) == res[mask].min()][0]


def get_resolution(*coords, min_resol=1.0e-8):
    """Compute resolution of 2-d grid points

    Parameters:
        X, Y, ... (np.array): scalar coordinates in each axis
        min_resol (float, optional): minimum resolution to consider.
            Default: 1.0e-8.

    Returns:
        pair of floats
    """
    return tuple([get_resolution_1d(c, min_resol=min_resol) for c in coords])


def pts_to_raster_meta(points_bounds, res):
    """Transform vector data coordinates to raster. Returns number of rows,
    columns and affine transformation

    If a raster of the given resolution doesn't exactly fit the given bounds,
    the raster might have slightly larger (but never smaller) bounds.

    Parameters:
        points_bounds (tuple): points total bounds (xmin, ymin, xmax, ymax)
        res (tuple): resolution of output raster (xres, yres)

    Returns:
        int, int, affine.Affine
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

def equal_crs(crs_one, crs_two):
    """Compare two crs

    Parameters:
        crs_one (dict or string or wkt): user crs
        crs_two (dict or string or wkt): user crs

    Returns:
        bool
    """
    return rasterio.crs.CRS.from_user_input(crs_one) == rasterio.crs.CRS.from_user_input(crs_two)

def _read_raster_reproject(src, src_crs, dst_meta, band=None, geometry=None, dst_crs=None,
                           transform=None, resampling=rasterio.warp.Resampling.nearest):
    """Helper function for `read_raster`"""
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
    """Read raster of bands and set 0 values to the masked ones. Each
    band is an event. Select region using window or geometry. Reproject
    input by proving dst_crs and/or (transform, width, height). Returns matrix
    in 2d: band x coordinates in 1d (can be reshaped to band x height x width)

    Parameters:
        file_name (str): name of the file
        band (list(int), optional): band number to read. Default: 1
        window (rasterio.windows.Window, optional): window to read
        geometry (shapely.geometry, optional): consider pixels only in shape
        dst_crs (crs, optional): reproject to given crs
        transform (rasterio.Affine): affine transformation to apply
        wdith (float): number of lons for transform
        height (float): number of lats for transform
        resampling (rasterio.warp.Resampling optional): resampling
            function used for reprojection to dst_crs

    Returns:
        dict (meta), np.array (band x coordinates_in_1d)
    """
    if not band:
        band = [1]
    LOGGER.info('Reading %s', file_name)
    if os.path.splitext(file_name)[1] == '.gz':
        file_name = '/vsigzip/' + file_name

    with rasterio.Env():
        with rasterio.open(file_name, 'r') as src:
            dst_meta = src.meta.copy()

            if dst_crs or transform:
                LOGGER.debug('Reprojecting ...')

                src_crs = src.crs if src_crs is None else src_crs
                if not src_crs:
                    src_crs = rasterio.crs.CRS.from_dict(DEF_CRS)
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
        dst_meta['crs'] = rasterio.crs.CRS.from_dict(DEF_CRS)

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
    if os.path.splitext(path)[1] == '.gz':
        path = '/vsigzip/' + path
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
        extra = (0.5 * ((shape[1] - 1) * res[0] - width),
                 0.5 * ((shape[0] - 1) * res[1] - height))
        bounds = (bounds[0] - extra[0] - 0.5 * res[0], bounds[1] - extra[1] - 0.5 * res[1],
                  bounds[2] + extra[0] + 0.5 * res[0], bounds[3] + extra[1] + 0.5 * res[1])

        if bounds[0] > 180:
            bounds = (bounds[0] - 360, bounds[1], bounds[2] - 360, bounds[3])

        window = src.window(*bounds)
        w_transform = src.window_transform(window)
        transform = rasterio.Affine(np.sign(w_transform[0]) * res[0], 0, w_transform[2],
                                    0, np.sign(w_transform[4]) * res[1], w_transform[5])

        if bounds[2] <= 180:
            data = src.read(bands, out_shape=shape, window=window,
                            resampling=resampling)
        else:
            # split up at antimeridian
            bounds_sub = [(bounds[0], bounds[1], 180, bounds[3]),
                          (-180, bounds[1], bounds[2] - 360, bounds[3])]
            ratio_left = (bounds_sub[0][2] - bounds_sub[0][0]) / (bounds[2] - bounds[0])
            shapes_sub = [(shape[0], int(shape[1] * ratio_left))]
            shapes_sub.append((shape[0], shape[1] - shapes_sub[0][1]))
            windows_sub = [src.window(*bds) for bds in bounds_sub]
            data = [src.read(bands, out_shape=shp, window=win, resampling=resampling)
                    for shp, win in zip(shapes_sub, windows_sub)]
            data = np.concatenate(data, axis=2)
    return data, transform

def read_raster_sample(path, lat, lon, intermediate_res=None, method='linear', fill_value=None):
    """Read point samples from raster file

    Parameters:
        path (str): path of the raster file
        lat (np.array): latitudes in file's CRS
        lon (np.array): longitudes in file's CRS
        intermediate_res (float, optional): If given, the raster is not read in its original
            resolution but in the given one. This can increase performance for
            files of very high resolution.
        method (str, optional): The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.
        fill_value (numeric, optional): The value used outside of the raster
            bounds. Default: The raster's nodata value or 0.

    Returns:
        np.array of same length as lat
    """
    if lat.size == 0:
        return np.zeros_like(lat)

    LOGGER.info('Sampling from %s', path)
    if os.path.splitext(path)[1] == '.gz':
        path = '/vsigzip/' + path

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

    Parameters:
        data (np.array): 2d numpy array containing the values
        interp_y (np.array): y-coordinates of points (corresp. to first axis of data)
        interp_x (np.array): x-coordinates of points (corresp. to second axis of data)
        transform (affine.Affine): affine transform defining the raster
        method (str, optional): The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.
        fill_value (numeric, optional): The value used outside of the raster
            bounds. Default: 0.

    Returns:
        np.array
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

    Parameters:
        data (np.array): 2d numpy array containing the values
        transform (affine.Affine): affine transform defining the raster
        res (float or pair of floats): new resolution
        method (str, optional): The interpolation method, passed to
            scipy.interp.interpn. Default: 'linear'.

    Return:
        np.array, affine.Affine
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
    """Read vector file format supported by fiona. Each field_name name is
    considered an event.

    Parameters:
        file_name (str): vector file with format supported by fiona and
            'geometry' field.
        field_name (list(str)): list of names of the columns with values.
        dst_crs (crs, optional): reproject to given crs

    Returns:
        np.array (lat), np.array (lon), geometry (GeiSeries), np.array (value)
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
    """Write raster in GeoTiff format

    Parameters:
        fle_name (str): file name to write
        data_matrix (np.array): 2d raster data. Either containing one band,
            or every row is a band and the column represents the grid in 1d.
        meta (dict): rasterio meta dictionary containing raster
            properties: width, height, crs and transform must be present
            at least (transform needs to contain upper left corner!)
        dtype (numpy dtype): a numpy dtype
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

def points_to_raster(points_df, val_names=None, res=0.0, raster_res=0.0, scheduler=None):
    """Compute raster matrix and transformation from value column

    Parameters:
        points_df (GeoDataFrame): contains columns latitude, longitude and those listed in
            the parameter `val_names`
        val_names (list of str, optional): The names of columns in `points_df` containing
            values. The raster will contain one band per column. Default: ['value']
        res (float, optional): resolution of current data in units of latitude
            and longitude, approximated if not provided.
        raster_res (float, optional): desired resolution of the raster
        scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”

    Returns:
        np.array, affine.Affine

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
    df_poly = points_df[val_names]
    if not scheduler:
        df_poly['geometry'] = apply_box(points_df)
    else:
        ddata = dd.from_pandas(points_df[['latitude', 'longitude']],
                               npartitions=cpu_count())
        df_poly['geometry'] = ddata.map_partitions(apply_box, meta=Polygon) \
                                   .compute(scheduler=scheduler)
    # construct raster
    xmin, ymin, xmax, ymax = (points_df.longitude.min(), points_df.latitude.min(),
                              points_df.longitude.max(), points_df.latitude.max())
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
        'crs': points_df.crs,
        'height': rows,
        'width': cols,
        'transform': ras_trans,
    }
    return raster_out, meta

def set_df_geometry_points(df_val, scheduler=None):
    """Set given geometry to given dataframe using dask if scheduler

    Parameters:
        df_val (DataFrame or GeoDataFrame): contains latitude and longitude columns
        scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
    """
    LOGGER.info('Setting geometry points.')
    def apply_point(df_exp):
        fun = lambda row: Point(row.longitude, row.latitude)
        return df_exp.apply(fun, axis=1)
    if not scheduler:
        df_val['geometry'] = apply_point(df_val)
    else:
        ddata = dd.from_pandas(df_val, npartitions=cpu_count())
        df_val['geometry'] = ddata.map_partitions(apply_point, meta=Point) \
                                  .compute(scheduler=scheduler)

def fao_code_def():
    """Generates list of FAO country codes and corresponding ISO numeric-3 codes

    Returns:
        iso_list (list): list of ISO numeric-3 codes
        faocode_list (list): list of FAO country codes
    """
    # FAO_FILE2: contains FAO country codes and correstponding ISO3 Code
    #           (http://www.fao.org/faostat/en/#definitions)
    fao_file = pd.read_csv(os.path.join(DATA_DIR, 'system', "FAOSTAT_data_country_codes.csv"))
    fao_code = getattr(fao_file, 'Country Code').values
    fao_iso = (getattr(fao_file, 'ISO3 Code').values).tolist()

    # create a list of ISO3 codes and corresponding fao country codes
    iso_list = list()
    faocode_list = list()
    for idx, iso in enumerate(fao_iso):
        if isinstance(iso, str):
            iso_list.append(country_iso_alpha2numeric(iso))
            faocode_list.append(int(fao_code[idx]))

    return iso_list, faocode_list

def country_faocode2iso(input_fao):
    """Convert FAO country code to ISO numeric-3 codes

    Parameters:
        input_fao (int or array): FAO country codes of countries (or single code)

    Returns:
        output_iso (int or array): ISO numeric-3 codes of countries (or single code)
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
    """Convert ISO numeric-3 codes to FAO country code

    Parameters:
        input_iso (int or array): ISO numeric-3 codes of countries (or single code)

    Returns:
        output_faocode (int or array): FAO country codes of countries (or single code)
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
