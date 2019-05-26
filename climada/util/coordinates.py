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
import os
import logging
import numpy as np
import fiona
from cartopy.io import shapereader
import shapely.vectorized
import shapely.ops
from shapely.geometry import Polygon
from sklearn.neighbors import BallTree
from fiona.crs import from_epsg
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, calculate_default_transform

from climada.util.constants import EARTH_RADIUS_KM
from climada.util.constants import DEF_CRS

LOGGER = logging.getLogger(__name__)

NE_EPSG = 4326
""" Natural Earth CRS EPSG """

NE_CRS = from_epsg(NE_EPSG)
""" Natural Earth CRS """

def grid_is_regular(coord):
    """Return True if grid is regular.

    Parameters:
        coord (np.array):
    """
    regular = False
    _, count_lat = np.unique(coord[:, 0], return_counts=True)
    _, count_lon = np.unique(coord[:, 1], return_counts=True)
    uni_lat_size = np.unique(count_lat).size
    uni_lon_size = np.unique(count_lon).size
    if uni_lat_size == uni_lon_size and uni_lat_size == 1 \
    and count_lat[0] > 1 and count_lon[0] > 1:
        regular = True
    return regular

def get_coastlines(extent=None, resolution=110):
    """Get latitudes and longitudes of the coast lines inside extent. All
    earth if no extent.

    Parameters:
        extent (tuple, optional): (min_lon, max_lon, min_lat, max_lat)
        resolution (float, optional): 10, 50 or 110. Resolution in m. Default:
            110m, i.e. 1:110.000.000

    Returns:
        np.array (lat, lon coastlines)
    """
    resolution = nat_earth_resolution(resolution)
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category='physical',
                                         name='coastline')
    with fiona.open(shp_file) as shp:
        coast_lon, coast_lat = [], []
        for line in shp:
            tup_lon, tup_lat = zip(*line['geometry']['coordinates'])
            coast_lon += list(tup_lon)
            coast_lat += list(tup_lat)
        coast = np.array([coast_lat, coast_lon]).transpose()

        if extent is None:
            return coast

        in_lon = np.logical_and(coast[:, 1] >= extent[0], coast[:, 1] <= extent[1])
        in_lat = np.logical_and(coast[:, 0] >= extent[2], coast[:, 0] <= extent[3])
        return coast[np.logical_and(in_lon, in_lat)].reshape(-1, 2)

def dist_to_coast(coord_lat, lon=None):
    """ Comput distance to coast from input points in meters.

    Parameters:
        coord_lat (np.array or tuple or float):
            - np.array with two columns, first for latitude of each point and
                second with longitude.
            - np.array with one dimension containing latitudes
            - tuple with first value latitude, second longitude
            - float with a latitude value
        lon (np.array or float, optional):
            - np.array with one dimension containing longitudes
            - float with a longitude value

    Returns:
        np.array
    """
    if lon is None:
        if isinstance(coord_lat, tuple):
            coord = np.array([[coord_lat[0], coord_lat[1]]])
        elif isinstance(coord_lat, np.ndarray):
            if coord_lat.shape[1] != 2:
                LOGGER.error('Missing longitude values.')
                raise ValueError
            coord = coord_lat
        else:
            LOGGER.error('Missing longitude values.')
            raise ValueError
    elif isinstance(lon, np.ndarray):
        if coord_lat.size != lon.size:
            LOGGER.error('Wrong input coordinates size: %s != %s',
                         coord_lat.size, lon.size)
            raise ValueError
        coord = np.empty((lon.size, 2))
        coord[:, 0] = coord_lat
        coord[:, 1] = lon
    elif isinstance(lon, float):
        if not isinstance(coord_lat, float):
            LOGGER.error('Wrong input coordinates values.')
            raise ValueError
        coord = np.array([[coord_lat, lon]])

    marg = 10
    lat = coord[:, 0]
    lon = coord[:, 1]
    coast = get_coastlines((np.min(lon) - marg, np.max(lon) + marg,
                            np.min(lat) - marg, np.max(lat) + marg), 10)

    tree = BallTree(np.radians(coast), metric='haversine')
    dist_coast, _ = tree.query(np.radians(coord), k=1, return_distance=True,
                               dualtree=True, breadth_first=False)
    return dist_coast.reshape(-1,) * EARTH_RADIUS_KM * 1000

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
        geom = [cntry_geom for cntry_geom in reader.geometries()]
        geom = shapely.ops.cascaded_union(geom)

    elif country_names:
        countries = list(reader.records())
        geom = [country.geometry for country in countries
                if (country.attributes['ISO_A3'] in country_names) or
                (country.attributes['WB_A3'] in country_names)]
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
    return geom

def coord_on_land(lat, lon, land_geom=None):
    """Check if point is on land (True) or water (False) of provided coordinates.
    All globe considered if no input countries.

    Parameters:
        lat (np.array): latitude of points
        lon (np.array): longitude of points
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
        land_geom = get_land_geometry(extent=(np.min(lon)-delta_deg, \
            np.max(lon)+delta_deg, np.min(lat)-delta_deg, \
            np.max(lat)+delta_deg), resolution=10)
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

    if country_names:
        if isinstance(country_names, str):
            country_names = [country_names]
        out = nat_earth[nat_earth.ISO_A3.isin(country_names)]

    elif extent:
        bbox = Polygon([
            (extent[0], extent[2]),
            (extent[0], extent[3]),
            (extent[1], extent[3]),
            (extent[1], extent[2])
        ])
        bbox = gpd.GeoSeries(bbox, crs=nat_earth.crs)
        bbox = gpd.GeoDataFrame({'geometry': bbox}, crs=nat_earth.crs)
        out = gpd.overlay(nat_earth, bbox, how="intersection")

    else:
        out = nat_earth

    return out

def get_resolution(lat, lon):
    """ Compute resolution of points in lat and lon

    Parameters:
        lat (np.array): latitude of points
        lon (np.array): longitude of points

    Returns:
        float
    """
    # ascending lat and lon
    res_lat, res_lon = np.diff(np.sort(lat)), np.diff(np.sort(lon))
    try:
        res_lat = res_lat[res_lat > 0].min()
    except ValueError:
        res_lat = 0
    try:
        res_lon = res_lon[res_lon > 0].min()
    except ValueError:
        res_lon = 0
    return res_lat, res_lon

def points_to_raster(points_bounds, res):
    """" Transform vector data coordinates to raster. Returns number of rows,
    columns and affine transformation

    Parameters:
        points_bounds (tuple): points total bounds (xmin, ymin, xmax, ymax)
        res (float): resolution of output raster

    Returns:
        int, int, affine.Affine
    """
    xmin, ymin, xmax, ymax = points_bounds
    rows = int(np.floor((ymax-ymin) /  res) + 1)
    cols = int(np.floor((xmax-xmin) / res) + 1)
    ras_trans = from_origin(xmin - res / 2, ymax + res / 2, res, res)
    return rows, cols, ras_trans

def equal_crs(crs_one, crs_two):
    """ Compare two crs

    Parameters:
        crs_one (dict or string or wkt): user crs
        crs_two (dict or string or wkt): user crs

    Returns:
        bool
    """
    return CRS.from_user_input(crs_one) == CRS.from_user_input(crs_two)

def read_raster(file_name, band=[1], src_crs=None, window=False, geometry=False,
                dst_crs=False, transform=None, width=None, height=None,
                resampling=Resampling.nearest):
    """ Read raster of bands and set 0 values to the masked ones. Each
    band is an event. Select region using window or geometry. Reproject
    input by proving dst_crs and/or (transform, width, height).

    Parameters:
        file_name (str): name of the file
        band (list(int), optional): band number to read. Default: 1
        window (rasterio.windows.Window, optional): window to read
        geometry (shapely.geometry, optional): consider pixels only in shape
        dst_crs (crs, optional): reproject to given crs
        transform (rasterio.Affine): affine transformation to apply
        wdith (float): number of lons for transform
        height (float): number of lats for transform
        resampling (rasterio.warp,.Resampling optional): resampling
            function used for reprojection to dst_crs

    Returns:
        dict (meta), np.array (intensity)
    """
    LOGGER.info('Reading %s', file_name)
    if os.path.splitext(file_name)[1] == '.gz':
        file_name = '/vsigzip/' + file_name
    with rasterio.Env():
        with rasterio.open(file_name, 'r') as src:
            if src_crs is None:
                src_meta = CRS.from_dict(DEF_CRS) if not src.crs else src.crs
            else:
                src_meta = src_crs
            if dst_crs or transform:
                LOGGER.debug('Reprojecting ...')
                if not dst_crs:
                    dst_crs = src_meta
                if not transform:
                    transform, width, height = calculate_default_transform(\
                        src_meta, dst_crs, src.width, src.height, *src.bounds)
                dst_meta = src.meta.copy()
                dst_meta.update({'crs': dst_crs,
                                 'transform': transform,
                                 'width': width,
                                 'height': height
                                })
                intensity = np.zeros((len(band), height, width))
                for idx_band, i_band in enumerate(band):
                    reproject(source=src.read(i_band),
                              destination=intensity[idx_band, :],
                              src_transform=src.transform,
                              src_crs=src_meta,
                              dst_transform=transform,
                              dst_crs=dst_crs,
                              resampling=resampling)
                    if np.isnan(dst_meta['nodata']):
                        intensity[idx_band, :][np.isnan(intensity[idx_band, :])] = 0
                    else:
                        intensity[idx_band, :][intensity[idx_band, :] == dst_meta['nodata']] = 0
                meta = dst_meta
                return meta, intensity.reshape((len(band), meta['height']*meta['width']))
            else:
                meta = src.meta.copy()
                if geometry:
                    inten, mask_trans = mask(src, geometry, crop=True, indexes=band)
                    if np.isnan(meta['nodata']):
                        inten[np.isnan(inten)] = 0
                    else:
                        inten[inten == meta['nodata']] = 0
                    meta.update({"height": inten.shape[1],
                                 "width": inten.shape[2],
                                 "transform": mask_trans})
                else:
                    masked_array = src.read(band, window=window, masked=True)
                    inten = masked_array.data
                    inten[masked_array.mask] = 0
                    if window:
                        meta.update({"height": window.height, \
                            "width": window.width, \
                            "transform": rasterio.windows.transform(window, src.transform)})
                if not meta['crs']:
                    meta['crs'] = CRS.from_dict(DEF_CRS)
                band_idx = np.array(band) - 1
                intensity = inten[band_idx, :]
                return meta, intensity.reshape((len(band), meta['height']*meta['width']))

def read_vector(file_name, inten_name=['intensity'], dst_crs=None):
    """ Read vector file format supported by fiona. Each intensity name is
    considered an event.

    Parameters:
        file_name (str): vector file with format supported by fiona and
            'geometry' field.
        inten_name (list(str)): list of names of the columns of the
            intensity of each event.
        dst_crs (crs, optional): reproject to given crs

    Returns:
        np.array (lat), np.array (lon), geometry (GeiSeries), np.array (intensity)
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
    intensity = np.zeros([len(inten_name), lat.size])
    for i_inten, inten in enumerate(inten_name):
        intensity[i_inten, :] = data_frame[inten].values
    return lat, lon, geometry, intensity
