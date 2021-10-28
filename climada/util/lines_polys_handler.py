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

"""
import pandas as pd
import numpy as np
import logging
from shapely.ops import unary_union
from climada.utils import coordinates as u_coord
import shapely as sh
import shapely.geometry as shgeom
from climada.util.constants import DEF_CRS, ONE_LAT_KM

LOGGER = logging.getLogger(__name__)

def poly_to_pnts(gdf, lon_res, lat_res):
    """


    Parameters
    ----------
    gdf : geodataframe
        Can be any CRS
    x_res : TYPE
        DESCRIPTION.
    y_res : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """


    gdf_points = gdf.copy()
    gdf_points['geometry'] = gdf.apply(
        lambda row: _interp_one_poly(row.geometry, lon_res, lat_res), axis=1)

    return gdf_points.explode()

def _interp_one_poly(poly, res_x, res_y):
    """


    Parameters
    ----------
    poly : shapely Polygon
        DESCRIPTION.
    res_x : TYPE
        Resolution in degrees (same as poly crs)
    res_y : TYPE
        Resolution in degrees (same as poly crs)

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if poly.is_empty:
        return shgeom.MultiPoint([])

    height, width, trafo = u_coord.pts_to_raster_meta(poly.bounds, (res_x, res_y))
    x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)
    in_geom = sh.vectorized.contains(poly, x_grid, y_grid)

    if sum(in_geom.flatten()) > 1:
        return shgeom.MultiPoint([(x, y) for x, y in
                            zip(x_grid[in_geom], y_grid[in_geom])])
    else:
        return shgeom.MultiPoint([poly.representative_point()])

def poly_to_pnts_m(gdf, x_res, y_res):
    """


    Parameters
    ----------
    gdf : TYPE
        Must in Default CRS
    x_res : TYPE
        Resolution in meters
    y_res : TYPE
        Resolution in meters

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if gdf.crs != DEF_CRS:
        raise Exception('''Expected a geographic CRS.
                        Please re-project to %s first.''' %DEF_CRS)

    gdf_points = gdf.copy()
    gdf_points['geometry'] = gdf.apply(
        lambda row: _interp_one_poly_m(row.geometry, x_res, y_res), axis=1)

    return gdf_points.explode()

def _interp_one_poly_m(poly, lat_res, lon_res):
    """


    Parameters
    ----------
    poly : Shapely polygon
        Must be in default CRS
    res_x : TYPE
        Resolution in meters
    res_y : TYPE
        Resolution in meters

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    if poly.is_empty:
        return shgeom.MultiPoint([])

    res_x, res_y = deg_res_to_m_res(poly.representative_point().y,
                                     lat_res, lon_res)

    return _interp_one_poly(poly, res_x, res_y)

# Conversion done for a sphere with a diameter of 40075000
def deg_res_to_m_res(lat, res_lat, res_lon):
    """
    Get a latitude dependent estimate for converting grid resolutions
    in metres to degrees.

    Parameters
    ----------
    lat : (float)
        latitude (in degrees) of the representative location
    dist : (float)
        distance in metres which should be converted to degrees lat & lon

    Returns
    -------
    res_x, res_y resolutions in degrees
    """

    m_per_onelon = 40075000 * np.cos(lat) / 360
    res_y = res_lon / m_per_onelon
    res_x = (res_lat/1000) / ONE_LAT_KM

    return res_x, res_y


# Should it only be square_meters? Maybe make interpolate polygon_meters
# and interpolate_polygons_degrees
# Remove hard-coded limit to EPSG:4326
def interpolate_polygons(gdf_poly, area_per_point):
    """For a GeoDataFrame with polygons, get equally distributed lat/lon pairs
    throughout the geometries, at a user-specified resolution (in terms of
    m2 area per point)

    Parameters
    ----------
    gdf_poly : gpd.GeoDataFrame
        with polygons to be interpolated
    area_per_point : float
        area in m2 which one point should represent

    Returns
    -------
    gdf_points : gpd.GeoDataFrame
        with multiindex: first level represents initial polygons, second level
        individual Point per row, retaining all other column infos
        belonging to its corresponding polygon

    See also
    --------
    * util.lines_polys_handler.point_exposure_from_polygons()
    """

    m_per_point = math.sqrt(area_per_point)

    if gdf_poly.crs != "EPSG:4326":
        raise Exception('''Expected a geographic CRS.
                        Please re-project to EPSG:4326 first.''')

    if gdf_poly.geometry.is_empty.any():
        LOGGER.info("Empty geometries encountered. Skipping those.")

    gdf_points = gdf_poly[~gdf_poly.geometry.is_empty].copy()

    gdf_points['geometry'] = gdf_poly.apply(
        lambda row: _interpolate_one_polygon(row.geometry, m_per_point), axis=1)

    return gdf_points.explode()

def agg_to_lines(exp_pnts, impact_pnts, agg_mode='sum'):

    # TODO: make a method of Impact(), go via saving entire impact matrix
    # TODO: think about how to include multi-index instead of requiring entire exp?

    """given an original line geometry, a converted point exposure and a
    resultingly calculated point impact, aggregate impacts back to shapes in
    original lines geodataframe outline.

    Parameters
    ----------

    Returns
    -------

    """
    impact_line = pd.DataFrame(index=exp_pnts.gdf.index,
                               data=impact_pnts.eai_exp, columns=['eai_exp'])

    if agg_mode == 'sum':
        return impact_line.groupby(level=0).eai_exp.sum()

    elif agg_mode == 'fraction':
        return impact_line.groupby(level=0).eai_exp.sum() / exp_pnts.gdf.groupby(level=0).value.sum()

    else:
        raise NotImplementedError

def agg_to_polygons(exp_pnts, impact_pnts, agg_mode='sum'):

    # TODO: make a method of Impact(), go via saving entire impact matrix
    # TODO: think about how to include multi-index instead of requiring entire exp?

    """given an original polygon geometry, a converted point exposure and a
    resultingly calculated point impact, aggregate impacts back to shapes in
    original polygons geodataframe outline.

    Parameters
    ----------

    Returns
    -------

    """
    impact_poly = pd.DataFrame(index=exp_pnts.gdf.index,
                               data=impact_pnts.eai_exp, columns=['eai_exp'])
    if agg_mode == 'sum':
        return impact_poly.groupby(level=0).eai_exp.sum()

    elif agg_mode == 'fraction':
        return impact_poly.groupby(level=0).eai_exp.sum() / exp_pnts.groupby(level=0).value.sum()

    else:
        raise ValueError(f"The aggregation mode {agg_mode} does not exist. Possible" +
                         " choices are 'sum' 'fraction'")

# Disaggretate constant, relative (divided evenly),
# Should there be two types of methods? One that disaggregates values,
# and one that computes the value.
def disaggregate_cnstly(gdf_interpol, val_per_point=None):
    """
    Disaggregate the values of an interpolated exposure gdf
    constantly among all points belonging to the initial shape.


    Parameters
    ----------
    gdf_interpol : gpd.GeoDataFrame
    val_per_point : float, optional
        value per interpolated point, in case no total value column given in
        gdf_interpol
    """
    primary_indices = np.unique(gdf_interpol.index.get_level_values(0))

    if val_per_point:
        val_per_point = val_per_point*np.ones(len(primary_indices))
    else:
        group = gdf_interpol.groupby(axis=0, level=0)
        val_per_point = group.value.mean() / group.count().iloc[:,0]

    for ix, val in zip(np.unique(gdf_interpol.index.get_level_values(0)),
                       val_per_point):
        gdf_interpol.at[ix, 'value']= val

    return gdf_interpol


# Does actually a change in resolution of litpop.
def disaggregate_litpop(gdf_interpol, gdf_shapes, countries):

    """
    disaggregate the values of an interpolated exposure gdf
    according to the litpop values contained within the initial shapes

    This loads the litpop exposure(s) of the countries into memory and cuts out
    those all the values within the original gdf_shapes.

    In a second step, the values of the original shapes are then disaggregated
    constantly onto the interpolated points within each shape.


    """
    # TODO. hotfix for current circular import exp.base <-> exp.litpop if put on top of module
    from climada.entity.exposures import litpop as lp
    exp = lp.LitPop()
    # TODO: Don't hard-code kwargs for exposure!
    exp.set_countries(countries,res_arcsec=30, reference_year=2015)

    # extract LitPop asset values for each shape
    shape_values = []
    for shape in gdf_shapes.geometry:
        shape_values.append(
            exp.gdf.loc[exp.gdf.geometry.within(shape)].value.values.sum())

    # evenly spread value per shape onto interpolated points
    group = gdf_interpol.groupby(axis=0, level=0)
    val_per_point = shape_values/group.count().iloc[:,0]
    for ix, val in zip(np.unique(gdf_interpol.index.get_level_values(0)),
                        val_per_point):
        gdf_interpol.at[ix, 'value']= val

    return gdf_interpol

def _make_union(gdf):
    """
    Solve issue of invalid geometries in MultiPolygons, which prevents that
    shapes can be combined into one unary union, save the respective Union
    """

    union1 = gdf[gdf.geometry.type == 'Polygon'].unary_union
    union2 = gdf[gdf.geometry.type != 'Polygon'].geometry.buffer(0).unary_union
    union_all = unary_union([union1, union2])

    return union_all

def invert_shapes(gdf_cutout, shape_outer):
    """
    given an outer shape and a gdf with geometries that are to be cut out,
    return the remaining multipolygon
    """
    return shape_outer - _make_union(gdf_cutout)