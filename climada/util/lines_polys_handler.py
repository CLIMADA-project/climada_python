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
from climada.util import coordinates as u_coord
import shapely as sh
import shapely.geometry as shgeom
from climada.util.constants import DEF_CRS, ONE_LAT_KM
import pyproj

LOGGER = logging.getLogger(__name__)

def poly_to_pnts(gdf_poly, lon_res, lat_res):
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


    gdf_points = gdf_poly.copy()
    gdf_points['geometry'] = gdf_points.apply(
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

def _interp_one_poly_m(poly, res_x, res_y, orig_crs):
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

    repr_pnt = poly.representative_point()
    lon_0, lat_0 = repr_pnt.x, repr_pnt.y

    project = pyproj.Transformer.from_proj(
        pyproj.Proj(orig_crs),
        pyproj.Proj("+proj=cea +lat_0=%f +lon_0=%f +units=m" %(lat_0, lon_0)),
        always_xy=True
    )
    poly_m = sh.ops.transform(project.transform, poly)

    height, width, trafo = u_coord.pts_to_raster_meta(poly_m.bounds, (res_x, res_y))
    x_grid, y_grid = u_coord.raster_to_meshgrid(trafo, width, height)

    in_geom = sh.vectorized.contains(poly_m, x_grid, y_grid)

    if sum(in_geom.flatten()) > 1:
        project_inv = pyproj.Transformer.from_proj(
            pyproj.Proj("+proj=cea +lat_0=%f +lon_0=%f +units=m" %(lat_0, lon_0)),
            pyproj.Proj(orig_crs),
            always_xy=True
        )
        x_poly, y_poly = project_inv.transform(x_grid[in_geom], y_grid[in_geom])
        poly_pnt = shgeom.MultiPoint([(x, y) for x, y in
                            zip(x_poly, y_poly)])
    else:
        poly_pnt = shgeom.MultiPoint([repr_pnt])

    return poly_pnt


def poly_to_pnts_m(gdf_poly, x_res, y_res):
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

    orig_crs = gdf_poly.crs
    gdf_points = gdf_poly.copy()

    gdf_points['geometry'] = gdf_points.apply(
        lambda row: _interp_one_poly_m(row.geometry, x_res, y_res, orig_crs), axis=1)


    return gdf_points.explode()


def line_to_pnts_m(gdf_lines, dist):

    """ Convert a GeoDataframe with LineString geometries to
    Point geometries, where Points are placed at a specified distance along the
    original LineString
    Important remark: LineString.interpolate() used here performs interpolation
    on a geodesic.

    Parameters
    ----------
    gdf_lines : gpd.GeoDataframe
        Geodataframe with line geometries
    point_dist : float
        Distance in metres apart from which the generated Points should be placed.

    Returns
    -------
    gdf_points : gpd.GeoDataFrame
        with individual Point per row, retaining all other column infos
        belonging to its corresponding line (incl. line length of original geom.
        and multi-index referring to original indexing)

    See also
    --------
    * util.coordinates.compute_geodesic_lengths()
    """

    gdf_points = gdf_lines.copy()
    line_lengths = u_coord.compute_geodesic_lengths(gdf_points)

    # split line lengths into relative fractions acc to point_dist (e.g. 0, 0.5, 1)
    dist_vectors = [
        np.linspace(0, 1, num=int(np.ceil(line_length/dist)+1))
        for line_length in line_lengths
        ]

    gdf_points['geometry'] = [shgeom.MultiPoint(
        [line.interpolate(dist, normalized=True) for dist in dist_vector])
        for line, dist_vector in zip(gdf_lines.geometry, dist_vectors)]

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
        return None

def disagg_gdf_avg(gdf_pnts):

    gdf_agg = gdf_pnts.copy()

    group = gdf_pnts.groupby(axis=0, level=0)
    gdf = group.value.mean() / group.value.count()

    gdf = gdf.reindex(gdf_pnts.index, level=0)
    gdf_agg['value'] = gdf

    return gdf_agg

def disagg_gdf_val(gdf_pnts, value_per_pnt):

    gdf_agg = gdf_pnts.copy()
    gdf_agg['value'] = value_per_pnt

    return gdf_agg


def disagg_poly_avg(gdf_poly_pnts):

    return disagg_gdf_avg(gdf_poly_pnts)

def disagg_poly_val(gdf_poly_pnts, value_per_pnt):

    gdf_agg = gdf_poly_pnts.copy()
    gdf_agg['value'] = value_per_pnt

    return gdf_agg

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