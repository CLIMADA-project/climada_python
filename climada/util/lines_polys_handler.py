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
import logging
import copy

import geopandas as gpd
import numpy as np
import shapely as sh
import scipy as sp
import shapely.geometry as shgeom
import cartopy.crs as ccrs

from shapely.ops import unary_union

from climada.engine import Impact
from climada.util import coordinates as u_coord

import pyproj

LOGGER = logging.getLogger(__name__)


def calc_geom_impact(
        exp, haz, impf_set, lon_res, lat_res,
        to_meters=False, disagg=None, agg_avg=False
        ):

    #discretize exposure
    exp_pnt = exp_geom_to_pnt(
        exp=exp, lon_res=lon_res, lat_res=lat_res,
        to_meters=to_meters, disagg=disagg
        )
    exp_pnt.assign_centroids(haz)

    return calc_impact_geom_pnt(exp_pnt=exp_pnt, impf_set=impf_set, haz=haz, agg_avg=agg_avg)

def calc_impact_geom_pnt(exp_pnt, impf_set, haz, agg_avg=False):
    # compute impact
    impact_pnt = Impact()
    impact_pnt.calc(exp_pnt, impf_set, haz, save_mat=True)

    return impact_pnt_agg(impact_pnt, exp_pnt, agg_avg)

def impact_pnt_agg(impact_pnt, exp_pnt, agg_avg):

    # aggregate impact
    mat_agg = aggregate_impact_mat(impact_pnt, exp_pnt.gdf, agg_avg)

    #Write to impact obj
    impact_agg = set_imp_mat(impact_pnt, mat_agg)

    #Add exposure representation points as coordinates
    repr_pnts = exp_pnt.gdf['geometry_orig'].apply(lambda x: x.representative_point())
    impact_agg.coord_exp = np.array([repr_pnts.y, repr_pnts.x]).transpose()
    #Add geometries
    impact_agg.geom_exp = exp_pnt.gdf.xs(0, level=1).set_geometry('geometry_orig').geometry.rename('geometry')

    return impact_agg

def exp_geom_to_pnt(exp, lon_res, lat_res, to_meters, disagg=None):

    # rasterize
    if to_meters:
        gdf_pnt = poly_to_pnts_m(exp.gdf.reset_index(drop=True), lon_res, lat_res)
    else:
        gdf_pnt = poly_to_pnts(exp.gdf.reset_index(drop=True), lon_res, lat_res)

    # disaggregate
    if disagg == 'avg':
        gdf_pnt = disagg_poly_avg(gdf_pnt)
    elif disagg == 'area':
        gdf_pnt = disagg_poly_val(gdf_pnt, lon_res * lat_res)
    elif disagg is None and 'value' not in gdf_pnt.columns:
        gdf_pnt['value'] = 1

    # set lat lon and centroids
    exp_pnt = exp.copy()
    exp_pnt.set_gdf(gdf_pnt)
    exp_pnt.set_lat_lon()

    return exp_pnt

def exp_line_to_pnt(exp, dist, disagg=None):

    gdf_pnt = line_to_pnts_m(exp.gdf.reset_index(drop=True), dist)

    # disaggregate
    if disagg == 'avg':
        gdf_pnt = disagg_line_avg(gdf_pnt)
    elif disagg == 'len':
        gdf_pnt = disagg_line_val(gdf_pnt, dist)
    elif disagg is None and 'value' not in gdf_pnt.columns:
        gdf_pnt['value'] = 1

    # set lat lon and centroids
    exp_pnt = exp.copy()
    exp_pnt.set_gdf(gdf_pnt)
    exp_pnt.set_lat_lon()

    return exp_pnt

def disagg_line_avg(gdf_line_pnts):

    gdf_agg = gdf_line_pnts.copy()

    group = gdf_line_pnts.groupby(axis=0, level=0)
    gdf = group.value.mean() / group.value.count()

    gdf = gdf.reindex(gdf_line_pnts.index, level=0)
    gdf_agg['value'] = gdf

    return gdf_agg

def disagg_line_val(gdf_line_pnts, value_per_pnt):

    gdf_agg = gdf_line_pnts.copy()
    gdf_agg['value'] = value_per_pnt

    return gdf_agg

def set_imp_mat(impact, mat):
    imp = copy.deepcopy(impact)
    imp.eai_exp = eai_exp_from_mat(mat, imp.frequency)
    imp.at_event = at_event_from_mat(mat)
    imp.aai_agg = aai_agg_from_at_event(imp.at_event, imp.frequency)
    imp.imp_mat = mat
    return imp


def eai_exp_from_mat(mat, freq):
    return np.einsum('ji,j->i', mat.todense(), freq)

def at_event_from_mat(mat):
    return np.squeeze(np.asarray(np.sum(mat, axis=1)))

def aai_agg_from_at_event(at_event, freq):
    return sum(at_event * freq)

def aggregate_impact_mat(imp_pnt, gdf_pnt, agg_avg):
    # aggregate impact
    mi = gdf_pnt.index
    row = mi.get_level_values(level=0).to_numpy()
    mask = np.zeros((len(row), len(np.unique(mi.droplevel(1)))))
    for i, m in enumerate(row):
        mask[i][m] = 1
    if agg_avg:
        mask /= mask.sum(axis=0)
    csr_mask = sp.sparse.csr_matrix(mask)
    return imp_pnt.imp_mat.dot(csr_mask)

def plot_eai_exp_geom(imp_geom, centered=False, figsize=(9, 13), **kwargs):
    kwargs['figsize'] = figsize
    if 'legend_kwds' not in kwargs:
        kwargs['legend_kwds'] = {
            'label': "Impact [%s]" %imp_geom.unit,
            'orientation': "horizontal"
            }
    if 'legend' not in kwargs:
        kwargs['legend'] = True
    gdf_plot = gpd.GeoDataFrame(imp_geom.geom_exp)
    gdf_plot['impact'] = imp_geom.eai_exp
    if centered:
        xmin, xmax = u_coord.lon_bounds(imp_geom.coord_exp[:,1])
        proj_plot = ccrs.PlateCarree(central_longitude=0.5 * (xmin + xmax))
        gdf_plot = gdf_plot.to_crs(proj_plot)
    return gdf_plot.plot(column = 'impact', **kwargs)


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
    gdf_points['geometry_pnt'] = gdf.apply(
        lambda row: _interp_one_poly(row.geometry, lon_res, lat_res), axis=1)
    gdf_points.rename(columns = {'geometry': 'geometry_orig'}, inplace=True)
    gdf_points.rename(columns = {'geometry_pnt': 'geometry'}, inplace=True)
    gdf_points.set_geometry('geometry', inplace=True)

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

    orig_crs = gdf.crs
    gdf_points = gdf.copy()
    gdf_points['geometry_pnt'] = gdf_points.apply(
        lambda row: _interp_one_poly_m(row.geometry, x_res, y_res, orig_crs), axis=1)
    gdf_points.rename(columns = {'geometry': 'geometry_orig'}, inplace=True)
    gdf_points.rename(columns = {'geometry_pnt': 'geometry'}, inplace=True)
    gdf_points.set_geometry('geometry', inplace=True)

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

def poly_to_equalarea_proj(poly, orig_crs):

    repr_pnt = poly.representative_point()
    lon_0, lat_0 = repr_pnt.x, repr_pnt.y

    project = pyproj.Transformer.from_proj(
        pyproj.Proj(orig_crs),
        pyproj.Proj("+proj=cea +lat_0=%f +lon_0=%f +units=m" %(lat_0, lon_0)),
        always_xy=True
    )
    return sh.ops.transform(project.transform, poly)

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
