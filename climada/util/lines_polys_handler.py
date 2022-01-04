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
        exp, impf_set, haz, lon_res, lat_res,
        to_meters=False, disagg=None, agg_avg=False
        ):
    """
    Compute impact for exposure with (multi-)polygons. Lat/Lon values are
    ignored.

    The polygons are first disaggrated to a grid with resolution
    lon_res/lat_res. The impact per point is then aggregated for each polygon.

    Parameters
    ----------
    exp : Exposures
        The exposure instance with exp.gdf.geometry containing (multi-)polygons
    impf_set : ImpactFuncSet
        The set of impact functions.
    haz : Hazard
        The hazard instance.
    lon_res : float
        Longitude resolution of the disaggregation grid.
    lat_res : float
        Latitude. resolution of the disaggregation grid.
    to_meters : bool, optional
       If True, the polygons are projected to an equal area projection before
       the disaggregation. lat/lon_res are then in meters. The exposures are
       then reprojected into the original projections before the impact
       calculation. The default is False.
    disagg : string, optional
        Disaggregation method. The default is None (i.e. use default disaggregation)
    agg_avg : bool, optional
        If True, the impact is averaged over all points in each polygon.
        If False, the impact is summed over all points in each polygon.
        The default is False.

    Returns
    -------
    Impact
        Impact object with the impact per polygon (lines of exp.gdf).

    See Also
    --------
    exp_geom_to_pnt: disaggregate exposures

    """

    #discretize exposure
    exp_pnt = exp_geom_to_pnt(
        exp=exp, lon_res=lon_res, lat_res=lat_res,
        to_meters=to_meters, disagg=disagg
        )
    exp_pnt.assign_centroids(haz)

    return calc_impact_pnt_agg(exp_pnt=exp_pnt, impf_set=impf_set, haz=haz, agg_avg=agg_avg)

def calc_impact_pnt_agg(exp_pnt, impf_set, haz, agg_avg=False):
    """
    Compute impact for an exposures with disaggregated geometries

    Parameters
    ----------
    exp_pnt : Exposures
        Exposures with a double index geodataframe, first for polygon geometries,
        second for the point disaggregation of the polygons.
    impf_set : ImpactFuncSet
        The set of impact functions.
    haz : Hazard
        The hazard instance.
    agg_avg : bool, optional
        If True, the impact is averaged over all points in each polygon.
        If False, the impact is summed over all points in each polygon.
        The default is False.

    Returns
    -------
    Impact
        Impact object with the impact per polygon.

    """
    # compute impact
    impact_pnt = Impact()
    impact_pnt.calc(exp_pnt, impf_set, haz, save_mat=True)

    return impact_pnt_agg(impact_pnt, exp_pnt, agg_avg)

def impact_pnt_agg(impact_pnt, exp_pnt, agg_avg):
    """
    Aggregate the impact per geometry

    Parameters
    ----------
    impact_pnt : Impact
        Impact object with impact per exposure point (lines of exp_pnt)
    exp_pnt : Exposures
        Exposures with a double index geodataframe, first for polygon geometries,
        second for the point disaggregation of the polygons.
    agg_avg : bool, optional
        If True, the impact is averaged over all points in each polygon.
        If False, the impact is summed over all points in each polygon.
        The default is False.

    Returns
    -------
    Impact
        Impact object with the impact per polygon.

    """

    # aggregate impact
    mat_agg = aggregate_impact_mat(impact_pnt, exp_pnt.gdf, agg_avg)

    #Write to impact obj
    impact_agg = set_imp_mat(impact_pnt, mat_agg)

    #Add exposure representation points as coordinates
    repr_pnts = gpd.GeoSeries(exp_pnt.gdf['geometry_orig'][:,0].apply(lambda x: x.representative_point()))
    impact_agg.coord_exp = np.array([repr_pnts.y, repr_pnts.x]).transpose()
    #Add geometries
    impact_agg.geom_exp = exp_pnt.gdf.xs(0, level=1).set_geometry('geometry_orig').geometry.rename('geometry')

    return impact_agg

def exp_geom_to_pnt(exp, lon_res, lat_res, to_meters, disagg=None):
    """
    Disaggregate exposures with polygon geometries to points

    Parameters
    ----------
    exp : Exposures
        The exposure instance with exp.gdf.geometry containing (multi-)polygons
    lon_res : float
        Longitude resolution of the disaggregation grid.
    lat_res : float
        Latitude. resolution of the disaggregation grid.
    to_meters : bool, optional
       If True, the polygons are projected to an equal area projection before
       the disaggregation. lat/lon_res are then in meters. The exposures are
       then reprojected into the original projections before the impact
       calculation. The default is False.
    disagg : string, optional
        Disaggregation method. The default is None (i.e. use default disaggregation)

    Returns
    -------
    exp_pnt : Exposures
        Exposures with a double index geodataframe, first for the polygon geometries of exp,
        second for the point disaggregation of the polygons.

    """

    # rasterize
    if to_meters:
        gdf_pnt = poly_to_pnts_m(exp.gdf.reset_index(drop=True), lon_res, lat_res)
    else:
        gdf_pnt = poly_to_pnts(exp.gdf.reset_index(drop=True), lon_res, lat_res)

    # disaggregate
    if disagg == 'avg':
        gdf_pnt = disagg_poly_avg(gdf_pnt)
    elif disagg == 'area':
        gdf_pnt = disagg_gdf_val(gdf_pnt, lon_res * lat_res)
    elif disagg is None and 'value' not in gdf_pnt.columns:
        gdf_pnt['value'] = 1

    # set lat lon and centroids
    exp_pnt = exp.copy()
    exp_pnt.set_gdf(gdf_pnt)
    exp_pnt.set_lat_lon()

    return exp_pnt

def set_imp_mat(impact, imp_mat):
    """
    Set Impact attributes from the impact matrix. Returns a copy.
    Overwrites eai_exp, at_event, aai_agg, imp_mat.

    Parameters
    ----------
    impact : Impact
        Impact instance.
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.

    Returns
    -------
    imp : Impact
        Copy of impact with eai_exp, at_event, aai_agg, imp_mat set.

    """
    imp = copy.deepcopy(impact)
    imp.eai_exp = eai_exp_from_mat(imp_mat, imp.frequency)
    imp.at_event = at_event_from_mat(imp_mat)
    imp.aai_agg = aai_agg_from_at_event(imp.at_event, imp.frequency)
    imp.imp_mat = imp_mat
    return imp

def eai_exp_from_mat(imp_mat, freq):
    """
    Compute impact for each exposures from the total impact matrix

    Parameters
    ----------
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.
    frequency : np.array
        annual frequency of events

    Returns
    -------
    eai_exp : np.array
        expected annual impact for each exposure

    """
    freq_mat = freq.reshape(len(freq), 1)
    return imp_mat.multiply(freq_mat).sum(axis=0).A1

def at_event_from_mat(imp_mat):
    """
    Compute impact for each hazard event from the total impact matrix

    Parameters
    ----------
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.

    Returns
    -------
    at_event : np.array
        impact for each hazard event

    """
    return np.squeeze(np.asarray(np.sum(imp_mat, axis=1)))

def aai_agg_from_at_event(at_event, freq):
    """
    Aggregate impact.at_event

    Parameters
    ----------
    at_event : np.array
        impact for each hazard event
    frequency : np.array
        annual frequency of event

    Returns
    -------
    float
        average annual impact aggregated

    """
    return sum(at_event * freq)

def aggregate_impact_mat(imp_pnt, gdf_pnt, agg_avg):
    """
    Aggregate impact matrix given geodataframe or disaggregated polygons.

    Parameters
    ----------
    impact_pnt : Impact
        Impact object with impact per point (lines of gdf_pnt)
    gdf_pnt : GeoDataFrame
        Exposures geodataframe with a double index, first for polygon geometries,
        second for the point disaggregation of the polygons.
    agg_avg : bool, optional
        If True, the impact is averaged over all points in each polygon.
        If False, the impact is summed over all points in each polygon.
        The default is False.

    Returns
    -------
    sparse.csr_matrix
        matrix num_events x num_polygons with impacts.

    """
    # aggregate impact
    mi = gdf_pnt.index
    col_geom = mi.get_level_values(level=0).to_numpy()
    row_pnt = np.arange(len(col_geom))
    if agg_avg:
        from collections import Counter
        geom_sizes = Counter(col_geom).values()
        mask = np.concatenate([np.ones(l) / l for l in geom_sizes])
    else:
        mask = np.ones(len(col_geom))
    csr_mask = sp.sparse.csr_matrix(
        (mask, (row_pnt, col_geom)),
         shape=(len(row_pnt), len(np.unique(col_geom)))
        )
    return imp_pnt.imp_mat.dot(csr_mask)

def plot_eai_exp_geom(imp_geom, centered=False, figsize=(9, 13), **kwargs):
    """
    Plot the average impact per exposure polygon.

    Parameters
    ----------
    imp_geom : Impact
        Impact instance with imp_geom set (i.e. computed from exposures with polygons)
    centered : bool, optional
        Center the plot. The default is False.
    figsize : (float, float), optional
        Figure size. The default is (9, 13).
    **kwargs : dict
        Keyword arguments for GeoDataFrame.plot()

    Returns
    -------
    ax:
        matplotlib axes instance

    """
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
    Disaggragate (multi-)polygons geodataframe to points

    Parameters
    ----------
    gdf : geodataframe
        Can be any CRS
    lon_res : float
        Resolution in longitudes (same units as gdf crs)
    lat_res : float
        Resolution in latitudes (same units as gdf crs)

    Returns
    -------
    geodataframe
        Geodataframe with a double index, first for polygon geometries,
        second for the point disaggregation of the polygons.

    """

    gdf_points = gdf.copy()
    gdf_points['geometry_pnt'] = gdf.apply(
        lambda row: _interp_one_poly(row.geometry, lon_res, lat_res), axis=1)
    gdf_points.rename(columns = {'geometry': 'geometry_orig'}, inplace=True)
    gdf_points.rename(columns = {'geometry_pnt': 'geometry'}, inplace=True)
    gdf_points.set_geometry('geometry', inplace=True)

    return gdf_points.explode()


def poly_to_pnts_m(gdf, x_res, y_res):
    """
    Disaggragate (multi-)polygons geodataframe to points

    Parameters
    ----------
    gdf : geodataframe
        Can be any CRS
    lon_res : float
        Resolution in longitudes in meters
    lat_res : float
        Resolution in latitudes in meters

    Returns
    -------
    geodataframe
        Geodataframe with a double index, first for polygon geometries,
        second for the point disaggregation of the polygons.

    """

    orig_crs = gdf.crs
    gdf_points = gdf.copy()
    gdf_points['geometry_pnt'] = gdf_points.apply(
        lambda row: _interp_one_poly_m(row.geometry, x_res, y_res, orig_crs), axis=1)
    gdf_points.rename(columns = {'geometry': 'geometry_orig'}, inplace=True)
    gdf_points.rename(columns = {'geometry_pnt': 'geometry'}, inplace=True)
    gdf_points.set_geometry('geometry', inplace=True)

    return gdf_points.explode()



def _interp_one_poly(poly, res_x, res_y):
    """
    Disaggragate a single polygon to points

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    res_x : float
        Resolution in x (same units as gdf crs)
    res_y : float
        Resolution in y (same units as gdf crs)

    Returns
    -------
    shapely multipoint
        Grid of points rasterizing the polygon

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
    Disaggragate a single polygon to points. Resolution in meters.

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    res_x : float
        Resolution in x in meters
    res_y : float
        Resolution in y in meters
    orig_crs: pyproj.CRS
        CRS of the polygon

    Returns
    -------
    shapely multipoint
        Grid of points rasterizing the polygon

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



def disagg_poly_avg(gdf_pnts):
    """
    Disaggragate value of geodataframes from polygons to points

    Parameters
    ----------
    gdf_pnts : geodataframe
        Geodataframe with a double index, first for polygon geometries,
        second for the point disaggregation of the polygons. The value column is assumed
        to represent values per polygon (first index).

    Returns
    -------
    gdf_agg : geodataframe
        The value per polygon is evenly distributed over the points per polygon.

    """

    gdf_agg = gdf_pnts.copy()

    group = gdf_pnts.groupby(axis=0, level=0)
    gdf = group.value.mean() / group.value.count()

    gdf = gdf.reindex(gdf_pnts.index, level=0)
    gdf_agg['value'] = gdf

    return gdf_agg

def disagg_gdf_val(gdf_pnts, value_per_pnt):
    """
    Assign same value to all geodataframe points

    Parameters
    ----------
    gdf_pnts : geodataframe
        Geodataframe with a double index, first for polygon geometries,
        second for the point disaggregation of the polygons. The value column is assumed
        to represent values per polygon (first index).
    value_per_pnt: float
        Value to assign to each point.

    Returns
    -------
    gdf_agg : geodataframe
        The value per point is value_per_pnt

    """

    gdf_agg = gdf_pnts.copy()
    gdf_agg['value'] = value_per_pnt

    return gdf_agg


def poly_to_equalarea_proj(poly, orig_crs):
    """
    Project polyong to Equal Area Cylindrical projection
    using a representative point as lat/lon reference.

    https://proj.org/operations/projections/cea.html

    Parameters
    ----------
    poly : shapely Polygon
        Polygon
    orig_crs: pyproj.CRS
        CRS of the polygon

    Returns
    -------
    poly : shapely Polygon
        Polygon in equal are projection

    """

    repr_pnt = poly.representative_point()
    lon_0, lat_0 = repr_pnt.x, repr_pnt.y

    project = pyproj.Transformer.from_proj(
        pyproj.Proj(orig_crs),
        pyproj.Proj("+proj=cea +lat_0=%f +lon_0=%f +units=m" %(lat_0, lon_0)),
        always_xy=True
    )
    return sh.ops.transform(project.transform, poly)


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


def invert_shapes(gdf_cutout, shape_outer):
    """
    given an outer shape and a gdf with geometries that are to be cut out,
    return the remaining multipolygon
    """
    return shape_outer - _make_union(gdf_cutout)

def _make_union(gdf):
    """
    Solve issue of invalid geometries in MultiPolygons, which prevents that
    shapes can be combined into one unary union, save the respective Union
    """

    union1 = gdf[gdf.geometry.type == 'Polygon'].unary_union
    union2 = gdf[gdf.geometry.type != 'Polygon'].geometry.buffer(0).unary_union
    union_all = unary_union([union1, union2])

    return union_all
