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

LOGGER = logging.getLogger(__name__)


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
        return impact_line.groupby(level=0).eai_exp.sum()/exp_pnts.groupby(level=0).value.sum()
    
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
        return impact_poly.groupby(level=0).eai_exp.sum()/exp_pnts.groupby(level=0).value.sum()
   
    else:
        raise NotImplementedError

def disaggregate_cnstly(gdf_interpol):
    """
    Disaggregate the values of an interpolated exposure gdf 
    constantly among all points belonging to the initial shape.
    
    Note
    ----
    Requires that the initial shapes from which the gdf was interpolated
    had a value column.
    
    Parameters
    ----------
    gdf_interpol : gpd.GeoDataFrame
    """
    
    group = gdf_interpol.groupby(axis=0, level=0)
    val_per_point = group.value.mean()/group.count().iloc[:,0]
    for ix, val in zip(np.unique(gdf_interpol.index.get_level_values(0)),
                       val_per_point):
        gdf_interpol.at[ix, 'value']= val
        
    return gdf_interpol

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