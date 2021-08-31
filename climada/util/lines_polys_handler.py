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
import geopandas as gpd
import logging

LOGGER = logging.getLogger(__name__)


def agg_point_impact_to_lines(gdf_lines, exp_points, imp_points, 
                              agg_mode='length'):
    """given an original line geometry, a converted point exposure and a 
    resultingly calculated point impact, aggregate impacts back to shapes in
    original lines geodataframe.
    
    Parameters
    ----------
    gdf_lines, 
    exp_points, 
    imp_points, 
    agg_mode : str
        'length' or 'value': whether the impact should be
    
    Returns
    -------
   agg_impact : gpd.GeoDataFrame of same height as gdf_lines with 'imp_frac' and 
        'imp_abs', referring to affected fraction of the line's total length / value
        and the absolute length / value impacted.
    """
    agg_impact = gpd.GeoDataFrame()
    
    if agg_mode == 'length':
        exp_points.gdf['affected'] = imp_points.eai_exp>0     
        agg_impact['imp_frac_l'] = (exp_points.gdf[exp_points.gdf.affected==True
                                                ].groupby(level=0)[agg_mode].sum() / 
                                 exp_points.gdf.groupby(level=0)[agg_mode].sum()
                                 ).fillna(0)
        agg_impact['imp_abs_l'] = exp_points.gdf[exp_points.gdf.affected==True
                                              ].groupby(level=0)[agg_mode].sum()
        agg_impact['imp_abs_l'] = agg_impact.imp_abs_l.fillna(0)
    elif agg_mode == 'value':
        exp_points.gdf['impact'] = imp_points.eai_exp
        agg_impact['imp_frac_v'] = (exp_points.gdf.groupby(level=0).impact.sum() / 
                                 exp_points.gdf.groupby(level=0)[agg_mode].sum())
        agg_impact['imp_abs_v'] = exp_points.gdf.groupby(level=0).impact.sum()

    return agg_impact

def agg_point_impact_to_polygons(gdf_polygons, exp_points, imp_points, 
                              agg_mode='length'):
    """given an original polygon geometry, a converted point exposure and a 
    resultingly calculated point impact, aggregate impacts back to shapes in
    original polygons geodataframe.
    
    Parameters
    ----------
    gdf_polygons, 
    exp_points, 
    imp_points, 
    agg_mode : str
        'length' or 'value': whether the impact should be
    
    Returns
    -------
    tuple of pandas.Series of same height as gdf_polygons with columns 'imp_frac' and 
        'imp_abs', referring to affected fraction of the line's total length / value
        and the absolute length / value impacted.
    """
    #TODO: implement
    if agg_mode == 'length':
        pass

    elif agg_mode == 'value':
        pass

    #return (imp_frac, imp_abs)
        