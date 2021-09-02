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
import pandas as pd
import logging

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

        