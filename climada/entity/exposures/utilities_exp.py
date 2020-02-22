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

Defines ManufacturingExp class, a subclass of Exposures, approximating global
exposure of the manufacturing industry sector (approximated using global 
emissions of NOx).

---

Source of underlying data:

Greenhouse gas & Air pollution Interactions and Synergies (GAINS) model,
International Institute for Applied Systems Analysis (IIASA),
2015, "ECLIPSE V5a global emission fields" [Data file],
http://www.iiasa.ac.at/web/home/research/researchPrograms/air/ECLIPSEv5a.html
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import cartopy.crs as ccrs
from tqdm import tqdm
from iso3166 import countries_by_alpha3 as ctry_iso3
from iso3166 import countries_by_numeric as ctry_ids
import netCDF4 as nc


import os
from geopandas import GeoDataFrame


from climada.entity.exposures.base import Exposures
from climada.entity.tag import Tag
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
import climada.util.hdf5_handler as hdf5
from climada.util.constants import ONE_LAT_KM, DEF_CRS
import climada.util.coordinates as co
from climada.util.interpolation import interpol_index
import climada.util.plot as u_plot
from climada.util.constants import DATA_DIR, SYSTEM_DIR, GLB_CENTROIDS_MAT 

LOGGER = logging.getLogger(__name__)

# File with raw data used for exposures. Source see file header.
# First for USA, second for rest of world.
ECLIPSE_FILE = os.path.join(SYSTEM_DIR, 'ECLIPSE_base_CLE_V5a_NOx.nc')


class UtilitiesExp(Exposures):
    """Approximates global manufacturing exposures. 
    Source of underlying data: 
    Greenhouse gas & Air pollution Interactions and Synergies (GAINS) model,
    International Institute for Applied Systems Analysis (IIASA),
    2015, "ECLIPSE V5a global emission fields" [Data file],
    http://www.iiasa.ac.at/web/home/research/researchPrograms/air/ECLIPSEv5a.html
    """
    def _constructor(self):
        return UtilitiesExp
    def init_utilities_exp(self, assign_centroids=False):
        """Initialize. ADD FILE DOWNLOAD CAPABILITIES LATER
        Parameters:
            assign_centroids (boolean): if False (default), no centroids are
            assigned to the data values. If True, assign centroids from the default 
            global centroids file (see GLB_CENTROIDS_MAT constant). Note: for
            MRIO analysis (supplychain module), this is required for obtaining
            region ids of exposures (not obtainable from source data file)."""
        
        data = nc.Dataset(ECLIPSE_FILE) # Using netCDF4 package.
        time = np.array(data.variables['time'][:])
        current_i = np.nonzero(time == 2015)[0][0] # Index of current year (actually 2015).
        lons = np.array(data.variables['lon'][:])
        lats = np.array(data.variables['lat'][:])
        raw_data = np.array(data.variables['emis_ind'][:])
        raw_data = (raw_data[current_i,:,:]).copy(); raw_data = raw_data.T
        # Flatten data matrix and adjust lons and lats accordingly. We now call
        # the raw data 'value', as in their resulting pd.Series representation:
        value = raw_data.flatten(order='F').copy(); del(raw_data)
        # We flattened with keeping column-wise order. Hence, for lat vector, 
        # we have to repeat each element k times, with k = length of original 
        # lon vector (note that in raw_data, rows represent lon and columns lat!).
        # For lon vector, we have to stack original lon vector n times, 
        # with n = length of original lat vector...
        lats_orig = lats.copy()
        lats = np.repeat(lats, len(lons))
        lons = np.tile(lons, len(lats_orig))
        # We only carry on with locations where emissions are > 0 :
        self['longitude'] = lons[value > 1e-1]
        self['latitude'] = lats[value > 1e-1]
#        For now, we use value 1, not actual emission values. This is an
#        interpretation question and assumptions have to be made for both
#        approaches. 
#        Update: USING 1 LEADS TO USELESS REGULAR GRID OF ASSETS ALMOST EVERYWHERE.
#        Problems are many extremely small values. Setting a minimal threshold
#        seems sensible. For now we set all values < 100t/year to zero (note that
#        ECLIPSE data is in kt); this has to be reviewed carefully later...
        self['value'] = value[value > 1e-1]
        if assign_centroids == True:
            # Read centroids to be able to assign region_ids (required for use
            # of exposures instance in mriot analysis:
            centr = Centroids(); centr.read_mat(GLB_CENTROIDS_MAT)
            centr.set_region_id()
            haz = Hazard('TC')
            haz.centroids = centr
            self.crs = centr.crs['init']
            self.assign_centroids(haz)
            region_ids = np.zeros_like(self['latitude'], dtype=int)
            for row_idx, centroid_idx in enumerate(self['centr_TC']):
                if centroid_idx >= 0 and len(centr.region_id) > centroid_idx:
                    region_ids[row_idx] = centr.region_id[centroid_idx]
                else:
                    region_ids[row_idx] = int(0)
            region_ids[region_ids == -99] = int(0)
            self['region_id'] = region_ids
            self.ref_year = 2015
            self.tag = Tag()
            self.tag.description = 'Global exposures of the manufacturing '\
                            'industry sector, approximated by global NOx emissions.'
            self.tag.file_name = ECLIPSE_FILE
            self.value_unit = 'kt NOx'
            self.check()            
        

        
        
        
    

 
            
            
                
                
            
        

