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
import netCDF4 as nc
import os
import tarfile
from climada.entity.exposures.base import Exposures
from climada.util.files_handler import download_file
from climada.entity.tag import Tag
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.util.constants import SYSTEM_DIR, GLB_CENTROIDS_MAT 

LOGGER = logging.getLogger(__name__)

ECLIPSE_FILE = 'ECLIPSE_base_CLE_V5a_NOx.nc'
ECLIPSE_FILE_url = ['https://iiasa.ac.at/web/home/research/researchPrograms/air/'
                    'ECLIPSE_V5a_baseline_CLE.tar.gz']
ECLIPSE_tar_FILE = ECLIPSE_FILE_url[0].split('/')[-1]

# TODO: review this class in light of the new strucutre of SupplyChain, i.e.
# no more global data needed

class ManufacturingExp(Exposures):
    """Approximates global manufacturing exposures. 
    Source of underlying data: 
    Greenhouse gas & Air pollution Interactions and Synergies (GAINS) model,
    International Institute for Applied Systems Analysis (IIASA),
    2015, "ECLIPSE V5a global emission fields" [Data file],
    http://www.iiasa.ac.at/web/home/research/researchPrograms/air/ECLIPSEv5a.html
    """
    def _constructor(self):
        return ManufacturingExp
    def init_manu_exp(self, assign_centroids=False):
        """Initialize.
        Parameters:
            assign_centroids (boolean): if False (default), no centroids are
            assigned to the data values. If True, assign centroids from the default 
            global centroids file (see GLB_CENTROIDS_MAT constant)"""

        if ECLIPSE_FILE not in os.listdir(SYSTEM_DIR):
            os.chdir(SYSTEM_DIR)
            try:
                download_file(ECLIPSE_FILE_url[0])
                LOGGER.debug('Download complete. Unzipping %s')
            except:
                LOGGER.error('Downloading manufacturing data failed.')
                raise
            
            with tarfile.open(ECLIPSE_tar_FILE) as f:
                f.extract(ECLIPSE_FILE)
                f.close()
            os.remove(ECLIPSE_tar_FILE)
        
        data = nc.Dataset(os.path.join(SYSTEM_DIR, ECLIPSE_FILE))
        time = np.array(data.variables['time'][:])
        current_i = np.nonzero(time == 2015)[0][0]
        lons = np.array(data.variables['lon'][:])
        lats = np.array(data.variables['lat'][:])
        raw_data = np.array(data.variables['emis_ind'][:])
        raw_data = (raw_data[current_i,:,:]).copy().T

        raw_data = raw_data.flatten(order='F').copy()
        # Adjust lat, lot to comply with the flattened data
        lats_orig = lats.copy()
        lats = np.repeat(lats, len(lons))
        lons = np.tile(lons, len(lats_orig))
        
        # Only consider location where emissions > than threshold. 
        # TODO: carefully review threshold
        self['longitude'] = lons[raw_data > 1e-1]
        self['latitude'] = lats[raw_data > 1e-1]
        self['value'] = raw_data[raw_data > 1e-1]
        
        # TODO: perhaps also include option of assigning user-defined centroids
        # also, as we already have lats and longs values, why do we need these?
        if assign_centroids == True:
            # Read centroids and assign region_ids
            centr = Centroids()
            centr.read_mat(GLB_CENTROIDS_MAT)
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
                                    'industry sector, approximated by global'\
                                    ' NOx emissions.'
            self.tag.file_name = os.path.join(SYSTEM_DIR, 
                                              'ECLIPSE_base_CLE_V5a_NOx.nc')
            self.value_unit = 'kt NOx'
            self.check()