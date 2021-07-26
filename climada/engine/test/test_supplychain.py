"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test Supplychain class.
"""

import unittest
import numpy as np

from climada import CONFIG
from climada.entity.exposures.base import Exposures
from climada.entity import ImpactFuncSet, ImpfTropCyclone
from climada.hazard.base import Hazard
from climada.engine.supplychain import SupplyChain
from climada.util.constants import EXP_DEMO_H5

HAZ_TEST_MAT = CONFIG.hazard.test_data.dir().joinpath('atl_prob_no_name.mat')
DIR_TEST_DATA = CONFIG.engine.test_data.dir()

class TestSupplyChain(unittest.TestCase):
    """Testing the SupplyChain class."""
    def test_read_wiot(self):
        """Test reading of wiod table."""
        sup = SupplyChain()
        sup.read_wiod16(year='test',
                        range_rows=(5,117), 
                        range_cols=(4,116), 
                        col_iso3=2, col_sectors=1)
        
        self.assertAlmostEqual(sup.mriot_data[0, 0], 12924.1797, places=3)
        self.assertAlmostEqual(sup.mriot_data[0, -1], 0, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, 0], 0, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1], 22.222, places=3)
        
        self.assertAlmostEqual(sup.mriot_data[0, 0], 
                               sup.mriot_data[sup.reg_pos[list(sup.reg_pos)[0]][0], 
                                              sup.reg_pos[list(sup.reg_pos)[0]][0]], 
                               places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1], 
                               sup.mriot_data[sup.reg_pos[list(sup.reg_pos)[-1]][-1], 
                                              sup.reg_pos[list(sup.reg_pos)[-1]][-1]], 
                               places=3)        
        self.assertEqual(np.shape(sup.mriot_data), (112, 112))
        self.assertAlmostEqual(sup.total_prod.sum(), 3533367.89439, places=3)

    def calc_sector_direct_impact(self):
        """Test running direct impact calculations."""
        
        sup = SupplyChain()
        sup.read_wiod16(year='test', 
                        range_rows=(5,117), 
                        range_cols=(4,116), 
                        col_iso3=2, col_sectors=1)
        
        # Tropical cyclone over Florida and Caribbean
        hazard = Hazard('TC')
        hazard.read_mat(HAZ_TEST_MAT)

        # Read demo entity values
        # Set the entity default file to the demo one
        exp = Exposures()
        exp.read_hdf5(EXP_DEMO_H5)
        exp.check()
        exp.gdf.region_id = 840 #assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc= ImpfTropCyclone()
        impf_tc.set_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()
        
        subsecs = list(range(10))+list(range(15,25))
        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      selected_subsec=subsecs)
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(),
                                sup.direct_impact[:, sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual(sup.reg_dir_imp[0], 'USA')
        self.assertAlmostEqual(sup.direct_impact.sum(),
                               sup.direct_impact[:, subsecs].sum(), places=3)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                               sup.direct_aai_agg[subsecs].sum(), places=3)
        
        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      selected_subsec='manufacturing')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual(sup.reg_dir_imp[0], 'USA')
        self.assertAlmostEqual(sup.direct_impact.sum(),
                               sup.direct_impact[:, range(4,23)].sum(), places=3)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                               sup.direct_aai_agg[range(4,23)].sum(), places=3)

        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      selected_subsec='agriculture')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual(sup.direct_impact.sum(),
                               sup.direct_impact[:,  range(0,1)].sum(), places=3)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                               sup.direct_aai_agg[ range(0,1)].sum(), places=3)

        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      selected_subsec='mining')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual(sup.direct_impact.sum(),
                               sup.direct_impact[:, range(3,4)].sum(), places=3)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                               sup.direct_aai_agg[range(3,4)].sum(), places=3)
        
        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      selected_subsec='service')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.reg_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual(sup.direct_impact.sum(),
                               sup.direct_impact[:, range(26,56)].sum(), places=3)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                               sup.direct_aai_agg[range(26,56)].sum(), places=3)

    def test_calc_sector_indirect_impact(self):
        """Test running indirect impact calculations."""
        
        sup = SupplyChain()
        sup.read_wiod16(year='test', 
                        range_rows=(5,117), 
                        range_cols=(4,116), 
                        col_iso3=2, col_sectors=1)

        # Tropical cyclone over Florida and Caribbean
        hazard = Hazard('TC')
        hazard.read_mat(HAZ_TEST_MAT)

        # Read demo entity values
        # Set the entity default file to the demo one
        exp = Exposures()
        exp.read_hdf5(EXP_DEMO_H5)
        exp.check()
        exp.gdf.region_id = 840 #assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc= ImpfTropCyclone()
        impf_tc.set_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        sup.calc_sector_direct_impact(hazard, exp, impf_set)
        sup.calc_indirect_impact(io_approach='ghosh')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.indirect_impact.shape)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.indirect_aai_agg.shape)
        self.assertAlmostEqual(sup.mriot_data.shape, sup.io_data['inverse'].shape)
        self.assertAlmostEqual(sup.io_data['risk_structure'].shape, 
                               (sup.mriot_data.shape[0], sup.mriot_data.shape[1],
                                sup.years.shape[0]))
        self.assertAlmostEqual('ghosh', sup.io_data['io_approach'])

        sup.calc_indirect_impact(io_approach='leontief')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.indirect_impact.shape)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.indirect_aai_agg.shape)
        self.assertAlmostEqual(sup.mriot_data.shape, sup.io_data['inverse'].shape)
        self.assertAlmostEqual(sup.io_data['risk_structure'].shape, 
                               (sup.mriot_data.shape[0], sup.mriot_data.shape[1],
                                sup.years.shape[0]))
        self.assertAlmostEqual('leontief', sup.io_data['io_approach'])

        sup.calc_indirect_impact(io_approach='eeioa')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.indirect_impact.shape)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.indirect_aai_agg.shape)
        self.assertAlmostEqual(sup.mriot_data.shape, sup.io_data['inverse'].shape)
        self.assertAlmostEqual(sup.io_data['risk_structure'].shape, 
                               (sup.mriot_data.shape[0], sup.mriot_data.shape[1],
                                sup.years.shape[0]))
        self.assertAlmostEqual('eeioa', sup.io_data['io_approach'])        

    def test_calc_sector_total_impact(self):
        """Test running total impact calculations.""" 
        sup = SupplyChain()
        sup.read_wiod16(year='test', 
                        range_rows=(5,117), 
                        range_cols=(4,116), 
                        col_iso3=2, col_sectors=1)

        # Tropical cyclone over Florida and Caribbean
        hazard = Hazard('TC')
        hazard.read_mat(HAZ_TEST_MAT)

        # Read demo entity values
        # Set the entity default file to the demo one
        exp = Exposures()
        exp.read_hdf5(EXP_DEMO_H5)
        exp.check()
        exp.gdf.region_id = 840 #assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc= ImpfTropCyclone()
        impf_tc.set_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()

        sup.calc_sector_direct_impact(hazard, exp, impf_set)        
        sup.calc_indirect_impact(io_approach='ghosh')
        sup.calc_total_impact()

        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.total_impact.shape)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.total_aai_agg.shape)
 
## Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSupplyChain)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
