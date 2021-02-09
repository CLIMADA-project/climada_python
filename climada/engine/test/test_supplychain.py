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

Test Supplychain class.
"""

import unittest
import os
import numpy as np

from climada.entity.exposures.base import Exposures
from climada.entity import ImpactFuncSet, IFTropCyclone
from climada.hazard.base import Hazard
from climada.engine.supplychain import SupplyChain
from climada.util.constants import SOURCE_DIR
from climada.util.constants import EXP_DEMO_H5

HAZ_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'hazard/test/data/')
HAZ_TEST_MAT = os.path.join(HAZ_DIR, 'atl_prob_no_name.mat')

TEST_DATA_DIR = os.path.join(SOURCE_DIR, 'engine', 'test', 'data')
TEST_WIOD = 'WIOTtest_Nov16_ROW.xlsb'
TEST_EXP = 'test_sup_exp.mat'
TEST_HAZ = 'test_hazard'
HAZ_DIR = os.path.join(SOURCE_DIR, 'hazard', 'test', 'data')
HAZ_TEST_MAT = os.path.join(HAZ_DIR, 'atl_prob_no_name.mat')

class TestSupplyChain(unittest.TestCase):
    """Testing the SupplyChain class."""
    def test_read_wiot(self):
        """Test reading of wiod table."""
        sup = SupplyChain()
        sup.read_wiot(year = 'test', file_path=TEST_DATA_DIR, rows_range=(5,117),
                      col_iso3=2, cols_data_range=(4,116), cols_sect_range=(1,61))

        self.assertAlmostEqual(sup.mriot_data[0, 0], 12924.1797, places=3)
        self.assertAlmostEqual(sup.mriot_data[0, -1], 0, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, 0], 0, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1], 22.222, places=3)
        
        self.assertAlmostEqual(sup.mriot_data[0, 0], 
                               sup.mriot_data[sup.cntry_pos[list(sup.cntry_pos)[0]][0], 
                                              sup.cntry_pos[list(sup.cntry_pos)[0]][0]], 
                               places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1], 
                               sup.mriot_data[sup.cntry_pos[list(sup.cntry_pos)[-1]][-1], 
                                              sup.cntry_pos[list(sup.cntry_pos)[-1]][-1]], 
                               places=3)        
        self.assertEqual(np.shape(sup.mriot_data), (112, 112))
        self.assertAlmostEqual(sup.total_prod.sum(), 3533367.89439, places=3)
        self.assertEqual(len(sup.countries_iso3), len(sup.countries))

    def test_calc_impact(self):
        """Test running direct and indirect impact calculations."""
        
        sup = SupplyChain()
        sup.read_wiot(year='test', file_path=TEST_DATA_DIR, rows_range=(5,117),
                      col_iso3=2, cols_data_range=(4,116), cols_sect_range=(1,61))

        # Tropical cyclone over Florida and Caribbean
        hazard = Hazard('TC')
        hazard.read_mat(HAZ_TEST_MAT)

        # Read demo entity values
        # Set the entity default file to the demo one
        exp = Exposures()
        exp.read_hdf5(EXP_DEMO_H5)
        exp.check()
        exp.region_id = 840 #assign right id for USA
        exp.assign_centroids(hazard)

        impf_tc= IFTropCyclone()
        impf_tc.set_emanuel_usa()
        impf_set = ImpactFuncSet()
        impf_set.append(impf_tc)
        impf_set.check()
        
        # Test direct impacts
        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      sec_subsec=None, 
                                      sector_type='manufacturing')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.cntry_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.cntry_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual(sup.cntry_dir_imp[0], 'USA')

        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      sec_subsec=None, 
                                      sector_type='agriculture')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.cntry_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.cntry_pos['USA']].sum(),
                                places = 3)

        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      sec_subsec=None, 
                                      sector_type='mining')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.cntry_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],),
                                sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.cntry_pos['USA']].sum(),
                                places = 3)
        
        sup.calc_sector_direct_impact(hazard, exp, impf_set,
                                      sec_subsec=None, 
                                      sector_type='service')
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.direct_impact.shape)
        self.assertAlmostEqual(sup.direct_impact.sum(), 
                                sup.direct_impact[:, sup.cntry_pos['USA']].sum(),
                                places = 3)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.direct_aai_agg.shape)
        self.assertAlmostEqual(sup.direct_aai_agg.sum(), 
                                sup.direct_aai_agg[sup.cntry_pos['USA']].sum(),
                                places = 3)
        # Test indirect impacts
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

        # Test total impacts
        sup.calc_total_impact()
        self.assertAlmostEqual((sup.years.shape[0], sup.mriot_data.shape[0]),
                                sup.total_impact.shape)
        self.assertAlmostEqual((sup.mriot_data.shape[0],), sup.total_aai_agg.shape)
 
## Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSupplyChain)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
    