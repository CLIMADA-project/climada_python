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
import glob
import numpy as np

from climada.entity.exposures.base import Exposures
from climada.entity.entity_def import Entity
from climada.hazard.base import Hazard
from climada.engine.supplychain import SupplyChain
from climada.util.constants import SOURCE_DIR

TEST_DATA_DIR = os.path.join(SOURCE_DIR, 'engine', 'test', 'data', 'supplychain')
TEST_EXP_DIR = os.path.join(TEST_DATA_DIR, 'test_exposures')
TEST_WIOD = 'test_wiod.xlsx'
TEST_EXP = 'test_sup_exp.mat'
TEST_HAZ = 'test_hazard'
HAZ_DIR = os.path.join(SOURCE_DIR, 'hazard', 'test', 'data')
HAZ_TEST_MAT = os.path.join(HAZ_DIR, 'atl_prob_no_name.mat')

class TestSupplyChain(unittest.TestCase):
    """Testing the SupplyChain class."""
    def test_read_wiod(self):
        """Test reading of wiod table with mini-version of
        original table."""
        sup = SupplyChain()
        sup.read_wiod(file_path=TEST_DATA_DIR, file_name=TEST_WIOD, \
                      rows=112, cols='E:DL', tot_prod_col='DM')
        mainsectors = ['agriculture', 'forestry_fishing', 'manufacturing',\
                            'mining_quarrying', 'services', 'utilities']
        self.assertAlmostEqual(sup.mriot_data[0, 0], 12924.1797, places=3)
        self.assertAlmostEqual(sup.mriot_data[-1, -1], 22.222, places=3)
        self.assertEqual(np.shape(sup.mriot_data), (112, 112))
        self.assertAlmostEqual(sum(sup.total_prod), 3533367.88300, places=3)
        self.assertEqual(len(sup.countries_iso3), sup.n_countries*sup.n_sectors)
        self.assertEqual(len(sup.countries_iso3), len(sup.countries))
        self.assertEqual(len(sup.countries_iso3), len(sup.sectors))
        self.assertEqual(set(sup.main_sectors), set(mainsectors))

        #Testing aggregation of labels:
        aggregated_dict_keys = ['sectors', 'aggregation_info', 'countries', \
                                'countries_iso3'] 
        aggregated_ctries = ['Australia', 'Austria']
        aggregation_info_keys = ['agriculture', 'forestry_fishing', \
                                 'manufacturing', 'mining_quarrying', \
                                 'services', 'utilities']
        
        self.assertEqual(list(sup.aggregated_mriot.keys()), \
                         aggregated_dict_keys)
        self.assertEqual(len(sup.aggregated_mriot['countries']), 12)
        self.assertEqual(len(sup.aggregated_mriot['countries']), \
                         len(sup.aggregated_mriot['countries_iso3']))
        self.assertEqual(len(sup.aggregated_mriot['countries']), \
                         len(sup.aggregated_mriot['sectors']))
        self.assertEqual(set(sup.aggregated_mriot['countries']), \
                         set(aggregated_ctries))
        self.assertEqual(set(sup.aggregated_mriot['aggregation_info'].keys()),
                         set(aggregation_info_keys))
        
    # THE FUNCTIONALITY OF METHOD DEFAULT EXPOSURE IS INHERENTLY DEPENDENT ON USING
    # THE ACTUAL EXPOSURE RAW DATA AND USING OTHER DATA FOR TESTING WOULD NOT
    # ACTUALLY TEST ITS FUNCTIONALITY. FOR NOW, TESTING IS NOT IMPLEMENTED DUE
    # TO LENGTHY COMPUTATION TIME.
    # def test_default_exp(self):
    #     """Test method which creates default exposure files, including the
    #     internal methods that are called by the main method to create
    #     the exposures. Note that these can't rely on simplified test data
    #     as their functionality is specifically tied to the respective raw
    #     data files. For succesful testing, the raw data files hence need to be
    #     in the correct location (data/system).
        
    #     Note that these methods do not do anything but writing new files. 
    #     We assert whether these files were created (in the proper location and
    #     with proper name).
        
    #     Note also that these tests also test the respective exposures classes
    #     modules that are used by default_exposures().
    #     """
    #     file_names = {'agriculture': 'GLB_agriculture_XXX',
    #           'forestry_fishing': 'GLB_forestry_fishing_XXX',
    #           'utilities': 'GLB_utilities_XXX',
    #           'services': 'GLB_services_XXX',
    #           'mining_quarrying': 'GLB_mining_quarrying_XXX',
    #           'manufacturing': 'GLB_manufacturing_XXX'}
    #     ## First the individual methods...
    #     ## Manufacturing:
    #     sup._create_manu_expo(file_names)
    #     required_file_in_folder = ['GLB_manufacturing_XXX']
    #     actual_file_in_folder = [os.path.basename(path) for path in \
    #                 (glob.glob(os.path.join(SUP_DATA_DIR, '*GLB_manu*')))]
    #     self.assertEqual(required_file_in_folder, actual_file_in_folder)
        
    #     ## Utilities:
    #     sup._create_utilities_expo(file_names)
    #     required_file_in_folder = ['GLB_utilities_XXX']
    #     actual_file_in_folder = [os.path.basename(path) for path in \
    #                 (glob.glob(os.path.join(SUP_DATA_DIR, '*GLB_util*')))]
    #     self.assertEqual(required_file_in_folder, actual_file_in_folder)
        
    #     ## Forestry and Fishing:
    #     sup._create_forest_expo(file_names)
    #     required_file_in_folder = ['GLB_forestry_fishing_XXX']
    #     actual_file_in_folder = [os.path.basename(path) for path in \
    #                 (glob.glob(os.path.join(SUP_DATA_DIR, '*GLB_forest*')))]
    #     self.assertEqual(required_file_in_folder, actual_file_in_folder)
        
    #     ## Mining and Quarrying:
    #     sup._create_mining_expo(file_names)
    #     required_file_in_folder = ['GLB_mining_quarrying_XXX']
    #     actual_file_in_folder = [os.path.basename(path) for path in \
    #                 (glob.glob(os.path.join(SUP_DATA_DIR, '*GLB_mining*')))]
    #     self.assertEqual(required_file_in_folder, actual_file_in_folder)
        
    #     ## Agriculture:
    #     sup._create_agri_expo(file_names)
    #     required_file_in_folder = ['GLB_agriculture_XXX']
    #     actual_file_in_folder = [os.path.basename(path) for path in \
    #                 (glob.glob(os.path.join(SUP_DATA_DIR, '*GLB_agri*')))]
    #     self.assertEqual(required_file_in_folder, actual_file_in_folder)
        
    #     ## Services:
    #     sup._create_services_expo()
    #     required_files_in_folder = 201
    #     actual_files_in_folder = len(glob.glob(os.path.join(SUP_DATA_DIR, \
    #                                                 '*_services_*')))
    #     self.assertEqual(required_files_in_folder, actual_files_in_folder)
        
    #     ## Now the main method...
    #     with self.assertRaises(UserWarning):
    #       sup.default_exposures()
    #     # Method does not return anything but writes new files. We assert
    #     # whether these files were created (in the proper location and with
    #     # proper name).
    #     required_files_in_folder = ['GLB_agriculture_XXX',\
    #                                 'GLB_forestry_fishing_XXX',\
    #                                 'GLB_manufacturing_XXX',\
    #                                 'GLB_mining_quarrying_XXX',\
    #                                 'GLB_services_XXX',\
    #                                 'GLB_utilities_XXX']
        
    #     actual_files_in_folder = [os.path.basename(path) for path in \
    #                         (glob.glob(os.path.join(SUP_DATA_DIR, '*GLB_*')))]
    #     self.assertEqual(required_files_in_folder, actual_files_in_folder)
        
    def test_prepare_exposures(self):
        sup = SupplyChain()
        sup.prepare_exposures(files_source=os.path.join(TEST_EXP_DIR, 'pre-preparation'),\
                              files_target=TEST_EXP_DIR, remove_restofw_ctries=False)
        required_files_in_folder = 18
        actual_files_in_folder = len(glob.glob(os.path.join(TEST_EXP_DIR, \
                                                            '*XXX')))
        self.assertEqual(required_files_in_folder, actual_files_in_folder) 
        
        che_exp = Exposures()
        che_exp.read_hdf5(os.path.join(TEST_EXP_DIR, 'pre-preparation', 'CHE_services_XXX'))
        chn_exp = Exposures()
        chn_exp.read_hdf5(os.path.join(TEST_EXP_DIR, 'pre-preparation', 'CHN_services_XXX'))
        row_exp = Exposures()
        row_exp.read_hdf5(os.path.join(TEST_EXP_DIR, 'ROW_services_XXX'))
        self.assertAlmostEqual(row_exp.value[0], che_exp.value[0]/(sum(che_exp.value)+sum(chn_exp.value)))
    
    # The following is a pseudo-test, as the method's core functionality - i.e. creating
    # the supplychain default global TC hazard - cannot be tested with dummy data.
    # create_default_haz() uses a sequence of climada functions which are tested separately
    # anyway, however.
    def test_create_default_haz(self):
        sup = SupplyChain()
        haz = sup.create_default_haz(save_haz=False, file_path=TEST_DATA_DIR, file_name='test_hazard')
        self.assertEqual(len(haz.event_id), len(haz.event_name))
        self.assertEqual(haz.tag.haz_type, 'TC')
    
    def test_calc_direct_impact(self):
        sup = SupplyChain()
        # Write dummy exposures to test data directory (required to proceed).
        unique_ctries, ind = np.unique(sup.countries_iso3, return_index=True)
        unique_ctries = np.append(unique_ctries[np.argsort(ind)], 'ROW')
        unique_msect, ind = np.unique(sup.main_sectors, return_index=True)
        unique_msect = list(unique_msect[np.argsort(ind)])
        for ctry in unique_ctries:
            for msect in unique_msect:
                exp = Exposures()
                ent = Entity()
                ent.read_mat(os.path.join(TEST_DATA_DIR, TEST_EXP))
                exp = ent.exposures
                exp.check()
                del(ent)
                exp.write_hdf5(os.path.join(TEST_EXP_DIR, 'imp_exp', ctry+'_'+msect+'_XXX'))
        required_files_in_folder = 18
        actual_files_in_folder = len(glob.glob(os.path.join(TEST_EXP_DIR, \
                                                'imp_exp','*XXX')))
        self.assertEqual(required_files_in_folder, actual_files_in_folder) 
        
        # Testing actual calculations. First create test hazard based on
        # default hazard file:
        haz = Hazard('TC')
        haz.read_hdf5(os.path.join(TEST_DATA_DIR, TEST_HAZ))
        sup.calc_direct_impact(haz, exp_source_path=os.path.join(TEST_EXP_DIR,\
                            'imp_exp'), imp_target_path=TEST_DATA_DIR)
        
        # Assertions:
        self.assertEqual(np.mean(sup.direct_impact[:,2]), sup.direct_aai_agg[2])
        self.assertEqual(len(sup.direct_impact[0,:]), len(sup.direct_aai_agg))
        
    def test_calc_indirect_impact(self):
        sup = SupplyChain()
        # Leontief approach:
        sup.calc_indirect_impact(io_approach='leontief')
        self.assertEqual(np.mean(sup.indirect_impact[:,2]), sup.indirect_aai_agg[2])
        self.assertEqual(len(sup.indirect_impact[0,:]), len(sup.indirect_aai_agg))
        shape = np.shape(sup.io_data['risk_structure'])
        self.assertEqual(shape, np.shape(sup.mriot_data) + (len(sup.years), ))
        self.assertEqual(np.shape(sup.mriot_data), np.shape(sup.io_data['inverse']))
        self.assertEqual(np.shape(sup.indirect_impact), np.shape(sup.direct_impact))
        # Ghosh approach:
        sup.calc_indirect_impact(io_approach='ghosh')
        sup.calc_indirect_impact(io_approach='leontief')
        self.assertEqual(np.mean(sup.indirect_impact[:,2]), sup.indirect_aai_agg[2])
        self.assertEqual(len(sup.indirect_impact[0,:]), len(sup.indirect_aai_agg))
        shape = np.shape(sup.io_data['risk_structure'])
        self.assertEqual(shape, np.shape(sup.mriot_data) + (len(sup.years), ))
        self.assertEqual(np.shape(sup.mriot_data), np.shape(sup.io_data['inverse']))
        self.assertEqual(np.shape(sup.indirect_impact), np.shape(sup.direct_impact))
        # Env extended io approach:
        sup.calc_indirect_impact(io_approach='eeio')
        sup.calc_indirect_impact(io_approach='leontief')
        self.assertEqual(np.mean(sup.indirect_impact[:,2]), sup.indirect_aai_agg[2])
        self.assertEqual(len(sup.indirect_impact[0,:]), len(sup.indirect_aai_agg))
        shape = np.shape(sup.io_data['risk_structure'])
        self.assertEqual(shape, np.shape(sup.mriot_data) + (len(sup.years), ))
        self.assertEqual(np.shape(sup.mriot_data), np.shape(sup.io_data['inverse']))
        self.assertEqual(np.shape(sup.indirect_impact), np.shape(sup.direct_impact))
        
    def test_calc_total_impact(self):
        sup = SupplyChain()
        sup.calc_total_impact()
        self.assertEqual(np.shape(sup.indirect_impact), np.shape(sup.total_impact))
        self.assertTrue(np.all(sup.indirect_impact+sup.direct_impact == sup.total_impact))

        
# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSupplyChain)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
        
        
        
