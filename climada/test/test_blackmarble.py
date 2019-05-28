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

Tests on Black marble.
"""

import unittest
import numpy as np
from cartopy.io import shapereader
from scipy import sparse

from climada.entity.exposures.black_marble import BlackMarble
from climada.entity.exposures.nightlight import load_nightlight_nasa, \
load_nightlight_noaa, NOAA_BORDER, cut_nl_nasa
from climada.entity.exposures import nightlight as nl_utils

class Test2013(unittest.TestCase):
    """Test black marble of previous in 2013."""

    def test_spain_pass(self):
        country_name = ['Spain']
        ent = BlackMarble()
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=1)
        self.assertIn('GDP ESP 2013: 1.362e+12.', cm.output[0])
        self.assertIn('Income group ESP 2013: 4.', cm.output[1])

        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=1)
        self.assertIn("Nightlights from NOAA's earth observation group for year 2013.", cm.output[0])
        self.assertIn("Processing country Spain.", cm.output[1])
        self.assertIn("Generating resolution of approx 1 km.", cm.output[2])
        self.assertTrue(np.isclose(ent.value.sum(), 1.362e+12*(4+1), 4))
        self.assertEqual(ent.crs, {'init': 'epsg:4326'})

    def test_sint_maarten_pass(self):
        country_name = ['Sint Maarten']
        ent = BlackMarble()
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertIn('GDP SXM 2014: 3.658e+08.', cm.output[0])
        self.assertIn('Income group SXM 2013: 4.', cm.output[1])
        self.assertEqual(ent.crs, {'init': 'epsg:4326'})

        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertIn("Nightlights from NOAA's earth observation group for year 2013.", cm.output[0])
        self.assertIn("Processing country Sint Maarten.", cm.output[1])
        self.assertIn("Generating resolution of approx 0.2 km.", cm.output[2])
        self.assertAlmostEqual(ent.value.sum(), 3.658e+08*(4+1))
        self.assertEqual(ent.crs, {'init': 'epsg:4326'})

    def test_anguilla_pass(self):
        country_name = ['Anguilla']
        ent = BlackMarble()
        ent.set_countries(country_name, 2013, res_km=0.2)
        self.assertEqual(ent.ref_year, 2013)
        self.assertIn("Anguilla 2013 GDP: 1.754e+08 income group: 3", ent.tag.description)
        self.assertAlmostEqual(ent.value.sum(), 1.754e+08*(3+1))
        self.assertEqual(ent.crs, {'init': 'epsg:4326'})

class Test1968(unittest.TestCase):
    """Test black marble of previous years to 1992."""
    def test_switzerland_pass(self):
        country_name = ['Switzerland']
        ent = BlackMarble()
        with self.assertLogs('climada.util.finance', level='INFO') as cm:
            ent.set_countries(country_name, 1968, res_km=0.5)
        self.assertIn('GDP CHE 1968: 1.894e+10.', cm.output[0])
        self.assertIn('Income group CHE 1987: 4.', cm.output[1])

        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 1968, res_km=0.5)
        self.assertIn("Nightlights from NOAA's earth observation group for year 1992.", cm.output[0])
        self.assertTrue("Processing country Switzerland." in cm.output[-2])
        self.assertTrue("Generating resolution of approx 0.5 km." in cm.output[-1])
        self.assertTrue(np.isclose(ent.value.sum(), 1.894e+10*(4+1), 4))
        self.assertEqual(ent.crs, {'init': 'epsg:4326'})

class Test2012(unittest.TestCase):
    """Test year 2012 flags."""
    
    def test_from_hr_flag_pass(self):
        """Check from_hr flag in set_countries method."""
        country_name = ['Turkey']
        
        ent = BlackMarble()
        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2012, res_km=5.0)
        self.assertTrue('NOAA' in cm.output[-3])
        size1 = ent.value.size
        self.assertTrue(np.isclose(ent.value.sum(), 8.740e+11*(3+1), 4))
    
        try:
            ent = BlackMarble()
            with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
                ent.set_countries(country_name, 2012, res_km=5.0, from_hr=True)
                self.assertTrue('NASA' in cm.output[-3])
                size2 = ent.value.size
                self.assertTrue(size1 < size2)
                self.assertTrue(np.isclose(ent.value.sum(), 8.740e+11*(3+1), 4))
        except TypeError:
            print('MemoryError caught')
            pass
        
    
        ent = BlackMarble()
        with self.assertLogs('climada.entity.exposures.black_marble', level='INFO') as cm:
            ent.set_countries(country_name, 2012, res_km=5.0, from_hr=False)
        self.assertTrue('NOAA' in cm.output[-3])
        self.assertTrue(np.isclose(ent.value.sum(), 8.740e+11*(3+1), 4))
        size3 = ent.value.size
        self.assertEqual(size1, size3)
        self.assertEqual(ent.crs, {'init': 'epsg:4326'})

class BMFuncs(unittest.TestCase):
    """Test plot functions."""
    def test_cut_nasa_esp_pass(self):
        """Test load_nightlight_nasa function."""
        shp_fn = shapereader.natural_earth(resolution='10m',
                                           category='cultural', 
                                           name='admin_0_countries')
        shp_file = shapereader.Reader(shp_fn)
        list_records = list(shp_file.records())
        for info_idx, info in enumerate(list_records):
            if info.attributes['ADM0_A3'] == 'AIA':
                bounds = info.bounds
        
        req_files = nl_utils.check_required_nl_files(bounds)
        files_exist, _ = nl_utils.check_nl_local_file_exists(req_files)
        nl_utils.download_nl_files(req_files, files_exist)
        
        try:
            nightlight, coord_nl = load_nightlight_nasa(bounds, req_files, 2016)
        except TypeError:
            print('MemoryError caught')
            return
    
        self.assertTrue(coord_nl[0, 0] < bounds[1])
        self.assertTrue(coord_nl[1, 0] < bounds[0])
        self.assertTrue(coord_nl[0, 0]+(nightlight.shape[0]-1)*coord_nl[0,1] > bounds[3])
        self.assertTrue(coord_nl[1, 0]+(nightlight.shape[1]-1)*coord_nl[1,1] > bounds[2])

    def test_load_noaa_pass(self):
        """Test load_nightlight_noaa function."""
        nightlight, coord_nl, fn_nl = load_nightlight_noaa(2013)
           
        self.assertEqual(coord_nl[0, 0], NOAA_BORDER[1])
        self.assertEqual(coord_nl[1, 0], NOAA_BORDER[0])
        self.assertEqual(coord_nl[0, 0]+(nightlight.shape[0]-1)*coord_nl[0,1], NOAA_BORDER[3])
        self.assertEqual(coord_nl[1, 0]+(nightlight.shape[1]-1)*coord_nl[1,1], NOAA_BORDER[2])

    def test_set_country_pass(self):
        """Test exposures attributes after black marble."""
        country_name = ['Switzerland', 'Germany']
        ent = BlackMarble()
        ent.set_countries(country_name, 2013, res_km=5.0)
        ent.check()
                
        self.assertEqual(np.unique(ent.region_id).size, 2)
        self.assertEqual(ent.ref_year, 2013)
        self.assertIn('Switzerland 2013 GDP: ', ent.tag.description)
        self.assertIn('Germany 2013 GDP: ', ent.tag.description)
        self.assertIn('income group: 4', ent.tag.description)
        self.assertIn('income group: 4', ent.tag.description)
        self.assertIn('F182013.v4c_web.stable_lights.avg_vis.p', ent.tag.file_name)
        self.assertIn('F182013.v4c_web.stable_lights.avg_vis.p', ent.tag.file_name)
        
    def test_cut_nl_nasa_1_pass(self):
        """Test cut_nl_nasa situation 2->3->4->5."""
        nl_mat = sparse.lil.lil_matrix([])
        in_lat = (21599, 21600)
        in_lon = (43199, 43200)
        # 0 2 4 6    (lat: Upper=0)   (lon: 0, 1, 2, 3)
        # 1 3 5 7    (lat: Lower=1)   (lon: 0, 1, 2, 3)
        in_lat_nb = (1, 0)
        in_lon_nb = (1, 2)
        
        idx_info = [2, -1, False]
        try:
            aux_nl = np.zeros((21600, 21600))
            aux_nl[21599, 21599] = 100
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                                            in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (1, 1))
            self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
            
            idx_info[0] = 3
            idx_info[1] = 2
            aux_nl[21599, 21599] = 0
            aux_nl[0, 21599] = 101
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (2, 1))
            self.assertEqual(nl_mat.tocsr()[0, 0], 101.0)
            self.assertEqual(nl_mat.tocsr()[1, 0], 100.0)
            
            idx_info[0] = 4
            idx_info[1] = 3
            aux_nl[0, 21599] = 0
            aux_nl[21599, 0] = 102
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (2, 2))
            self.assertEqual(nl_mat.tocsr()[0, 0], 101.0)
            self.assertEqual(nl_mat.tocsr()[1, 0], 100.0)
            self.assertEqual(nl_mat.tocsr()[0, 1], 0.0)
            self.assertEqual(nl_mat.tocsr()[1, 1], 102.0)
    
            idx_info[0] = 5
            idx_info[1] = 4
            aux_nl[21599, 0] = 0
            aux_nl[0, 0] = 103
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (2, 2))
            self.assertEqual(nl_mat.tocsr()[0, 0], 101.0)
            self.assertEqual(nl_mat.tocsr()[1, 0], 100.0)
            self.assertEqual(nl_mat.tocsr()[0, 1], 103.0)
            self.assertEqual(nl_mat.tocsr()[1, 1], 102.0)
        except MemoryError:
            print('MemoryError caught')
            pass
        
    def test_cut_nl_nasa_2_pass(self):
        """Test cut_nl_nasa situation 3->5."""
        nl_mat = sparse.lil.lil_matrix([])
        in_lat = (21599, 21599)
        in_lon = (43199, 43200)
        # 0 2 4 6    (lat: Upper=0)   (lon: 0, 1, 2, 3)
        # 1 3 5 7    (lat: Lower=1)   (lon: 0, 1, 2, 3)
        in_lat_nb = (1, 1)
        in_lon_nb = (1, 2)
        
        idx_info = [3, -1, False]
        try:
            aux_nl = np.zeros((21600, 21600))
            aux_nl[0, 21599] = 100
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (1, 1))
            self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
            
            idx_info[0] = 5
            idx_info[1] = 3
            aux_nl[0, 21599] = 0
            aux_nl[0, 0] = 101
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (1, 2))
            self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
            self.assertEqual(nl_mat.tocsr()[0, 1], 101.0)
        except MemoryError:
            print('MemoryError caught')
            pass

    def test_cut_nl_nasa_3_pass(self):
        """Test cut_nl_nasa situation 2->4."""
        nl_mat = sparse.lil.lil_matrix([])
        in_lat = (21600, 21600)
        in_lon = (43199, 43200)
        # 0 2 4 6    (lat: Upper=0)   (lon: 0, 1, 2, 3)
        # 1 3 5 7    (lat: Lower=1)   (lon: 0, 1, 2, 3)
        in_lat_nb = (0, 0)
        in_lon_nb = (1, 2)
    
        idx_info = [2, -1, False]
        try:
            aux_nl = np.zeros((21600, 21600))
            aux_nl[21599, 21599] = 100
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (1, 1))
            self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
            
            idx_info[0] = 4
            idx_info[1] = 2
            aux_nl[21599, 21599] = 0
            aux_nl[21599, 0] = 101
            cut_nl_nasa(aux_nl, idx_info, nl_mat, in_lat, 
                        in_lon, in_lat_nb, in_lon_nb)
            
            self.assertEqual(nl_mat.shape, (1, 2))
            self.assertEqual(nl_mat.tocsr()[0, 0], 100.0)
            self.assertEqual(nl_mat.tocsr()[0, 1], 101.0)
        except MemoryError:
            print('MemoryError caught')
            pass

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(Test2013)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(Test1968))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(Test2012))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(BMFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
