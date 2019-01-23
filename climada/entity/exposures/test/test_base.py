"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test Exposure base class.
"""
import os
import unittest
import numpy as np
import pandas as pd
import geopandas
from sklearn.neighbors import DistanceMetric
from climada.util.coordinates import coord_on_land

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, INDICATOR_IF, INDICATOR_CENTR, add_sea
from climada.hazard.base import Hazard
from climada.util.constants import ENT_TEMPLATE_XLS, ONE_LAT_KM

DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'test', 'data')
ENT_TEST_XLS = os.path.join(DATA_DIR, 'demo_today.xlsx')

def good_exposures():
    """Followng values are defined for each exposure"""
    data = {}
    data['latitude'] = np.array([ 1, 2, 3])
    data['longitude'] = np.array([ 2, 3, 4])
    data['value'] = np.array([1, 2, 3])
    data['deductible'] = np.array([1, 2, 3])
    data[INDICATOR_IF + 'NA'] = np.array([1, 2, 3])
    data['category_id'] = np.array([1, 2, 3])
    data['region_id'] = np.array([1, 2, 3])
    data[INDICATOR_CENTR + 'TC'] = np.array([1, 2, 3])
    
    expo = Exposures(geopandas.GeoDataFrame(data=data))
    return expo

class TestAssign(unittest.TestCase):
    """Check assign function"""

    def test_assign_pass(self):
        """ Check that assigned attribute is correctly set."""
        # Fill with dummy values the GridPoints
        expo = good_exposures()
        # Fill with dummy values the centroids
        haz = Hazard()
        haz.tag.haz_type = 'TC'
        haz.centroids.coord = np.ones((expo.shape[0]+6, 2))
        # assign
        expo.assign(haz)

        # check assigned variable has been set with correct length
        self.assertEqual(expo.shape[0], len(expo[INDICATOR_CENTR + 'TC']))

class TestChecker(unittest.TestCase):
    """Test logs of check function """

    def test_info_logs(self):
        """Wrong exposures definition"""
        expo = good_exposures()

        with self.assertLogs('climada.entity.exposures.base', level='INFO') as cm:
            expo.check()
        self.assertIn('tag metadata set to default value', cm.output[0])
        self.assertIn('ref_year metadata set to default value', cm.output[1])
        self.assertIn('value_unit metadata set to default value', cm.output[2])
        self.assertIn('geometry not set', cm.output[3])
        self.assertIn('cover not set', cm.output[4])
        
    def test_error_logs(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo = expo.drop(['longitude'],axis=1)

        with self.assertLogs('climada.entity.exposures.base', level='ERROR') as cm:
            with self.assertRaises(ValueError):
                expo.check()
        self.assertIn('longitude missing', cm.output[0])

class TestReader(unittest.TestCase):
    """ Check constructor Exposures through DataFrames readers """

    def test_read_template(self):
        """Wrong exposures definition"""
        df = pd.read_excel(ENT_TEMPLATE_XLS)
        exp_df = Exposures(df)
        # set metadata
        exp_df.ref_year = 2020
        exp_df.tag = Tag(ENT_TEMPLATE_XLS, 'ENT_TEMPLATE_XLS')
        exp_df.value_unit = 'XSD'
        exp_df.check()

class TestAddSea(unittest.TestCase):
    """ Check constructor Exposures through DataFrames readers """
    def test_add_sea_pass(self):
        """Test add_sea function with fake data."""
        exp = Exposures()

        exp['value'] = np.arange(0, 1.0e6, 1.0e5)

        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp['latitude'] = np.linspace(min_lat, max_lat, 10)
        exp['longitude'] = np.linspace(min_lon, max_lon, 10)
        exp['region_id'] = np.ones(10)
        exp['if_TC'] = np.ones(10)
        exp.ref_year = 2015
        exp.value_unit = 'XSD'

        sea_coast = 100
        sea_res_km = 50
        sea_res = (sea_coast, sea_res_km)
        exp_sea = add_sea(exp, sea_res)
        exp_sea.check()

        sea_coast /= ONE_LAT_KM
        sea_res_km /= ONE_LAT_KM

        min_lat = min_lat - sea_coast
        max_lat = max_lat + sea_coast
        min_lon = min_lon - sea_coast
        max_lon = max_lon + sea_coast
        self.assertEqual(np.min(exp_sea.latitude), min_lat)
        self.assertEqual(np.min(exp_sea.longitude), min_lon)
        self.assertTrue(np.array_equal(exp_sea.value.values[:10], np.arange(0, 1.0e6, 1.0e5)))
        self.assertEqual(exp_sea.ref_year, exp.ref_year)
        self.assertEqual(exp_sea.value_unit, exp.value_unit)

        on_sea_lat = exp_sea.latitude.values[11:]
        on_sea_lon = exp_sea.longitude.values[11:]
        res_on_sea = coord_on_land(on_sea_lat, on_sea_lon)
        res_on_sea = np.logical_not(res_on_sea)
        self.assertTrue(np.all(res_on_sea))

        dist = DistanceMetric.get_metric('haversine')
        self.assertAlmostEqual(dist.pairwise([[exp_sea.longitude.values[-1], \
            exp_sea.latitude.values[-1]], [exp_sea.longitude.values[-2], \
            exp_sea.latitude.values[-2]]])[0][1], sea_res_km)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestChecker)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAssign))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReader))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAddSea))
unittest.TextTestRunner(verbosity=2).run(TESTS)
