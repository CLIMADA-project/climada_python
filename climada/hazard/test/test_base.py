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

Test Hazard base class.
"""

import unittest
import datetime as dt
from pathlib import Path
import numpy as np
from scipy import sparse
from pathos.pools import ProcessPool as Pool

from climada import CONFIG
from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
import climada.util.dates_times as u_dt
from climada.util.constants import DEF_FREQ_UNIT, HAZ_TEMPLATE_XLS, HAZ_DEMO_FL
import climada.util.coordinates as u_coord

from climada.test import get_test_file
import climada.hazard.test as hazard_test


DATA_DIR :Path = CONFIG.hazard.test_data.dir()
"""
Directory for writing (and subsequent reading) of temporary files created during tests.
"""
HAZ_TEST_MAT :Path = Path(hazard_test.__file__).parent.joinpath('data', 'atl_prob_no_name.mat')
"""
Hazard test file from Git repository. Fraction is 1. Format: matlab.
"""
HAZ_TEST_TC :Path = get_test_file('test_tc_florida')
"""
Hazard test file from Data API: Hurricanes from 1851 to 2011 over Florida with 100 centroids.
Fraction is empty. Format: HDF5.
"""


def dummy_hazard():
    fraction = sparse.csr_matrix([[0.02, 0.03, 0.04],
                                  [0.01, 0.01, 0.01],
                                  [0.3, 0.1, 0.0],
                                  [0.3, 0.2, 0.0]])
    intensity = sparse.csr_matrix([[0.2, 0.3, 0.4],
                                   [0.1, 0.1, 0.01],
                                   [4.3, 2.1, 1.0],
                                   [5.3, 0.2, 0.0]])

    return Hazard(
        "TC",
        intensity=intensity,
        fraction=fraction,
        centroids=Centroids.from_lat_lon(
            np.array([1, 3, 5]), np.array([2, 4, 6])),
        event_id=np.array([1, 2, 3, 4]),
        event_name=['ev1', 'ev2', 'ev3', 'ev4'],
        date=np.array([1, 2, 3, 4]),
        orig=np.array([True, False, False, True]),
        frequency=np.array([0.1, 0.5, 0.5, 0.2]),
        frequency_unit='1/week',
        units='m/s',
        file_name="file1.mat",
        description="Description 1",
    )

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    def setUp(self):
        """Test fixure: Build a valid hazard"""
        centroids = Centroids.from_lat_lon(np.array([1, 3]), np.array([2, 3]))
        centroids.region_id = np.array([1, 2])
        self.hazard = Hazard(
            "TC",
            centroids=centroids,
            event_id=np.array([1, 2, 3]),
            event_name=['A', 'B', 'C'],
            frequency=np.array([1, 2, 3]),
            # events x centroids
            intensity=sparse.csr_matrix([[1, 2], [1, 2], [1, 2]]),
            fraction=sparse.csr_matrix([[1, 2], [1, 2], [1, 2]]),
        )

    def test_check_empty_fraction(self):
        """Test empty fraction"""
        self.hazard.fraction = sparse.csr_matrix(self.hazard.intensity.shape)
        self.hazard.check()

    def test_init_empty_fraction(self):
        """Test initializing a Hazard without fraction"""
        hazard = Hazard(
            "TC",
            centroids=self.hazard.centroids,
            event_id=self.hazard.event_id,
            event_name=self.hazard.event_name,
            frequency=self.hazard.frequency,
            intensity=self.hazard.intensity
        )
        hazard.check()
        np.testing.assert_array_equal(hazard.fraction.shape, hazard.intensity.shape)
        self.assertEqual(hazard.fraction.nnz, 0)  # No nonzero entries

    def test_check_wrongCentroids_fail(self):
        """Wrong hazard definition"""
        self.hazard.centroids.region_id = np.array([1, 2, 3, 4])
        with self.assertRaises(ValueError):
            self.hazard.check()

    def test_check_wrongFreq_fail(self):
        """Wrong hazard definition"""
        self.hazard.frequency = np.array([1, 2])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('Invalid Hazard.frequency size: 3 != 2.', str(cm.exception))

    def test_check_wrongInten_fail(self):
        """Wrong hazard definition"""
        self.hazard.intensity = sparse.csr_matrix([[1, 2], [1, 2]])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('Invalid Hazard.intensity row size: 3 != 2.', str(cm.exception))

    def test_check_wrongFrac_fail(self):
        """Wrong hazard definition"""
        self.hazard.fraction = sparse.csr_matrix([[1], [1], [1]])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('Invalid Hazard.fraction column size: 2 != 1.', str(cm.exception))

    def test_check_wrongEvName_fail(self):
        """Wrong hazard definition"""
        self.hazard.event_name = ['M']

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('Invalid Hazard.event_name size: 3 != 1.', str(cm.exception))

    def test_check_wrongId_fail(self):
        """Wrong hazard definition"""
        self.hazard.event_id = np.array([1, 2, 1])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('There are events with the same identifier.', str(cm.exception))

    def test_check_wrong_date_fail(self):
        """Wrong hazard definition"""
        self.hazard.date = np.array([1, 2])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('Invalid Hazard.date size: 3 != 2.', str(cm.exception))

    def test_check_wrong_orig_fail(self):
        """Wrong hazard definition"""
        self.hazard.orig = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn('Invalid Hazard.orig size: 3 != 4.', str(cm.exception))

    def test_event_name_to_id_pass(self):
        """Test event_name_to_id function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        self.assertEqual(haz.get_event_id('event001')[0], 1)
        self.assertEqual(haz.get_event_id('event084')[0], 84)

    def test_event_name_to_id_fail(self):
        """Test event_name_to_id function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        with self.assertRaises(ValueError) as cm:
            haz.get_event_id('1050')
        self.assertIn('No event with name: 1050', str(cm.exception))

    def test_event_id_to_name_pass(self):
        """Test event_id_to_name function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        self.assertEqual(haz.get_event_name(2), 'event002')
        self.assertEqual(haz.get_event_name(48), 'event048')

    def test_event_id_to_name_fail(self):
        """Test event_id_to_name function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        with self.assertRaises(ValueError) as cm:
            haz.get_event_name(1050)
        self.assertIn('No event with id: 1050', str(cm.exception))

    def test_get_date_strings_pass(self):
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        haz.event_name[5] = 'HAZEL'
        haz.event_name[10] = 'HAZEL'

        self.assertEqual(len(haz.get_event_date('HAZEL')), 2)
        self.assertEqual(haz.get_event_date('HAZEL')[0],
                         u_dt.date_to_str(haz.date[5]))
        self.assertEqual(haz.get_event_date('HAZEL')[1],
                         u_dt.date_to_str(haz.date[10]))

        self.assertEqual(haz.get_event_date(2)[0], u_dt.date_to_str(haz.date[1]))

        self.assertEqual(len(haz.get_event_date()), haz.date.size)
        self.assertEqual(haz.get_event_date()[560],
                         u_dt.date_to_str(haz.date[560]))

class TestRemoveDupl(unittest.TestCase):
    """Test remove_duplicates method."""

    def test_equal_same(self):
        """Append the same hazard and remove duplicates, obtain initial hazard."""
        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        haz2 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        haz1.append(haz2)
        haz1.remove_duplicates()
        haz1.check()
        self.assertEqual(haz1.event_name, haz2.event_name)
        self.assertTrue(np.array_equal(haz1.event_id, haz2.event_id))
        self.assertTrue(np.array_equal(haz1.frequency, haz2.frequency))
        self.assertEqual(haz1.frequency_unit, haz2.frequency_unit)
        self.assertTrue(np.array_equal(haz1.date, haz2.date))
        self.assertTrue(np.array_equal(haz1.orig, haz2.orig))
        self.assertTrue(np.array_equal(haz1.intensity.toarray(), haz2.intensity.toarray()))
        self.assertTrue(np.array_equal(haz1.fraction.toarray(), haz2.fraction.toarray()))
        self.assertTrue((haz1.intensity != haz2.intensity).nnz == 0)
        self.assertTrue((haz1.fraction != haz2.fraction).nnz == 0)
        self.assertEqual(haz1.units, haz2.units)
        self.assertEqual(haz1.tag.file_name, [haz2.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz2.tag.haz_type)
        self.assertEqual(haz1.tag.description, [haz2.tag.description, haz2.tag.description])

    def test_same_events_same(self):
        """Append hazard with same events and diff centroids. After removing
        duplicate events, initial events are obtained with 0 intensity and
        fraction in new appended centroids."""
        haz1 = dummy_hazard()
        centroids = Centroids.from_lat_lon(np.array([7, 9, 11]), np.array([8, 10, 12]))
        fraction = sparse.csr_matrix([[0.22, 0.32, 0.44],
                                      [0.11, 0.11, 0.11],
                                      [0.32, 0.11, 0.99],
                                      [0.32, 0.22, 0.88]])
        intensity = sparse.csr_matrix([[0.22, 3.33, 6.44],
                                       [1.11, 0.11, 1.11],
                                       [8.33, 4.11, 4.4],
                                       [9.33, 9.22, 1.77]])
        haz2 = Hazard(
            "TC",
            centroids=centroids,
            event_id=haz1.event_id,
            event_name=haz1.event_name,
            frequency=haz1.frequency,
            frequency_unit = "1/week",
            date = haz1.date,
            fraction=fraction,
            intensity=intensity,
            units="m/s",
            file_name="file2.mat",
            description="Description 2"
        )

        haz1.append(haz2)
        haz1.remove_duplicates()
        haz1.check()

        # expected values
        haz_res = dummy_hazard()
        haz_res.intensity = sparse.hstack(
            [haz_res.intensity, sparse.csr_matrix((haz_res.intensity.shape[0], 3))], format='csr')
        haz_res.fraction = sparse.hstack(
            [haz_res.fraction, sparse.csr_matrix((haz_res.fraction.shape[0], 3))], format='csr')
        self.assertTrue(np.array_equal(haz_res.intensity.toarray(),
                                       haz1.intensity.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(np.array_equal(haz_res.fraction.toarray(),
                                       haz1.fraction.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        self.assertEqual(haz1.event_name, haz_res.event_name)
        self.assertTrue(np.array_equal(haz1.date, haz_res.date))
        self.assertTrue(np.array_equal(haz1.orig, haz_res.orig))
        self.assertTrue(np.array_equal(haz1.event_id,
                                       haz_res.event_id))
        self.assertTrue(np.array_equal(haz1.frequency, haz_res.frequency))
        self.assertEqual(haz1.frequency_unit, haz_res.frequency_unit)
        self.assertEqual(haz_res.units, haz1.units)

        self.assertEqual(haz1.tag.file_name,
                         [haz_res.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz_res.tag.haz_type)
        self.assertEqual(haz1.tag.description,
                         [haz_res.tag.description, haz2.tag.description])

class TestSelect(unittest.TestCase):
    """Test select method."""

    def test_select_event_name(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(event_names=['ev4', 'ev1'])

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.2, 0.1])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(sel_haz.fraction.toarray(),
                                       np.array([[0.3, 0.2, 0.0],
                                                 [0.02, 0.03, 0.04]])))
        self.assertTrue(np.array_equal(sel_haz.intensity.toarray(),
                                       np.array([[5.3, 0.2, 0.0],
                                                 [0.2, 0.3, 0.4]])))
        self.assertEqual(sel_haz.event_name, ['ev4', 'ev1'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_event_id(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(event_id=[4, 1])

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.2, 0.1])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(sel_haz.fraction.toarray(),
                                       np.array([[0.3, 0.2, 0.0],
                                                 [0.02, 0.03, 0.04]])))
        self.assertTrue(np.array_equal(sel_haz.intensity.toarray(),
                                       np.array([[5.3, 0.2, 0.0],
                                                 [0.2, 0.3, 0.4]])))
        self.assertEqual(sel_haz.event_name, ['ev4', 'ev1'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_orig_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(orig=True)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([1, 4])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([1, 4])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.1, 0.2])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(
            sel_haz.fraction.toarray(), np.array([[0.02, 0.03, 0.04], [0.3, 0.2, 0.0]])))
        self.assertTrue(np.array_equal(
            sel_haz.intensity.toarray(), np.array([[0.2, 0.3, 0.4], [5.3, 0.2, 0.0]])))
        self.assertEqual(sel_haz.event_name, ['ev1', 'ev4'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_syn_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(orig=False)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(
            sel_haz.fraction.toarray(), np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0]])))
        self.assertTrue(np.array_equal(
            sel_haz.intensity.toarray(), np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0]])))
        self.assertEqual(sel_haz.event_name, ['ev2', 'ev3'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=(2, 4))

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3, 4])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3, 4])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5, 0.2])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(
            sel_haz.fraction.toarray(), np.array([[0.01, 0.01, 0.01],
                                                  [0.3, 0.1, 0.0],
                                                  [0.3, 0.2, 0.0]])))
        self.assertTrue(np.array_equal(
            sel_haz.intensity.toarray(), np.array([[0.1, 0.1, 0.01],
                                                   [4.3, 2.1, 1.0],
                                                   [5.3, 0.2, 0.0]])))
        self.assertEqual(sel_haz.event_name, ['ev2', 'ev3', 'ev4'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_str_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=('0001-01-02', '0001-01-03'))

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(
            sel_haz.fraction.toarray(), np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0]])))
        self.assertTrue(np.array_equal(
            sel_haz.intensity.toarray(), np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0]])))
        self.assertEqual(sel_haz.event_name, ['ev2', 'ev3'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_and_orig_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=(2, 4), orig=False)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(
            sel_haz.fraction.toarray(), np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0]])))
        self.assertTrue(np.array_equal(
            sel_haz.intensity.toarray(), np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0]])))
        self.assertEqual(sel_haz.event_name, ['ev2', 'ev3'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_wrong_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=(6, 8), orig=False)
        self.assertEqual(sel_haz, None)

    def test_select_date_invalid_pass(self):
        """Test select with invalid date values"""
        haz = dummy_hazard()

        # lists and numpy arrays should work just like tuples
        sel_haz = haz.select(date=['0001-01-02', '0001-01-03'])
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        sel_haz = haz.select(date=np.array([2, 4]))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3, 4])))

        # iterables of length not exactly 2 are invalid
        with self.assertRaises(ValueError) as context:
            haz.select(date=(3,))
        self.assertTrue("not enough values to unpack" in str(context.exception))
        with self.assertRaises(ValueError) as context:
            haz.select(date=(1, 2, 3))
        self.assertTrue("too many values to unpack" in str(context.exception))

        # only strings and numbers are valid
        with self.assertRaises(TypeError) as context:
            haz.select(date=({}, {}))

    def test_select_reg_id_pass(self):
        """Test select region of centroids."""
        haz = dummy_hazard()
        haz.centroids.region_id = np.array([5, 7, 9])
        sel_haz = haz.select(date=(2, 4), orig=False, reg_id=9)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord.squeeze(),
                                       haz.centroids.coord[2, :]))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(sel_haz.fraction.toarray(), np.array([[0.01], [0.0]])))
        self.assertTrue(np.array_equal(sel_haz.intensity.toarray(), np.array([[0.01], [1.0]])))
        self.assertEqual(sel_haz.event_name, ['ev2', 'ev3'])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_tight_pass(self):
        """Test select tight box around hazard"""

        #intensity select
        haz = dummy_hazard()
        haz.intensity[:, -1] = 0.0
        sel_haz = haz.select_tight()

        self.assertTrue(np.array_equal(sel_haz.centroids.coord.squeeze(),
                                       haz.centroids.coord[:-1, :]))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, haz.event_id))
        self.assertTrue(np.array_equal(sel_haz.date, haz.date))
        self.assertTrue(np.array_equal(sel_haz.orig, haz.orig))
        self.assertTrue(np.array_equal(sel_haz.frequency, haz.frequency))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(sel_haz.fraction.toarray(),
                                       haz.fraction[:,:-1].toarray()))
        self.assertTrue(np.array_equal(sel_haz.intensity.toarray(),
                                       haz.intensity[:,:-1].toarray()))
        self.assertEqual(sel_haz.event_name, haz.event_name)
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

        #fraction select
        haz = dummy_hazard()
        haz.fraction[:, -1] = 0.0

        sel_haz = haz.select_tight(val='fraction')

        self.assertTrue(np.array_equal(sel_haz.centroids.coord.squeeze(),
                                       haz.centroids.coord[:-1, :]))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, haz.event_id))
        self.assertTrue(np.array_equal(sel_haz.date, haz.date))
        self.assertTrue(np.array_equal(sel_haz.orig, haz.orig))
        self.assertTrue(np.array_equal(sel_haz.frequency, haz.frequency))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(sel_haz.fraction.toarray(),
                                       haz.fraction[:,:-1].toarray()))
        self.assertTrue(np.array_equal(sel_haz.intensity.toarray(),
                                       haz.intensity[:,:-1].toarray()))
        self.assertEqual(sel_haz.event_name, haz.event_name)
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)


        haz = dummy_hazard()
        haz.intensity[:, -1] = 0.0

        # small buffer: zero field is discarded
        sel_haz = haz.select_tight(buffer=0.1)
        self.assertTrue(np.array_equal(sel_haz.centroids.coord.squeeze(),
                                       haz.centroids.coord[:-1, :]))
        # large buffer: zero field is retained
        sel_haz = haz.select_tight(buffer=10)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord.squeeze(),
                                       haz.centroids.coord))
        self.assertEqual(sel_haz.tag, haz.tag)
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, haz.event_id))
        self.assertTrue(np.array_equal(sel_haz.date, haz.date))
        self.assertTrue(np.array_equal(sel_haz.orig, haz.orig))
        self.assertTrue(np.array_equal(sel_haz.frequency, haz.frequency))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(np.array_equal(sel_haz.fraction.toarray(),
                                       haz.fraction.toarray()))
        self.assertTrue(np.array_equal(sel_haz.intensity.toarray(),
                                       haz.intensity.toarray()))
        self.assertEqual(sel_haz.event_name, haz.event_name)
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)


class TestAppend(unittest.TestCase):
    """Test append method."""

    def test_append_empty_fill(self):
        """Append an empty. Obtain initial hazard."""
        def _check_hazard(hazard):
            # expected values
            haz1_orig = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
            self.assertEqual(hazard.event_name, haz1_orig.event_name)
            self.assertTrue(np.array_equal(hazard.event_id, haz1_orig.event_id))
            self.assertTrue(np.array_equal(hazard.date, haz1_orig.date))
            self.assertTrue(np.array_equal(hazard.orig, haz1_orig.orig))
            self.assertTrue(np.array_equal(hazard.frequency, haz1_orig.frequency))
            self.assertEqual(hazard.frequency_unit, haz1_orig.frequency_unit)
            self.assertTrue((hazard.intensity != haz1_orig.intensity).nnz == 0)
            self.assertTrue((hazard.fraction != haz1_orig.fraction).nnz == 0)
            self.assertEqual(hazard.units, haz1_orig.units)
            self.assertEqual(hazard.tag.file_name, haz1_orig.tag.file_name)
            self.assertEqual(hazard.tag.haz_type, haz1_orig.tag.haz_type)
            self.assertEqual(hazard.tag.description, haz1_orig.tag.description)

        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        haz2 = Hazard('TC')
        haz2.centroids.geometry.crs = 'epsg:4326'
        haz1.append(haz2)
        haz1.check()
        _check_hazard(haz1)

        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        haz2 = Hazard('TC')
        haz2.centroids.geometry.crs = 'epsg:4326'
        haz2.append(haz1)
        haz2.check()
        _check_hazard(haz2)

    def test_same_centroids_extend(self):
        """Append hazard with same centroids, different events."""
        haz1 = dummy_hazard()
        fraction = sparse.csr_matrix([[0.2, 0.3, 0.4],
                                      [0.1, 0.1, 0.1],
                                      [0.3, 0.1, 0.9],
                                      [0.3, 0.2, 0.8]])
        intensity = sparse.csr_matrix([[0.2, 3.3, 6.4],
                                       [1.1, 0.1, 1.01],
                                       [8.3, 4.1, 4.0],
                                       [9.3, 9.2, 1.7]])
        haz2 = Hazard('TC',
                      file_name='file2.mat',
                      description='Description 2',
                      centroids=haz1.centroids,
                      event_id=np.array([5, 6, 7, 8]),
                      event_name=['ev5', 'ev6', 'ev7', 'ev8'],
                      frequency=np.array([0.9, 0.75, 0.75, 0.22]),
                      frequency_unit='1/week',
                      units="m/s",
                      fraction=fraction,
                      intensity=intensity,
                      )

        haz1.append(haz2)
        haz1.check()

        # expected values
        haz1_orig = dummy_hazard()
        exp_inten = np.zeros((8, 3))
        exp_inten[0:4, 0:3] = haz1_orig.intensity.toarray()
        exp_inten[4:8, 0:3] = haz2.intensity.toarray()
        exp_frac = np.zeros((8, 3))
        exp_frac[0:4, 0:3] = haz1_orig.fraction.toarray()
        exp_frac[4:8, 0:3] = haz2.fraction.toarray()

        self.assertEqual(haz1.event_id.size, 8)
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        for i_ev in range(haz1.event_id.size):
            self.assertTrue(any((haz1.intensity[i_ev].toarray() == exp_inten).all(1)))
            self.assertTrue(any((haz1.fraction[i_ev].toarray() == exp_frac).all(1)))
            self.assertTrue(haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name)
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id))
            self.assertTrue(haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency))

        self.assertEqual(haz1.centroids.size, 3)
        self.assertTrue(np.array_equal(haz1.centroids.coord, haz2.centroids.coord))
        self.assertEqual(haz1.tag.file_name,
                         [haz1_orig.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description,
                         [haz1_orig.tag.description, haz2.tag.description])

    def test_incompatible_type_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = dummy_hazard()
        haz2 = dummy_hazard()
        haz2.tag.haz_type = 'WS'
        haz2.tag.file_name = 'file2.mat'
        haz2.tag.description = 'Description 2'
        with self.assertRaises(ValueError) as cm:
            haz1.append(haz2)

    def test_incompatible_units_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = dummy_hazard()
        haz2 = dummy_hazard()
        haz2.units = 'km/h'
        with self.assertRaises(ValueError) as cm:
            haz1.append(haz2)

    def test_incompatible_freq_units_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = dummy_hazard()
        haz2 = dummy_hazard()
        haz2.frequency_unit = '1/month'
        with self.assertRaises(ValueError) as cm:
            haz1.append(haz2)

    def test_all_different_extend(self):
        """Append totally different hazard."""
        haz1 = dummy_hazard()

        fraction = sparse.csr_matrix([[0.2, 0.3, 0.4],
                                      [0.1, 0.1, 0.1],
                                      [0.3, 0.1, 0.9],
                                      [0.3, 0.2, 0.8]])
        intensity = sparse.csr_matrix([[0.2, 3.3, 6.4],
                                       [1.1, 0.1, 1.01],
                                       [8.3, 4.1, 4.0],
                                       [9.3, 9.2, 1.7]])
        haz2 = Hazard('TC',
                      date=np.ones((4,)),
                      orig=np.ones((4,)),
                      file_name='file2.mat',
                      description='Description 2',
                      centroids=Centroids.from_lat_lon(
                          np.array([7, 9, 11]), np.array([8, 10, 12])),
                      event_id=np.array([5, 6, 7, 8]),
                      event_name=['ev5', 'ev6', 'ev7', 'ev8'],
                      frequency=np.array([0.9, 0.75, 0.75, 0.22]),
                      frequency_unit='1/week',
                      units='m/s',
                      intensity=intensity,
                      fraction=fraction)

        haz1.append(haz2)
        haz1.check()

        # expected values
        haz1_orig = dummy_hazard()
        exp_inten = np.zeros((8, 6))
        exp_inten[0:4, 0:3] = haz1_orig.intensity.toarray()
        exp_inten[4:8, 3:6] = haz2.intensity.toarray()
        exp_frac = np.zeros((8, 6))
        exp_frac[0:4, 0:3] = haz1_orig.fraction.toarray()
        exp_frac[4:8, 3:6] = haz2.fraction.toarray()
        self.assertEqual(haz1.event_id.size, 8)
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        for i_ev in range(haz1.event_id.size):
            self.assertTrue(any((haz1.intensity[i_ev].toarray() == exp_inten).all(1)))
            self.assertTrue(any((haz1.fraction[i_ev].toarray() == exp_frac).all(1)))
            self.assertTrue(haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name)
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id))
            self.assertTrue(haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency))

        self.assertEqual(haz1.centroids.size, 6)
        self.assertEqual(haz1_orig.units, haz1.units)
        self.assertEqual(haz1_orig.frequency_unit, haz1.frequency_unit)
        self.assertEqual(haz1.tag.file_name,
                         [haz1_orig.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_orig.tag.haz_type)
        self.assertEqual(haz1.tag.description,
                         [haz1_orig.tag.description, haz2.tag.description])

    def test_same_events_append(self):
        """Append hazard with same events (and diff centroids).
        Events are appended with all new centroids columns."""
        haz1 = dummy_hazard()
        fraction = sparse.csr_matrix([[0.22, 0.32, 0.44],
                                      [0.11, 0.11, 0.11],
                                      [0.32, 0.11, 0.99],
                                      [0.32, 0.22, 0.88]])
        intensity = sparse.csr_matrix([[0.22, 3.33, 6.44],
                                       [1.11, 0.11, 1.11],
                                       [8.33, 4.11, 4.4],
                                       [9.33, 9.22, 1.77]])
        haz2 = Hazard('TC',
                      file_name='file2.mat',
                      description='Description 2',
                      centroids=Centroids.from_lat_lon(
                          np.array([7, 9, 11]), np.array([8, 10, 12])),
                      event_id=haz1.event_id,
                      event_name=haz1.event_name.copy(),
                      frequency=haz1.frequency,
                      frequency_unit=haz1.frequency_unit,
                      date=haz1.date,
                      units='m/s',
                      fraction=fraction,
                      intensity=intensity)

        haz1.append(haz2)

        # expected values
        haz1_ori = dummy_hazard()
        res_inten = np.zeros((8, 6))
        res_inten[0:4, 0:3] = haz1_ori.intensity.toarray()
        res_inten[4:, 3:] = haz2.intensity.toarray()

        res_frac = np.zeros((8, 6))
        res_frac[0:4, 0:3] = haz1_ori.fraction.toarray()
        res_frac[4:, 3:] = haz2.fraction.toarray()

        self.assertTrue(np.array_equal(res_inten,
                                       haz1.intensity.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(np.array_equal(res_frac,
                                       haz1.fraction.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        self.assertEqual(haz1.event_name,
                         haz1_ori.event_name + haz2.event_name)
        self.assertTrue(np.array_equal(haz1.date,
                                       np.append(haz1_ori.date, haz2.date)))
        self.assertTrue(np.array_equal(haz1.orig,
                                       np.append(haz1_ori.orig, haz2.orig)))
        self.assertTrue(np.array_equal(haz1.event_id, np.arange(1, 9)))
        self.assertTrue(np.array_equal(haz1.frequency,
                                       np.append(haz1_ori.frequency, haz2.frequency)))
        self.assertEqual(haz1_ori.frequency_unit, haz1.frequency_unit)
        self.assertEqual(haz1_ori.units, haz1.units)

        self.assertEqual(haz1.tag.file_name,
                         [haz1_ori.tag.file_name, haz2.tag.file_name])
        self.assertEqual(haz1.tag.haz_type, haz1_ori.tag.haz_type)
        self.assertEqual(haz1.tag.description,
                         [haz1_ori.tag.description, haz2.tag.description])

    def test_concat_pass(self):
        """Test concatenate function."""

        haz_1 = Hazard("TC",
                       file_name='file1.mat',
                       description='Description 1',
                       centroids=Centroids.from_lat_lon(
                           np.array([1, 3, 5]), np.array([2, 4, 6])),
                       event_id=np.array([1]),
                       event_name=['ev1'],
                       date=np.array([1]),
                       orig=np.array([True]),
                       frequency=np.array([1.0]),
                       frequency_unit='1/week',
                       fraction=sparse.csr_matrix([[0.02, 0.03, 0.04]]),
                       intensity=sparse.csr_matrix([[0.2, 0.3, 0.4]]),
                       units='m/s',)

        haz_2 = Hazard("TC",
                       file_name='file2.mat',
                       description='Description 2',
                       centroids=Centroids.from_lat_lon(
                           np.array([1, 3, 5]), np.array([2, 4, 6])),
                       event_id=np.array([1]),
                       event_name=['ev2'],
                       date=np.array([2]),
                       orig=np.array([False]),
                       frequency=np.array([1.0]),
                       frequency_unit='1/week',
                       fraction=sparse.csr_matrix([[1.02, 1.03, 1.04]]),
                       intensity=sparse.csr_matrix([[1.2, 1.3, 1.4]]),
                       units='m/s',)

        haz = Hazard.concat([haz_1, haz_2])

        hres_frac = sparse.csr_matrix([[0.02, 0.03, 0.04],
                                       [1.02, 1.03, 1.04]])
        hres_inten = sparse.csr_matrix([[0.2, 0.3, 0.4],
                                        [1.2, 1.3, 1.4]])

        self.assertIsInstance(haz, Hazard)
        self.assertTrue(sparse.isspmatrix_csr(haz.intensity))
        self.assertTrue(np.array_equal(haz.intensity.toarray(), hres_inten.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz.fraction))
        self.assertTrue(np.array_equal(haz.fraction.toarray(), hres_frac.toarray()))
        self.assertEqual(haz.units, haz_2.units)
        self.assertTrue(np.array_equal(haz.frequency, np.array([1.0, 1.0])))
        self.assertEqual(haz.frequency_unit, haz_2.frequency_unit)
        self.assertTrue(np.array_equal(haz.orig, np.array([True, False])))
        self.assertTrue(np.array_equal(haz.date, np.array([1, 2])))
        self.assertTrue(np.array_equal(haz.event_id, np.array([1, 2])))
        self.assertEqual(haz.event_name, ['ev1', 'ev2'])
        self.assertTrue(np.array_equal(haz.centroids.coord, haz_1.centroids.coord))
        self.assertTrue(np.array_equal(haz.centroids.coord, haz_2.centroids.coord))
        self.assertEqual(haz.tag.file_name, ['file1.mat', 'file2.mat'])
        self.assertEqual(haz.tag.description, ['Description 1', 'Description 2'])

    def test_append_new_var_pass(self):
        """New variable appears if hazard to append is empty."""
        haz = dummy_hazard()
        haz.frequency_unit = haz.get_default('frequency_unit')
        haz.new_var = np.ones(haz.size)

        app_haz = Hazard('TC')
        app_haz.append(haz)
        self.assertIn('new_var', app_haz.__dict__)

    def test_append_raise_type_error(self):
        """Raise error if hazards of different class"""
        haz1 = Hazard('TC', units='m/s')
        from climada.hazard import TropCyclone
        haz2 = TropCyclone()
        with self.assertRaises(TypeError):
            haz1.append(haz2)

    def test_concat_raise_value_error(self):
        """Raise error if hazards with different units of type"""
        haz1 = Hazard('TC', units='m/s')
        haz3 = Hazard('EQ')
        with self.assertRaises(ValueError):
             Hazard.concat([haz1, haz3])

        haz4 = Hazard('TC', units='cm')
        with self.assertRaises(ValueError):
             Hazard.concat([haz1, haz4])

    def test_change_centroids(self):
        """Set new centroids for hazard"""
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        haz_1 = Hazard('TC',
                       file_name='file1.mat',
                       description='Description 1',
                       centroids=cent1,
                       event_id=np.array([1]),
                       event_name=['ev1'],
                       date=np.array([1]),
                       orig=np.array([True]),
                       frequency=np.array([1.0]),
                       frequency_unit='1/week',
                       fraction=sparse.csr_matrix([[0.02, 0.03]]),
                       intensity=sparse.csr_matrix([[0.2, 0.3]]),
                       units='m/s',)

        lat2, lon2 = np.array([0, 1, 3]), np.array([0, -1, 3])
        on_land2 = np.array([True, True, False])
        cent2 = Centroids(lat=lat2, lon=lon2, on_land=on_land2)

        haz_2 = haz_1.change_centroids(cent2)

        self.assertTrue(np.array_equal(haz_2.intensity.toarray(),
                               np.array([[0.2, 0.3, 0.]])))
        self.assertTrue(np.array_equal(haz_2.fraction.toarray(),
                               np.array([[0.02, 0.03, 0.]])))
        self.assertTrue(np.array_equal(haz_2.event_id, np.array([1])))
        self.assertTrue(np.array_equal(haz_2.event_name, ['ev1']))
        self.assertTrue(np.array_equal(haz_2.orig, [True]))
        self.assertEqual(haz_2.tag.description, 'Description 1')

        """Test error for projection"""
        lat3, lon3 = np.array([0.5, 3]), np.array([-0.5, 3])
        on_land3 = np.array([True, True, False])
        cent3 = Centroids(lat=lat3, lon=lon3, on_land=on_land3)

        with self.assertRaises(ValueError) as cm:
            haz_1.change_centroids(cent3, threshold=100)
        self.assertIn('two hazard centroids are mapped to the same centroids', str(cm.exception))


    def test_change_centroids_raster(self):
        """Set new centroids for hazard"""
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        haz_1 = Hazard('TC',
                       file_name='file1.mat',
                       description='Description 1',
                       centroids=cent1,
                       event_id=np.array([1]),
                       event_name=['ev1'],
                       date=np.array([1]),
                       orig=np.array([True]),
                       frequency=np.array([1.0]),
                       frequency_unit='1/week',
                       fraction=sparse.csr_matrix([[0.02, 0.03]]),
                       intensity=sparse.csr_matrix([[0.2, 0.3]]),
                       units='m/s',)


        """Test with raster centroids"""
        cent4 = Centroids.from_pnt_bounds(points_bounds=(-1, 0, 0, 1), res=1)
        cent4.lat, cent1.lon = np.array([0, 1]), np.array([0, -1])
        cent4.on_land = np.array([True, True])

        haz_4 = haz_1.change_centroids(cent4)

        self.assertTrue(np.array_equal(haz_4.intensity.toarray(),
                               np.array([[0.3, 0.0, 0.0, 0.2]])))
        self.assertTrue(np.array_equal(haz_4.fraction.toarray(),
                               np.array([[0.03, 0.0, 0.0, 0.02]])))
        self.assertTrue(np.array_equal(haz_4.event_id, np.array([1])))
        self.assertTrue(np.array_equal(haz_4.event_name, ['ev1']))
        self.assertTrue(np.array_equal(haz_4.orig, [True]))
        self.assertEqual(haz_4.tag.description, 'Description 1')


class TestStats(unittest.TestCase):
    """Test return period statistics"""

    def test_degenerate_pass(self):
        """Test degenerate call."""
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        return_period = np.array([25, 50, 100, 250])
        haz.intensity = sparse.csr.csr_matrix(np.zeros(haz.intensity.shape))
        inten_stats = haz.local_exceedance_inten(return_period)
        self.assertTrue(np.array_equal(inten_stats, np.zeros((4, 100))))

    def test_ref_all_pass(self):
        """Compare against reference."""
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        return_period = np.array([25, 50, 100, 250])
        inten_stats = haz.local_exceedance_inten(return_period)

        self.assertAlmostEqual(inten_stats[0][0], 55.424015590131290)
        self.assertAlmostEqual(inten_stats[1][0], 67.221687644669998)
        self.assertAlmostEqual(inten_stats[2][0], 79.019359699208721)
        self.assertAlmostEqual(inten_stats[3][0], 94.615033842370963)

        self.assertAlmostEqual(inten_stats[1][66], 70.608592953031405)
        self.assertAlmostEqual(inten_stats[3][33], 88.510983305123631)
        self.assertAlmostEqual(inten_stats[2][99], 79.717518054203623)

class TestYearset(unittest.TestCase):
    """Test return period statistics"""

    def test_ref_pass(self):
        """Test against reference."""
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        orig_year_set = haz.calc_year_set()

        self.assertTrue(np.array_equal(np.array(list(orig_year_set.keys())),
                                       np.arange(1851, 2012)))
        self.assertTrue(np.array_equal(orig_year_set[1851],
                                       np.array([1, 11, 21, 31])))
        self.assertTrue(np.array_equal(orig_year_set[1958],
                                       np.array([8421, 8431, 8441, 8451, 8461, 8471, 8481,
                                                 8491, 8501, 8511])))
        self.assertTrue(np.array_equal(orig_year_set[1986],
                                       np.array([11101, 11111, 11121, 11131, 11141, 11151])))
        self.assertTrue(np.array_equal(orig_year_set[1997],
                                       np.array([12221, 12231, 12241, 12251, 12261, 12271,
                                                 12281, 12291])))
        self.assertTrue(np.array_equal(orig_year_set[2006],
                                       np.array([13571, 13581, 13591, 13601, 13611, 13621,
                                                 13631, 13641, 13651, 13661])))
        self.assertTrue(np.array_equal(orig_year_set[2010],
                                       np.array([14071, 14081, 14091, 14101, 14111, 14121,
                                                 14131, 14141, 14151, 14161, 14171, 14181,
                                                 14191, 14201, 14211, 14221, 14231, 14241,
                                                 14251])))

class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the Hazard class"""

    def test_hazard_pass(self):
        """Read an hazard excel file correctly."""

        # Read demo excel file
        description = 'One single file.'
        hazard = Hazard.from_excel(HAZ_TEMPLATE_XLS, description=description, haz_type='TC')

        # Check results
        n_events = 100
        n_centroids = 45

        self.assertEqual(hazard.units, '')

        self.assertEqual(hazard.centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(hazard.centroids.coord[0][0], -25.95)
        self.assertEqual(hazard.centroids.coord[0][1], 32.57)
        self.assertEqual(hazard.centroids.coord[n_centroids - 1][0], -24.7)
        self.assertEqual(hazard.centroids.coord[n_centroids - 1][1], 33.88)

        self.assertEqual(len(hazard.event_name), 100)
        self.assertEqual(hazard.event_name[12], 'event013')

        self.assertEqual(hazard.event_id.dtype, int)
        self.assertEqual(hazard.event_id.shape, (n_events,))
        self.assertEqual(hazard.event_id[0], 1)
        self.assertEqual(hazard.event_id[n_events - 1], 100)

        self.assertEqual(hazard.date.dtype, int)
        self.assertEqual(hazard.date.shape, (n_events,))
        self.assertEqual(hazard.date[0], 675874)
        self.assertEqual(hazard.date[n_events - 1], 676329)

        self.assertEqual(hazard.event_name[0], 'event001')
        self.assertEqual(hazard.event_name[50], 'event051')
        self.assertEqual(hazard.event_name[-1], 'event100')

        self.assertEqual(hazard.frequency.dtype, float)
        self.assertEqual(hazard.frequency.shape, (n_events,))
        self.assertEqual(hazard.frequency[0], 0.01)
        self.assertEqual(hazard.frequency[n_events - 2], 0.001)

        self.assertEqual(hazard.frequency_unit, DEF_FREQ_UNIT)

        self.assertEqual(hazard.intensity.dtype, float)
        self.assertEqual(hazard.intensity.shape, (n_events, n_centroids))

        self.assertEqual(hazard.fraction.dtype, float)
        self.assertEqual(hazard.fraction.shape, (n_events, n_centroids))
        self.assertEqual(hazard.fraction[0, 0], 1)
        self.assertEqual(hazard.fraction[10, 19], 1)
        self.assertEqual(hazard.fraction[n_events - 1, n_centroids - 1], 1)

        self.assertTrue(np.all(hazard.orig))

        # tag hazard
        self.assertEqual(hazard.tag.file_name, HAZ_TEMPLATE_XLS)
        self.assertEqual(hazard.tag.description, description)
        self.assertEqual(hazard.tag.haz_type, 'TC')

class TestReaderMat(unittest.TestCase):
    """Test reader functionality of the ExposuresExcel class"""

    def test_hazard_pass(self):
        """Read a hazard mat file correctly."""
        # Read demo excel file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)

        # Check results
        n_events = 14450
        n_centroids = 100

        self.assertEqual(hazard.units, 'm/s')

        self.assertEqual(hazard.centroids.coord.shape, (n_centroids, 2))

        self.assertEqual(hazard.event_id.dtype, int)
        self.assertEqual(hazard.event_id.shape, (n_events,))

        self.assertEqual(hazard.frequency.dtype, float)
        self.assertEqual(hazard.frequency.shape, (n_events,))

        self.assertEqual(hazard.frequency_unit, DEF_FREQ_UNIT)

        self.assertEqual(hazard.intensity.dtype, float)
        self.assertEqual(hazard.intensity.shape, (n_events, n_centroids))
        self.assertEqual(hazard.intensity[12, 46], 12.071393519949979)
        self.assertEqual(hazard.intensity[13676, 49], 17.228323602220616)

        self.assertEqual(hazard.fraction.dtype, float)
        self.assertEqual(hazard.fraction.shape, (n_events, n_centroids))
        self.assertEqual(hazard.fraction[8454, 98], 1)
        self.assertEqual(hazard.fraction[85, 54], 0)

        self.assertEqual(len(hazard.event_name), n_events)
        self.assertEqual(hazard.event_name[124], 125)

        self.assertEqual(len(hazard.date), n_events)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[0]).year, 1851)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[0]).month, 6)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[0]).day, 25)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[78]).year, 1852)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[78]).month, 9)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[78]).day, 22)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[-1]).year, 2011)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[-1]).month, 11)
        self.assertEqual(dt.datetime.fromordinal(hazard.date[-1]).day, 6)

        self.assertTrue(hazard.orig[0])
        self.assertTrue(hazard.orig[11580])
        self.assertTrue(hazard.orig[4940])
        self.assertFalse(hazard.orig[3551])
        self.assertFalse(hazard.orig[10651])
        self.assertFalse(hazard.orig[4818])

        # tag hazard
        self.assertEqual(hazard.tag.file_name, str(HAZ_TEST_MAT))
        self.assertEqual(hazard.tag.description,
                         ' TC hazard event set, generated 14-Nov-2017 10:09:05')
        self.assertEqual(hazard.tag.haz_type, 'TC')

class TestHDF5(unittest.TestCase):
    """Test reader functionality of the ExposuresExcel class"""

    def test_write_read_pass(self):
        """Read a hazard mat file correctly."""
        file_name = str(DATA_DIR.joinpath('test_haz.h5'))

        # Read demo matlab file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        hazard.event_name = list(map(str, hazard.event_name))
        for todense_flag in [False, True]:
            if todense_flag:
                hazard.write_hdf5(file_name, todense=todense_flag)
            else:
                hazard.write_hdf5(file_name)

            haz_read = Hazard.from_hdf5(file_name)

            self.assertEqual(str(hazard.tag.file_name), haz_read.tag.file_name)
            self.assertIsInstance(haz_read.tag.file_name, str)
            self.assertEqual(hazard.tag.haz_type, haz_read.tag.haz_type)
            self.assertIsInstance(haz_read.tag.haz_type, str)
            self.assertEqual(hazard.tag.description, haz_read.tag.description)
            self.assertIsInstance(haz_read.tag.description, str)
            self.assertEqual(hazard.units, haz_read.units)
            self.assertIsInstance(haz_read.units, str)
            self.assertTrue(np.array_equal(hazard.centroids.coord, haz_read.centroids.coord))
            self.assertTrue(u_coord.equal_crs(hazard.centroids.crs, haz_read.centroids.crs))
            self.assertTrue(np.array_equal(hazard.event_id, haz_read.event_id))
            self.assertTrue(np.array_equal(hazard.frequency, haz_read.frequency))
            self.assertEqual(hazard.frequency_unit, haz_read.frequency_unit)
            self.assertIsInstance(haz_read.frequency_unit, str)
            self.assertTrue(np.array_equal(hazard.event_name, haz_read.event_name))
            self.assertIsInstance(haz_read.event_name, list)
            self.assertIsInstance(haz_read.event_name[0], str)
            self.assertTrue(np.array_equal(hazard.date, haz_read.date))
            self.assertTrue(np.array_equal(hazard.orig, haz_read.orig))
            self.assertTrue(np.array_equal(hazard.intensity.toarray(),
                                           haz_read.intensity.toarray()))
            self.assertIsInstance(haz_read.intensity, sparse.csr_matrix)
            self.assertTrue(np.array_equal(hazard.fraction.toarray(), haz_read.fraction.toarray()))
            self.assertIsInstance(haz_read.fraction, sparse.csr_matrix)

    def test_write_read_unsupported_type(self):
        """Check if the write command correctly handles unsupported types"""
        file_name = str(DATA_DIR.joinpath('test_unsupported.h5'))

        # Define an unsupported type
        class CustomID:
            id = 1

        # Create a hazard with unsupported type as attribute
        hazard = dummy_hazard()
        hazard.event_id = CustomID()

        # Write the hazard and check the logs for the correct warning
        with self.assertLogs(logger="climada.hazard.base", level="WARN") as cm:
            hazard.write_hdf5(file_name)
        self.assertIn("write_hdf5: the class member event_id is skipped", cm.output[0])

        # Load the file again and compare to previous instance
        hazard_read = Hazard.from_hdf5(file_name)
        self.assertEqual(hazard.tag.description, hazard_read.tag.description)
        self.assertTrue(np.array_equal(hazard.date, hazard_read.date))
        self.assertTrue(np.array_equal(hazard_read.event_id, np.array([])))  # Empty array


class TestCentroids(unittest.TestCase):
    """Test return period statistics"""

    def test_reproject_raster_pass(self):
        """Test reproject_raster reference."""
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL])
        haz_fl.check()

        haz_fl.reproject_raster(dst_crs='epsg:2202')

        self.assertEqual(haz_fl.intensity.shape, (1, 1046408))
        self.assertIsInstance(haz_fl.intensity, sparse.csr_matrix)
        self.assertIsInstance(haz_fl.fraction, sparse.csr_matrix)
        self.assertEqual(haz_fl.fraction.shape, (1, 1046408))
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.meta['crs'], 'epsg:2202'))
        self.assertEqual(haz_fl.centroids.meta['width'], 968)
        self.assertEqual(haz_fl.centroids.meta['height'], 1081)
        self.assertEqual(haz_fl.fraction.min(), 0)
        self.assertEqual(haz_fl.fraction.max(), 1)
        self.assertEqual(haz_fl.intensity.min(), -9999)
        self.assertTrue(haz_fl.intensity.max() < 4.7)

    def test_raster_to_vector_pass(self):
        """Test raster_to_vector method"""
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL')
        haz_fl.check()
        meta_orig = haz_fl.centroids.meta
        inten_orig = haz_fl.intensity
        fract_orig = haz_fl.fraction

        haz_fl.raster_to_vector()

        self.assertEqual(haz_fl.centroids.meta, dict())
        self.assertAlmostEqual(haz_fl.centroids.lat.min(),
                               meta_orig['transform'][5]
                               + meta_orig['height'] * meta_orig['transform'][4]
                               - meta_orig['transform'][4] / 2)
        self.assertAlmostEqual(haz_fl.centroids.lat.max(),
                               meta_orig['transform'][5] + meta_orig['transform'][4] / 2)
        self.assertAlmostEqual(haz_fl.centroids.lon.max(),
                               meta_orig['transform'][2]
                               + meta_orig['width'] * meta_orig['transform'][0]
                               - meta_orig['transform'][0] / 2)
        self.assertAlmostEqual(haz_fl.centroids.lon.min(),
                               meta_orig['transform'][2] + meta_orig['transform'][0] / 2)
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.crs, meta_orig['crs']))
        self.assertTrue(np.allclose(haz_fl.intensity.data, inten_orig.data))
        self.assertTrue(np.allclose(haz_fl.fraction.data, fract_orig.data))

    def test_reproject_vector_pass(self):
        """Test reproject_vector"""
        haz_fl = Hazard('FL',
                        event_id=np.array([1]),
                        date=np.array([1]),
                        frequency=np.array([1]),
                        orig=np.array([1]),
                        event_name=['1'],
                        intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
                        fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
                        centroids=Centroids.from_lat_lon(
                            np.array([1, 2, 3]), np.array([1, 2, 3])),)
        haz_fl.check()

        haz_fl.reproject_vector(dst_crs='epsg:2202')
        self.assertTrue(np.allclose(haz_fl.centroids.lat,
                                    np.array([331585.4099637291, 696803.88, 1098649.44])))
        self.assertTrue(np.allclose(haz_fl.centroids.lon,
                                    np.array([11625664.37925186, 11939560.43, 12244857.13])))
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.crs, 'epsg:2202'))
        self.assertTrue(np.allclose(haz_fl.intensity.toarray(), np.array([0.5, 0.2, 0.1])))
        self.assertTrue(np.allclose(haz_fl.fraction.toarray(), np.array([0.5, 0.2, 0.1]) / 2))

    def test_vector_to_raster_pass(self):
        """Test vector_to_raster"""
        haz_fl = Hazard('FL',
                        event_id=np.array([1]),
                        date=np.array([1]),
                        frequency=np.array([1]),
                        orig=np.array([1]),
                        event_name=['1'],
                        intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
                        fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
                        centroids=Centroids.from_lat_lon(
                            np.array([1, 2, 3]), np.array([1, 2, 3])),)
        haz_fl.check()

        haz_fl.vector_to_raster()
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.meta['crs'], 'epsg:4326'))
        self.assertAlmostEqual(haz_fl.centroids.meta['transform'][0], 1.0)
        self.assertAlmostEqual(haz_fl.centroids.meta['transform'][1], 0)
        self.assertAlmostEqual(haz_fl.centroids.meta['transform'][2], 0.5)
        self.assertAlmostEqual(haz_fl.centroids.meta['transform'][3], 0)
        self.assertAlmostEqual(haz_fl.centroids.meta['transform'][4], -1.0)
        self.assertAlmostEqual(haz_fl.centroids.meta['transform'][5], 3.5)
        self.assertEqual(haz_fl.centroids.meta['height'], 3)
        self.assertEqual(haz_fl.centroids.meta['width'], 3)
        self.assertEqual(haz_fl.centroids.lat.size, 0)
        self.assertEqual(haz_fl.centroids.lon.size, 0)
        self.assertTrue(haz_fl.intensity.min() >= 0)
        self.assertTrue(haz_fl.intensity.max() <= 0.5)
        self.assertTrue(haz_fl.fraction.min() >= 0)
        self.assertTrue(haz_fl.fraction.max() <= 0.5 / 2)


class TestClear(unittest.TestCase):
    """Test clear method"""

    def test_clear(self):
        """Clear method clears everything"""
        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        haz1.units = "m"
        haz1.frequency_unit = "1/m"
        haz1.foo = np.arange(10)
        haz1.clear()
        self.assertEqual(list(vars(haz1.tag).values()), ['', '', ''])
        self.assertEqual(haz1.units, '')
        self.assertEqual(haz1.frequency_unit, DEF_FREQ_UNIT)
        self.assertEqual(haz1.centroids.size, 0)
        self.assertEqual(len(haz1.event_name), 0)
        for attr in vars(haz1).keys():
            if attr not in ['tag', 'units', 'event_name', 'pool', 'frequency_unit']:
                self.assertEqual(getattr(haz1, attr).size, 0)
        self.assertIsNone(haz1.pool)

    def test_clear_pool(self):
        """Clear method should not clear a process pool"""
        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type='TC')
        pool = Pool(nodes=2)
        haz1.pool = pool
        haz1.check()
        haz1.clear()
        self.assertEqual(haz1.pool, pool)
        pool.close()
        pool.join()
        pool.clear()

def dummy_step_impf(haz):
    from climada.entity import ImpactFunc
    intensity = (0, 1, haz.intensity.max())
    impf = ImpactFunc.from_step_impf(intensity)
    return impf

class TestImpactFuncs(unittest.TestCase):
    """Test methods mainly for computing impacts"""
    def test_haz_type(self):
        """Test haz_type property"""
        haz = dummy_hazard()
        self.assertEqual(haz.haz_type, 'TC')
        haz.tag.haz_type = 'random'
        self.assertEqual(haz.haz_type, 'random')

    def test_cent_exp_col(self):
        """Test return of centroid exposures column"""
        haz = dummy_hazard()
        self.assertEqual(haz.centr_exp_col, 'centr_TC')
        haz.tag.haz_type = 'random'
        self.assertEqual(haz.centr_exp_col, 'centr_random')
        haz = Hazard()
        self.assertEqual(haz.centr_exp_col, 'centr_')

    def test_get_mdr(self):
        haz = dummy_hazard()
        impf = dummy_step_impf(haz)

        #single index
        for idx in range(3):
            cent_idx = np.array([idx])
            mdr = haz.get_mdr(cent_idx, impf)
            true_mdr = np.digitize(haz.intensity[:, idx].toarray(), [0, 1]) - 1
            np.testing.assert_array_almost_equal(mdr.toarray(), true_mdr)

        #repeated index
        cent_idx = np.array([0, 0, 1])
        mdr = haz.get_mdr(cent_idx, impf)
        true_mdr = np.digitize(haz.intensity[:, cent_idx].toarray(), [0, 1]) - 1
        np.testing.assert_array_almost_equal(mdr.toarray(), true_mdr)

        #mdr is not zero at 0
        impf.mdd += 1
        #single index
        for idx in range(3):
            cent_idx = np.array([idx])
            mdr = haz.get_mdr(cent_idx, impf)
            true_mdr = np.digitize(haz.intensity[:, idx].toarray(), [0, 1])
            np.testing.assert_array_almost_equal(mdr.toarray(), true_mdr)

    def test_get_paa(self):
        haz = dummy_hazard()
        impf = dummy_step_impf(haz)

        idx = [0, 1]
        cent_idx = np.array(idx)
        paa = haz.get_paa(cent_idx, impf)
        true_paa = np.ones(haz.intensity[:, idx].shape)
        np.testing.assert_array_almost_equal(paa.toarray(), true_paa)

        #repeated index
        idx = [0, 0]
        cent_idx = np.array(idx)
        paa = haz.get_paa(cent_idx, impf)
        true_paa = np.ones(haz.intensity[:, idx].shape)
        np.testing.assert_array_almost_equal(paa.toarray(), true_paa)

        #paa is not zero at 0
        impf.paa += 1
        #repeated index
        idx = [0, 0, 1]
        cent_idx = np.array(idx)
        paa = haz.get_paa(cent_idx, impf)
        true_paa = np.ones(haz.intensity[:, idx].shape) + 1
        np.testing.assert_array_almost_equal(paa.toarray(), true_paa)

    def test_get_fraction(self):
        haz = dummy_hazard()

        #standard index
        idx = [0, 1]
        cent_idx = np.array(idx)
        frac = haz._get_fraction(cent_idx)
        true_frac = haz.fraction[:, idx]
        np.testing.assert_array_equal(frac.toarray(), true_frac.toarray())

        #repeated index
        idx = [0, 0]
        cent_idx = np.array(idx)
        frac = haz._get_fraction(cent_idx)
        true_frac = haz.fraction[:, idx]
        np.testing.assert_array_equal(frac.toarray(), true_frac.toarray())

        #index is None
        cent_idx = None
        frac = haz._get_fraction(cent_idx)
        true_frac = haz.fraction
        np.testing.assert_array_equal(frac.toarray(), true_frac.toarray())

        #test empty fraction
        haz.fraction = sparse.csr_matrix(haz.fraction.shape)
        frac = haz._get_fraction()
        self.assertIsNone(frac)

        frac = haz._get_fraction(np.array([0, 1]))
        self.assertIsNone(frac)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHDF5))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestReaderMat))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRemoveDupl))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSelect))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStats))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestYearset))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroids))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestClear))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
