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
from pathlib import Path

import numpy as np
from pathos.pools import ProcessPool as Pool
from scipy import sparse

import climada.util.coordinates as u_coord
import climada.util.dates_times as u_dt
from climada import CONFIG
from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
from climada.test import get_test_file
from climada.util.constants import HAZ_TEMPLATE_XLS

DATA_DIR: Path = CONFIG.hazard.test_data.dir()
"""
Directory for writing (and subsequent reading) of temporary files created during tests.
"""

HAZ_TEST_TC: Path = get_test_file("test_tc_florida")
"""
Hazard test file from Data API: Hurricanes from 1851 to 2011 over Florida with 100 centroids.
Fraction is empty. Format: HDF5.
"""


def dummy_hazard():
    fraction = sparse.csr_matrix(
        [[0.02, 0.03, 0.04], [0.01, 0.01, 0.01], [0.3, 0.1, 0.0], [0.3, 0.2, 0.0]]
    )
    intensity = sparse.csr_matrix(
        [[0.2, 0.3, 0.4], [0.1, 0.1, 0.01], [4.3, 2.1, 1.0], [5.3, 0.2, 0.0]]
    )

    return Hazard(
        "TC",
        intensity=intensity,
        fraction=fraction,
        centroids=Centroids(lat=np.array([1, 3, 5]), lon=np.array([2, 4, 6])),
        event_id=np.array([1, 2, 3, 4]),
        event_name=["ev1", "ev2", "ev3", "ev4"],
        date=np.array([1, 2, 3, 4]),
        orig=np.array([True, False, False, True]),
        frequency=np.array([0.1, 0.5, 0.5, 0.2]),
        frequency_unit="1/week",
        units="m/s",
    )


class TestLoader(unittest.TestCase):
    """Test loading functions from the Hazard class"""

    def setUp(self):
        """Test fixure: Build a valid hazard"""
        centroids = Centroids(
            lat=np.array([1, 3]),
            lon=np.array([2, 3]),
            region_id=np.array([1, 2]),
        )
        self.hazard = Hazard(
            "TC",
            centroids=centroids,
            event_id=np.array([1, 2, 3]),
            event_name=["A", "B", "C"],
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
            intensity=self.hazard.intensity,
        )
        hazard.check()
        np.testing.assert_array_equal(hazard.fraction.shape, hazard.intensity.shape)
        self.assertEqual(hazard.fraction.nnz, 0)  # No nonzero entries

    def test_check_wrongFreq_fail(self):
        """Wrong hazard definition"""
        self.hazard.frequency = np.array([1, 2])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn("Invalid Hazard.frequency size: 3 != 2.", str(cm.exception))

    def test_check_wrongInten_fail(self):
        """Wrong hazard definition"""
        self.hazard.intensity = sparse.csr_matrix([[1, 2], [1, 2]])
        with self.assertRaisesRegex(
            ValueError, "Invalid Hazard.intensity row size: 3 != 2."
        ):
            self.hazard.check()

    def test_check_wrongFrac_fail(self):
        """Wrong hazard definition"""
        self.hazard.fraction = sparse.csr_matrix([[1], [1], [1]])
        with self.assertRaisesRegex(
            ValueError, "Invalid Hazard.fraction column size: 2 != 1."
        ):
            self.hazard.check()

    def test_check_wrongEvName_fail(self):
        """Wrong hazard definition"""
        self.hazard.event_name = ["M"]

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn("Invalid Hazard.event_name size: 3 != 1.", str(cm.exception))

    def test_check_wrongId_fail(self):
        """Wrong hazard definition"""
        self.hazard.event_id = np.array([1, 2, 1])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn("There are events with the same identifier.", str(cm.exception))

    def test_check_wrong_date_fail(self):
        """Wrong hazard definition"""
        self.hazard.date = np.array([1, 2])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn("Invalid Hazard.date size: 3 != 2.", str(cm.exception))

    def test_check_wrong_orig_fail(self):
        """Wrong hazard definition"""
        self.hazard.orig = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError) as cm:
            self.hazard.check()
        self.assertIn("Invalid Hazard.orig size: 3 != 4.", str(cm.exception))

    def test_event_name_to_id_pass(self):
        """Test event_name_to_id function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        self.assertEqual(haz.get_event_id("event001")[0], 1)
        self.assertEqual(haz.get_event_id("event084")[0], 84)

    def test_event_name_to_id_fail(self):
        """Test event_name_to_id function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        with self.assertRaises(ValueError) as cm:
            haz.get_event_id("1050")
        self.assertIn("No event with name: 1050", str(cm.exception))

    def test_event_id_to_name_pass(self):
        """Test event_id_to_name function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        self.assertEqual(haz.get_event_name(2), "event002")
        self.assertEqual(haz.get_event_name(48), "event048")

    def test_event_id_to_name_fail(self):
        """Test event_id_to_name function."""
        haz = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        with self.assertRaises(ValueError) as cm:
            haz.get_event_name(1050)
        self.assertIn("No event with id: 1050", str(cm.exception))

    def test_get_date_strings_pass(self):
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        haz.event_name[5] = "HAZEL"
        haz.event_name[10] = "HAZEL"

        self.assertEqual(len(haz.get_event_date("HAZEL")), 2)
        self.assertEqual(haz.get_event_date("HAZEL")[0], u_dt.date_to_str(haz.date[5]))
        self.assertEqual(haz.get_event_date("HAZEL")[1], u_dt.date_to_str(haz.date[10]))

        self.assertEqual(haz.get_event_date(2)[0], u_dt.date_to_str(haz.date[1]))

        self.assertEqual(len(haz.get_event_date()), haz.date.size)
        self.assertEqual(haz.get_event_date()[560], u_dt.date_to_str(haz.date[560]))

    def test_check_matrices(self):
        """Test the check_matrices method"""
        hazard = Hazard("TC")
        hazard.fraction = sparse.csr_matrix(np.zeros((2, 2)))
        hazard.check_matrices()  # No error, fraction.nnz = 0
        hazard.fraction = sparse.csr_matrix(np.ones((2, 2)))
        with self.assertRaisesRegex(
            ValueError, "Intensity and fraction matrices must have the same shape"
        ):
            hazard.check_matrices()
        hazard.intensity = sparse.csr_matrix(np.ones((2, 3)))
        with self.assertRaisesRegex(
            ValueError, "Intensity and fraction matrices must have the same shape"
        ):
            hazard.check_matrices()

        # Check that matrices are pruned
        hazard.intensity[:] = 0
        hazard.fraction = sparse.csr_matrix(([0], [0], [0, 1, 1]), shape=(2, 3))
        hazard.check_matrices()
        for attr in ("intensity", "fraction"):
            with self.subTest(matrix=attr):
                matrix = getattr(hazard, attr)
                self.assertEqual(matrix.nnz, 0)
                self.assertTrue(matrix.has_canonical_format)


class TestRemoveDupl(unittest.TestCase):
    """Test remove_duplicates method."""

    def test_equal_same(self):
        """Append the same hazard and remove duplicates, obtain initial hazard."""
        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        haz2 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        haz1.append(haz2)
        haz1.remove_duplicates()
        haz1.check()
        self.assertEqual(haz1.event_name, haz2.event_name)
        self.assertTrue(np.array_equal(haz1.event_id, haz2.event_id))
        self.assertTrue(np.array_equal(haz1.frequency, haz2.frequency))
        self.assertEqual(haz1.frequency_unit, haz2.frequency_unit)
        self.assertTrue(np.array_equal(haz1.date, haz2.date))
        self.assertTrue(np.array_equal(haz1.orig, haz2.orig))
        self.assertTrue(
            np.array_equal(haz1.intensity.toarray(), haz2.intensity.toarray())
        )
        self.assertTrue(
            np.array_equal(haz1.fraction.toarray(), haz2.fraction.toarray())
        )
        self.assertTrue((haz1.intensity != haz2.intensity).nnz == 0)
        self.assertTrue((haz1.fraction != haz2.fraction).nnz == 0)
        self.assertEqual(haz1.units, haz2.units)
        self.assertEqual(haz1.haz_type, haz2.haz_type)

    def test_same_events_same(self):
        """Append hazard with same events and diff centroids. After removing
        duplicate events, initial events are obtained with 0 intensity and
        fraction in new appended centroids."""
        haz1 = dummy_hazard()
        centroids = Centroids(lat=np.array([7, 9, 11]), lon=np.array([8, 10, 12]))
        fraction = sparse.csr_matrix(
            [
                [0.22, 0.32, 0.44],
                [0.11, 0.11, 0.11],
                [0.32, 0.11, 0.99],
                [0.32, 0.22, 0.88],
            ]
        )
        intensity = sparse.csr_matrix(
            [
                [0.22, 3.33, 6.44],
                [1.11, 0.11, 1.11],
                [8.33, 4.11, 4.4],
                [9.33, 9.22, 1.77],
            ]
        )
        haz2 = Hazard(
            "TC",
            centroids=centroids,
            event_id=haz1.event_id,
            event_name=haz1.event_name,
            frequency=haz1.frequency,
            frequency_unit="1/week",
            date=haz1.date,
            fraction=fraction,
            intensity=intensity,
            units="m/s",
        )

        haz1.append(haz2)
        haz1.remove_duplicates()
        haz1.check()

        # expected values
        haz_res = dummy_hazard()
        haz_res.intensity = sparse.hstack(
            [haz_res.intensity, sparse.csr_matrix((haz_res.intensity.shape[0], 3))],
            format="csr",
        )
        haz_res.fraction = sparse.hstack(
            [haz_res.fraction, sparse.csr_matrix((haz_res.fraction.shape[0], 3))],
            format="csr",
        )
        self.assertTrue(
            np.array_equal(haz_res.intensity.toarray(), haz1.intensity.toarray())
        )
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(
            np.array_equal(haz_res.fraction.toarray(), haz1.fraction.toarray())
        )
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        self.assertEqual(haz1.event_name, haz_res.event_name)
        self.assertTrue(np.array_equal(haz1.date, haz_res.date))
        self.assertTrue(np.array_equal(haz1.orig, haz_res.orig))
        self.assertTrue(np.array_equal(haz1.event_id, haz_res.event_id))
        self.assertTrue(np.array_equal(haz1.frequency, haz_res.frequency))
        self.assertEqual(haz1.frequency_unit, haz_res.frequency_unit)
        self.assertEqual(haz_res.units, haz1.units)
        self.assertEqual(haz1.haz_type, haz_res.haz_type)


class TestSelect(unittest.TestCase):
    """Test select method."""

    def test_select_event_name(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(event_names=["ev4", "ev1"])

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.2, 0.1])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.3, 0.2, 0.0], [0.02, 0.03, 0.04]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[5.3, 0.2, 0.0], [0.2, 0.3, 0.4]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev4", "ev1"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_event_id(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(event_id=[4, 1])

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.2, 0.1])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.3, 0.2, 0.0], [0.02, 0.03, 0.04]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[5.3, 0.2, 0.0], [0.2, 0.3, 0.4]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev4", "ev1"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_event_id(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(event_id=np.array([4, 1]))

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([4, 1])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.2, 0.1])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.3, 0.2, 0.0], [0.02, 0.03, 0.04]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[5.3, 0.2, 0.0], [0.2, 0.3, 0.4]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev4", "ev1"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_orig_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(orig=True)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([1, 4])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([1, 4])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([True, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.1, 0.2])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.02, 0.03, 0.04], [0.3, 0.2, 0.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[0.2, 0.3, 0.4], [5.3, 0.2, 0.0]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev1", "ev4"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_syn_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(orig=False)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev2", "ev3"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=(2, 4))

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3, 4])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3, 4])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False, True])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5, 0.2])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0], [0.3, 0.2, 0.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0], [5.3, 0.2, 0.0]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev2", "ev3", "ev4"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_str_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=("0001-01-02", "0001-01-03"))

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev2", "ev3"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_date_and_orig_pass(self):
        """Test select historical events."""
        haz = dummy_hazard()
        sel_haz = haz.select(date=(2, 4), orig=False)

        self.assertTrue(np.array_equal(sel_haz.centroids.coord, haz.centroids.coord))
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(
                sel_haz.fraction.toarray(),
                np.array([[0.01, 0.01, 0.01], [0.3, 0.1, 0.0]]),
            )
        )
        self.assertTrue(
            np.array_equal(
                sel_haz.intensity.toarray(),
                np.array([[0.1, 0.1, 0.01], [4.3, 2.1, 1.0]]),
            )
        )
        self.assertEqual(sel_haz.event_name, ["ev2", "ev3"])
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
        sel_haz = haz.select(date=["0001-01-02", "0001-01-03"])
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
        haz.centroids.gdf["region_id"] = np.array([5, 7, 9])
        sel_haz = haz.select(date=(2, 4), orig=False, reg_id=9)

        self.assertTrue(
            np.array_equal(sel_haz.centroids.coord.squeeze(), haz.centroids.coord[2, :])
        )
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.date, np.array([2, 3])))
        self.assertTrue(np.array_equal(sel_haz.orig, np.array([False, False])))
        self.assertTrue(np.array_equal(sel_haz.frequency, np.array([0.5, 0.5])))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(sel_haz.fraction.toarray(), np.array([[0.01], [0.0]]))
        )
        self.assertTrue(
            np.array_equal(sel_haz.intensity.toarray(), np.array([[0.01], [1.0]]))
        )
        self.assertEqual(sel_haz.event_name, ["ev2", "ev3"])
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_tight_pass(self):
        """Test select tight box around hazard"""

        # intensity select
        haz = dummy_hazard()
        haz.intensity[:, -1] = 0.0
        sel_haz = haz.select_tight()

        self.assertTrue(
            np.array_equal(
                sel_haz.centroids.coord.squeeze(), haz.centroids.coord[:-1, :]
            )
        )
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, haz.event_id))
        self.assertTrue(np.array_equal(sel_haz.date, haz.date))
        self.assertTrue(np.array_equal(sel_haz.orig, haz.orig))
        self.assertTrue(np.array_equal(sel_haz.frequency, haz.frequency))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(sel_haz.fraction.toarray(), haz.fraction[:, :-1].toarray())
        )
        self.assertTrue(
            np.array_equal(sel_haz.intensity.toarray(), haz.intensity[:, :-1].toarray())
        )
        self.assertEqual(sel_haz.event_name, haz.event_name)
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

        # fraction select
        haz = dummy_hazard()
        haz.fraction[:, -1] = 0.0

        sel_haz = haz.select_tight(val="fraction")

        self.assertTrue(
            np.array_equal(
                sel_haz.centroids.coord.squeeze(), haz.centroids.coord[:-1, :]
            )
        )
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, haz.event_id))
        self.assertTrue(np.array_equal(sel_haz.date, haz.date))
        self.assertTrue(np.array_equal(sel_haz.orig, haz.orig))
        self.assertTrue(np.array_equal(sel_haz.frequency, haz.frequency))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(sel_haz.fraction.toarray(), haz.fraction[:, :-1].toarray())
        )
        self.assertTrue(
            np.array_equal(sel_haz.intensity.toarray(), haz.intensity[:, :-1].toarray())
        )
        self.assertEqual(sel_haz.event_name, haz.event_name)
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

        haz = dummy_hazard()
        haz.intensity[:, -1] = 0.0

        # small buffer: zero field is discarded
        sel_haz = haz.select_tight(buffer=0.1)
        self.assertTrue(
            np.array_equal(
                sel_haz.centroids.coord.squeeze(), haz.centroids.coord[:-1, :]
            )
        )
        # large buffer: zero field is retained
        sel_haz = haz.select_tight(buffer=10)

        self.assertTrue(
            np.array_equal(sel_haz.centroids.coord.squeeze(), haz.centroids.coord)
        )
        self.assertEqual(sel_haz.units, haz.units)
        self.assertTrue(np.array_equal(sel_haz.event_id, haz.event_id))
        self.assertTrue(np.array_equal(sel_haz.date, haz.date))
        self.assertTrue(np.array_equal(sel_haz.orig, haz.orig))
        self.assertTrue(np.array_equal(sel_haz.frequency, haz.frequency))
        self.assertEqual(sel_haz.frequency_unit, haz.frequency_unit)
        self.assertTrue(
            np.array_equal(sel_haz.fraction.toarray(), haz.fraction.toarray())
        )
        self.assertTrue(
            np.array_equal(sel_haz.intensity.toarray(), haz.intensity.toarray())
        )
        self.assertEqual(sel_haz.event_name, haz.event_name)
        self.assertIsInstance(sel_haz, Hazard)
        self.assertIsInstance(sel_haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(sel_haz.fraction, sparse.csr_matrix)

    def test_select_new_fraction_zero(self):
        """Check if a new fraction of only zeros is handled correctly"""
        hazard = dummy_hazard()
        hazard.centroids.gdf["region_id"] = [1, 1, 2]

        # Select a part of the hazard where fraction is zero only
        with self.assertRaisesRegex(
            RuntimeError,
            "Your selection created a Hazard object where the fraction matrix is zero "
            "everywhere",
        ):
            hazard.select(event_id=[3, 4], reg_id=[2])

        # Error should not be thrown if we set everything to zero
        # NOTE: Setting the values of `data` to zero instead of the matrix values will
        #       add explicitly stored zeros. Therefore, this test explicitly checks if
        #       `eliminate_zeros` is called on `fraction` during `select`.
        hazard.fraction.data[...] = 0
        selection = hazard.select(event_id=[3, 4], reg_id=[2])
        np.testing.assert_array_equal(selection.fraction.toarray(), [[0], [0]])


class TestAppend(unittest.TestCase):
    """Test append method."""

    def test_append_empty_fill(self):
        """Append an empty. Obtain initial hazard."""

        def _check_hazard(hazard):
            # expected values
            haz1_orig = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
            self.assertEqual(hazard.event_name, haz1_orig.event_name)
            self.assertTrue(np.array_equal(hazard.event_id, haz1_orig.event_id))
            self.assertTrue(np.array_equal(hazard.date, haz1_orig.date))
            self.assertTrue(np.array_equal(hazard.orig, haz1_orig.orig))
            self.assertTrue(np.array_equal(hazard.frequency, haz1_orig.frequency))
            self.assertEqual(hazard.frequency_unit, haz1_orig.frequency_unit)
            self.assertTrue((hazard.intensity != haz1_orig.intensity).nnz == 0)
            self.assertTrue((hazard.fraction != haz1_orig.fraction).nnz == 0)
            self.assertEqual(hazard.units, haz1_orig.units)
            self.assertEqual(hazard.haz_type, haz1_orig.haz_type)

        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        haz2 = Hazard("TC")
        haz2.centroids.geometry.crs = "epsg:4326"
        haz1.append(haz2)
        haz1.check()
        _check_hazard(haz1)

        haz1 = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")
        haz2 = Hazard("TC")
        haz2.centroids.geometry.crs = "epsg:4326"
        haz2.append(haz1)
        haz2.check()
        _check_hazard(haz2)

    def test_same_centroids_extend(self):
        """Append hazard with same centroids, different events."""
        haz1 = dummy_hazard()
        fraction = sparse.csr_matrix(
            [[0.2, 0.3, 0.4], [0.1, 0.1, 0.1], [0.3, 0.1, 0.9], [0.3, 0.2, 0.8]]
        )
        intensity = sparse.csr_matrix(
            [[0.2, 3.3, 6.4], [1.1, 0.1, 1.01], [8.3, 4.1, 4.0], [9.3, 9.2, 1.7]]
        )
        haz2 = Hazard(
            "TC",
            centroids=haz1.centroids,
            event_id=np.array([5, 6, 7, 8]),
            event_name=["ev5", "ev6", "ev7", "ev8"],
            frequency=np.array([0.9, 0.75, 0.75, 0.22]),
            frequency_unit="1/week",
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
            self.assertTrue(
                haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name
            )
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(
                haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id)
            )
            self.assertTrue(
                haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency)
            )

        self.assertEqual(haz1.centroids.size, 3)
        self.assertTrue(np.array_equal(haz1.centroids.coord, haz2.centroids.coord))
        self.assertEqual(haz1.haz_type, haz1_orig.haz_type)

    def test_incompatible_type_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = dummy_hazard()
        haz2 = dummy_hazard()
        haz2.haz_type = "WS"
        with self.assertRaises(ValueError) as cm:
            haz1.append(haz2)

    def test_incompatible_units_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = dummy_hazard()
        haz2 = dummy_hazard()
        haz2.units = "km/h"
        with self.assertRaises(ValueError) as cm:
            haz1.append(haz2)

    def test_incompatible_freq_units_fail(self):
        """Raise error when append two incompatible hazards."""
        haz1 = dummy_hazard()
        haz2 = dummy_hazard()
        haz2.frequency_unit = "1/month"
        with self.assertRaises(ValueError) as cm:
            haz1.append(haz2)

    def test_all_different_extend(self):
        """Append totally different hazard."""
        haz1 = dummy_hazard()

        fraction = sparse.csr_matrix(
            [[0.2, 0.3, 0.4], [0.1, 0.1, 0.1], [0.3, 0.1, 0.9], [0.3, 0.2, 0.8]]
        )
        intensity = sparse.csr_matrix(
            [[0.2, 3.3, 6.4], [1.1, 0.1, 1.01], [8.3, 4.1, 4.0], [9.3, 9.2, 1.7]]
        )
        haz2 = Hazard(
            "TC",
            date=np.ones((4,)),
            orig=np.ones((4,)),
            centroids=Centroids(lat=np.array([7, 9, 11]), lon=np.array([8, 10, 12])),
            event_id=np.array([5, 6, 7, 8]),
            event_name=["ev5", "ev6", "ev7", "ev8"],
            frequency=np.array([0.9, 0.75, 0.75, 0.22]),
            frequency_unit="1/week",
            units="m/s",
            intensity=intensity,
            fraction=fraction,
        )

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
            self.assertTrue(
                haz1.event_name[i_ev] in haz1_orig.event_name + haz2.event_name
            )
            self.assertTrue(haz1.date[i_ev] in np.append(haz1_orig.date, haz2.date))
            self.assertTrue(haz1.orig[i_ev] in np.append(haz1_orig.orig, haz2.orig))
            self.assertTrue(
                haz1.event_id[i_ev] in np.append(haz1_orig.event_id, haz2.event_id)
            )
            self.assertTrue(
                haz1.frequency[i_ev] in np.append(haz1_orig.frequency, haz2.frequency)
            )

        self.assertEqual(haz1.centroids.size, 6)
        self.assertEqual(haz1_orig.units, haz1.units)
        self.assertEqual(haz1_orig.frequency_unit, haz1.frequency_unit)
        self.assertEqual(haz1.haz_type, haz1_orig.haz_type)

    def test_same_events_append(self):
        """Append hazard with same events (and diff centroids).
        Events are appended with all new centroids columns."""
        haz1 = dummy_hazard()
        fraction = sparse.csr_matrix(
            [
                [0.22, 0.32, 0.44],
                [0.11, 0.11, 0.11],
                [0.32, 0.11, 0.99],
                [0.32, 0.22, 0.88],
            ]
        )
        intensity = sparse.csr_matrix(
            [
                [0.22, 3.33, 6.44],
                [1.11, 0.11, 1.11],
                [8.33, 4.11, 4.4],
                [9.33, 9.22, 1.77],
            ]
        )
        haz2 = Hazard(
            "TC",
            centroids=Centroids(lat=np.array([7, 9, 11]), lon=np.array([8, 10, 12])),
            event_id=haz1.event_id,
            event_name=haz1.event_name.copy(),
            frequency=haz1.frequency,
            frequency_unit=haz1.frequency_unit,
            date=haz1.date,
            units="m/s",
            fraction=fraction,
            intensity=intensity,
        )

        haz1.append(haz2)

        # expected values
        haz1_ori = dummy_hazard()
        res_inten = np.zeros((8, 6))
        res_inten[0:4, 0:3] = haz1_ori.intensity.toarray()
        res_inten[4:, 3:] = haz2.intensity.toarray()

        res_frac = np.zeros((8, 6))
        res_frac[0:4, 0:3] = haz1_ori.fraction.toarray()
        res_frac[4:, 3:] = haz2.fraction.toarray()

        self.assertTrue(np.array_equal(res_inten, haz1.intensity.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.intensity))
        self.assertTrue(np.array_equal(res_frac, haz1.fraction.toarray()))
        self.assertTrue(sparse.isspmatrix_csr(haz1.fraction))
        self.assertEqual(haz1.event_name, haz1_ori.event_name + haz2.event_name)
        self.assertTrue(np.array_equal(haz1.date, np.append(haz1_ori.date, haz2.date)))
        self.assertTrue(np.array_equal(haz1.orig, np.append(haz1_ori.orig, haz2.orig)))
        self.assertTrue(np.array_equal(haz1.event_id, np.arange(1, 9)))
        self.assertTrue(
            np.array_equal(
                haz1.frequency, np.append(haz1_ori.frequency, haz2.frequency)
            )
        )
        self.assertEqual(haz1_ori.frequency_unit, haz1.frequency_unit)
        self.assertEqual(haz1_ori.units, haz1.units)

        self.assertEqual(haz1.haz_type, haz1_ori.haz_type)

    def test_concat_pass(self):
        """Test concatenate function."""

        haz_1 = Hazard(
            "TC",
            centroids=Centroids(
                lat=np.array([1, 3, 5]), lon=np.array([2, 4, 6]), crs="epsg:4326"
            ),
            event_id=np.array([1]),
            event_name=["ev1"],
            date=np.array([1]),
            orig=np.array([True]),
            frequency=np.array([1.0]),
            frequency_unit="1/week",
            fraction=sparse.csr_matrix([[0.02, 0.03, 0.04]]),
            intensity=sparse.csr_matrix([[0.2, 0.3, 0.4]]),
            units="m/s",
        )

        haz_2 = Hazard(
            "TC",
            centroids=Centroids(
                lat=np.array([1, 3, 5]), lon=np.array([2, 4, 6]), crs="epsg:4326"
            ),
            event_id=np.array([1]),
            event_name=["ev2"],
            date=np.array([2]),
            orig=np.array([False]),
            frequency=np.array([1.0]),
            frequency_unit="1/week",
            fraction=sparse.csr_matrix([[1.02, 1.03, 1.04]]),
            intensity=sparse.csr_matrix([[1.2, 1.3, 1.4]]),
            units="m/s",
        )

        haz = Hazard.concat([haz_1, haz_2])

        hres_frac = sparse.csr_matrix([[0.02, 0.03, 0.04], [1.02, 1.03, 1.04]])
        hres_inten = sparse.csr_matrix([[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]])

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
        self.assertEqual(haz.event_name, ["ev1", "ev2"])
        self.assertTrue(np.array_equal(haz.centroids.coord, haz_1.centroids.coord))
        self.assertTrue(np.array_equal(haz.centroids.coord, haz_2.centroids.coord))
        self.assertEqual(haz.centroids.crs, haz_1.centroids.crs)

    def test_append_new_var_pass(self):
        """New variable appears if hazard to append is empty."""
        haz = dummy_hazard()
        haz.frequency_unit = haz.get_default("frequency_unit")
        haz.new_var = np.ones(haz.size)

        app_haz = Hazard("TC")
        app_haz.append(haz)
        self.assertIn("new_var", app_haz.__dict__)

    def test_append_raise_type_error(self):
        """Raise error if hazards of different class"""
        haz1 = Hazard("TC", units="m/s")
        from climada.hazard import TropCyclone

        haz2 = TropCyclone()
        with self.assertRaises(TypeError):
            haz1.append(haz2)

    def test_concat_raise_value_error(self):
        """Raise error if hazards with different units, type or crs"""
        haz1 = Hazard(
            "TC", units="m/s", centroids=Centroids(lat=[], lon=[], crs="epsg:4326")
        )
        haz3 = Hazard("EQ")
        with self.assertRaisesRegex(ValueError, "different types"):
            Hazard.concat([haz1, haz3])

        haz4 = Hazard("TC", units="cm")
        with self.assertRaisesRegex(ValueError, "different units"):
            Hazard.concat([haz1, haz4])

        haz5 = Hazard("TC", centroids=Centroids(lat=[], lon=[], crs="epsg:7777"))
        with self.assertRaisesRegex(ValueError, "different CRS"):
            Hazard.concat([haz1, haz5])

    def test_change_centroids(self):
        """Set new centroids for hazard"""
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        haz_1 = Hazard(
            "TC",
            centroids=cent1,
            event_id=np.array([1]),
            event_name=["ev1"],
            date=np.array([1]),
            orig=np.array([True]),
            frequency=np.array([1.0]),
            frequency_unit="1/week",
            fraction=sparse.csr_matrix([[0.02, 0.03]]),
            intensity=sparse.csr_matrix([[0.2, 0.3]]),
            units="m/s",
        )

        lat2, lon2 = np.array([0, 1, 3]), np.array([0, -1, 3])
        on_land2 = np.array([True, True, False])
        cent2 = Centroids(lat=lat2, lon=lon2, on_land=on_land2)

        haz_2 = haz_1.change_centroids(cent2)

        self.assertTrue(
            np.array_equal(haz_2.intensity.toarray(), np.array([[0.2, 0.3, 0.0]]))
        )
        self.assertTrue(
            np.array_equal(haz_2.fraction.toarray(), np.array([[0.02, 0.03, 0.0]]))
        )
        self.assertTrue(np.array_equal(haz_2.event_id, np.array([1])))
        self.assertTrue(np.array_equal(haz_2.event_name, ["ev1"]))
        self.assertTrue(np.array_equal(haz_2.orig, [True]))

        """Test error for projection"""
        lat3, lon3 = np.array([0.5, 3, 1]), np.array([-0.5, 3, 1])
        on_land3 = np.array([True, True, False])
        cent3 = Centroids(lat=lat3, lon=lon3, on_land=on_land3)

        with self.assertRaises(ValueError) as cm:
            haz_1.change_centroids(cent3, threshold=100)
        self.assertIn(
            "two hazard centroids are mapped to the same centroids", str(cm.exception)
        )

    def test_change_centroids_raster(self):
        """Set new centroids for hazard"""
        lat, lon = np.array([0, 1]), np.array([0, -1])
        on_land = np.array([True, True])
        cent1 = Centroids(lat=lat, lon=lon, on_land=on_land)

        haz_1 = Hazard(
            "TC",
            centroids=cent1,
            event_id=np.array([1]),
            event_name=["ev1"],
            date=np.array([1]),
            orig=np.array([True]),
            frequency=np.array([1.0]),
            frequency_unit="1/week",
            fraction=sparse.csr_matrix([[0.02, 0.03]]),
            intensity=sparse.csr_matrix([[0.2, 0.3]]),
            units="m/s",
        )

        """Test with raster centroids"""
        cent4 = Centroids.from_pnt_bounds(points_bounds=(-1, 0, 0, 1), res=1)

        haz_4 = haz_1.change_centroids(cent4)

        self.assertTrue(
            np.array_equal(haz_4.intensity.toarray(), np.array([[0.3, 0.0, 0.0, 0.2]]))
        )
        self.assertTrue(
            np.array_equal(haz_4.fraction.toarray(), np.array([[0.03, 0.0, 0.0, 0.02]]))
        )
        self.assertTrue(np.array_equal(haz_4.event_id, np.array([1])))
        self.assertTrue(np.array_equal(haz_4.event_name, ["ev1"]))
        self.assertTrue(np.array_equal(haz_4.orig, [True]))


class TestStats(unittest.TestCase):
    """Test return period statistics"""

    def test_degenerate_pass(self):
        """Test degenerate call."""
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        return_period = np.array([25, 50, 100, 250])
        haz.intensity = sparse.csr_matrix(np.zeros(haz.intensity.shape))
        inten_stats = (
            haz.local_exceedance_intensity(return_period)[0]
            .values[:, 1:]
            .T.astype(float)
        )
        np.testing.assert_allclose(inten_stats, np.full((4, 100), np.nan))

    def test_local_exceedance_intensity(self):
        """Test local exceedance frequencies with lin lin interpolation"""
        haz = dummy_hazard()
        haz.intensity = sparse.csr_matrix([[1.0, 3.0, 1.0], [2.0, 3.0, 0.0]])
        haz.intensity_thres = 0.5
        haz.frequency = np.full(2, 1.0)
        return_period = np.array([0.5, 2.0 / 3.0, 1.0])
        # first centroid has intensities 1,2 with cum frequencies 2,1
        # first centroid has intensities 3 with cum frequencies 2 (due to grouping of values)
        # third centroid has intensities 1 with cum frequencies 1
        # testing at frequencies 2, 1.5, 1
        inten_stats, _, _ = haz.local_exceedance_intensity(
            return_period,
            log_frequency=False,
            log_intensity=False,
            method="extrapolate_constant",
        )
        np.testing.assert_allclose(
            inten_stats[inten_stats.columns[1:]].values,
            np.array([[1.0, 1.5, 2.0], [3.0, 3.0, 3.0], [0.0, 0.0, 1.0]]),
        )

    def test_local_exceedance_intensity_methods(self):
        """Test local exceedance frequencies with different methods"""
        haz = dummy_hazard()
        haz.intensity = sparse.csr_matrix(
            [[0, 0, 1e1], [0.2, 1e1, 1e2], [1e3, 1e3, 1e3]]
        )
        haz.intensity_thres = 0.5
        haz.frequency = np.array([1.0, 0.1, 0.01])
        return_period = (1000, 30, 0.1)
        # first centroid has intensities 1e3 with frequencies .01, cum freq .01
        # second centroid has intensities 1e1, 1e3 with cum frequencies .1, .01, cum freq .11, .01
        # third centroid has intensities 1e1, 1e2, 1e3 with cum frequencies 1., .1, .01, cum freq 1.11, .11, .01
        # testing at frequencies .001, .033, 10.

        # test stepfunction
        inten_stats, _, _ = haz.local_exceedance_intensity(
            return_periods=(1000, 30, 0.1), method="stepfunction"
        )
        np.testing.assert_allclose(
            inten_stats.values[:, 1:].astype(float),
            np.array([[1e3, 0, 0], [1e3, 1e1, 0], [1e3, 1e2, 0]]),
        )

        # test log log extrapolation
        inten_stats, _, _ = haz.local_exceedance_intensity(
            return_periods=(1000, 30, 0.1), method="extrapolate"
        )
        np.testing.assert_allclose(
            inten_stats.values[:, 1:].astype(float),
            np.array([[1e3, 0, 0], [1e5, 1e2, 1e-3], [1e4, 300, 1]]),
            rtol=0.8,
        )

        # test log log interpolation and extrapolation with constant
        inten_stats, _, _ = haz.local_exceedance_intensity(
            return_periods=(1000, 30, 0.1), method="extrapolate_constant"
        )
        np.testing.assert_allclose(
            inten_stats.values[:, 1:].astype(float),
            np.array([[1e3, 0, 0], [1e3, 1e2, 0], [1e3, 300, 0]]),
            rtol=0.8,
        )

        # test log log interpolation and no extrapolation
        inten_stats, _, _ = haz.local_exceedance_intensity(
            return_periods=(1000, 30, 0.1)
        )
        np.testing.assert_allclose(
            inten_stats.values[:, 1:].astype(float),
            np.array(
                [[np.nan, np.nan, np.nan], [np.nan, 1e2, np.nan], [np.nan, 300, np.nan]]
            ),
            rtol=0.8,
        )

        # test lin lin interpolation without extrapolation
        inten_stats, _, _ = haz.local_exceedance_intensity(
            return_periods=(1000, 30, 0.1),
            log_frequency=False,
            log_intensity=False,
            method="extrapolate_constant",
        )
        np.testing.assert_allclose(
            inten_stats.values[:, 1:].astype(float),
            np.array([[1e3, 0, 0], [1e3, 750, 0], [1e3, 750, 0]]),
            rtol=0.8,
        )

    def test_local_return_period(self):
        """Test local return periods with lin lin interpolation"""
        haz = dummy_hazard()
        haz.intensity = sparse.csr_matrix([[1.0, 4.0, 1.0], [2.0, 2.0, 0.0]])
        haz.frequency = np.full(2, 1.0)
        threshold_intensities = np.array([1.0, 2.0, 3.0])
        # first centroid has intensities 1,2 with cum frequencies 2,1
        # second centroid has intensities 2, 4 with cum frequencies 2, 1
        # third centroid has intensities 1 with cum frequencies 1 (0 intensity is neglected)
        # testing at intensities 1, 2, 3
        return_stats, _, _ = haz.local_return_period(
            threshold_intensities,
            log_frequency=False,
            log_intensity=False,
            min_intensity=0,
            method="extrapolate_constant",
        )
        np.testing.assert_allclose(
            return_stats[return_stats.columns[1:]].values,
            np.array([[0.5, 1.0, np.nan], [0.5, 0.5, 2.0 / 3], [1.0, np.nan, np.nan]]),
        )

    def test_local_return_period_methods(self):
        """Test local return periods different methods"""
        haz = dummy_hazard()
        haz.intensity = sparse.csr_matrix(
            [[0, 0, 1e1], [0.0, 1e1, 1e2], [1e3, 1e3, 1e3]]
        )
        haz.intensity_thres = 0.5
        haz.frequency = np.array([1.0, 0.1, 0.01])
        # first centroid has intensities 1e3 with frequencies .01, cum freq .01
        # second centroid has intensities 1e1, 1e3 with cum frequencies .1, .01, cum freq .11, .01
        # third centroid has intensities 1e1, 1e2, 1e3 with cum frequencies 1., .1, .01, cum freq 1.11, .11, .01
        # testing at intensities .1, 300, 1e4

        # test stepfunction
        return_stats, _, _ = haz.local_return_period(
            threshold_intensities=(0.1, 300, 1e5), method="stepfunction"
        )
        np.testing.assert_allclose(
            return_stats.values[:, 1:].astype(float),
            np.array(
                [[100, 100, np.nan], [1 / 0.11, 100, np.nan], [1 / 1.11, 100, np.nan]]
            ),
        )

        # test log log extrapolation
        return_stats, _, _ = haz.local_return_period(
            threshold_intensities=(0.1, 300, 1e5), method="extrapolate"
        )
        np.testing.assert_allclose(
            return_stats.values[:, 1:].astype(float),
            np.array([[100, 100, np.nan], [1.0, 30, 1e3], [0.01, 30, 1e4]]),
            rtol=0.8,
        )

        # test log log interpolation and extrapolation with constant
        return_stats, _, _ = haz.local_return_period(
            threshold_intensities=(0.1, 300, 1e5), method="extrapolate_constant"
        )
        np.testing.assert_allclose(
            return_stats.values[:, 1:].astype(float),
            np.array(
                [[100, 100, np.nan], [1 / 0.11, 30, np.nan], [1 / 1.11, 30, np.nan]]
            ),
            rtol=0.8,
        )

        # test log log interpolation and no extrapolation
        return_stats, _, _ = haz.local_return_period(
            threshold_intensities=(0.1, 300, 1e5)
        )
        np.testing.assert_allclose(
            return_stats.values[:, 1:].astype(float),
            np.array(
                [[np.nan, np.nan, np.nan], [np.nan, 30, np.nan], [np.nan, 30, np.nan]]
            ),
            rtol=0.8,
        )


class TestYearset(unittest.TestCase):
    """Test return period statistics"""

    def test_ref_pass(self):
        """Test against reference."""
        haz = Hazard.from_hdf5(HAZ_TEST_TC)
        orig_year_set = haz.calc_year_set()

        self.assertTrue(
            np.array_equal(np.array(list(orig_year_set.keys())), np.arange(1851, 2012))
        )
        self.assertTrue(np.array_equal(orig_year_set[1851], np.array([1, 11, 21, 31])))
        self.assertTrue(
            np.array_equal(
                orig_year_set[1958],
                np.array([8421, 8431, 8441, 8451, 8461, 8471, 8481, 8491, 8501, 8511]),
            )
        )
        self.assertTrue(
            np.array_equal(
                orig_year_set[1986],
                np.array([11101, 11111, 11121, 11131, 11141, 11151]),
            )
        )
        self.assertTrue(
            np.array_equal(
                orig_year_set[1997],
                np.array([12221, 12231, 12241, 12251, 12261, 12271, 12281, 12291]),
            )
        )
        self.assertTrue(
            np.array_equal(
                orig_year_set[2006],
                np.array(
                    [
                        13571,
                        13581,
                        13591,
                        13601,
                        13611,
                        13621,
                        13631,
                        13641,
                        13651,
                        13661,
                    ]
                ),
            )
        )
        self.assertTrue(
            np.array_equal(
                orig_year_set[2010],
                np.array(
                    [
                        14071,
                        14081,
                        14091,
                        14101,
                        14111,
                        14121,
                        14131,
                        14141,
                        14151,
                        14161,
                        14171,
                        14181,
                        14191,
                        14201,
                        14211,
                        14221,
                        14231,
                        14241,
                        14251,
                    ]
                ),
            )
        )


class TestCentroids(unittest.TestCase):
    """Test return period statistics"""

    def test_reproject_vector_pass(self):
        """Test reproject_vector"""
        haz_fl = Hazard(
            "FL",
            event_id=np.array([1]),
            date=np.array([1]),
            frequency=np.array([1]),
            orig=np.array([1]),
            event_name=["1"],
            intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
            fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
            centroids=Centroids(lat=np.array([1, 2, 3]), lon=np.array([1, 2, 3])),
        )
        haz_fl.check()

        haz_fl.reproject_vector(dst_crs="epsg:2202")
        self.assertTrue(
            np.allclose(
                haz_fl.centroids.lat,
                np.array([331585.4099637291, 696803.88, 1098649.44]),
            )
        )
        self.assertTrue(
            np.allclose(
                haz_fl.centroids.lon,
                np.array([11625664.37925186, 11939560.43, 12244857.13]),
            )
        )
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.crs, "epsg:2202"))
        self.assertTrue(
            np.allclose(haz_fl.intensity.toarray(), np.array([0.5, 0.2, 0.1]))
        )
        self.assertTrue(
            np.allclose(haz_fl.fraction.toarray(), np.array([0.5, 0.2, 0.1]) / 2)
        )


def dummy_step_impf(haz):
    from climada.entity import ImpactFunc

    intensity = (0, 1, haz.intensity.max())
    impf = ImpactFunc.from_step_impf(intensity, haz_type=haz.haz_type)
    return impf


class TestImpactFuncs(unittest.TestCase):
    """Test methods mainly for computing impacts"""

    def test_haz_type(self):
        """Test haz_type property"""
        haz = dummy_hazard()
        self.assertEqual(haz.haz_type, "TC")
        haz.haz_type = "random"
        self.assertEqual(haz.haz_type, "random")

    def test_cent_exp_col(self):
        """Test return of centroid exposures column"""
        haz = dummy_hazard()
        self.assertEqual(haz.centr_exp_col, "centr_TC")
        haz.haz_type = "random"
        self.assertEqual(haz.centr_exp_col, "centr_random")
        haz = Hazard()
        self.assertEqual(haz.centr_exp_col, "centr_")

    def test_get_mdr(self):
        haz = dummy_hazard()
        impf = dummy_step_impf(haz)

        # single index
        for idx in range(3):
            cent_idx = np.array([idx])
            mdr = haz.get_mdr(cent_idx, impf)
            true_mdr = np.digitize(haz.intensity[:, idx].toarray(), [0, 1]) - 1
            np.testing.assert_array_almost_equal(mdr.toarray(), true_mdr)

        # repeated index
        cent_idx = np.array([0, 0, 1])
        mdr = haz.get_mdr(cent_idx, impf)
        true_mdr = np.digitize(haz.intensity[:, cent_idx].toarray(), [0, 1]) - 1
        np.testing.assert_array_almost_equal(mdr.toarray(), true_mdr)

        # mdr is not zero at 0
        impf.mdd += 1
        # single index
        for idx in range(3):
            cent_idx = np.array([idx])
            mdr = haz.get_mdr(cent_idx, impf)
            true_mdr = np.digitize(haz.intensity[:, idx].toarray(), [0, 1])
            np.testing.assert_array_almost_equal(mdr.toarray(), true_mdr)

        # #case with zeros everywhere
        cent_idx = np.array([0, 0, 1])
        impf.mdd = np.array([0, 0, 0, 1])
        # how many non-zeros values are expected
        num_nz_values = 5
        mdr = haz.get_mdr(cent_idx, impf)
        self.assertEqual(mdr.nnz, num_nz_values)

    def test_get_paa(self):
        haz = dummy_hazard()
        impf = dummy_step_impf(haz)

        idx = [0, 1]
        cent_idx = np.array(idx)
        paa = haz.get_paa(cent_idx, impf)
        true_paa = np.ones(haz.intensity[:, idx].shape)
        np.testing.assert_array_almost_equal(paa.toarray(), true_paa)

        # repeated index
        idx = [0, 0]
        cent_idx = np.array(idx)
        paa = haz.get_paa(cent_idx, impf)
        true_paa = np.ones(haz.intensity[:, idx].shape)
        np.testing.assert_array_almost_equal(paa.toarray(), true_paa)

        # paa is not zero at 0
        impf.paa += 1
        # repeated index
        idx = [0, 0, 1]
        cent_idx = np.array(idx)
        paa = haz.get_paa(cent_idx, impf)
        true_paa = np.ones(haz.intensity[:, idx].shape) + 1
        np.testing.assert_array_almost_equal(paa.toarray(), true_paa)

    def test_get_fraction(self):
        haz = dummy_hazard()

        # standard index
        idx = [0, 1]
        cent_idx = np.array(idx)
        frac = haz._get_fraction(cent_idx)
        true_frac = haz.fraction[:, idx]
        np.testing.assert_array_equal(frac.toarray(), true_frac.toarray())

        # repeated index
        idx = [0, 0]
        cent_idx = np.array(idx)
        frac = haz._get_fraction(cent_idx)
        true_frac = haz.fraction[:, idx]
        np.testing.assert_array_equal(frac.toarray(), true_frac.toarray())

        # index is None
        cent_idx = None
        frac = haz._get_fraction(cent_idx)
        true_frac = haz.fraction
        np.testing.assert_array_equal(frac.toarray(), true_frac.toarray())

        # test empty fraction
        haz.fraction = sparse.csr_matrix(haz.fraction.shape)
        frac = haz._get_fraction()
        self.assertIsNone(frac)

        frac = haz._get_fraction(np.array([0, 1]))
        self.assertIsNone(frac)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRemoveDupl))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSelect))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestStats))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestYearset))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAppend))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCentroids))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactFuncs))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
