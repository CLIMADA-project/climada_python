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

import numpy as np

from climada.hazard.base import Hazard
from climada.hazard.test.test_base import DATA_DIR, dummy_hazard
from climada.util.constants import DEF_FREQ_UNIT, HAZ_TEMPLATE_XLS


class TestReaderExcel(unittest.TestCase):
    """Test reader functionality of the Hazard class"""

    def test_hazard_pass(self):
        """Read an hazard excel file correctly."""

        # Read demo excel file
        hazard = Hazard.from_excel(HAZ_TEMPLATE_XLS, haz_type="TC")

        # Check results
        n_events = 100
        n_centroids = 45

        self.assertEqual(hazard.units, "")

        self.assertEqual(hazard.centroids.coord.shape, (n_centroids, 2))
        self.assertEqual(hazard.centroids.coord[0][0], -25.95)
        self.assertEqual(hazard.centroids.coord[0][1], 32.57)
        self.assertEqual(hazard.centroids.coord[n_centroids - 1][0], -24.7)
        self.assertEqual(hazard.centroids.coord[n_centroids - 1][1], 33.88)

        self.assertEqual(len(hazard.event_name), 100)
        self.assertEqual(hazard.event_name[12], "event013")

        self.assertEqual(hazard.event_id.dtype, int)
        self.assertEqual(hazard.event_id.shape, (n_events,))
        self.assertEqual(hazard.event_id[0], 1)
        self.assertEqual(hazard.event_id[n_events - 1], 100)

        self.assertEqual(hazard.date.dtype, int)
        self.assertEqual(hazard.date.shape, (n_events,))
        self.assertEqual(hazard.date[0], 675874)
        self.assertEqual(hazard.date[n_events - 1], 676329)

        self.assertEqual(hazard.event_name[0], "event001")
        self.assertEqual(hazard.event_name[50], "event051")
        self.assertEqual(hazard.event_name[-1], "event100")

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

        self.assertEqual(hazard.haz_type, "TC")


class TestHDF5(unittest.TestCase):
    """Test reader functionality of the ExposuresExcel class"""

    def test_write_read_unsupported_type(self):
        """Check if the write command correctly handles unsupported types"""
        file_name = str(DATA_DIR.joinpath("test_unsupported.h5"))

        # Define an unsupported type
        class CustomID:
            id = 1

        # Create a hazard with unsupported type as attribute
        hazard = dummy_hazard()
        hazard.event_id = CustomID()

        # Write the hazard and check the logs for the correct warning
        with self.assertLogs(logger="climada.hazard.io", level="WARN") as cm:
            hazard.write_hdf5(file_name)
        self.assertIn("write_hdf5: the class member event_id is skipped", cm.output[0])

        # Load the file again and compare to previous instance
        hazard_read = Hazard.from_hdf5(file_name)
        self.assertTrue(np.array_equal(hazard.date, hazard_read.date))
        self.assertTrue(
            np.array_equal(hazard_read.event_id, np.array([]))
        )  # Empty array


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReaderExcel)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestHDF5))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
