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

Test of dates_times module
"""

import unittest
import numpy as np
import collections

import climada.util.yearsets as yearsets
from climada.engine import Impact
import climada.util.dates_times as u_dt


IMP = Impact()
IMP.at_event = np.arange(10,110,10)
IMP.frequency = np.array(np.ones(10)*0.2)

SAMPLING_VECT = [np.array([0]), np.array([4]), np.array([1]), np.array([2, 5, 7, 9, 6]),
                 np.array([8]), np.array([3]), np.array([2, 6]), np.array([1]),
                 np.array([3,5]), np.array([])]

YEAR_LIST = list(range(2000, 2010))

class TestYearSets(unittest.TestCase):
    """Test yearset functions"""

    def test_impact_yearset(self):
        """Test computing a yearly impact (yimp) for a given list of years (YEAR_LIST)
        from an impact (IMP) and a sampling vector (SAMPLING_VECT)"""
        yimp, sampling_vect = yearsets.impact_yearset(IMP, YEAR_LIST, correction_fac=False)

        self.assertAlmostEqual(len(sampling_vect), len(YEAR_LIST))

    def test_impact_yearset_sampling_vect(self):
        """Test computing a yearly impact (yimp) for a given list of years (YEAR_LIST)
        from an impact (IMP) and a sampling vector (SAMPLING_VECT)"""
        yimp = yearsets.impact_yearset_from_sampling_vect(IMP, YEAR_LIST, SAMPLING_VECT, False)

        self.assertAlmostEqual(yimp.at_event[3], 340)
        self.assertEqual(u_dt.date_to_str(yimp.date)[0], '2000-01-01')
        self.assertAlmostEqual(np.sum(yimp.at_event), 770)

    def test_sample_from_poisson(self):
        """Test sampling amount of events per year."""
        n_sample_years = 1000
        lam = np.sum(IMP.frequency)
        events_per_year = yearsets.sample_from_poisson(n_sample_years, lam)

        self.assertEqual(events_per_year.size, n_sample_years)
        self.assertAlmostEqual(np.round(np.mean(events_per_year)), 2)

    def test_sample_events(self):
        """Test the sampling of 34 events out of a pool of 20 events."""
        events_per_year = np.array([0, 2, 2, 2, 1, 2, 3, 2, 2, 0, 2, 1, 2, 2, 2, 3, 5, 0, 1, 0])
        frequencies = np.array(np.ones(20)*0.2)

        sampling_vect = yearsets.sample_events(events_per_year, frequencies)

        self.assertEqual(len(sampling_vect), len(events_per_year))
        self.assertEqual(len(np.concatenate(sampling_vect).ravel()), np.sum(events_per_year))
        self.assertEqual(len(np.unique(list(collections.Counter(np.concatenate(sampling_vect).ravel()).values()))), 2)

    def test_computing_imp_per_year(self):
        """Test the calculation of impacts per year from a given sampling dictionary."""
        imp_per_year = yearsets.compute_imp_per_year(IMP, SAMPLING_VECT)
        self.assertEqual(imp_per_year[0], 10)

    def test_correction_fac(self):
        """Test the calculation of a correction factor as the ration of the expected annual
        impact (eai) of the event_impacts and the eai of the annual_impacts"""
        imp_per_year = yearsets.compute_imp_per_year(IMP, SAMPLING_VECT)
        correction_factor = yearsets.calculate_correction_fac(imp_per_year, IMP)

        self.assertAlmostEqual(correction_factor, 1.42857143)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestYearSets)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
