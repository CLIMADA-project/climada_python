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

Test of dates_times module
"""

import unittest
import numpy as np
import collections

import climada.util.yearsets as yearsets
from climada.engine import Impact
import climada.util.dates_times as u_dt


EVENT_IMPACTS = Impact()
EVENT_IMPACTS.at_event = np.arange(10,110,10)
EVENT_IMPACTS.frequency = np.array(np.ones(10)*0.2)

SAMPLING_VECT = [np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9]),
                 np.array([0,1]), np.array([2,3]), np.array([4,5]), np.array([6,7]), np.array([8,9])]

N_SAMPLED_YEARS = 10
YEAR_LIST = list(range(2000, 2010))

class TestYearSets(unittest.TestCase):
    """Test yearset functions"""
    def test_impact_yearset(self):
        """Test computing an annual_impacts object for a given range of years (N_SAMPLED_YEARS)
        from an event_impacts object and a sampling dictionary"""
        annual_impacts, sampling_dict = yearsets.impact_yearset(EVENT_IMPACTS, N_SAMPLED_YEARS,
                                                                SAMPLING_VECT, False)

        self.assertEqual(annual_impacts.at_event[0], 30)
        self.assertEqual(annual_impacts.date[1], 366)

    def test_impact_yearset_yearlist(self):
        """Test computing an annual_impacts object for a given list of years (YEAR_LIST)
        from an event_impacts object and a sampling dictionary"""
        annual_impacts, sampling_dict = yearsets.impact_yearset(EVENT_IMPACTS, YEAR_LIST,
                                                                SAMPLING_VECT, False)

        self.assertEqual(annual_impacts.at_event[0], 30)
        self.assertEqual(u_dt.date_to_str(annual_impacts.date)[0], '2000-01-01')

    def test_sample_events(self):
        """Test the sampling of 10 events out of a pool of 20 events."""
        tot_n_events = 10
        frequencies = np.array(np.ones(20)*0.2)
        
        selected_events = yearsets.sample_events(tot_n_events, frequencies)

        self.assertEqual(len(selected_events), tot_n_events)
        self.assertEqual(len(np.unique(selected_events)), tot_n_events)


    def test_sample_repetitive_events(self):
        """Test the sampling of 20 events out of a pool of 10 events. (events are sampled
        repetitively, however the times they are sampled differs by a max of 1 count.)"""
        tot_n_events = 20
        n_input_events = 10

        selected_events = yearsets.sample_events(tot_n_events, EVENT_IMPACTS.frequency)

        self.assertEqual(len(selected_events), tot_n_events)
        self.assertEqual(len(np.unique(selected_events)), n_input_events)
        self.assertEqual(np.unique(list(collections.Counter(selected_events).values())), 2)

    # def test_sampling_vect(self):
    #     """Test generating a sampling dictionary with a mean of 2 events per year."""
    #     n_sampled_years = 100000
        
    #     sampling_dict = yearsets.create_sampling_vect(n_sampled_years, EVENT_IMPACTS)
        # self.assertAlmostEqual(np.round(np.mean(sampling_dict['events_per_year'])), 2)
        # self.assertTrue(len(np.unique(list(collections.Counter(sampling_dict['selected_events']).values()))), 2)


    # def test_correction_fac(self):
    #     """Test the calculation of a correction factor as the ration of the expected annual
    #     impact (eai) of the event_impacts and the eai of the annual_impacts"""
    #     correction_factor = yearsets.calculate_correction_fac(impact_per_year, EVENT_IMPACTS)

    #     self.assertAlmostEqual(correction_factor, 1)

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestYearSets)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
