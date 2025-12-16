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

Tests for impact_calc_strat

"""

import unittest
from unittest.mock import MagicMock, patch

from climada.engine import Impact
from climada.entity import ImpactFuncSet
from climada.entity.exposures import Exposures
from climada.hazard import Hazard
from climada.trajectories import Snapshot
from climada.trajectories.impact_calc_strat import ImpactCalcComputation


class TestImpactCalcComputation(unittest.TestCase):
    def setUp(self):
        self.mock_snapshot0 = MagicMock(spec=Snapshot)
        self.mock_snapshot0.exposure = MagicMock(spec=Exposures)
        self.mock_snapshot0.hazard = MagicMock(spec=Hazard)
        self.mock_snapshot0.impfset = MagicMock(spec=ImpactFuncSet)
        self.mock_snapshot1 = MagicMock(spec=Snapshot)
        self.mock_snapshot1.exposure = MagicMock(spec=Exposures)
        self.mock_snapshot1.hazard = MagicMock(spec=Hazard)
        self.mock_snapshot1.impfset = MagicMock(spec=ImpactFuncSet)

        self.impact_calc_computation = ImpactCalcComputation()

    @patch.object(ImpactCalcComputation, "compute_impacts_pre_transfer")
    def test_compute_impacts(self, mock_calculate_impacts_for_snapshots):
        mock_impacts = MagicMock(spec=Impact)
        mock_calculate_impacts_for_snapshots.return_value = mock_impacts

        result = self.impact_calc_computation.compute_impacts(
            exp=self.mock_snapshot0.exposure,
            haz=self.mock_snapshot0.hazard,
            vul=self.mock_snapshot0.impfset,
        )

        self.assertEqual(result, mock_impacts)
        mock_calculate_impacts_for_snapshots.assert_called_once_with(
            self.mock_snapshot0.exposure,
            self.mock_snapshot0.hazard,
            self.mock_snapshot0.impfset,
        )

    def test_calculate_impacts_for_snapshots(self):
        mock_imp_E0H0 = MagicMock(spec=Impact)

        with patch(
            "climada.trajectories.impact_calc_strat.ImpactCalc"
        ) as mock_impact_calc:
            mock_impact_calc.return_value.impact.side_effect = [mock_imp_E0H0]

            result = self.impact_calc_computation.compute_impacts_pre_transfer(
                exp=self.mock_snapshot0.exposure,
                haz=self.mock_snapshot0.hazard,
                vul=self.mock_snapshot0.impfset,
            )

            self.assertEqual(result, mock_imp_E0H0)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestImpactCalcComputation)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
