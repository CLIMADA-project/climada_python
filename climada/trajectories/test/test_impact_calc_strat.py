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

import numpy as np
from scipy.sparse import csr_matrix

from climada.trajectories.impact_calc_strat import (
    Impact,
    ImpactCalcComputation,
    Snapshot,
)


class TestImpactCalcComputation(unittest.TestCase):
    def setUp(self):
        self.mock_snapshot0 = MagicMock(spec=Snapshot)
        self.mock_snapshot1 = MagicMock(spec=Snapshot)

        self.impact_calc_computation = ImpactCalcComputation()

    @patch.object(ImpactCalcComputation, "_calculate_impacts_for_snapshots")
    @patch.object(ImpactCalcComputation, "_apply_risk_transfer")
    def test_compute_impacts(
        self, mock_apply_risk_transfer, mock_calculate_impacts_for_snapshots
    ):
        mock_impacts = (
            MagicMock(spec=Impact),
            MagicMock(spec=Impact),
            MagicMock(spec=Impact),
            MagicMock(spec=Impact),
        )
        mock_calculate_impacts_for_snapshots.return_value = mock_impacts

        result = self.impact_calc_computation.compute_impacts(
            self.mock_snapshot0, self.mock_snapshot1, 0.1, 0.9, False
        )

        self.assertEqual(result, mock_impacts)
        mock_calculate_impacts_for_snapshots.assert_called_once_with(
            self.mock_snapshot0, self.mock_snapshot1
        )
        mock_apply_risk_transfer.assert_called_once_with(mock_impacts, 0.1, 0.9, False)

    def test_calculate_impacts_for_snapshots(self):
        mock_imp_E0H0 = MagicMock(spec=Impact)
        mock_imp_E1H0 = MagicMock(spec=Impact)
        mock_imp_E0H1 = MagicMock(spec=Impact)
        mock_imp_E1H1 = MagicMock(spec=Impact)

        with patch(
            "climada.trajectories.impact_calc_strat.ImpactCalc"
        ) as mock_impact_calc:
            mock_impact_calc.return_value.impact.side_effect = [
                mock_imp_E0H0,
                mock_imp_E1H0,
                mock_imp_E0H1,
                mock_imp_E1H1,
            ]

            result = self.impact_calc_computation._calculate_impacts_for_snapshots(
                self.mock_snapshot0, self.mock_snapshot1
            )

            self.assertEqual(
                result, (mock_imp_E0H0, mock_imp_E1H0, mock_imp_E0H1, mock_imp_E1H1)
            )

    def test_apply_risk_transfer(self):
        mock_imp_E0H0 = MagicMock(spec=Impact)
        mock_imp_E0H0.imp_mat = MagicMock(spec=csr_matrix)

        mock_imp_E1H0 = MagicMock(spec=Impact)
        mock_imp_E1H0.imp_mat = MagicMock(spec=csr_matrix)

        mock_imp_E0H1 = MagicMock(spec=Impact)
        mock_imp_E0H1.imp_mat = MagicMock(spec=csr_matrix)

        mock_imp_E1H1 = MagicMock(spec=Impact)
        mock_imp_E1H1.imp_mat = MagicMock(spec=csr_matrix)

        mock_impacts = (mock_imp_E0H0, mock_imp_E1H0, mock_imp_E0H1, mock_imp_E1H1)

        mock_imp_resi = MagicMock(spec=csr_matrix)

        with patch.object(
            self.impact_calc_computation,
            "calculate_residual_or_risk_transfer_impact_matrix",
        ) as mock_calc_risk_transfer:
            mock_calc_risk_transfer.return_value = mock_imp_resi
            self.impact_calc_computation._apply_risk_transfer(
                mock_impacts, 0.1, 0.9, False
            )

            self.assertIs(mock_impacts[0].imp_mat, mock_imp_resi)
            self.assertIs(mock_impacts[1].imp_mat, mock_imp_resi)
            self.assertIs(mock_impacts[2].imp_mat, mock_imp_resi)
            self.assertIs(mock_impacts[3].imp_mat, mock_imp_resi)

    def test_calculate_residual_or_risk_transfer_impact_matrix(self):
        imp_mat = MagicMock()
        imp_mat.sum.return_value.A1 = np.array([100, 200, 300])
        imp_mat.multiply.return_value = "rescaled_matrix"

        result = self.impact_calc_computation.calculate_residual_or_risk_transfer_impact_matrix(
            imp_mat, 0.1, 0.9, True
        )
        self.assertEqual(result, "rescaled_matrix")

        result = self.impact_calc_computation.calculate_residual_or_risk_transfer_impact_matrix(
            imp_mat, 0.1, 0.9, False
        )
        self.assertEqual(result, "rescaled_matrix")


if __name__ == "__main__":
    unittest.main()
