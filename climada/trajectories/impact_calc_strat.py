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

This modules implements the Snapshot and SnapshotsCollection classes.

"""

import copy
from abc import ABC, abstractmethod

import numpy as np

from climada.engine.impact import Impact
from climada.engine.impact_calc import ImpactCalc
from climada.trajectories.snapshot import Snapshot


class ImpactComputationStrategy(ABC):
    """Interface for impact computation strategies."""

    @abstractmethod
    def compute_impacts(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        risk_transf_attach: float | None,
        risk_transf_cover: float | None,
        calc_residual: bool,
    ) -> tuple:
        pass


class ImpactCalcComputation(ImpactComputationStrategy):
    """Default impact computation strategy."""

    def compute_impacts(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        risk_transf_attach: float | None,
        risk_transf_cover: float | None,
        calc_residual: bool = False,
    ):
        impacts = self._calculate_impacts_for_snapshots(snapshot0, snapshot1)
        self._apply_risk_transfer(
            impacts, risk_transf_attach, risk_transf_cover, calc_residual
        )
        return impacts

    def _calculate_impacts_for_snapshots(
        self, snapshot0: Snapshot, snapshot1: Snapshot
    ):
        """Calculate impacts for the given snapshots and impact function set."""
        imp_E0H0 = ImpactCalc(
            snapshot0.exposure, snapshot0.impfset, snapshot0.hazard
        ).impact()
        imp_E1H0 = ImpactCalc(
            snapshot1.exposure, snapshot1.impfset, snapshot0.hazard
        ).impact()
        imp_E0H1 = ImpactCalc(
            snapshot0.exposure, snapshot0.impfset, snapshot1.hazard
        ).impact()
        imp_E1H1 = ImpactCalc(
            snapshot1.exposure, snapshot1.impfset, snapshot1.hazard
        ).impact()
        return imp_E0H0, imp_E1H0, imp_E0H1, imp_E1H1

    def _apply_risk_transfer(
        self,
        impacts: tuple[Impact, Impact, Impact, Impact],
        risk_transf_attach: float | None,
        risk_transf_cover: float | None,
        calc_residual: bool,
    ):
        """Apply risk transfer to the calculated impacts."""
        if risk_transf_attach is not None and risk_transf_cover is not None:
            for imp in impacts:
                imp.imp_mat = self.calculate_residual_or_risk_transfer_impact_matrix(
                    imp.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
                )

    def calculate_residual_or_risk_transfer_impact_matrix(
        self, imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
    ):
        """
        Calculate either the residual or the risk transfer impact matrix.

        The impact matrix is adjusted based on the total impact for each event.
        When calculating the residual impact, the result is the total impact minus
        the risk layer. The risk layer is defined as the minimum of the cover and
        the maximum of the difference between the total impact and the attachment.
        If `calc_residual` is False, the function returns the risk layer matrix
        instead of the residual.

        Parameters
        ----------
        imp_mat : scipy.sparse.csr_matrix
            The original impact matrix to be scaled.
        attachment : float, optional
            The attachment point for the risk layer.
        cover : float, optional
            The maximum coverage for the risk layer.
        calc_residual : bool, default=True
            Determines if the function calculates the residual (if True) or the
            risk layer (if False).

        Returns
        -------
        scipy.sparse.csr_matrix
            The adjusted impact matrix, either residual or risk transfer.

        """

        if risk_transf_attach and risk_transf_cover:
            imp_mat = copy.deepcopy(imp_mat)
            # Calculate the total impact per event
            total_at_event = imp_mat.sum(axis=1).A1
            # Risk layer at event
            transfer_at_event = np.minimum(
                np.maximum(total_at_event - risk_transf_attach, 0), risk_transf_cover
            )
            residual_at_event = np.maximum(total_at_event - transfer_at_event, 0)

            # Calculate either the residual or transfer impact matrix
            # Choose the denominator to rescale the impact values
            if calc_residual:
                numerator = residual_at_event
            else:
                numerator = transfer_at_event

            rescale_impact_values = np.divide(
                numerator,
                total_at_event,
                out=np.zeros_like(numerator, dtype=float),
                where=total_at_event != 0,
            )

            # The multiplication is broadcasted across the columns for each row
            result_matrix = imp_mat.multiply(rescale_impact_values[:, np.newaxis])

            return result_matrix

        else:

            return imp_mat
