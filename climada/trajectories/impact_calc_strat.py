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
        future: tuple[int, int, int],
        risk_transf_attach: float | None,
        risk_transf_cover: float | None,
        calc_residual: bool,
    ) -> Impact:
        pass


class ImpactCalcComputation(ImpactComputationStrategy):
    """Default impact computation strategy."""

    def compute_impacts(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        future: tuple[int, int, int],
        risk_transf_attach: float | None,
        risk_transf_cover: float | None,
        calc_residual: bool = False,
    ):
        impact = self.compute_impacts_pre_transfer(snapshot0, snapshot1, future)
        self._apply_risk_transfer(
            impact, risk_transf_attach, risk_transf_cover, calc_residual
        )
        return impact

    def compute_impacts_pre_transfer(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        future: tuple[int, int, int],
    ) -> Impact:
        exp = snapshot1.exposure if future[0] else snapshot0.exposure
        haz = snapshot1.hazard if future[1] else snapshot0.hazard
        vul = snapshot1.impfset if future[2] else snapshot0.impfset
        return ImpactCalc(exposures=exp, impfset=vul, hazard=haz).impact()

    def _apply_risk_transfer(
        self,
        impact: Impact,
        risk_transf_attach: float | None,
        risk_transf_cover: float | None,
        calc_residual: bool,
    ):
        """Apply risk transfer to the calculated impacts."""
        if risk_transf_attach is not None and risk_transf_cover is not None:
            impact.imp_mat = self.calculate_residual_or_risk_transfer_impact_matrix(
                impact.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
            )

    def calculate_residual_or_risk_transfer_impact_matrix(
        self, imp_mat, attachement, cover, calc_residual
    ):
        """Calculate either the residual or the risk transfer impact matrix
        from a global risk transfer mechanism.

        To compute the transfered risk, the function first computes for each event,
        the (positive) difference between their total impact and `attachment`.
        The transfered risk for each event is then defined as the minimum between
        this value and `cover`. The residual impact is the total impact minus
        the transfered risk.
        The impact matrix is then adjusted by multiply impact per centroids
        by the ratio between residual risk and total impact for each event.
        As such, the risk transfer is shared among all impacted exposure points
        and equaly distributed.

        If `calc_residual` is False, the function returns the transfered risk
        at each points instead of the residual risk by using the ratio between
        transfered risk and total impact instead.

        Parameters
        ----------
        imp_mat : scipy.sparse.csr_matrix
            The original impact matrix to be scaled.
        attachment : float
            The attachment point for the risk layer.
        cover : float
            The maximum coverage for the risk layer.
        calc_residual : bool, default=True
            Determines if the function calculates the residual (if True) or the
            risk layer (if False).

        Returns
        -------
        scipy.sparse.csr_matrix
            The adjusted impact matrix, either residual or risk transfer.

        Warnings
        --------

        This transfer capability is different and not exclusive with the one
        implemented in ImpactCalc, which is defined at the centroid level.
        The mechanism here corresponds to a global cover applied to the whole
        region studied.

        """
        imp_mat = copy.deepcopy(imp_mat)
        # Calculate the total impact per event
        total_at_event = imp_mat.sum(axis=1).A1
        # Risk layer at event
        transfer_at_event = np.minimum(
            np.maximum(total_at_event - attachement, 0), cover
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
