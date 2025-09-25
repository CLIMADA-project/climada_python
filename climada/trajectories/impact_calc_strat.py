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

This modules implements the impact computation strategy objects for trajectories.

"""

import copy
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from scipy import sparse

from climada.engine.impact import Impact
from climada.engine.impact_calc import ImpactCalc
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard


class ImpactComputationStrategy(ABC):
    """
    Interface for impact computation strategies.

    This abstract class defines the contract for all concrete strategies
    responsible for calculating and optionally modifying the total impact
    based on a set of inputs (exposure, hazard, vulnerability).
    """

    @abstractmethod
    def compute_impacts(
        self,
        exp: Exposures,
        haz: Hazard,
        vul: ImpactFuncSet,
        risk_transf_attach: Optional[float] = None,
        risk_transf_cover: Optional[float] = None,
        calc_residual: bool = True,
    ) -> Impact:
        """
        Calculates the total impact, including optional risk transfer application.

        Parameters
        ----------
        exp : Exposures
            The exposure data.
        haz : Hazard
            The hazard data (e.g., event intensity).
        vul : ImpactFuncSet
            The set of vulnerability functions.
        risk_transf_attach : float, optional
            The attachment point (deductible) for the global risk transfer mechanism.
            If None, default to 0.
        risk_transf_cover : float, optional
            The cover limit for the risk transfer mechanism. If None, the cover
            is assumed to be infinite (only the attachment applies).
        calc_residual : bool, default=True
            If True, the function returns the residual impact (after risk transfer).
            If False, it returns the transferred impact (the part covered by the
            risk transfer).

        Returns
        -------
        Impact
            An object containing the computed total impact matrix and metrics.

        See Also
        --------
        ImpactCalcComputation : The default implementation of this interface.
        """
        pass


class ImpactCalcComputation(ImpactComputationStrategy):
    """
    Default impact computation strategy.

    This strategy first calculates the raw impact using the standard
    :class:`ImpactCalc` logic and then applies a global risk transfer mechanism.
    """

    def compute_impacts(
        self,
        exp: Exposures,
        haz: Hazard,
        vul: ImpactFuncSet,
        risk_transf_attach: Optional[float] = None,
        risk_transf_cover: Optional[float] = None,
        calc_residual: bool = False,
    ) -> Impact:
        """
        Calculates the impact and applies the risk transfer mechanism.

        This overrides the abstract method to implement the default strategy.

        Parameters
        ----------
        exp : Exposures
            The exposure data.
        haz : Hazard
            The hazard data.
        vul : ImpactFuncSet
            The set of vulnerability functions.
        risk_transf_attach : float, optional
            The attachment point (deductible) for the risk transfer mechanism.
        risk_transf_cover : float, optional
            The cover limit for the risk transfer mechanism.
        calc_residual : bool, default=False
            If True, returns the residual impact. If False, returns the transferred impact.

        Returns
        -------
        Impact
            The final impact object (either residual or transferred).
        """
        impact = self.compute_impacts_pre_transfer(exp, haz, vul)
        self._apply_risk_transfer(
            impact, risk_transf_attach, risk_transf_cover, calc_residual
        )
        return impact

    def compute_impacts_pre_transfer(
        self,
        exp: Exposures,
        haz: Hazard,
        vul: ImpactFuncSet,
    ) -> Impact:
        """
        Calculates the raw impact matrix before any risk transfer is applied.

        Parameters
        ----------
        exp : Exposures
            The exposure data.
        haz : Hazard
            The hazard data.
        vul : ImpactFuncSet
            The set of vulnerability functions.

        Returns
        -------
        Impact
            An Impact object containing the raw, pre-transfer impact matrix.
        """
        return ImpactCalc(exposures=exp, impfset=vul, hazard=haz).impact()

    def _apply_risk_transfer(
        self,
        impact: Impact,
        risk_transf_attach: Optional[float],
        risk_transf_cover: Optional[float],
        calc_residual: bool,
    ) -> None:
        """
        Applies risk transfer logic and modifies the Impact object in-place.

        Parameters
        ----------
        impact : Impact
            The Impact object whose impact matrix will be modified.
        risk_transf_attach : float, optional
            The attachment point.
        risk_transf_cover : float, optional
            The cover limit.
        calc_residual : bool
            Determines whether to set the matrix to the residual or transferred impact.
        """
        if risk_transf_attach is not None or risk_transf_cover is not None:
            # Assuming impact.imp_mat is a sparse matrix, which is a common pattern
            impact.imp_mat = self.calculate_residual_or_risk_transfer_impact_matrix(
                impact.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
            )

    def calculate_residual_or_risk_transfer_impact_matrix(
        self,
        imp_mat: Union[
            sparse.csr_matrix, Any
        ],  # Use Any if sparse.csr_matrix is too restrictive
        attachement: Optional[float],
        cover: Optional[float],
        calc_residual: bool,
    ) -> Union[sparse.csr_matrix, Any]:
        """
        Calculates either the residual or the risk transfer impact matrix
        based on a global risk transfer mechanism.

        This function modifies the original impact matrix values proportionally
        based on the total event impact relative to the attachment and cover.

        Parameters
        ----------
        imp_mat : scipy.sparse.csr_matrix or object with .data
            The original impact matrix (events x exposure points).
        attachement : float, optional
            The attachment point (deductible).
        cover : float, optional
            The cover limit.
        calc_residual : bool
            If True, the function returns the residual impact matrix.
            If False, it returns the transferred risk impact matrix.

        Returns
        -------
        scipy.sparse.csr_matrix or object with .data
            The adjusted impact matrix (residual or transferred).

        Notes
        -----
        The calculation is performed event-wise:

        1. **Total Impact**: Calculate the total impact for each event
           (sum of impacts across all exposure points).
        2. **Transferred Risk per Event**: Defined as:
           $$\min(\max(0, \text{Total Impact} - \text{attachement}), \text{cover})$$
        3. **Residual Risk per Event**:
           $$\text{Total Impact} - \text{Transferred Risk per Event}$$
        4. **Adjustment**: The original impact per exposure point is scaled
           by the ratio of (Residual Risk / Total Impact) or
           (Transferred Risk / Total Impact) for that event.
           This ensures the risk transfer is shared proportionally among all
           impacted exposure points.
        """
        imp_mat = copy.deepcopy(imp_mat)
        # Calculate the total impact per event
        total_at_event = imp_mat.sum(axis=1).A1
        # Risk layer at event
        attachement = 0 if attachement is None else attachement
        cover = np.inf if cover is None else cover
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
