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

This modules implements the impact computation strategy objects for risk
trajectories.

"""

from abc import ABC, abstractmethod

from climada.engine.impact import Impact
from climada.engine.impact_calc import ImpactCalc
from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.hazard.base import Hazard

__all__ = ["ImpactCalcComputation"]


# The following is acceptable.
# We design a pattern, and currently it requires only to
# define the compute_impacts method.
# pylint: disable=too-few-public-methods
class ImpactComputationStrategy(ABC):
    """
    Interface for impact computation strategies.

    This abstract class defines the contract for all concrete strategies
    responsible for calculating and optionally modifying with a risk transfer,
    the impact computation, based on a set of inputs (exposure, hazard, vulnerability).

    It revolves around a `compute_impacts()` method that takes as arguments
    the three dimensions of risk (exposure, hazard, vulnerability) and return an
    Impact object.
    """

    @abstractmethod
    def compute_impacts(
        self,
        exp: Exposures,
        haz: Hazard,
        vul: ImpactFuncSet,
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

        Returns
        -------
        Impact
            An object containing the computed total impact matrix and metrics.

        See Also
        --------
        ImpactCalcComputation : The default implementation of this interface.
        """


class ImpactCalcComputation(ImpactComputationStrategy):
    r"""
    Default impact computation strategy using the core engine of climada.

    This strategy first calculates the raw impact using the standard
    :class:`ImpactCalc` logic.

    """

    def compute_impacts(
        self,
        exp: Exposures,
        haz: Hazard,
        vul: ImpactFuncSet,
    ) -> Impact:
        """
        Calculates the impact.

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
            The final impact object.
        """
        return ImpactCalc(exposures=exp, impfset=vul, hazard=haz).impact()
