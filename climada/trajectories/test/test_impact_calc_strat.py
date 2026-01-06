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

from unittest.mock import MagicMock, patch

import pytest

from climada.engine import Impact
from climada.entity import ImpactFuncSet
from climada.entity.exposures import Exposures
from climada.hazard import Hazard
from climada.trajectories import Snapshot
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)

# --- Fixtures ---


@pytest.fixture
def mock_snapshot():
    """Provides a snapshot with mocked exposure, hazard, and impact functions."""
    snap = MagicMock(spec=Snapshot)
    snap.exposure = MagicMock(spec=Exposures)
    snap.hazard = MagicMock(spec=Hazard)
    snap.impfset = MagicMock(spec=ImpactFuncSet)
    return snap


@pytest.fixture
def strategy():
    """Provides an instance of the ImpactCalcComputation strategy."""
    return ImpactCalcComputation()


# --- Tests ---
def test_interface_compliance(strategy):
    """Ensure the class correctly inherits from the Abstract Base Class."""
    assert isinstance(strategy, ImpactComputationStrategy)
    assert isinstance(strategy, ImpactCalcComputation)


def test_compute_impacts(strategy, mock_snapshot):
    """Test that compute_impacts calls the pre-transfer method correctly."""
    mock_impacts = MagicMock(spec=Impact)

    # We patch the ImpactCalc within trajectories
    with patch("climada.trajectories.impact_calc_strat.ImpactCalc") as mock_ImpactCalc:
        mock_ImpactCalc.return_value.impact.return_value = mock_impacts
        result = strategy.compute_impacts(
            exp=mock_snapshot.exposure,
            haz=mock_snapshot.hazard,
            vul=mock_snapshot.impfset,
        )
        mock_ImpactCalc.assert_called_once_with(
            exposures=mock_snapshot.exposure,
            impfset=mock_snapshot.impfset,
            hazard=mock_snapshot.hazard,
        )
        mock_ImpactCalc.return_value.impact.assert_called_once()
        assert result == mock_impacts


def test_cannot_instantiate_abstract_base_class():
    """Ensure ImpactComputationStrategy cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        ImpactComputationStrategy()  # type: ignore


@pytest.mark.parametrize("invalid_input", [None, 123, "string"])
def test_compute_impacts_type_errors(strategy, invalid_input):
    """
    Smoke test: Ensure that if ImpactCalc raises errors due to bad input,
    the strategy correctly propagates them.
    """
    with pytest.raises(AttributeError):
        strategy.compute_impacts(invalid_input, invalid_input, invalid_input)
