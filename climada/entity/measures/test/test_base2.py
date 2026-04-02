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

Test Measure classes.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from climada.entity.measures.base import Measure
from climada.entity.measures.helper import (
    helper_exposure,
    helper_hazard,
    helper_impfset,
)


def test_measure_init(hazard, exposures, impfset):
    """Test if Measure initializes with default identity functions."""
    meas = Measure(name="test_meas")
    assert meas.name == "test_meas"
    assert meas.measure_effects(exposures, impfset, hazard) == (
        exposures,
        impfset,
        hazard,
    )


def test_apply_enforce_copy(hazard, exposures, impfset):
    """Verify that the original object is not modified when enforce_copy=True."""

    def double_intensity(haz):
        haz.intensity *= 2
        return haz

    def double_mdd(impfset):
        impfset.get_func(haz_type="TEST_HAZARD_TYPE", fun_id=1).mdd *= 2
        return impfset

    def double_value(exp):
        exp.gdf["value"] *= 2
        return exp

    meas = Measure(
        "scale",
        hazard_change=double_intensity,
        impfset_change=double_mdd,
        exposures_change=double_value,
    )
    original_haz = hazard.intensity.toarray().copy()
    original_impf = impfset.get_func(haz_type="TEST_HAZARD_TYPE", fun_id=1).mdd.copy()
    original_exp = exposures.gdf["value"].to_numpy().copy()

    transformed_haz = meas.apply_to_hazard(hazard, enforce_copy=True)
    transformed_impfset = meas.apply_to_impfset(impfset, enforce_copy=True)
    transformed_exp = meas.apply_to_exposures(exposures, enforce_copy=True)

    assert np.array_equal(hazard.intensity.toarray(), original_haz)
    assert np.array_equal(transformed_haz.intensity.toarray(), original_haz * 2)

    assert np.array_equal(
        impfset.get_func(haz_type="TEST_HAZARD_TYPE", fun_id=1).mdd, original_impf
    )
    assert np.array_equal(
        transformed_impfset.get_func(haz_type="TEST_HAZARD_TYPE", fun_id=1).mdd,
        original_impf * 2,
    )

    assert np.array_equal(exposures.gdf["value"].to_numpy(), original_exp)
    assert np.array_equal(transformed_exp.gdf["value"], original_exp * 2)


def test_apply_to_all(exposures, impfset, hazard):
    """Test the bulk apply method."""
    meas = Measure("identity")
    result = meas.apply(exposures, impfset, hazard)

    assert "exposure" in result
    assert "hazard" in result
    assert "impfset" in result
