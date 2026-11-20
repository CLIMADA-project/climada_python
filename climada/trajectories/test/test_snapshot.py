import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFunc, ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.hazard import Hazard
from climada.trajectories.snapshot import Snapshot
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5

# --- Fixtures ---


@pytest.fixture(scope="module")
def shared_data():
    """Load heavy HDF5 data once per module to speed up tests."""
    exposure = Exposures.from_hdf5(EXP_DEMO_H5)
    hazard = Hazard.from_hdf5(HAZ_DEMO_H5)
    impfset = ImpactFuncSet(
        [
            ImpactFunc(
                "TC",
                3,
                intensity=np.array([0, 20]),
                mdd=np.array([0, 0.5]),
                paa=np.array([0, 1]),
            )
        ]
    )
    return exposure, hazard, impfset


@pytest.fixture
def mock_context(shared_data):
    """Provides the exposure/hazard/impfset and a pre-configured mock measure."""
    exp, haz, impf = shared_data

    # Setup Mock Measure
    mock_measure = MagicMock(spec=Measure)
    mock_measure.name = "Test Measure"

    modified_exp = MagicMock(spec=Exposures)
    modified_haz = MagicMock(spec=Hazard)
    modified_imp = MagicMock(spec=ImpactFuncSet)

    mock_measure.apply.return_value = (modified_exp, modified_imp, modified_haz)

    return {
        "exp": exp,
        "haz": haz,
        "imp": impf,
        "measure": mock_measure,
        "mod_exp": modified_exp,
        "mod_haz": modified_haz,
        "mod_imp": modified_imp,
        "date": pd.Timestamp("2023"),
    }


# --- Tests ---


@pytest.mark.parametrize(
    "input_date,expected",
    [
        ("2023", pd.Timestamp(2023, 1, 1)),
        ("2023-01-01", pd.Timestamp(2023, 1, 1)),
        (np.datetime64("2023-01-01"), pd.Timestamp(2023, 1, 1)),
        (datetime.date(2023, 1, 1), pd.Timestamp(2023, 1, 1)),
        (pd.Timestamp(2023, 1, 1), pd.Timestamp(2023, 1, 1)),
    ],
)
def test_init_valid_dates(mock_context, input_date, expected):
    """Test various valid date input formats using parametrization."""
    snapshot = Snapshot(
        exposure=mock_context["exp"],
        hazard=mock_context["haz"],
        impfset=mock_context["imp"],
        date=input_date,
    )
    assert snapshot.date == expected


def test_init_invalid_date_format(mock_context):
    with pytest.raises(ValueError, match=r"String must be in a valid date format"):
        Snapshot(
            exposure=mock_context["exp"],
            hazard=mock_context["haz"],
            impfset=mock_context["imp"],
            date="invalid-date",
        )


def test_init_invalid_date_type(mock_context):
    with pytest.raises(
        TypeError,
        match=r"Unsupported type",
    ):
        Snapshot(
            exposure=mock_context["exp"],
            hazard=mock_context["haz"],
            impfset=mock_context["imp"],
            date=2023.5,  # type: ignore
        )


def test_properties(mock_context):
    snapshot = Snapshot(
        exposure=mock_context["exp"],
        hazard=mock_context["haz"],
        impfset=mock_context["imp"],
        date=mock_context["date"],
    )

    # Check that it's a deep copy (new reference)
    assert snapshot.exposure is not mock_context["exp"]
    assert snapshot.hazard is not mock_context["haz"]

    assert snapshot.measure is None

    # Check data equality
    pd.testing.assert_frame_equal(snapshot.exposure.gdf, mock_context["exp"].gdf)
    assert snapshot.hazard.haz_type == mock_context["haz"].haz_type
    assert snapshot.impfset == mock_context["imp"]
    assert snapshot.date == mock_context["date"]


def test_reference(mock_context):
    snapshot = Snapshot(
        exposure=mock_context["exp"],
        hazard=mock_context["haz"],
        impfset=mock_context["imp"],
        date=mock_context["date"],
        ref_only=True,
    )

    # Check that it is a reference
    assert snapshot.exposure is mock_context["exp"]
    assert snapshot.hazard is mock_context["haz"]
    assert snapshot.impfset is mock_context["imp"]
    assert snapshot.measure is None


def test_apply_measure(mock_context):
    snapshot = Snapshot(
        exposure=mock_context["exp"],
        hazard=mock_context["haz"],
        impfset=mock_context["imp"],
        date=mock_context["date"],
    )
    new_snapshot = snapshot.apply_measure(mock_context["measure"])

    assert new_snapshot.measure is not None
    assert new_snapshot.measure.name == "Test Measure"
    assert new_snapshot.exposure == mock_context["mod_exp"]
    assert new_snapshot.hazard == mock_context["mod_haz"]
    assert new_snapshot.impfset == mock_context["mod_imp"]
    assert new_snapshot.date == mock_context["date"]
