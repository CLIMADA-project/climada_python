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

Integration tests for the measure pipeline:
MeasureConfig -> Measure -> modified (Exposures, ImpactFuncSet, Hazard) -> Impact.

Expected values follow the analytical design of ``conftest.py``
(linear impact function, baseline AAI = 18, at_event = [0, 0, 500, 3750, 0]).
"""

from unittest.mock import patch

import numpy as np
import pytest

from climada.engine.impact_calc import ImpactCalc
from climada.engine.test.test_impact_calc import exposure
from climada.entity.measures.measure import Measure
from climada.entity.measures.measure_config import (
    CostIncomeConfig,
    ExposuresModifierConfig,
    HazardModifierConfig,
    ImpfsetModifierConfig,
    MeasureConfig,
)
from climada.hazard.base import Hazard

LOGGER = "climada.entity.measures.measure.LOGGER"
BASELINE_AAI = 18.0

# Note: at_event() does not care about the frequency
BASELINE_AT_EVENT = [0.0, 0.0, 500.0, 3750.0, 15000.0]
CATEGORIES = [1, 1, 2, 1, 1, 3]


@pytest.fixture
def triplet(exposures_factory, impfset, hazard):
    """Return (exposures, impfset, hazard) with centroids assigned."""
    exposures = exposures_factory(categories=CATEGORIES)
    exposures.assign_centroids(hazard)
    return exposures, impfset, hazard


@pytest.fixture
def config_factory(hazard):
    """Return a function building a MeasureConfig from modifier overrides."""

    def _make_config(
        name="measure",
        impfset=None,
        hazard_mod=None,
        exposures_mod=None,
        cost_income=None,
    ):
        """
        Parameters
        ----------
        name : str, optional
            Measure name.
        impfset, hazard_mod, exposures_mod, cost_income : dict, optional
            Keyword arguments of the corresponding sub-configuration.

        Returns
        -------
        MeasureConfig
        """
        haz_type = hazard.haz_type
        return MeasureConfig(
            name=name,
            haz_type=haz_type,
            impfset_modifier=ImpfsetModifierConfig(
                haz_type=haz_type, **(impfset or {})
            ),
            hazard_modifier=HazardModifierConfig(
                haz_type=haz_type, **(hazard_mod or {})
            ),
            exposures_modifier=ExposuresModifierConfig(**(exposures_mod or {})),
            cost_income=CostIncomeConfig(**(cost_income or {})),
        )

    return _make_config


def _aai(measure, triplet, **kwargs):
    """Return the AAI of the triplet after applying `measure`."""
    return measure.calc_impact(*triplet, **kwargs).aai_agg


class TestBaseline:
    def test_identity_measure_reproduces_analytical_impact(self, triplet):
        impact = Measure("baseline").calc_impact(*triplet)
        assert impact.aai_agg == pytest.approx(BASELINE_AAI)
        np.testing.assert_allclose(impact.at_event, BASELINE_AT_EVENT)

    def test_default_config_is_identity(self, triplet, config_factory):
        measure = Measure.from_config(config_factory())
        assert _aai(measure, triplet) == pytest.approx(BASELINE_AAI)


class TestConfigModifiers:
    @pytest.mark.parametrize(
        "modifiers, expected_aai",
        [
            ({"hazard_mod": {"haz_int_mult": 0.5}}, 9.0),
            ({"hazard_mod": {"haz_freq_mult": 2.0}}, 36.0),
            (
                {"hazard_mod": {"haz_freq_add": -0.005}},
                0.5,
            ),  # Only event 3 with 0.001 freq
            ({"hazard_mod": {"haz_freq_add": -0.01}}, 0.0),  # clipped at zero
            ({"hazard_mod": {"impact_rp_cutoff": 200}}, 15.0),  # event 3 removed
            ({"impfset": {"impf_mdd_mult": 0.5}}, 9.0),
            ({"impfset": {"impf_paa_mult": 0.5}}, 9.0),
            ({"impfset": {"impf_int_mult": 0.5}}, 9.0),  # inverse intensity axis
            ({"exposures_mod": {"set_to_zero": [1]}}, 14.0),
            (
                {
                    "hazard_mod": {"haz_int_mult": 0.5},
                    "impfset": {"impf_mdd_mult": 0.5},
                },
                4.5,
            ),
        ],
        ids=[
            "haz_int_mult",
            "haz_freq_mult",
            "haz_freq_add",
            "haz_freq_add_clipped",
            "rp_cutoff",
            "mdd_mult",
            "paa_mult",
            "impf_int_mult",
            "set_to_zero",
            "haz_and_mdd",
        ],
    )
    def test_aai_after_modifier(self, triplet, config_factory, modifiers, expected_aai):
        measure = Measure.from_config(config_factory(**modifiers))
        assert _aai(measure, triplet) == pytest.approx(expected_aai)

    def test_rp_cutoff_keeps_event_below_threshold(self, triplet, config_factory):
        measure = Measure.from_config(
            config_factory(hazard_mod={"impact_rp_cutoff": 200})
        )
        impact = measure.calc_impact(*triplet)
        np.testing.assert_allclose(impact.at_event, [0.0, 0.0, 0.0, 3750.0, 15000.0])

    def test_rp_cutoff_keeps_event_below_threshold_edge(self, triplet, config_factory):
        measure = Measure.from_config(
            config_factory(hazard_mod={"impact_rp_cutoff": 250})
        )
        impact = measure.calc_impact(*triplet)
        np.testing.assert_allclose(impact.at_event, [0.0, 0.0, 0.0, 0.0, 15000.0])


class TestImpfIds:
    @pytest.mark.parametrize("impf_ids", [None, "all", 1, [1]])
    def test_targeted_function_is_modified(self, triplet, config_factory, impf_ids):
        measure = Measure.from_config(
            config_factory(impfset={"impf_mdd_mult": 0.5, "impf_ids": impf_ids})
        )
        assert _aai(measure, triplet) == pytest.approx(9.0)

    def test_other_function_leaves_impact_unchanged(self, triplet, config_factory):
        measure = Measure.from_config(
            config_factory(impfset={"impf_mdd_mult": 0.5, "impf_ids": [2]})
        )
        assert _aai(measure, triplet) == pytest.approx(BASELINE_AAI)

    def test_invalid_type_raises_value_error(self, triplet, config_factory):
        measure = Measure.from_config(config_factory(impfset={"impf_ids": 2.5}))
        with pytest.raises(ValueError, match="invalid"):
            measure.apply_impfset_changes(triplet[1])


class TestReplacementFromDisk:
    def test_hazard_replacement(
        self, triplet, config_factory, hazard_factory, tmp_path
    ):
        path = tmp_path / "hazard.hdf5"
        hazard_factory(intensity_scale=0.5).write_hdf5(str(path))

        measure = Measure.from_config(
            config_factory(hazard_mod={"new_hazard_path": str(path)})
        )
        assert _aai(measure, triplet) == pytest.approx(9.0)

    def test_hazard_replacement_then_modifier(
        self, triplet, config_factory, hazard_factory, tmp_path
    ):
        path = tmp_path / "hazard.hdf5"
        hazard_factory(intensity_scale=0.5).write_hdf5(str(path))

        with pytest.warns(UserWarning, match="Both new hazard"):
            config = config_factory(
                hazard_mod={"new_hazard_path": str(path), "haz_int_mult": 0.5}
            )
        assert _aai(Measure.from_config(config), triplet) == pytest.approx(4.5)

    def test_impfset_replacement(
        self, triplet, config_factory, impfset_factory, tmp_path
    ):
        path = tmp_path / "impfset.xlsx"
        impfset_factory(paa_scale=0.5).write_excel(str(path))

        measure = Measure.from_config(
            config_factory(impfset={"new_impfset_path": str(path)})
        )
        assert _aai(measure, triplet) == pytest.approx(9.0)

    def test_impfset_replacement_then_modifier(
        self, triplet, config_factory, impfset_factory, tmp_path
    ):
        path = tmp_path / "impfset.xlsx"
        impfset_factory(paa_scale=0.5).write_excel(str(path))

        with pytest.warns(UserWarning, match="Both new impfset"):
            config = config_factory(
                impfset={"new_impfset_path": str(path), "impf_mdd_mult": 0.5}
            )
        assert _aai(Measure.from_config(config), triplet) == pytest.approx(4.5)

    def test_exposures_replacement(
        self, triplet, config_factory, exposures_factory, tmp_path
    ):
        path = tmp_path / "exposures.hdf5"
        exposures_factory(categories=CATEGORIES, value_factor=0.5).write_hdf5(str(path))

        measure = Measure.from_config(
            config_factory(exposures_mod={"new_exposures_path": str(path)})
        )
        assert _aai(measure, triplet) == pytest.approx(9.0)

    @pytest.mark.parametrize(
        "override",
        [
            {"hazard_mod": {"new_hazard_path": "nil"}},
            {"impfset": {"new_impfset_path": "nil"}},
            {"exposures_mod": {"new_exposures_path": "nil"}},
        ],
        ids=["hazard", "impfset", "exposures"],
    )
    def test_nil_path_means_no_replacement(self, triplet, config_factory, override):
        measure = Measure.from_config(config_factory(**override))
        assert _aai(measure, triplet) == pytest.approx(BASELINE_AAI)


class TestCopySemantics:
    @pytest.fixture
    def measure(self, config_factory):
        return Measure.from_config(
            config_factory(
                hazard_mod={"haz_int_mult": 0.5},
                impfset={"impf_mdd_mult": 0.5},
                exposures_mod={"set_to_zero": [1]},
            )
        )

    def test_enforce_copy_preserves_inputs(self, measure, triplet):
        exposures, impfset, hazard = triplet
        intensity = hazard.intensity.toarray().copy()
        values = exposures.gdf["value"].to_numpy().copy()

        measure.apply(exposures, impfset, hazard, enforce_copy=True)

        np.testing.assert_array_equal(hazard.intensity.toarray(), intensity)
        np.testing.assert_array_equal(exposures.gdf["value"].to_numpy(), values)
        assert _aai(Measure("baseline"), triplet) == pytest.approx(BASELINE_AAI)

    def test_no_copy_modifies_inputs_in_place(self, measure, triplet):
        exposures, impfset, hazard = triplet
        intensity = hazard.intensity.toarray().copy()

        measure.apply(exposures, impfset, hazard, enforce_copy=False)

        np.testing.assert_allclose(hazard.intensity.toarray(), 0.5 * intensity)
        assert exposures.gdf["value"].iloc[1] == 0


class TestCalcImpact:
    def test_matches_manual_apply_and_impact_calc(self, triplet, config_factory):
        measure = Measure.from_config(
            config_factory(
                hazard_mod={"haz_int_mult": 0.5},
                impfset={"impf_paa_mult": 0.5},
            )
        )
        new_exp, new_impfset, new_haz = measure.apply(*triplet)
        manual = ImpactCalc(new_exp, new_impfset, new_haz).impact(save_mat=False)

        impact = measure.calc_impact(*triplet)

        assert impact.aai_agg == pytest.approx(manual.aai_agg)
        np.testing.assert_allclose(impact.at_event, manual.at_event)

    def test_warns_and_assigns_missing_centroids(
        self, exposures_factory, impfset, hazard
    ):
        exposures = exposures_factory(categories=CATEGORIES)
        with patch(LOGGER) as mock_logger:
            impact = Measure("baseline").calc_impact(exposures, impfset, hazard)

        mock_logger.warning.assert_called_once()
        assert hazard.centr_exp_col in exposures.gdf.columns
        assert impact.aai_agg == pytest.approx(BASELINE_AAI)

    def test_no_warning_when_centroids_assigned(self, triplet):
        with patch(LOGGER) as mock_logger:
            Measure("baseline").calc_impact(*triplet)

        mock_logger.warning.assert_not_called()


class TestCustomFunction:
    def test_custom_measure(self, measure_on_exposures, triplet):
        impact = measure_on_exposures.calc_impact(
            *triplet, kwargs_exposures={"reduction_factor": 0.1}
        )
        assert impact.aai_agg == pytest.approx(BASELINE_AAI * 0.9)

    def test_custom_measure_missing_kwarg(self, measure_on_exposures, triplet):
        with pytest.raises(TypeError, match="kwargs_exposures"):
            measure_on_exposures.calc_impact(*triplet)

    def test_wrong_custom_measure(self, triplet):
        def wrong_change(arg1):
            return "wrong return type"

        meas = Measure(name="", exposures_changes=wrong_change)
        with pytest.raises(TypeError, match="did not return an Exposures"):
            meas.apply(*triplet)

        meas = Measure(name="", hazard_changes=wrong_change)
        with pytest.raises(TypeError, match="did not return an Hazard"):
            meas.apply(*triplet)

        meas = Measure(name="", impfset_changes=wrong_change)
        with pytest.raises(TypeError, match="did not return an ImpactFuncSet"):
            meas.apply(*triplet)
