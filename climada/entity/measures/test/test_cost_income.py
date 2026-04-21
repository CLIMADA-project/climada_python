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

Unit tests for the CostIncome class.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from climada.entity.measures.cost_income import CostIncome


class TestInit:
    def test_defaults(self):
        ci = CostIncome()
        assert ci.init_cost == 0.0
        assert ci.periodic_cost == 0.0
        assert ci.periodic_income == 0.0
        assert ci.cost_growth_rate == 0.0
        assert ci.income_growth_rate == 0.0
        assert ci.freq == "Y"
        assert ci.custom_cash_flows is None

    def test_costs_stored_negative(self):
        ci = CostIncome(init_cost=100, periodic_cost=50)
        assert ci.init_cost == -100.0
        assert ci.periodic_cost == -50.0

    def test_costs_already_negative_stay_negative(self):
        ci = CostIncome(init_cost=-200, periodic_cost=-30)
        assert ci.init_cost == -200.0
        assert ci.periodic_cost == -30.0

    def test_income_stored_positive(self):
        ci = CostIncome(periodic_income=-80)
        assert ci.periodic_income == 80.0

    def test_mkt_price_year_default_is_current_year(self):
        ci = CostIncome()
        assert ci.mkt_price_year.year == datetime.today().year

    def test_mkt_price_year_custom(self):
        ci = CostIncome(mkt_price_year=2015)
        assert ci.mkt_price_year.year == 2015

    def test_custom_cash_flows_processed(self):
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-06-01"],
                "cost": [100, 200],
                "income": [50, 60],
            }
        )
        ci = CostIncome(custom_cash_flows=df, freq="Y")
        # After resampling to yearly, should have one row per year
        assert isinstance(ci.custom_cash_flows, pd.DataFrame)
        assert "cost" in ci.custom_cash_flows.columns
        # Costs should be negative after processing
        assert (ci.custom_cash_flows["cost"] <= 0).all()


# ---------------------------------------------------------------------------
# from_dict / from_config / from_yaml
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_basic(self):
        d = {
            "mkt_price_year": 2020,
            "init_cost": 500,
            "periodic_cost": 100,
            "periodic_income": 200,
            "cost_yearly_growth_rate": 0.02,
            "income_yearly_growth_rate": 0.03,
            "freq": "Y",
        }
        ci = CostIncome.from_dict(d)
        assert ci.init_cost == -500.0
        assert ci.periodic_income == 200.0
        assert ci.cost_growth_rate == 0.02

    def test_defaults_for_missing_keys(self):
        ci = CostIncome.from_dict({})
        assert ci.init_cost == 0.0
        assert ci.freq == "Y"

    def test_with_custom_cash_flows(self):
        d = {
            "freq": "Y",
            "custom_cash_flows": [
                {"date": "2021-01-01", "cost": 100, "income": 50},
            ],
        }
        ci = CostIncome.from_dict(d)
        assert ci.custom_cash_flows is not None


class TestFromYaml:
    def test_from_yaml(self, tmp_path):
        yaml_content = """
cost_income:
  mkt_price_year: 2020
  init_cost: 1000
  periodic_cost: 200
  periodic_income: 300
  cost_yearly_growth_rate: 0.01
  income_yearly_growth_rate: 0.02
  freq: Y
"""
        p = tmp_path / "ci.yaml"
        p.write_text(yaml_content)
        ci = CostIncome.from_yaml(str(p))
        assert ci.init_cost == -1000.0
        assert ci.periodic_income == 300.0


# ---------------------------------------------------------------------------
# _freq_to_days
# ---------------------------------------------------------------------------


class TestFreqToDays:
    def test_yearly(self):
        result = CostIncome._freq_to_days("Y")
        assert result == "365d"

    def test_monthly(self):
        result = CostIncome._freq_to_days("M")
        assert result == "30d"

    def test_daily(self):
        result = CostIncome._freq_to_days("D")
        assert result == "1d"

    def test_invalid(self):
        with pytest.raises(ValueError):
            CostIncome._freq_to_days("INVALID_FREQ_XYZ")


# ---------------------------------------------------------------------------
# _get_width_days
# ---------------------------------------------------------------------------


class TestGetWidthDays:
    def test_yearly(self):
        ci = CostIncome(freq="Y")
        assert ci._get_width_days() == 365.0

    def test_3yearly(self):
        ci = CostIncome(freq="3Y")
        assert ci._get_width_days() == 3 * 365.0

    def test_monthly(self):
        ci = CostIncome(freq="M")
        assert ci._get_width_days() == 30.0

    def test_daily(self):
        ci = CostIncome(freq="D")
        assert ci._get_width_days() == 1.0


# ---------------------------------------------------------------------------
# _calc_at_date
# ---------------------------------------------------------------------------


class TestCalcAtDate:
    def test_before_impl_date_is_zero(self):
        ci = CostIncome(mkt_price_year=2020, init_cost=1000, periodic_income=500)
        impl = pd.Timestamp("2021-01-01")
        curr = pd.Timestamp("2020-01-01")
        net, cost, inc = ci.calc_at_date(impl, curr)
        assert net == 0.0
        assert cost == 0.0
        assert inc == 0.0

    def test_at_impl_date_uses_init_cost(self):
        ci = CostIncome(mkt_price_year=2021, init_cost=1000, periodic_income=0)
        impl = pd.Timestamp("2021-01-01")
        net, cost, inc = ci.calc_at_date(impl, impl)
        assert cost == pytest.approx(-1000.0, rel=1e-3)
        assert inc == pytest.approx(0.0)

    def test_after_impl_date_uses_periodic_cost(self):
        ci = CostIncome(mkt_price_year=2020, periodic_cost=200, periodic_income=0)
        impl = pd.Timestamp("2021-01-01")
        curr = pd.Timestamp("2022-01-01")
        net, cost, inc = ci.calc_at_date(impl, curr)
        assert cost < 0
        assert abs(cost) == 200

    def test_income_growth_applied(self):
        ci = CostIncome(
            mkt_price_year=2020, periodic_income=100, income_yearly_growth_rate=0.10
        )
        impl = pd.Timestamp("2020-01-01")
        curr = pd.Timestamp("2021-01-01")
        _, _, inc = ci.calc_at_date(impl, curr)
        expected = 100 * (1.10**1.0)
        assert inc == pytest.approx(expected, rel=1e-2)

    def test_net_equals_income_plus_cost(self):
        ci = CostIncome(mkt_price_year=2020, periodic_cost=100, periodic_income=150)
        impl = pd.Timestamp("2020-01-01")
        curr = pd.Timestamp("2021-01-01")
        net, cost, inc = ci.calc_at_date(impl, curr)
        assert net == pytest.approx(cost + inc)

    def test_custom_cash_flows_added(self):
        df = pd.DataFrame(
            {
                "date": ["2021-01-01"],
                "cost": [500.0],
                "income": [200.0],
            }
        )
        ci = CostIncome(mkt_price_year=2021, custom_cash_flows=df, freq="Y")
        impl = pd.Timestamp("2021-01-01")
        curr = pd.Timestamp("2021-01-01")
        net, cost, inc = ci.calc_at_date(impl, curr)
        assert inc == pytest.approx(200.0, rel=1e-3)
        assert cost == pytest.approx(-500.0, rel=1e-3)


# ---------------------------------------------------------------------------
# calc_cash_flows
# ---------------------------------------------------------------------------


class TestCalcCashFlows:
    def test_returns_three_arrays(self):
        ci = CostIncome(mkt_price_year=2020, periodic_income=100)
        net, costs, incs = ci.calc_cash_flows("2020-01-01", "2020-01-01", "2025-01-01")
        assert isinstance(net, np.ndarray)
        assert isinstance(costs, np.ndarray)
        assert isinstance(incs, np.ndarray)

    def test_length_matches_periods(self):
        ci = CostIncome(freq="Y")
        net, costs, incs = ci.calc_cash_flows("2020-01-01", "2020-01-01", "2024-01-01")
        periods = pd.period_range("2020-01-01", "2024-01-01", freq="Y")
        assert len(net) == len(periods)

    def test_zero_cost_income(self):
        ci = CostIncome()
        net, costs, incs = ci.calc_cash_flows("2020-01-01", "2020-01-01", "2023-01-01")
        np.testing.assert_array_equal(net, 0.0)

    def test_nonzero_cost_income(self):
        ci = CostIncome(
            mkt_price_year=2020, init_cost=5000, periodic_cost=200, periodic_income=1000
        )
        net, cost, income = ci.calc_cash_flows(
            impl_date="2020-01-01", start_date="2019-01-01", end_date="2025-01-01"
        )
        np.testing.assert_array_equal(
            net, [0.0, -5000.0, 800.0, 800.0, 800.0, 800.0, 800.0]
        )
        np.testing.assert_array_equal(
            cost, [0.0, -5000.0, -200.0, -200.0, -200.0, -200.0, -200.0]
        )
        np.testing.assert_array_equal(
            income, [0.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0]
        )


# ---------------------------------------------------------------------------
# calc_total
# ---------------------------------------------------------------------------


class TestCalcTotal:
    def test_total_is_sum_of_cash_flows(self):
        ci = CostIncome(mkt_price_year=2020, periodic_income=100, periodic_cost=50)
        net_arr, cost_arr, inc_arr = ci.calc_cash_flows(
            "2020-01-01", "2020-01-01", "2024-01-01"
        )
        total_net, total_cost, total_inc = ci.calc_total(
            "2020-01-01", "2020-01-01", "2024-01-01"
        )
        assert total_net == pytest.approx(float(np.sum(net_arr)))
        assert total_cost == pytest.approx(float(np.sum(cost_arr)))
        assert total_inc == pytest.approx(float(np.sum(inc_arr)))

    def test_returns_floats(self):
        ci = CostIncome()
        result = ci.calc_total("2020-01-01", "2020-01-01", "2022-01-01")
        assert all(isinstance(v, (float, np.floating)) for v in result)


# ---------------------------------------------------------------------------
# to_dataframe
# ---------------------------------------------------------------------------


class TestToDataframe:
    def test_columns(self):
        ci = CostIncome(periodic_income=100)
        df = ci.to_dataframe("2020-01-01", "2020-01-01", "2023-01-01")
        assert set(df.columns) == {"date", "net", "cost", "income"}

    def test_row_count(self):
        ci = CostIncome(freq="Y")
        df = ci.to_dataframe("2020-01-01", "2020-01-01", "2022-01-01")
        expected = len(pd.period_range("2020-01-01", "2022-01-01", freq="Y"))
        assert len(df) == expected


# ---------------------------------------------------------------------------
# comb_cost_income
# ---------------------------------------------------------------------------


class TestCombCostIncome:
    def test_costs_are_summed(self):
        ci1 = CostIncome(mkt_price_year=2020, init_cost=100, periodic_cost=50)
        ci2 = CostIncome(mkt_price_year=2020, init_cost=200, periodic_cost=30)
        combined = CostIncome.comb_cost_income([ci1, ci2])
        assert combined.init_cost == -300.0
        assert combined.periodic_cost == -80.0

    def test_incomes_are_summed(self):
        ci1 = CostIncome(mkt_price_year=2020, periodic_income=100)
        ci2 = CostIncome(mkt_price_year=2020, periodic_income=200)
        combined = CostIncome.comb_cost_income([ci1, ci2])
        assert combined.periodic_income == 300.0

    def test_mismatched_mkt_price_year_raises(self):
        ci1 = CostIncome(mkt_price_year=2020)
        ci2 = CostIncome(mkt_price_year=2021)
        with pytest.raises(ValueError, match="market price years"):
            CostIncome.comb_cost_income([ci1, ci2])

    def test_mismatched_cost_growth_rate_raises(self):
        ci1 = CostIncome(mkt_price_year=2020, cost_yearly_growth_rate=0.02)
        ci2 = CostIncome(mkt_price_year=2020, cost_yearly_growth_rate=0.05)
        with pytest.raises(ValueError, match="cost_growth_rate"):
            CostIncome.comb_cost_income([ci1, ci2])

    def test_mismatched_income_growth_rate_raises(self):
        ci1 = CostIncome(mkt_price_year=2020, income_yearly_growth_rate=0.01)
        ci2 = CostIncome(mkt_price_year=2020, income_yearly_growth_rate=0.03)
        with pytest.raises(ValueError, match="income_growth_rate"):
            CostIncome.comb_cost_income([ci1, ci2])

    def test_single_element_list(self):
        ci = CostIncome(mkt_price_year=2020, init_cost=500, periodic_income=100)
        combined = CostIncome.comb_cost_income([ci])
        assert combined.init_cost == -500.0
        assert combined.periodic_income == 100.0

    def test_preserves_growth_rates(self):
        ci1 = CostIncome(
            mkt_price_year=2020,
            cost_yearly_growth_rate=0.02,
            income_yearly_growth_rate=0.03,
        )
        ci2 = CostIncome(
            mkt_price_year=2020,
            cost_yearly_growth_rate=0.02,
            income_yearly_growth_rate=0.03,
        )
        combined = CostIncome.comb_cost_income([ci1, ci2])
        assert combined.cost_growth_rate == 0.02
        assert combined.income_growth_rate == 0.03

    def test_merges_custom_cash_flows(self):
        df1 = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-03-01"],
                "cost": [100, 200],
                "income": [50, 60],
            }
        )
        df2 = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-03-01", "2020-04-01"],
                "cost": [100, 200, 300],
                "income": [50, 60, 70],
            }
        )

        expected = pd.DataFrame(
            {
                "date": ["2020-01", "2020-02", "2020-03", "2020-04"],
                "cost": [-200, 0, -400, -300],
                "income": [100, 0, 120, 70],
            }
        )

        expected["date"] = pd.to_datetime(expected["date"])
        expected = expected.set_index("date")
        expected = expected.resample("MS").sum()

        ci1 = CostIncome(
            mkt_price_year=2020, periodic_income=100, freq="M", custom_cash_flows=df1
        )
        ci2 = CostIncome(
            mkt_price_year=2020, periodic_cost=50, freq="M", custom_cash_flows=df2
        )

        combined = CostIncome.comb_cost_income([ci1, ci2])
        pd.testing.assert_frame_equal(combined.custom_cash_flows, expected)
