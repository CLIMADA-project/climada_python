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

This modules implements the CalcRiskMetrics classes.

CalcRiskMetrics are used to compute risk metrics (and intermediate requirements)
in between two snapshots.

As these computations are not always required and can become "heavy", a so called "lazy"
approach is used: computation is only done when required, and then stored.

"""

import logging

import numpy as np
import pandas as pd

from climada.engine.impact import Impact
from climada.entity._legacy_measures.base import Measure
from climada.trajectories.constants import (
    AAI_METRIC_NAME,
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    EAI_METRIC_NAME,
    GROUP_COL_NAME,
    GROUP_ID_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    RISK_COL_NAME,
    RP_VALUE_PREFIX,
    UNIT_COL_NAME,
)
from climada.trajectories.impact_calc_strat import ImpactComputationStrategy
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CalcRiskMetricsPoints",
]

_CACHE_SETTINGS = {"ENABLE_LAZY_CACHE": False}


def lazy_property(method):
    """
    Decorator that converts a method into a cached, lazy-evaluated property.

    This decorator is intended for properties that require heavy computation.
    The result is calculated only when first accessed and then stored in a
    corresponding private attribute (e.g., a method named `impact` will
    cache its result in `_impact`).

    Parameters
    ----------
    method : callable
        The method to be converted into a lazy property.

    Returns
    -------
    property
        A property object that handles the caching logic and attribute access.

    Notes
    -----
    The caching behavior can be globally toggled via the
    `_CACHE_SETTINGS["ENABLE_LAZY_CACHE"]` flag. If disabled, the
    method will be re-evaluated on every access.

    """
    attr_name = f"_{method.__name__}"

    @property
    def _lazy(self):
        if not _CACHE_SETTINGS.get("ENABLE_LAZY_CACHE", True):
            return method(self)

        if getattr(self, attr_name) is None:
            setattr(self, attr_name, method(self))

        return getattr(self, attr_name)

    return _lazy


class CalcRiskMetricsPoints:
    """This class handles the computation of impacts for a list of `Snapshot`.

    Note that most attribute like members are properties with their own docstring.

    Attributes
    ----------

    impact_computation_strategy: ImpactComputationStrategy, optional
        The method used to calculate the impact from the (Haz,Exp,Vul) of the snapshots.
        Defaults to ImpactCalc
    measure: Measure, optional
        The measure applied to snapshots. Defaults to None.

    Notes
    -----

    This class is intended for internal computation.
    """

    def __init__(
        self,
        snapshots: list[Snapshot],
        impact_computation_strategy: ImpactComputationStrategy,
    ) -> None:
        """Initialize a new `CalcRiskMetricsPoints`

        This initializes and instantiate a new `CalcRiskMetricsPoints` object.
        No computation is done at initialisation and only done "just in time".

        Parameters
        ----------
        snapshots : List[Snapshot]
            The `Snapshot` list to compute risk for.
        impact_computation_strategy: ImpactComputationStrategy, optional
            The method used to calculate the impact from the (Haz,Exp,Vul) of the two snapshots.
            Defaults to ImpactCalc

        """

        self._init_impact_data()
        self.snapshots = snapshots
        self.impact_computation_strategy = impact_computation_strategy
        self._date_idx = pd.DatetimeIndex(
            [snap.date for snap in self.snapshots],
            name=DATE_COL_NAME,
        )
        self.measure = None
        try:
            self._group_id = np.unique(
                np.concatenate(
                    [
                        snap.exposure.gdf[GROUP_ID_COL_NAME]
                        for snap in self.snapshots
                        if GROUP_ID_COL_NAME in snap.exposure.gdf.columns
                    ]
                )
            )
        except ValueError as exc:
            error_message = str(exc).lower()
            if "need at least one array to concatenate" in error_message:
                self._group_id = np.array([])
            else:
                raise

    def _init_impact_data(self):
        """Util method that resets computed data, for instance when
        changing the computation strategy.

        """
        self._impacts = None
        self._eai_gdf = None
        self._per_date_eai = None
        self._per_date_aai = None

    _reset_impact_data = _init_impact_data

    @property
    def impact_computation_strategy(self) -> ImpactComputationStrategy:
        """The method used to calculate the impact from the (Haz,Exp,Vul)
        of the snapshots.

        """
        return self._impact_computation_strategy

    @impact_computation_strategy.setter
    def impact_computation_strategy(self, value, /):
        if not isinstance(value, ImpactComputationStrategy):
            raise ValueError(
                "The provided value is not an ImpactComputationStrategy object. See the trajectory module documentation for more information on how to define your own impact computation strategies."
            )

        self._impact_computation_strategy = value
        self._reset_impact_data()

    @lazy_property
    def impacts(self) -> list[Impact]:
        """Return Impact object for the different snapshots."""

        return [
            self.impact_computation_strategy.compute_impacts(
                snap.exposure, snap.hazard, snap.impfset
            )
            for snap in self.snapshots
        ]

    @lazy_property
    def per_date_eai(self) -> np.ndarray:
        """Expected annual impacts per snapshot."""

        return np.array([imp.eai_exp for imp in self.impacts])

    @lazy_property
    def per_date_aai(self) -> np.ndarray:
        """Average annual impacts per snapshot."""

        return np.array([imp.aai_agg for imp in self.impacts])

    def calc_eai_gdf(self) -> pd.DataFrame:
        """Convenience function returning a DataFrame
        from `per_date_eai`.

        This can easily be merged with the GeoDataFrame of
        the exposure object of one of the `Snapshot`.

        Notes
        -----

        The DataFrame from the first snapshot of the list is used
        as a basis (notably for `value` and `group_id`).

        """

        metric_df = pd.DataFrame(self.per_date_eai, index=self._date_idx)
        metric_df = metric_df.reset_index().melt(
            id_vars=DATE_COL_NAME, var_name=COORD_ID_COL_NAME, value_name=RISK_COL_NAME
        )
        eai_gdf = pd.concat(
            [
                snap.exposure.gdf.reset_index(names=[COORD_ID_COL_NAME]).assign(
                    date=snap.date.as_unit(self._date_idx.unit)
                )
                for snap in self.snapshots
            ]
        )
        if GROUP_ID_COL_NAME in eai_gdf.columns:
            eai_gdf = eai_gdf[[DATE_COL_NAME, COORD_ID_COL_NAME, GROUP_ID_COL_NAME]]
        else:
            eai_gdf[[GROUP_ID_COL_NAME]] = pd.NA
            eai_gdf = eai_gdf[[DATE_COL_NAME, COORD_ID_COL_NAME, GROUP_ID_COL_NAME]]

        eai_gdf = eai_gdf.merge(metric_df, on=[DATE_COL_NAME, COORD_ID_COL_NAME])
        eai_gdf = eai_gdf.rename(columns={GROUP_ID_COL_NAME: GROUP_COL_NAME})
        eai_gdf[GROUP_COL_NAME] = pd.Categorical(
            eai_gdf[GROUP_COL_NAME], categories=self._group_id
        )
        eai_gdf[METRIC_COL_NAME] = EAI_METRIC_NAME
        eai_gdf[MEASURE_COL_NAME] = (
            self.measure.name if self.measure else NO_MEASURE_VALUE
        )
        eai_gdf[UNIT_COL_NAME] = self.snapshots[0].exposure.value_unit
        return eai_gdf

    def calc_aai_metric(self) -> pd.DataFrame:
        """Compute a DataFrame of the AAI for each snapshot."""

        aai_df = pd.DataFrame(
            index=self._date_idx, columns=[RISK_COL_NAME], data=self.per_date_aai
        )
        aai_df[GROUP_COL_NAME] = pd.Categorical(
            [pd.NA] * len(aai_df), categories=self._group_id
        )
        aai_df[METRIC_COL_NAME] = AAI_METRIC_NAME
        aai_df[MEASURE_COL_NAME] = (
            self.measure.name if self.measure else NO_MEASURE_VALUE
        )
        aai_df[UNIT_COL_NAME] = self.snapshots[0].exposure.value_unit
        aai_df.reset_index(inplace=True)
        return aai_df

    def calc_aai_per_group_metric(self) -> pd.DataFrame | None:
        """Compute a DataFrame of the AAI distinguised per group id
        in the exposures, for each snapshot.

        """

        if len(self._group_id) < 1:
            LOGGER.warning(
                "No group id defined in the Exposures object. Per group aai will be empty."
            )
            return None

        eai_pres_groups = self.calc_eai_gdf()[
            [DATE_COL_NAME, COORD_ID_COL_NAME, GROUP_COL_NAME, RISK_COL_NAME]
        ].copy()
        aai_per_group_df = eai_pres_groups.groupby(
            [DATE_COL_NAME, GROUP_COL_NAME], as_index=False, observed=True
        )[RISK_COL_NAME].sum()
        aai_per_group_df[METRIC_COL_NAME] = AAI_METRIC_NAME
        aai_per_group_df[MEASURE_COL_NAME] = (
            self.measure.name if self.measure else NO_MEASURE_VALUE
        )
        aai_per_group_df[UNIT_COL_NAME] = self.snapshots[0].exposure.value_unit
        return aai_per_group_df

    def calc_return_periods_metric(self, return_periods: list[int]) -> pd.DataFrame:
        """Compute a DataFrame of the estimated impacts for a list
        of return periods, for each snapshot.

        Parameters
        ----------

        return_periods : list of int
            The return periods to estimate impacts for.
        """

        per_date_rp = np.array(
            [
                imp.calc_freq_curve(return_per=return_periods).impact
                for imp in self.impacts
            ]
        )
        rp_df = pd.DataFrame(
            index=self._date_idx, columns=return_periods, data=per_date_rp
        ).melt(value_name=RISK_COL_NAME, var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df[GROUP_COL_NAME] = pd.Categorical(
            [pd.NA] * len(rp_df), categories=self._group_id
        )
        rp_df[METRIC_COL_NAME] = RP_VALUE_PREFIX + "_" + rp_df["rp"].astype(str)
        rp_df = rp_df.drop("rp", axis=1)
        rp_df[MEASURE_COL_NAME] = (
            self.measure.name if self.measure else NO_MEASURE_VALUE
        )
        rp_df[UNIT_COL_NAME] = self.snapshots[0].exposure.value_unit
        return rp_df

    def apply_measure(self, measure: Measure) -> "CalcRiskMetricsPoints":
        """Creates a new `CalcRiskMetricsPoints` object by applying the effects
        of the given measure.

        The effects of the measure are applied to all the snapshots contained
        in the initial `CalcRiskMetricsPoints` and a new `CalcRiskMetricsPoints`
        containing the modified snapshots is returned.

        Parameters
        ----------
        measure : Measure
            The measure to apply.

        Returns
        -------

        CalcRiskMetricsPoints
            The risk period with given measure applied.

        """
        snapshots = [snap.apply_measure(measure) for snap in self.snapshots]
        risk_period = CalcRiskMetricsPoints(
            snapshots,
            self.impact_computation_strategy,
        )

        risk_period.measure = measure
        return risk_period
