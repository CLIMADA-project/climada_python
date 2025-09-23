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

This modules implements the CalcRiskPeriod class.

CalcRiskPeriod are used to compute risk metrics (and intermediate requirements)
in between two snapshots.

As these computations are not always required and can become "heavy", a so called "lazy"
approach is used: computation is only done when required, and then stored.

"""

import itertools
import logging
from abc import ABC

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from climada.engine.impact import Impact
from climada.engine.impact_calc import ImpactCalc
from climada.entity.measures.base import Measure
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.interpolation import (
    AllLinearStrategy,
    InterpolationStrategy,
    InterpolationStrategyBase,
    linear_interp_arrays,
)
from climada.trajectories.snapshot import Snapshot
from climada.util import log_level

LOGGER = logging.getLogger(__name__)


def lazy_property(method):
    # This function is used as a decorator for properties
    # that require "heavy" computation and are not always needed.
    # When requested, if a property is none, it uses the corresponding
    # computation method and caches the result in the corresponding
    # private attribute
    attr_name = f"_{method.__name__}"

    @property
    def _lazy(self):
        if getattr(self, attr_name) is None:
            # LOGGER.debug(
            #     f"Computing {method.__name__} for {self._snapshot0.date}-{self._snapshot1.date} with {meas_n}."
            # )
            setattr(self, attr_name, method(self))
        return getattr(self, attr_name)

    return _lazy


class CalcRiskMetricsPoints:

    def __init__(
        self,
        snapshots: list[Snapshot],
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ) -> None:
        self._reset_impact_data()
        self.snapshots = snapshots
        self.impact_computation_strategy = (
            impact_computation_strategy or ImpactCalcComputation()
        )
        self._date_idx = pd.DatetimeIndex(
            [snap.date for snap in self.snapshots], name="date"
        )
        self.measure = None
        self._groups_id = np.unique(
            np.concatenate(
                [
                    snap.exposure.gdf["group_id"]
                    for snap in self.snapshots
                    if "group_id" in snap.exposure.gdf.columns
                ]
            )
        )

    def _reset_impact_data(self):
        """Util method that resets computed data, for instance when changing the time resolution."""
        self._impacts = None
        self._eai_gdf = None
        self._per_date_eai = None
        self._per_date_aai = None

    @lazy_property
    def impacts(self) -> list[Impact]:
        return [
            self.impact_computation_strategy.compute_impacts(
                snap.exposure, snap.hazard, snap.impfset
            )
            for snap in self.snapshots
        ]

    @lazy_property
    def per_date_eai(self) -> np.ndarray:
        """Expected annual impacts per date with changing exposure, changing hazard and changing vulnerability"""
        return np.array([imp.eai_exp for imp in self.impacts])

    @lazy_property
    def per_date_aai(self) -> np.ndarray:
        """Average annual impacts per date with changing exposure, changing hazard and changing vulnerability"""
        return np.array([imp.aai_agg for imp in self.impacts])

    @lazy_property
    def eai_gdf(self) -> pd.DataFrame:
        """Convenience function returning a GeoDataFrame (with both datetime and coordinates) from `per_date_eai`.

        Notes
        -----

        The GeoDataFrame from the starting snapshot is used as a basis (notably for `value` and `group_id`).
        """
        return self.calc_eai_gdf()

    def calc_eai_gdf(self) -> pd.DataFrame:
        """Merges the per date EAIs of the risk period with the GeoDataframe of the exposure of the starting snapshot."""
        df = pd.DataFrame(self.per_date_eai, index=self._date_idx)
        df = df.reset_index().melt(
            id_vars="date", var_name="coord_id", value_name="risk"
        )
        eai_gdf = pd.concat(
            [snap.exposure.gdf[["group_id"]] for snap in self.snapshots]
        )
        eai_gdf["coord_id"] = eai_gdf.index
        eai_gdf = eai_gdf.merge(df, on="coord_id")
        eai_gdf = eai_gdf.rename(columns={"group_id": "group"})
        eai_gdf["group"] = pd.Categorical(eai_gdf["group"], categories=self._groups_id)
        eai_gdf["metric"] = "eai"
        eai_gdf["measure"] = self.measure.name if self.measure else "no_measure"
        return eai_gdf

    def calc_aai_metric(self) -> pd.DataFrame:
        """Compute a DataFrame of the AAI at each dates of the risk period (including changes in exposure, hazard and vulnerability)."""
        aai_df = pd.DataFrame(
            index=self._date_idx, columns=["risk"], data=self.per_date_aai
        )
        aai_df["group"] = pd.Categorical(
            [pd.NA] * len(aai_df), categories=self._groups_id
        )
        aai_df["metric"] = "aai"
        aai_df["measure"] = self.measure.name if self.measure else "no_measure"
        aai_df.reset_index(inplace=True)
        return aai_df

    def calc_aai_per_group_metric(self) -> pd.DataFrame:
        """Compute a DataFrame of the AAI distinguised per group id in the exposures, at each dates of the risk period (including changes in exposure, hazard and vulnerability).

        Notes
        -----

        If group id changes between starting and ending snapshots of the risk period, the AAIs are linearly interpolated (with a warning for transparency).
        """

        eai_pres_groups = self.eai_gdf[["date", "coord_id", "group", "risk"]].copy()
        aai_per_group_df = eai_pres_groups.groupby(
            ["date", "group"], as_index=False, observed=True
        )["risk"].sum()
        aai_per_group_df["metric"] = "aai"
        aai_per_group_df["measure"] = (
            self.measure.name if self.measure else "no_measure"
        )
        return aai_per_group_df

    def calc_return_periods_metric(self, return_periods: list[int]) -> pd.DataFrame:
        """Compute a DataFrame of the estimated impacts for a list of return periods, at each dates of the risk period (including changes in exposure, hazard and vulnerability).

        Parameters
        ----------

        return_periods : list of int
            The return periods to estimate impacts for.
        """

        # currently mathematicaly wrong, but approximatively correct, to be reworked when concatenating the impact matrices for the interpolation
        per_date_rp = np.array(
            [
                imp.calc_freq_curve(return_per=return_periods).impact
                for imp in self.impacts
            ]
        )
        rp_df = pd.DataFrame(
            index=self._date_idx, columns=return_periods, data=per_date_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = pd.Categorical(
            [pd.NA] * len(rp_df), categories=self._groups_id
        )
        rp_df["metric"] = "rp_" + rp_df["rp"].astype(str)
        rp_df["measure"] = self.measure.name if self.measure else "no_measure"
        return rp_df

    def apply_measure(self, measure: Measure) -> "CalcRiskMetricsPoints":
        """Creates a new `CalcRiskPeriod` object with a measure.

        The given measure is applied to both snapshot of the risk period.

        Parameters
        ----------
        measure : Measure
            The measure to apply.

        Returns
        -------

        CalcRiskPeriod
            The risk period with given measure applied.

        """
        snapshots = [snap.apply(measure) for snap in self.snapshots]
        risk_period = CalcRiskMetricsPoints(
            snapshots,
            self.impact_computation_strategy,
        )

        risk_period.measure = measure
        return risk_period


class CalcRiskMetricsPeriod:
    """Handles the computation of impacts for a risk period.

    This object handles the interpolations and computations of risk metrics in
    between two given snapshots, along a DateTimeIndex build from either a
    `time_resolution` (which must be a valid "freq" string to build a DateTimeIndex)
    and defaults to "Y" (start of the year) or `time_points` integer argument, in which case
    the DateTimeIndex will have that many periods.

    Note that most attribute like members are properties with their own docstring.

    Attributes
    ----------

    date_idx: pd.PeriodIndex
        The date index for the different interpolated points between the two snapshots
    interpolation_strategy: InterpolationStrategy, optional
        The approach used to interpolate impact matrices in between the two snapshots, linear by default.
    impact_computation_strategy: ImpactComputationStrategy, optional
        The method used to calculate the impact from the (Haz,Exp,Vul) of the two snapshots.
        Defaults to ImpactCalc
    measure: Measure, optional
        The measure to apply to both snapshots. Defaults to None.

    Notes
    -----

    This class is intended for internal computation. Users should favor `RiskTrajectory` objects.
    """

    def __init__(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        time_resolution: str | None = None,
        time_points: int | None = None,
        interpolation_strategy: InterpolationStrategyBase | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
    ):
        """Initialize a new `CalcRiskPeriod`

        This initializes and instantiate a new CalcRiskPeriod object. No heavy
        computation is done at that point.

        Parameters
        ----------
        snapshot0 : Snapshot
            The `Snapshot` at the start of the risk period.
        snapshot1 : Snapshot
            The `Snapshot` at the end of the risk period.
        time_resolution : str, optional
            One of pandas date offset strings or corresponding objects. See :func:`pandas.date_range`.
        time_points : int, optional
            Number of periods to generate for the PeriodIndex.
        interpolation_strategy: InterpolationStrategy, optional
            The approach used to interpolate impact matrices in between the two snapshots, linear by default.
        impact_computation_strategy: ImpactComputationStrategy, optional
            The method used to calculate the impact from the (Haz,Exp,Vul) of the two snapshots.
            Defaults to ImpactCalc

        Notes
        -----

        If both `time_points` and `freq` are given, a consistency check between the two is made.

        """

        LOGGER.debug("Instantiating new CalcRiskPeriod.")
        self._snapshot0 = snapshot0
        self._snapshot1 = snapshot1
        self.date_idx = self._set_date_idx(
            date1=snapshot0.date,
            date2=snapshot1.date,
            periods=time_points,
            freq=time_resolution,
            name="date",
        )
        self.interpolation_strategy = interpolation_strategy or AllLinearStrategy()
        self.impact_computation_strategy = (
            impact_computation_strategy or ImpactCalcComputation()
        )
        self.measure = None  # Only possible to set with apply_measure to make sure snapshots are consistent

        self._group_id_E0 = (
            self.snapshot_start.exposure.gdf["group_id"].values
            if "group_id" in self.snapshot_start.exposure.gdf.columns
            else np.array([])
        )
        self._group_id_E1 = (
            self.snapshot_end.exposure.gdf["group_id"].values
            if "group_id" in self.snapshot_end.exposure.gdf.columns
            else np.array([])
        )
        self._groups_id = np.unique(
            np.concatenate([self._group_id_E0, self._group_id_E1])
        )

    def _reset_impact_data(self):
        """Util method that resets computed data, for instance when changing the time resolution."""
        for fut in list(itertools.product([0, 1], repeat=3)):
            setattr(self, f"_E{fut[0]}H{fut[1]}V{fut[2]}", None)

        for fut in list(itertools.product([0, 1], repeat=2)):
            setattr(self, f"_imp_mats_H{fut[0]}V{fut[1]}", None)
            setattr(self, f"_per_date_eai_H{fut[0]}V{fut[1]}", None)
            setattr(self, f"_per_date_aai_H{fut[0]}V{fut[1]}", None)

        self._eai_gdf = None
        self._per_date_eai = None
        self._per_date_aai = None
        self._per_date_return_periods_H0, self._per_date_return_periods_H1 = None, None

    @staticmethod
    def _set_date_idx(
        date1: str | pd.Timestamp,
        date2: str | pd.Timestamp,
        periods: int | None = None,
        freq: str | None = None,
        name: str | None = None,
    ) -> pd.PeriodIndex:
        """Generate a date range index based on the provided parameters.

        Parameters
        ----------
        date1 : str or pd.Timestamp
            The start date of the date range.
        date2 : str or pd.Timestamp
            The end date of the date range.
        periods : int, optional
            Number of date points to generate. If None, `freq` must be provided.
        freq : str, optional
            Frequency string for the date range. If None, `periods` must be provided.
        name : str, optional
            Name of the resulting date range index.

        Returns
        -------
        pd.PeriodIndex
            A PeriodIndex representing the date range.

        Raises
        ------
        ValueError
            If the number of periods and frequency given to date_range are inconsistent.
        """
        if periods is not None and freq is not None:
            points = None
        else:
            points = periods

        ret = pd.period_range(
            date1,
            date2,
            periods=points,
            freq=freq,  # type: ignore
            name=name,
        )
        if periods is not None and len(ret) != periods:
            raise ValueError(
                "Number of periods and frequency given to date_range are inconsistent."
            )

        if ret.freq != freq:
            LOGGER.debug(
                f"Given interval frequency ( {ret.freq} ) and infered interval frequency differ ( {freq} )."
            )

        return ret

    @property
    def snapshot_start(self) -> Snapshot:
        """The `Snapshot` at the start of the risk period."""
        return self._snapshot0

    @property
    def snapshot_end(self) -> Snapshot:
        """The `Snapshot` at the end of the risk period."""
        return self._snapshot1

    @property
    def date_idx(self) -> pd.PeriodIndex:
        """The pandas PeriodIndex representing the time dimension of the risk period."""
        return self._date_idx

    @date_idx.setter
    def date_idx(self, value, /):
        if not isinstance(value, pd.PeriodIndex):
            raise ValueError("Not a PeriodIndex")

        self._date_idx = value  # Avoids weird hourly data
        self._time_points = len(self.date_idx)
        self._time_resolution = self.date_idx.freq
        self._reset_impact_data()

    @property
    def time_points(self) -> int:
        """The numbers of different time points (dates) in the risk period."""
        return self._time_points

    @time_points.setter
    def time_points(self, value, /):
        if not isinstance(value, int):
            raise ValueError("Not an int")

        self.date_idx = pd.date_range(
            self.snapshot_start.date, self.snapshot_end.date, periods=value, name="date"
        )

    @property
    def time_resolution(self) -> str:
        """The time resolution of the risk periods, expressed as a pandas interval frequency string."""
        return self._time_resolution

    @time_resolution.setter
    def time_resolution(self, value, /):
        freq = pd.tseries.frequencies.to_offset(value)
        self.date_idx = pd.date_range(
            self.snapshot_start.date, self.snapshot_end.date, freq=freq, name="date"
        )

    @property
    def interpolation_strategy(self) -> InterpolationStrategyBase:
        """The approach used to interpolate impact matrices in between the two snapshots."""
        return self._interpolation_strategy

    @interpolation_strategy.setter
    def interpolation_strategy(self, value, /):
        if not isinstance(value, InterpolationStrategyBase):
            raise ValueError("Not an interpolation strategy")

        self._interpolation_strategy = value
        self._reset_impact_data()

    @property
    def impact_computation_strategy(self) -> ImpactComputationStrategy:
        """The method used to calculate the impact from the (Haz,Exp,Vul) of the two snapshots."""
        return self._impact_computation_strategy

    @impact_computation_strategy.setter
    def impact_computation_strategy(self, value, /):
        if not isinstance(value, ImpactComputationStrategy):
            raise ValueError("Not an interpolation strategy")

        self._impact_computation_strategy = value
        self._reset_impact_data()

    ##### Impact objects cube / Risk Cube #####

    @lazy_property
    def E0H0V0(self) -> Impact:
        """Impact object corresponding to starting exposure, starting hazard and starting vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_start.exposure,
            self.snapshot_start.hazard,
            self.snapshot_start.impfset,
        )

    @lazy_property
    def E1H0V0(self) -> Impact:
        """Impact object corresponding to future exposure, starting hazard and starting vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_end.exposure,
            self.snapshot_start.hazard,
            self.snapshot_start.impfset,
        )

    @lazy_property
    def E0H1V0(self) -> Impact:
        """Impact object corresponding to starting exposure, future hazard and starting vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_start.exposure,
            self.snapshot_end.hazard,
            self.snapshot_start.impfset,
        )

    @lazy_property
    def E1H1V0(self) -> Impact:
        """Impact object corresponding to future exposure, future hazard and starting vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_end.exposure,
            self.snapshot_end.hazard,
            self.snapshot_start.impfset,
        )

    @lazy_property
    def E0H0V1(self) -> Impact:
        """Impact object corresponding to starting exposure, starting hazard and future vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_start.exposure,
            self.snapshot_start.hazard,
            self.snapshot_end.impfset,
        )

    @lazy_property
    def E1H0V1(self) -> Impact:
        """Impact object corresponding to future exposure, starting hazard and future vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_end.exposure,
            self.snapshot_start.hazard,
            self.snapshot_end.impfset,
        )

    @lazy_property
    def E0H1V1(self) -> Impact:
        """Impact object corresponding to starting exposure, future hazard and future vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_start.exposure,
            self.snapshot_end.hazard,
            self.snapshot_end.impfset,
        )

    @lazy_property
    def E1H1V1(self) -> Impact:
        """Impact object corresponding to future exposure, future hazard and future vulnerability."""
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot_end.exposure,
            self.snapshot_end.hazard,
            self.snapshot_end.impfset,
        )

    ###############################

    ### Impact Matrices arrays ####

    @lazy_property
    def imp_mats_H0V0(self) -> list:
        """List of `time_points` impact matrices with changing exposure, starting hazard and starting vulnerability."""
        return self.interpolation_strategy.interp_over_exposure_dim(
            self.E0H0V0.imp_mat, self.E1H0V0.imp_mat, self.time_points
        )

    @lazy_property
    def imp_mats_H1V0(self) -> list:
        """List of `time_points` impact matrices with changing exposure, future hazard and starting vulnerability."""
        return self.interpolation_strategy.interp_over_exposure_dim(
            self.E0H1V0.imp_mat, self.E1H1V0.imp_mat, self.time_points
        )

    @lazy_property
    def imp_mats_H0V1(self) -> list:
        """List of `time_points` impact matrices with changing exposure, starting hazard and future vulnerability."""
        return self.interpolation_strategy.interp_over_exposure_dim(
            self.E0H0V1.imp_mat, self.E1H0V1.imp_mat, self.time_points
        )

    @lazy_property
    def imp_mats_H1V1(self) -> list:
        """List of `time_points` impact matrices with changing exposure, future hazard and future vulnerability."""
        return self.interpolation_strategy.interp_over_exposure_dim(
            self.E0H1V1.imp_mat, self.E1H1V1.imp_mat, self.time_points
        )

    ###############################

    ########## Base EAI ###########

    @lazy_property
    def per_date_eai_H0V0(self) -> np.ndarray:
        """Expected annual impacts for changing exposure, starting hazard and starting vulnerability."""
        return calc_per_date_eais(
            self.imp_mats_H0V0, self.snapshot_start.hazard.frequency
        )

    @lazy_property
    def per_date_eai_H1V0(self) -> np.ndarray:
        """Expected annual impacts for changing exposure, future hazard and starting vulnerability."""
        return calc_per_date_eais(
            self.imp_mats_H1V0, self.snapshot_end.hazard.frequency
        )

    @lazy_property
    def per_date_eai_H0V1(self) -> np.ndarray:
        """Expected annual impacts for changing exposure, starting hazard and future vulnerability."""
        return calc_per_date_eais(
            self.imp_mats_H0V1, self.snapshot_start.hazard.frequency
        )

    @lazy_property
    def per_date_eai_H1V1(self) -> np.ndarray:
        """Expected annual impacts for changing exposure, future hazard and future vulnerability."""
        return calc_per_date_eais(
            self.imp_mats_H1V1, self.snapshot_end.hazard.frequency
        )

    ##################################

    ######### Specific AAIs ##########

    @lazy_property
    def per_date_aai_H0V0(self) -> np.ndarray:
        """Average annual impacts for changing exposure, starting hazard and starting vulnerability."""
        return calc_per_date_aais(self.per_date_eai_H0V0)

    @lazy_property
    def per_date_aai_H1V0(self) -> np.ndarray:
        """Average annual impacts for changing exposure, future hazard and starting vulnerability."""
        return calc_per_date_aais(self.per_date_eai_H1V0)

    @lazy_property
    def per_date_aai_H0V1(self) -> np.ndarray:
        """Average annual impacts for changing exposure, starting hazard and future vulnerability."""
        return calc_per_date_aais(self.per_date_eai_H0V1)

    @lazy_property
    def per_date_aai_H1V1(self) -> np.ndarray:
        """Average annual impacts for changing exposure, future hazard and future vulnerability."""
        return calc_per_date_aais(self.per_date_eai_H1V1)

    #################################

    ######### Specific RPs  #########

    def per_date_return_periods_H0V0(self, return_periods: list[int]) -> np.ndarray:
        """Estimated impacts per dates for given return periods, with changing exposure, starting hazard and starting vulnerability."""
        return calc_per_date_rps(
            self.imp_mats_H0V0, self.snapshot_start.hazard.frequency, return_periods
        )

    def per_date_return_periods_H1V0(self, return_periods: list[int]) -> np.ndarray:
        """Estimated impacts per dates for given return periods, with changing exposure, future hazard and starting vulnerability."""
        return calc_per_date_rps(
            self.imp_mats_H1V0, self.snapshot_end.hazard.frequency, return_periods
        )

    def per_date_return_periods_H0V1(self, return_periods: list[int]) -> np.ndarray:
        """Estimated impacts per dates for given return periods, with changing exposure, starting hazard and future vulnerability."""
        return calc_per_date_rps(
            self.imp_mats_H0V1, self.snapshot_start.hazard.frequency, return_periods
        )

    def per_date_return_periods_H1V1(self, return_periods: list[int]) -> np.ndarray:
        """Estimated impacts per dates for given return periods, with changing exposure, future hazard and future vulnerability."""
        return calc_per_date_rps(
            self.imp_mats_H1V1, self.snapshot_end.hazard.frequency, return_periods
        )

    ##################################

    ### Fully interpolated metrics ###

    @lazy_property
    def per_date_eai(self) -> np.ndarray:
        """Expected annual impacts per date with changing exposure, changing hazard and changing vulnerability"""
        return self.calc_eai()

    @lazy_property
    def per_date_aai(self) -> np.ndarray:
        """Average annual impacts per date with changing exposure, changing hazard and changing vulnerability"""
        return calc_per_date_aais(self.per_date_eai)

    @lazy_property
    def eai_gdf(self) -> gpd.GeoDataFrame:
        """Convenience function returning a GeoDataFrame (with both datetime and coordinates) from `per_date_eai`.

        Notes
        -----

        The GeoDataFrame from the starting snapshot is used as a basis (notably for `value` and `group_id`).
        """
        return self.calc_eai_gdf()

    ####################################

    ### Metrics from impact matrices ###

    # These methods might go in a utils file instead, to be reused
    # for a no interpolation case (and maybe the timeseries?)

    ####################################

    ##### Interpolation of metrics #####

    def calc_eai(self) -> np.ndarray:
        """Compute the EAIs at each date of the risk period (including changes in exposure, hazard and vulnerability)."""
        per_date_eai_H0V0, per_date_eai_H1V0, per_date_eai_H0V1, per_date_eai_H1V1 = (
            self.per_date_eai_H0V0,
            self.per_date_eai_H1V0,
            self.per_date_eai_H0V1,
            self.per_date_eai_H1V1,
        )
        per_date_eai_V0 = self.interpolation_strategy.interp_over_hazard_dim(
            per_date_eai_H0V0, per_date_eai_H1V0
        )
        per_date_eai_V1 = self.interpolation_strategy.interp_over_hazard_dim(
            per_date_eai_H0V1, per_date_eai_H1V1
        )
        per_date_eai = self.interpolation_strategy.interp_over_vulnerability_dim(
            per_date_eai_V0, per_date_eai_V1
        )
        return per_date_eai

    def calc_eai_gdf(self) -> gpd.GeoDataFrame:
        """Merges the per date EAIs of the risk period with the GeoDataframe of the exposure of the starting snapshot."""
        df = pd.DataFrame(self.per_date_eai, index=self.date_idx)
        df = df.reset_index().melt(
            id_vars="date", var_name="coord_id", value_name="risk"
        )
        eai_gdf = self.snapshot_start.exposure.gdf[["group_id"]]
        eai_gdf["coord_id"] = eai_gdf.index
        eai_gdf = eai_gdf.merge(df, on="coord_id")
        eai_gdf = eai_gdf.rename(columns={"group_id": "group"})
        eai_gdf["group"] = pd.Categorical(eai_gdf["group"], categories=self._groups_id)
        eai_gdf["metric"] = "eai"
        eai_gdf["measure"] = self.measure.name if self.measure else "no_measure"
        return eai_gdf

    def calc_aai_metric(self) -> pd.DataFrame:
        """Compute a DataFrame of the AAI at each dates of the risk period (including changes in exposure, hazard and vulnerability)."""
        aai_df = pd.DataFrame(
            index=self.date_idx, columns=["risk"], data=self.per_date_aai
        )
        aai_df["group"] = pd.Categorical(
            [pd.NA] * len(aai_df), categories=self._groups_id
        )
        aai_df["metric"] = "aai"
        aai_df["measure"] = self.measure.name if self.measure else "no_measure"
        aai_df.reset_index(inplace=True)
        return aai_df

    def calc_aai_per_group_metric(self) -> pd.DataFrame:
        """Compute a DataFrame of the AAI distinguised per group id in the exposures, at each dates of the risk period (including changes in exposure, hazard and vulnerability).

        Notes
        -----

        If group id changes between starting and ending snapshots of the risk period, the AAIs are linearly interpolated (with a warning for transparency).
        """
        if len(self._group_id_E0) < 1 or len(self._group_id_E1) < 1:
            LOGGER.warning(
                "No group id defined in at least one of the Exposures object. Per group aai will be empty."
            )
            return pd.DataFrame()

        eai_pres_groups = self.eai_gdf[["date", "coord_id", "group", "risk"]].copy()
        aai_per_group_df = eai_pres_groups.groupby(
            ["date", "group"], as_index=False, observed=True
        )["risk"].sum()
        if not np.array_equal(self._group_id_E0, self._group_id_E1):
            LOGGER.warning(
                "Group id are changing between present and future snapshot. Per group AAI will be linearly interpolated."
            )
            eai_fut_groups = self.eai_gdf.copy()
            eai_fut_groups["group"] = pd.Categorical(
                np.tile(self._group_id_E1, len(self.date_idx)),
                categories=self._groups_id,
            )
            aai_fut_groups = eai_fut_groups.groupby(["date", "group"], as_index=False)[
                "risk"
            ].sum()
            aai_per_group_df["risk"] = linear_interp_arrays(
                aai_per_group_df["risk"].values, aai_fut_groups["risk"].values
            )

        aai_per_group_df["metric"] = "aai"
        aai_per_group_df["measure"] = (
            self.measure.name if self.measure else "no_measure"
        )
        return aai_per_group_df

    def calc_return_periods_metric(self, return_periods: list[int]) -> pd.DataFrame:
        """Compute a DataFrame of the estimated impacts for a list of return periods, at each dates of the risk period (including changes in exposure, hazard and vulnerability).

        Parameters
        ----------

        return_periods : list of int
            The return periods to estimate impacts for.
        """

        # currently mathematicaly wrong, but approximatively correct, to be reworked when concatenating the impact matrices for the interpolation
        per_date_rp_H0V0, per_date_rp_H1V0, per_date_rp_H0V1, per_date_rp_H1V1 = (
            self.per_date_return_periods_H0V0(return_periods),
            self.per_date_return_periods_H1V0(return_periods),
            self.per_date_return_periods_H0V1(return_periods),
            self.per_date_return_periods_H1V1(return_periods),
        )
        per_date_rp_V0 = self.interpolation_strategy.interp_over_hazard_dim(
            per_date_rp_H0V0, per_date_rp_H1V0
        )
        per_date_rp_V1 = self.interpolation_strategy.interp_over_hazard_dim(
            per_date_rp_H0V1, per_date_rp_H1V1
        )
        per_date_rp = self.interpolation_strategy.interp_over_vulnerability_dim(
            per_date_rp_V0, per_date_rp_V1
        )
        rp_df = pd.DataFrame(
            index=self.date_idx, columns=return_periods, data=per_date_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = pd.Categorical(
            [pd.NA] * len(rp_df), categories=self._groups_id
        )
        rp_df["metric"] = "rp_" + rp_df["rp"].astype(str)
        rp_df["measure"] = self.measure.name if self.measure else "no_measure"
        return rp_df

    def calc_risk_components_metric(self) -> pd.DataFrame:
        """Compute a DataFrame of the individual components of risk (impact), at each dates of the risk period (including changes in exposure, hazard and vulnerability)."""
        per_date_aai_V0 = self.interpolation_strategy.interp_over_hazard_dim(
            self.per_date_aai_H0V0, self.per_date_aai_H1V0
        )
        per_date_aai_H0 = self.interpolation_strategy.interp_over_vulnerability_dim(
            self.per_date_aai_H0V0, self.per_date_aai_H0V1
        )
        df = pd.DataFrame(
            {
                "total risk": self.per_date_aai,
                "base risk": self.per_date_aai[0],
                "exposure contribution": self.per_date_aai_H0V0 - self.per_date_aai[0],
                "hazard contribution": per_date_aai_V0
                - (self.per_date_aai_H0V0 - self.per_date_aai[0])
                - self.per_date_aai[0],
                "vulnerability contribution": per_date_aai_H0
                - self.per_date_aai[0]
                - (self.per_date_aai_H0V0 - self.per_date_aai[0]),
            },
            index=self.date_idx,
        )
        df["interaction contribution"] = df["total risk"] - (
            df["base risk"]
            + df["exposure contribution"]
            + df["hazard contribution"]
            + df["vulnerability contribution"]
        )
        df = df.melt(
            value_vars=[
                "base risk",
                "exposure contribution",
                "hazard contribution",
                "vulnerability contribution",
                "interaction contribution",
            ],
            var_name="metric",
            value_name="risk",
            ignore_index=False,
        )
        df.reset_index(inplace=True)
        df["group"] = pd.Categorical([pd.NA] * len(df), categories=self._groups_id)
        df["measure"] = self.measure.name if self.measure else "no_measure"
        return df

    def apply_measure(self, measure: Measure) -> "CalcRiskMetricsPeriod":
        """Creates a new `CalcRiskPeriod` object with a measure.

        The given measure is applied to both snapshot of the risk period.

        Parameters
        ----------
        measure : Measure
            The measure to apply.

        Returns
        -------

        CalcRiskPeriod
            The risk period with given measure applied.

        """
        snap0 = self.snapshot_start.apply_measure(measure)
        snap1 = self.snapshot_end.apply_measure(measure)

        risk_period = CalcRiskMetricsPeriod(
            snap0,
            snap1,
            self.time_resolution,
            self.time_points,
            self.interpolation_strategy,
            self.impact_computation_strategy,
        )

        risk_period.measure = measure
        return risk_period


def calc_per_date_eais(imp_mats: list[csr_matrix], frequency: np.ndarray) -> np.ndarray:
    """
    Calculate expected average impact (EAI) values from a list of impact matrices corresponding
    to impacts at different dates (with possible changes along exposure, hazard and vulnerability).

    Parameters
    ----------
    imp_mats : list of np.ndarray
        List of impact matrices.
    frequency : np.ndarray
        Hazard frequency values.

    Returns
    -------
    np.ndarray
        2D array of EAI (1D) for each dates.
    """
    per_date_eai_exp = np.array(
        [ImpactCalc.eai_exp_from_mat(imp_mat, frequency) for imp_mat in imp_mats]
    )
    return per_date_eai_exp


def calc_per_date_aais(per_date_eai_exp: np.ndarray) -> np.ndarray:
    """
    Calculate per_date aggregate annual impact (AAI) values resulting from a list arrays corresponding
    to EAI at different dates (with possible changes along exposure, hazard and vulnerability).

    Parameters
    ----------
    per_date_eai_exp: np.ndarray
        EAIs arrays.

    Returns
    -------
    np.ndarray
        1D array of AAI (0D) for each dates.
    """
    per_date_aai = np.array(
        [ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in per_date_eai_exp]
    )
    return per_date_aai


def calc_per_date_rps(
    imp_mats: list[csr_matrix],
    frequency: np.ndarray,
    return_periods: list[int],
) -> np.ndarray:
    """
    Calculate per date return period impact values from a list of impact matrices corresponding
    to impacts at different dates (with possible changes along exposure, hazard and vulnerability).

    Parameters
    ----------
    imp_mats: list of scipy.crs_matrix
        List of impact matrices.
    frequency: np.ndarray
        Frequency values.
    return_periods : list of int
        Return periods to calculate impact values for.

    Returns
    -------
    np.ndarray
        2D array of impacts per return periods (1D) for each dates.
    """
    rp = np.array(
        [calc_freq_curve(imp_mat, frequency, return_periods) for imp_mat in imp_mats]
    )
    return rp


def calc_freq_curve(imp_mat_intrpl, frequency, return_per=None) -> np.ndarray:
    """
    Calculate the estimated impacts for given return periods.

    Parameters
    ----------

    imp_mat_intrpl: scipy.csr_matrix
        An impact matrix.
    frequency: np.ndarray
        The frequency of the hazard.
    return_per: np.ndarray
        The return periods to compute impacts for.

    Returns
    -------
    np.ndarray
       The estimated impacts for the different return periods.
    """

    at_event = np.sum(imp_mat_intrpl, axis=1).A1

    # Sort descendingly the impacts per events
    sort_idxs = np.argsort(at_event)[::-1]
    # Calculate exceedence frequency
    exceed_freq = np.cumsum(frequency[sort_idxs])
    # Set return period and impact exceeding frequency
    ifc_return_per = 1 / exceed_freq[::-1]
    ifc_impact = at_event[sort_idxs][::-1]

    if return_per is not None:
        interp_imp = np.interp(return_per, ifc_return_per, ifc_impact)
        ifc_return_per = return_per
        ifc_impact = interp_imp

    return ifc_impact
