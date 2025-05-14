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

import logging

import numpy as np
import pandas as pd

from climada.engine.impact_calc import ImpactCalc
from climada.entity.measures.base import Measure
from climada.trajectories.impact_calc_strat import (
    ImpactCalcComputation,
    ImpactComputationStrategy,
)
from climada.trajectories.interpolation import (
    InterpolationStrategy,
    LinearInterpolation,
)
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)


def lazy_property(method):
    # This function is used as a decorator for properties
    # that require "heavy" computation and are not always needed
    # if the property is none, it uses the corresponding computation method
    # and stores the result in the corresponding private attribute
    attr_name = f"_{method.__name__}"

    @property
    def _lazy(self):
        if getattr(self, attr_name) is None:
            # meas_n = self.measure.name if self.measure else "no_measure"
            # LOGGER.debug(
            #     f"Computing {method.__name__} for {self._snapshot0.date}-{self._snapshot1.date} with {meas_n}."
            # )
            setattr(self, attr_name, method(self))
        return getattr(self, attr_name)

    return _lazy


class CalcRiskPeriod:
    """Handles the computation of impacts for a risk period.

    This object handles the interpolations and computations of risk metrics in
    between two given snapshots, along a DateTime index build on
    `interval_freq` or `time_points`.

    Attributes
    ----------

    date_idx: pd.DateTimeIndex
        The date index for the different interpolated points between the two snapshots
    interpolation_strategy: InterpolationStrategy, optional
        The approach used to interpolate impact matrices in between the two snapshots, linear by default.
    impact_computation_strategy: ImpactComputationStrategy, optional
        The method used to calculate the impact from the (Haz,Exp,Vul) of the two snapshots.
        Defaults to ImpactCalc
    risk_transf_attach: float, optional
        The attachement of risk transfer to apply. Defaults to None.
    risk_transf_cover: float, optional
        The cover of risk transfer to apply. Defaults to None.
    calc_residual: bool, optional
        A boolean stating whether the residual (True) or transfered risk (False) is retained when doing
        the risk transfer. Defaults to False.
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
        interval_freq: str | None = "AS-JAN",
        time_points: int | None = None,
        interpolation_strategy: InterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
        risk_transf_attach: float | None = None,
        risk_transf_cover: float | None = None,
        calc_residual: bool = False,
    ):
        LOGGER.info("Instantiating new CalcRiskPeriod.")
        self._snapshot0 = snapshot0
        self._snapshot1 = snapshot1
        self.date_idx = CalcRiskPeriod._set_date_idx(
            date1=snapshot0.date,
            date2=snapshot1.date,
            periods=time_points,
            freq=interval_freq,
            name="date",
        )
        self.interpolation_strategy = interpolation_strategy or LinearInterpolation()
        self.impact_computation_strategy = (
            impact_computation_strategy or ImpactCalcComputation()
        )
        self.risk_transf_attach = risk_transf_attach
        self.risk_transf_cover = risk_transf_cover
        self.calc_residual = calc_residual
        self.measure = None  # Only possible to set with apply_measure to make sure snapshots are consistent

        self._group_id_E0 = self.snapshot0.exposure.gdf["group_id"].values
        self._group_id_E1 = self.snapshot1.exposure.gdf["group_id"].values

    def _reset_impact_data(self):
        self._impacts_arrays = None
        self._imp_mats_H0, self._imp_mats_H1 = None, None
        self._imp_mats_E0, self._imp_mats_E1 = None, None
        self._per_date_eai_H0, self._per_date_eai_H1 = None, None
        self._per_date_aai_H0, self._per_date_aai_H1 = None, None
        self._eai_gdf = None
        self._per_date_return_periods_H0, self._per_date_return_periods_H1 = None, None

    @staticmethod
    def _set_date_idx(
        date1: str | pd.Timestamp,
        date2: str | pd.Timestamp,
        periods: int | None = None,
        freq: str | None = None,
        name: str | None = None,
    ) -> pd.DatetimeIndex:
        """
        Generate a date range index based on the provided parameters.

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
        pd.DatetimeIndex
            A DatetimeIndex representing the date range.

        Raises
        ------
        ValueError
            If the number of periods and frequency given to date_range are inconsistent.
        """
        if periods is not None and freq is not None:
            points = None
        else:
            points = periods

        ret = pd.date_range(
            date1,
            date2,
            periods=points,
            freq=freq,  # type: ignore
            name=name,
            normalize=True,
        )
        if periods is not None and len(ret) != periods:
            raise ValueError(
                "Number of periods and frequency given to date_range are inconsistant"
            )

        if pd.infer_freq(ret) != freq:
            LOGGER.debug(
                f"Given interval frequency ( {pd.infer_freq(ret)} ) and infered interval frequency differ ( {freq} )."
            )

        return ret

    @property
    def snapshot0(self):
        return self._snapshot0

    @property
    def snapshot1(self):
        return self._snapshot1

    @property
    def date_idx(self):
        return self._date_idx

    @date_idx.setter
    def date_idx(self, value, /):
        if not isinstance(value, pd.DatetimeIndex):
            raise ValueError("Not a DatetimeIndex")

        self._date_idx = value.normalize()
        self._time_points = len(self.date_idx)
        self._interval_freq = pd.infer_freq(self.date_idx)
        self._prop_H1 = np.linspace(0, 1, num=self.time_points)
        self._prop_H0 = 1 - self._prop_H1
        self._reset_impact_data()

    @property
    def time_points(self):
        return self._time_points

    @time_points.setter
    def time_points(self, value, /):
        if not isinstance(value, int):
            raise ValueError("Not an int")

        self.date_idx = pd.date_range(
            self.snapshot0.date, self.snapshot1.date, periods=value, name="date"
        )

    @property
    def interval_freq(self):
        return self._interval_freq

    @interval_freq.setter
    def interval_freq(self, value, /):
        freq = pd.tseries.frequencies.to_offset(value)
        self.date_idx = pd.date_range(
            self.snapshot0.date, self.snapshot1.date, freq=freq, name="date"
        )

    @property
    def interpolation_strategy(self):
        return self._interpolation_strategy

    @interpolation_strategy.setter
    def interpolation_strategy(self, value, /):
        if not isinstance(value, InterpolationStrategy):
            raise ValueError("Not an interpolation strategy")

        self._interpolation_strategy = value
        self._reset_impact_data()

    @property
    def impact_computation_strategy(self):
        return self._impact_computation_strategy

    @impact_computation_strategy.setter
    def impact_computation_strategy(self, value, /):
        if not isinstance(value, ImpactComputationStrategy):
            raise ValueError("Not an interpolation strategy")

        self._impact_computation_strategy = value
        self._reset_impact_data()

    @lazy_property
    def impacts_arrays(self):
        return self.impact_computation_strategy.compute_impacts(
            self.snapshot0,
            self.snapshot1,
            self.risk_transf_attach,
            self.risk_transf_cover,
            self.calc_residual,
        )

    @property
    def _E0H0(self):
        return self.impacts_arrays[0]

    @property
    def _E1H0(self):
        return self.impacts_arrays[1]

    @property
    def _E0H1(self):
        return self.impacts_arrays[2]

    @property
    def _E1H1(self):
        return self.impacts_arrays[3]

    @property
    def risk_transf_attach(self):
        return self._risk_transfer_attach

    @risk_transf_attach.setter
    def risk_transf_attach(self, value, /):
        self._risk_transfer_attach = value
        self._reset_impact_data()

    @property
    def risk_transf_cover(self):
        return self._risk_transfer_cover

    @risk_transf_cover.setter
    def risk_transf_cover(self, value, /):
        self._risk_transfer_cover = value
        self._reset_impact_data()

    @property
    def calc_residual(self):
        return self._calc_residual

    @calc_residual.setter
    def calc_residual(self, value, /):
        if not isinstance(value, bool):
            raise ValueError("Not a boolean")

        self._calc_residual = value
        self._reset_impact_data()

    @lazy_property
    def imp_mats_H0(self):
        return self.interpolation_strategy.interpolate(
            self._E0H0, self._E1H0, self.time_points
        )

    @lazy_property
    def imp_mats_H1(self):
        return self.interpolation_strategy.interpolate(
            self._E0H1, self._E1H1, self.time_points
        )

    @lazy_property
    def imp_mats_E0(self):
        return self.interpolation_strategy.interpolate(
            self._E0H0, self._E0H1, self.time_points
        )

    @lazy_property
    def imp_mats_E1(self):
        return self.interpolation_strategy.interpolate(
            self._E1H0, self._E1H1, self.time_points
        )

    @lazy_property
    def per_date_eai_H0(self):
        return self.calc_per_date_eais(
            self.imp_mats_H0, self.snapshot0.hazard.frequency
        )

    @lazy_property
    def per_date_eai_H1(self):
        return self.calc_per_date_eais(
            self.imp_mats_H1, self.snapshot1.hazard.frequency
        )

    @lazy_property
    def per_date_aai_H0(self):
        return self.calc_per_date_aais(self.per_date_eai_H0)

    @lazy_property
    def per_date_aai_H1(self):
        return self.calc_per_date_aais(self.per_date_eai_H1)

    @lazy_property
    def eai_gdf(self):
        return self.calc_eai_gdf()

    def per_date_return_periods_H0(self, return_periods) -> np.ndarray:
        return self.calc_per_date_rps(
            self.imp_mats_H0, self.snapshot0.hazard.frequency, return_periods
        )

    def per_date_return_periods_H1(self, return_periods) -> np.ndarray:
        return self.calc_per_date_rps(
            self.imp_mats_H1, self.snapshot1.hazard.frequency, return_periods
        )

    @classmethod
    def calc_per_date_eais(cls, imp_mats, frequency) -> np.ndarray:
        """
        Calculate per_date expected average impact (EAI) values for two scenarios.

        Parameters
        ----------
        imp_mats_0 : list of np.ndarray
            List of interpolated impact matrices for scenario 0.
        imp_mats_1 : list of np.ndarray
            List of interpolated impact matrices for scenario 1.
        frequency_0 : np.ndarray
            Frequency values associated with scenario 0.
        frequency_1 : np.ndarray
            Frequency values associated with scenario 1.

        Returns
        -------
        tuple
            Tuple containing:
            - per_date_eai_exp_0 : list of float
                per date expected annual impacts for scenario 0.
            - per_date_eai_exp_1 : list of float
                per date expected annual impacts for scenario 1.
        """
        per_date_eai_exp = np.array(
            [ImpactCalc.eai_exp_from_mat(imp_mat, frequency) for imp_mat in imp_mats]
        )
        return per_date_eai_exp

    @staticmethod
    def calc_per_date_aais(per_date_eai_exp) -> np.ndarray:
        """
        Calculate per_date aggregate annual impact (AAI) values for two scenarios.

        Parameters
        ----------
        per_date_eai_exp_0 : list of float
            Per_Date expected annual impacts for scenario 0.
        per_date_eai_exp_1 : list of float
            Per_Date expected annual impacts for scenario 1.

        Returns
        -------
        tuple
            Tuple containing:
            - per_date_aai_0 : list of float
                Aggregate annual impact values for scenario 0.
            - per_date_aai_1 : list of float
                Aggregate annual impact values for scenario 1.
        """
        per_date_aai = np.array(
            [ImpactCalc.aai_agg_from_eai_exp(eai_exp) for eai_exp in per_date_eai_exp]
        )
        return per_date_aai

    @classmethod
    def calc_per_date_rps(cls, imp_mats, frequency, return_periods) -> np.ndarray:
        """
        Calculate per_date return period impact values for two scenarios.

        Parameters
        ----------
        imp_mats_0 : list of np.ndarray
            List of interpolated impact matrices for scenario 0.
        imp_mats_1 : list of np.ndarray
            List of interpolated impact matrices for scenario 1.
        frequency_0 : np.ndarray
            Frequency values for scenario 0.
        frequency_1 : np.ndarray
            Frequency values for scenario 1.
        return_periods : list of int
            Return periods to calculate impact values for.

        Returns
        -------
        tuple
            Tuple containing:
            - rp_0 : list of np.ndarray
                Per_Date return period impact values for scenario 0.
            - rp_1 : list of np.ndarray
                Per_Date return period impact values for scenario 1.
        """
        rp = np.array(
            [
                cls.calc_freq_curve(imp_mat, frequency, return_periods)
                for imp_mat in imp_mats
            ]
        )
        return rp

    @classmethod
    def calc_freq_curve(cls, imp_mat_intrpl, frequency, return_per=None) -> np.ndarray:
        """
        Calculate the frequency curve

        Parameters:
        imp_mat_intrpl (np.array): The interpolated impact matrix
        frequency (np.array): The frequency of the hazard
        return_per (np.array): The return period

        Returns:
        ifc_return_per (np.array): The impact exceeding frequency
        ifc_impact (np.array): The impact exceeding the return period
        """

        # Calculate the at_event make the np.array
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

    def calc_eai_gdf(self):
        per_date_eai_H0, per_date_eai_H1 = (self.per_date_eai_H0, self.per_date_eai_H1)
        per_date_eai = np.multiply(
            self._prop_H0.reshape(-1, 1), per_date_eai_H0
        ) + np.multiply(self._prop_H1.reshape(-1, 1), per_date_eai_H1)
        df = pd.DataFrame(per_date_eai, index=self.date_idx)
        df = df.reset_index().melt(
            id_vars="date", var_name="coord_id", value_name="risk"
        )
        eai_gdf = self.snapshot1.exposure.gdf
        eai_gdf["coord_id"] = eai_gdf.index
        eai_gdf = eai_gdf.merge(df, on="coord_id")
        eai_gdf = eai_gdf.rename(
            columns={"group_id": "group", "value": "exposure_value"}
        )
        eai_gdf["metric"] = "eai"
        eai_gdf["measure"] = self.measure.name if self.measure else "no_measure"
        return eai_gdf

    def calc_aai_metric(self):
        per_date_aai_H0, per_date_aai_H1 = self.per_date_aai_H0, self.per_date_aai_H1
        per_date_aai = self._prop_H0 * per_date_aai_H0 + self._prop_H1 * per_date_aai_H1
        aai_df = pd.DataFrame(index=self.date_idx, columns=["risk"], data=per_date_aai)
        aai_df["group"] = pd.NA
        aai_df["metric"] = "aai"
        aai_df["measure"] = self.measure.name if self.measure else "no_measure"
        aai_df.reset_index(inplace=True)
        return aai_df

    def calc_aai_per_group_metric(self):
        aai_per_group_df = []
        for group in np.unique(
            np.concatenate(np.array([self._group_id_E0, self._group_id_E1]), axis=0)
        ):
            group_idx_E0 = np.where(self._group_id_E0 == group)[0]
            group_idx_E1 = np.where(self._group_id_E1 == group)[0]
            per_date_aai_H0, per_date_aai_H1 = (
                self.per_date_eai_H0[:, group_idx_E0].sum(axis=1),
                self.per_date_eai_H1[:, group_idx_E1].sum(axis=1),
            )
            per_date_aai = (
                self._prop_H0 * per_date_aai_H0 + self._prop_H1 * per_date_aai_H1
            )
            df = pd.DataFrame(index=self.date_idx, columns=["risk"], data=per_date_aai)
            df["group"] = group
            aai_per_group_df.append(df)

        aai_per_group_df = pd.concat(aai_per_group_df)
        aai_per_group_df["metric"] = "aai"
        aai_per_group_df["measure"] = (
            self.measure.name if self.measure else "no_measure"
        )
        aai_per_group_df.reset_index(inplace=True)
        return aai_per_group_df

    def calc_return_periods_metric(self, return_periods):
        rp_0, rp_1 = (
            self.per_date_return_periods_H0(return_periods),
            self.per_date_return_periods_H1(return_periods),
        )
        per_date_rp = np.multiply(self._prop_H0.reshape(-1, 1), rp_0) + np.multiply(
            self._prop_H1.reshape(-1, 1), rp_1
        )
        rp_df = pd.DataFrame(
            index=self.date_idx, columns=return_periods, data=per_date_rp
        ).melt(value_name="risk", var_name="rp", ignore_index=False)
        rp_df.reset_index(inplace=True)
        rp_df["group"] = pd.NA
        rp_df["metric"] = "rp_" + rp_df["rp"].astype(str)
        rp_df["measure"] = self.measure.name if self.measure else "no_measure"
        return rp_df

    def calc_risk_components_metric(self):
        per_date_aai_H0, per_date_aai_H1 = self.per_date_aai_H0, self.per_date_aai_H1
        per_date_aai = self._prop_H0 * per_date_aai_H0 + self._prop_H1 * per_date_aai_H1

        risk_dev_0 = per_date_aai_H0 - per_date_aai[0]
        risk_cc_0 = per_date_aai - (risk_dev_0 + per_date_aai[0])
        df = pd.DataFrame(
            {
                "base risk": per_date_aai - (risk_dev_0 + risk_cc_0),
                "delta from exposure": risk_dev_0,
                "delta from hazard": risk_cc_0,
            },
            index=self.date_idx,
        )
        df = df.melt(
            value_vars=["base risk", "delta from exposure", "delta from hazard"],
            var_name="metric",
            value_name="risk",
            ignore_index=False,
        )
        df.reset_index(inplace=True)
        df["group"] = pd.NA
        df["measure"] = self.measure.name if self.measure else "no_measure"
        return df

    def apply_measure(self, measure: Measure):
        snap0 = self.snapshot0.apply_measure(measure)
        snap1 = self.snapshot1.apply_measure(measure)

        risk_period = CalcRiskPeriod(
            snap0,
            snap1,
            self.interval_freq,
            self.time_points,
            self.interpolation_strategy,
            self.impact_computation_strategy,
            self.risk_transf_attach,
            self.risk_transf_cover,
            self.calc_residual,
        )

        risk_period.measure = measure
        return risk_period
