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
    attr_name = f"_{method.__name__}"

    @property
    def _lazy(self):
        if getattr(self, attr_name) is None:
            setattr(self, attr_name, method(self))
        return getattr(self, attr_name)

    return _lazy


class CalcRiskPeriod:
    """Handles the computation of impacts for a risk period."""

    def __init__(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        interval_freq: str | None = "YS",
        time_points: int | None = None,
        interpolation_strategy: InterpolationStrategy | None = None,
        impact_computation_strategy: ImpactComputationStrategy | None = None,
        risk_transf_attach: float | None = None,
        risk_transf_cover: float | None = None,
        calc_residual: bool = False,
        measure: Measure | None = None,
    ):
        self.snapshot0 = snapshot0
        self.snapshot1 = snapshot1
        self.date_idx = pd.date_range(
            snapshot0.date,
            snapshot1.date,
            periods=time_points,
            freq=interval_freq,  # type: ignore
            name="date",
        )
        self.time_points = len(self.date_idx)
        self.interval_freq = pd.infer_freq(self.date_idx)
        self.measure = measure
        self._prop_H1 = np.linspace(0, 1, num=self.time_points)
        self._prop_H0 = 1 - self._prop_H1
        self.interpolation_strategy = interpolation_strategy or LinearInterpolation()
        self.impact_computation_strategy = (
            impact_computation_strategy or ImpactCalcComputation()
        )
        self._E0H0, self._E1H0, self._E0H1, self._E1H1 = (
            self.impact_computation_strategy.compute_impacts(
                snapshot0,
                snapshot1,
                risk_transf_attach,
                risk_transf_cover,
                calc_residual,
            )
        )
        self._imp_mats_H0, self._imp_mats_H1 = None, None
        self._imp_mats_E0, self._imp_mats_E1 = None, None
        self._per_date_eai_H0, self._per_date_eai_H1 = None, None
        self._per_date_aai_H0, self._per_date_aai_H1 = None, None
        self._per_date_return_periods_H0, self._per_date_return_periods_H1 = None, None
        self._group_id_E0 = self.snapshot0.exposure.gdf["group_id"].values
        self._group_id_E1 = self.snapshot1.exposure.gdf["group_id"].values

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
            group_idx_E0 = np.where(self._group_id_E0 != group)
            group_idx_E1 = np.where(self._group_id_E1 != group)
            per_date_aai_H0, per_date_aai_H1 = (
                self.per_date_eai_H0[:, group_idx_E0].sum(),
                self.per_date_eai_H1[:, group_idx_E1].sum(),
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
        rp_0, rp_1 = self.per_date_return_periods_H0(
            return_periods
        ), self.per_date_return_periods_H1(return_periods)
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
