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

"""

import bisect
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.compromise_rankings import copeland
from pyrepo_mcda.mcda_methods import MULTIMOORA, SAW, SPOTIS, TOPSIS, VIKOR

from climada.engine.option_appraisal.MCDM.criterion import Criterion

MCDM_DEFAULT = {"Topsis": TOPSIS(), "Saw": SAW(), "Vikor": VIKOR()}  #'AHP': AHP(),
"""Default MCDM ranking method"""

COMPROMISE_DEFAULT = {
    "copeland": copeland,
}
"""Default Compromise approach"""


class MCA_Calc:
    def __init__(
        self,
        risk_metrics: pd.DataFrame,
        criteria: list[Criterion],
        criteria_weights: dict[str, float] | None = None,
        metrics_col: str | None = None,
        metrics_weights: dict[str, float] | None = None,
        constraints: list[str] | None = None,
        groups_col: str | None = None,
        groups_weights: dict[str, float] | None = None,
        options_col: str = "measure",
        mcdm_methods: dict | None = None,
        compromise_method: dict | None = None,
    ) -> None:
        self._criteria_weights = None
        self._groups_weights = None
        self._metrics_weights = None

        self._norm_criteria_weights = None
        self._norm_groups_weights = None
        self._norm_metrics_weights = None
        self.risk_metrics = risk_metrics.copy()
        self.criteria = criteria
        self.options_col = options_col
        self.options_names = self.risk_metrics[options_col].unique()
        self.metrics_col = metrics_col
        self.metrics_names = self.risk_metrics[metrics_col].unique()
        self.groups_col = groups_col
        self.groups_names = self.risk_metrics[groups_col].unique()

        self.criteria_weights = (
            criteria_weights
            if criteria_weights
            else pd.Series(
                [1 / len(criteria)] * len(criteria),
                index=self.criteria_cols,
                name="criteria weights",
            )
        )
        self.groups_weights = groups_weights
        self.metrics_weights = metrics_weights
        self.constraints = constraints
        self.mcdm_methods = mcdm_methods if mcdm_methods else MCDM_DEFAULT
        self.compromise_method = (
            compromise_method if compromise_method else COMPROMISE_DEFAULT
        )

    @property
    def criteria(self) -> list[Criterion]:
        return self._criteria

    @criteria.setter
    def criteria(self, value: Any, /):
        if not isinstance(value, Iterable) and not all(
            isinstance(c, Criterion) for c in value
        ):
            raise ValueError("Criterias should be a list of Criterion objects")

        fail_crit = [
            c
            for c in value
            if not self.criteria_col_in_risk_metrics(c, self.risk_metrics)
        ]
        if len(fail_crit) > 0:
            raise ValueError(f"{fail_crit} not found in risk metric dataframe")

        self._criteria = sorted(value, key=lambda x: x.column_name)

    @property
    def criteria_names(self) -> list[str]:
        return [c.name for c in self.criteria]

    @property
    def criteria_cols(self) -> list[str]:
        return [c.column_name for c in self.criteria]

    @property
    def criteria_type(self) -> list[bool]:
        return [c.obj_maximise for c in self.criteria]

    def add_criterion(self, criterion: Criterion, criterion_values: pd.Series):
        tmp = self.risk_metrics.copy()
        tmp[criterion.column_name] = tmp[self.options_col].map(criterion_values)
        self.risk_metrics = tmp
        bisect.insort(self._criteria, criterion, key=lambda x: x.column_name)

    @property
    def criteria_weights(self) -> pd.Series:
        return self._criteria_weights

    @criteria_weights.setter
    def criteria_weights(self, value: Any, /):
        if self.criteria_cols != list(value.keys()):
            fail_weights = [k for k, _ in value.items() if k not in self.criteria_cols]
            fail_crit = [k for k in self.criteria_cols if k not in value.keys()]
            if len(fail_weights) > 0:
                raise ValueError(f"{fail_weights} not found in criteria columns")
            if len(fail_crit) > 0:
                raise ValueError(f"{fail_crit} not found in weights")

        self._criteria_weights = pd.Series(value, name="criteria weights").sort_index()
        self._norm_criteria_weights = self.normalize_weights(self._criteria_weights)
        self._update_norm_weights()

    @property
    def groups_weights(self) -> pd.Series:
        return self._groups_weights

    @groups_weights.setter
    def groups_weights(self, value: Any, /):
        if value is None and len(self.groups_names) == 0:
            self._groups_weights = None
            self._norm_groups_weights = None

        if value is None:
            self._groups_weights = pd.Series(
                [1 / len(self.groups_names)] * len(self.groups_names),
                index=self.groups_names,
                name="group weights",
            )

        if value:
            if self.groups_col != list(value.keys()):
                fail_weights = [
                    k for k, _ in value.items() if k not in self.groups_names
                ]
                fail_group = [k for k in self.groups_names if k not in value.keys()]
                if len(fail_weights) > 0:
                    raise ValueError(f"{fail_weights} not found in criteria columns")
                if len(fail_group) > 0:
                    raise ValueError(f"{fail_group} not found in weights")

            self._groups_weights = pd.Series(value, name="group weights").sort_index()

        self._norm_groups_weights = self.normalize_weights(self._groups_weights)
        self._update_norm_weights()

    @property
    def metrics_weights(self) -> pd.Series:
        return self._metrics_weights

    @metrics_weights.setter
    def metrics_weights(self, value: Any, /):
        if value is None:
            self._metrics_weights = pd.Series(
                [1 / len(self.metrics_names)] * len(self.metrics_names),
                index=self.metrics_names,
                name="metric weights",
            )

        if value:
            if list(self.metrics_names) != list(value.keys()):
                fail_weights = [
                    k for k, _ in value.items() if k not in self.metrics_names
                ]
                fail_metrics = [k for k in self.metrics_names if k not in value.keys()]
                if len(fail_weights) > 0:
                    raise ValueError(f"{fail_weights} not found in criteria columns")
                if len(fail_metrics) > 0:
                    raise ValueError(f"{fail_metrics} not found in weights")

            self._metrics_weights = pd.Series(value, name="metric weights").sort_index()

        self._norm_metrics_weights = self.normalize_weights(self._metrics_weights)
        self._update_norm_weights()

    # @property
    # def groups_col(self) -> list[str]:
    #     return self._groups

    # @groups_col.setter
    # def groups_col(self, value: Any, /):
    #     if value:
    #         fail_groups = [
    #             k for k in value if k not in self.risk_metrics.columns
    #         ]
    #         if len(fail_groups) > 0:
    #             raise ValueError(f"{fail_groups} not found in risk metric dataframe")

    #         self._groups = value

    def criteria_from_group(self, group_name):
        pass
        tmp = self.risk_metrics.copy()

    @property
    def scenario_cols(self) -> list[str]:
        return self._scenario_cols

    @scenario_cols.setter
    def scenario_cols(self, value: Any, /):
        if value:
            fail_sce = [k for k in value if k not in self.risk_metrics.columns]
            if len(fail_sce) > 0:
                raise ValueError(f"{fail_sce} not found in risk metric dataframe")

        self._scenario_cols = value

    @property
    def weights(self):
        return self._normalized_weights

    def _update_norm_weights(self):
        if (
            self.criteria_weights is not None
            and self.metrics_weights is not None
            and self.groups_weights is not None
        ):
            index = self.pivoted_risk_metrics().columns
            values = [
                self._norm_groups_weights[idx[0]]
                * self._norm_metrics_weights[idx[1]]
                * self._norm_criteria_weights[idx[2]]
                for idx in index
            ]
            self._normalized_weights = pd.Series(values, index=index)
        else:
            self._normalized_weights = None

    def pivoted_risk_metrics(self):
        df = self.risk_metrics.copy().loc[
            :,
            [self.options_col]
            + self.criteria_cols
            + [self.metrics_col]
            + [self.groups_col],
        ]
        index = self.options_col
        columns = [self.metrics_col, self.groups_col]
        df = df.pivot_table(index=index, columns=columns)
        # df.columns.name = [self.groups_col,self.metrics_col,"criteria"]
        df.columns = df.columns.reorder_levels(
            order=[self.groups_col, self.metrics_col, None]
        )
        df = df.sort_index(axis=1)
        return df

    def calc_rankings(
        self,
        mcdm_methods=None,
        compromise_method=None,
        constraints=None,
        rank_filt=None,
        derived_columns=None,
    ):
        """
        Calculate rankings for a DecisionMatrix instance using specified Multi-Criteria Decision Making (MCDM) methods.

        Parameters:
        - mcdm_methods: dict, optional
            Dictionary of MCDM methods to use for ranking. Defaults to the MCDM_DEFAULT dictionary.
        - comp_ranks: dict, optional
            Dictionary of compromised ranking functions to use. Defaults to the COMP_DEFAULT dictionary.
        - constraints: dict, optional
            Dictionary of constraints to filter the data. Defaults to an empty dictionary.
        - rank_filt: dict, optional
            Dictionary of filters to apply to the ranking. Defaults to an empty dictionary.
        - derived_columns: dict, optional
            Dictionary of derived columns to calculate. Defaults to an empty dictionary.

        Returns:
        - ranks_output: RanksOutput
            An instance of the RanksOutput class containing the rankings.
        """
        mcdm_methods = mcdm_methods if mcdm_methods else self.mcdm_methods
        compromise_method = (
            compromise_method if compromise_method else self.compromise_method
        )
        constraints = constraints if constraints else self.constraints
        weights = self.weights.to_numpy()
        reps = len(self.weights) / len(self.criteria_type)
        types = np.tile(np.where(self.criteria_type, 1, -1), reps=int(reps))
        risk_metrics = self.pivoted_risk_metrics().copy()
        risk_metrics = risk_metrics.replace(0, np.finfo(float).eps)
        if constraints:
            for constraint in constraints:
                risk_metrics = risk_metrics.query(constraint)

        # TODO
        mca_df = []  # self.risk_metrics.loc[:, self.options_col].copy()
        for method_name, method in mcdm_methods.items():
            mca_df.append(
                pd.Series(
                    self.rank_matrix(
                        risk_metrics, weights=weights, types=types, method=method
                    ),
                    name=method_name,
                )
            )

        mca_df = pd.concat(mca_df, axis=1)

        if compromise_method:
            compromise = []
            for comp_method_name, comp_method in compromise_method.items():
                compromise.append(
                    pd.Series(
                        comp_method(mca_df[list(mcdm_methods.keys())]),
                        name=comp_method_name,
                        index=risk_metrics.index,
                    )
                )
            mca_df = pd.concat([mca_df, pd.concat(compromise, axis=0)], axis=1)

        return mca_df

        # for method_name, method in mcdm_methods.items():
        #     grouped = risk_metrics.groupby(
        #         self.scenario_cols, as_index=False, group_keys=False
        #     )[self.criteria_cols]
        #     tmp = []
        #     for _, group in grouped:
        #         group[method_name] = self.rank_matrix(group, weights=weights, types=types, method=method)
        #         tmp.append(group[method_name])
        #     mca_df.append(pd.concat(tmp,axis=0))
        # mca_df = pd.concat(mca_df, axis=1)
        # mca_df = pd.concat(
        #     [risk_metrics.loc[:, [self.options_col] + self.scenario_cols], mca_df],
        #     axis=1,
        # )
        # if compromise_method:
        #     compromise = []
        #     for method_name, method in compromise_method.items():
        #         grouped = mca_df.groupby(
        #             self.scenario_cols, as_index=False, group_keys=False
        #         )[list(mcdm_methods.keys())]
        #         for _, group in grouped:
        #             group[method_name] = method(group)
        #             compromise.append(group[method_name])

        #     mca_df = pd.concat([mca_df, pd.concat(compromise, axis=0)], axis=1)
        # return mca_df

    @staticmethod
    def rank_matrix(df, weights, types, method):
        matrix = df.to_numpy()
        if isinstance(method, SPOTIS):
            bounds_min = np.amin(matrix, axis=0)
            bounds_max = np.amax(matrix, axis=0)
            bounds = np.vstack((bounds_min, bounds_max))
            # Calculate the preference values of alternatives
            prefs = method(matrix, weights, types, bounds)
        else:
            prefs = method(matrix, weights, types)
        if isinstance(method, MULTIMOORA):
            return pd.Series(prefs, index=df.index)
        elif isinstance(method, (VIKOR, SPOTIS)):
            prefs = rank_preferences(prefs, reverse=False)
            return pd.Series(prefs, index=df.index)
        else:
            prefs = rank_preferences(prefs, reverse=True)
            return pd.Series(prefs, index=df.index)

    @staticmethod
    def criteria_col_in_risk_metrics(
        crit: Criterion, risk_metrics: pd.DataFrame
    ) -> bool:
        return crit.column_name in risk_metrics.columns

    @staticmethod
    def normalize_weights(weights: pd.Series) -> pd.Series:
        return weights / weights.sum()
