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
import functools
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.compromise_rankings import copeland
from pyrepo_mcda.mcda_methods import MULTIMOORA, SAW, SPOTIS, TOPSIS, VIKOR
from pyrepo_mcda.sensitivity_analysis_weights_values import (
    Sensitivity_analysis_weights_values,
)

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
        self._sensitivity_analyser = Sensitivity_analysis_weights_values()

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
        return pd.Series(
            [c.obj_maximise for c in self.criteria], index=self.criteria_cols
        )

    def add_criterion(
        self, criterion: Criterion, criterion_values: pd.Series, weight=None
    ):
        tmp = self.risk_metrics.copy()
        tmp[criterion.column_name] = tmp[self.options_col].map(criterion_values)
        self.risk_metrics = tmp
        bisect.insort(self._criteria, criterion, key=lambda x: x.column_name)
        if not weight:
            weight = 1 / (len(self.criteria_weights) + 1)

        tmp = self.criteria_weights.copy()
        tmp[criterion.column_name] = weight
        self.criteria_weights = tmp

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

    @staticmethod
    def sub_select_df(df, sub_selection):
        return df[
            functools.reduce(
                lambda x, y: x & y,
                (
                    df[col].isin([val] if not isinstance(val, list) else val)
                    for col, val in sub_selection.items()
                ),
            )
        ]

    def pivoted_risk_metrics(self, sub_selection=None):
        df = self.risk_metrics.copy().loc[
            :,
            [self.options_col]
            + self.criteria_cols
            + [self.metrics_col]
            + [self.groups_col],
        ]
        if sub_selection is not None:
            df = self.sub_select_df(df, sub_selection)

        index = self.options_col
        columns = [self.metrics_col, self.groups_col]
        df = df.pivot_table(index=index, columns=columns)
        # df.columns.name = [self.groups_col,self.metrics_col,"criteria"]
        df.columns = df.columns.reorder_levels(
            order=[self.groups_col, self.metrics_col, "criteria"]
        )
        df = df.sort_index(axis=1)
        return df

    def individual_rank(self, sub_selection=None):
        weights = self.weights.reset_index(name="weight")

        if sub_selection is not None:
            weights = self.sub_select_df(weights, sub_selection)

        reps = len(weights) / len(self.criteria_type)
        types = np.tile(np.where(self.criteria_type, 1, -1), reps=int(reps))
        risk_metrics = self.pivoted_risk_metrics(sub_selection).copy() * types
        return risk_metrics.rank(axis=0, ascending=False, method="max")

    def _mapped_criteria_types(self, weights):
        reps = len(weights) / len(self.criteria_type[weights["criteria"]])
        return np.tile(
            np.where(self.criteria_type[weights["criteria"]], 1, -1), reps=int(reps)
        )

    def calc_rankings(
        self,
        mcdm_methods=None,
        compromise_method=None,
        constraints=None,
        sub_selection=None,
    ):
        """
        Calculate rankings for a DecisionMatrix instance using specified Multi-Criteria Decision Making (MCDM) methods.

        Parameters:
        - mcdm_methods: dict, optional
            Dictionary of MCDM methods to use for ranking. Defaults to the MCDM_DEFAULT dictionary.
        - comp_ranks: dict, optional
            Dictionary of compromised ranking functions to use. Defaults to the COMP_DEFAULT dictionary.
        - constraints: list, optional
            List of constraints (pandas query strings) to filter the data. Defaults to an empty list.
        - sub_selection: dict, optional
            Dictionary of groups, metrics to sub-select

        Returns:
        - ranks_output: RanksOutput
            An instance of the RanksOutput class containing the rankings.
        """
        mcdm_methods = mcdm_methods if mcdm_methods else self.mcdm_methods
        compromise_method = (
            compromise_method if compromise_method else self.compromise_method
        )
        constraints = constraints if constraints else self.constraints
        weights = self.weights.reset_index(name="weight")
        if sub_selection is not None:
            weights = self.sub_select_df(weights, sub_selection)

        types = self._mapped_criteria_types(weights)
        risk_metrics = self.pivoted_risk_metrics(sub_selection).copy()
        risk_metrics = risk_metrics.replace(0, np.finfo(float).eps)
        if constraints:
            for constraint in constraints:
                # TODO update types if constraint
                risk_metrics = risk_metrics.query(constraint)

        # TODO
        mca_df = []  # self.risk_metrics.loc[:, self.options_col].copy()
        for method_name, method in mcdm_methods.items():
            mca_df.append(
                pd.Series(
                    self.rank_matrix(
                        risk_metrics,
                        weights=weights["weight"].to_numpy(),
                        types=types,
                        method=method,
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

    def calc_sensitivity(self, crit_col, method=TOPSIS()):
        # TODO Can only work for one selected group and metric, decide on how to implement for more than that
        sub_selection = {"group": "All", "metric": "aai"}
        matrix = self.pivoted_risk_metrics(sub_selection)
        weights = self.sub_select_df(
            self.weights.reset_index(name="weight"), sub_selection
        )
        weights_values = np.arange(0.05, 0.95, 0.1)
        types = self._mapped_criteria_types(weights)
        data_sens = self._sensitivity_analyser(
            matrix.values,
            weights_values,
            types,
            method,
            self.criteria_cols.index(crit_col),
        )
        data_sens.index = self.options_names
        return data_sens

    def plot_weight_sensitivity(self, crit_cols=None, method=TOPSIS()):
        crit_cols = crit_cols if crit_cols else self.criteria_cols
        for j, name in enumerate(crit_cols):
            data_sens = self.calc_sensitivity(name, method)
            plot_lineplot_sensitivity(
                data_sens, "TOPSIS", name, "Weight value", "value"
            )

    @staticmethod
    def criteria_col_in_risk_metrics(
        crit: Criterion, risk_metrics: pd.DataFrame
    ) -> bool:
        return crit.column_name in risk_metrics.columns

    @staticmethod
    def normalize_weights(weights: pd.Series) -> pd.Series:
        return weights / weights.sum()


def plot_lineplot_sensitivity(
    data_sens, method_name, criterion_name, x_title, filename=""
):
    """
    Visualization method to display line chart of alternatives rankings obtained with
    modification of weight of given criterion.

    Parameters
    ----------
        df_plot : DataFrame
            DataFrame containing rankings of alternatives obtained with different weight of
            selected criterion. The particular rankings are contained in subsequent columns of
            DataFrame.

        method_name : str
            Name of chosen MCDA method, i.e. `TOPSIS`, `VIKOR`, `CODAS`, `WASPAS`, `MULTIMOORA`, `MABAC`, `EDAS`, `SPOTIS`

        criterion_name : str
            Name of chosen criterion whose weight is modified

        x_title : str
            Title of x axis

        filename : str
            Name of file to save this chart

    Examples
    ----------
    >>> plot_lineplot_sensitivity(df_plot, method_name, criterion_name, x_title, filename)
    """
    plt.figure(figsize=(8, 4))
    for j in range(data_sens.shape[0]):

        plt.plot(data_sens.iloc[j, :], linewidth=2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(
            "  " + data_sens.index[j],
            (x_max, data_sens.iloc[j, -1]),
            fontsize=12,
            style="italic",
            horizontalalignment="left",
        )

    plt.xlabel(x_title, fontsize=12)
    plt.ylabel("Rank", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.title(method_name + ", modification of " + criterion_name + " weight")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    criterion_name = criterion_name.replace("$", "")
    criterion_name = criterion_name.replace("{", "")
    criterion_name = criterion_name.replace("}", "")
    # plt.savefig('./results/' + 'sensitivity_' + 'lineplot_' + method_name + '_' + criterion_name + '_' + filename + '.eps')
    plt.show()
