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

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from climada.engine.option_appraisal.MCDM.category import (
    CategorizedObject,
    CategoryLike,
    CategorySpace,
    CriteriaCategory,
)
from climada.engine.option_appraisal.MCDM.constants import (
    CRITERION_DEFAULT_COLNAME,
    DATE_DEFAULT_COLNAME,
    DEFAULT_CATEGORY_WEIGHT,
    DEFAULT_CRITERION_BASE_WEIGHT,
    FUTURE_DATE_DEFAULT_COLNAME,
    IMPORTANCE_MATCH,
    NO_MEASURE_DEFAULT_NAME,
    OPTIONS_DEFAULT_COLNAME,
    PRESENT_DATE_DEFAULT_COLNAME,
)
from climada.engine.option_appraisal.MCDM.mcda_methods import APPROACH_FN, MCDAApproach
from climada.engine.option_appraisal.MCDM.weights import WeightedItem

LOGGER = logging.getLogger(__name__)


class Criterion(CategorizedObject, WeightedItem):
    def __init__(
        self,
        name: str,
        categories: Optional[
            Union[CategoryLike, Sequence[CategoryLike], Set[CriteriaCategory]]
        ] = None,
        space: Optional[CategorySpace] = None,
        data: pd.Series = None,
        obj_maximize: bool = True,
        base_weight: float = DEFAULT_CRITERION_BASE_WEIGHT,
    ) -> None:
        CategorizedObject.__init__(self, name, categories, space)
        WeightedItem.__init__(self, base_weight)
        self.data = data
        self.data.name = name
        self.obj_maximize = obj_maximize

    def __repr__(self, indent=4) -> str:
        """
        Provides a custom, formatted, multi-line string representation
        of the Criterion object for better readability.
        """
        indent_space = " " * indent
        # Format the Categories Set
        # Use a list comprehension to format each category on a new line
        formatted_categories = ",\n".join(
            f"{cat.__repr__(indent=indent*2)}"
            for cat in sorted(
                list(self.categories), key=lambda c: c.name
            )  # Sorting for consistency
        )

        return (
            f"Criterion(\n"
            f"{indent_space}name='{self.name}',\n"
            f"{indent_space}categories={{\n{formatted_categories}\n{indent_space}}},\n"
            f"{indent_space}obj_maximise={self.obj_maximize}\n"
            f"{indent_space}base_weight={self.weight}\n"
            f"{indent_space}average_weight_from_categories={self.weight_from_category}\n"
            f")"
        )

    @property
    def category_weights(self):
        return {cat.name: cat.weight for cat in self.categories}

    @property
    def weight_from_category(self):
        return np.array([cat.effective_weight for cat in self.categories]).prod() ** (
            1 / len(self.categories)
        )


class CriteriaSet:
    def __init__(
        self,
        criteria: list[Criterion],
        category_weights: dict[str, float] | None = None,
        criteria_weights: dict[str, float] | None = None,
    ) -> None:
        self.criteria = criteria
        self._category_space = criteria[0].category_space
        # self.criteria_base_weights = criteria_weights
        self.category_weights = category_weights

    def display(self):
        lines = []

        total_weights = self.criteria_total_weights(active_only=False)

        active = [c for c in self.criteria if total_weights[c.name] > 0]
        inactive = [c for c in self.criteria if total_weights[c.name] == 0]

        # Header
        lines.append(
            f"CriteriaSet  {len(active)} active criteria  |  {len(self.category_space.all_categories)} categories"
        )
        if inactive:
            lines.append(f"             {len(inactive)} inactive criteria (weight = 0)")
        lines.append("=" * 60)

        # Category weights section
        lines.append("\nCategories")
        lines.append("-" * 60)
        cat_types = self.category_space.category_types
        for cat_type in sorted(t for t in cat_types if t is not None):
            cats = self.category_space.select_categories_by_type(cat_type)
            lines.append(f"  [{cat_type}]")
            for cat in sorted(cats, key=lambda c: c.name):
                bar = _weight_bar(cat.weight)
                lines.append(f"    {cat.name:<30}  {bar}  {cat.weight:.3f}")

        # Criteria weights section
        if len(active) > 0:
            bar_width = len(active) if len(active) < 50 else 50
            max_len = max(len(crit.name) for crit in active)
            total_sum = sum(total_weights[c.name] for c in active)
            lines.append(f"\nCriteria{' ' * (max_len)}Weights")
            lines.append("-" * 8 + " " * (max_len) + "-" * 9)
            total_weights = self.criteria_total_weights()
            for crit in sorted(active, key=lambda c: -total_weights[c.name]):
                total = total_weights[crit.name]
                effective = total / total_sum if total_sum > 0 else 0.0
                bar = _weight_bar(effective, width=len(active) * 2)
                lines.append(
                    f"  {crit.name:<{max_len+4}}  "
                    f"base={crit.weight:.5f}  total={total_weights[crit.name]:.5f}  "
                    f"effective={effective:.5f}   "
                    f"{bar}"
                )

            lines.append("")
        print("\n".join(lines))

    @classmethod
    def from_risk_metrics(
        cls,
        risk_metrics: pd.DataFrame,
        category_types: list[str],
        criteria_cols: list[str],
        options_colname: str = OPTIONS_DEFAULT_COLNAME,
        excluded_value_cols=None,
        criteria_min: Optional[list[str]] = None,
    ) -> "CriteriaSet":
        criteria_min = [] if criteria_min is None else criteria_min
        if excluded_value_cols:
            risk_metrics = risk_metrics[
                [col for col in risk_metrics.columns if col not in excluded_value_cols]
            ].copy()
        if (
            DATE_DEFAULT_COLNAME in risk_metrics.columns
            and DATE_DEFAULT_COLNAME not in category_types
        ):
            if risk_metrics[DATE_DEFAULT_COLNAME].nunique() > 1:
                LOGGER.info(
                    f"'{DATE_DEFAULT_COLNAME}' column with more than one value found in risk metric dataframe. Will apply default treatment: will define a category '{PRESENT_DATE_DEFAULT_COLNAME}' for earliest date and '{FUTURE_DATE_DEFAULT_COLNAME}' for latest one. You can make every date a category by explicitly including the '{DATE_DEFAULT_COLNAME}' column in the category_types."
                )
                max_date = risk_metrics[DATE_DEFAULT_COLNAME].max()
                min_date = risk_metrics[DATE_DEFAULT_COLNAME].min()
                risk_metrics = risk_metrics.loc[
                    risk_metrics[DATE_DEFAULT_COLNAME].isin([min_date, max_date])
                ]
                risk_metrics[DATE_DEFAULT_COLNAME] = risk_metrics[
                    DATE_DEFAULT_COLNAME
                ].map(
                    {
                        min_date: PRESENT_DATE_DEFAULT_COLNAME,
                        max_date: FUTURE_DATE_DEFAULT_COLNAME,
                    }
                )
                category_types.append(DATE_DEFAULT_COLNAME)

        cols_not_used = [
            col
            for col in risk_metrics.columns
            if col
            not in category_types
            + criteria_cols
            + [options_colname]
            + excluded_value_cols
        ]
        for col in cols_not_used:
            if risk_metrics[col].nunique() > 1:
                raise ValueError(
                    f"Column {col} is not defined as a category type or a criteria nor is excluded. As it has more that one unique value, I don't know how to handle it. Either add it to excluded_value_cols if it is a criterion values column you do not want or conversely to criteria_cols. If it is an identifier column either add it to category_type or subselect the dataframe to have a unique value."
                )

        risk_metrics = risk_metrics[
            category_types + criteria_cols + [options_colname]
        ].copy()
        risk_metrics = risk_metrics.loc[
            risk_metrics[OPTIONS_DEFAULT_COLNAME] != NO_MEASURE_DEFAULT_NAME
        ]
        risk_metrics = risk_metrics.melt(
            id_vars=category_types + [options_colname],
            value_vars=criteria_cols,
            var_name=CRITERION_DEFAULT_COLNAME,
        )
        groups = risk_metrics.set_index(options_colname).groupby(
            category_types + [CRITERION_DEFAULT_COLNAME], as_index=False, observed=False
        )["value"]
        for gr in category_types:
            LOGGER.info(
                f"Categories found in type '{gr}': {risk_metrics[gr].astype(str).unique()}"
            )

        LOGGER.info(f"Total number of possible criteria: {len(groups)}")

        cat_space = CategorySpace()
        crits = []
        for group_name, group in groups:
            cats = [
                (str(col), str(val))
                for col, val in zip(
                    category_types + [CRITERION_DEFAULT_COLNAME], group_name
                )
            ]
            for cat in cats:
                cat_space.add_category(name=cat[1], category_type=cat[0])
            crit_fullname = f"{'-'.join(['_'.join(c) for c in cats])}"
            crits.append(
                Criterion(
                    crit_fullname,
                    categories=[cat[1] for cat in cats],
                    data=group,
                    space=cat_space,
                    obj_maximize=(all(c not in crit_fullname for c in criteria_min)),
                )
            )

        return cls(criteria=crits)

    @property
    def category_space(self):
        return self._category_space

    @property
    def criteria(self):
        return self._criteria

    @criteria.setter
    def criteria(self, value, /):
        self._check_consistency(value)
        self._criteria = [crit for crit in value]

    @property
    def criteria_names(self):
        return [crit.name for crit in self.criteria]

    @property
    def criteria_matrix(self):
        return pd.concat([crit.data for crit in self.criteria], axis=1)

    @property
    def criteria_types(self):
        return np.array([1 if crit.obj_maximize else -1 for crit in self.criteria])

    @property
    def criteria_with_weight(self):
        return list(self.criteria_total_weights().keys())

    def add_criteria(self, criteria: Criterion | list[Criterion]):
        if not isinstance(criteria, list):
            criteria = [criteria]

        # TODO: Warn duplicates
        # Overwrite?
        criteria = [c for c in criteria if c.name not in self.criteria_names]
        self.criteria = self.criteria + criteria

    @staticmethod
    def _check_consistency(criteria):
        if not isinstance(criteria, list):
            raise ValueError("Criteria must be a list of Criterion.")

        if not all([isinstance(criterion, Criterion) for criterion in criteria]):
            raise ValueError("Criteria must be a list of Criterion.")

        first_index = criteria[0].data.index.sort_values()
        mismatched_indices_info = []

        # Iterate through the criteria starting from the second element (index 1)
        for i, criterion in enumerate(criteria[1:]):
            current_index = criterion.data.index.sort_values()

            if not first_index.equals(current_index):
                # Find the specific differences (elements in one index but not the other)
                diff_1 = first_index.difference(current_index)
                diff_2 = current_index.difference(first_index)

                mismatched_indices_info.append(
                    f"Criterion {i + 1} (Name: {criterion.name if hasattr(criterion, 'name') else 'N/A'}) "
                    f"has an index mismatch with Criterion 0."
                    f"\n  -> Unique to Criterion 0: {list(diff_1)}"
                    f"\n  -> Unique to Criterion {i + 1}: {list(diff_2)}"
                )

        if mismatched_indices_info:
            infos = "\n".join(mismatched_indices_info)
            raise ValueError(
                "All criteria must have the same index (options) to be combined."
                f"\n\nDetails of Mismatches:\n\n{infos}"
            )

        if not all([criteria[0].space is criterion.space for criterion in criteria]):
            raise ValueError("Criteria must share the same space of categories.")

    def criteria_total_weights(
        self,
        categories_influence: float = 0.5,
        base_weight_influence: float = 0.5,
        active_only: bool = True,
    ) -> dict[str, float]:
        """Compute weighted combination of base weight and category-derived weight.

        Parameters
        ----------
        categories_influence : float
            Weight given to category-derived score. Must sum to 1 with
            ``base_weight_influence``.
        base_weight_influence : float
            Weight given to the criterion's base weight.
        active_only : bool
            If True (default), only criteria with non-zero total weight are returned.

        Returns
        -------
        dict[str, float]
            Mapping of criterion name to total weight.
        """
        if categories_influence + base_weight_influence != 1.0:
            raise ValueError(
                "categories_influence and base_weight_influence must sum to 1."
            )

        weights = {
            crit.name: crit.weight * base_weight_influence
            + crit.weight_from_category * categories_influence
            for crit in self.criteria
        }

        if active_only:
            return {k: v for k, v in weights.items() if v > 0}
        return weights

        weights = {
            crit.name: crit.weight * base_weight_influence
            + crit.weight_from_category * categories_influence
            for crit in self.criteria
        }

        if active_only:
            return {k: v for k, v in weights.items() if v > 0}
        return weights

    @property
    def criteria_base_weights(self):
        return {v.name: v.weight for v in self.criteria}

    def get_criteria(self, name):
        for crit in self.criteria:
            if crit.name == name:
                return crit

    @property
    def category_effective_weights(self):
        return self.category_space.effective_weights

    def set_criterion_weight(self, name: str, weight) -> None:
        """Set the base weight of a single criterion.

        Parameters
        ----------
        name : str
            Criterion name.
        weight : float or str
            New weight value.
        """
        crit = self.get_criteria(name)
        if crit is None:
            raise KeyError(f"Criterion '{name}' not found.")
        crit.weight = weight

    def reset_category_weights(self) -> None:
        self.category_space.reset_weights()

    def all_equal_category_weights(self) -> None:
        self.category_space.reset_weights(weight=1.0)

    def update_category_weights(self, weights: dict[str, float]) -> None:
        """Set the weight of a single category.

        Parameters
        ----------
        name : str
            Category name.
        weight : float or str
            New weight value.
        """
        for name, weight in weights.items():
            self.category_space.set_weight(name, weight)

    def get_criteria_by_category(
        self,
        categories: Union[
            "CriteriaCategory", str, List[Union["CriteriaCategory", str]]
        ],
    ) -> List["Criterion"]:
        """
        Retrieves all Criterion objects that are linked to any of the
        given categories or any of their subcategories.

        :param categories: A single CriteriaCategory/string or a list of them
                           to filter by.
        :return: A list of matching Criterion objects.
        """

        # Handle the case where a single string is passed (not a list)
        if not isinstance(categories, list):
            categories = [categories]

        matching_criteria = [
            criterion
            for criterion in self.criteria
            # Check if the criterion matches ANY category in the filter list
            if any(
                # We rely on the criterion's method which uses the recursive
                # is_descendant_of logic.
                criterion.has_category(cat)
                for cat in categories
            )
        ]

        return matching_criteria

    @property
    def active_criteria_matrix(self) -> pd.DataFrame:
        """Criteria matrix sub-selected for criteria with non-zero total weight.

        Returns
        -------
        pd.DataFrame
            Columns are active criterion names, index is options.
        """
        active_names = self.criteria_with_weight
        return self.criteria_matrix[active_names]

    def normalized_criteria_matrix(
        self,
        scaler=None,
    ) -> pd.DataFrame:
        """Criteria matrix normalized using a scikit-learn-compatible scaler.

        Only active criteria (non-zero total weight) are included.

        Parameters
        ----------
        scaler : sklearn-compatible transformer, optional
            Must implement ``fit_transform(X)``. Defaults to
            ``sklearn.preprocessing.MinMaxScaler()``.

        Returns
        -------
        pd.DataFrame
            Normalized criteria matrix with same index and columns as
            ``active_criteria_matrix``.
        """
        if scaler is None:
            scaler = MinMaxScaler()

        matrix = self.active_criteria_matrix
        return pd.DataFrame(
            scaler.fit_transform(matrix),
            index=matrix.index,
            columns=matrix.columns,
        )

    def score_matrix(
        self,
        approach: MCDAApproach | str = MCDAApproach.SAW,
        scaler=None,
    ) -> pd.Series:
        """Score options using a MCDA approach.

        Parameters
        ----------
        approach : MCDAApproach or str
            Scoring method. One of ``MCDAApproach.SAW`` or ``MCDAApproach.TOPSIS``.
            Strings ``"saw"`` and ``"topsis"`` are also accepted.
        scaler : sklearn-compatible transformer, optional
            Scaler passed to ``normalized_criteria_matrix``.
            Defaults to ``MinMaxScaler()``.

        Returns
        -------
        pd.Series
            Scores indexed by option, sorted descending.
        """
        if isinstance(approach, str):
            approach = MCDAApproach(approach.lower())

        fn = APPROACH_FN[approach]

        matrix = self.normalized_criteria_matrix(scaler=scaler)
        active_criteria = [c for c in self.criteria if c.name in matrix.columns]

        total_weights = self.criteria_total_weights()
        raw_weights = np.array([total_weights[c.name] for c in active_criteria])
        weights = raw_weights / raw_weights.sum()  # normalise to sum=1

        criteria_types = np.array(
            [1 if c.obj_maximize else -1 for c in active_criteria]
        )

        scores = fn(matrix.values, weights, criteria_types)
        return pd.Series(scores, index=matrix.index, name=approach.value).sort_values(
            ascending=False
        )

    def display_space(self) -> None:
        """
        Prints the entire category hierarchy registered in the system using ASCII art.
        It handles multiple roots and is robust against multiple inheritance.
        """
        self.category_space.display()


def _weight_bar(weight: float, width: int = 8) -> str:
    """ASCII progress bar for a weight in [0, 1].

    Parameters
    ----------
    weight : float
        Value in [0, 1].
    width : int
        Total bar characters.

    Returns
    -------
    str
        e.g. ``[██░░░░░░]``
    """
    filled = round(weight * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"
