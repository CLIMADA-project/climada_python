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
from typing import List, Optional, Set, Union

import pandas as pd

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
from climada.engine.option_appraisal.MCDM.weights import WeightedItem

LOGGER = logging.getLogger(__name__)


class Criterion(CategorizedObject, WeightedItem):
    def __init__(
        self,
        name: str,
        categories: Optional[
            Union[CategoryLike, List[CategoryLike], Set[CriteriaCategory]]
        ] = None,
        space: Optional[CategorySpace] = None,
        data: pd.Series = None,
        obj_maximise: bool = True,
        base_weight: float = DEFAULT_CRITERION_BASE_WEIGHT,
    ) -> None:
        CategorizedObject.__init__(self, name, categories, space)
        WeightedItem.__init__(self, base_weight)
        self.data = data
        self.data.name = name
        self.obj_maximize = obj_maximise

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
            f"{indent_space}total_weight_from_categories={self.average_weight_from_category}\n"
            f")"
        )

    @property
    def category_weights(self):
        return {cat.name: cat.weight for cat in self.categories}

    @property
    def average_weight_from_category(self):
        return sum([cat.weight for cat in self.categories]) / len(self.categories)


class CriteriaSet:
    def __init__(
        self,
        criteria: list[Criterion],
        categories_weights: dict[str, float] | None = None,
        criteria_weights: dict[str, float] | None = None,
    ) -> None:
        self.criteria = criteria
        self._category_space = criteria[0].category_space
        self.criteria_weights = criteria_weights
        self.categories_weights = categories_weights

    @classmethod
    def from_risk_metrics(
        cls,
        risk_metrics: pd.DataFrame,
        category_types: list[str],
        criteria_cols: list[str],
        options_colname: str = OPTIONS_DEFAULT_COLNAME,
        excluded_value_cols=None,
    ) -> "CriteriaSet":
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
            crits.append(
                Criterion(
                    f"{'-'.join(['_'.join(c) for c in cats])}",
                    categories=[cat[1] for cat in cats],
                    data=group,
                    space=cat_space,
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
        self._criteria = {crit.name: crit for crit in value}
        self._criteria_matrix = pd.concat([item.data for item in value], axis=1)

    @property
    def criteria_matrix(self):
        return self._criteria_matrix

    @staticmethod
    def _check_consistency(criteria):
        if not isinstance(criteria, list):
            raise ValueError("Criteria must be a list of Criterion.")

        if not all([isinstance(criterion, Criterion) for criterion in criteria]):
            raise ValueError("Criteria must be a list of Criterion.")

        if not all(
            [
                criteria[0].data.index.equals(criterion.data.index)
                for criterion in criteria
            ]
        ):
            raise ValueError("Criteria do not all have the same index (options).")

        if not all([criteria[0].space is criterion.space for criterion in criteria]):
            raise ValueError("Criteria must share the same space of categories.")

    @property
    def criteria_weights(self):
        return {k: v.weight for k, v in self.criteria}

    @criteria_weights.setter
    def criteria_weights(self, value, /):
        if value is None:
            LOGGER.info(
                f"Resetting criteria base weight to default value ({DEFAULT_CRITERION_BASE_WEIGHT})"
            )
            for crit in self.criteria.values():
                crit.weight = DEFAULT_CRITERION_BASE_WEIGHT
        else:
            no_match = [k for k in value.keys() if k not in self.criteria.keys()]
            no_weight = [k for k in self.criteria.keys() if k not in value.keys()]
            if len(no_match) > 0:
                LOGGER.warning(
                    f"Some weights do not correspond to any criteria: {no_match}"
                )

            if len(no_weight) > 0:
                LOGGER.warning(
                    f"No weight given for one or more criteria (will use existing or default ({DEFAULT_CRITERION_BASE_WEIGHT})) {no_weight}"
                )

            for k, v in value.items():
                self.criteria[k].weight = v

    @property
    def category_weights(self):
        return self.category_space.category_weights

    @category_weights.setter
    def category_weights(self, value, /):
        self.category_space.category_weights = value

    def set_weight_by_category_type(self, value: dict[str, float]):
        no_match = [
            k for k in value.keys() if k not in self.category_space.category_types
        ]
        if len(no_match) > 0:
            LOGGER.warning(
                f"Some weights do not correspond to any category type: {no_match}"
            )

        categories = {
            cat_type: self.category_space.select_categories_by_type(cat_type)
            for cat_type in value.keys()
            if cat_type not in no_match
        }
        category_weights = {
            cat.name: value[cat_type]
            for cat_type in value.keys()
            if cat_type not in no_match
            for cat in categories[cat_type]
        }
        self.category_weights = self.category_weights | category_weights

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

    def display_space(self) -> None:
        """
        Prints the entire category hierarchy registered in the system using ASCII art.
        It handles multiple roots and is robust against multiple inheritance.
        """
        self.category_space.display()
