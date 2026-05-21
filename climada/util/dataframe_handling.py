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

Define functions to handle with coordinates
"""

import pandas as pd


def reorder_dataframe_columns(
    dataframe: pd.DataFrame, priority_order: list[str], keep_remaining: bool = True
) -> pd.DataFrame:
    """
    Applies a column priority list to a DataFrame to reorder its columns.

    This function is robust to cases where:
    1. Columns in 'priority_order' are not in the DataFrame (they are ignored).
    2. Columns in the DataFrame are not in 'priority_order'.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The input DataFrame.
    priority_order: list[str]
        A list of strings defining the desired column
        order. Columns listed first have higher priority.
    keep_remaining: bool
        If True, any columns in the DataFrame but NOT in
        'priority_order' will be appended to the end in their
        original relative order. If False, these columns
        are dropped.

    Returns:
        pd.DataFrame: The DataFrame with columns reordered according to the priority list.
    """

    present_priority_columns = [
        col for col in priority_order if col in dataframe.columns
    ]

    new_column_order = present_priority_columns

    if keep_remaining:
        remaining_columns = [
            col for col in dataframe.columns if col not in present_priority_columns
        ]

        new_column_order.extend(remaining_columns)

    return dataframe[new_column_order]
