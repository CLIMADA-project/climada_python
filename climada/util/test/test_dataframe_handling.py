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

Test coordinates module.
"""

import pandas as pd

from climada.util.dataframe_handling import reorder_dataframe_columns


def test_reorder_dataframe_columns():
    # Setup: Create a sample DataFrame
    df = pd.DataFrame(
        {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8], "E": [9, 10]}
    )

    # Test Case 1: Standard reordering with keep_remaining=True
    # Priority: C, A (should move C and A to front, keep B, D, E in original order)
    priority = ["C", "A", "Z"]  # 'Z' is not in df, should be ignored
    result = reorder_dataframe_columns(df, priority, keep_remaining=True)

    expected_cols = ["C", "A", "B", "D", "E"]
    assert (
        list(result.columns) == expected_cols
    ), f"Expected {expected_cols}, got {list(result.columns)}"

    # Test Case 2: Dropping remaining columns (keep_remaining=False)
    # Priority: B, D (should only keep B and D)
    priority = ["B", "D"]
    result = reorder_dataframe_columns(df, priority, keep_remaining=False)

    expected_cols = ["B", "D"]
    assert (
        list(result.columns) == expected_cols
    ), f"Expected {expected_cols}, got {list(result.columns)}"

    # Test Case 3: All priority columns missing
    priority = ["X", "Y"]
    result = reorder_dataframe_columns(df, priority, keep_remaining=True)

    # Should return original order since no priority matches
    expected_cols = ["A", "B", "C", "D", "E"]
    assert list(result.columns) == expected_cols

    # Test Case 4: Empty priority list
    priority = []
    result = reorder_dataframe_columns(df, priority, keep_remaining=True)

    expected_cols = ["A", "B", "C", "D", "E"]
    assert list(result.columns) == expected_cols
