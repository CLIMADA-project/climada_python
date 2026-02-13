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

Define functions to parse strings
"""

from itertools import chain


def parse_mapping_string(map_str: str | None) -> dict[int, int] | None:
    """
    Parses strings like "1to8, 5to2" into {1: 8, 5: 2}.
    """
    if map_str is None or map_str == "" or map_str == "nil":
        return None
    # 1. Split by comma to get individual pairs: ['1to8', '5to2']
    pairs = (p.strip() for p in map_str.split(",") if p.strip())

    # 2. Split each pair by 'to' and convert to integers
    def split_pair(p):
        k, v = p.split("to")
        return int(k), int(v)

    # 3. Construct the dictionary
    return dict(split_pair(p) for p in pairs)


def parse_range(s: str | None) -> list | None:
    """
    Parses strings like "1,4-6,10,12-14" into [1,4,5,6,10,12,13,14].
    """
    if s is None or s == "nil" or s == "":
        return None
    try:
        return list(
            chain.from_iterable(
                range(r[0], r[-1] + 1)
                for r in [[int(i) for i in part.split("-")] for part in s.split(",")]
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid string format for zeroing assets: {s} (Ex: '1,4-6,10')"
        ) from exc
