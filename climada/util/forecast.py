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

Define Forecast base class.
"""

from typing import Any

import numpy as np


def check_attribute_shapes(obj_act: Any, attr_act: str, obj_exp: Any, attr_exp: str):
    """Compare the shapes of attributes of two objects.

    Raises
    ------
    ValueError
        If the shapes do not match
    """
    shape_actual = getattr(obj_act, attr_act).shape
    shape_expected = getattr(obj_exp, attr_exp).shape
    if shape_actual != shape_expected:
        raise ValueError(
            f"Shape mismatch between {type(obj_act).__name__}.{attr_act} "
            f"{shape_actual} and {type(obj_exp).__name__}.{attr_exp} {shape_expected}"
        )


class Forecast:
    """Mixin class for forecast data.

    Attributes
    ----------
    lead_time : np.ndarray
        Array of forecast lead times, given as timedelta64 objects.
        Represents the lead times of the forecasts.
    member : np.ndarray
        Array of ensemble member identifiers, given as integers.
        Represents different forecast ensemble members.
    """

    def __init__(
        self,
        lead_time: np.ndarray | None = None,
        member: np.ndarray | None = None,
        **kwargs,
    ):
        """Initialize Forecast.

        Parameters
        ----------
        lead_time : np.ndarray or None, optional
            Forecast lead times. Default is empty array.
        member : np.ndarray or None, optional
            Ensemble member identifiers. Default is empty array.
        """

        self.lead_time = (
            np.asarray(lead_time) if lead_time is not None else np.array([])
        )
        self.member = np.asarray(member) if member is not None else np.array([])
        super().__init__(**kwargs)
