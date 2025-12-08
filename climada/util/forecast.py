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

import numpy as np


class Forecast:
    """Mixin class for forecast data.

    Attributes
    ----------
    lead_time : np.ndarray
        Array of forecast lead times, given as datetime64 objects.
        Represents the time points for which forecasts are made.
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
