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
Cost functions for impact function calibration module
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_squared_log_error


def mse(data: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None) -> float:
    """Weighted mean squared error

    See
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    """
    return mean_squared_error(data, predicted, sample_weight=weights)


def msle(data: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None) -> float:
    """Weighted mean squared logarithmic error

    See
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html
    """
    return mean_squared_log_error(data, predicted, sample_weight=weights)
