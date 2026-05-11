from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class MCDAApproach(Enum):
    SAW = "saw"
    TOPSIS = "topsis"


def _saw(
    matrix: np.ndarray,
    weights: np.ndarray,
    criteria_types: np.ndarray,
) -> np.ndarray:
    """Simple Additive Weighting.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_options, n_criteria)
        Normalized criteria values.
    weights : np.ndarray, shape (n_criteria,)
        Normalized weights summing to 1.
    criteria_types : np.ndarray, shape (n_criteria,)
        +1 for maximization, -1 for minimization criteria.

    Returns
    -------
    np.ndarray, shape (n_options,)
        SAW scores.
    """
    signed = matrix * criteria_types  # flip minimization criteria
    return signed @ weights


def _topsis(
    matrix: np.ndarray,
    weights: np.ndarray,
    criteria_types: np.ndarray,
) -> np.ndarray:
    """Technique for Order of Preference by Similarity to Ideal Solution.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_options, n_criteria)
        Normalized criteria values.
    weights : np.ndarray, shape (n_criteria,)
        Normalized weights summing to 1.
    criteria_types : np.ndarray, shape (n_criteria,)
        +1 for maximization, -1 for minimization criteria.

    Returns
    -------
    np.ndarray, shape (n_options,)
        TOPSIS closeness scores in [0, 1].
    """
    weighted = matrix * weights

    # Ideal best/worst depend on criterion direction
    ideal_best = np.where(
        criteria_types == 1, weighted.max(axis=0), weighted.min(axis=0)
    )
    ideal_worst = np.where(
        criteria_types == 1, weighted.min(axis=0), weighted.max(axis=0)
    )

    d_best = np.linalg.norm(weighted - ideal_best, axis=1)
    d_worst = np.linalg.norm(weighted - ideal_worst, axis=1)

    return d_worst / (d_best + d_worst)


APPROACH_FN = {
    MCDAApproach.SAW: _saw,
    MCDAApproach.TOPSIS: _topsis,
}
