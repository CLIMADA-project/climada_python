"""Default functions"""
from typing import Sequence, Optional

import pandas as pd
import numpy as np

from climada.engine import Impact

def rmse(impact: pd.DataFrame, data: pd.DataFrame):
    return np.sqrt(np.mean(((impact - data) ** 2).to_numpy()))

def rmsf(impact: pd.DataFrame, data: pd.DataFrame):
    return np.sqrt(np.mean((((impact + 1) / (data + 1)) ** 2).to_numpy()))

def impact_at_reg(impact: Impact, region_ids: Optional[Sequence] = None):
    return impact.impact_at_reg(agg_regions=region_ids)
