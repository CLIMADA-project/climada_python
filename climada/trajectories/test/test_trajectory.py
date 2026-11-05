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

unit tests for RiskTrajectory (Being an abstract )

"""

import datetime
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from climada.entity.disc_rates.base import DiscRates
from climada.trajectories.constants import AAI_METRIC_NAME
from climada.trajectories.snapshot import Snapshot
from climada.trajectories.trajectory import (
    DEFAULT_ALLGROUP_NAME,
    DEFAULT_RP,
    RiskTrajectory,
)


@pytest.fixture
def mock_snapshots():
    """Provides a list of mock Snapshot objects with sequential dates."""
    snaps = []
    for year in [2023, 2024, 2025]:
        m = MagicMock(spec=Snapshot)
        m.date = datetime.date(year, 1, 1)
        snaps.append(m)
    return snaps


def test_abstract():
    with pytest.raises(TypeError, match="abstract class"):
        RiskTrajectory(mock_snapshots)  # type: ignore
