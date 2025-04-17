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
Tests for cross calibration module
"""
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import numpy as np
import numpy.testing as npt

from climada.util.calibrate.cross_calibrate import (
    EnsembleOptimizerOutput,
    SingleEnsembleOptimizerOutput,
)


class TestEnsembleOptimizerOutput(unittest.TestCase):
    """Test the EnsembleOptimizerOutput"""

    def setUp(self):
        """Initialize single outputs"""
        self.output1 = SingleEnsembleOptimizerOutput(
            params={"param1": 1.0, "param2": 2.0},
            target=1,
            event_info={
                "event_id": np.array([1, 2]),
                "region_id": np.array([1, 2]),
                "event_name": ["a", "b"],
            },
        )
        self.output2 = SingleEnsembleOptimizerOutput(
            params={"param1": 1.1, "param2": 2.1},
            target=2,
            event_info={
                "event_name": [1],
                "event_id": np.array([3]),
                "region_id": np.array([4]),
            },
        )

    def test_from_outputs(self):
        """Test 'from_outputs' initialization"""
        out = EnsembleOptimizerOutput.from_outputs([self.output1, self.output2])
        data = out.data

        # Test MultiIndex columns
        npt.assert_array_equal(data["Parameters"].columns, ["param1", "param2"])
        npt.assert_array_equal(
            data["Event"].columns, ["event_id", "region_id", "event_name"]
        )

        # Test parameters
        npt.assert_array_equal(data[("Parameters", "param1")], [1.0, 1.1])
        npt.assert_array_equal(data[("Parameters", "param2")], [2.0, 2.1])

        # Test event info
        pdt.assert_series_equal(
            data[("Event", "event_id")],
            pd.Series([np.array([1, 2]), np.array([3])], name=("Event", "event_id")),
        )
        pdt.assert_series_equal(
            data[("Event", "region_id")],
            pd.Series([np.array([1, 2]), np.array([4])], name=("Event", "region_id")),
        )
        pdt.assert_series_equal(
            data[("Event", "event_name")],
            pd.Series([["a", "b"], [1]], name=("Event", "event_name")),
        )

    def test_cycling(self):
        """Test correct cycling to files"""
        with TemporaryDirectory() as tmp:
            filepath = Path(tmp, "file.h5")

            out = EnsembleOptimizerOutput.from_outputs([self.output1, self.output2])
            out.to_hdf(filepath)

            out_new = EnsembleOptimizerOutput.from_hdf(filepath)
            pdt.assert_frame_equal(out.data, out_new.data)

    @unittest.skip("Cycling with CSV does not preserve data types")
    def test_cycling_csv(self):
        """Test correct cycling with CSV"""
        with TemporaryDirectory() as tmp:
            filepath = Path(tmp, "file.csv")

            out = EnsembleOptimizerOutput.from_outputs([self.output1, self.output2])
            out.to_csv(filepath)

            out_new = EnsembleOptimizerOutput.from_csv(filepath)
            pdt.assert_frame_equal(out.data, out_new.data)
