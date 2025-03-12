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
Tests for calibration module
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, create_autospec, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from climada.engine import ImpactCalc
from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Centroids, Hazard
from climada.util.calibrate import Input, OutputEvaluator
from climada.util.calibrate.base import Optimizer, Output


class ConcreteOptimizer(Optimizer):
    """An instance for testing. Implements 'run' without doing anything"""

    def run(self, **_):
        pass


def hazard():
    """Create a dummy hazard instance"""
    lat = [1, 2]
    lon = [0, 1]
    centroids = Centroids.from_lat_lon(lat=lat, lon=lon)
    event_id = np.array([1, 3, 10])
    intensity = csr_matrix([[1, 0.1], [2, 0.2], [3, 2]])
    return Hazard(
        event_id=event_id, centroids=centroids, intensity=intensity, haz_type="TEST"
    )


def exposure():
    """Create a dummy exposure instance"""
    return Exposures(
        data=dict(
            longitude=[0, 1, 100],
            latitude=[1, 2, 50],
            value=[1, 0.1, 1e6],
            impf_TEST=[1, 1, 1],
        )
    )


class TestInputPostInit(unittest.TestCase):
    """Test the post_init dunder method of Input"""

    def setUp(self):
        """Create default input instance"""
        # Create the hazard instance
        self.hazard = hazard()

        # Create the exposure instance
        self.exposure = exposure()

        # Create some data
        self.data_events = [10, 3]
        self.data = pd.DataFrame(data={"a": [1, 2]}, index=self.data_events)

        # Create dummy funcs
        self.impact_to_dataframe = lambda _: pd.DataFrame()
        self.cost_func = lambda impact, data: 1.0
        self.impact_func_gen = lambda **kwargs: ImpactFuncSet()

    def test_post_init_calls(self):
        """Test if post_init calls stuff correctly using mocks"""
        # Create mocks
        exposure_mock = create_autospec(Exposures())

        # Default
        Input(
            hazard=self.hazard,
            exposure=exposure_mock,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_creator=self.impact_func_gen,
            impact_to_dataframe=self.impact_to_dataframe,
        )
        exposure_mock.assign_centroids.assert_called_once_with(self.hazard)
        exposure_mock.reset_mock()

        # Default
        Input(
            hazard=self.hazard,
            exposure=exposure_mock,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_creator=self.impact_func_gen,
            impact_to_dataframe=self.impact_to_dataframe,
            assign_centroids=False,
        )
        exposure_mock.assign_centroids.assert_not_called()

    def test_post_init(self):
        """Test if post_init results in a sensible hazard and exposure"""
        # Create input
        input = Input(
            hazard=self.hazard,
            exposure=self.exposure,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_creator=self.impact_func_gen,
            impact_to_dataframe=self.impact_to_dataframe,
        )

        # Check centroids assignment
        npt.assert_array_equal(input.exposure.gdf["centr_TEST"], [0, 1, -1])

    def test_align_impact(self):
        """Check alignment of impact and data"""
        input = Input(
            hazard=hazard(),
            exposure=exposure(),
            data=pd.DataFrame(
                data={"col1": [1, 2], "col2": [2, 3]}, index=[0, 1], dtype="float"
            ),
            cost_func=lambda x, y: (x + y).sum(axis=None),
            impact_func_creator=lambda _: ImpactFuncSet([ImpactFunc()]),
            # Mock the dataframe creation by ignoring the argument
            impact_to_dataframe=lambda _: pd.DataFrame(
                data={"col2": [1, 2], "col3": [2, 3]}, index=[1, 2], dtype="float"
            ),
        )

        # missing_data_value = np.nan
        data_aligned, impact_df_aligned = input.impact_to_aligned_df(None)
        data_aligned_test = pd.DataFrame(
            data={
                "col1": [1, 2, np.nan],
                "col2": [2, 3, np.nan],
                "col3": [np.nan, np.nan, np.nan],
            },
            index=[0, 1, 2],
            dtype="float",
        )
        pd.testing.assert_frame_equal(data_aligned, data_aligned_test)
        impact_df_aligned_test = pd.DataFrame(
            data={"col1": [0, 0, 0], "col2": [0, 1, 0], "col3": [0, 0, 0]},
            index=[0, 1, 2],
            dtype="float",
        )
        pd.testing.assert_frame_equal(impact_df_aligned, impact_df_aligned_test)

        # Check fillna
        data_aligned, impact_df_aligned = input.impact_to_aligned_df(None, fillna=0)
        pd.testing.assert_frame_equal(data_aligned, data_aligned_test.fillna(0))
        pd.testing.assert_frame_equal(impact_df_aligned, impact_df_aligned_test)

        # Different missing data value
        input.missing_data_value = 0.0
        data_aligned, impact_df_aligned = input.impact_to_aligned_df(None)
        pd.testing.assert_frame_equal(data_aligned, data_aligned_test.fillna(0))
        pd.testing.assert_frame_equal(
            impact_df_aligned,
            pd.DataFrame(
                data={"col1": [0, 0, 0], "col2": [0, 1, 2], "col3": [0, 2, 3]},
                index=[0, 1, 2],
                dtype="float",
            ),
        )

        # Check error
        with self.assertRaisesRegex(ValueError, "NaN values computed in impact!"):
            input.impact_to_dataframe = lambda _: pd.DataFrame(
                data={"col1": [np.nan], "col2": [2, 3]}, index=[1, 2]
            )
            data_aligned, impact_df_aligned = input.impact_to_aligned_df(None)


class TestOptimizer(unittest.TestCase):
    """Base class for testing optimizers. Creates an input mock"""

    def setUp(self):
        """Mock the input"""
        self.input = Input(
            hazard=hazard(),
            exposure=exposure(),
            data=pd.DataFrame(data={"col1": [1, 2], "col2": [2, 3]}, index=[0, 1]),
            cost_func=lambda x, y: (x + y).sum(axis=None),
            impact_func_creator=lambda _: ImpactFuncSet([ImpactFunc()]),
            impact_to_dataframe=lambda x: x.impact_at_reg(),
        )
        self.optimizer = ConcreteOptimizer(self.input)


class TestOuput(unittest.TestCase):
    """Test the optimizer output"""

    def test_cycle(self):
        """Test if cycling an output object works"""
        output = Output(params={"p1": 1.0, "p_2": 10}, target=2.0)
        with TemporaryDirectory() as tmpdir:
            outfile = Path(tmpdir, "out.h5")
            output.to_hdf5(outfile)
            self.assertTrue(outfile.is_file())
            output_2 = Output.from_hdf5(outfile)
        self.assertEqual(output.target, output_2.target)
        self.assertDictEqual(output.params, output_2.params)


class TestOutputEvaluator(unittest.TestCase):
    """Test the output evaluator"""

    def setUp(self):
        """Create Input and Output"""
        self.input = Input(
            hazard=hazard(),
            exposure=exposure(),
            data=pd.DataFrame(),
            impact_func_creator=MagicMock(),
            # Should not be called
            impact_to_dataframe=lambda _: None,
            cost_func=lambda _: None,
        )
        self.output = Output(params={"p1": 1, "p2": 2.0}, target=0.0)

    @patch("climada.util.calibrate.base.ImpactCalc", autospec=True)
    def test_init(self, mock):
        """Test initialization"""
        self.input.exposure.value_unit = "my_unit"
        self.input.impact_func_creator.return_value = "impact_func"
        impact_calc_mock = MagicMock(ImpactCalc)
        mock.return_value = impact_calc_mock
        impact_calc_mock.impact = MagicMock()
        impact_calc_mock.impact.return_value = "impact"

        out_eval = OutputEvaluator(self.input, self.output)
        self.assertEqual(out_eval.impf_set, "impact_func")
        self.assertEqual(out_eval.impact, "impact")
        self.assertEqual(out_eval._impact_label, "Impact [my_unit]")

        self.input.impact_func_creator.assert_called_with(p1=1, p2=2.0)
        mock.assert_called_with(self.input.exposure, "impact_func", self.input.hazard)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInputPostInit)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptimizer))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOutputEvaluator))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
