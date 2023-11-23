"""Tests for calibration module"""

import unittest
from unittest.mock import create_autospec

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix

from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Hazard, Centroids

from climada.util.calibrate import Input
from climada.util.calibrate.base import Optimizer


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
    return Hazard(event_id=event_id, centroids=centroids, intensity=intensity)


def exposure():
    """Create a dummy exposure instance"""
    return Exposures(
        data=dict(
            longitude=[0, 1, 100],
            latitude=[1, 2, 50],
            value=[1, 0.1, 1e6],
            impf_=[1, 1, 1],
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

        # Check hazard and exposure
        self.assertIn("centr_", input.exposure.gdf)
        npt.assert_array_equal(input.exposure.gdf["centr_"], [0, 1, -1])

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
        pd.testing.assert_frame_equal(
            impact_df_aligned,
            impact_df_aligned_test
        )

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


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestInputPostInit)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptimizer))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
