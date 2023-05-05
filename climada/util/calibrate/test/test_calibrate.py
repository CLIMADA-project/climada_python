"""Tests for calibration module"""

import unittest
from unittest.mock import create_autospec

import numpy as np
import numpy.testing as npt
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from climada.entity import Exposures, ImpactFuncSet
from climada.hazard import Hazard, Centroids

from ..impact_func import Input, ScipyMinimizeOptimizer


def hazard():
    """Create a dummy hazard instance"""
    lat = [1, 2]
    lon = [0, 1]
    centroids = Centroids.from_lat_lon(lat=lat, lon=lon)
    event_id = np.array([1, 3, 10])
    intensity = csr_matrix([[1, 1], [2, 2], [3, 3]])
    return Hazard(event_id=event_id, centroids=centroids, intensity=intensity)


def exposure():
    """Create a dummy exposure instance"""
    return Exposures(data=dict(longitude=[0, 1, 100], latitude=[1, 2, 50]))


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
        self.cost_func = lambda impact, data: 1.0
        self.impact_func_gen = lambda **kwargs: ImpactFuncSet()

    def test_post_init_calls(self):
        """Test if post_init calls stuff correctly using mocks"""
        # Create mocks
        hazard_mock_1 = create_autospec(Hazard, instance=True)
        hazard_mock_2 = create_autospec(Hazard, instance=True)
        exposure_mock = create_autospec(Exposures, instance=True)

        # Make first hazard mock return another instance
        hazard_mock_1.select.return_value = hazard_mock_2

        # Create input
        input = Input(
            hazard=hazard_mock_1,
            exposure=exposure_mock,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_gen=self.impact_func_gen,
        )

        # Query checks
        hazard_mock_1.select.assert_called_once_with(event_id=self.data_events)
        self.assertNotEqual(input.hazard, hazard_mock_1)
        self.assertEqual(input.hazard, hazard_mock_2)
        exposure_mock.assign_centroids.assert_called_once_with(hazard_mock_2)

    def test_post_init(self):
        """Test if post_init results in a sensible hazard and exposure"""
        # Create input
        input = Input(
            hazard=self.hazard,
            exposure=self.exposure,
            data=self.data,
            cost_func=self.cost_func,
            impact_func_gen=self.impact_func_gen,
        )

        # Check hazard and exposure
        npt.assert_array_equal(input.hazard.event_id, self.data.index)
        self.assertIn("centr_", input.exposure.gdf)
        npt.assert_array_equal(input.exposure.gdf["centr_"], [0, 1, -1])


class TestScipyMinimizeOptimizer(unittest.TestCase):
    """Tests for the optimizer based on scipy.optimize.minimize"""

    def setUp(self):
        """Mock the input and create the optimizer"""
        self.input = create_autospec(Input, instance=True)
        self.optimizer = ScipyMinimizeOptimizer(self.input)

    def test_kwargs_to_impact_func_gen(self):
        """Test the _kwargs_to_impact_func_gen method"""
        # _param_names is empty in the beginning
        x = np.array([1, 2, 3])
        self.assertDictEqual(self.optimizer._kwargs_to_impact_func_gen(x), {})

        # Now populate it and try again
        self.optimizer._param_names = ["x_2", "x_1", "x_3"]
        result = {"x_2": 1, "x_1": 2, "x_3": 3}
        self.assertDictEqual(self.optimizer._kwargs_to_impact_func_gen(x), result)

        # Other arguments are ignored
        self.assertDictEqual(
            self.optimizer._kwargs_to_impact_func_gen(x, x + 3), result
        )

        # Array is flattened, iterator stops
        self.assertDictEqual(
            self.optimizer._kwargs_to_impact_func_gen(np.array([[1, 2], [3, 4]])),
            result,
        )

    def test_select_by_keys(self):
        """Test the _select_by_keys method"""
        param_names = ["a", "b", "c", "d"]
        mapping = dict(zip(param_names, [1, "2", (1, 2)]))

        # _param_names is empty in the beginning
        self.assertListEqual(self.optimizer._select_by_param_names(mapping), [])

        # Set _param_names
        self.optimizer._param_names = param_names

        # Check result
        self.assertListEqual(
            self.optimizer._select_by_param_names(mapping), [1, "2", (1, 2), None]
        )
