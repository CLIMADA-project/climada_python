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
Tests for ensemble calibration module
"""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import create_autospec, patch

import numpy as np
import numpy.testing as npt
import pandas as pd
import pandas.testing as pdt

from climada.util.calibrate.base import Input, Output
from climada.util.calibrate.ensemble import (
    AverageEnsembleOptimizer,
    EnsembleOptimizer,
    EnsembleOptimizerOutput,
    SingleEnsembleOptimizerOutput,
    TragedyEnsembleOptimizer,
    event_info_from_input,
    sample_data,
    sample_weights,
)

from .test_base import ConcreteOptimizer, exposure, hazard


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

    def test_from_outputs_empty(self):
        """Test 'from_outputs' with empty list"""
        out = EnsembleOptimizerOutput.from_outputs([])
        self.assertTrue(out.data.empty)

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

    def test_to_input_var(self):
        """Test creating an Unsequa InputVar from the output"""

        def impf_creator(**params):
            """Stub impf creator"""
            return params

        invar = EnsembleOptimizerOutput.from_outputs(
            [self.output1, self.output2]
        ).to_input_var(
            impact_func_creator=impf_creator,
            haz_id_dict={"TC": [0, 1]},
            bounds_impfi=(0, 1),
        )
        self.assertDictEqual(invar.func(IFi=None, IL=0), self.output1.params)
        self.assertDictEqual(invar.func(IFi=None, IL=1), self.output2.params)
        self.assertListEqual(list(invar.distr_dict.keys()), ["IFi", "IL"])


class TestSampleData(unittest.TestCase):
    """Test sample_data function"""

    def test_sample_data(self):
        """Test sampling of a Data Frame"""
        df = pd.DataFrame([[0, 1, 2], [3, 4, 5]], index=[1, 2], columns=["a", "b", "c"])
        samples = [(0, 0), (0, 2), (1, 1)]

        pdt.assert_frame_equal(
            sample_data(df, samples),
            pd.DataFrame(
                [[0, np.nan, 2], [np.nan, 4, np.nan]],
                index=df.index,
                columns=df.columns,
            ),
        )


class TestSampleWeights(unittest.TestCase):
    """Test sample_weights function"""

    def test_sample_weights(self):
        """Test sampling of data weights"""
        df = pd.DataFrame([[0, 1, 2], [3, 4, 5]], index=[1, 2], columns=["a", "b", "c"])
        samples = [(0, 0), (0, 2), (1, 1), (0, 2)]

        pdt.assert_frame_equal(
            sample_weights(df, samples),
            pd.DataFrame(
                [[0, 0, 4], [0, 4, 0]],
                index=df.index,
                columns=df.columns,
                dtype="float",
            ),
        )


class TestEventInfoFromInput(unittest.TestCase):
    """Test retrieving event information from the input"""

    def setUp(self):
        """Create input"""
        self.input = Input(
            hazard=hazard(),
            exposure=exposure(),
            data=pd.DataFrame(
                [[1, np.nan], [10, np.nan], [np.nan, np.nan]],
                index=[1, 3, 10],
                columns=["a", "b"],
            ),
            cost_func=lambda _: None,
            impact_func_creator=lambda _: None,
            impact_to_dataframe=lambda _: None,
            assign_centroids=True,
        )
        self.input.hazard.centroids.gdf["region_id"] = ["a", "b"]
        self.input.hazard.event_name = ["event1", "event2", 3]

    def test_info_valid_hazard(self):
        """Test retrieving event information from the input"""
        info = event_info_from_input(self.input)
        self.assertListEqual(list(info.keys()), ["event_id", "region_id", "event_name"])
        npt.assert_array_equal(info["event_id"], np.array([1, 3]))
        npt.assert_array_equal(info["region_id"], np.array(["a"]))
        self.assertListEqual(info["event_name"], ["event1", "event2"])

    def test_info_invalid_hazard(self):
        """Test retrieving event information if selection somehow failed"""
        self.input.data.set_index(pd.Index([100, 101, 102]), inplace=True)
        info = event_info_from_input(self.input)
        self.assertListEqual(info["event_name"], [])


class ConcreteEnsembleOptimizer(EnsembleOptimizer):
    """Concrete instantiation of an ensemble optimizer"""

    def input_from_sample(self, sample):
        inp = copy.copy(self.input)  # NOTE: Shallow copy!
        inp.data = sample_data(inp.data, sample)
        return inp


@patch("climada.util.calibrate.bayesian_optimizer.BayesianOptimizer")
class TestEnsembleOptimizer(unittest.TestCase):
    """Test the AverageEnsembleOptimizer"""

    def setUp(self):
        """Create input and optimizer"""
        self.input = Input(
            hazard=hazard(),
            exposure=exposure(),
            data=pd.DataFrame(
                [[1, 2], [10, 11], [100, np.nan]],
                index=[1, 3, 10],
                columns=["a", "b"],
            ),
            cost_func=lambda _: None,
            impact_func_creator=lambda _: None,
            impact_to_dataframe=lambda _: None,
            assign_centroids=False,
        )

    @patch("climada.util.calibrate.ensemble.ProcessPool")
    def test_run(self, pool_class_mock, opt_class_mock):
        """Test initialization"""
        # Mock the optimizer class
        opt_mock = opt_class_mock.return_value
        output = Output(params={"p1": 0.1, "p2": 2}, target=0.2)
        opt_mock.run.return_value = output

        # Mock the process pool (context manager)
        pool_mock = pool_class_mock.return_value.__enter__.return_value
        pool_mock.imap.side_effect = map

        self.opt = ConcreteEnsembleOptimizer(
            input=self.input,
            optimizer_type=opt_class_mock,
            optimizer_init_kwargs={"foo": "bar", "random_state": 2},
        )
        self.opt.samples = [[(0, 0)], [(1, 0), (1, 1)], [(2, 0)]]

        outputs = []
        for proc in (1, 3):
            with self.subTest(processes=proc):
                ens_out = self.opt.run(processes=proc, bar="baz")
                outputs.append(ens_out)

                if proc > 1:
                    pool_class_mock.assert_called_once_with(nodes=proc)

                self.assertEqual(ens_out.data.shape[0], len(self.opt.samples))

                # Test passing init_kwargs
                self.assertEqual(opt_class_mock.call_args.kwargs["foo"], "bar")

                # Test update_init_kwargs
                self.assertListEqual(
                    [call[1]["random_state"] for call in opt_class_mock.call_args_list],
                    [2, 3, 4],
                )

                # Test passing run kwargs
                self.assertEqual(opt_mock.run.call_args.kwargs["bar"], "baz")

                # Test passing the input and sampling
                pdt.assert_frame_equal(
                    opt_class_mock.call_args_list[0][0][0].data,
                    pd.DataFrame(
                        [[1, np.nan], [np.nan, np.nan], [np.nan, np.nan]],
                        index=[1, 3, 10],
                        columns=["a", "b"],
                    ),
                )
                pdt.assert_frame_equal(
                    opt_class_mock.call_args_list[1][0][0].data,
                    pd.DataFrame(
                        [[np.nan, np.nan], [10, 11], [np.nan, np.nan]],
                        index=[1, 3, 10],
                        columns=["a", "b"],
                    ),
                )
                pdt.assert_frame_equal(
                    opt_class_mock.call_args_list[2][0][0].data,
                    pd.DataFrame(
                        [[np.nan, np.nan], [np.nan, np.nan], [100, np.nan]],
                        index=[1, 3, 10],
                        columns=["a", "b"],
                    ),
                )

                # Reset mock calls
                opt_class_mock.reset_mock()
                opt_mock.reset_mock()
                pool_class_mock.reset_mock()
                pool_mock.reset_mock()

        pdt.assert_frame_equal(outputs[0].data, outputs[1].data)

    def test_run_empty_samples(self, opt_class_mock):
        """Test execution with empty samples list"""
        self.opt = ConcreteEnsembleOptimizer(
            input=self.input,
            optimizer_type=opt_class_mock,
            optimizer_init_kwargs={"foo": "bar", "random_state": 2},
        )
        ens_out = self.opt.run(processes=1, bar="baz")
        self.assertTrue(ens_out.data.empty)


class DummyInput:
    def __init__(self, df):
        self.data = df
        self.stub = "a"
        self.hazard = create_autospec(hazard())
        self.hazard.select.return_value = self.hazard
        self.data_weights = None


class TestAverageEnsembleOptimizer(unittest.TestCase):
    """Test the AverageEnsembleOptimizer"""

    def setUp(self):
        # Sample DataFrame with some NaNs
        data = pd.DataFrame({"a": [1.0, None, None, 2.0], "b": [None, None, 3.0, 4.0]})
        self.input = DummyInput(data)

    def test_post_init_sampling(self):
        opt = AverageEnsembleOptimizer(
            input=self.input, sample_fraction=0.5, optimizer_type=ConcreteOptimizer
        )
        samples = np.array(opt.samples)
        self.assertTupleEqual(samples.shape, (20, 2, 2))

        opt = AverageEnsembleOptimizer(
            input=self.input,
            ensemble_size=11,
            sample_fraction=0.8,  # Will cause rounding
            optimizer_type=ConcreteOptimizer,
        )
        samples = np.array(opt.samples)
        self.assertTupleEqual(samples.shape, (11, 3, 2))

        opt = AverageEnsembleOptimizer(
            input=self.input,
            ensemble_size=2,
            sample_fraction=0.95,  # Will cause rounding, always select all
            optimizer_type=ConcreteOptimizer,
        )

        samples = [sorted([tuple(idx) for idx in arr]) for arr in opt.samples]
        npt.assert_array_equal(samples[0], [[0, 0], [2, 1], [3, 0], [3, 1]])
        npt.assert_array_equal(samples[0], samples[1])

    def test_sampling_replace(self):
        """Test if replacement works"""
        data = pd.DataFrame({"a": [1.0]})
        self.input = DummyInput(data)
        opt = AverageEnsembleOptimizer(
            input=self.input,
            ensemble_size=1,
            sample_fraction=3,
            replace=True,
            optimizer_type=ConcreteOptimizer,
        )
        npt.assert_array_equal(opt.samples, [[(0, 0), (0, 0), (0, 0)]])

    def test_invalid_sample_fraction(self):
        with self.assertRaisesRegex(ValueError, "Sample fraction"):
            AverageEnsembleOptimizer(
                input=self.input,
                sample_fraction=0,
                optimizer_type=ConcreteOptimizer,
            )
        with self.assertRaisesRegex(ValueError, "Sample fraction"):
            AverageEnsembleOptimizer(
                input=self.input,
                sample_fraction=1.2,
                replace=False,
                optimizer_type=ConcreteOptimizer,
            )

        # Should not throw
        AverageEnsembleOptimizer(
            input=self.input,
            sample_fraction=1.1,
            replace=True,
            optimizer_type=ConcreteOptimizer,
        )

    def test_invalid_ensemble_size(self):
        with self.assertRaisesRegex(ValueError, "Ensemble size must be >=1"):
            AverageEnsembleOptimizer(
                input=self.input,
                ensemble_size=0,
                optimizer_type=ConcreteOptimizer,
            )

    def test_random_state_determinism(self):
        opt1 = AverageEnsembleOptimizer(
            input=self.input,
            random_state=123,
            optimizer_type=ConcreteOptimizer,
        )
        opt2 = AverageEnsembleOptimizer(
            input=self.input,
            random_state=123,
            optimizer_type=ConcreteOptimizer,
        )
        for s1, s2 in zip(opt1.samples, opt2.samples):
            np.testing.assert_array_equal(s1, s2)

    def test_input_from_sample(self):
        opt = AverageEnsembleOptimizer(
            input=self.input,
            optimizer_type=ConcreteOptimizer,
        )
        inp = opt.input_from_sample([(0, 0), (3, 1), (0, 0)])

        self.assertIsNot(inp, self.input)
        self.assertIs(inp.stub, self.input.stub)
        pd.testing.assert_frame_equal(
            inp.data,
            pd.DataFrame({"a": [1.0, None, None, None], "b": [None, None, None, 4.0]}),
        )
        pd.testing.assert_frame_equal(
            inp.data_weights,
            pd.DataFrame({"a": [2.0, 0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0, 1.0]}),
        )


class TestTragedyEnsembleOptimizer(unittest.TestCase):
    """Test the TragedyEnsembleOptimizer"""

    def setUp(self):
        # Sample DataFrame with some NaNs
        data = pd.DataFrame({"a": [1.0, None, None, 2.0], "b": [None, None, 3.0, 4.0]})
        self.input = DummyInput(data)

    def test_post_init_sampling(self):
        opt = TragedyEnsembleOptimizer(
            input=self.input, optimizer_type=ConcreteOptimizer
        )
        samples = np.array(opt.samples)
        self.assertTupleEqual(samples.shape, (4, 1, 2))
        npt.assert_array_equal(samples, [[[0, 0]], [[2, 1]], [[3, 0]], [[3, 1]]])

        opt = TragedyEnsembleOptimizer(
            input=self.input,
            ensemble_size=2,
            optimizer_type=ConcreteOptimizer,
        )
        samples = np.array(opt.samples)
        self.assertTupleEqual(samples.shape, (2, 1, 2))

    def test_invalid_ensemble_size(self):
        with self.assertRaisesRegex(ValueError, "Ensemble size must be >=1"):
            TragedyEnsembleOptimizer(
                input=self.input,
                ensemble_size=0,
                optimizer_type=ConcreteOptimizer,
            )
        with self.assertRaisesRegex(ValueError, "here: 4"):
            TragedyEnsembleOptimizer(
                input=self.input,
                ensemble_size=5,
                optimizer_type=ConcreteOptimizer,
            )

    def test_random_state_determinism(self):
        opt1 = TragedyEnsembleOptimizer(
            input=self.input,
            random_state=2,
            optimizer_type=ConcreteOptimizer,
        )
        opt2 = TragedyEnsembleOptimizer(
            input=self.input,
            random_state=2,
            optimizer_type=ConcreteOptimizer,
        )
        for s1, s2 in zip(opt1.samples, opt2.samples):
            np.testing.assert_array_equal(s1, s2)

    def test_input_from_sample(self):
        opt = TragedyEnsembleOptimizer(
            input=self.input,
            optimizer_type=ConcreteOptimizer,
        )
        inp = opt.input_from_sample([(3, 0)])
        self.assertIsNot(inp, self.input)
        self.assertIs(inp.stub, self.input.stub)
        pdt.assert_frame_equal(inp.data, pd.DataFrame({"a": [2.0]}, index=[3]))
        inp.hazard.select.assert_called_once_with(event_id=pd.Index([3]))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestSampleData)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEventInfoFromInput))
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestEnsembleOptimizerOutput)
    )
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEnsembleOptimizer))
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestAverageEnsembleOptimizer)
    )
    TESTS.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestTragedyEnsembleOptimizer)
    )
    unittest.TextTestRunner(verbosity=2).run(TESTS)
