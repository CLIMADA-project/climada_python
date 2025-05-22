import copy
import datetime
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from climada.entity.exposures import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.measures.base import Measure
from climada.hazard import Hazard
from climada.trajectories.snapshot import Snapshot
from climada.util.constants import EXP_DEMO_H5, HAZ_DEMO_H5


class TestSnapshot(unittest.TestCase):

    def setUp(self):
        # Create mock objects for testing
        self.mock_exposure = Exposures.from_hdf5(EXP_DEMO_H5)
        self.mock_hazard = Hazard.from_hdf5(HAZ_DEMO_H5)
        self.mock_impfset = ImpactFuncSet(
            [
                ImpactFunc(
                    "TC",
                    3,
                    intensity=np.array([0, 20]),
                    mdd=np.array([0, 0.5]),
                    paa=np.array([0, 1]),
                )
            ]
        )
        self.mock_measure = MagicMock(spec=Measure)
        self.mock_measure.name = "Test Measure"

        # Setup mock return values for measure.apply
        self.mock_modified_exposure = MagicMock(spec=Exposures)
        self.mock_modified_hazard = MagicMock(spec=Hazard)
        self.mock_modified_impfset = MagicMock(spec=ImpactFuncSet)
        self.mock_measure.apply.return_value = (
            self.mock_modified_exposure,
            self.mock_modified_impfset,
            self.mock_modified_hazard,
        )

    def test_init_with_int_date(self):
        snapshot = Snapshot(
            self.mock_exposure, self.mock_hazard, self.mock_impfset, 2023
        )
        self.assertEqual(snapshot.date, datetime.date(2023, 1, 1))

    def test_init_with_str_date(self):
        snapshot = Snapshot(
            self.mock_exposure, self.mock_hazard, self.mock_impfset, "2023-01-01"
        )
        self.assertEqual(snapshot.date, datetime.date(2023, 1, 1))

    def test_init_with_date_object(self):
        date_obj = datetime.date(2023, 1, 1)
        snapshot = Snapshot(
            self.mock_exposure, self.mock_hazard, self.mock_impfset, date_obj
        )
        self.assertEqual(snapshot.date, date_obj)

    def test_init_with_invalid_date(self):
        with self.assertRaises(ValueError):
            Snapshot(
                self.mock_exposure, self.mock_hazard, self.mock_impfset, "invalid-date"
            )

    def test_init_with_invalid_type(self):
        with self.assertRaises(TypeError):
            Snapshot(self.mock_exposure, self.mock_hazard, self.mock_impfset, 2023.5)

    def test_properties(self):
        snapshot = Snapshot(
            self.mock_exposure, self.mock_hazard, self.mock_impfset, 2023
        )

        # We want a new reference
        self.assertIsNot(snapshot.exposure, self.mock_exposure)
        self.assertIsNot(snapshot.hazard, self.mock_hazard)
        self.assertIsNot(snapshot.impfset, self.mock_impfset)

        # But we want equality
        pd.testing.assert_frame_equal(snapshot.exposure.gdf, self.mock_exposure.gdf)

        self.assertEqual(snapshot.hazard.haz_type, self.mock_hazard.haz_type)
        self.assertEqual(snapshot.hazard.intensity.nnz, self.mock_hazard.intensity.nnz)
        self.assertEqual(snapshot.hazard.size, self.mock_hazard.size)

        self.assertEqual(snapshot.impfset, self.mock_impfset)

    def test_apply_measure(self):
        snapshot = Snapshot(
            self.mock_exposure, self.mock_hazard, self.mock_impfset, 2023
        )
        new_snapshot = snapshot.apply_measure(self.mock_measure)

        self.assertIsNotNone(new_snapshot.measure)
        self.assertEqual(new_snapshot.measure.name, "Test Measure")
        self.assertEqual(new_snapshot.exposure, self.mock_modified_exposure)
        self.assertEqual(new_snapshot.hazard, self.mock_modified_hazard)
        self.assertEqual(new_snapshot.impfset, self.mock_modified_impfset)


if __name__ == "__main__":
    unittest.main()
