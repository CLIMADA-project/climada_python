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

Unit Tests for LitPop class.
"""

import unittest

import numpy as np
from rasterio import Affine
import rasterio
from rasterio.crs import CRS
import copy
from climada.entity.exposures.litpop import litpop as lp


def data_arrays_demo(number_of_arrays=2):
    """init demo data arrays (2d) for LitPop core calculations"""
    data_arrays = list()
    if number_of_arrays > 0:
        data_arrays.append(np.array([[0, 1, 2], [3, 4, 5]]))
        # array([[0, 1, 2],
        #       [3, 4, 5]])
    if number_of_arrays > 1:
        data_arrays.append(np.array([[10, 10, 10], [1, 1, 1]]))
        # array([[10, 10, 10],
        #       [1, 1, 1]])
    if number_of_arrays > 2:
        data_arrays.append(np.array([[0, 1, 10], [0, 1, 10]]))
        # array([[0, 1, 10],
        #       [0, 1, 10]])
    if number_of_arrays > 3:
        data_arrays.append([[0, 1, 10, 100], [0, 1, 10, 100]])
        # [[0, 1, 10, 100],
        #  [0, 1, 10, 100]]
    return data_arrays


def data_arrays_resampling_demo():
    """init demo data arrays (2d) and meta data for resampling"""
    data_arrays = list()
    # demo pop:
    data_arrays.append(np.array([[0, 1, 2], [3, 4, 5]], dtype="float32"))
    data_arrays.append(np.array([[0, 1, 2], [3, 4, 5]], dtype="float32"))
    # array([[0, 1, 2],
    #       [3, 4, 5]])
    # demo nightlight:
    data_arrays.append(
        np.array(
            [
                [2, 10, 0, 0, 0, 0],
                [10, 2, 10, 0, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 1, 1],
            ],
            dtype="float32",
        )
    )
    # array([[ 2., 10.,  0.,  0.,  0.,  0.],
    #        [10.,  2., 10.,  0.,  0.,  0.],
    #        [ 0.,  0.,  0.,  0.,  1.,  1.],
    #        [ 1.,  0.,  0.,  0.,  1.,  1.]], dtype=float32)]

    meta_list = [
        {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": -3.4028230607370965e38,
            "width": 3,
            "height": 2,
            "count": 1,
            "crs": CRS.from_epsg(4326),
            #'crs': CRS.from_epsg(4326),
            "transform": Affine(1, 0.0, -10, 0.0, -1, 40),
        },
        {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": -3.4028230607370965e38,
            "width": 3,
            "height": 2,
            "count": 1,
            "crs": CRS.from_epsg(4326),
            #'crs': CRS.from_epsg(4326),
            "transform": Affine(
                1, 0.0, -10, 0.0, -1, 41
            ),  # shifted by 1 degree latitude to the north
        },
        {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": None,
            "width": 6,
            "height": 4,
            "count": 1,
            "crs": CRS.from_epsg(4326),
            # 'crs': CRS.from_epsg(32662),
            "transform": Affine(0.5, 0.0, -10, 0.0, -0.5, 40),  # higher resolution
        },
    ]
    return data_arrays, meta_list

def create_target_grid(meta, res_arcsec=None):
    """Create a properly cropped target grid with the right resolution.

    Parameters
    ----------
    meta : dict
        Metadata dictionary from an original raster grid.
    res_arcsec : int, optional
        Desired resolution in arcseconds. If None, uses the original resolution.

    Returns
    -------
    target_grid : dict
        Dictionary containing the correctly aligned target grid metadata.
    """
    import copy

    # Make a copy of the original metadata
    target_grid = copy.deepcopy(meta)

    if res_arcsec is not None:
        res_deg = res_arcsec / 3600  # Convert arcseconds to degrees

        # Ensure it aligns with the original grid
        aligned_lon_min = -180 + (round((meta["transform"][2] - (-180)) / res_deg) * res_deg)
        aligned_lat_max = 90 - (round((90 - meta["transform"][5]) / res_deg) * res_deg)

        # Define new affine transform
        target_grid["transform"] = Affine(
            res_deg, 0, aligned_lon_min,
            0, -res_deg, aligned_lat_max
        )

        # Compute width and height
        width = int(round(meta["width"] * (meta["transform"].a / res_deg)))
        height = int(round(meta["height"] * (abs(meta["transform"].e) / res_deg)))

        target_grid["width"] = width
        target_grid["height"] = height

    return target_grid

class TestLitPop(unittest.TestCase):
    """Test LitPop Class methods and functions"""

    def test_reproject_input_data_downsample(self):
        """test function reproject_input_data downsampling lit to pop grid
        (default resampling for LitPop)"""
        data_in, meta_list = data_arrays_resampling_demo()
        #
        data_out, meta_out = lp.reproject_input_data(
            data_in,
            meta_list,
            target_grid=meta_list[0]
        )
        # test reference data unchanged:
        np.testing.assert_array_equal(data_in[0], data_out[0])
        # test northward shift:
        np.testing.assert_array_equal(data_in[1][1, :], data_out[1][0, :])
        # test reprojected nl data:
        reference_array = np.array(
            [[5.020408, 2.267857, 0.12244898], [1.1224489, 0.6785714, 0.7346939]],
            dtype="float32",
        )
        np.testing.assert_array_almost_equal_nulp(reference_array, data_out[2])

    def test_reproject_input_data_downsample_conserve_sum(self):
        """test function reproject_input_data downsampling with conservation of sum"""
        data_in, meta_list = data_arrays_resampling_demo()
        data_out, meta_out = lp.reproject_input_data(
            data_in,
            meta_list,
            target_grid = meta_list[0],
            conserve="sum",
        )
        # test reference data unchanged:
        np.testing.assert_array_equal(data_in[0], data_out[0])
        # test conserve sum:
        for i, _ in enumerate(data_in):
            self.assertAlmostEqual(data_in[i].sum(), data_out[i].sum())

    def test_target_grid_alignment(self):
        """Test if reprojection correctly aligns to target grid"""

        # Step 1: Define original synthetic dataset
        orig_res = 1.0
        orig_transform = Affine(orig_res, 0, -10.0, 0, -orig_res, 40.0)
        
        data_in, meta_list = data_arrays_resampling_demo()
        meta_list[0]["transform"] = orig_transform
        meta_list[0]["width"] = 3
        meta_list[0]["height"] = 3

        # Step 2: Define target grid (coarser resolution)
        target_res = 2.0  # New resolution (downsample from 1° to 2°)
        target_transform = Affine(target_res, 0, -10.0, 0, -target_res, 40.0)
        
        target_grid = {
            "driver": "GTiff",
            "dtype": "float32",
            "nodata": meta_list[0]["nodata"],
            "width": 2,  # (3° / 2°) ≈ 2
            "height": 2,  # (3° / 2°) ≈ 2
            "count": 1,
            "crs": meta_list[0]["crs"],
            "transform": target_transform,
        }

        # Step 3: Reproject
        data_out, meta_out = lp.reproject_input_data(
            data_in,
            meta_list,
            target_grid=target_grid,
            resampling=rasterio.warp.Resampling.bilinear
        )

        # Step 4: Validate output
        self.assertEqual(meta_out["transform"], target_transform, "Transform does not match target grid!")
        self.assertEqual(meta_out["width"], target_grid["width"], "Width mismatch!")
        self.assertEqual(meta_out["height"], target_grid["height"], "Height mismatch!")

        # Check for correct data structure
        self.assertEqual(data_out[0].shape, (2, 2), "Output data shape is incorrect!")

    def test_reproject_input_data_downsample_conserve_mean(self):
        """test function reproject_input_data downsampling with conservation of sum"""
        data_in, meta_list = data_arrays_resampling_demo()
        data_out, meta_out = lp.reproject_input_data(
            data_in,
            meta_list,
            target_grid=meta_list[1],
            conserve="mean",
        )
        # test reference data unchanged:
        np.testing.assert_array_equal(data_in[1], data_out[1])
        # test conserve sum:
        for i, _ in enumerate(data_in):
            self.assertAlmostEqual(data_in[i].mean(), data_out[i].mean(), places=5)

    def test_reproject_input_data_upsample(self):
        """test function reproject_input_data with upsampling
        (usually not required for LitPop)"""
        data_in, meta_list = data_arrays_resampling_demo()
        data_out, meta_out = lp.reproject_input_data(
            data_in,
            meta_list,
            target_grid=meta_list[2],
        )
        # test reference data unchanged:
        np.testing.assert_array_equal(data_in[2], data_out[2])
        # test northward shift:
        np.testing.assert_array_equal(data_out[0][2, :], data_out[1][0, :])
        np.testing.assert_array_equal(data_out[0][3, :], data_out[1][1, :])
        # test reprojected nl data:
        reference_array = np.array(
            [
                [0.0, 0.25, 0.75, 1.25, 1.75, 2.0],
                [0.75, 1.0, 1.5, 2.0, 2.5, 2.75],
                [2.25, 2.5, 3.0, 3.5, 4.0, 4.25],
                [3.0, 3.25, 3.75, 4.25, 4.75, 5.0],
            ],
            dtype="float32",
        )
        np.testing.assert_array_equal(reference_array, data_out[0])

    def test_gridpoints_core_calc_input_errors(self):
        """test for ValueErrors and TypeErrors due to wrong input to function
        gridpoints_core_calc"""
        data = data_arrays_demo(2)
        # negative offset:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data, offsets=[2, -1])
        # negative exponents:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data, exponents=[2, -1])

        # different shapes:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data_arrays_demo(4))

        # wrong format:
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data, exponents=["a", "b"])
        data.append("hello i am a string")
        with self.assertRaises(ValueError):
            lp.gridpoints_core_calc(data)

    def test_gridpoints_core_calc_default_1(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with default exponents and offsets - 1 array"""
        data_arrays = data_arrays_demo(1)  # get list with 1 demo array
        result_array = lp.gridpoints_core_calc(data_arrays)
        results_check = data_arrays[0]

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[1, 1], results_check[1, 1])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_default_2(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with default exponents and offsets- 2 arrays"""
        data_arrays = data_arrays_demo(2)  # get list with 2 demo arrays
        result_array = lp.gridpoints_core_calc(data_arrays)
        results_check = data_arrays[0] * data_arrays[1]

        self.assertEqual(result_array[0, 0], results_check[0, 0])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_default_3(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with default exponents and offsets- 3 arrays"""
        data_arrays = data_arrays_demo(3)
        result_array = lp.gridpoints_core_calc(data_arrays)
        results_check = data_arrays[0] * data_arrays[1] * data_arrays[2]

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[1, 1], results_check[1, 1])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)
        # self.assertEqual(result_array, data_arrays[0] * data_arrays[1])

    def test_gridpoints_core_calc_exp(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed exponents"""
        data_arrays = data_arrays_demo(2)  # get list with 2 demo arrays
        exp = [2, 1]
        result_array = lp.gridpoints_core_calc(data_arrays, exponents=exp)
        results_check = data_arrays[0] * data_arrays[0] * data_arrays[1]

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[0, 2], results_check[0, 2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

        exp = [2, 0.1]
        result_array = lp.gridpoints_core_calc(data_arrays, exponents=exp)
        results_check = data_arrays[0] * data_arrays[0] * (data_arrays[1] ** 0.1)
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_offsets(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed offsets"""
        data_arrays = data_arrays_demo(2)  # get list with 2 demo arrays
        offsets = [1, 10]
        result_array = lp.gridpoints_core_calc(data_arrays, offsets=offsets)
        results_check = (data_arrays[0] + 1) * (10 + data_arrays[1])

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[0, 2], results_check[0, 2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_offsets_exp(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed offsets and exponents"""
        data_arrays = data_arrays_demo(2)  # get list with 2 demo arrays
        offsets = [0, 10]
        exp = [2, 1]
        result_array = lp.gridpoints_core_calc(
            data_arrays, offsets=offsets, exponents=exp
        )
        results_check = (data_arrays[0]) * (data_arrays[0]) * (10 + data_arrays[1])
        results_check2 = np.array([[0, 20, 80], [99, 176, 275]])

        self.assertEqual(result_array.shape, results_check.shape)
        self.assertEqual(result_array[0, 2], results_check[0, 2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)
        self.assertEqual(result_array[1, 2], results_check2[1, 2])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check2)

    def test_gridpoints_core_calc_rescale(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with rescaling (default exponents and offsets)"""
        data_arrays = data_arrays_demo(2)  # get list with 2 demo arrays
        result_array = lp.gridpoints_core_calc(data_arrays, total_val_rescale=2.5)
        results_check = (
            (data_arrays[0] * data_arrays[1])
            * 2.5
            / np.sum(data_arrays[0] * data_arrays[1])
        )

        self.assertAlmostEqual(result_array.sum(), 2.5)
        self.assertEqual(result_array[0, 1], results_check[0, 1])
        np.testing.assert_array_almost_equal_nulp(result_array, results_check)

    def test_gridpoints_core_calc_offsets_exp_rescale(self):
        """test function gridpoints_core_calc, i.e. core data combination
        on grid point level with changed offsets and exponents and rescaling"""
        data_arrays = data_arrays_demo(2)  # get list with 2 demo arrays
        offsets = [0.2, 3]
        exp = [0.5, 1.7]
        tot = -7
        result_array = lp.gridpoints_core_calc(
            data_arrays, offsets=offsets, exponents=exp, total_val_rescale=tot
        )
        results_check = np.array(data_arrays[0] + 0.2, dtype=float) ** exp[0] * (
            np.array(data_arrays[1] + 3.0, dtype=float) ** exp[1]
        )
        results_check = results_check * tot / results_check.sum()
        self.assertEqual(result_array.shape, results_check.shape)
        self.assertAlmostEqual(result_array.sum(), tot)
        self.assertEqual(result_array[1, 2], results_check[1, 2])
        np.testing.assert_allclose(result_array, results_check)

    def test_grp_read_pass(self):
        """test _grp_read() to pass and return either dict with admin1 values or None"""
        result = lp._grp_read("JPN")
        if result is not None:
            self.assertIsInstance(result, dict)
            self.assertIn("Fukuoka", result.keys())
            self.assertIsInstance(result["Saga"], float)

    def test_fail_get_total_value_per_country_pop(self):
        "test _get_total_value_per_country fails for pop"
        with self.assertRaises(NotImplementedError):
            lp._get_total_value_per_country("XXX", "pop", None)

    def test_get_total_value_per_country_none(self):
        "test _get_total_value_per_country pass with None"
        value = lp._get_total_value_per_country("XXX", "none", None)
        self.assertEqual(value, None)

    def test_get_total_value_per_country_norm(self):
        "test _get_total_value_per_country pass with 1"
        value = lp._get_total_value_per_country("XXX", "norm", None)
        self.assertEqual(value, 1)

    def test_get_total_value_per_country_gdp(self):
        "test _get_total_value_per_country get number for gdp"
        gdp_togo = lp._get_total_value_per_country("TGO", "gdp", 2010)
        gdp_switzerland = lp._get_total_value_per_country("CHE", "gdp", 2222)
        value_switzerland = lp._get_total_value_per_country("CHE", "income_group", 2222)
        self.assertIsInstance(gdp_togo, float)
        # value for income_group = gdp * income group:
        self.assertEqual(value_switzerland, 5 * gdp_switzerland)

    def test_get_total_value_per_country_pc(self):
        "test _get_total_value_per_country get number for pc of Poland"
        value = lp._get_total_value_per_country("POL", "pc", 2015)
        self.assertIsInstance(value, float)

    def test_get_total_value_per_country_nfw(self):
        "test _get_total_value_per_country get number for pc of Poland"
        value = lp._get_total_value_per_country("POL", "nfw", 2015)
        self.assertIsInstance(value, float)

    def test_get_value_unit_pass(self):
        """test get_value_unit pass"""
        self.assertEqual(lp.get_value_unit("pop"), "people")
        self.assertEqual(lp.get_value_unit("gdp"), "USD")
        self.assertEqual(lp.get_value_unit("pc"), "USD")
        self.assertEqual(lp.get_value_unit("nfw"), "USD")
        self.assertEqual(lp.get_value_unit("none"), "")


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLitPop)
    # TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestUncertainty))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
