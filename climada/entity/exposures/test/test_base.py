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

Test Exposure base class.
"""

import unittest

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import scipy as sp
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Point, Polygon
from sklearn.metrics import DistanceMetric

import climada.util.coordinates as u_coord
from climada import CONFIG
from climada.entity import LitPop
from climada.entity.exposures.base import (
    DEF_REF_YEAR,
    DEF_VALUE_UNIT,
    INDICATOR_CENTR,
    INDICATOR_IMPF,
    Exposures,
    add_sea,
)
from climada.hazard.base import Centroids, Hazard
from climada.util.constants import DEF_CRS, ENT_TEMPLATE_XLS, HAZ_DEMO_FL, ONE_LAT_KM

DATA_DIR = CONFIG.exposures.test_data.dir()


def good_exposures():
    """Followng values are defined for each exposure"""
    data = {}
    data["latitude"] = np.array([1, 2, 3])
    data["longitude"] = np.array([2, 3, 4])
    data["value"] = np.array([1, 2, 3])
    data["deductible"] = np.array([1, 2, 3])
    data[INDICATOR_IMPF + "NA"] = np.array([1, 2, 3])
    data["category_id"] = np.array([1, 2, 3])
    data["region_id"] = np.array([1, 2, 3])
    data[INDICATOR_CENTR + "TC"] = np.array([1, 2, 3])

    expo = Exposures(gpd.GeoDataFrame(data=data))
    return expo


class TestFuncs(unittest.TestCase):
    """Check assign function"""

    def test_assign_pass(self):
        """Check that attribute `assigned` is correctly set."""
        np_rand = np.random.RandomState(123456789)

        haz = Hazard.from_raster(
            [HAZ_DEMO_FL], haz_type="FL", window=Window(10, 20, 50, 60)
        )
        ncentroids = haz.centroids.size

        exp = Exposures(
            crs=haz.centroids.crs,
            lon=np.concatenate(
                [
                    haz.centroids.lon,
                    haz.centroids.lon + 0.001 * (-0.5 + np_rand.rand(ncentroids)),
                ]
            ),
            lat=np.concatenate(
                [
                    haz.centroids.lat,
                    haz.centroids.lat + 0.001 * (-0.5 + np_rand.rand(ncentroids)),
                ]
            ),
        )
        expected_result = np.concatenate([np.arange(ncentroids), np.arange(ncentroids)])

        # make sure that it works for both float32 and float64
        for test_dtype in [np.float64, np.float32]:
            haz.centroids.gdf["lat"] = haz.centroids.lat.astype(test_dtype)
            haz.centroids.gdf["lon"] = haz.centroids.lon.astype(test_dtype)
            exp.assign_centroids(haz)
            self.assertEqual(exp.gdf.shape[0], len(exp.hazard_centroids("FL")))
            np.testing.assert_array_equal(exp.hazard_centroids("FL"), expected_result)
            exp.assign_centroids(Hazard(), overwrite=False)
            self.assertEqual(exp.gdf.shape[0], len(exp.hazard_centroids("FL")))
            np.testing.assert_array_equal(exp.hazard_centroids("FL"), expected_result)

    def test__init__meta_type(self):
        """Check if meta of type list raises a ValueError in __init__"""
        with self.assertRaises(TypeError) as cm:
            Exposures(meta="{}")
        self.assertEqual("meta must be of type dict", str(cm.exception))

    def test__init__geometry_type(self):
        """Check that initialization fails when `geometry` is given as a `str` argument"""
        with self.assertRaises(TypeError) as cm:
            Exposures(geometry="myname")
        self.assertEqual(
            "Exposures is not able to handle customized 'geometry' column names.",
            str(cm.exception),
        )

    def test__init__mda_in_kwargs(self):
        """Check if `_metadata` attributes are instantiated correctly for sub-classes of
        ``Exposures``"""
        litpop = LitPop(exponents=2)
        self.assertEqual(litpop.exponents, 2)
        litpop = LitPop(meta=dict(exponents=3))
        self.assertEqual(litpop.exponents, 3)

    def test_read_raster_pass(self):
        """from_raster"""
        exp = Exposures.from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        self.assertTrue(u_coord.equal_crs(exp.crs, DEF_CRS))
        self.assertAlmostEqual(
            exp.latitude.max(), 10.248220966978932 - 0.009000000000000341 / 2
        )
        self.assertAlmostEqual(
            exp.latitude.min(),
            10.248220966978932 - 0.009000000000000341 / 2 - 59 * 0.009000000000000341,
        )
        self.assertAlmostEqual(
            exp.longitude.min(), -69.2471495969998 + 0.009000000000000341 / 2
        )
        self.assertAlmostEqual(
            exp.longitude.max(),
            -69.2471495969998 + 0.009000000000000341 / 2 + 49 * 0.009000000000000341,
        )
        self.assertEqual(len(exp.gdf), 60 * 50)
        self.assertAlmostEqual(
            exp.gdf["value"].values.reshape((60, 50))[25, 12], 0.056825936
        )

    def test_assign_raster_pass(self):
        """Test assign_centroids with raster hazard"""
        # explicit, easy-to-understand raster centroids for hazard
        meta = {
            "count": 1,
            "crs": DEF_CRS,
            "width": 20,
            "height": 10,
            "transform": rasterio.Affine(1.5, 0.0, -20, 0.0, -1.4, 8),
        }
        haz = Hazard("FL", centroids=Centroids.from_meta(meta))

        # explicit points with known results (see `expected_result` for details)
        exp = Exposures(
            crs=DEF_CRS,
            lon=[
                -20.1,
                -20.0,
                -19.8,
                -19.0,
                -18.6,
                -18.4,
                -19.0,
                -19.0,
                -19.0,
                -19.0,
                -20.1,
                0.0,
                10.1,
                10.1,
                10.1,
                0.0,
                -20.2,
                -20.3,
                -6.4,
                9.8,
                0.0,
            ],
            lat=[
                7.3,
                7.3,
                7.3,
                7.3,
                7.3,
                7.3,
                8.1,
                7.9,
                6.7,
                6.5,
                8.1,
                8.2,
                8.3,
                0.0,
                -6.1,
                -6.2,
                -6.3,
                0.0,
                -1.9,
                -1.7,
                0.0,
            ],
        )
        exp.assign_centroids(haz)

        expected_result = [
            # constant y-value, varying x-value
            0,
            0,
            0,
            0,
            0,
            1,
            # constant x-value, varying y-value
            0,
            0,
            0,
            20,
            # out of bounds: topleft, top, topright, right, bottomright, bottom, bottomleft, left
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            # some explicit points within the raster
            149,
            139,
            113,
        ]
        np.testing.assert_array_equal(
            exp.gdf[INDICATOR_CENTR + "FL"].values, expected_result
        )

    def test_assign_raster_same_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures.from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        haz = Hazard.from_raster(
            [HAZ_DEMO_FL], haz_type="FL", window=Window(10, 20, 50, 60)
        )
        exp.assign_centroids(haz)
        np.testing.assert_array_equal(
            exp.gdf[INDICATOR_CENTR + "FL"].values,
            np.arange(haz.centroids.size, dtype=int),
        )

    # Test fails because exposures stores the crs in the meta attribute as rasterio object,
    # while the centroids stores the crs in the geodataframe, which is not a rasterio object.
    # The comparison in assign_centroids then fails.
    def test_assign_large_hazard_subset_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures.from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.latitude[[0, 1]] = exp.latitude[[1, 0]]
        exp.longitude[[0, 1]] = exp.longitude[[1, 0]]
        exp.check()
        haz = Hazard.from_raster([HAZ_DEMO_FL], haz_type="FL")
        exp.assign_centroids(haz)
        assigned_centroids = haz.centroids.select(
            sel_cen=exp.gdf[INDICATOR_CENTR + "FL"].values
        )
        np.testing.assert_array_equal(
            np.unique(assigned_centroids.lat), np.unique(exp.latitude)
        )
        np.testing.assert_array_equal(
            np.unique(assigned_centroids.lon), np.unique(exp.longitude)
        )

    def test_affected_total_value(self):
        haz_type = "RF"
        gdf = gpd.GeoDataFrame(
            {
                "value": [1, 2, 3, 4, 5, 6],
                "latitude": [1, 2, 3, 4, 1, 0],
                "longitude": [-1, -2, -3, -4, 0, 1],
                "centr_" + haz_type: [0, 2, 2, 3, -1, 4],
            }
        )
        exp = Exposures(gdf, crs=4326)
        intensity = sp.sparse.csr_matrix(np.array([[0, 0, 1, 10, 2], [-1, 0, 0, 1, 2]]))
        cent = Centroids(lat=np.array([1, 2, 3, 4]), lon=np.array([-1, -2, -3, -4]))
        haz = Hazard(
            haz_type=haz_type, centroids=cent, intensity=intensity, event_id=[1, 2]
        )

        # do not reassign centroids
        tot_val = exp.affected_total_value(
            haz, threshold_affected=0, overwrite_assigned_centroids=False
        )
        self.assertEqual(tot_val, np.sum(exp.gdf["value"][[1, 2, 3, 5]]))
        tot_val = exp.affected_total_value(
            haz, threshold_affected=3, overwrite_assigned_centroids=False
        )
        self.assertEqual(tot_val, np.sum(exp.gdf["value"][[3]]))
        tot_val = exp.affected_total_value(
            haz, threshold_affected=-2, overwrite_assigned_centroids=False
        )
        self.assertEqual(tot_val, np.sum(exp.gdf["value"][[0, 1, 2, 3, 5]]))
        tot_val = exp.affected_total_value(
            haz, threshold_affected=11, overwrite_assigned_centroids=False
        )
        self.assertEqual(tot_val, 0)

        # reassign centroids (i.e. to [0, 1, 2, 3, -1, -1])
        tot_val = exp.affected_total_value(
            haz, threshold_affected=11, overwrite_assigned_centroids=True
        )
        self.assertEqual(tot_val, 0)
        tot_val = exp.affected_total_value(
            haz, threshold_affected=0, overwrite_assigned_centroids=False
        )
        self.assertEqual(tot_val, 7)
        tot_val = exp.affected_total_value(
            haz, threshold_affected=3, overwrite_assigned_centroids=False
        )
        self.assertEqual(tot_val, 4)


class TestChecker(unittest.TestCase):
    """Test logs of check function"""

    def test_error_logs_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.gdf.drop(["value"], inplace=True, axis=1)

        with self.assertRaises(ValueError) as cm:
            expo.check()
        self.assertIn("value missing", str(cm.exception))

    def test_error_logs_wrong_crs(self):
        """Ambiguous crs definition"""
        expo = good_exposures()  # epsg:4326

        # all good
        _expo = Exposures(expo.gdf, meta={"crs": 4326}, crs=DEF_CRS)
        self.assertEqual(expo.crs, _expo.crs)

        # still good: crs in argument and meta override crs from data frame
        _expo = Exposures(expo.gdf, meta={"crs": 4230})
        self.assertNotEqual(expo.crs, _expo.crs)

        _expo = Exposures(expo.gdf, crs="epsg:4230")
        self.assertTrue(u_coord.equal_crs(_expo.crs, 4230))

        # bad: direct and indirect (meta) argument conflict
        with self.assertRaises(ValueError) as cm:
            _expo = Exposures(expo.gdf, meta={"crs": 4230}, crs=4326)
        self.assertIn(
            "conflicting arguments: the given crs is different", str(cm.exception)
        )


class TestIO(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""

    def test_read_template_pass(self):
        """Wrong exposures definition"""
        df = pd.read_excel(ENT_TEMPLATE_XLS)
        exp_df = Exposures(df)
        # set metadata
        exp_df.ref_year = 2020
        exp_df.value_unit = "XSD"
        exp_df.check()

    def test_io_hdf5_pass(self):
        """write and read hdf5"""
        exp_df = Exposures(pd.read_excel(ENT_TEMPLATE_XLS), crs="epsg:32632")
        exp_df.check()
        # set metadata
        exp_df.ref_year = 2020
        exp_df.value_unit = "XSD"

        file_name = DATA_DIR.joinpath("test_hdf5_exp.h5")

        # pd.errors.PerformanceWarning should be suppressed. Therefore, make sure that
        # PerformanceWarning would result in test failure here
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
            exp_df.write_hdf5(file_name)

        exp_read = Exposures.from_hdf5(file_name)

        self.assertEqual(exp_df.ref_year, exp_read.ref_year)
        self.assertEqual(exp_df.value_unit, exp_read.value_unit)
        self.assertEqual(exp_df.description, exp_read.description)
        np.testing.assert_array_equal(exp_df.latitude, exp_read.latitude)
        np.testing.assert_array_equal(exp_df.longitude, exp_read.longitude)
        np.testing.assert_array_equal(exp_df.value, exp_read.value)
        np.testing.assert_array_equal(
            exp_df.data["deductible"].values, exp_read.data["deductible"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["cover"].values, exp_read.data["cover"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["region_id"].values, exp_read.data["region_id"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["category_id"].values, exp_read.data["category_id"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["impf_TC"].values, exp_read.data["impf_TC"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["centr_TC"].values, exp_read.data["centr_TC"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["impf_FL"].values, exp_read.data["impf_FL"].values
        )
        np.testing.assert_array_equal(
            exp_df.data["centr_FL"].values, exp_read.data["centr_FL"].values
        )

        self.assertTrue(
            u_coord.equal_crs(exp_df.crs, exp_read.crs),
            f"{exp_df.crs} and {exp_read.crs} are different",
        )
        self.assertTrue(u_coord.equal_crs(exp_df.gdf.crs, exp_read.gdf.crs))


class TestAddSea(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""

    def test_add_sea_pass(self):
        """Test add_sea function with fake data."""
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12

        exp = Exposures(
            data=dict(
                value=np.arange(0, 1.0e6, 1.0e5),
                latitude=np.linspace(min_lat, max_lat, 10),
                longitude=np.linspace(min_lon, max_lon, 10),
                region_id=np.ones(10),
                impf_TC=np.ones(10),
            ),
            ref_year=2015,
            value_unit="XSD",
        )
        exp.check()

        sea_coast = 100
        sea_res_km = 50
        sea_res = (sea_coast, sea_res_km)
        exp_sea = add_sea(exp, sea_res)
        exp_sea.check()

        sea_coast /= ONE_LAT_KM
        sea_res_km /= ONE_LAT_KM

        min_lat = min_lat - sea_coast
        max_lat = max_lat + sea_coast
        min_lon = min_lon - sea_coast
        max_lon = max_lon + sea_coast
        self.assertEqual(np.min(exp_sea.latitude), min_lat)
        self.assertEqual(np.min(exp_sea.longitude), min_lon)
        np.testing.assert_array_equal(exp_sea.value[:10], np.arange(0, 1.0e6, 1.0e5))
        self.assertEqual(exp_sea.ref_year, exp.ref_year)
        self.assertEqual(exp_sea.value_unit, exp.value_unit)

        on_sea_lat = exp_sea.latitude[11:]
        on_sea_lon = exp_sea.longitude[11:]
        res_on_sea = u_coord.coord_on_land(on_sea_lat, on_sea_lon)
        res_on_sea = ~res_on_sea
        self.assertTrue(np.all(res_on_sea))

        dist = DistanceMetric.get_metric("haversine")
        self.assertAlmostEqual(
            dist.pairwise(
                [
                    [exp_sea.longitude[-1], exp_sea.latitude[-1]],
                    [exp_sea.longitude[-2], exp_sea.latitude[-2]],
                ]
            )[0][1],
            sea_res_km,
        )


class TestConcat(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""

    def setUp(self):
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp = Exposures(
            crs="epsg:3395",
            value=np.arange(0, 1.0e6, 1.0e5),
            lat=np.linspace(min_lat, max_lat, 10),
            lon=np.linspace(min_lon, max_lon, 10),
            ref_year=2015,
            value_unit="XSD",
            data=dict(
                region_id=np.ones(10),
                impf_TC=np.ones(10),
            ),
        )
        self.dummy = exp

    def test_concat_pass(self):
        """Test concat function with fake data."""

        self.dummy.check()

        catexp = Exposures.concat(
            [
                self.dummy,
                self.dummy.gdf,
                pd.DataFrame(self.dummy.gdf.values, columns=self.dummy.gdf.columns),
                self.dummy,
            ]
        )
        self.assertEqual(
            list(self.dummy.gdf.columns), ["region_id", "impf_TC", "geometry", "value"]
        )
        self.assertEqual(self.dummy.gdf.shape, (10, 4))
        self.assertEqual(catexp.gdf.shape, (40, 4))
        self.assertTrue(u_coord.equal_crs(catexp.crs, "epsg:3395"))

    def test_concat_fail(self):
        """Test failing concat function with fake data."""

        with self.assertRaises(TypeError):
            Exposures.concat(
                [self.dummy, self.dummy.gdf, self.dummy.gdf.values, self.dummy]
            )


class TestGeoDFFuncs(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""

    def test_copy_pass(self):
        """Test copy function."""
        exp = good_exposures()
        exp.check()
        exp_copy = exp.copy()
        self.assertIsInstance(exp_copy, Exposures)
        self.assertTrue(u_coord.equal_crs(exp_copy.crs, exp.crs))
        self.assertEqual(exp_copy.ref_year, exp.ref_year)
        self.assertEqual(exp_copy.value_unit, exp.value_unit)
        self.assertEqual(exp_copy.description, exp.description)
        np.testing.assert_array_equal(exp_copy.latitude, exp.latitude)
        np.testing.assert_array_equal(exp_copy.longitude, exp.longitude)

    def test_to_crs_inplace_pass(self):
        """Test to_crs function inplace."""
        exp = good_exposures()
        exp.check()
        exp.to_crs("epsg:3395", inplace=True)
        self.assertIsInstance(exp, Exposures)
        self.assertTrue(u_coord.equal_crs(exp.crs, "epsg:3395"))
        self.assertEqual(exp.ref_year, DEF_REF_YEAR)
        self.assertEqual(exp.value_unit, DEF_VALUE_UNIT)
        self.assertEqual(exp.description, None)

    def test_to_crs_pass(self):
        """Test to_crs function copy."""
        exp = good_exposures()
        exp.check()
        exp_tr = exp.to_crs("epsg:3395")
        self.assertIsInstance(exp, Exposures)
        self.assertTrue(u_coord.equal_crs(exp.crs, DEF_CRS))
        self.assertTrue(u_coord.equal_crs(exp_tr.crs, "epsg:3395"))
        self.assertEqual(exp_tr.ref_year, DEF_REF_YEAR)
        self.assertEqual(exp_tr.value_unit, DEF_VALUE_UNIT)
        self.assertEqual(exp_tr.description, None)

    def test_constructor_pass(self):
        """Test initialization with input GeoDataFrame"""
        in_gpd = gpd.GeoDataFrame(
            dict(latitude=range(10), longitude=[0] * 10, value=np.zeros(10))
        )
        in_exp = Exposures(in_gpd, ref_year=2015)
        self.assertEqual(in_exp.ref_year, 2015)
        np.testing.assert_array_equal(in_exp.value, np.zeros(10))
        self.assertEqual(in_exp.gdf.geometry[0], Point(0, 0))

    def test_error_on_access_item(self):
        """Test error output when trying to access items as in CLIMADA 1.x"""
        expo = good_exposures()
        with self.assertRaises(TypeError) as err:
            expo["value"] = 3
        self.assertIn("CLIMADA 2", str(err.exception))
        self.assertIn("gdf", str(err.exception))

    def test_set_gdf(self):
        """Test setting the GeoDataFrame"""
        empty_gdf = gpd.GeoDataFrame()
        gdf_without_geometry = good_exposures().gdf
        good_exp = good_exposures()
        good_exp.set_crs(crs="epsg:3395")
        gdf_with_geometry = good_exp.gdf

        probe = Exposures()
        self.assertRaises(ValueError, probe.set_gdf, pd.DataFrame())

        probe.set_gdf(empty_gdf)
        self.assertTrue(probe.gdf.equals(gpd.GeoDataFrame().set_geometry([])))
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.gdf.crs))

        probe.set_gdf(gdf_with_geometry)
        self.assertTrue(probe.gdf.equals(gdf_with_geometry))
        self.assertTrue(u_coord.equal_crs("epsg:3395", gdf_with_geometry.crs))
        self.assertTrue(
            u_coord.equal_crs("epsg:3395", probe.crs), f"unexpected: {probe.crs}"
        )
        self.assertTrue(u_coord.equal_crs("epsg:3395", probe.gdf.crs))

        probe.set_gdf(gdf_without_geometry)
        self.assertTrue(probe.gdf.equals(good_exposures().gdf))
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.gdf.crs))

    def test_set_crs(self):
        """Test setting the CRS"""
        empty_gdf = gpd.GeoDataFrame()
        gdf_without_geometry = good_exposures().gdf
        good_exp = good_exposures()
        gdf_with_geometry = good_exp.gdf

        probe = Exposures(gdf_without_geometry)
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        probe.set_crs("epsg:3395")
        self.assertTrue(u_coord.equal_crs("epsg:3395", probe.crs))

        probe = Exposures(gdf_with_geometry)
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        probe.set_crs(DEF_CRS)
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        probe.set_crs("epsg:3395")
        self.assertTrue(u_coord.equal_crs("epsg:3395", probe.crs))

    def test_to_crs_epsg_crs(self):
        """Check that if crs and epsg are both provided a ValueError is raised"""
        with self.assertRaises(ValueError) as cm:
            Exposures.to_crs(self, crs="GCS", epsg=26915)
        self.assertEqual("one of crs or epsg must be None", str(cm.exception))

    def test_latlon_with_polygons(self):
        """Check for proper error message if the data frame contains non-Point shapes"""
        poly = Polygon(
            [(10.0, 0.0), (10.0, 1.0), (11.0, 1.0), (11.0, 0.0), (10.0, 0.0)]
        )
        point = Point((1, -1))
        multi = MultiPolygon(
            [
                (
                    ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
                    [((0.1, 1.1), (0.1, 1.2), (0.2, 1.2), (0.2, 1.1))],
                )
            ]
        )
        poly = Polygon()
        exp = Exposures(geometry=[poly, point, multi, poly])
        with self.assertRaises(ValueError) as valer:
            exp.latitude
        self.assertEqual(
            "Can only calculate latitude from Points."
            " GeoDataFrame contains Polygon, MultiPolygon."
            " Please see the lines_polygons module tutorial.",
            str(valer.exception),
        )
        with self.assertRaises(ValueError) as valer:
            exp.longitude
        self.assertEqual(
            "Can only calculate longitude from Points."
            " GeoDataFrame contains Polygon, MultiPolygon."
            " Please see the lines_polygons module tutorial.",
            str(valer.exception),
        )


class TestImpactFunctions(unittest.TestCase):
    """Test impact function handling"""

    def test_get_impf_column(self):
        """Test the get_impf_column"""
        expo = good_exposures()

        # impf column is 'impf_NA'
        self.assertEqual("impf_NA", expo.get_impf_column("NA"))
        self.assertRaises(ValueError, expo.get_impf_column)
        self.assertRaises(ValueError, expo.get_impf_column, "HAZ")

        # removed impf column
        expo.data.drop(columns="impf_NA", inplace=True)
        self.assertRaises(ValueError, expo.get_impf_column, "NA")
        self.assertRaises(ValueError, expo.get_impf_column)

        # default (anonymous) impf column
        expo.data["impf_"] = 1
        self.assertEqual("impf_", expo.get_impf_column())
        self.assertEqual("impf_", expo.get_impf_column("HAZ"))

        # rename impf column to old style column name
        expo.data.rename(columns={"impf_": "if_"}, inplace=True)
        self.assertEqual("if_", expo.get_impf_column())
        self.assertEqual("if_", expo.get_impf_column("HAZ"))

        # rename impf column to old style column name
        expo.data.rename(columns={"if_": "if_NA"}, inplace=True)
        self.assertEqual("if_NA", expo.get_impf_column("NA"))
        self.assertRaises(ValueError, expo.get_impf_column)
        self.assertRaises(ValueError, expo.get_impf_column, "HAZ")

        # add anonymous impf column
        expo.data["impf_"] = expo.region_id
        self.assertEqual("if_NA", expo.get_impf_column("NA"))
        self.assertEqual("impf_", expo.get_impf_column())
        self.assertEqual("impf_", expo.get_impf_column("HAZ"))


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestFuncs)
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestChecker))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIO))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAddSea))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestConcat))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGeoDFFuncs))
    TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestImpactFunctions))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
