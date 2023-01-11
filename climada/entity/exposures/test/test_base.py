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
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import DistanceMetric
import rasterio
from rasterio.windows import Window

from climada import CONFIG
from climada.entity.exposures.base import Exposures, INDICATOR_IMPF, \
     INDICATOR_CENTR, add_sea, DEF_REF_YEAR, DEF_VALUE_UNIT
from climada.entity import LitPop
from climada.entity.tag import Tag
from climada.hazard.base import Hazard, Centroids
from climada.util.constants import ENT_TEMPLATE_XLS, ONE_LAT_KM, DEF_CRS, HAZ_DEMO_FL
import climada.util.coordinates as u_coord

DATA_DIR = CONFIG.exposures.test_data.dir()

def good_exposures():
    """Followng values are defined for each exposure"""
    data = {}
    data['latitude'] = np.array([1, 2, 3])
    data['longitude'] = np.array([2, 3, 4])
    data['value'] = np.array([1, 2, 3])
    data['deductible'] = np.array([1, 2, 3])
    data[INDICATOR_IMPF + 'NA'] = np.array([1, 2, 3])
    data['category_id'] = np.array([1, 2, 3])
    data['region_id'] = np.array([1, 2, 3])
    data[INDICATOR_CENTR + 'TC'] = np.array([1, 2, 3])

    expo = Exposures(gpd.GeoDataFrame(data=data))
    return expo

class TestFuncs(unittest.TestCase):
    """Check assign function"""

    def test_assign_pass(self):
        """Check that attribute `assigned` is correctly set."""
        np_rand = np.random.RandomState(123456789)

        haz = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL', window=Window(10, 20, 50, 60))
        haz.raster_to_vector()
        ncentroids = haz.centroids.size

        exp = Exposures(crs=haz.centroids.crs)

        # some are matching exactly, some are geographically close
        exp.gdf['longitude'] = np.concatenate([
            haz.centroids.lon, haz.centroids.lon + 0.001 * (-0.5 + np_rand.rand(ncentroids))])
        exp.gdf['latitude'] = np.concatenate([
            haz.centroids.lat, haz.centroids.lat + 0.001 * (-0.5 + np_rand.rand(ncentroids))])
        expected_result = np.concatenate([np.arange(ncentroids), np.arange(ncentroids)])

        # make sure that it works for both float32 and float64
        for test_dtype in [np.float64, np.float32]:
            haz.centroids.lat = haz.centroids.lat.astype(test_dtype)
            haz.centroids.lon = haz.centroids.lon.astype(test_dtype)
            exp.assign_centroids(haz)
            self.assertEqual(exp.gdf.shape[0], len(exp.gdf[INDICATOR_CENTR + 'FL']))
            np.testing.assert_array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values, expected_result)
            exp.assign_centroids(Hazard(), overwrite=False)
            self.assertEqual(exp.gdf.shape[0], len(exp.gdf[INDICATOR_CENTR + 'FL']))
            np.testing.assert_array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values, expected_result)

    def test__init__meta_type(self):
        """ Check if meta of type list raises a ValueError in __init__"""
        with self.assertRaises(ValueError) as cm:
            Exposures(meta=[])
        self.assertEqual("meta must be a dictionary",
                      str(cm.exception))

    def test__init__geometry_type(self):
        """Check that initialization fails when `geometry` is given as a `str` argument"""
        with self.assertRaises(ValueError) as cm:
            Exposures(geometry='myname')
        self.assertEqual("Exposures is not able to handle customized 'geometry' column names.",
                         str(cm.exception))

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
        self.assertAlmostEqual(exp.gdf['latitude'].max(),
                               10.248220966978932 - 0.009000000000000341 / 2)
        self.assertAlmostEqual(exp.gdf['latitude'].min(),
                               10.248220966978932 - 0.009000000000000341
                               / 2 - 59 * 0.009000000000000341)
        self.assertAlmostEqual(exp.gdf['longitude'].min(),
                               -69.2471495969998 + 0.009000000000000341 / 2)
        self.assertAlmostEqual(exp.gdf['longitude'].max(),
                               -69.2471495969998 + 0.009000000000000341
                               / 2 + 49 * 0.009000000000000341)
        self.assertEqual(len(exp.gdf), 60 * 50)
        self.assertAlmostEqual(exp.gdf.value.values.reshape((60, 50))[25, 12], 0.056825936)

    def test_assign_raster_pass(self):
        """Test assign_centroids with raster hazard"""
        # explicit, easy-to-understand raster centroids for hazard
        meta = {
            'count': 1, 'crs': DEF_CRS,
            'width': 20, 'height': 10,
            'transform': rasterio.Affine(1.5, 0.0, -20, 0.0, -1.4, 8)
        }
        haz = Hazard('FL', centroids=Centroids(meta=meta))

        # explicit points with known results (see `expected_result` for details)
        exp = Exposures(crs=DEF_CRS)
        exp.gdf['longitude'] = np.array([
            -20.1, -20.0, -19.8, -19.0, -18.6, -18.4,
            -19.0, -19.0, -19.0, -19.0,
            -20.1, 0.0, 10.1, 10.1, 10.1, 0.0, -20.2, -20.3,
            -6.4, 9.8, 0.0,
        ])
        exp.gdf['latitude'] = np.array([
            7.3, 7.3, 7.3, 7.3, 7.3, 7.3,
            8.1, 7.9, 6.7, 6.5,
            8.1, 8.2, 8.3, 0.0, -6.1, -6.2, -6.3, 0.0,
            -1.9, -1.7, 0.0,
        ])
        exp.assign_centroids(haz)

        expected_result = [
            # constant y-value, varying x-value
            -1, 0, 0, 0, 0, 1,
            # constant x-value, varying y-value
            -1, 0, 0, 20,
            # out of bounds: topleft, top, topright, right, bottomright, bottom, bottomleft, left
            -1, -1, -1, -1, -1, -1, -1, -1,
            # some explicit points within the raster
            149, 139, 113,
        ]
        np.testing.assert_array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values, expected_result)


    def test_assign_raster_same_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures.from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.check()
        haz = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL', window=Window(10, 20, 50, 60))
        exp.assign_centroids(haz)
        np.testing.assert_array_equal(exp.gdf[INDICATOR_CENTR + 'FL'].values,
                                      np.arange(haz.centroids.size, dtype=int))

    def test_assign_large_hazard_subset_pass(self):
        """Test assign_centroids with raster hazard"""
        exp = Exposures.from_raster(HAZ_DEMO_FL, window=Window(10, 20, 50, 60))
        exp.gdf.latitude[[0, 1]] = exp.gdf.latitude[[1, 0]]
        exp.gdf.longitude[[0, 1]] = exp.gdf.longitude[[1, 0]]
        exp.check()
        haz = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL')
        haz.raster_to_vector()
        exp.assign_centroids(haz)
        assigned_centroids = haz.centroids.select(sel_cen=exp.gdf[INDICATOR_CENTR + 'FL'].values)
        np.testing.assert_array_equal(assigned_centroids.lat, exp.gdf.latitude)
        np.testing.assert_array_equal(assigned_centroids.lon, exp.gdf.longitude)

    def test_affected_total_value(self):
        exp = Exposures.from_raster(HAZ_DEMO_FL, window=Window(25, 90, 10, 5))
        haz = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL', window=Window(25, 90, 10, 5))
        exp.assign_centroids(haz)
        tot_val = exp.affected_total_value(haz)
        self.assertEqual(tot_val, np.sum(exp.gdf.value))
        new_centr = exp.gdf.centr_FL
        new_centr[6] = -1
        exp.gdf.centr_FL = new_centr
        tot_val = exp.affected_total_value(haz)
        self.assertAlmostEqual(tot_val, np.sum(exp.gdf.value) - exp.gdf.value[6], places=4)
        new_vals = exp.gdf.value
        new_vals[7] = 0
        exp.gdf.value = new_vals
        tot_val = exp.affected_total_value(haz)
        self.assertAlmostEqual(tot_val, np.sum(exp.gdf.value) - exp.gdf.value[6], places=4)
        exp.gdf.centr_FL = -1
        tot_val = exp.affected_total_value(haz)
        self.assertEqual(tot_val, 0)

class TestChecker(unittest.TestCase):
    """Test logs of check function"""

    def test_error_logs_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.gdf.drop(['longitude'], inplace=True, axis=1)

        with self.assertRaises(ValueError) as cm:
            expo.check()
        self.assertIn('longitude missing', str(cm.exception))

    def test_error_logs_wrong_crs(self):
        """Ambiguous crs definition"""
        expo = good_exposures()
        expo.set_geometry_points()  # sets crs to 4326

        # all good
        _expo = Exposures(expo.gdf, meta={'crs':4326}, crs=DEF_CRS)

        with self.assertRaises(ValueError) as cm:
            _expo = Exposures(expo.gdf, meta={'crs':4230}, crs=4326)
        self.assertIn("Inconsistent CRS definition, crs and meta arguments don't match",
                      str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _expo = Exposures(expo.gdf, meta={'crs':4230})
        self.assertIn("Inconsistent CRS definition, data doesn't match meta or crs argument",
                      str(cm.exception))

        with self.assertRaises(ValueError) as cm:
            _expo = Exposures(expo.gdf, crs='epsg:4230')
        self.assertIn("Inconsistent CRS definition, data doesn't match meta or crs argument",
                      str(cm.exception))

        _expo = Exposures(expo.gdf)
        _expo.meta['crs'] = 'epsg:4230'
        with self.assertRaises(ValueError) as cm:
            _expo.check()
        self.assertIn("Inconsistent CRS definition, gdf (EPSG:4326) attribute doesn't match "
                      "meta (epsg:4230) attribute.", str(cm.exception))

    def test_error_geometry_fail(self):
        """Wrong exposures definition"""
        expo = good_exposures()
        expo.set_geometry_points()
        expo.gdf.latitude.values[0] = 5

        with self.assertRaises(ValueError):
            expo.check()

class TestIO(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""

    def test_read_template_pass(self):
        """Wrong exposures definition"""
        df = pd.read_excel(ENT_TEMPLATE_XLS)
        exp_df = Exposures(df)
        # set metadata
        exp_df.ref_year = 2020
        exp_df.tag = Tag(ENT_TEMPLATE_XLS, 'ENT_TEMPLATE_XLS')
        exp_df.value_unit = 'XSD'
        exp_df.check()

    def test_io_hdf5_pass(self):
        """write and read hdf5"""
        exp_df = Exposures(pd.read_excel(ENT_TEMPLATE_XLS), crs="epsg:32632")
        exp_df.set_geometry_points()
        exp_df.check()
        # set metadata
        exp_df.ref_year = 2020
        exp_df.tag = Tag(ENT_TEMPLATE_XLS, 'ENT_TEMPLATE_XLS')
        exp_df.value_unit = 'XSD'

        file_name = DATA_DIR.joinpath('test_hdf5_exp.h5')

        # pd.errors.PerformanceWarning should be suppressed. Therefore, make sure that
        # PerformanceWarning would result in test failure here
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=pd.errors.PerformanceWarning)
            exp_df.write_hdf5(file_name)

        exp_read = Exposures.from_hdf5(file_name)

        self.assertEqual(exp_df.ref_year, exp_read.ref_year)
        self.assertEqual(exp_df.value_unit, exp_read.value_unit)
        self.assertDictEqual(exp_df.meta, exp_read.meta)
        self.assertTrue(u_coord.equal_crs(exp_df.crs, exp_read.crs))
        self.assertTrue(u_coord.equal_crs(exp_df.gdf.crs, exp_read.gdf.crs))
        self.assertEqual(exp_df.tag.file_name, exp_read.tag.file_name)
        self.assertEqual(exp_df.tag.description, exp_read.tag.description)
        np.testing.assert_array_equal(exp_df.gdf.latitude.values, exp_read.gdf.latitude.values)
        np.testing.assert_array_equal(exp_df.gdf.longitude.values, exp_read.gdf.longitude.values)
        np.testing.assert_array_equal(exp_df.gdf.value.values, exp_read.gdf.value.values)
        np.testing.assert_array_equal(exp_df.gdf.deductible.values, exp_read.gdf.deductible.values)
        np.testing.assert_array_equal(exp_df.gdf.cover.values, exp_read.gdf.cover.values)
        np.testing.assert_array_equal(exp_df.gdf.region_id.values, exp_read.gdf.region_id.values)
        np.testing.assert_array_equal(exp_df.gdf.category_id.values, exp_read.gdf.category_id.values)
        np.testing.assert_array_equal(exp_df.gdf.impf_TC.values, exp_read.gdf.impf_TC.values)
        np.testing.assert_array_equal(exp_df.gdf.centr_TC.values, exp_read.gdf.centr_TC.values)
        np.testing.assert_array_equal(exp_df.gdf.impf_FL.values, exp_read.gdf.impf_FL.values)
        np.testing.assert_array_equal(exp_df.gdf.centr_FL.values, exp_read.gdf.centr_FL.values)

        for point_df, point_read in zip(exp_df.gdf.geometry.values, exp_read.gdf.geometry.values):
            self.assertEqual(point_df.x, point_read.x)
            self.assertEqual(point_df.y, point_read.y)

class TestAddSea(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""
    def test_add_sea_pass(self):
        """Test add_sea function with fake data."""
        exp = Exposures()
        exp.gdf['value'] = np.arange(0, 1.0e6, 1.0e5)
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp.gdf['latitude'] = np.linspace(min_lat, max_lat, 10)
        exp.gdf['longitude'] = np.linspace(min_lon, max_lon, 10)
        exp.gdf['region_id'] = np.ones(10)
        exp.gdf['impf_TC'] = np.ones(10)
        exp.ref_year = 2015
        exp.value_unit = 'XSD'
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
        self.assertEqual(np.min(exp_sea.gdf.latitude), min_lat)
        self.assertEqual(np.min(exp_sea.gdf.longitude), min_lon)
        np.testing.assert_array_equal(exp_sea.gdf.value.values[:10], np.arange(0, 1.0e6, 1.0e5))
        self.assertEqual(exp_sea.ref_year, exp.ref_year)
        self.assertEqual(exp_sea.value_unit, exp.value_unit)

        on_sea_lat = exp_sea.gdf.latitude.values[11:]
        on_sea_lon = exp_sea.gdf.longitude.values[11:]
        res_on_sea = u_coord.coord_on_land(on_sea_lat, on_sea_lon)
        res_on_sea = ~res_on_sea
        self.assertTrue(np.all(res_on_sea))

        dist = DistanceMetric.get_metric('haversine')
        self.assertAlmostEqual(dist.pairwise([
            [exp_sea.gdf.longitude.values[-1], exp_sea.gdf.latitude.values[-1]],
            [exp_sea.gdf.longitude.values[-2], exp_sea.gdf.latitude.values[-2]],
        ])[0][1], sea_res_km)


class TestConcat(unittest.TestCase):
    """Check constructor Exposures through DataFrames readers"""
    def setUp(self):
        exp = Exposures(crs='epsg:3395')
        exp.gdf['value'] = np.arange(0, 1.0e6, 1.0e5)
        min_lat, max_lat = 27.5, 30
        min_lon, max_lon = -18, -12
        exp.gdf['latitude'] = np.linspace(min_lat, max_lat, 10)
        exp.gdf['longitude'] = np.linspace(min_lon, max_lon, 10)
        exp.gdf['region_id'] = np.ones(10)
        exp.gdf['impf_TC'] = np.ones(10)
        exp.ref_year = 2015
        exp.value_unit = 'XSD'
        self.dummy = exp

    def test_concat_pass(self):
        """Test concat function with fake data."""

        self.dummy.check()

        catexp = Exposures.concat([self.dummy, self.dummy.gdf, pd.DataFrame(self.dummy.gdf.values, columns=self.dummy.gdf.columns), self.dummy])
        self.assertEqual(self.dummy.gdf.shape, (10,5))
        self.assertEqual(catexp.gdf.shape, (40,5))
        self.assertTrue(u_coord.equal_crs(catexp.crs, 'epsg:3395'))

    def test_concat_fail(self):
        """Test failing concat function with fake data."""

        with self.assertRaises(TypeError):
            Exposures.concat([self.dummy, self.dummy.gdf, self.dummy.gdf.values, self.dummy])


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
        self.assertEqual(exp_copy.tag.description, exp.tag.description)
        self.assertEqual(exp_copy.tag.file_name, exp.tag.file_name)
        np.testing.assert_array_equal(exp_copy.gdf.latitude.values, exp.gdf.latitude.values)
        np.testing.assert_array_equal(exp_copy.gdf.longitude.values, exp.gdf.longitude.values)

    def test_to_crs_inplace_pass(self):
        """Test to_crs function inplace."""
        exp = good_exposures()
        exp.set_geometry_points()
        exp.check()
        exp.to_crs('epsg:3395', inplace=True)
        self.assertIsInstance(exp, Exposures)
        self.assertTrue(u_coord.equal_crs(exp.crs, 'epsg:3395'))
        self.assertEqual(exp.ref_year, DEF_REF_YEAR)
        self.assertEqual(exp.value_unit, DEF_VALUE_UNIT)
        self.assertEqual(exp.tag.description, '')
        self.assertEqual(exp.tag.file_name, '')

    def test_to_crs_pass(self):
        """Test to_crs function copy."""
        exp = good_exposures()
        exp.set_geometry_points()
        exp.check()
        exp_tr = exp.to_crs('epsg:3395')
        self.assertIsInstance(exp, Exposures)
        self.assertTrue(u_coord.equal_crs(exp.crs, DEF_CRS))
        self.assertTrue(u_coord.equal_crs(exp_tr.crs, 'epsg:3395'))
        self.assertEqual(exp_tr.ref_year, DEF_REF_YEAR)
        self.assertEqual(exp_tr.value_unit, DEF_VALUE_UNIT)
        self.assertEqual(exp_tr.tag.description, '')
        self.assertEqual(exp_tr.tag.file_name, '')

    def test_constructor_pass(self):
        """Test initialization with input GeoDataFrame"""
        in_gpd = gpd.GeoDataFrame()
        in_gpd['value'] = np.zeros(10)
        in_gpd.ref_year = 2015
        in_exp = Exposures(in_gpd, ref_year=2015)
        self.assertEqual(in_exp.ref_year, 2015)
        np.testing.assert_array_equal(in_exp.gdf.value, np.zeros(10))

    def test_error_on_access_item(self):
        """Test error output when trying to access items as in CLIMADA 1.x"""
        expo = good_exposures()
        with self.assertRaises(TypeError) as err:
            expo['value'] = 3
        self.assertIn("CLIMADA 2", str(err.exception))
        self.assertIn("gdf", str(err.exception))

    def test_set_gdf(self):
        """Test setting the GeoDataFrame"""
        empty_gdf = gpd.GeoDataFrame()
        gdf_without_geometry = good_exposures().gdf
        good_exp = good_exposures()
        good_exp.set_crs(crs='epsg:3395')
        good_exp.set_geometry_points()
        gdf_with_geometry = good_exp.gdf

        probe = Exposures()
        self.assertRaises(ValueError, probe.set_gdf, pd.DataFrame())

        probe.set_gdf(empty_gdf)
        self.assertTrue(probe.gdf.equals(gpd.GeoDataFrame()))
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        self.assertFalse(hasattr(probe.gdf, "crs"))

        probe.set_gdf(gdf_with_geometry)
        self.assertTrue(probe.gdf.equals(gdf_with_geometry))
        self.assertTrue(u_coord.equal_crs('epsg:3395', probe.crs))
        self.assertTrue(u_coord.equal_crs('epsg:3395', probe.gdf.crs))

        probe.set_gdf(gdf_without_geometry)
        self.assertTrue(probe.gdf.equals(good_exposures().gdf))
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        self.assertFalse(hasattr(probe.gdf, "crs"))

    def test_set_crs(self):
        """Test setting the CRS"""
        empty_gdf = gpd.GeoDataFrame()
        gdf_without_geometry = good_exposures().gdf
        good_exp = good_exposures()
        good_exp.set_geometry_points()
        gdf_with_geometry = good_exp.gdf

        probe = Exposures(gdf_without_geometry)
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        probe.set_crs('epsg:3395')
        self.assertTrue(u_coord.equal_crs('epsg:3395', probe.crs))

        probe = Exposures(gdf_with_geometry)
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        probe.set_crs(DEF_CRS)
        self.assertTrue(u_coord.equal_crs(DEF_CRS, probe.crs))
        self.assertRaises(ValueError, probe.set_crs, 'epsg:3395')
        self.assertTrue(u_coord.equal_crs('EPSG:4326', probe.meta.get('crs')))

    def test_to_crs_epsg_crs(self):
        """ Check that if crs and epsg are both provided a ValueError is raised"""
        with self.assertRaises(ValueError) as cm:
            Exposures.to_crs(self, crs='GCS', epsg=26915)
        self.assertEqual("one of crs or epsg must be None", str(cm.exception))

class TestImpactFunctions(unittest.TestCase):
    """Test impact function handling"""
    def test_get_impf_column(self):
        """Test the get_impf_column"""
        expo = good_exposures()

        # impf column is 'impf_NA'
        self.assertEqual('impf_NA', expo.get_impf_column('NA'))
        self.assertRaises(ValueError, expo.get_impf_column)
        self.assertRaises(ValueError, expo.get_impf_column, 'HAZ')

        # removed impf column
        expo.gdf.drop(columns='impf_NA', inplace=True)
        self.assertRaises(ValueError, expo.get_impf_column, 'NA')
        self.assertRaises(ValueError, expo.get_impf_column)

        # default (anonymous) impf column
        expo.check()
        self.assertEqual('impf_', expo.get_impf_column())
        self.assertEqual('impf_', expo.get_impf_column('HAZ'))

        # rename impf column to old style column name
        expo.gdf.rename(columns={'impf_': 'if_'}, inplace=True)
        expo.check()
        self.assertEqual('if_', expo.get_impf_column())
        self.assertEqual('if_', expo.get_impf_column('HAZ'))

        # rename impf column to old style column name
        expo.gdf.rename(columns={'if_': 'if_NA'}, inplace=True)
        expo.check()
        self.assertEqual('if_NA', expo.get_impf_column('NA'))
        self.assertRaises(ValueError, expo.get_impf_column)
        self.assertRaises(ValueError, expo.get_impf_column, 'HAZ')

        # add anonymous impf column
        expo.gdf['impf_'] = expo.gdf['region_id']
        self.assertEqual('if_NA', expo.get_impf_column('NA'))
        self.assertEqual('impf_', expo.get_impf_column())
        self.assertEqual('impf_', expo.get_impf_column('HAZ'))


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
