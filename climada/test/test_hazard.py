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

Test Hazard base class.
"""

import unittest
import numpy as np
import datetime as dt
from pathlib import Path
from scipy import sparse

from climada import CONFIG
from climada.hazard import tc_tracks as tc
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.hazard.storm_europe import StormEurope
from climada.util.constants import (HAZ_DEMO_FL, WS_DEMO_NC)
from climada.util.api_client import Client
from climada.util import coordinates as u_coord
from climada.test import get_test_file

DATA_DIR = CONFIG.test_data.dir()

# Hazard test file from Git repository. Fraction is 1. Format: matlab.
HAZ_TEST_MAT :Path = get_test_file('atl_prob_no_name', file_format='matlab')

class TestCentroids(unittest.TestCase):
    """Test centroids functionalities"""

    def test_read_write_raster_pass(self):
        """Test write_raster: Hazard from raster data"""
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL])
        haz_fl.haz_type = 'FL'
        haz_fl.check()

        self.assertEqual(haz_fl.intensity.shape, (1, 1032226))
        self.assertEqual(haz_fl.intensity.min(), -9999.0)
        self.assertAlmostEqual(haz_fl.intensity.max(), 4.662774085998535)

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'))

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')])
        haz_fl.haz_type = 'FL'
        self.assertTrue(np.allclose(haz_fl.intensity.toarray(), haz_read.intensity.toarray()))
        self.assertEqual(np.unique(np.array(haz_fl.fraction.toarray())).size, 2)

    def test_read_raster_pool_pass(self):
        """Test from_raster constructor with pool"""
        from pathos.pools import ProcessPool as Pool
        pool = Pool()
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL], haz_type='FL', pool=pool)
        haz_fl.check()

        self.assertEqual(haz_fl.intensity.shape, (1, 1032226))
        self.assertEqual(haz_fl.intensity.min(), -9999.0)
        self.assertAlmostEqual(haz_fl.intensity.max(), 4.662774085998535)
        pool.close()
        pool.join()

    def test_read_write_vector_pass(self):
        """Test write_raster: Hazard from vector data"""
        haz_fl = Hazard('FL',
                        event_id=np.array([1]),
                        date=np.array([1]),
                        frequency=np.array([1]),
                        orig=np.array([1]),
                        event_name=['1'],
                        intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
                        fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
                        centroids=Centroids.from_lat_lon(
                            np.array([1, 2, 3]), np.array([1, 2, 3])),)
        haz_fl.check()

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'))

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')], haz_type='FL')
        self.assertEqual(haz_read.intensity.shape, (1, 9))
        self.assertTrue(np.allclose(np.unique(np.array(haz_read.intensity.toarray())),
                                    np.array([0.0, 0.1, 0.2, 0.5])))

    def test_write_fraction_pass(self):
        """Test write_raster with fraction"""
        haz_fl = Hazard('FL',
                        event_id=np.array([1]),
                        date=np.array([1]),
                        frequency=np.array([1]),
                        orig=np.array([1]),
                        event_name=['1'],
                        intensity=sparse.csr_matrix(np.array([0.5, 0.2, 0.1])),
                        fraction=sparse.csr_matrix(np.array([0.5, 0.2, 0.1]) / 2),
                        centroids=Centroids.from_lat_lon(
                            np.array([1, 2, 3]), np.array([1, 2, 3])),)
        haz_fl.check()

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'), intensity=False)

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')],
                                      files_fraction=[DATA_DIR.joinpath('test_write_hazard.tif')],
                                      haz_type='FL')
        self.assertEqual(haz_read.intensity.shape, (1, 9))
        self.assertEqual(haz_read.fraction.shape, (1, 9))
        self.assertTrue(np.allclose(np.unique(np.array(haz_read.fraction.toarray())),
                                    np.array([0.0, 0.05, 0.1, 0.25])))
        self.assertTrue(np.allclose(np.unique(np.array(haz_read.intensity.toarray())),
                                    np.array([0.0, 0.05, 0.1, 0.25])))


class TestStormEurope(unittest.TestCase):
    """Test methods to create StormEurope object"""

    def test_from_footprints(self):
        """Test from_footprints constructor, using one small test files"""

        def _test_first(haz):
            """Test the expected first entry of the hazard"""
            self.assertEqual(haz.haz_type, "WS")
            self.assertEqual(haz.units, "m/s")
            self.assertEqual(haz.event_id.size, 1)
            self.assertEqual(haz.date.size, 1)
            self.assertEqual(dt.datetime.fromordinal(haz.date[0]).year, 1999)
            self.assertEqual(dt.datetime.fromordinal(haz.date[0]).month, 12)
            self.assertEqual(dt.datetime.fromordinal(haz.date[0]).day, 26)
            self.assertEqual(haz.event_id[0], 1)
            self.assertEqual(haz.event_name[0], "Lothar")
            self.assertIsInstance(haz.intensity, sparse.csr_matrix)
            self.assertIsInstance(haz.fraction, sparse.csr_matrix)
            self.assertEqual(haz.intensity.shape, (1, 9944))
            self.assertEqual(haz.fraction.shape, (1, 9944))
            self.assertEqual(haz.frequency[0], 1.0)

        # Load first entry
        storms = StormEurope.from_footprints(
            WS_DEMO_NC[0]
        )
        _test_first(storms)

        # Omit the second file, should be the same result
        storms = StormEurope.from_footprints(WS_DEMO_NC, files_omit=str(WS_DEMO_NC[1]))
        _test_first(storms)

        # Now load both
        storms = StormEurope.from_footprints(WS_DEMO_NC)

        self.assertEqual(storms.haz_type, "WS")
        self.assertEqual(storms.units, "m/s")
        self.assertEqual(storms.event_id.size, 2)
        self.assertEqual(storms.date.size, 2)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).year, 1999)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).month, 12)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).day, 26)
        self.assertEqual(storms.event_id[0], 1)
        self.assertEqual(storms.event_name[0], "Lothar")
        self.assertIsInstance(storms.intensity, sparse.csr_matrix)
        self.assertIsInstance(storms.fraction, sparse.csr_matrix)
        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(storms.fraction.shape, (2, 9944))

    def test_icon_read(self):
        """test reading from icon grib"""
        # for this test the forecast file is supposed to be already downloaded from the dwd
        # another download would fail because the files are available for 24h only
        # instead, we download it as a test dataset through the climada data api
        apiclient = Client()
        ds = apiclient.get_dataset_info(
            name="test_storm_europe_icon_2021012800", status="test_dataset"
        )
        dsdir, _ = apiclient.download_dataset(ds)
        haz = StormEurope.from_icon_grib(
            dt.datetime(2021, 1, 28),
            dt.datetime(2021, 1, 28),
            model_name="test",
            grib_dir=dsdir,
            delete_raw_data=False,
        )
        self.assertEqual(haz.haz_type, "WS")
        self.assertEqual(haz.units, "m/s")
        self.assertEqual(haz.event_id.size, 40)
        self.assertEqual(haz.date.size, 40)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).year, 2021)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).month, 1)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).day, 28)
        self.assertEqual(haz.event_id[-1], 40)
        self.assertEqual(haz.event_name[-1], "2021-01-28_ens40")
        self.assertIsInstance(haz.intensity, sparse.csr_matrix)
        self.assertIsInstance(haz.fraction, sparse.csr_matrix)
        self.assertEqual(haz.intensity.shape, (40, 49))
        self.assertAlmostEqual(haz.intensity.max(), 17.276321, places=3)
        self.assertEqual(haz.fraction.shape, (40, 49))
        with self.assertLogs("climada.hazard.storm_europe", level="WARNING") as cm:
            with self.assertRaises(ValueError):
                haz = StormEurope.from_icon_grib(
                    dt.datetime(2021, 1, 28, 6),
                    dt.datetime(2021, 1, 28),
                    model_name="test",
                    grib_dir=CONFIG.hazard.test_data.str(),
                    delete_raw_data=False,
                )
        self.assertEqual(len(cm.output), 1)
        self.assertIn("event definition is inaccuratly implemented", cm.output[0])


class TestTcTracks(unittest.TestCase):
    """Test methods to create TcTracks objects from netcdf"""

    def test_ibtracs_with_basin(self):
        """Filter TCs by (genesis) basin."""
        # South Atlantic (not usually a TC location at all)
        tc_track = tc.TCTracks.from_ibtracs_netcdf(basin="SA")
        self.assertEqual(tc_track.size, 3)

        # the basin is not necessarily the genesis basin
        tc_track = tc.TCTracks.from_ibtracs_netcdf(
            year_range=(1995, 1995), basin="SP", estimate_missing=True
        )
        self.assertEqual(tc_track.size, 6)
        self.assertEqual(tc_track.data[0].basin[0], "SP")
        self.assertEqual(tc_track.data[5].basin[0], "SI")

        # genesis in NI
        tc_track = tc.TCTracks.from_ibtracs_netcdf(
            year_range=(1994, 1994), genesis_basin="NI", estimate_missing=True
        )
        self.assertEqual(tc_track.size, 5)
        for tr in tc_track.data:
            self.assertEqual(tr.basin[0], "NI")

        # genesis in EP, but crosses WP at some point
        tc_track = tc.TCTracks.from_ibtracs_netcdf(
            year_range=(2002, 2003), basin="WP", genesis_basin="EP"
        )
        self.assertEqual(tc_track.size, 3)
        for tr in tc_track.data:
            self.assertEqual(tr.basin[0], "EP")
            self.assertIn("WP", tr.basin)

    def test_cutoff_tracks(self):
        tc_track = tc.TCTracks.from_ibtracs_netcdf(storm_id="1986226N30276")
        tc_track.equal_timestep()
        with self.assertLogs("climada.hazard.tc_tracks_synth", level="DEBUG") as cm:
            tc_track.calc_perturbed_trajectories(nb_synth_tracks=10)
        self.assertIn(
            "The following generated synthetic tracks moved beyond "
            "the range of [-70, 70] degrees latitude",
            cm.output[1],
        )


class TestBase(unittest.TestCase):
    """Test methods to create Hazard objects from hdf5, matlab mat
    and raster."""

    def test_write_read_pass(self):
        """Read a hazard mat file correctly."""

        file_name = str(DATA_DIR.joinpath("test_haz.h5"))
        # Read demo matlab file
        hazard = Hazard.from_mat(HAZ_TEST_MAT)
        hazard.event_name = list(map(str, hazard.event_name))
        for todense_flag in [False, True]:
            if todense_flag:
                hazard.write_hdf5(file_name, todense=todense_flag)
            else:
                hazard.write_hdf5(file_name)

            haz_read = Hazard.from_hdf5(file_name)

            self.assertEqual(hazard.haz_type, haz_read.haz_type)
            self.assertIsInstance(haz_read.haz_type, str)
            self.assertEqual(hazard.units, haz_read.units)
            self.assertIsInstance(haz_read.units, str)
            self.assertTrue(
                np.array_equal(hazard.centroids.coord, haz_read.centroids.coord)
            )
            self.assertTrue(
                u_coord.equal_crs(hazard.centroids.crs, haz_read.centroids.crs)
            )
            self.assertTrue(np.array_equal(hazard.event_id, haz_read.event_id))
            self.assertTrue(np.array_equal(hazard.frequency, haz_read.frequency))
            self.assertEqual(hazard.frequency_unit, haz_read.frequency_unit)
            self.assertIsInstance(haz_read.frequency_unit, str)
            self.assertTrue(np.array_equal(hazard.event_name, haz_read.event_name))
            self.assertIsInstance(haz_read.event_name, list)
            self.assertIsInstance(haz_read.event_name[0], str)
            self.assertTrue(np.array_equal(hazard.date, haz_read.date))
            self.assertTrue(np.array_equal(hazard.orig, haz_read.orig))
            self.assertTrue(
                np.array_equal(hazard.intensity.toarray(), haz_read.intensity.toarray())
            )
            self.assertIsInstance(haz_read.intensity, sparse.csr_matrix)
            self.assertTrue(
                np.array_equal(hazard.fraction.toarray(), haz_read.fraction.toarray())
            )
            self.assertIsInstance(haz_read.fraction, sparse.csr_matrix)

    def test_raster_to_vector_pass(self):
        """Test raster_to_vector method"""

        haz_fl = Hazard.from_raster([HAZ_DEMO_FL], haz_type="FL")
        haz_fl.check()
        meta_orig = haz_fl.centroids.meta
        inten_orig = haz_fl.intensity
        fract_orig = haz_fl.fraction

        haz_fl.raster_to_vector()

        self.assertEqual(haz_fl.centroids.meta, dict())
        self.assertAlmostEqual(
            haz_fl.centroids.lat.min(),
            meta_orig["transform"][5]
            + meta_orig["height"] * meta_orig["transform"][4]
            - meta_orig["transform"][4] / 2,
        )
        self.assertAlmostEqual(
            haz_fl.centroids.lat.max(),
            meta_orig["transform"][5] + meta_orig["transform"][4] / 2,
        )
        self.assertAlmostEqual(
            haz_fl.centroids.lon.max(),
            meta_orig["transform"][2]
            + meta_orig["width"] * meta_orig["transform"][0]
            - meta_orig["transform"][0] / 2,
        )
        self.assertAlmostEqual(
            haz_fl.centroids.lon.min(),
            meta_orig["transform"][2] + meta_orig["transform"][0] / 2,
        )
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.crs, meta_orig["crs"]))
        self.assertTrue(np.allclose(haz_fl.intensity.data, inten_orig.data))
        self.assertTrue(np.allclose(haz_fl.fraction.data, fract_orig.data))

    def test_reproject_raster_pass(self):
        """Test reproject_raster reference."""

        haz_fl = Hazard.from_raster([HAZ_DEMO_FL])
        haz_fl.check()

        haz_fl.reproject_raster(dst_crs="epsg:2202")

        self.assertEqual(haz_fl.intensity.shape, (1, 1046408))
        self.assertIsInstance(haz_fl.intensity, sparse.csr_matrix)
        self.assertIsInstance(haz_fl.fraction, sparse.csr_matrix)
        self.assertEqual(haz_fl.fraction.shape, (1, 1046408))
        self.assertTrue(u_coord.equal_crs(haz_fl.centroids.meta["crs"], "epsg:2202"))
        self.assertEqual(haz_fl.centroids.meta["width"], 968)
        self.assertEqual(haz_fl.centroids.meta["height"], 1081)
        self.assertEqual(haz_fl.fraction.min(), 0)
        self.assertEqual(haz_fl.fraction.max(), 1)
        self.assertEqual(haz_fl.intensity.min(), -9999)
        self.assertTrue(haz_fl.intensity.max() < 4.7)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroids)
    TESTS.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStormEurope))
    TESTS.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTcTracks))
    TESTS.addTest(unittest.TestLoader().loadTestsFromTestCase(TestBase))
    unittest.TextTestRunner(verbosity=2).run(TESTS)
