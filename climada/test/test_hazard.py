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
from scipy import sparse

from climada import CONFIG
from climada.hazard import tc_tracks as tc
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.hazard.storm_europe import StormEurope
from climada.util.constants import (HAZ_DEMO_FL, WS_DEMO_NC)
from climada.util.api_client import Client

DATA_DIR = CONFIG.test_data.dir()

class TestCentroids(unittest.TestCase):
    """Test centroids functionalities"""

    def test_read_write_raster_pass(self):
        """Test write_raster: Hazard from raster data"""
        haz_fl = Hazard.from_raster([HAZ_DEMO_FL])
        haz_fl.tag.haz_type = 'FL'
        haz_fl.check()

        self.assertEqual(haz_fl.intensity.shape, (1, 1032226))
        self.assertEqual(haz_fl.intensity.min(), -9999.0)
        self.assertAlmostEqual(haz_fl.intensity.max(), 4.662774085998535)

        haz_fl.write_raster(DATA_DIR.joinpath('test_write_hazard.tif'))

        haz_read = Hazard.from_raster([DATA_DIR.joinpath('test_write_hazard.tif')])
        haz_fl.tag.haz_type = 'FL'
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
    """ Test methods to create StormEurope object """

    def test_from_footprints(self):
        """Test from_footprints constructor, using one small test files"""
        def _test_first(haz):
            """Test the expected first entry of the hazard"""
            self.assertEqual(haz.tag.haz_type, 'WS')
            self.assertEqual(haz.units, 'm/s')
            self.assertEqual(haz.event_id.size, 1)
            self.assertEqual(haz.date.size, 1)
            self.assertEqual(dt.datetime.fromordinal(haz.date[0]).year, 1999)
            self.assertEqual(dt.datetime.fromordinal(haz.date[0]).month, 12)
            self.assertEqual(dt.datetime.fromordinal(haz.date[0]).day, 26)
            self.assertEqual(haz.event_id[0], 1)
            self.assertEqual(haz.event_name[0], 'Lothar')
            self.assertIsInstance(haz.intensity,
                                  sparse.csr.csr_matrix)
            self.assertIsInstance(haz.fraction,
                                  sparse.csr.csr_matrix)
            self.assertEqual(haz.intensity.shape, (1, 9944))
            self.assertEqual(haz.fraction.shape, (1, 9944))
            self.assertEqual(haz.frequency[0], 1.0)

        # Load first entry
        storms = StormEurope.from_footprints(
            WS_DEMO_NC[0], description='test_description')
        _test_first(storms)

        # Omit the second file, should be the same result
        storms = StormEurope.from_footprints(WS_DEMO_NC, files_omit=str(WS_DEMO_NC[1]))
        _test_first(storms)

        # Now load both
        storms = StormEurope.from_footprints(WS_DEMO_NC, description='test_description')

        self.assertEqual(storms.tag.haz_type, 'WS')
        self.assertEqual(storms.units, 'm/s')
        self.assertEqual(storms.event_id.size, 2)
        self.assertEqual(storms.date.size, 2)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).year, 1999)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).month, 12)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).day, 26)
        self.assertEqual(storms.event_id[0], 1)
        self.assertEqual(storms.event_name[0], 'Lothar')
        self.assertIsInstance(storms.intensity,
                              sparse.csr.csr_matrix)
        self.assertIsInstance(storms.fraction,
                              sparse.csr.csr_matrix)
        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(storms.fraction.shape, (2, 9944))

    def test_icon_read(self):
        """test reading from icon grib"""
        # for this test the forecast file is supposed to be already downloaded from the dwd
        # another download would fail because the files are available for 24h only
        # instead, we download it as a test dataset through the climada data api
        apiclient = Client()
        ds = apiclient.get_dataset_info(name='test_storm_europe_icon_2021012800', status='test_dataset')
        dsdir, _ = apiclient.download_dataset(ds)
        haz = StormEurope.from_icon_grib(
            dt.datetime(2021, 1, 28),
            dt.datetime(2021, 1, 28),
            model_name='test',
            grib_dir=dsdir,
            delete_raw_data=False)
        self.assertEqual(haz.tag.haz_type, 'WS')
        self.assertEqual(haz.units, 'm/s')
        self.assertEqual(haz.event_id.size, 40)
        self.assertEqual(haz.date.size, 40)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).year, 2021)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).month, 1)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).day, 28)
        self.assertEqual(haz.event_id[-1], 40)
        self.assertEqual(haz.event_name[-1], '2021-01-28_ens40')
        self.assertIsInstance(haz.intensity,
                              sparse.csr.csr_matrix)
        self.assertIsInstance(haz.fraction,
                              sparse.csr.csr_matrix)
        self.assertEqual(haz.intensity.shape, (40, 49))
        self.assertAlmostEqual(haz.intensity.max(), 17.276321,places=3)
        self.assertEqual(haz.fraction.shape, (40, 49))
        with self.assertLogs('climada.hazard.storm_europe', level='WARNING') as cm:
            with self.assertRaises(ValueError):
                haz = StormEurope.from_icon_grib(
                    dt.datetime(2021, 1, 28, 6),
                    dt.datetime(2021, 1, 28),
                    model_name='test',
                    grib_dir=CONFIG.hazard.test_data.str(),
                    delete_raw_data=False)
        self.assertEqual(len(cm.output), 1)
        self.assertIn('event definition is inaccuratly implemented', cm.output[0])

class TestTcTracks(unittest.TestCase):
    """ Test methods to create TcTracks objects from netcdf"""

    def test_ibtracs_with_basin(self):
        """Filter TCs by (genesis) basin."""
        # South Atlantic (not usually a TC location at all)
        tc_track = tc.TCTracks.from_ibtracs_netcdf(basin="SA")
        self.assertEqual(tc_track.size, 3)

        # the basin is not necessarily the genesis basin
        tc_track = tc.TCTracks.from_ibtracs_netcdf(
            year_range=(1995, 1995), basin="SP", estimate_missing=True)
        self.assertEqual(tc_track.size, 6)
        self.assertEqual(tc_track.data[0].basin[0], 'SP')
        self.assertEqual(tc_track.data[5].basin[0], 'SI')

        # genesis in NI
        tc_track = tc.TCTracks.from_ibtracs_netcdf(
            year_range=(1994, 1994), genesis_basin="NI", estimate_missing=True)
        self.assertEqual(tc_track.size, 5)
        for tr in tc_track.data:
            self.assertEqual(tr.basin[0], "NI")

        # genesis in EP, but crosses WP at some point
        tc_track = tc.TCTracks.from_ibtracs_netcdf(
            year_range=(2002, 2003), basin="WP", genesis_basin="EP")
        self.assertEqual(tc_track.size, 3)
        for tr in tc_track.data:
            self.assertEqual(tr.basin[0], "EP")
            self.assertIn("WP", tr.basin)
        
    def test_cutoff_tracks(self):
        tc_track = tc.TCTracks.from_ibtracs_netcdf(storm_id='1986226N30276')
        tc_track.equal_timestep()
        with self.assertLogs('climada.hazard.tc_tracks_synth', level='DEBUG') as cm:
            tc_track.calc_perturbed_trajectories(nb_synth_tracks=10)
        self.assertIn('The following generated synthetic tracks moved beyond '
                      'the range of [-70, 70] degrees latitude', cm.output[1])

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestCentroids)
    TESTS.addTest(unittest.TestLoader().loadTestsFromTestCase(TestStormEurope))
    TESTS.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTcTracks))
    unittest.TextTestRunner(verbosity=2).run(TESTS)