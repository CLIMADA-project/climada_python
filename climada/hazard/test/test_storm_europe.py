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

Test StormEurope class
"""

import copy
import unittest
import datetime as dt
import numpy as np
from scipy import sparse

from climada import CONFIG
from climada.hazard.storm_europe import StormEurope, generate_WS_forecast_hazard
from climada.hazard.centroids.centr import DEF_VAR_EXCEL, Centroids
from climada.util.constants import WS_DEMO_NC
from climada.util.api_client import Client


DATA_DIR = CONFIG.hazard.test_data.dir()

class TestReader(unittest.TestCase):
    """Test loading functions from the StormEurope class"""

    def test_centroids_from_nc(self):
        """Test if centroids can be constructed correctly"""
        cent = StormEurope._centroids_from_nc(WS_DEMO_NC[0])

        self.assertTrue(isinstance(cent, Centroids))
        self.assertEqual(cent.size, 9944)

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

    def test_read_with_ref(self):
        """Test from_footprints while passing in a reference raster."""
        storms = StormEurope.from_footprints(WS_DEMO_NC, ref_raster=WS_DEMO_NC[1])

        self.assertEqual(storms.tag.haz_type, 'WS')
        self.assertEqual(storms.units, 'm/s')
        self.assertEqual(storms.event_id.size, 2)
        self.assertEqual(storms.date.size, 2)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).year, 1999)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).month, 12)
        self.assertEqual(dt.datetime.fromordinal(storms.date[0]).day, 26)
        self.assertEqual(storms.event_id[0], 1)
        self.assertEqual(storms.event_name[0], 'Lothar')
        self.assertTrue(isinstance(storms.intensity, sparse.csr.csr_matrix))
        self.assertTrue(isinstance(storms.fraction, sparse.csr.csr_matrix))
        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(storms.fraction.shape, (2, 9944))

    def test_read_with_cent(self):
        """Test from_footprints while passing in a Centroids object"""
        var_names = copy.deepcopy(DEF_VAR_EXCEL)
        var_names['sheet_name'] = 'fp_centroids-test'
        var_names['col_name']['region_id'] = 'iso_n3'
        test_centroids = Centroids.from_excel(
            DATA_DIR.joinpath('fp_centroids-test.xls'),
            var_names=var_names
            )
        storms = StormEurope.from_footprints(WS_DEMO_NC, centroids=test_centroids)

        self.assertEqual(storms.intensity.shape, (2, 9944))
        self.assertEqual(
            np.count_nonzero(
                ~np.isnan(storms.centroids.region_id)
            ),
            6401
        )

    def test_set_ssi(self):
        """Test set_ssi with both dawkins and wisc_gust methodology."""
        storms = StormEurope.from_footprints(WS_DEMO_NC)

        storms.set_ssi(method='dawkins')
        ssi_dawg = np.asarray([1.44573572e+09, 6.16173724e+08])
        self.assertTrue(
            np.allclose(storms.ssi, ssi_dawg)
        )

        storms.set_ssi(method='wisc_gust')
        ssi_gusty = np.asarray([1.42124571e+09, 5.86870673e+08])
        self.assertTrue(
            np.allclose(storms.ssi, ssi_gusty)
        )

        storms.set_ssi(threshold=20, on_land=False)
        ssi_special = np.asarray([2.96582030e+09, 1.23980294e+09])
        self.assertTrue(
            np.allclose(storms.ssi, ssi_special)
        )

    def test_generate_prob_storms(self):
        """Test the probabilistic storm generator; calls _hist2prob as well as
        Centroids.set_region_id()"""
        storms = StormEurope.from_footprints(WS_DEMO_NC)
        storms_prob = storms.generate_prob_storms()

        self.assertEqual(
            np.count_nonzero(storms.centroids.region_id),
            6402
            # here, we don't rasterise; we check if the centroids lie in a
            # polygon. that is to say, it's not the majority of a raster pixel,
            # but the centroid's location that is decisive
        )
        self.assertEqual(storms_prob.size, 60)
        self.assertTrue(np.allclose((1 / storms_prob.frequency).astype(int), 330))
        self.assertAlmostEqual(storms.frequency.sum(),
                               storms_prob.frequency.sum())
        self.assertEqual(np.count_nonzero(storms_prob.orig), 2)
        self.assertEqual(storms_prob.centroids.size, 3054)
        self.assertIsInstance(storms_prob.intensity,
                              sparse.csr.csr_matrix)

    def test_cosmoe_read(self):
        """test reading from cosmo-e netcdf"""
        haz = StormEurope.from_cosmoe_file(
            DATA_DIR.joinpath('storm_europe_cosmoe_forecast_vmax_testfile.nc'),
            run_datetime=dt.datetime(2018,1,1),
            event_date=dt.datetime(2018,1,3))
        self.assertEqual(haz.tag.haz_type, 'WS')
        self.assertEqual(haz.units, 'm/s')
        self.assertEqual(haz.event_id.size, 21)
        self.assertEqual(haz.date.size, 21)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).year, 2018)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).month, 1)
        self.assertEqual(dt.datetime.fromordinal(haz.date[0]).day, 3)
        self.assertEqual(haz.event_id[-1], 21)
        self.assertEqual(haz.event_name[-1], '2018-01-03_ens21')
        self.assertIsInstance(haz.intensity,
                              sparse.csr.csr_matrix)
        self.assertIsInstance(haz.fraction,
                              sparse.csr.csr_matrix)
        self.assertEqual(haz.intensity.shape, (21, 25))
        self.assertAlmostEqual(haz.intensity.max(), 36.426735,places=3)
        self.assertEqual(haz.fraction.shape, (21, 25))

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

    def test_generate_forecast(self):
        """ testing generating a forecast """
        hazard, haz_model, run_datetime, event_date = generate_WS_forecast_hazard(
            run_datetime=dt.datetime(2018,1,1),
            event_date=dt.datetime(2018,1,3),
            haz_model='cosmo2e_file',
            haz_raw_storage=DATA_DIR.joinpath('storm_europe_cosmoe_forecast' +
                                              '_vmax_testfile.nc'),
            save_haz=False,
            )
        self.assertEqual(run_datetime.year, 2018)
        self.assertEqual(run_datetime.month, 1)
        self.assertEqual(run_datetime.day, 1)
        self.assertEqual(event_date.day, 3)
        self.assertEqual(hazard.event_name[-1], '2018-01-03_ens21')
        self.assertEqual(haz_model, 'C2E')

# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
