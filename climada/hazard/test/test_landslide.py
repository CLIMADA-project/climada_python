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

Unit test landslide module.
"""
import unittest
import geopandas as gpd
import numpy as np
import shapely
from scipy import sparse

from climada import CONFIG
from climada.hazard.landslide import Landslide, sample_events_from_probs
import climada.util.coordinates as u_coord

DATA_DIR = CONFIG.hazard.test_data.dir()
LS_HIST_FILE = DATA_DIR / 'test_ls_hist.shp'
LS_PROB_FILE = DATA_DIR / 'test_ls_prob.tif'

class TestLandslideModule(unittest.TestCase):

    def test_set_ls_hist(self):
        """ Test function set_ls_hist()"""
        LS_hist = Landslide()
        LS_hist.set_ls_hist(bbox=(20,40,23,46),
                                  input_gdf=LS_HIST_FILE)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.tag.haz_type, 'LS')
        self.assertEqual(np.unique(LS_hist.intensity.data),np.array([1]))
        self.assertEqual(np.unique(LS_hist.fraction.data),np.array([1]))
        self.assertTrue((LS_hist.frequency.data<=1).all())

        input_gdf = gpd.read_file(LS_HIST_FILE)
        LS_hist = Landslide()
        LS_hist.set_ls_hist(bbox=(20,40,23,46),
                                  input_gdf=input_gdf)
        self.assertEqual(LS_hist.size, 272)
        self.assertEqual(LS_hist.tag.haz_type, 'LS')
        self.assertEqual(np.unique(LS_hist.intensity.data),np.array([1]))
        self.assertEqual(np.unique(LS_hist.fraction.data),np.array([1]))
        self.assertTrue((LS_hist.frequency.data<=1).all())

    def test_set_ls_prob(self):
        """ Test the function set_ls_prob()"""
        LS_prob = Landslide()
        n_years=1000
        LS_prob.set_ls_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE, n_years=n_years,
                            dist='binom')

        self.assertEqual(LS_prob.tag.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(1, 43200))
        self.assertEqual(LS_prob.fraction.shape,(1, 43200))
        self.assertTrue(max(LS_prob.intensity.data)<=1)
        self.assertEqual(min(LS_prob.intensity.data),0)
        self.assertTrue(max(LS_prob.fraction.data)<=n_years)
        self.assertEqual(min(LS_prob.fraction.data),0)
        self.assertEqual(LS_prob.frequency.shape, (1, 43200))
        self.assertEqual(min(LS_prob.frequency.data),0)
        self.assertTrue(max(LS_prob.frequency.data)<=1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

        LS_prob = Landslide()
        n_years=300
        LS_prob.set_ls_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE,
                            dist='poisson', n_years=n_years)
        self.assertEqual(LS_prob.tag.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(1, 43200))
        self.assertEqual(LS_prob.fraction.shape,(1, 43200))
        self.assertTrue(max(LS_prob.intensity.data)<=1)
        self.assertEqual(min(LS_prob.intensity.data),0)
        self.assertTrue(max(LS_prob.fraction.data)<=n_years)
        self.assertEqual(min(LS_prob.fraction.data),0)
        self.assertEqual(LS_prob.frequency.shape, (1, 43200))
        self.assertEqual(min(LS_prob.frequency.data),0)
        self.assertTrue(max(LS_prob.frequency.data)<=1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

        LS_prob = Landslide()
        corr_fact = 1.8*10e6
        LS_prob.set_ls_prob(bbox=(8,45,11,46),
                            path_sourcefile=LS_PROB_FILE,
                            dist='poisson', corr_fact=corr_fact)
        self.assertEqual(LS_prob.tag.haz_type, 'LS')
        self.assertEqual(LS_prob.intensity.shape,(1, 43200))
        self.assertEqual(LS_prob.fraction.shape,(1, 43200))
        self.assertTrue(max(LS_prob.intensity.data)<=1)
        self.assertEqual(min(LS_prob.intensity.data),0)
        self.assertTrue(max(LS_prob.fraction.data)<=n_years)
        self.assertEqual(min(LS_prob.fraction.data),0)
        self.assertEqual(LS_prob.frequency.shape, (1, 43200))
        self.assertEqual(min(LS_prob.frequency.data),0)
        self.assertTrue(max(LS_prob.frequency.data)<=1/n_years)
        self.assertEqual(LS_prob.centroids.crs.to_epsg(), 4326)
        self.assertTrue(LS_prob.centroids.coord.max() <= 46)
        self.assertTrue(LS_prob.centroids.coord.min() >= 8)

    def test_sample_events_from_probs(self):
        bbox = (8,45,11,46)
        corr_fact = 10e6
        n_years = 400
        __, prob_matrix = u_coord.read_raster(
            LS_PROB_FILE, geometry=[shapely.geometry.box(*bbox, ccw=True)])
        prob_matrix = sparse.csr_matrix(prob_matrix/corr_fact)

        ev_matrix = sample_events_from_probs(prob_matrix, n_years,
                                             dist='binom')
        self.assertTrue(max(ev_matrix.data) <= n_years)
        self.assertEqual(min(ev_matrix.data), 0)

        ev_matrix = sample_events_from_probs(prob_matrix/corr_fact, n_years,
                                             dist='poisson')
        self.assertTrue(max(ev_matrix.data) <= n_years)
        self.assertEqual(min(ev_matrix.data), 0)


if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestLandslideModule)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
