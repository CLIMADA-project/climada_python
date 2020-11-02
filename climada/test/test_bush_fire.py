"""
Test Bush fire class
"""

import os
import unittest
import numpy as np

from climada.hazard.bush_fire import BushFire


DATA_DIR = os.path.join(os.path.dirname(__file__), '../test/data/')
TEST_FIRMS = os.path.join(DATA_DIR, "BF_FIRMS.csv")

description = ''

class TestBushFire(unittest.TestCase):
    """Test loading functions from the BushFire class"""

    def test_hist_events_pass(self):
        """ Test set_hist_events """
        bf = BushFire()
        bf.set_hist_events(TEST_FIRMS)
        bf.check()

        self.assertEqual(bf.tag.haz_type, 'BF')
        self.assertEqual(bf.units, 'K')
        self.assertTrue(np.allclose(bf.event_id, np.arange(1, 13)))
        self.assertTrue(np.allclose(bf.date,
            np.array([736186, 736185, 736185, 736184, 736184, 736185, 736184, 736185,
               736185, 736184, 736185, 736186])))
        self.assertTrue(np.allclose(bf.orig, np.ones(12, bool)))
        self.assertEqual(bf.event_name, ['1', '2', '3', '4', '5', '6', '7', '8', '9',
            '10', '11', '12'])
        self.assertTrue(np.allclose(bf.frequency, np.ones(12)))
        self.assertEqual(bf.intensity.shape, (12, 51042))
        self.assertEqual(bf.fraction.shape, (12, 51042))
        self.assertEqual(bf.intensity[0, :].nonzero()[1][0], 42159)
        self.assertEqual(bf.intensity[3, :].nonzero()[1][5], 27737)
        self.assertEqual(bf.intensity[7, :].nonzero()[1][4], 18821)
        self.assertAlmostEqual(bf.intensity[0, 42159], 324.1)
        self.assertAlmostEqual(bf.intensity[3, 27737], 392.5)
        self.assertAlmostEqual(bf.intensity[7, 18821], 312.8)
        self.assertAlmostEqual(bf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)

    def test_hist_event_set_pass(self):
        """ Test set_hist_event_year_set """
        bf = BushFire()
        bf.set_hist_event_year_set(TEST_FIRMS)
        
        self.assertEqual(bf.tag.haz_type, 'BF')
        self.assertEqual(bf.units, 'K')
        self.assertTrue(np.allclose(bf.event_id, np.arange(1, 2)))
        self.assertTrue(np.allclose(bf.date, np.array([735964])))
        self.assertTrue(np.allclose(bf.orig, np.ones(1, bool)))
        self.assertEqual(bf.event_name, ['2016'])
        self.assertTrue(np.allclose(bf.frequency, np.ones(1)))
        self.assertEqual(bf.intensity.shape, (1, 51042))
        self.assertEqual(bf.fraction.shape, (1, 51042))
        self.assertEqual(bf.intensity[0, :].nonzero()[1][0], 47)
        self.assertEqual(bf.intensity[0, :].nonzero()[1][20], 1296)
        self.assertEqual(bf.intensity[0, :].nonzero()[1][100], 4005)
        self.assertAlmostEqual(bf.intensity[0, 47], 305.0)
        self.assertAlmostEqual(bf.intensity[0, 1296], 342.5)
        self.assertAlmostEqual(bf.intensity[0, 4005], 318.5)
        self.assertAlmostEqual(bf.intensity[0, 8000], 0.0)
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)

    def test_proba_event_set_pass(self):
        """ Test probabilistic set_probabilistic_event_year_set """
        bf = BushFire()
        bf.set_hist_event_year_set(TEST_FIRMS)
        bf.set_probabilistic_event_year_set(1,[3,4])

        self.assertEqual(bf.size, 2)
        orig = np.zeros(2, bool)
        orig[0] = True
        self.assertTrue(np.allclose(bf.orig, orig))
        self.assertEqual(bf.event_name, ['2016', '2'])
        self.assertEqual(bf.intensity.shape, (2, 51042))
        self.assertEqual(bf.fraction.shape, (2, 51042))
        self.assertEqual(bf.intensity[1, :].nonzero()[1][0], 5080)
        self.assertEqual(bf.intensity[1, :].nonzero()[1][4], 32991)
        self.assertEqual(bf.intensity[0, :].nonzero()[1][11], 939)       
        self.assertAlmostEqual(bf.intensity[1, 5080], 334.3)
        self.assertAlmostEqual(bf.intensity[1, 32991], 309.9)
        self.assertAlmostEqual(bf.intensity[0, 939], 356.0)
        self.assertAlmostEqual(bf.fraction.max(), 1.0)
        self.assertAlmostEqual(bf.fraction.min(), 0.0)


# Execute Tests
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestBushFire)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
