"""
test plots
"""
import unittest
import matplotlib.pyplot as plt

from climada.hazard.source_mat import HazardMat
from climada.util.constants import HAZ_DEMO_MAT

class TestPlotter(unittest.TestCase):
    """Test plot functions."""

    def setUp(self):
        self.hazard = HazardMat(HAZ_DEMO_MAT)
        plt.ion()

    def tearDown(self):
        plt.close('all')

    def test_hazard_intensity(self):
        """Generate all possible plots of the hazard intensity."""
        myfig = self.hazard.plot_intensity(event_id=36)
        self.assertIn('Event 36: NNN_1185106_gen5', myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_intensity(event_id=-1)
        self.assertIn('1-largest Event 3899: NNN_1190604_gen8', \
                      myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_intensity(event_id=-4)
        self.assertIn('4-largest Event 5489: NNN_1192804_gen8', \
                      myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_intensity(event_id=0)
        self.assertIn('TC max intensity at each point', \
                      myfig._suptitle.get_text())
        plt.close(myfig)

        myfig = self.hazard.plot_intensity(centr_id=59)
        self.assertIn('Centroid 59: (29.0, -79.0)', myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_intensity(centr_id=-1)
        self.assertIn('1-largest Centroid 100: (30.0, -75.0)', \
                      myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_intensity(centr_id=-4)
        self.assertIn('4-largest Centroid 70: (30.0, -78.0)', \
                      myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_intensity(centr_id=0)
        self.assertIn('TC max intensity at each event', \
                      myfig._suptitle.get_text())
        plt.close(myfig)

    def test_hazard_fraction(self):
        """Generate all possible plots of the hazard fraction."""
        myfig = self.hazard.plot_fraction(event_id=36)
        self.assertIn('Event 36: NNN_1185106_gen5', myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_fraction(event_id=-1)
        self.assertIn('1-largest Event 11898: GORDON_gen7', \
                      myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_fraction(centr_id=59)
        self.assertIn('Centroid 59: (29.0, -79.0)', myfig._suptitle.get_text())
        plt.close(myfig)
        myfig = self.hazard.plot_fraction(centr_id=-1)
        self.assertIn('Centroid 80: (30.0, -77.0)', myfig._suptitle.get_text())
        plt.close(myfig)

# Execute TestPlotter
suite = unittest.TestLoader().loadTestsFromTestCase(TestPlotter)
unittest.TextTestRunner(verbosity=2).run(suite)
