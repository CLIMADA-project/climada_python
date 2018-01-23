"""
Test Exposure base class.
"""

import unittest
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.source_mat import HazardMat
from climada.util.constants import HAZ_DEMO_MAT

class TestLoader(unittest.TestCase):
    """Test loading funcions from the Hazard class"""

    @staticmethod
    def good_hazard():
        """Define well a hazard"""
        haz = Hazard()
        haz.centroids = Centroids()
        haz.centroids.region_id = np.array([1, 2])
        haz.centroids.id = np.array([1, 2])
        haz.centroids.coord = np.array([[1, 2], [1, 2]])
        haz.event_id = np.array([1, 2, 3])
        haz.event_name = ['A', 'B', 'C']
        haz.frequency = np.array([1, 2, 3])
        # events x centroids
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])
        haz.fraction = sparse.csr_matrix([[1, 2], [1, 2], [1, 2]])

        return haz

    def test_check_wrongCentroids_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.centroids.region_id = np.array([1, 2, 3, 4])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Centroids.region_id size: 2 != 4', \
                         str(error.exception))

    def test_check_wrongFreq_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.frequency = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.frequency size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongInten_fail(self):
        """Wrong hazard definition"""
        haz = self.good_hazard()
        haz.intensity = sparse.csr_matrix([[1, 2], [1, 2]])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.intensity row size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongFrac_fail(self):
        """Wrong exposures definition"""
        haz = self.good_hazard()
        haz.fraction = sparse.csr_matrix([[1], [1], [1]])

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.fraction column size: 2 != 1', \
                         str(error.exception))

    def test_check_wrongEvName_fail(self):
        """Wrong exposures definition"""
        haz = self.good_hazard()
        haz.event_name = ['M']

        with self.assertRaises(ValueError) as error:
            haz.check()
        self.assertEqual('Invalid Hazard.event_name size: 3 != 1', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """Load function not implemented"""
        haz = Hazard()
        with self.assertRaises(NotImplementedError):
            haz.load('filename')

    def test_read_notimplemented(self):
        """Read function not implemented"""
        haz = Hazard()
        with self.assertRaises(NotImplementedError):
            haz.read('filename')

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            Hazard('filename')

class TestPlotter(unittest.TestCase):
    """Test plot functions."""

    def setUp(self):
        self.hazard = HazardMat(HAZ_DEMO_MAT)
        plt.ion()

    def tearDown(self):
        plt.close('all')
    
    def test_plot_intensity(self):
        """Generate all possible plots of the intensity."""
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

    def test_plot_fraction(self):
        """Generate all possible plots of the fraction."""
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

# Execute TestAssign
suite = unittest.TestLoader().loadTestsFromTestCase(TestLoader)
suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPlotter))
unittest.TextTestRunner(verbosity=2).run(suite)
