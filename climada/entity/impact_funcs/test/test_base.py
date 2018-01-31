"""
Test ImpactFuncs class.
"""

import unittest
import numpy as np

from climada.entity.impact_funcs.base import ImpactFuncs, ImpactFunc
from climada.util.constants import ENT_DEMO_XLS

class TestLoader(unittest.TestCase):
    """Test loading funcions from the ImpactFuncs class"""

    def test_check_wrongPAA_fail(self):
        """Wrong PAA definition"""
        imp_fun = ImpactFuncs()
        imp_id = 1
        haz_type = 'TC'
        imp_fun.data[haz_type] = {imp_id:ImpactFunc()}
        imp_fun.data[haz_type][imp_id].id = imp_id
        imp_fun.data[haz_type][imp_id].haz_type = haz_type
        imp_fun.data[haz_type][imp_id].intensity = np.array([1, 2, 3])
        imp_fun.data[haz_type][imp_id].mdd = np.array([1, 2, 3])
        imp_fun.data[haz_type][imp_id].paa = np.array([1, 2])

        with self.assertRaises(ValueError) as error:
            imp_fun.check()
        self.assertEqual('Invalid ImpactFunc.paa size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongMDD_fail(self):
        """Wrong MDD definition"""
        imp_fun = ImpactFuncs()
        imp_id = 1
        haz_type = 'TC'
        imp_fun.data[haz_type] = {imp_id:ImpactFunc()}
        imp_fun.data[haz_type][imp_id].id = imp_id
        imp_fun.data[haz_type][imp_id].haz_type = haz_type
        imp_fun.data[haz_type][imp_id].intensity = np.array([1, 2, 3])
        imp_fun.data[haz_type][imp_id].mdd = np.array([1, 2])
        imp_fun.data[haz_type][imp_id].paa = np.array([1, 2, 3])

        with self.assertRaises(ValueError) as error:
            imp_fun.check()
        self.assertEqual('Invalid ImpactFunc.mdd size: 3 != 2', \
                         str(error.exception))

    def test_check_wrongID_fail(self):
        """Wrong id definition"""
        imp_fun = ImpactFuncs()
        imp_id = 1
        haz_type = 'TC'
        imp_fun.data[haz_type] = {imp_id:ImpactFunc()}
        imp_fun.data[haz_type][imp_id].id = 0
        imp_fun.data[haz_type][imp_id].haz_type = haz_type
        with self.assertRaises(ValueError) as error:
            imp_fun.check()
        self.assertEqual('Wrong ImpactFunc.id: 1 != 0', \
                         str(error.exception))

    def test_check_wrongType_fail(self):
        """Wrong hazard type definition"""
        imp_fun = ImpactFuncs()
        imp_id = 1
        haz_type = 'TC'
        imp_fun.data[haz_type] = {imp_id:ImpactFunc()}
        imp_fun.data[haz_type][imp_id].id = imp_id
        imp_fun.data[haz_type][imp_id].haz_type = 'null'
        with self.assertRaises(ValueError) as error:
            imp_fun.check()
        self.assertEqual('Wrong ImpactFunc.haz_type: TC != null', \
                         str(error.exception))

    def test_load_notimplemented(self):
        """"Load function not implemented"""
        imp_fun = ImpactFuncs()
        with self.assertRaises(NotImplementedError):
            imp_fun.load(ENT_DEMO_XLS)

    def test_read_notimplemented(self):
        """Read function not implemented"""
        imp_fun = ImpactFuncs()
        with self.assertRaises(NotImplementedError):
            imp_fun.read(ENT_DEMO_XLS)

    def test_constructfile_notimplemented(self):
        """Constructor from file not implemented"""
        with self.assertRaises(NotImplementedError):
            ImpactFuncs(ENT_DEMO_XLS)

class TestInterpolation(unittest.TestCase):
    """Impact function interpolation test"""

    def test_wrongAttribute_fail(self):
        """Interpolation of wrong variable fails."""
        imp_fun = ImpactFunc()
        intensity = 3
        with self.assertRaises(ValueError) as error:
            imp_fun.interpolate(intensity, 'mdg')
        self.assertEqual('Attribute of the impact function mdg not found.',\
                         str(error.exception))

    def test_mdd_pass(self):
        """Interpolation of wrong variable fails."""
        imp_fun = ImpactFunc()
        imp_fun.intensity = np.array([0,1])
        imp_fun.mdd = np.array([1,2])
        imp_fun.paa = np.array([3,4])
        intensity = 0.5
        resul = imp_fun.interpolate(intensity, 'mdd')
        self.assertEqual(1.5, resul)

    def test_paa_pass(self):
        """Interpolation of wrong variable fails."""
        imp_fun = ImpactFunc()
        imp_fun.intensity = np.array([0,1])
        imp_fun.mdd = np.array([1,2])
        imp_fun.paa = np.array([3,4])
        intensity = 0.5
        resul = imp_fun.interpolate(intensity, 'paa')
        self.assertEqual(3.5, resul)        

if __name__ == '__main__':
    unittest.main()
