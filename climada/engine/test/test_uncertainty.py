#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:16:25 2020

@author: ckropf
"""
import unittest

from climada.entity import ImpactFunc, ImpactFuncSet
import numpy as np

class TestUncVar(unittest.TestCase):
    pass

def xhi(v, v_half, vmin):
    """
    impact function parameter (c.f. (Knutson 2011))

    Parameters
    ----------
    v : float
        intensity (wind speed)
    v_half : float
        intensity at half curve.
    vmin : float
        minimum intensity

    Returns
    -------
    float
        impact function xhi parameter 

    """
    
    return max([(v - vmin), 0]) / (v_half - vmin)
  
def imp_fun_param(v, G, v_half, vmin, k):
    """
    impact function formula from (Knutson 2011)

    Parameters
    ----------
    v : float
        intensity (wind speed)
    G : float
        Max impact. 
    v_half : float
        intensity at half curve.
    vmin : float
        minimum intensity
    k : float
        curve exponent (slope).

    Returns
    -------
    float
        impact value at given intensity v

    """
    
    return G * xhi(v, v_half, vmin)**k / (1 + xhi(v, v_half, vmin)**k)

    
def imp_fun_tc(G=1, v_half=84.7, vmin=25.7, k=3, _id=1):
    """
    Parametrized impact function from (Knutson 2011)

    Parameters
    ----------
    G : float, optional
        Max impact. The default is 1.
    v_half : float, optional
        intensity at half curve. The default is 84.7.
    vmin : float, optional
        minimum intensity. The default is 25.7.
    k : float, optional
        curve exponent (slope). The default is 3.
    _id : int, optional
        impact function id. The default is 1.

    Returns
    -------
    imp_fun : climada.ImpactFunc
        Impact function with given parameters

    """
    
    imp_fun = ImpactFunc()
    imp_fun.haz_type = 'TC'
    imp_fun.id = _id
    imp_fun.intensity_unit = 'm/s'
    imp_fun.intensity = np.linspace(0, 150, num=100)
    imp_fun.mdd = np.repeat(1, len(imp_fun.intensity))
    imp_fun.paa = np.array([imp_fun_param(v, G, v_half, vmin, k) for v in imp_fun.intensity])
    imp_fun.check()
    impf_set = ImpactFuncSet()
    impf_set.append(imp_fun)
    return impf_set
    
if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUncVar)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
    