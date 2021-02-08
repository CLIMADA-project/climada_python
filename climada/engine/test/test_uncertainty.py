"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Test uncertainty module.
"""



import unittest

from climada.entity import ImpactFunc, ImpactFuncSet
import numpy as np
from climada.entity import BlackMarble, Entity
from climada.hazard import TropCyclone
import os
from climada.engine.uncertainty import UncVar, UncImpact, UncCostBenefit
import scipy as sp
from pathos.pools import ProcessPool as Pool


CURR_DIR = "/Users/ckropf/Documents/Climada/Uncertainty"

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


def dummy_exp():
    file_name = os.path.join(CURR_DIR, "exp_AIA.h5")
    exp = BlackMarble()
    exp.read_hdf5(file_name)
    return exp

def dummy_haz(x=1):
    file_name = os.path.join(CURR_DIR, "tc_AIA.h5")
    haz= TropCyclone()
    haz.read_hdf5(file_name)
    haz.intensity = haz.intensity.multiply(x)
    return haz

HAZ_TEST_MAT = '/Users/ckropf/Documents/Climada/climada_python/climada/hazard/test/data/atl_prob_no_name.mat'
ENT_TEST_MAT = '/Users/ckropf/Documents/Climada/climada_python/climada/entity/exposures/test/data/demo_today.mat'
def dummy_ent():
    entity = Entity()
    entity.read_mat(ENT_TEST_MAT)
    entity.check()
    entity.measures._data['TC'] = entity.measures._data.pop('XX')
    for meas in entity.measures.get_measure('TC'):
        meas.haz_type = 'TC'
    entity.check()
    return entity


class TestUncVar(unittest.TestCase):

    exp = dummy_exp()
    haz = dummy_haz()
    impf = imp_fun_tc

    distr_dict = {"G": sp.stats.uniform(0.8,1),
                  "v_half": sp.stats.uniform(50, 100),
                  "vmin": sp.stats.norm(15,30),
                  "k": sp.stats.uniform(1, 5)
                  }
    impf_unc = UncVar(impf, distr_dict)

    impf_unc.plot_distr()

    unc = UncImpact(exp, impf_unc, haz)
    unc.make_sample(N=1)
    unc.calc_impact_distribution(calc_eai_exp=True)
    unc.calc_impact_sensitivity()

    unc.plot_metric_distribution(['aai_agg', 'freq_curve'])
    unc.plot_rp_distribution()

    pool = Pool()
    haz_unc = UncVar(dummy_haz, {'x': sp.stats.norm(1, 1)})
    ent = dummy_ent()
    unc = UncCostBenefit(haz_unc, ent)
    unc.make_sample(N=1)
    unc.calc_cost_benefit_distribution(pool=pool)
    unc.calc_cost_benefit_sensitivity()
    pool.close()
    pool.join()
    pool.clear()

    unc.plot_metric_distribution(list(unc.metrics.keys())[0:6])

if __name__ == "__main__":
    TESTS = unittest.TestLoader().loadTestsFromTestCase(TestUncVar)
    unittest.TextTestRunner(verbosity=2).run(TESTS)
