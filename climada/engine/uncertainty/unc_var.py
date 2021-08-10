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

Define UncVar class.
"""

import copy
from functools import partial
from itertools import zip_longest

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from climada.entity import Entity, DiscRates

__all__ = ['UncVar']

FIG_W, FIG_H = 8, 5 #default figize width/heigh column/work multiplicators

class UncVar():
    """
    Uncertainty variable

    An uncertainty variable requires a single or multi-parameter function.
    The parameters must follow a given distribution. The uncertainty
    variables are the input parameters of the model.

    Attributes
    ----------
    distr_dict : dict
        Distribution of the uncertainty parameters. Keys are uncertainty
        parameters names and Values are probability density distribution
        from scipy.stats package
        https://docs.scipy.org/doc/scipy/reference/stats.html
    labels : list
        Names of the uncertainty parameters (keys of distr_dict)
    uncvar_func : function
        User defined python fucntion with the uncertainty parameters
        as input kwargs and which returns a climada object.

    Notes
    -----
    A few default Variables are defined for Hazards, Exposures and
    Impact Fucntions.


    Examples
    --------

    Categorical variable function: LitPop exposures with m,n exponents in [0,5]
        import scipy as sp
        def litpop_cat(m, n):
            exp = Litpop()
            exp.set_country('CHE', exponent=[m, n])
            return exp
        distr_dict = {
            'm': sp.stats.randint(low=0, high=5),
            'n': sp.stats.randint(low=0, high=5)
            }
        unc_var_cat = UncVar(uncvar_func=litpop_cat, distr_dict=distr_dict)

    Continuous variable function: Impact function for TC
        import scipy as sp
        def imp_fun_tc(G, v_half, vmin, k, _id=1):
            imp_fun = ImpactFunc()
            imp_fun.haz_type = 'TC'
            imp_fun.id = _id
            imp_fun.intensity_unit = 'm/s'
            imp_fun.intensity = np.linspace(0, 150, num=100)
            imp_fun.mdd = np.repeat(1, len(imp_fun.intensity))
            imp_fun.paa = np.array([sigmoid_function(v, G, v_half, vmin, k)
                                    for v in imp_fun.intensity])
            imp_fun.check()
            impf_set = ImpactFuncSet()
            impf_set.append(imp_fun)
            return impf_set
        distr_dict = {"G": sp.stats.uniform(0.8, 1),
              "v_half": sp.stats.uniform(50, 100),
              "vmin": sp.stats.norm(loc=15, scale=30),
              "k": sp.stats.randint(low=1, high=9)
              }
        unc_var_cont = UncVar(uncvar_func=imp_fun_tc, distr_dict=distr_dict)

    """

    def __init__(self, uncvar_func, distr_dict):
        """
        Initialize UncVar

        Parameters
        ----------
        uncvar_func : function
            Variable defined as a function of the uncertainty parameters
        distr_dict : dict
            Dictionary of the probability density distributions of the
            uncertainty parameters, with keys matching the keyword
            arguments (i.e. uncertainty parameters) of the uncvar_func
            function.
            The distribution must be of type scipy.stats
            https://docs.scipy.org/doc/scipy/reference/stats.html

        """
        self.labels = list(distr_dict.keys())
        self.distr_dict = distr_dict
        self.uncvar_func = uncvar_func

    def evaluate(self, **params):
        """
        Return the value of uncertainty cariable.

        By default, the value of the average is returned.

        Parameters
        ----------
        **params : all input parameters from self.unc_var
            Params will be passed to self.unc_func.

        Returns
        -------
        unc_func(**params) : climada object
            Output of the uncertainty variable.

        """

        if not params:
            params = {
                param: distr.mean()
                for param, distr in self.distr_dict.items()
                }
        return self.uncvar_func(**params)


    def plot(self, figsize=None):
        """
        Plot the distributions of the parameters of the uncertainty variable.

        Parameters
        ----------
        figsize: tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:
                nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
                figsize = (ncols * FIG_W, nrows * FIG_H)

        Returns
        -------
        axes: matplotlib.pyplot.figure, matplotlib.pyplot.axes
            The figure and axes handle of the plot.

        """

        nplots = len(self.distr_dict)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        if figsize is None:
            figsize = (ncols * FIG_W, nrows * FIG_H)
        _fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])
        for ax, name_distr in zip_longest(flat_axes,
                                    self.distr_dict.items(),
                                    fillvalue=None):
            if name_distr is None:
                ax.remove()
                continue
            (param_name, distr) = name_distr
            x = np.linspace(distr.ppf(1e-10), distr.ppf(1-1e-10), 100)
            ax.plot(x, distr.pdf(x), label=param_name)
            ax.legend()
        return axes


    @staticmethod
    def var_to_uncvar(var):
        """
        Returns an uncertainty variable with no distribution if var is not
        an UncVar. Else, returns var.

        Parameters
        ----------
        var : climada.uncertainty.UncVar or any other CLIMADA object

        Returns
        -------
        UncVar
            var if var is UncVar, else UncVar with var and no distribution.

        """

        if isinstance(var, UncVar):
            return var

        return UncVar(uncvar_func=lambda: var, distr_dict={})

    def haz_unc(haz, bounds_ev=None, bounds_int=None, bounds_freq=None):
        """
        Default hazard uncertainty variable

        Three types of uncertainty can be added:
            1- sub-sampling events from the total event set
            2- scale the intensity of all events (homogeneously)
            3- scale the frequency of all events (homogeneously)

        Parameters
        ----------
        haz : climada.hazard
            The base hazard
        bounds_ev : TYPE, optional
            DESCRIPTION. The default is None.
        bounds_int : TYPE, optional
            DESCRIPTION. The default is None.
        bounds_freq : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        unc_var
            Uncertainty variable for a hazard object.

        """
        kwargs = {'haz': haz}
        if bounds_ev is None:
            kwargs['HE'] = None
        if bounds_int is None:
            kwargs['HI'] = None
        if bounds_freq is None:
            kwargs['HF'] = None
        return UncVar(
            partial(_haz_uncfunc, **kwargs),
            _haz_unc_dict(bounds_ev, bounds_int, bounds_freq)
            )

    def exp_unc(exp, bounds_totval=None, bounds_noise=None):
        """


        Parameters
        ----------
        exp : TYPE
            DESCRIPTION.
        bounds_totval : TYPE, optional
            DESCRIPTION. The default is None.
        bounds_noise : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        kwargs = {'exp': exp, 'bounds_noise': bounds_noise}
        if bounds_noise is None:
            kwargs['EN'] = None
        if bounds_totval is None:
            kwargs['ET'] = None
        return UncVar(
            partial(_exp_uncfunc, **kwargs),
            _exp_unc_dict(bounds_totval, bounds_noise)
            )

    def impfset_unc(impf_set, bounds_impf=None, haz_type='TC', fun_id=1):
        """


        Parameters
        ----------
        impf_set : TYPE
            DESCRIPTION.
        bounds_impf : TYPE, optional
            DESCRIPTION. The default is None.
        haz_type : TYPE, optional
            DESCRIPTION. The default is 'TC'.
        fun_id : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        kwargs = {}
        if bounds_impf is None:
            kwargs['IF'] = None
        return UncVar(
            partial(_impfset_uncfunc, impf_set=impf_set, haz_type=haz_type, fun_id=fun_id, **kwargs),
            _impfset_unc_dict(bounds_impf)
        )

    def ent_unc(bounds_disk, bounds_cost, bounds_totval, bounds_noise,
                bounds_impf, impf_set, disc_rate,
                exp, meas_set):
        """


        Parameters
        ----------
        bounds_disk : TYPE
            DESCRIPTION.
        bounds_cost : TYPE
            DESCRIPTION.
        bounds_totval : TYPE
            DESCRIPTION.
        bounds_noise : TYPE
            DESCRIPTION.
        bounds_impf : TYPE
            DESCRIPTION.
        impf_set : TYPE
            DESCRIPTION.
        disc_rate : TYPE
            DESCRIPTION.
        exp : TYPE
            DESCRIPTION.
        meas_set : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return UncVar(
            partial(_ent_unc_func, bounds_noise=bounds_noise, impf_set=impf_set, disc_rate=disc_rate,
                     exp=exp, meas_set=meas_set),
            _ent_unc_dict(bounds_totval, bounds_noise, bounds_impf, bounds_disk, bounds_cost)
        )

    def entfut_unc(bounds_cost, bounds_eg, bounds_noise,
                bounds_impf, impf_set, exp, meas_set):
        """


        Parameters
        ----------
        bounds_cost : TYPE
            DESCRIPTION.
        bounds_eg : TYPE
            DESCRIPTION.
        bounds_noise : TYPE
            DESCRIPTION.
        bounds_impf : TYPE
            DESCRIPTION.
        impf_set : TYPE
            DESCRIPTION.
        exp : TYPE
            DESCRIPTION.
        meas_set : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return UncVar(
            partial(_entfut_unc_func, bounds_noise=bounds_noise, impf_set=impf_set,
                     exp=exp, meas_set=meas_set),
            _entfut_unc_dict(bounds_eg, bounds_noise, bounds_impf, bounds_cost)
        )



#Hazard
def _haz_uncfunc(HE, HI, HF, haz):
    """


    Parameters
    ----------
    HE : TYPE
        DESCRIPTION.
    HI : TYPE
        DESCRIPTION.
    HF : TYPE
        DESCRIPTION.
    haz : TYPE
        DESCRIPTION.

    Returns
    -------
    haz_tmp : TYPE
        DESCRIPTION.

    """
    haz_tmp = copy.deepcopy(haz)
    if HE is not None:
        nb = int(np.round(haz_tmp.size * HE))
        event_names = np.random.choice(haz_tmp.event_name, nb)
        haz_tmp = haz_tmp.select(event_names=event_names)
    if HI is not None:
        haz_tmp.intensity = haz_tmp.intensity.multiply(HI)
    if HF is not None:
        haz_tmp.frequency = np.multiply(haz_tmp.frequency, HF)
    return haz_tmp

def _haz_unc_dict(bounds_ev, bounds_int, bounds_freq):
    """


    Parameters
    ----------
    bounds_ev : TYPE
        DESCRIPTION.
    bounds_int : TYPE
        DESCRIPTION.
    bounds_freq : TYPE
        DESCRIPTION.

    Returns
    -------
    hud : TYPE
        DESCRIPTION.

    """
    hud = {}
    if bounds_ev is not None:
        emin, edelta = bounds_ev[0], bounds_ev[1] - bounds_ev[0]
        hud['HE'] = sp.stats.uniform(emin, edelta)
    if bounds_int is not None:
        imin, idelta = bounds_int[0], bounds_int[1] - bounds_int[0]
        hud['HI'] = sp.stats.uniform(imin, idelta)
    if bounds_freq is not None:
        fmin, fdelta = bounds_freq[0], bounds_freq[1] - bounds_freq[0]
        hud['HF'] = sp.stats.uniform(fmin, fdelta)
    return hud

#Exposure
def _exp_uncfunc(EN, ET, exp, bounds_noise):
    """


    Parameters
    ----------
    EN : TYPE
        DESCRIPTION.
    ET : TYPE
        DESCRIPTION.
    exp : TYPE
        DESCRIPTION.
    bounds_noise : TYPE
        DESCRIPTION.

    Returns
    -------
    exp_tmp : TYPE
        DESCRIPTION.

    """
    exp_tmp = exp.copy(deep=True)
    if EN is not None:
        rnd_vals = np.random.uniform(bounds_noise[0], bounds_noise[1], size = len(exp_tmp.gdf))
        exp_tmp.gdf.value *= rnd_vals
    if ET is not None:
        exp_tmp.gdf.value *= ET
    return exp_tmp

def _exp_unc_dict(bounds_totval, bounds_noise):
    """


    Parameters
    ----------
    bounds_totval : TYPE
        DESCRIPTION.
    bounds_noise : TYPE
        DESCRIPTION.

    Returns
    -------
    eud : TYPE
        DESCRIPTION.

    """
    eud = {}
    if bounds_totval is not None:
        tmin, tmax = bounds_totval[0], bounds_totval[1] - bounds_totval[0]
        eud['ET'] = sp.stats.uniform(tmin, tmax)
    if bounds_noise is not None:
        eud['EN'] = sp.stats.uniform(0, 1)
    return eud

#Impact function set
# def _impfset_uncfunc(IF, impf_set, haz_type='TC', fun_id=1):
#     """


#     Parameters
#     ----------
#     IF : TYPE
#         DESCRIPTION.
#     impf_set : TYPE
#         DESCRIPTION.
#     haz_type : TYPE, optional
#         DESCRIPTION. The default is 'TC'.
#     fun_id : TYPE, optional
#         DESCRIPTION. The default is 1.

#     Returns
#     -------
#     impf_set_tmp : TYPE
#         DESCRIPTION.

#     """
#     impf_set_tmp = copy.deepcopy(impf_set)
#     if IF is not None:
#         new_mdd = np.minimum(impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd * IF, 1.0)
#         impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd = new_mdd
#     return impf_set_tmp

# def _impfset_unc_dict(bounds_impf):
#     """


#     Parameters
#     ----------
#     bounds_impf : TYPE
#         DESCRIPTION.

#     Returns
#     -------
#     iud : TYPE
#         DESCRIPTION.

#     """
#     iud = {}
#     if bounds_impf is not None:
#         xmin, xdelta = bounds_impf[0], bounds_impf[1] - bounds_impf[0]
#         iud['IF'] = sp.stats.uniform(xmin, xdelta)
#     return iud


#Impact function set
def _impfset_uncfunc(IFi, MDD, PAA, impf_set, haz_type='TC', fun_id=1):
    """


    Parameters
    ----------
    IF : TYPE
        DESCRIPTION.
    impf_set : TYPE
        DESCRIPTION.
    haz_type : TYPE, optional
        DESCRIPTION. The default is 'TC'.
    fun_id : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    impf_set_tmp : TYPE
        DESCRIPTION.

    """
    impf_set_tmp = copy.deepcopy(impf_set)
    if MDD is not None:
        new_mdd = np.minimum(impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd * MDD, 1.0)
        impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd = new_mdd
    if PAA is not None:
        new_paa = np.minimum(impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).paa * PAA, 1.0)
        impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).paa = new_paa
    if IFi is not None:
        new_int = np.maximumm(impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).intensity + IFi, 0.0)
        impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).intensity = new_int
    return impf_set_tmp

def _impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa):
    """


    Parameters
    ----------
    bounds_impf : TYPE
        DESCRIPTION.

    Returns
    -------
    iud : TYPE
        DESCRIPTION.

    """
    iud = {}
    if bounds_impfi is not None:
        xmin, xdelta = bounds_impfi[0], bounds_impfi[1] - bounds_impfi[0]
        iud['IFi'] = sp.stats.uniform(xmin, xdelta)
    if bounds_paa is not None:
        xmin, xdelta = bounds_paa[0], bounds_paa[1] - bounds_paa[0]
        iud['PAA'] = sp.stats.uniform(xmin, xdelta)
    if bounds_mdd is not None:
        xmin, xdelta = bounds_mdd[0], bounds_mdd[1] - bounds_mdd[0]
        iud['MDD'] = sp.stats.uniform(xmin, xdelta)
    return iud

#Entity
def _disc_uncfunc(DR, disc_rate):
    """


    Parameters
    ----------
    DR : TYPE
        DESCRIPTION.
    disc_rate : TYPE
        DESCRIPTION.

    Returns
    -------
    disc : TYPE
        DESCRIPTION.

    """
    disc = copy.deepcopy(disc_rate)
    disc.rates = np.ones(disc.years.size) * DR
    return disc

def _disc_unc_dict(bounds_disk):
    """


    Parameters
    ----------
    bounds_disk : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    dmin, ddelta = bounds_disk[0], bounds_disk[1] - bounds_disk[0]
    return  {'DR': sp.stats.uniform(dmin, ddelta)}

def _meas_set_uncfunc(CO, meas_set):
    """


    Parameters
    ----------
    CO : TYPE
        DESCRIPTION.
    meas_set : TYPE
        DESCRIPTION.

    Returns
    -------
    meas_set_tmp : TYPE
        DESCRIPTION.

    """
    meas_set_tmp = copy.deepcopy(meas_set)
    for haz_type in meas_set_tmp.get_hazard_types():
        for meas in meas_set_tmp.get_measure(haz_type=haz_type):
            meas.cost *= CO
    return meas_set_tmp

def _meas_set_unc_dict(bounds_cost):
    """


    Parameters
    ----------
    bounds_cost : TYPE
        DESCRIPTION.

    Returns
    -------
    dict
        DESCRIPTION.

    """
    cmin, cdelta = bounds_cost[0], bounds_cost[1] - bounds_cost[0]
    return {'CO': sp.stats.uniform(cmin, cdelta)}


def _ent_unc_func(EN, ET, IF, CO, DR, bounds_noise,
                 impf_set, disc_rate, exp, meas_set):
    ent = Entity()
    if EN is None or ET is None:
        ent.exposures = exp
    else:
        ent.exposures = _exp_uncfunc(EN, ET, exp, bounds_noise)
    if IF is None:
        ent.impact_func = impf_set
    else:
        ent.impact_funcs = _impfset_uncfunc(IF, impf_set=impf_set)
    if CO is None:
        ent.measures = meas_set
    else:
        ent.measures = _meas_set_uncfunc(CO, meas_set=meas_set)
    if DR is None:
        ent.disc_rates = disc_rate
    else:
        ent.disc_rates = _disc_uncfunc(DR, disc_rate)
    return ent

def _ent_unc_dict(bounds_totval, bounds_noise, bounds_impf, bounds_disk, bounds_cost):
    """


    Parameters
    ----------
    bounds_totval : TYPE
        DESCRIPTION.
    bounds_noise : TYPE
        DESCRIPTION.
    bounds_impf : TYPE
        DESCRIPTION.
    bounds_disk : TYPE
        DESCRIPTION.
    bounds_cost : TYPE
        DESCRIPTION.

    Returns
    -------
    ent_unc_dict : TYPE
        DESCRIPTION.

    """
    ent_unc_dict = _exp_unc_dict(bounds_totval, bounds_noise)
    ent_unc_dict.update(_impfset_unc_dict(bounds_impf))
    ent_unc_dict.update(_disc_unc_dict(bounds_disk))
    ent_unc_dict.update(_meas_set_unc_dict(bounds_cost))
    return  ent_unc_dict

def _entfut_unc_func(ENf, EG, IFf, CO, bounds_noise,
                 impf_set, exp, meas_set):
    """


    Parameters
    ----------
    ENf : TYPE
        DESCRIPTION.
    EG : TYPE
        DESCRIPTION.
    IFf : TYPE
        DESCRIPTION.
    CO : TYPE
        DESCRIPTION.
    bounds_noise : TYPE
        DESCRIPTION.
    impf_set : TYPE
        DESCRIPTION.
    exp : TYPE
        DESCRIPTION.
    meas_set : TYPE
        DESCRIPTION.

    Returns
    -------
    ent : TYPE
        DESCRIPTION.

    """
    ent = Entity()
    if ENf is None or EG is None:
        ent.exposures = exp
    else:
        ent.exposures = _exp_uncfunc(EN=ENf, ET=EG, exp=exp, bounds_noise=bounds_noise)
    if IFf is None:
        ent.impact_func = impf_set
    else:
        ent.impact_funcs = _impfset_uncfunc(IFf, impf_set=impf_set)
    if CO is None:
        ent.measures = meas_set
    else:
        ent.measures = _meas_set_uncfunc(CO, meas_set=meas_set)
    ent.disc_rates = DiscRates() #Disc rate of future entity ignored in cost_benefit.calc()
    return ent

def _entfut_unc_dict(bounds_eg=None, bounds_noise=None, bounds_impf=None,
                     bounds_cost=None):
    """


    Parameters
    ----------
    bounds_eg : TYPE, optional
        DESCRIPTION. The default is None.
    bounds_noise : TYPE, optional
        DESCRIPTION. The default is None.
    bounds_impf : TYPE, optional
        DESCRIPTION. The default is None.
    bounds_cost : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    eud : TYPE
        DESCRIPTION.

    """
    eud = {}
    if bounds_eg is not None:
        gmin, gmax = bounds_eg[0], bounds_eg[1] - bounds_eg[0]
        eud['EG'] = sp.stats.uniform(gmin, gmax)
    if bounds_noise is not None:
        eud['ENf'] = sp.stats.uniform(0, 1)
    if bounds_impf  is  not  None:
        xmin, xdelta = bounds_impf[0], bounds_impf[1] - bounds_impf[0]
        eud.update({'IFf' : sp.stats.uniform(xmin, xdelta)})
    if bounds_cost is not None:
        eud.update(_meas_set_unc_dict(bounds_cost))
    return  eud

