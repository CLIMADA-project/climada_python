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

Define Uncertainty class.
"""

import copy
from functools import partial

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['UncVar']

FIG_W, FIG_H = 8, 5 #default figize width/heigh column/work multiplicators

class UncVar():
    """
    Uncertainty variable

    An uncertainty variable requires a single or multi-parameter function.
    The parameters must follow a given distribution.

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
        as kwargs and which returns a climada object.


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

        Returns
        -------
        None.

        """
        self.labels = list(distr_dict.keys())
        self.distr_dict = distr_dict
        self.uncvar_func = uncvar_func

    def evaluate(self, **params):
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

    def exp_unc(exp, bounds_totval, bounds_noise):
        kwargs = {'exp': exp}
        if bounds_noise is None:
            kwargs['EN'] = None
        return UncVar(
            partial(_exp_uncfunc, **kwargs),
            _exp_unc_dict(bounds_totval, bounds_noise)
            )

    def impfset_unc(impf_set, bounds_impf, haz_type, fun_id=1):
        return UncVar(
            partial(_impfset_unc_func, impf_set=impf_set, haz_type=haz_type, fun_id=fun_id),
            _impfset_unc_dict(bounds_impf)
        )


def _haz_uncfunc(HE, HI, HF, haz):
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
    hud = {}
    if bounds_ev is not None:
        emin, edelta = bounds_ev[0], bounds_ev[1] - bounds_ev[0]
        hud['HE'] = sp.stat.uniform(emin, edelta)
    if bounds_int is not None:
        imin, idelta = bounds_int[0], bounds_int[1] - bounds_int[0]
        hud['HI'] = sp.stat.uniform(imin, idelta)
    if bounds_freq is not None:
        fmin, fdelta = bounds_freq[0], bounds_freq[1] - bounds_freq[0]
        hud['HF'] = sp.stat.uniform(fmin, fdelta)
    return hud

def _exp_uncfunc(EN, ET, exp, bounds_noise):
    exp_tmp = exp.copy(deep=True)
    if EN is not None:
        rnd_vals = np.random.uniform(bounds_noise[0], bounds_noise[1], size = len(exp_tmp.gdf))
        exp_tmp.gdf.value *= rnd_vals
    if ET is not None:
        exp_tmp.gdf.value *= ET
    return exp_tmp

def _exp_unc_dict(bounds_totval, bounds_noise):
    eud = {}
    if bounds_totval is not None:
        tmin, tmax = bounds_totval[0], bounds_totval[1] - bounds_totval[0]
        eud['ET'] = sp.stats.uniform(tmin, tmax)
    if bounds_noise is not None:
        eud['EN'] = sp.stats.uniform(0, 1)
    return eud

def _impfset_unc_func(IF, impf_set, haz_type='TC', fun_id=1):
    impf_set_tmp = copy.deepcopy(impf_set)
    new_mdd = np.minimum(impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd * IF, 1.0)
    impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd = new_mdd
    return impf_set_tmp

def _impfset_unc_dict(bounds_impf):
    xmin, xdelta = bounds_impf[0], bounds_impf[1] - bounds_impf[0]
    return {'IF' : sp.stats.uniform(xmin, xdelta)}
