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

Define InputVar class.
"""

import copy
from functools import partial
from itertools import zip_longest

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from climada.entity import Entity, DiscRates

__all__ = ['InputVar']

FIG_W, FIG_H = 8, 5 #default figize width/heigh column/work multiplicators

class InputVar():
    """
    Input variable for the uncertainty analysis

    An uncertainty input variable requires a single or multi-parameter function.
    The parameters must follow a given distribution. The uncertainty input
    variables are the input parameters of the model.

    Attributes
    ----------
    distr_dict : dict
        Distribution of the uncertainty parameters. Keys are uncertainty
        parameters names and Values are probability density distribution
        from the scipy.stats package
        https://docs.scipy.org/doc/scipy/reference/stats.html
    labels : list
        Names of the uncertainty parameters (keys of distr_dict)
    func : function
        User defined python fucntion with the uncertainty parameters
        as keyword arguements and which returns a climada object.

    Notes
    -----
    A few default Variables are defined for Hazards, Exposures,
    Impact Fucntions, Measures and Entities.


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
        iv_cat = InputVar(func=litpop_cat, distr_dict=distr_dict)

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
        iv_cont = InputVar(func=imp_fun_tc, distr_dict=distr_dict)

    """

    def __init__(self, func, distr_dict):
        """
        Initialize InputVar

        Parameters
        ----------
        func : function
            Variable defined as a function of the uncertainty parameters
        distr_dict : dict
            Dictionary of the probability density distributions of the
            uncertainty parameters, with keys matching the keyword
            arguments (i.e. uncertainty parameters) of the func
            function.
            The distribution must be of type scipy.stats
            https://docs.scipy.org/doc/scipy/reference/stats.html

        """
        self.labels = list(distr_dict.keys())
        self.distr_dict = distr_dict
        self.func = func

    def evaluate(self, **params):
        """
        Return the value of uncertainty input variable.

        By default, the value of the average is returned.

        Parameters
        ----------
        **params : optional
            Input parameters will be passed to self.InputVar_func.

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
        return self.func(**params)


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
    def var_to_inputvar(var):
        """
        Returns an uncertainty variable with no distribution if var is not
        an InputVar. Else, returns var.

        Parameters
        ----------
        var : climada.uncertainty.InputVar or any other CLIMADA object

        Returns
        -------
        InputVar
            var if var is InputVar, else InputVar with var and no distribution.

        """

        if isinstance(var, InputVar):
            return var

        return InputVar(func=lambda: var, distr_dict={})

    @staticmethod
    def haz(haz, bounds_ev=None, bounds_int=None, bounds_freq=None):
        """
        Helper wrapper for basic hazard uncertainty input variable

        The following types of uncertainties can be added:
        HE: sub-sampling events from the total event set
            The number of events in each sub-sample is sampled
            uniformly from a distribution with (min, max) = bounds_ev
        HI: scale the intensity of all events (homogeneously)
            The instensity of all events is multiplied by a number
            sampled uniformly from a distribution with (min, max) = bounds_int
        HF: scale the frequency of all events (homogeneously)
            The frequency of all events is multiplied by a number
            sampled uniformly from a distribution with (min, max) = bounds_freq

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        haz : climada.hazard.Hazard
            The base hazard
        bounds_ev : (min, max), optional
            Bounds of the uniform distribution for the number of events
            to be sampled per sample. The default is None.
        bounds_int : (min, max), optional
            Bounds of the uniform distribution for the homogeneous intensity
            scaling. The default is None.
        bounds_freq : TYPE, optional
            Bounds of the uniform distribution for the homogeneous frequency
            scaling. The default is None.

        Returns
        -------
        climada.engine.uncertainty_quantification.input_var.InputVar
            Uncertainty input variable for a hazard object.

        """
        kwargs = {'haz': haz}
        if bounds_ev is None:
            kwargs['HE'] = None
        if bounds_int is None:
            kwargs['HI'] = None
        if bounds_freq is None:
            kwargs['HF'] = None
        return InputVar(
            partial(_haz_uncfunc, **kwargs),
            _haz_unc_dict(bounds_ev, bounds_int, bounds_freq)
            )

    @staticmethod
    def exp(exp, bounds_totval=None, bounds_noise=None):
        """
        Helper wrapper for basic exposure uncertainty input variable

        The following types of uncertainties can be added:
        ET: scale the total value (homogeneously)
            The value at each exposure point is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_totvalue
        EN: mutliplicative noise (inhomogeneous)
            The value of each exposure point is independently multiplied by
            a random number sampled uniformly from a distribution
            with (min, max) = bounds_noise

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        exp : climada.entity.exposures.Exposures
            The base exposure.
        bounds_totval : (min, max), optional
            Bounds of the uniform distribution for the homogeneous total value
            scaling.. The default is None.
        bounds_noise : (min, max), optional
            Bounds of the uniform distribution to scale each exposure point
            independently. The default is None.

        Returns
        -------
        climada.engine.uncertainty_quantification.input_var.InputVar
            Uncertainty input variable for an exposure object.

        """
        kwargs = {'exp': exp, 'bounds_noise': bounds_noise}
        if bounds_noise is None:
            kwargs['EN'] = None
        if bounds_totval is None:
            kwargs['ET'] = None
        return InputVar(
            partial(_exp_uncfunc, **kwargs),
            _exp_unc_dict(bounds_totval, bounds_noise)
            )

    @staticmethod
    def impfset(impf_set, bounds_mdd=None, bounds_paa=None,
                    bounds_impfi=None, haz_type='TC', fun_id=1):
        """
        Helper wrapper for basic impact function set uncertainty input variable.

        One impact function (chosen with haz_type and fun_id) is characterized.

        The following types of uncertainties can be added:
        MDD: scale the mdd (homogeneously)
            The value of mdd at each intensity is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_mdd
        PAA: scale the paa (homogeneously)
            The value of paa at each intensity is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_paa
        IFi: shift the intensity (homogeneously)
            The value intensity are all summed with a random number
            sampled uniformly from a distribution with
            (min, max) = bounds_int

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        impf_set : climada.entity.impact_funcs.impact_func_set.ImpactFuncSet
            The base impact function set.
        bounds_mdd : (min, max), optional
            Bounds of the uniform distribution for the homogeneous mdd
            scaling. The default is None.
        bounds_paa : (min, max), optional
            Bounds of the uniform distribution for the homogeneous paa
            scaling. The default is None.
        bounds_impfi : (min, max), optional
            Bounds of the uniform distribution for the homogeneous shift
            of intensity. The default is None.
        haz_type : str, optional
            The hazard type of the impact function. The default is 'TC'.
        fun_id : int, optional
            The id of the impact function. The default is 1.

        Returns
        -------
        climada.engine.uncertainty_quantification.input_var.InputVar
            DESCRIPTION.

        """
        kwargs = {}
        if bounds_mdd is None:
            kwargs['MDD'] = None
        if bounds_paa is None:
            kwargs['PAA'] = None
        if bounds_impfi is None:
            kwargs['IFi'] = None
        return InputVar(
            partial(
                _impfset_uncfunc, impf_set=impf_set, haz_type=haz_type,
                fun_id=fun_id, **kwargs
                ),
            _impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa)
        )

    @staticmethod
    def ent(impf_set, disc_rate, exp, meas_set,
            bounds_disc=None, bounds_cost=None, bounds_totval=None,
            bounds_noise=None, bounds_mdd=None, bounds_paa=None,
            bounds_impfi=None):
        """
        Helper wrapper for basic entity set uncertainty input variable.

        The following types of uncertainties can be added:
        DR: scale the discount rates (homogeneously)
            The value of the discounts in each year is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_disc
        CO: scale the cost (homogeneously)
            The cost of all measures is multiplied by the same number
            sampled uniformly from a distribution with
            (min, max) = bounds_cost
        ET: scale the total value (homogeneously)
            The value at each exposure point is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_totvalue
        EN: mutliplicative noise (inhomogeneous)
            The value of each exposure point is independently multiplied by
            a random number sampled uniformly from a distribution
            with (min, max) = bounds_noise
        MDD: scale the mdd (homogeneously)
            The value of mdd at each intensity is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_mdd
        PAA: scale the paa (homogeneously)
            The value of paa at each intensity is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_paa
        IFi: shift the intensity (homogeneously)
            The value intensity are all summed with a random number
            sampled uniformly from a distribution with
            (min, max) = bounds_int


        If a bounds is None, this parameter is assumed to have no uncertainty.


        Parameters
        ----------
        bounds_disk : (min, max), optional
            Bounds of the uniform distribution for the homogeneous discount
            rate scaling. The default is None.
        bounds_cost :(min, max), optional
            Bounds of the uniform distribution for the homogeneous cost
            of all measures scaling. The default is None.
        bounds_totval : (min, max), optional
            Bounds of the uniform distribution for the homogeneous total
            exposure value scaling. The default is None.
        bounds_noise : (min, max), optional
            Bounds of the uniform distribution to scale each exposure point
            independently. The default is None.
        bounds_mdd : (min, max), optional
            Bounds of the uniform distribution for the homogeneous mdd
            scaling. The default is None.
        bounds_paa : (min, max), optional
            Bounds of the uniform distribution for the homogeneous paa
            scaling. The default is None.
        bounds_int : (min, max), optional
            Bounds of the uniform distribution for the homogeneous shift
            of intensity. The default is None.
        impf_set : climada.engine.impact_funcs.impact_func_set.ImpactFuncSet
            The base impact function set.
        disc_rate : climada.entity.disc_rates.base.DiscRates
            The base discount rates.
        exp : climada.entity.exposures.base.Exposure
            The base exposure.
        meas_set : climada.entity.measures.measure_set.MeasureSet
            The base measures.

        Returns
        -------
        climada.engine.uncertainty_quantification.input_var.InputVar
            Entity uncertainty input variable

        """
        kwargs = {}
        if bounds_mdd is None:
            kwargs['MDD'] = None
        if bounds_paa is None:
            kwargs['PAA'] = None
        if bounds_impfi is None:
            kwargs['IFi'] = None
        if bounds_disc is None:
            kwargs['DR'] = None
        if bounds_cost is None:
            kwargs['CO'] = None
        if bounds_totval is None:
            kwargs['ET'] = None
        if bounds_noise is None:
            kwargs['EN'] = None


        return InputVar(
            partial(_ent_unc_func, impf_set=impf_set, disc_rate=disc_rate,
                    bounds_noise=bounds_noise,
                    exp=exp, meas_set=meas_set, **kwargs),
            _ent_unc_dict(bounds_totval, bounds_noise, bounds_impfi, bounds_mdd,
                          bounds_paa, bounds_disc, bounds_cost)
        )

    @staticmethod
    def entfut(impf_set, exp, meas_set,
               bounds_cost=None, bounds_eg=None, bounds_noise=None,
                bounds_impfi=None, bounds_mdd=None, bounds_paa=None
                ):
        """
        Helper wrapper for basic future entity set uncertainty input variable.

        The following types of uncertainties can be added:
        CO: scale the cost (homogeneously)
            The cost of all measures is multiplied by the same number
            sampled uniformly from a distribution with
            (min, max) = bounds_cost
        EG: scale the exposures growth (homogeneously)
            The value at each exposure point is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_eg
        EN: mutliplicative noise (inhomogeneous)
            The value of each exposure point is independently multiplied by
            a random number sampled uniformly from a distribution
            with (min, max) = bounds_noise
        MDD: scale the mdd (homogeneously)
            The value of mdd at each intensity is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_mdd
        PAA: scale the paa (homogeneously)
            The value of paa at each intensity is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_paa
        IFi: shift the impact function intensity (homogeneously)
            The value intensity are all summed with a random number
            sampled uniformly from a distribution with
            (min, max) = bounds_impfi


        If a bounds is None, this parameter is assumed to have no uncertainty.


        Parameters
        ----------
        bounds_cost :(min, max), optional
            Bounds of the uniform distribution for the homogeneous cost
            of all measures scaling. The default is None.
        bounds_eg : (min, max), optional
            Bounds of the uniform distribution for the homogeneous total
            exposure growth scaling. The default is None.
        bounds_noise : (min, max), optional
            Bounds of the uniform distribution to scale each exposure point
            independently. The default is None.
        bounds_mdd : (min, max), optional
            Bounds of the uniform distribution for the homogeneous mdd
            scaling. The default is None.
        bounds_paa : (min, max), optional
            Bounds of the uniform distribution for the homogeneous paa
            scaling. The default is None.
        bounds_impfi : (min, max), optional
            Bounds of the uniform distribution for the homogeneous shift
            of intensity. The default is None.
        impf_set : climada.engine.impact_funcs.impact_func_set.ImpactFuncSet
            The base impact function set.
        exp : climada.entity.exposures.base.Exposure
            The base exposure.
        meas_set : climada.entity.measures.measure_set.MeasureSet
            The base measures.

        Returns
        -------
        climada.engine.uncertainty_quantification.input_var.InputVar
            Entity uncertainty input variable

        """

        kwargs = {}
        if bounds_mdd is None:
            kwargs['MDD'] = None
        if bounds_paa is None:
            kwargs['PAA'] = None
        if bounds_impfi is None:
            kwargs['IFi'] = None
        if bounds_cost is None:
            kwargs['CO'] = None
        if bounds_eg is None:
            kwargs['EG'] = None
        if bounds_noise is None:
            kwargs['EN'] = None

        return InputVar(
            partial(_entfut_unc_func, bounds_noise=bounds_noise, impf_set=impf_set,
                     exp=exp, meas_set=meas_set, **kwargs),
            _entfut_unc_dict(bounds_eg=bounds_eg, bounds_noise=bounds_noise,
                             bounds_impfi=bounds_impfi, bounds_paa=bounds_paa,
                             bounds_mdd=bounds_mdd, bounds_cost=bounds_cost)
        )


#Hazard
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

#Impact function set
def _impfset_uncfunc(IFi, MDD, PAA, impf_set, haz_type='TC', fun_id=1):
    impf_set_tmp = copy.deepcopy(impf_set)
    if MDD is not None:
        new_mdd = np.minimum(
            impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd * MDD,
            1.0
            )
        impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).mdd = new_mdd
    if PAA is not None:
        new_paa = np.minimum(
            impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).paa * PAA,
            1.0
            )
        impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).paa = new_paa
    if IFi is not None:
        new_int = np.maximumm(
            impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).intensity + IFi,
            0.0
            )
        impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).intensity = new_int
    return impf_set_tmp

def _impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa):
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
    disc = copy.deepcopy(disc_rate)
    if DR is not None:
        disc.rates = np.ones(disc.years.size) * DR
    return disc

def _disc_unc_dict(bounds_disk):
    if bounds_disk is None:
        return {}
    dmin, ddelta = bounds_disk[0], bounds_disk[1] - bounds_disk[0]
    return  {'DR': sp.stats.uniform(dmin, ddelta)}

def _meas_set_uncfunc(CO, meas_set):
    meas_set_tmp = copy.deepcopy(meas_set)
    for haz_type in meas_set_tmp.get_hazard_types():
        for meas in meas_set_tmp.get_measure(haz_type=haz_type):
            meas.cost *= CO
    return meas_set_tmp

def _meas_set_unc_dict(bounds_cost):
    cmin, cdelta = bounds_cost[0], bounds_cost[1] - bounds_cost[0]
    return {'CO': sp.stats.uniform(cmin, cdelta)}

def _ent_unc_func(EN, ET, IFi, MDD, PAA, CO, DR, bounds_noise,
                 impf_set, disc_rate, exp, meas_set):
    ent = Entity()
    if EN is None and ET is None:
        ent.exposures = exp
    else:
        ent.exposures = _exp_uncfunc(EN, ET, exp, bounds_noise)
    if MDD is None and PAA is None and IFi is None:
        ent.impact_funcs = impf_set
    else:
        ent.impact_funcs = _impfset_uncfunc(IFi, MDD, PAA, impf_set=impf_set)
    if CO is None:
        ent.measures = meas_set
    else:
        ent.measures = _meas_set_uncfunc(CO, meas_set=meas_set)
    if DR is None:
        ent.disc_rates = disc_rate
    else:
        ent.disc_rates = _disc_uncfunc(DR, disc_rate)
    return ent

def _ent_unc_dict(bounds_totval, bounds_noise, bounds_impfi, bounds_mdd,
                  bounds_paa, bounds_disk, bounds_cost):
    ent_unc_dict = _exp_unc_dict(bounds_totval, bounds_noise)
    ent_unc_dict.update(_impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa))
    ent_unc_dict.update(_disc_unc_dict(bounds_disk))
    ent_unc_dict.update(_meas_set_unc_dict(bounds_cost))
    return  ent_unc_dict

def _entfut_unc_func(EN, EG, IFi, MDD, PAA, CO, bounds_noise,
                 impf_set, exp, meas_set):
    ent = Entity()
    if EN is None and EG is None:
        ent.exposures = exp
    else:
        ent.exposures = _exp_uncfunc(EN=EN, ET=EG, exp=exp, bounds_noise=bounds_noise)
    if IFi is None and PAA is None and MDD is None:
        ent.impact_funcs = impf_set
    else:
        ent.impact_funcs = _impfset_uncfunc(IFi, MDD, PAA, impf_set=impf_set)
    if CO is None:
        ent.measures = meas_set
    else:
        ent.measures = _meas_set_uncfunc(CO, meas_set=meas_set)
    ent.disc_rates = DiscRates() #Disc rate of future entity ignored in cost_benefit.calc()
    return ent

def _entfut_unc_dict(bounds_impfi, bounds_mdd,
                  bounds_paa, bounds_eg, bounds_noise,
                  bounds_cost):
    eud = {}
    if bounds_eg is not None:
        gmin, gmax = bounds_eg[0], bounds_eg[1] - bounds_eg[0]
        eud['EG'] = sp.stats.uniform(gmin, gmax)
    if bounds_noise is not None:
        eud['EN'] = sp.stats.uniform(0, 1)
    eud.update(_impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa))
    if bounds_cost is not None:
        eud.update(_meas_set_unc_dict(bounds_cost))
    return eud
