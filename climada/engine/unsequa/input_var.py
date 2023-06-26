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
import logging
from typing import Dict

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from climada.entity import Entity, DiscRates

LOGGER = logging.getLogger(__name__)

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

    >>> import scipy as sp
    >>> def litpop_cat(m, n):
    ...     exp = Litpop.from_countries('CHE', exponent=[m, n])
    ...     return exp
    >>> distr_dict = {
    ...     'm': sp.stats.randint(low=0, high=5),
    ...     'n': sp.stats.randint(low=0, high=5)
    ...     }
    >>> iv_cat = InputVar(func=litpop_cat, distr_dict=distr_dict)

    Continuous variable function: Impact function for TC

    >>> import scipy as sp
    >>> def imp_fun_tc(G, v_half, vmin, k, _id=1):
    ...     intensity = np.linspace(0, 150, num=100)
    ...     mdd = np.repeat(1, len(intensity))
    ...     paa = np.array([sigmoid_function(v, G, v_half, vmin, k)
    ...                     for v in intensity])
    ...     imp_fun = ImpactFunc(haz_type='TC',
    ...                          id=_id,
    ...                          intensity_unit='m/s',
    ...                          intensity=intensity,
    ...                          mdd=mdd,
    ...                          paa=paa)
    ...     imp_fun.check()
    ...     impf_set = ImpactFuncSet([imp_fun])
    ...     return impf_set
    >>> distr_dict = {"G": sp.stats.uniform(0.8, 1),
    ...       "v_half": sp.stats.uniform(50, 100),
    ...       "vmin": sp.stats.norm(loc=15, scale=30),
    ...       "k": sp.stats.randint(low=1, high=9)
    ...       }
    >>> iv_cont = InputVar(func=imp_fun_tc, distr_dict=distr_dict)
    """

    def __init__(
        self,
        func: callable,
        distr_dict: Dict,
    ):
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
            Input parameters will be passed to self.func.

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

            >>> nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
            >>> figsize = (ncols * FIG_W, nrows * FIG_H)

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
            low = distr.ppf(1e-10)
            high = distr.ppf(1-1e-10)
            n = 100
            try:
                x = np.linspace(low, high, n)
                ax.plot(x, distr.pdf(x), label=param_name)
            except AttributeError:
                if (high - low) > n:
                    x = np.arange(low, high, int((high-low) / n))
                else:
                    x = np.arange(low, high+1)
                ax.vlines(x, 0, distr.pmf(x), label=param_name)
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
    def haz(haz_list, n_ev=None, bounds_int=None, bounds_frac=None, bounds_freq=None):
        """
        Helper wrapper for basic hazard uncertainty input variable

        The following types of uncertainties can be added:

        HE: sub-sampling events from the total event set
            For each sub-sample, n_ev events are sampled with replacement.
            HE is the value of the seed
            for  the uniform random number generator.
        HI: scale the intensity of all events (homogeneously)
            The instensity of all events is multiplied by a number
            sampled uniformly from a distribution with (min, max) = bounds_int
        HA: scale the fraction of all events (homogeneously)
            The fraction of all events is multiplied by a number
            sampled uniformly from a distribution with (min, max) = bounds_frac
        HF: scale the frequency of all events (homogeneously)
            The frequency of all events is multiplied by a number
            sampled uniformly from a distribution with (min, max) = bounds_freq
        HL: sample uniformly from hazard list
            From the provided list of hazard is elements are uniformly
            sampled. For example, Hazards outputs from dynamical models
            for different input factors.

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        haz : List of climada.hazard.Hazard
            The list of base hazard. Can be one or many to uniformly sample
            from.
        n_ev : int, optional
            Number of events to be subsampled per sample. Can be equal or
            larger than haz.size. The default is None.
        bounds_int : (float, float), optional
            Bounds of the uniform distribution for the homogeneous intensity
            scaling. The default is None.
        bounds_frac : (float, float), optional
            Bounds of the uniform distribution for the homogeneous fraction
            scaling. The default is None.
        bounds_freq : (float, float), optional
            Bounds of the uniform distribution for the homogeneous frequency
            scaling. The default is None.

        Returns
        -------
        climada.engine.unsequa.input_var.InputVar
            Uncertainty input variable for a hazard object.

        """
        n_haz = len(haz_list)
        kwargs = {'haz_list': haz_list, 'n_ev': n_ev}
        if n_ev is None:
            kwargs['HE'] = None
        if bounds_int is None:
            kwargs['HI'] = None
        if bounds_frac is None:
            kwargs['HA'] = None
        if bounds_freq is None:
            kwargs['HF'] = None
        if n_haz == 1:
            kwargs['HL'] = 0
        return InputVar(
            partial(_haz_uncfunc, **kwargs),
            _haz_unc_dict(n_ev, bounds_int, bounds_frac, bounds_freq, n_haz)
            )

    @staticmethod
    def exp(exp_list, bounds_totval=None, bounds_noise=None):
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
            with (min, max) = bounds_noise. EN is the value of the seed
            for  the uniform random number generator.
        EL: sample uniformly from exposure list
            From the provided list of exposure is elements are uniformly
            sampled. For example, LitPop instances with different exponents.

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        exp_list : list of climada.entity.exposures.Exposures
            The list of base exposure. Can be one or many to uniformly sample
            from.
        bounds_totval : (float, float), optional
            Bounds of the uniform distribution for the homogeneous total value
            scaling.. The default is None.
        bounds_noise : (float, float), optional
            Bounds of the uniform distribution to scale each exposure point
            independently. The default is None.

        Returns
        -------
        climada.engine.unsequa.input_var.InputVar
            Uncertainty input variable for an exposure object.

        """
        n_exp = len(exp_list)
        kwargs = {'exp_list': exp_list, 'bounds_noise': bounds_noise}
        if bounds_noise is None:
            kwargs['EN'] = None
        if bounds_totval is None:
            kwargs['ET'] = None
        if n_exp == 1:
            kwargs['EL'] = 0
        return InputVar(
            partial(_exp_uncfunc, **kwargs),
            _exp_unc_dict(bounds_totval=bounds_totval,
                          bounds_noise=bounds_noise,
                          n_exp=n_exp)
            )

    @staticmethod
    def impfset(impf_set_list, haz_id_dict= None, bounds_mdd=None, bounds_paa=None,
                bounds_impfi=None):
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
        IL: sample uniformly from impact function set list
            From the provided list of impact function sets elements are uniformly
            sampled. For example, impact functions obtained from different
            calibration methods.


        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        impf_set_list : list of ImpactFuncSet
            The list of  base impact function set. Can be one or many to
            uniformly sample from. The impact function ids must identical
            for all impact function sets.
        bounds_mdd : (float, float), optional
            Bounds of the uniform distribution for the homogeneous mdd
            scaling. The default is None.
        bounds_paa : (float, float), optional
            Bounds of the uniform distribution for the homogeneous paa
            scaling. The default is None.
        bounds_impfi : (float, float), optional
            Bounds of the uniform distribution for the homogeneous shift
            of intensity. The default is None.
        haz_id_dict : dict(), optional
            Dictionary of the impact functions affected by uncertainty.
            Keys are hazard types (str), values are a list of impact
            function id (int).
            Default is impsf_set.get_ids() i.e. all impact functions in the set

        Returns
        -------
        climada.engine.unsequa.input_var.InputVar
            Uncertainty input variable for an impact function set object.

        """
        n_impf_set = len(impf_set_list)
        kwargs = {'impf_set_list': impf_set_list}
        if bounds_mdd is None:
            kwargs['MDD'] = None
        if bounds_paa is None:
            kwargs['PAA'] = None
        if bounds_impfi is None:
            kwargs['IFi'] = None
        if haz_id_dict is None:
            haz_id_dict = impf_set_list[0].get_ids()
        if n_impf_set == 1:
            kwargs['IL'] = 0
        return InputVar(
            partial(
                _impfset_uncfunc, haz_id_dict=haz_id_dict,
                **kwargs
                ),
            _impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa, n_impf_set)
        )

    @staticmethod
    def ent(impf_set_list, disc_rate, exp_list, meas_set, haz_id_dict,
            bounds_disc=None, bounds_cost=None, bounds_totval=None,
            bounds_noise=None, bounds_mdd=None, bounds_paa=None,
            bounds_impfi=None):
        """
        Helper wrapper for basic entity set uncertainty input variable.

        Important: only the impact function defined by haz_type and
        fun_id will be affected by bounds_impfi, bounds_mdd, bounds_paa.

        The following types of uncertainties can be added:

        DR: value of constant discount rate (homogeneously)
            The value of the discounts in each year is
            sampled uniformly from a distribution with
            (min, max) = bounds_disc
        CO: scale the cost (homogeneously)
            The cost of all measures is multiplied by the same number
            sampled uniformly from a distribution with
            (min, max) = bounds_cost
        ET: scale the total value (homogeneously)
            The value at each exposure point is multiplied by a number
            sampled uniformly from a distribution with
            (min, max) = bounds_totval
        EN: mutliplicative noise (inhomogeneous)
            The value of each exposure point is independently multiplied by
            a random number sampled uniformly from a distribution
            with (min, max) = bounds_noise. EN is the value of the seed
            for  the uniform random number generator.
        EL: sample uniformly from exposure list
            From the provided list of exposure is elements are uniformly
            sampled. For example, LitPop instances with different exponents.
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
        IL: sample uniformly from impact function set list
            From the provided list of impact function sets elements are uniformly
            sampled. For example, impact functions obtained from different
            calibration methods.

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        bounds_disk : (float, float), optional
            Bounds of the uniform distribution for the homogeneous discount
            rate scaling. The default is None.
        bounds_cost :(float, float), optional
            Bounds of the uniform distribution for the homogeneous cost
            of all measures scaling. The default is None.
        bounds_totval : (float, float), optional
            Bounds of the uniform distribution for the homogeneous total
            exposure value scaling. The default is None.
        bounds_noise : (float, float), optional
            Bounds of the uniform distribution to scale each exposure point
            independently. The default is None.
        bounds_mdd : (float, float), optional
            Bounds of the uniform distribution for the homogeneous mdd
            scaling. The default is None.
        bounds_paa : (float, float), optional
            Bounds of the uniform distribution for the homogeneous paa
            scaling. The default is None.
        bounds_impfi : (float, float), optional
            Bounds of the uniform distribution for the homogeneous shift
            of intensity. The default is None.
        impf_set_list : list of ImpactFuncSet
            The list of  base impact function set. Can be one or many to
            uniformly sample from. The impact function ids must identical
            for all impact function sets.
        disc_rate : climada.entity.disc_rates.base.DiscRates
            The base discount rates.
        exp_list : [climada.entity.exposures.base.Exposure]
            The list of base exposure. Can be one or many to uniformly sample
            from.
        meas_set : climada.entity.measures.measure_set.MeasureSet
            The base measures.
        haz_id_dict : dict
            Dictionary of the impact functions affected by uncertainty.
            Keys are hazard types (str), values are a list of impact
            function id (int).

        Returns
        -------
        climada.engine.unsequa.input_var.InputVar
            Entity uncertainty input variable

        """
        n_exp = len(exp_list)
        n_impf_set = len(impf_set_list)

        kwargs = {}
        if bounds_mdd is None:
            kwargs['MDD'] = None
        if bounds_paa is None:
            kwargs['PAA'] = None
        if bounds_impfi is None:
            kwargs['IFi'] = None
        if n_impf_set== 1:
            kwargs['IL'] = 0
        if bounds_disc is None:
            kwargs['DR'] = None
        if bounds_cost is None:
            kwargs['CO'] = None
        if bounds_totval is None:
            kwargs['ET'] = None
        if bounds_noise is None:
            kwargs['EN'] = None
        if n_exp == 1:
            kwargs['EL'] = 0

        return InputVar(
            partial(_ent_unc_func,
                    impf_set_list=impf_set_list, haz_id_dict=haz_id_dict,
                    disc_rate=disc_rate, bounds_noise=bounds_noise,
                    exp_list=exp_list, meas_set=meas_set, **kwargs
                    ),
            _ent_unc_dict(bounds_totval=bounds_totval, bounds_noise=bounds_noise,
                          bounds_impfi=bounds_impfi, n_impf_set=n_impf_set,
                          bounds_mdd=bounds_mdd,
                          bounds_paa=bounds_paa, bounds_disc=bounds_disc,
                          bounds_cost=bounds_cost, n_exp=n_exp,)
        )

    @staticmethod
    def entfut(impf_set_list, exp_list, meas_set, haz_id_dict,
               bounds_cost=None, bounds_eg=None, bounds_noise=None,
               bounds_impfi=None, bounds_mdd=None, bounds_paa=None,
               ):
        """
        Helper wrapper for basic future entity set uncertainty input variable.

        Important: only the impact function defined by haz_type and
        fun_id will be affected by bounds_impfi, bounds_mdd, bounds_paa.

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
            with (min, max) = bounds_noise. EN is the value of the seed
            for  the uniform random number generator.
        EL: sample uniformly from exposure list
            From the provided list of exposure is elements are uniformly
            sampled. For example, LitPop instances with different exponents.
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
        IL: sample uniformly from impact function set list
            From the provided list of impact function sets elements are uniformly
            sampled. For example, impact functions obtained from different
            calibration methods.

        If a bounds is None, this parameter is assumed to have no uncertainty.

        Parameters
        ----------
        bounds_cost :(float, float), optional
            Bounds of the uniform distribution for the homogeneous cost
            of all measures scaling. The default is None.
        bounds_eg : (float, float), optional
            Bounds of the uniform distribution for the homogeneous total
            exposure growth scaling. The default is None.
        bounds_noise : (float, float), optional
            Bounds of the uniform distribution to scale each exposure point
            independently. The default is None.
        bounds_mdd : (float, float), optional
            Bounds of the uniform distribution for the homogeneous mdd
            scaling. The default is None.
        bounds_paa : (float, float), optional
            Bounds of the uniform distribution for the homogeneous paa
            scaling. The default is None.
        bounds_impfi : (float, float), optional
            Bounds of the uniform distribution for the homogeneous shift
            of intensity. The default is None.
        impf_set_list : list of ImpactFuncSet
            The list of  base impact function set. Can be one or many to
            uniformly sample from. The impact function ids must identical
            for all impact function sets.
        exp_list : [climada.entity.exposures.base.Exposure]
            The list of base exposure. Can be one or many to uniformly sample
            from.
        meas_set : climada.entity.measures.measure_set.MeasureSet
            The base measures.
        haz_id_dict : dict
            Dictionary of the impact functions affected by uncertainty.
            Keys are hazard types (str), values are a list of impact
            function id (int).

        Returns
        -------
        climada.engine.unsequa.input_var.InputVar
            Entity uncertainty input variable

        """
        n_exp = len(exp_list)
        n_impf_set = len(impf_set_list)

        kwargs = {}
        if bounds_mdd is None:
            kwargs['MDD'] = None
        if bounds_paa is None:
            kwargs['PAA'] = None
        if bounds_impfi is None:
            kwargs['IFi'] = None
        if n_impf_set == 1:
            kwargs['IL'] = 0
        if bounds_cost is None:
            kwargs['CO'] = None
        if bounds_eg is None:
            kwargs['EG'] = None
        if bounds_noise is None:
            kwargs['EN'] = None
        if n_exp == 1:
            kwargs['EL'] = 0

        return InputVar(
            partial(_entfut_unc_func, haz_id_dict=haz_id_dict,
                    bounds_noise=bounds_noise, impf_set_list=impf_set_list,
                    exp_list=exp_list, meas_set=meas_set, **kwargs),
            _entfut_unc_dict(bounds_eg=bounds_eg, bounds_noise=bounds_noise,
                             bounds_impfi=bounds_impfi, n_impf_set=n_impf_set,
                             bounds_paa=bounds_paa,
                             bounds_mdd=bounds_mdd, bounds_cost=bounds_cost,
                             n_exp=n_exp)
        )


#Hazard
def _haz_uncfunc(HE, HI, HA, HF, HL, haz_list, n_ev):
    haz_tmp = copy.deepcopy(haz_list[int(HL)])
    if HE is not None:
        rng = np.random.RandomState(int(HE))
        event_id = list(rng.choice(haz_tmp.event_id, int(n_ev)))
        haz_tmp = haz_tmp.select(event_id=event_id)
    if HI is not None:
        haz_tmp.intensity = haz_tmp.intensity.multiply(HI)
    if HA is not None:
        haz_tmp.fraction = haz_tmp.fraction.multiply(HA)
    if HF is not None:
        haz_tmp.frequency = np.multiply(haz_tmp.frequency, HF)
    return haz_tmp

def _haz_unc_dict(n_ev, bounds_int, bounds_frac, bounds_freq, n_haz):
    hud = {}
    if n_ev is not None:
        hud['HE'] = sp.stats.randint(0, 2**32 - 1) #seed for rnd generator
    if bounds_int is not None:
        imin, idelta = bounds_int[0], bounds_int[1] - bounds_int[0]
        hud['HI'] = sp.stats.uniform(imin, idelta)
    if bounds_frac is not None:
        amin, adelta = bounds_frac[0], bounds_frac[1] - bounds_frac[0]
        hud['HA'] = sp.stats.uniform(amin, adelta)
    if bounds_freq is not None:
        fmin, fdelta = bounds_freq[0], bounds_freq[1] - bounds_freq[0]
        hud['HF'] = sp.stats.uniform(fmin, fdelta)
    if n_haz > 1:
        hud['HL'] = sp.stats.randint(0, n_haz)
    return hud

#Exposure
def _exp_uncfunc(EN, ET, EL, exp_list, bounds_noise):
    exp_tmp = exp_list[int(EL)].copy(deep=True)
    if EN is not None:
        rng = np.random.RandomState(int(EN))
        rnd_vals = rng.uniform(bounds_noise[0], bounds_noise[1], size = len(exp_tmp.gdf))
        exp_tmp.gdf.value *= rnd_vals
    if ET is not None:
        exp_tmp.gdf.value *= ET
    return exp_tmp

def _exp_unc_dict(bounds_totval, bounds_noise, n_exp):
    eud = {}
    if bounds_totval is not None:
        tmin, tmax = bounds_totval[0], bounds_totval[1] - bounds_totval[0]
        eud['ET'] = sp.stats.uniform(tmin, tmax)
    if bounds_noise is not None:
        eud['EN'] = sp.stats.randint(0, 2**32 - 1) #seed for rnd generator
    if n_exp > 1:
        eud['EL'] = sp.stats.randint(0, n_exp)
    return eud

#Impact function set
def _impfset_uncfunc(IFi, MDD, PAA, IL, impf_set_list, haz_id_dict):
    impf_set_tmp = copy.deepcopy(impf_set_list[int(IL)])
    for haz_type, fun_id_list in haz_id_dict.items():
        for fun_id in fun_id_list:
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
                new_int = np.maximum(
                    impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).intensity + IFi,
                    0.0
                    )
                impf_set_tmp.get_func(haz_type=haz_type, fun_id=fun_id).intensity = new_int
    return impf_set_tmp

def _impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa, n_impf_set):
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
    if n_impf_set > 1:
        iud['IL'] = sp.stats.randint(0, n_impf_set)
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
    if CO is not None:
        for haz_type in meas_set_tmp.get_hazard_types():
            for meas in meas_set_tmp.get_measure(haz_type=haz_type):
                meas.cost *= CO
    return meas_set_tmp

def _meas_set_unc_dict(bounds_cost):
    cmin, cdelta = bounds_cost[0], bounds_cost[1] - bounds_cost[0]
    return {'CO': sp.stats.uniform(cmin, cdelta)}

def _ent_unc_func(EN, ET, EL, IFi, IL, MDD, PAA, CO, DR, bounds_noise,
                 impf_set_list, haz_id_dict, disc_rate, exp_list, meas_set):

    exposures = _exp_uncfunc(EN, ET, EL, exp_list, bounds_noise)
    impact_func_set = _impfset_uncfunc(IFi, MDD, PAA, IL, impf_set_list=impf_set_list,
                                            haz_id_dict=haz_id_dict)
    measure_set = _meas_set_uncfunc(CO, meas_set=meas_set)
    disc_rates = _disc_uncfunc(DR, disc_rate)
    return Entity(exposures, disc_rates, impact_func_set, measure_set)

def _ent_unc_dict(bounds_totval, bounds_noise, bounds_impfi, bounds_mdd,
                  bounds_paa, n_impf_set, bounds_disc, bounds_cost, n_exp):
    ent_unc_dict = _exp_unc_dict(bounds_totval, bounds_noise, n_exp)
    ent_unc_dict.update(_impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa, n_impf_set))
    ent_unc_dict.update(_disc_unc_dict(bounds_disc))
    ent_unc_dict.update(_meas_set_unc_dict(bounds_cost))
    return  ent_unc_dict

def _entfut_unc_func(EN, EG, EL, IFi, IL, MDD, PAA, CO, bounds_noise,
                 impf_set_list, haz_id_dict, exp_list, meas_set):
    exposures = _exp_uncfunc(EN=EN, ET=EG, EL=EL, exp_list=exp_list, bounds_noise=bounds_noise)
    impact_funcs = _impfset_uncfunc(IFi, MDD, PAA, IL, impf_set_list=impf_set_list,
                                            haz_id_dict=haz_id_dict)
    measures = _meas_set_uncfunc(CO, meas_set=meas_set)
    disc_rates = DiscRates() #Disc rate of future entity ignored in cost_benefit.calc()
    return Entity(exposures, disc_rates, impact_funcs, measures)

def _entfut_unc_dict(bounds_impfi, bounds_mdd,
                  bounds_paa, n_impf_set, bounds_eg, bounds_noise,
                  bounds_cost, n_exp):
    eud = {}
    if bounds_eg is not None:
        gmin, gmax = bounds_eg[0], bounds_eg[1] - bounds_eg[0]
        eud['EG'] = sp.stats.uniform(gmin, gmax)
    if bounds_noise is not None:
        eud['EN'] = sp.stats.randint(0, 2**32 - 1) #seed for rnd generator
    if n_exp > 1:
        eud['EL'] = sp.stats.randint(0, n_exp)
    eud.update(_impfset_unc_dict(bounds_impfi, bounds_mdd, bounds_paa, n_impf_set))
    if bounds_cost is not None:
        eud.update(_meas_set_unc_dict(bounds_cost))
    return eud
