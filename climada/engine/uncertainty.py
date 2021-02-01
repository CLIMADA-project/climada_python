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

Define Uncertainty class.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from functools import partial

import SALib.sample as salibs
import SALib.analyze as saliba

from climada.engine import Impact
from climada.engine.cost_benefit import CostBenefit, risk_aai_agg
from climada.util.value_representation import value_to_monetary_unit as vtm

LOGGER = logging.getLogger(__name__)


class UncVar():
    """
    Uncertainty variable

    An uncertainty variable requires a single or multi-parameter function.
    The parameters must follow a given distribution.

    Examples
    --------

    Categorical variable function: LitPop exposures with m,n exponents in [0,5]
        import scipy as sp
        def litpop_cat(m, n):
            exp = Litpop()
            exp.set_country('CHE', exponent=[m, n])
            return exp
        distr_dict = {
            m: sp.stats.randint(low=0, high=5),
            n: sp.stats.randint(low=0, high=5)
            }
        unc_var_cat = UncVar(litpop_cat, distr_dict)

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
        unc_var_cont = UncVar(imp_fun_tc, distr_dict)

    """

    def __init__(self, unc_var, distr_dict):
        """
        Initialize UncVar

        Parameters
        ----------
        unc_var : function
            Variable defined as a function of the uncertainty parameters
        distr_dict : dict
            Dictionary of the probability density distributions of the
            uncertainty parameters, with keys the matching the keyword
            arguments (i.e. uncertainty parameters) of the unc_var function.
            The distribution must be of type scipy.stats
            https://docs.scipy.org/doc/scipy/reference/stats.html

        Returns
        -------
        None.

        """
        self.labels = list(distr_dict.keys())
        self.distr_dict = distr_dict
        self.unc_var = unc_var

    def plot_distr(self):
        """
        Plot the distributions of the parameters of the uncertainty variable.

        Returns
        -------
        fig, ax: matplotlib.pyplot.fig, matplotlib.pyplot.ax
            The figure and axis handle of the plot.

        """
        nplots = len(self.distr_dict)
        nrows, ncols = int(nplots / 3) + 1, min(nplots, 3)
        fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for ax, (param_name, distr) in zip(axis.flatten(), self.distr_dict.items()):
            x = np.linspace(distr.ppf(0.001), distr.ppf(0.999), 100)
            ax.plot(x, distr.pdf(x), label=param_name)
            ax.legend()
        return fig, axis

    def evaluate(self, kwargs):
        """
        Evaluate the uncertainty variable.

        Parameters
        ----------
        kwargs :
            These parameters will be passed to self.unc_var.
            They must be the input parameters of the uncertainty variable .

        Returns
        -------

            Evaluated uncertainty variable

        """
        return self.unc_var(**kwargs)


class Uncertainty():
    """
    Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() or climada.engine.costbenefit.CostBenefit()
    object.

    """

    def __init__(self, unc_vars=None, pool=None):
        """Initialize Unc

        Parameters
        ----------
        unc_vars : list of uncertainty variables of type
            climade.engine.uncertainty.UncVar
        pool : pathos.pools.ProcessPool
            Pool of CPUs for parralel computations. Default is None.
        Returns
        -------
        None.

        """
        
        if unc_vars:
            self.unc_vars = {}

        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        self.params = pd.DataFrame()
        self.problem = {}
        self.metrics = {}


    @property
    def n_runs(self):
        """
        The effective number of runs needed for the sample size self.n_samples.

        Returns
        -------
        int
            effective number of runs

        """

        if isinstance(self.params, pd.DataFrame):
            return self.params.shape[0]
        else:
            return 0
        
    @property
    def param_labels(self):
        """
        Labels of all uncertainty parameters.

        Returns
        -------
        list of strings
            Labels of all uncertainty parameters.

        """
        return list(self.distr_dict.keys())


    @property
    def distr_dict(self):
        """
        Dictionary of the distribution of all the parameters of all variables
        listed in self.unc_vars

        Returns
        -------
        distr_dict : dict( sp.stats objects )
            Dictionary of all distributions.

        """

        distr_dict = dict()
        for unc_var in self.unc_vars.values():
            distr_dict.update(unc_var.distr_dict)
        return distr_dict
   
    
    def make_sample(self, N, sampling_method='saltelli', **kwargs):
        """
        Make a sample for all parameters with their respective
        distributions using the chosen sampling_method from SALib.
        https://salib.readthedocs.io/en/latest/api.html
 
        Parameters
        ----------
        N : int
            Number of samples as defined in SALib.sample.saltelli.sample().
        kwargs: 
            Keyword arguments will be passed to the SALib sample method.
        Returns
        -------
        None.

        """
        self.sampling_method = sampling_method
        self.n_samples = N
        uniform_base_sample = self._make_uniform_base_sample(**kwargs)
        df_params = pd.DataFrame(uniform_base_sample, columns=self.param_labels)
        for param in list(df_params):
            df_params[param] = df_params[param].apply(
                self.distr_dict[param].ppf
                )
        self.params = df_params
        
        
    def _make_uniform_base_sample(self, **kwargs):
        """
        Make a uniform distributed [0,1] sample for the defined 
        uncertainty parameters (self.param_labels) with the chosen
        method from SALib (self.sampling_method)
        https://salib.readthedocs.io/en/latest/api.html

        Parameters
        ----------
        kwargs: 
            Keyword arguments will be passed to the SALib sample method.

        Returns
        -------
        sample_params : np.matrix
            Returns a NumPy matrix containing the sampled uncertainty
            parameters using the defined sampling method (self.sampling_method)

        """
        
        problem = {
            'num_vars' : len(self.param_labels),
            'names' : self.param_labels,
            'bounds' : [[0, 1]]*len(self.param_labels)
            }
        self.problem = problem
        salib_sampling_method = getattr(salibs, self.sampling_method)
        sample_params = salib_sampling_method.sample(problem = problem,
                                                     N = self.n_samples,
                                                     **kwargs)
        return sample_params


    def est_comp_time(self):
        """
        Estimate the computation time

        Returns
        -------
        None.

        """
        raise NotImplementedError()
    
        
    def _calc_metric_sensitivity(self, analysis_method, **kwargs):
        """
        Compute the sensitivity indices using SALib

        Parameters
        ----------
        analysis_method : str
            sensitivity analysis method (as named in SALib.analyse)
        **kwargs : keyword arguments
            Are passed to the chose SALib analyse method.

        Returns
        -------
        sensitivity_dict : dict
            Dictionnary of the sensitivity indices. Keys are the 
            metrics names, values the sensitivity indices dictionnary
            as returned by SALib.

        """
        
        sensitivity_dict = {}
        for name, df_metric in self.metrics.items():
            sensitivity_dict[name] = {}
            for metric in df_metric:
                Y = df_metric[metric].to_numpy()
                sensitivity_index = analysis_method.analyze(self.problem, Y, **kwargs)
                sensitivity_dict[name].update({metric: sensitivity_index})
                
        return sensitivity_dict
    
    
    @staticmethod
    def _var_or_uncvar(var):
        """
        Returns uncertainty variable with no distribution if var is not
        an UncVar. Else, returns var.

        Parameters
        ----------
        var : Object or else

        Returns
        -------
        UncVar
            var if var is UncVar, else UncVar with var and no distribution.

        """
        
        if isinstance(var, UncVar):
            return var
        
        return UncVar(unc_var=lambda: var, distr_dict={})
    
    
    def plot_metric_distribution(self, metric_list):
        """
        Plot the distribution of values.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        fig : TYPE
            DESCRIPTION.
        axes : TYPE
            DESCRIPTION.

        """
        
        if not self.metrics:
            raise ValueError("No uncertainty data present for this emtrics. "+
                    "Please run an uncertainty analysis first.")
            
        df_values = pd.DataFrame()
        for metric in metric_list:
            df_values = df_values.append(self.metrics[metric])
            
        df_values_log = df_values.apply(np.log10).copy()
        df_values_log = df_values_log.replace([np.inf, -np.inf], np.nan)
        cols = df_values_log.columns
        nplots = len(cols)
        nrows, ncols = int(nplots / 3) + 1, min(nplots, 3)
        fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize=(nrows*7, ncols * 3.5),
                                 sharex=True,
                                 sharey=True)    
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]

        for ax, col in zip(flat_axes, cols):
            data = df_values_log[col]
            data.hist(ax=ax,  bins=100, density=True, histtype='step')
            avg = df_values[col].mean()
            std = df_values[col].std()
            ax.plot([np.log10(avg), np.log10(avg)], [0, 1],
                    color='red', linestyle='dashed',
                    label="avg=%.2f%s" %vtm(avg))
            ax.plot([np.log10(avg) - np.log10(std) / 2,
                     np.log10(avg) + np.log10(std) / 2],
                    [0.3, 0.3], color='red',
                    label="std=%.2f%s" %vtm(std))
            ax.set_title(col)
            ax.set_xlabel('value [log10]')
            ax.set_ylabel('density of events')
            ax.legend()

        return fig, axes

    
    
    
class UncImpact(Uncertainty):
    
    def __init__(self, exp_unc, impf_unc, haz_unc, pool=None):
        """Initialize Unc

        Parameters
        ----------
        exp_unc : climada.engine.uncertainty.UncVar or climada.entity.Exposure
            Exposure uncertainty variable or Exposure
        impf_unc : climada.engine.uncertainty.UncVar or climada.entity.ImpactFuncSet
            Impactfunction uncertainty variable or Impact function
        haz_unc : climada.engine.uncertainty.UncVar or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard
        pool : pathos.pools.ProcessPool
            Pool of CPUs for parralel computations. Default is None.

        Returns
        -------
        None.

        """

        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None
            
        self.unc_vars = {'exp': self._var_or_uncvar(exp_unc),
                         'impf': self._var_or_uncvar(impf_unc),
                         'haz': self._var_or_uncvar(haz_unc),
                         }

        self.params = pd.DataFrame()
        self.problem = {}
        self.metrics = {'aai_agg': pd.DataFrame([]),
                        'freq_curve': pd.DataFrame([]),
                        'eai_exp': pd.DataFrame([]),
                        'at_event':  pd.DataFrame([])}
    
    
    def calc_impact_distribution(self,
                             rp=None,
                             calc_eai_exp=False,
                             calc_at_event=False,
                             ):
        """
        Computes the impact for each of the parameters set defined in
        uncertainty.params.
    
        By default, the aggregated average annual impact
        (impact.aai_agg) and the excees impact at return periods rp 
        (imppact.calc_freq_curve(self.rp).impact) is computed.
        Optionally, eai_exp and at_event is computed (this may require 
        a larger amount of memory if n_samples and/or the number of centroids
        is large).
    
        Parameters
        ----------
        rp : list(int), optional
            Return period in years to be computed.
            The default is [5, 10, 20, 50, 100, 250].
        calc_eai_exp : boolean, optional
            Toggle computation of the impact at each centroid location.
            The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.
    
        Returns
        -------
        None.
    
        """

        if rp is None:
            rp=[5, 10, 20, 50, 100, 250]
    
        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event
    
        #Compute impact distributions
        if self.pool:
            chunksize = min(self.n_runs // self.pool.ncpus, 100)
            imp_metrics = self.pool.map(self._map_impact_eval,
                                           self.params.iterrows(),
                                           chunsize = chunksize)
    
        else:
            imp_metrics = map(self._map_impact_eval, self.params.iterrows())
                       
        [aai_agg_list, freq_curve_list,
         eai_exp_list, at_event_list] = list(zip(*imp_metrics))

        # Assign computed impact distribution data to self
        self.metrics['aai_agg']  = pd.DataFrame(aai_agg_list,
                                                columns = ['aai_agg'])
        
        self.metrics['freq_curve'] = pd.DataFrame(freq_curve_list,
                                    columns=['rp' + str(n) for n in rp])
        self.metrics['eai_exp'] =  pd.DataFrame(eai_exp_list)
        self.metrics['at_event'] = pd.DataFrame(at_event_list)


    def _map_impact_eval(self, param_sample):
        """
        Map to compute impact for all parameter samples in parrallel

        Parameters
        ----------
        param_sample : pd.DataFrame.iterrows()
            Generator of the parameter samples

        Returns
        -------
         : list
            impact metrics list for all samples containing aai_agg, rp_curve,
            eai_exp and at_event.

        """

        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        exp_params = param_sample[1][self.unc_vars['exp'].labels].to_dict()
        haz_params = param_sample[1][self.unc_vars['haz'].labels].to_dict()
        impf_params = param_sample[1][self.unc_vars['impf'].labels].to_dict()

        exp = self.unc_vars['exp'].evaluate(exp_params)
        haz = self.unc_vars['haz'].evaluate(haz_params)
        impf = self.unc_vars['impf'].evaluate(impf_params)

        imp = Impact()
        imp.calc(exposures=exp, impact_funcs=impf, hazard=haz)
        
        
        # Extract from climada.impact the chosen metrics
        freq_curve = imp.calc_freq_curve(self.rp).impact
        
        if self.calc_eai_exp:
            eai_exp = imp.eai_exp
        else:
            eai_exp = np.array([])
            
        if self.calc_at_event:
            at_event= imp.at_event
        else:
            at_event = np.array([])

        return [imp.aai_agg, freq_curve, eai_exp, at_event]
    
    
    def calc_impact_sensitivity(self, method='sobol', **kwargs):
        """
        Compute the sensitivity indices using the SALib library:
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    
        Simple introduction to default Sobol sensitivity
        https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis
    
        Parameters
        ----------
        method : string, optional
            Choose the method for the sensitivity analysis. Note that
            SALib recommends pairs of sampling anad analysis algorithms.
            We recommend users to respect these pairings. 
            Defaul: 'sobol' 
            Note that for the default 'sobol', negative sensitivity
            indices indicate that the algorithm has not converged. In this
            case, please restart the uncertainty and sensitivity analysis
            with an increased number of samples.
        **kwargs :
            These parameters will be passed to chosen SALib.analyze routine.
    
        Returns
        -------
        sensitivity_analysis : dict
            Dictionary with keys the uncertainty parameter labels.
            For each uncertainty parameter, the item is another dictionary
            with keys the sensitivity indices (the direct ouput from
            the chosen SALib.analyse method)  
        """
        
        if self.params.empty:
            raise ValueError("I found no samples. Please produce first"
                             " samples using Uncertainty.make_sample().")
            
        if not self.metrics:
            raise ValueError("I found no impact data. Please compute"
                             " the impact distribution first using"+
                             " UncImpact.calc_impact_distribution()")
                              
        analysis_method = getattr(saliba, method)
    
        
        sensitivity_analysis = self._calc_metric_sensitivity(analysis_method, **kwargs)
        self.sensitivity = sensitivity_analysis
    
        return sensitivity_analysis
    
    
class UncCostBenefit(Uncertainty):
    
    def __init__(self, haz_unc, ent_unc, haz_fut_unc=None, ent_fut_unc=None,
                 future_year=None, risk_func=risk_aai_agg, pool=None):

         if pool:
             self.pool = pool
             LOGGER.info('Using %s CPUs.', self.pool.ncpus)
         else:
             self.pool = None
             
         self.unc_vars = {'haz': self._var_or_uncvar(haz_unc),
                          'ent': self._var_or_uncvar(ent_unc),
                          'haz_fut': self._var_or_uncvar(haz_fut_unc),
                          'ent_fut': self._var_or_uncvar(ent_fut_unc)
                          }
    
         self.params = pd.DataFrame()
         self.problem = {}
         self.metrics = {'tot_climate_risk': None,
                         'benefit': None,
                         'cost_ben_ratio': None,
                         'imp_meas_present': None,
                         'imp_meas_future': None}
        
    
    
    def calc_cost_benefit_distribution(self, **kwargs):
   
        #Compute impact distributions
        if self.pool:
            chunksize = min(self.n_runs // self.pool.ncpus, 100)
            cb_metrics = self.pool.map(partial(self._map_costben_eval, **kwargs),
                                           self.params.iterrows(),
                                           chunsize = chunksize)
    
        else:
            cb_metrics = map(self._map_costben_eval, self.params.iterrows())
                       
        [imp_meas_present, imp_meas_future,
         tot_climate_risk, benefit, cost_ben_ratio] = cb_metrics

        # Assign computed impact distribution data to self
        
        self.metrics['tot_climate_risk'] = \
            pd.DataFrame(tot_climate_risk, columns = ['tot_climate_risk'])
            
        df_ben = pd.DataFrame()
        for ben_dict in benefit:
            df_ben.append(pd.DataFrame(ben_dict))
        self.metrics['benefit'] = df_ben
        
        df_costbenratio = pd.DataFrame()
        for costbenratio_dict in cost_ben_ratio:
            df_costbenratio.append(pd.DataFrame(costbenratio_dict))
        self.metrics['cost_ben_ratio'] = df_costbenratio
        
        df_imp_meas_pres = pd.DataFrame()
        for impd in imp_meas_present:
            risk = impd['risk']
            risk_transf = impd['risk_transf']
            cost_meas, cost_ins = impd['cost']
            freq_curve = impd['efc']
            df_tmp = pd.DataFrame(risk, risk_transf, cost_meas, cost_ins, *freq_curve,
                              columns = ['risk', 'risk_transf', 'cost_meas',
                                         'cost_ins', *impd['efc'].return_per]
                              )
            df_imp_meas_pres.append(df_tmp)
        self.metrics['imp_meas_present'] = df_imp_meas_pres
        
        df_imp_meas_fut = pd.DataFrame()
        for impd in imp_meas_future:
            risk = impd['risk']
            risk_transf = impd['risk_transf']
            cost_meas, cost_ins = impd['cost']
            freq_curve = impd['efc']
            df_tmp = pd.DataFrame(risk, risk_transf, cost_meas, cost_ins, *freq_curve,
                              columns = ['risk', 'risk_transf', 'cost_meas',
                                         'cost_ins', *impd['efc'].return_per]
                              )
            df_imp_meas_fut.append(df_tmp)
        self.metrics['imp_meas_future'] = df_imp_meas_fut


    def _map_costben_eval(self, param_sample, **kwargs):
        """
        Map to compute impact for all parameter samples in parrallel

        Parameters
        ----------
        param_sample : pd.DataFrame.iterrows()
            Generator of the parameter samples

        Returns
        -------


        """

        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        haz_params = param_sample[1][self.unc_vars['haz'].labels].to_dict()
        ent_params = param_sample[1][self.unc_vars['ent'].labels].to_dict()
        haz_fut_params = param_sample[1][self.unc_vars['haz_fut'].labels].to_dict()
        ent_fut_params = param_sample[1][self.unc_vars['ent_fut'].labels].to_dict()

        haz = self.unc_vars['haz'].evaluate(haz_params)
        ent = self.unc_vars['ent'].evaluate(ent_params)
        haz_fut = self.unc_vars['haz_fut'].evaluate(haz_fut_params)
        ent_fut = self.unc_vars['ent_fut'].evaluate(ent_fut_params)

        cb = CostBenefit()
        cb.calc(hazard=haz, entity=ent, haz_future=haz_fut, ent_future=ent_fut,
                save_imp=False, **kwargs)
        
        # Extract from climada.impact the chosen metrics
        return  [cb.imp_meas_present,
                 cb.imp_meas_future,
                 cb.tot_climate_risk,
                 cb.benefit,
                 cb.cost_ben_ratio
                 ]

        
    def calc_cost_benefit_sensitivity(self,  method, **kwargs):
        
        if self.params.empty:
            raise ValueError("I found no samples. Please produce first"
                             " samples using Uncertainty.make_sample().")
            
        if not self.metrics:
            raise ValueError("I found no impact data. Please compute"
                             " the impact distribution first using"+
                             " UncImpact.calc_impact_distribution()")
                              
        analysis_method = getattr(saliba, method)
    
        
        sensitivity_analysis = self._calc_metric_sensitivity(analysis_method, **kwargs)
        self.sensitivity = sensitivity_analysis
    
        return sensitivity_analysis


class UncRobustness():
    """
    Compute variance from multiplicative Gaussian noise
    """
    
    def __init__(self):
        raise NotImplementedError()
