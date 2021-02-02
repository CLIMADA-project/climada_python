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

import logging
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from climada.engine import Impact
from climada.engine.cost_benefit import CostBenefit
from climada.util.value_representation import value_to_monetary_unit as vtm

LOGGER = logging.getLogger(__name__)

# Future planed features:
# - Add 'efc' (frequency curve) to UncCostBenenfit
# - Make the Robustness class for generic noise robustness tests
# - Add estimate of computation time and suggest surrogate models if needed


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
        fig, ax: matplotlib.pyplot.figure, matplotlib.pyplot.axes
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

    def __init__(self, unc_vars, params=None, problem=None,
                 metrics=None, pool=None):
        """
        Initialize Uncertainty

        Parameters
        ----------
        unc_vars : list of climade.engine.uncertainty.UncVar variables
            list of uncertainty variables
        params : pd.DataFrame, optional
            DataFrame of sampled parameter values.
            The default is pd.DataFrame().
        problem : dict, optional
            The description of the uncertainty variables and their
            distribution as used in SALib.
            https://salib.readthedocs.io/en/latest/getting-started.html.
            The default is {}.
        metrics : dict(), optional
            Dictionnary of the metrics evaluation. The default is {}.
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.
        """

        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        if params is None:
            params = pd.DataFrame()

        if problem is None:
            problem = {}

        if metrics is None:
            metrics = {}

        self.unc_vars = unc_vars
        self.params = params
        self.problem = problem
        self.metrics = metrics


    @property
    def n_runs(self):
        """
        The effective number of runs needed for the sample size self.n_samples.

        Returns
        -------
        n_runs: int
            effective number of runs

        """

        if isinstance(self.params, pd.DataFrame):
            return self.params.shape[0]
        return 0

    @property
    def param_labels(self):
        """
        Labels of all uncertainty parameters.

        Returns
        -------
        param_labels: list of strings
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
        sampling_method: string
            The sampling method as defined in SALib. Possible choices:
            'saltelli', 'fast_sampler', 'latin', 'morris', 'dgsm', 'ff'
            https://salib.readthedocs.io/en/latest/api.html
            The default is 'saltelli'
        kwargs:
            Keyword arguments are passed to the SALib sampling method.

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
            Keyword arguments are passed to the SALib sample method.

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
        #To import a submodule from a module use 'from_list' necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        salib_sampling_method = getattr(
            __import__('SALib.sample',
                       fromlist=[self.sampling_method]
                       ),
            self.sampling_method
            )
        sample_params = salib_sampling_method.sample(problem = problem,
                                                     N = self.n_samples,
                                                     **kwargs)
        return sample_params


    def est_comp_time(self):
        """
        Estimate the computation time
        """
        raise NotImplementedError()


    def _calc_metric_sensitivity(self, analysis_method, **kwargs):
        """
        Compute the sensitivity indices using SALib

        Parameters
        ----------
        analysis_method : str
            sensitivity analysis method from SALib.analyse
            https://salib.readthedocs.io/en/latest/api.html
            Possible choices:
                'fast', 'rbd_fact', 'morris', 'sobol', 'delta', 'ff'
            The default is 'sobol'.
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


    def plot_metric_distribution(self, metric_list=None):
        """
        Plot the distribution of values.

        Parameters
        ----------
        metric_list: list
            List of metrics to plot the distribution.
            The default is None.

        Raises
        ------
        ValueError
            If no metric distribution was computed the plot cannot be made.

        Returns
        -------
        fig, axes: matplotlib.pyplot.figure, matplotlib.pyplot.axes
            The figure and axis handle of the plot.

        """

        if not self.metrics:
            raise ValueError("No uncertainty data present for this emtrics. "+
                    "Please run an uncertainty analysis first.")

        if metric_list is None:
            metric_list = list(self.metrics.keys())

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
    """
    Impact uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() object.

    """

    def __init__(self, exp_unc, impf_unc, haz_unc, pool=None):
        """Initialize UncImpact

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

        """


        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None


        unc_vars = {'exp': self._var_or_uncvar(exp_unc),
                    'impf': self._var_or_uncvar(impf_unc),
                    'haz': self._var_or_uncvar(haz_unc),
                    }
        params = pd.DataFrame()
        problem = {}
        metrics = {'aai_agg': pd.DataFrame([]),
                   'freq_curve': pd.DataFrame([]),
                   'eai_exp': pd.DataFrame([]),
                   'at_event':  pd.DataFrame([])
                   }

        Uncertainty.__init__(self, unc_vars=unc_vars, pool=pool,
                             params=params, problem=problem, metrics=metrics)


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
            Return periods in years to be computed.
            The default is [5, 10, 20, 50, 100, 250].
        calc_eai_exp : boolean, optional
            Toggle computation of the impact at each centroid location.
            The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.

        Raises
        ------
        ValueError:
            If no sampling parameters defined, the distribution cannot
            be computed.

        """

        if self.params.empty:
            raise ValueError("No sample was found. Please create one first"
                             "using UncImpact.make_sample(N)")

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
            eai_exp (np.array([]) if self.calc_eai_exp=False) and at_event
            (np.array([]) if self.calc_at_event=False).

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
            Sensitivity analysis method from SALib.analyse.
            Possible choices:
                'fast', 'rbd_fact', 'morris', 'sobol', 'delta', 'ff'
            Note that SALib recommends pairs of sampling anad analysis
            algorithms. We recommend users to respect these pairings.
            https://salib.readthedocs.io/en/latest/api.html
            The Default is 'sobol'.
            Note that for the default 'sobol', negative sensitivity
            indices indicate that the algorithm has not converged. In this
            case, please restart the uncertainty and sensitivity analysis
            with an increased number of samples.
        **kwargs : keyword arguments
            These parameters are passed to the chosen SALib.analyze routine.

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

        #To import a submodule from a module use 'from_list' necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        analysis_method = getattr(
            __import__('SALib.analyze',
                       fromlist=[method]
                       ),
            method
            )


        sensitivity_analysis = self._calc_metric_sensitivity(analysis_method, **kwargs)
        self.sensitivity = sensitivity_analysis

        return sensitivity_analysis


class UncCostBenefit(Uncertainty):
    """
    Cost Benefit Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.costbenefit.CostBenefit() object.

    """

    def __init__(self, haz_unc, ent_unc, haz_fut_unc=None, ent_fut_unc=None,
                 pool=None):
        """Initialize UncCostBenefit

        Parameters
        ----------
        haz_unc : climada.engine.uncertainty.UncVar
                  or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard for the present Hazard
            in climada.engine.CostBenefit.calc
        ent_unc : climada.engine.uncertainty.UncVar
                  or climada.entity.Entity
            Entity uncertainty variable or Entity for the future Entity
            in climada.engine.CostBenefit.calc
        haz_unc_fut: climada.engine.uncertainty.UncVar
                     or climada.hazard.Hazard, optional
            Hazard uncertainty variable or Hazard for the future Hazard
            in climada.engine.CostBenefit.calc
            The Default is None.
        ent_fut_unc : climada.engine.uncertainty.UncVar
                      or climada.entity.Entity, optional
            Entity uncertainty variable or Entity for the future Entity
            in climada.engine.CostBenefit.calc
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.

        """


        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        unc_vars = {'haz': self._var_or_uncvar(haz_unc),
                         'ent': self._var_or_uncvar(ent_unc),
                         'haz_fut': self._var_or_uncvar(haz_fut_unc),
                         'ent_fut': self._var_or_uncvar(ent_fut_unc)
                         }

        params = pd.DataFrame()
        problem = {}
        metrics =  {'tot_climate_risk': None,
                    'benefit': None,
                    'cost_ben_ratio': None,
                    'imp_meas_present': None,
                    'imp_meas_future': None}

        Uncertainty.__init__(self, unc_vars=unc_vars, pool=pool,
                             params=params, problem=problem, metrics=metrics)


    def calc_cost_benefit_distribution(self, **kwargs):
        """
        Computes the cost benefit for each of the parameters set defined in
        uncertainty.params.

        By default, imp_meas_present, imp_meas_future, tot_climate_risk,
        benefit, cost_ben_ratio are computed.

        Parameters
        ----------
        **kwargs : keyword arguments
            These parameters are passed to
            climada.engine.CostBenefit.calc().

        Returns
        -------
        None.

        """

        if self.params.empty:
            LOGGER.info("No sample was found. Please create one first"
                        "using UncImpact.make_sample(N)")
            return None

        #Compute impact distributions
        if self.pool:
            chunksize = min(self.n_runs // self.pool.ncpus, 100)
            cb_metrics = self.pool.map(partial(self._map_costben_eval, **kwargs),
                                           self.params.iterrows(),
                                           chunsize = chunksize)

        else:
            cb_metrics = map(partial(self._map_costben_eval, **kwargs),
                             self.params.iterrows())

        [imp_meas_present,
         imp_meas_future,
         tot_climate_risk,
         benefit,
         cost_ben_ratio] = list(zip(*cb_metrics)) #Transpose list of list

        # Assign computed impact distribution data to self
        self.metrics['tot_climate_risk'] = \
            pd.DataFrame(tot_climate_risk, columns = ['tot_climate_risk'])

        self.metrics['benefit'] = pd.DataFrame(benefit)
        self.metrics['cost_ben_ratio'] = pd.DataFrame(cost_ben_ratio)


        imp_metric_names = ['risk', 'risk_transf', 'cost_meas',
                            'cost_ins']

        for imp_meas, name in zip([imp_meas_present, imp_meas_future],
                                  ['imp_meas_present', 'imp_meas_future']):
            df_imp_meas = pd.DataFrame()
            if imp_meas[0]:
                for imp in imp_meas:
                    met_dic = {}
                    for meas, imp_dic in imp.items():
                        metrics = [imp_dic['risk'],
                                   imp_dic['risk_transf'],
                                   *imp_dic['cost']]
                        dic_tmp = {meas + '-' + m_name: [m_value]
                                   for m_name, m_value
                                   in zip(imp_metric_names, metrics)
                                    }
                        met_dic.update(dic_tmp)
                    df_imp_meas = df_imp_meas.append(pd.DataFrame(met_dic))
            self.metrics[name] = df_imp_meas

        LOGGER.info("Currently the freq_curve is not saved. Please " +
                    "change the risk_func if return period information " +
                    "needed")

        return None


    def _map_costben_eval(self, param_sample, **kwargs):
        """
        Map to compute cost benefit for all parameter samples in parallel

        Parameters
        ----------
        param_sample : pd.DataFrame.iterrows()
            Generator of the parameter samples

        Returns
        -------
         : list
            icost benefit metrics list for all samples containing
            imp_meas_present, imp_meas_future, tot_climate_risk,
            benefit, cost_ben_ratio

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



    def calc_cost_benefit_sensitivity(self,  method='sobol', **kwargs):
        """
        Compute the sensitivity indices using the SALib library:
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis

        Simple introduction to default Sobol sensitivity
        https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis

        Parameters
        ----------
        method : string, optional
            Sensitivity analysis method from SALib.analyse.
            Possible choices:
                'fast', 'rbd_fact', 'morris', 'sobol', 'delta', 'ff'
            Note that SALib recommends pairs of sampling anad analysis
            algorithms. We recommend users to respect these pairings.
            https://salib.readthedocs.io/en/latest/api.html
            The Default is 'sobol'.
            Note that for the default 'sobol', negative sensitivity
            indices indicate that the algorithm has not converged. In this
            case, please restart the uncertainty and sensitivity analysis
            with an increased number of samples.
        **kwargs : keyword arguments
            These parameters are passed to the chosen SALib.analyze routine.

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

        #To import a submodule from a module 'from_list' is necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        analysis_method = getattr(
            __import__('SALib.analyze',
                       fromlist=[method]
                       ),
            method
            )


        sensitivity_analysis = self._calc_metric_sensitivity(analysis_method, **kwargs)
        self.sensitivity = sensitivity_analysis

        return sensitivity_analysis


class UncRobustness():
    """
    Compute variance from multiplicative Gaussian noise
    """

    def __init__(self):
        raise NotImplementedError()
        