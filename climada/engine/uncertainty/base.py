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

__all__ = ['UncVar', 'Uncertainty']

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest

from climada.util.value_representation import value_to_monetary_unit as u_vtm
from climada.util.value_representation import sig_dig as u_sig_dig

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
            'm': sp.stats.randint(low=0, high=5),
            'n': sp.stats.randint(low=0, high=5)
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
            uncertainty parameters, with keys matching the keyword
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


    def plot(self):
        """
        Plot the distributions of the parameters of the uncertainty variable.

        Returns
        -------
        fig, axes: matplotlib.pyplot.figure, matplotlib.pyplot.axes
            The figure and axis handle of the plot.

        """

        nplots = len(self.distr_dict)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
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
            x = np.linspace(distr.ppf(0.001), distr.ppf(0.999), 100)
            ax.plot(x, distr.pdf(x), label=param_name)
            ax.legend()
        return fig, axes


    def evaluate(self, uncvar_kwargs):
        """
        Evaluate the uncertainty variable.

        Parameters
        ----------
        uncvar_kwargs : dict
            These parameters will be passed to self.unc_var.
            They must be the input parameters of the uncertainty variable .

        Returns
        -------

            Evaluated uncertainty variable

        """
        return self.unc_var(**uncvar_kwargs)
    
    
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

        return UncVar(unc_var=lambda: var, distr_dict={})



class Uncertainty():
    """
    Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() or climada.engine.costbenefit.CostBenefit()
    object.

    """

    def __init__(self, unc_vars, samples=None, metrics=None,
                 sensitivity=None):
        """
        Initialize Uncertainty

        Parameters
        ----------
        unc_vars : dict
            keys are names and values are climade.engine.uncertainty.UncVar 
        samples : pd.DataFrame, optional
            DataFrame of sampled parameter values. Column names must be
            parameter names (all labels) from all unc_vars.
            The default is pd.DataFrame().
        metrics : dict(), optional
            Dictionnary of the CLIMADA metrics (outputs from 
            Impact.calc() or CostBenefits.calcl()) for which the uncertainty
            distribution (and optionally the sensitivity) will be computed. 
            For each sample (row of samples), each metric must have a definite
            value.             
            Metrics are named directly after their defining attributes:
                Impact: ['aai_agg', 'freq_curve', 'eai_exp', 'at_event']
                CostBenefits: ['tot_climate_risk', 'benefit', 'cost_ben_ratio',
                               'imp_meas_present', 'imp_meas_future']
            Keys are metric names and values are pd.DataFrame with values 
            for each parameter sample (one row per sample).            
            The default is {}.
        sensitivity: dict(), optional
            Dictionnary of the sensitivity analysis for each uncertainty
            parameter.
            The default is {}.
        """
    
        self.unc_vars = unc_vars if unc_vars else {}
        self.samples_df = samples if samples is not None else pd.DataFrame(
            columns = self.param_labels)
        self.sampling_method = None
        self.metrics = metrics if metrics is not None else {}
        self.sensitivity = sensitivity if sensitivity is not None else {}
        self.check()
    
    
    def check(self):
        """
        Check if the data variables are consistent

        Returns
        -------
        check: boolean
            True if data is consistent.

        """
        check = True
        check &= (self.param_labels == self.samples_df.columns.to_list())
        if not check:
            raise ValueError("Parameter names from unc_vars do not "
                             "correspond to parameters names of sample")
        for metric, df_distr in self.metrics.items():
            if not df_distr.empty:
                check &= (len(df_distr) == self.n_samples)
                if not check:
                    raise ValueError(f"Metric f{metric} has less values than"
                             " the number of samples {self.n_samples}")
                
                if df_distr.isnull().values.any():
                    LOGGER.warning("At least one metric evaluated to Nan for " 
                        "one cominbation of uncertainty parameters containend " 
                        "in sample. Note that the sensitivity analysis will " 
                        "then return Nan. " 
                        "See https://github.com/SALib/SALib/issues/237")
        return check
        
    
    @property
    def n_samples(self):
        """
        The effective number of samples

        Returns
        -------
        n_samples: int
            effective number of samples

        """

        if isinstance(self.samples_df, pd.DataFrame):
            return self.samples_df.shape[0]
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
    
    @property
    def problem(self):
        """
        The description of the uncertainty variables and their
        distribution as used in SALib.
        https://salib.readthedocs.io/en/latest/getting-started.html.

        Returns
        -------
        problem : dict
            Salib problem dictionnary.

        """
        return {
            'num_vars' : len(self.param_labels),
            'names' : self.param_labels,
            'bounds' : [[0, 1]]*len(self.param_labels)
            }


    def make_sample(self, N, sampling_method='saltelli',
                          sampling_kwargs=None):
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
        sampling_kwargs: dict()
            Optional keyword arguments of the chosen SALib sampling method.


        Returns
        -------
        df_samples: pd.DataFrame()
            Dataframe of the generated samples 
            (one row = one sample, columns = uncertainty parameters)
        """
        
        self.samples_df = sampling_method
        uniform_base_sample = self._make_uniform_base_sample(N,
                                                             sampling_method,
                                                             sampling_kwargs)
        df_samples = pd.DataFrame(uniform_base_sample, columns=self.param_labels)
        for param in list(df_samples):
            df_samples[param] = df_samples[param].apply(
                self.distr_dict[param].ppf
                )
        self.samples_df = df_samples
        LOGGER.info("Effective number of made samples: %d" %self.n_samples)
        return df_samples


    def _make_uniform_base_sample(self, N, sampling_method='saltelli',
                                  sampling_kwargs=None):
        """
        Make a uniform distributed [0,1] sample for the defined
        uncertainty parameters (self.param_labels) with the chosen
        method from SALib (self.sampling_method)
        https://salib.readthedocs.io/en/latest/api.html

        Parameters
        ----------
        N:
            Number of samples as defined for the SALib sample method.
            Note that the effective number of created samples might be
            larger (c.f. SALib)
        sampling_method: string
            The sampling method as defined in SALib. Possible choices:
            'saltelli', 'fast_sampler', 'latin', 'morris', 'dgsm', 'ff'
            https://salib.readthedocs.io/en/latest/api.html
            The default is 'saltelli'
        sampling_kwargs: dict()
            Optional keyword arguments of the chosen SALib sampling method.

        Returns
        -------
        sample_uniform : np.matrix
            Returns a NumPy matrix containing the sampled uncertainty
            parameters using the defined sampling method (self.sampling_method)

        """
        
        if sampling_kwargs is None: sampling_kwargs = {}

        #To import a submodule from a module use 'from_list' necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        salib_sampling_method = getattr(
            __import__('SALib.sample',
                       fromlist=[sampling_method]
                       ),
            sampling_method
            )
        sample_uniform = salib_sampling_method.sample(problem = self.problem,
                                                     N = N,
                                                     **sampling_kwargs)
        return sample_uniform


    def est_comp_time(self, time_one_run, pool=None):
        """      
        Estimate the computation time

        Parameters
        ----------
        time_one_run : int/float
            Estimated computation time for one parameter set in seconds
        pool : pathos.pool, optional
            pool that would be used for parallel computation.
            The default is None.

        Returns
        -------
        Estimated computation time in secs.

        """
        time_one_run = u_sig_dig(time_one_run, n_sig_dig=3)
        if time_one_run > 5:
            LOGGER.warning("Computation time for one set of parameters is "+
                "%.2fs. This is suspiciously long. Possible " % time_one_run +
                "Potential reasons: unc_vars are loading data, centroids not"+
                " assigned to exp before defining unc_var, ..." +
                "\n If computation cannot be reduced, consider using" 
                " a surrogate model https://www.uqlab.com/")

        ncpus = pool.ncpus if pool else 1
        total_time = self.n_samples * time_one_run / ncpus
        
        return total_time


    def calc_sensitivity(self, salib_method='sobol', method_kwargs=None):
        """
        Compute the sensitivity indices using SALib. Prior to doing this
        sensitivity analysis, one must compute the distribution of the output
        metrics values (i.e. self.metrics is defined) for all the parameter 
        samples (rows of self.samples_df).
        
        According to Wikipedia, sensitivity analysis is “the study of how the
        uncertainty in the output of a mathematical model or system (numerical
        or otherwise) can be apportioned to different sources of uncertainty
        in its inputs.” The sensitivity of each input is often represented by
        a numeric value, called the sensitivity index. Sensitivity indices
        come in several forms:

        First-order indices: measures the contribution to the output variance
        by a single model input alone.
        Second-order indices: measures the contribution to the output variance
        caused by the interaction of two model inputs.
        Total-order index: measures the contribution to the output variance
        caused by a model input, including both its first-order effects
        (the input varying alone) and all higher-order interactions.


        Parameters
        ----------
        salib_method : str
            sensitivity analysis method from SALib.analyse
            Possible choices:
                'fast', 'rbd_fact', 'morris', 'sobol', 'delta', 'ff'
            The default is 'sobol'.
            Note that in Salib, sampling methods and sensitivity analysis
            methods should be used in specific pairs.
            https://salib.readthedocs.io/en/latest/api.html
        method_kwargs: dict(), optional
            Keyword arguments of the chosen SALib analyse method.
            The default is to use SALib's default arguments.
        Returns
        -------
        sensitivity_dict : dict
            Dictionnary of the sensitivity indices. Keys are the
            metrics names, values the sensitivity indices dictionnary
            as returned by SALib.

        """
        
        check_salib(self.sampling_method, salib_method)
        
        #To import a submodule from a module use 'from_list' necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        method = getattr(
            __import__('SALib.analyze',
                       fromlist=[salib_method]
                       ),
            salib_method
            )
        
        #Certaint Salib method required model input (X) and output (Y), others
        #need only ouput (Y)
        salib_kwargs = method.analyze.__code__.co_varnames
        X = self.samples_df.to_numpy() if 'X' in salib_kwargs else None

        if method_kwargs is None: method_kwargs = {} 
        sensitivity_dict = {}
        for name, df_metric in self.metrics.items():
            sensitivity_dict[name] = {}
            for metric in df_metric:
                Y = df_metric[metric].to_numpy()
                if X is not None:
                    sensitivity_index = method.analyze(self.problem, X, Y,
                                                            **method_kwargs)
                else:
                    sensitivity_index = method.analyze(self.problem, Y,
                                                            **method_kwargs)
                sensitivity_dict[name].update({metric: sensitivity_index})
        
        self.sensitivity = sensitivity_dict

        return sensitivity_dict


    def plot_distribution(self, metric_list=None):
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
            raise ValueError("No uncertainty data present for these metrics. "+
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
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize=(nrows*7, ncols * 3.5),
                                 sharex=True,
                                 sharey=True)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])

        for ax, col in zip_longest(flat_axes, cols, fillvalue=None):
            if col is None:
                ax.remove()
                continue
            data = df_values_log[col]
            if data.empty:
                ax.remove()
                continue
            data.hist(ax=ax,  bins=100, density=True, histtype='step')
            avg = df_values[col].mean()
            std = df_values[col].std()
            ax.plot([np.log10(avg), np.log10(avg)], [0, 1],
                    color='red', linestyle='dashed',
                    label="avg=%.2f%s" %u_vtm(avg))
            ax.plot([np.log10(avg) - np.log10(std) / 2,
                     np.log10(avg) + np.log10(std) / 2],
                    [0.3, 0.3], color='red',
                    label="std=%.2f%s" %u_vtm(std))
            ax.set_title(col)
            ax.set_xlabel('value [log10]')
            ax.set_ylabel('density of events')
            ax.legend()

        return fig, axes


    def plot_sample(self):
        """
        Plot the sample distributions of the uncertainty parameters.
        
        Raises
        ------
        ValueError
            If no sample was computed the plot cannot be made.

        Returns
        -------
        fig, ax: matplotlib.pyplot.figure, matplotlib.pyplot.axes
            The figure and axis handle of the plot.

        """
        
        if self.samples_df.empty:
            raise ValueError("No uncertainty sample present."+
                    "Please make a sample first.")

        nplots = len(self.param_labels)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for ax, label in zip_longest(axis.flatten(), self.param_labels, fillvalue=None):
            if label is None:
                ax.remove()
                continue
            self.samples_df[label].hist(ax=ax, bins=100) 
            ax.set_title(label)
            ax.set_xlabel('value')
            ax.set_ylabel('Sample count')
            
        return fig, axis

   
    def plot_sensitivity(self, salib_si='S1', metric_list=None):
        """
        Plot the first order sensitivity indices of the chosen metric.

        Parameters
        ----------
        salib_si: string, optional
            The first order sensitivity index to plot (see SALib option)
            https://salib.readthedocs.io/en/latest/basics.html
            The default is S1.
            
        metric_list: list of strings, optional
            List of metrics to plot the sensitivity
            The default is ['aai_agg', 'freq_curve', 'tot_climate_risk', 
                           'benefit', 'cost_ben_ratio', 'imp_meas_present',
                           'imp_meas_future', 'tot_value']

        Raises
        ------
        ValueError
            If no sensitivity was computed the plot cannot be made.

        Returns
        -------
        fig, axes: matplotlib.pyplot.figure, matplotlib.pyplot.axes
            The figure and axis handle of the plot.

        """

        if not self.metrics:
            raise ValueError("No sensitivity present for this metrics. "
                    "Please run a sensitivity analysis first.")
            
        if metric_list is None:
            metric_list = ['aai_agg', 'freq_curve', 'tot_climate_risk', 
                           'eai_exp', 'at_event', 'tot_value',
                           'benefit', 'cost_ben_ratio', 'imp_meas_present',
                           'imp_meas_future', 'tot_value']
            metric_list = list(set(metric_list) & set(self.metrics.keys()))
            
            
        nplots = len(metric_list)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize=(nrows*9, ncols * 3.5),
                                 sharex=True,
                                 sharey=True)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])
        
        for ax, metric in zip_longest(flat_axes, metric_list, fillvalue=None):
            if metric is None:
                ax.remove()
                continue
            si_dict = self.sensitivity[metric]
            S = {label: si[salib_si] for label, si in si_dict.items()}
            df_S = pd.DataFrame(S)
            if df_S.empty:
                ax.remove()
                continue
            try:
                S_conf = {
                    label: si[salib_si + '_conf']
                    for label, si in si_dict.items()
                    }
            except KeyError:
                S_conf = []
            df_S_conf = pd.DataFrame(S_conf)
            if df_S_conf.empty:
                df_S.plot(ax=ax, kind='bar')
            else:
                df_S.plot(ax=ax, kind='bar', yerr=df_S_conf)
            ax.set_xticklabels(self.param_labels, rotation=0)
            ax.set_title(salib_si + ' - ' + metric)
        plt.tight_layout()
            
        return fig, axes
    
    
SALIB_COMPATIBILITY = {
    'fast': ['fast_sampler'],
    'rbd_fast': ['latin'] ,
    'morris': ['morris'],
    'sobol' : ['saltelli'],
    'delta' : ['latin'],
    'dgsm' : ['fast_sampler', 'latin', 'morris', 'saltelli', 'latin', 'ff'], 
    'ff' : ['ff'],
    }

def check_salib(sampling_method, sensitivity_method):
    """
    Checks whether the chosen salib sampling method and sensitivity method
    respect the pairing recommendation by the salib package.

    Parameters
    ----------
    sampling_method : str
        Name of the sampling method.
    sensitivity_method : str
        Name of the sensitivity analysis method.

    Returns
    -------
    bool
        True if sampling and sensitivity methods respect the recommended
        pairing.

    """
    
    if sampling_method not in SALIB_COMPATIBILITY[sensitivity_method]:
        LOGGER.warning("The chosen combination of sensitivity method (%s)"
            " and sampling method (%s) does not correspond to the recommendation"
            " of the salib pacakge."
            "\n https://salib.readthedocs.io/en/latest/api.html"
            %(sampling_method, sensitivity_method)
            )
        return False
    return True
    
