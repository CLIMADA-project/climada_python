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

from climada.util.value_representation import value_to_monetary_unit as u_vtm

LOGGER = logging.getLogger(__name__)

# Future planed features:
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

    def __init__(self, unc_vars, sample=None, metrics=None,
                 sensitivity=None):
        """
        Initialize Uncertainty

        Parameters
        ----------
        unc_vars : list of climade.engine.uncertainty.UncVar 
            list of uncertainty variables
        sample : pd.DataFrame, optional
            DataFrame of sampled parameter values. Column names must be
            parameter names (all labels) from all unc_vars.
            The default is pd.DataFrame().
        metrics : dict(), optional
            Dictionnary of the CLIMADA metrics for which the sensitivity
            will be computed. For each sample, each metric must have a
            definite value.
            Keys are metric names (e.g. 'aai_agg', 'freq_curve') and 
            values are pd.DataFrame with values for each parameter sample
            (one row per sample).
            The default is {}.
        sensitivity: dict(), optional
            Dictionnary of the sensitivity analysis for each uncertainty
            parameter.
            The default is None.
        """
    
        self.unc_vars = unc_vars if unc_vars else []
        self.sample = sample if sample else pd.DataFrame(
            columns = self.param_labels)
        self.metrics = metrics if metrics else {}
        self.sensitivity = sensitivity if sensitivity else None
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
        check &= (self.param_labels == self.sample.columns.to_list())
        if not check:
            raise ValueError("Parameter names from unc_vars do not "
                             "correspond to parameters names of sample")
        for metric, df_distr in self.metrics.items():
            if df_distr.empty:
                continue
            check &= (len(df_distr) == self.n_samples)
            if not check:
                raise ValueError(f"Metric f{metric} has less values than the "
                             "number of samples {self.n_samples}")
            
            if df_distr.isnull().values.any():
                LOGGER.warning("At least one metric evaluated to Nan for " +
                    "one cominbation of uncertainty parameters containend " +
                    "in sample. Note that the sensitivity analysis will " +
                    "then return Nan. " +
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

        if isinstance(self.sample, pd.DataFrame):
            return self.sample.shape[0]
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

        """
        
        uniform_base_sample = self._make_uniform_base_sample(N,
                                                             sampling_method,
                                                             sampling_kwargs)
        df_samples = pd.DataFrame(uniform_base_sample, columns=self.param_labels)
        for param in list(df_samples):
            df_samples[param] = df_samples[param].apply(
                self.distr_dict[param].ppf
                )
        self.sample = df_samples
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


    def est_comp_time(self):
        """
        Estimate the computation time
        """
        raise NotImplementedError()


    def calc_sensitivity(self, salib_method='sobol', method_kwargs=None):
        """
        Compute the sensitivity indices using SALib

        Parameters
        ----------
        salib_method : str
            sensitivity analysis method from SALib.analyse
            https://salib.readthedocs.io/en/latest/api.html
            Possible choices:
                'fast', 'rbd_fact', 'morris', 'sobol', 'delta', 'ff'
            The default is 'sobol'.
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
        
        #To import a submodule from a module use 'from_list' necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        method = getattr(
            __import__('SALib.analyze',
                       fromlist=[salib_method]
                       ),
            salib_method
            )

        if method_kwargs is None: method_kwargs = {} 
        sensitivity_dict = {}
        for name, df_metric in self.metrics.items():
            sensitivity_dict[name] = {}
            for metric in df_metric:
                Y = df_metric[metric].to_numpy()
                sensitivity_index = method.analyze(self.problem, Y,
                                                            **method_kwargs)
                sensitivity_dict[name].update({metric: sensitivity_index})
        
        self.sensitivity = sensitivity_dict

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
        
        if self.sample.empty:
            raise ValueError("No uncertainty sample present."+
                    "Please make a sample first.")

        nplots = len(self.param_labels)
        nrows, ncols = int(nplots / 3) + 1, min(nplots, 3)
        fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for ax, label in zip(axis.flatten(), self.param_labels):
            self.sample[label].hist(ax=ax, bins=100) 
            ax.set_title(label)
            ax.set_xlabel('value')
            ax.set_ylabel('Sample count')
            ax.legend()
            
        return fig, axis


   
    def plot_sensitivity(self, salib_si='ST', metric_list=None):
        """
        Plot the first order sensitivity indices of the chosen metric.

        Parameters
        ----------
        salib_si: string, optional
            The sensitivity index to plot (see SALib option)
            https://salib.readthedocs.io/en/latest/basics.html
            Possible choices: "S1", "ST", "S2" (only if calc_second_order=True)
            The default is ST
            
        metric_list: list of strings, optional
            List of metrics to plot the sensitivity
            The default is ['aai_agg', 'freq_curve', ]

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
            raise ValueError("No sensitivity present for this emtrics. "+
                    "Please run a sensitivity analysis first.")
            
        if metric_list is None:
            metric_list = ['aai_agg', 'freq_curve', 'tot_climate_risk',
                           'benefit', 'cost_ben_ratio', 'imp_meas_present',
                           'imp_meas_future']
            metric_list = list(set(metric_list) & set(self.metrics.keys()))
            
            
        nplots = len(metric_list)
        nrows, ncols = int(nplots / 3) + 1, min(nplots, 3)
        fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize=(nrows*9, ncols * 3.5),
                                 sharex=True,
                                 sharey=True)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]
        
        for ax, metric in zip(flat_axes, metric_list):
            si_dict = self.sensitivity[metric]
            S = {label: si[salib_si] for label, si in si_dict.items()}
            S_conf = {
                label: si[salib_si + '_conf']
                for label, si in si_dict.items()
                }
            df_S = pd.DataFrame(S)
            df_S_conf = pd.DataFrame(S_conf)
            if df_S.empty: continue
            df_S.plot(ax=ax, kind='bar', yerr=df_S_conf)
            ax.set_xticklabels(self.param_labels, rotation=0)
            ax.set_title('S1 - ' + metric)
            
        return fig, axes