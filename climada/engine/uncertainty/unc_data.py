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

__all__ = ['UncVar', 'UncData']

import logging
import json
import h5py

from itertools import zip_longest
from pathlib import Path

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from climada.util.value_representation import value_to_monetary_unit as u_vtm
from climada.util.value_representation import sig_dig as u_sig_dig
from climada.util import plot as u_plot
from climada.util.config import setup_logging as u_setup_logging
import climada.util.hdf5_handler as u_hdf5
from climada import CONFIG

LOGGER = logging.getLogger(__name__)
u_setup_logging()

# Metrics that are multi-dimensional
METRICS_2D = ['eai_exp_unc_df', 'eai_exp_sens_df',
              'at_event_unc_df', 'at_event_sens_df']

DATA_DIR = CONFIG.engine.uncertainty.local_data.user_data.dir()

FIG_W, FIG_H = 8, 5 #default figize width/heigh column/work multiplicators

#Table of recommended pairing between salib sampling and sensitivity methods
# NEEDS TO BE UPDATED REGULARLY!!
SALIB_COMPATIBILITY = {
    'fast': ['fast_sampler'],
    'rbd_fast': ['latin'] ,
    'morris': ['morris'],
    'sobol' : ['saltelli'],
    'delta' : ['latin'],
    'dgsm' : ['fast_sampler', 'latin', 'morris', 'saltelli', 'latin', 'ff'],
    'ff' : ['ff'],
    }

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


class UncData():
    """
    Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() or climada.engine.costbenefit.CostBenefit()
    object.

    Attributes
    ----------
    samples_df : pandas.DataFrame
        Values of the sampled uncertainty parameters. It has n_samples rows
        and one column per uncertainty parameter.
    sampling_method : str
        Name of the sampling method from SAlib.
        https://salib.readthedocs.io/en/latest/api.html#
    n_samples : int
        Effective number of samples (number of rows of samples_df)
    param_labels : list
        Name of all the uncertainty parameters
    distr_dict : dict
        Comon flattened dictionary of all the distr_dic list in unc_vars.
        It represents the distribution of all the uncertainty parameters.
    problem_sa : dict
        The description of the uncertainty variables and their
        distribution as used in SALib.
        https://salib.readthedocs.io/en/latest/basics.html.
    """

    _metadata = ['sampling_method', 'sampling_kwargs', 'sensitivity_method',
                 'sensitivity_kwargs']

    def __init__(self):
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
            dictionary of the CLIMADA metrics (outputs from
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
            dictionary of the sensitivity analysis for each uncertainty
            parameter.
            The default is {}.
        """
        #MetaData
        # self.sensitivity_method = ''
        # self.sensitivity_kwargs = ()
        # self.cost_benefit_kwargs = ()
        #Sample
        self.samples_df = pd.DataFrame()
        self.samples_df.attrs['sampling_method'] = ''
        self.samples_df.attrs['sampling_kwargs'] = ()
        #Imact
        # self.aai_agg_unc_df = pd.DataFrame()
        # self.freq_curve_unc_df = pd.DataFrame()
        # self.at_event_unc_df = pd.DataFrame()
        # self.eai_exp_unc_df =pd.DataFrame()
        # self.tot_value_unc_df = pd.DataFrame()
        # self.aai_agg_sens_df = pd.DataFrame()
        # self.freq_curve_sens_df = pd.DataFrame()
        # self.at_event_sens_df = pd.DataFrame()
        # self.eai_exp_sens_df = pd.DataFrame()
        # self.tot_value_sens_df = pd.DataFrame()
        #CostBenefit
        # self.tot_climate_risk_unc_df = pd.DataFrame()
        # self.benefit_unc_df = pd.DataFrame()
        # self.cost_ben_ratio_unc_df = pd.DataFrame()
        # self.imp_meas_present_unc_df = pd.DataFrame()
        # self.imp_meas_future_unc_df = pd.DataFrame()
        # self.tot_climate_risk_sens_df = pd.DataFrame()
        # self.benefit_sens_df = pd.DataFrame()
        # self.cost_ben_ratio_sens_df = pd.DataFrame()
        # self.imp_meas_present_sens_df = pd.DataFrame()
        # self.imp_meas_future_sens_df = pd.DataFrame()

    def check_salib(self, sensitivity_method):
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

        if self.sampling_method not in SALIB_COMPATIBILITY[sensitivity_method]:
            LOGGER.warning("The chosen combination of sensitivity method (%s)"
                " and sampling method (%s) does not correspond to the"
                " recommendation of the salib pacakge."
                "\n https://salib.readthedocs.io/en/latest/api.html",
                self.sampling_method, sensitivity_method
                )
            return False
        return True

    @property
    def sampling_method(self):
        return self.samples_df.attrs['sampling_method']

    @property
    def sampling_kwargs(self):
        return self.samples_df.attrs['sampling_kwargs']

    @property
    def n_samples(self):
        """
        The effective number of samples

        Returns
        -------
        n_samples: int
            effective number of samples

        """

        return self.samples_df.shape[0]

    @property
    def param_labels(self):
        """
        Labels of all uncertainty parameters.

        Returns
        -------
        param_labels: list of strings
            Labels of all uncertainty parameters.

        """
        return list(self.samples_df)

    @property
    def problem_sa(self):
        """
        The description of the uncertainty variables and their
        distribution as used in SALib.
        https://salib.readthedocs.io/en/latest/basics.html

        Returns
        -------
        problem_sa : dict
            Salib problem dictionary.

        """
        return {
            'num_vars' : len(self.param_labels),
            'names' : self.param_labels,
            'bounds' : [[0, 1]]*len(self.param_labels)
            }

    @property
    def uncertainty_metrics(self):
        unc_metric_list = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                if not attr_value.empty and 'unc' in attr_name:
                    unc_metric_list.append(attr_name)
        return unc_metric_list

    @property
    def sensitivity_metrics(self):
        sens_metric_list = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                if not attr_value.empty and 'sens' in attr_name:
                    sens_metric_list.append(attr_name)
        return sens_metric_list

    def get_uncertainty(self, metric_list=None):
        if metric_list is None:
            metric_list = self.uncertainty_metrics
        try:
            unc_df = pd.concat(
                [getattr(self, metric) for metric in metric_list],
                axis=1
                )
        except AttributeError:
            return pd.DataFrame([])
        return unc_df

    def get_sensitivity(self, metric_list, salib_si):
        df_all = pd.DataFrame([])
        df_meta = pd.DataFrame([])
        for metric in metric_list:
            try:
                submetric_df = getattr(self, metric)
            except AttributeError:
                continue
            if not submetric_df.empty:
                submetric_df = submetric_df[submetric_df['si'] == salib_si]
                df_all = pd.concat(
                    [df_all, submetric_df.select_dtypes('number')],
                    axis=1
                    )
                if df_meta.empty:
                    df_meta = submetric_df.drop(submetric_df.select_dtypes('number').columns, axis=1)
        return pd.concat([df_meta, df_all], axis=1)

    def plot_sample(self, figsize=None):
        """
        Plot the sample distributions of the uncertainty parameters.

        Parameters
        ---------
        figsize: tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:
                nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
                figsize = (ncols * FIG_W, nrows * FIG_H)

        Raises
        ------
        ValueError
            If no sample was computed the plot cannot be made.

        Returns
        -------
        axes: matplotlib.pyplot.axes
            The axis handle of the plot.

        """

        if self.samples_df.empty:
            raise ValueError("No uncertainty sample present."+
                    "Please make a sample first.")

        nplots = len(self.param_labels)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        if not figsize:
            figsize = (ncols * FIG_W, nrows * FIG_H)
        _fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for ax, label in zip_longest(axes.flatten(),
                                     self.param_labels,
                                     fillvalue=None):
            if label is None:
                ax.remove()
                continue
            self.samples_df[label].hist(ax=ax, bins=100)
            ax.set_title(label)
            ax.set_xlabel('value')
            ax.set_ylabel('Sample count')

        return axes

    def plot_uncertainty(self, metric_list=None, figsize=None, log=False):
     """
     Plot the distribution of values of the risk metrics over the sampled
     input parameters (i.e. plot the "uncertainty" distributions).

     Parameters
     ----------
     metric_list: list, optional
         List of metrics to plot the distribution.
         The default is None.
     figsize: tuple(int or float, int or float), optional
         The figsize argument of matplotlib.pyplot.subplots()
         The default is derived from the total number of plots (nplots) as:
             nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
             figsize = (ncols * FIG_W, nrows * FIG_H)
     log: boolean
         Use log10 scale for x axis. Default is False

     Raises
     ------
     ValueError
         If no metric distribution was computed the plot cannot be made.

     Returns
     -------
     axes: matplotlib.pyplot.axes
         The axes handle of the plot.

     """
     fontsize = 18 #default label fontsize

     if not self.uncertainty_metrics:
         raise ValueError("No uncertainty data present for these metrics. "+
                 "Please run an uncertainty analysis first.")

     if metric_list is None:
         metric_list = [
             metric
             for metric in self.uncertainty_metrics
             if metric not in METRICS_2D
             ]

     unc_df = self.get_uncertainty(metric_list)

     if log:
         unc_df_plt = unc_df.apply(np.log10).copy()
         unc_df_plt = unc_df_plt.replace([np.inf, -np.inf], np.nan)
     else:
         unc_df_plt = unc_df.copy()

     cols = unc_df_plt.columns
     nplots = len(cols)
     nrows, ncols = int(np.ceil(nplots / 2)), min(nplots, 2)
     if not figsize:
         figsize = (ncols * FIG_W, nrows * FIG_H)
     _fig, axes = plt.subplots(nrows = nrows,
                              ncols = ncols,
                              figsize = figsize)
     if nplots > 1:
         flat_axes = axes.flatten()
     else:
         flat_axes = np.array([axes])

     for ax, col in zip_longest(flat_axes, cols, fillvalue=None):
         if col is None:
             ax.remove()
             continue
         data = unc_df_plt[col]
         if data.empty:
             ax.remove()
             continue
         data.hist(ax=ax, bins=30, density=True, histtype='bar',
                   color='lightsteelblue', edgecolor='black')
         try:
             data.plot.kde(ax=ax, color='darkblue', linewidth=4, label='')
         except np.linalg.LinAlgError:
             pass
         avg, std = unc_df[col].mean(), unc_df[col].std()
         _, ymax = ax.get_ylim()
         if log:
             avg_plot = np.log10(avg)
         else:
             avg_plot = avg
         ax.axvline(avg_plot, color='darkorange', linestyle='dashed', linewidth=2,
                 label="avg=%.2f%s" %u_vtm(avg))
         if log:
             std_m, std_p = np.log10(avg - std), np.log10(avg + std)
         else:
             std_m, std_p = avg - std, avg + std
         ax.plot([std_m, std_p],
                 [0.3 * ymax, 0.3 * ymax], color='black',
                 label="std=%.2f%s" %u_vtm(std))
         ax.set_title(col)
         if log:
             ax.set_xlabel('value [log10]')
         else:
             ax.set_xlabel('value')
         ax.set_ylabel('density of events')
         ax.legend(fontsize=fontsize-2)

         ax.tick_params(labelsize=fontsize)
         for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
             item.set_fontsize(fontsize)
     plt.tight_layout()

     return axes


    def plot_rp_uncertainty(self, figsize=(8, 6)):
        """
        Plot the distribution of return period values.

        Parameters
        ----------
        figsize: tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is (8, 6)

        Raises
        ------
        ValueError
            If no metric distribution was computed the plot cannot be made.

        Returns
        -------
        ax: matplotlib.pyplot.axes
            The axis handle of the plot.

        """

        if self.freq_curve_unc_df.empty:
            raise ValueError("No return period uncertainty data present "
                    "Please run an uncertainty analysis with the desired "
                    "return period specified.")


        unc_df = self.freq_curve_unc_df

        _fig, ax = plt.subplots(figsize=figsize)

        min_l, max_l = unc_df.min().min(), unc_df.max().max()

        for n, (_name, values) in enumerate(unc_df.iteritems()):
            count, division = np.histogram(values, bins=10)
            count = count / count.max()
            losses = [(bin_i + bin_f )/2 for (bin_i, bin_f) in zip(division[:-1], division[1:])]
            ax.plot([min_l, max_l], [2*n, 2*n], color='k', alpha=0.5)
            ax.fill_between(losses, count + 2*n, 2*n)

        ax.set_xlim(min_l, max_l)
        ax.set_ylim(0, 2*(n+1))
        ax.set_xlabel('impact')
        ax.set_ylabel('return period [years]')
        ax.set_yticks(np.arange(0, 2*(n+1), 2))
        ax.set_yticklabels(unc_df.columns)

        return ax


    def plot_sensitivity(self, salib_si='S1', salib_si_conf='S1_conf',
                         metric_list=None, figsize=None):
        """
        Plot one of the first order sensitivity indices of the chosen
        metric(s). This requires that a senstivity analysis was already
        performed.

        E.g. For the sensitivity analysis method 'sobol', the choices
        are ['S1', 'ST'], for 'delta' the  choices are ['delta', 'S1'].

        For more information see the SAlib documentation:
        https://salib.readthedocs.io/en/latest/basics.html

        Parameters
        ----------
        salib_si: string, optional
            The first order (one value per metric output) sensitivity index
            to plot. This must be a key of the sensitivity dictionaries in
            self.sensitivity[metric] for each metric in metric_list.
            The default is S1.
        metric_list: list of strings, optional
            List of metrics to plot the sensitivity. If a metric is not found
            in self.sensitivity, it is ignored.
            The default is all metrics from Impact.calc or CostBenefit.calc:
            ['aai_agg', 'freq_curve', 'tot_climate_risk', 'benefit',
             'cost_ben_ratio', 'imp_meas_present', 'imp_meas_future',
             'tot_value']
        figsize: tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:
                nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
                figsize = (ncols * FIG_W, nrows * FIG_H)

        Raises
        ------
        ValueError
            If no sensitivity is available the plot cannot be made.

        Returns
        -------
        axes: matplotlib.pyplot.axes
            The axes handle of the plot.

        """

        if not self.sensitivity_metrics:
            raise ValueError("No sensitivity present. "
                    "Please run a sensitivity analysis first.")

        if metric_list is None:
            metric_list = [
                metric
                for metric in self.sensitivity_metrics
                if metric not in METRICS_2D
                ]

        nplots = len(metric_list)
        nrows, ncols = int(np.ceil(nplots / 2)), min(nplots, 2)
        if not figsize:
            figsize = (ncols * FIG_W, nrows * FIG_H)
        _fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize = figsize,
                                 sharex = True,
                                 sharey = True)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])

        for ax, metric in zip_longest(flat_axes, metric_list, fillvalue=None):
            if metric is None:
                ax.remove()
                continue
            df_S = self.get_sensitivity([metric], salib_si).select_dtypes('number')
            if df_S.empty:
                ax.remove()
                continue
            df_S_conf = self.get_sensitivity([metric], salib_si_conf)
            if df_S_conf.empty:
                df_S.plot(ax=ax, kind='bar')
            df_S.plot(ax=ax, kind='bar', yerr=df_S_conf)
            ax.set_xticklabels(self.param_labels, rotation=0)
            ax.set_title(salib_si + ' - ' + metric.replace('_sens_df', ''))
        plt.tight_layout()

        return axes

    def plot_sensitivity_second_order(self, salib_si='S2', salib_si_conf='S2_conf',
                                      metric_list=None, figsize=None):
        """Plot second order sensitivity indices as matrix.

        This requires that a senstivity analysis was already performed with
        a method that returns second-order sensitivity indices.

        The sensitivity indices or their confidence interval can be shown.

        E.g. For the sensitivity analysis method 'sobol', the choices
        are ['S2', 'S2_conf'].

        For more information see the SAlib documentation:
        https://salib.readthedocs.io/en/latest/basics.html

        Parameters
        ----------
        salib_si: string, optional
            The second order (one value per metric output) sensitivity index
            to plot. This must be a key of the sensitivity dictionaries in
            self.sensitivity[metric] for each metric in metric_list.
            The default is S2.
        metric_list: list of strings, optional
            List of metrics to plot the sensitivity. If a metric is not found
            in self.sensitivity, it is ignored. For a metric with submetrics,
            e.g. 'freq_curve', all submetrics (e.g. 'rp5') are plotted on
            separate axis. Submetrics (e.g. 'rp5') are also valid choices.
            The default is all metrics and their submetrics from Impact.calc
            or CostBenefit.calc:
            ['aai_agg', 'freq_curve', 'tot_climate_risk', 'benefit',
             'cost_ben_ratio', 'imp_meas_present', 'imp_meas_future',
             'tot_value']
        figsize: tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:
                nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
                figsize = (ncols * 5, nrows * 5)

        Raises
        ------
        ValueError
            If no sensitivity is available the plot cannot be made.

        Returns
        -------
        axes: matplotlib.pyplot.axes
            The axes handle of the plot.

        """

        if not self.sensitivity_metrics:
            raise ValueError("No sensitivity present for this metrics. "
                    "Please run a sensitivity analysis first.")

        if metric_list is None:
            metric_list = [
                metric
                for metric in self.sensitivity_metrics
                if metric not in METRICS_2D
                ]

        #all the lowest level metrics (e.g. rp10) directly or as
        #submetrics of the metrics in metrics_list
        df_S = self.get_sensitivity(metric_list, salib_si).select_dtypes('number')
        df_S_conf = self.get_sensitivity(metric_list, salib_si_conf).select_dtypes('number')

        nplots = len(df_S.columns)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        if not figsize:
            figsize = (ncols * 5, nrows * 5)
        _fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize = figsize)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])

        for ax, submetric in zip(flat_axes, df_S.columns):
            #Make matrix symmetric
            s2_matrix = np.triu(
                np.reshape(
                    df_S[submetric].to_numpy(),
                    (len(self.param_labels), -1)
                    )
                )
            s2_matrix = s2_matrix + s2_matrix.T - np.diag(np.diag(s2_matrix))
            ax.matshow(s2_matrix, cmap='coolwarm')
            s2_conf_matrix = np.triu(
                np.reshape(
                    df_S_conf[submetric].to_numpy(),
                    (len(self.param_labels), -1)
                    )
                )
            s2_conf_matrix = s2_conf_matrix + s2_conf_matrix.T - \
                np.diag(np.diag(s2_conf_matrix))
            for i in range(len(s2_matrix)):
                for j in range(len(s2_matrix)):
                    if np.isnan(s2_matrix[i, j]):
                        ax.text(j, i, np.nan,
                           ha="center", va="center",
                           color="k", fontsize='medium')
                    else:
                        ax.text(j, i,
                            str(round(s2_matrix[i, j], 2)) + u'\n\u00B1' +  #\u00B1 = +-
                            str(round(s2_conf_matrix[i, j], 2)),
                            ha="center", va="center",
                            color="k", fontsize='medium')

            ax.set_title(salib_si + ' - ' + submetric, fontsize=18)
            labels = self.param_labels
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, fontsize=16)
            ax.set_yticklabels(labels, fontsize=16)
        plt.tight_layout()

        return axes

    def plot_sensitivity_map(self, exp, salib_si='S1', figsize=(8, 6)):
        """
        Plot a map of the largest sensitivity index in each exposure point

        Parameters
        ----------
        exp : climada.exposure
            The exposure from which to take the coordinates
        salib_si : str, optional
            The name of the sensitivity index to plot.
            The default is 'S1'.
        figsize: tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is (8, 6)

        Raises
        ------
        ValueError
            If no sensitivity data is found, raise error.

        Returns
        -------
        ax: matplotlib.pyplot.axes
            The axis handle of the plot.

        """

        try:
            si_eai_df = self.get_sensitivity(['eai_exp_sens_df'], salib_si).select_dtypes('number')
            eai_max_si_idx = si_eai_df.idxmax().to_numpy()

        except KeyError as verr:
            raise ValueError("No sensitivity indices found for"
                  " impact.eai_exp. Please compute sensitivity first using"
                  " UncCalcImpact.calc_sensitivity(unc_data, calc_eai_exp=True)"
                  ) from verr
        if len(eai_max_si_idx) != len(exp.gdf):
            LOGGER.error("The length of the sensitivity data "
                  "%d does not match the number "
                  "of points %d in the given exposure. "
                  "Please check the exposure or recompute the sensitivity  "
                  "using UncCalcImpact.calc_sensitivity(calc_eai_exp=True)",
                  len(eai_max_si_idx), len(exp.gdf)
                  )
            return None

        if exp is None:
            exp_input_vals = self.samples_df.loc[0][
                self.unc_vars['exp'].labels
                ].to_dict()
            exp = self.unc_vars['exp'].uncvar_func(**exp_input_vals)

        plot_val = np.array([eai_max_si_idx]).astype(float)
        coord = np.array([exp.gdf.latitude, exp.gdf.longitude]).transpose()
        ax = u_plot.geo_scatter_categorical(
                plot_val, coord,
                var_name='Largest sensitivity index ' + salib_si,
                title='Sensitivity map',
                cat_name= self.param_labels,
                figsize=figsize
                )

        return ax

    def save_hdf5(self, filename=None):
        """
        Save the samples_df dataframe to .csv

        Parameters
        ----------
        filename : str or pathlib.Path, optional
            The filename with absolute or relative path.
            The default name is "unc_data + datetime.now() + .csv" and
            the default path is taken from climada.config

        Returns
        -------
        save_path : pathlib.Path
            Path to the saved file

        """
        if filename is None:
            filename = "unc_data" + dt.datetime.now().strftime(
                                                            "%Y-%m-%d-%H%M%S"
                                                            )
            filename = Path(DATA_DIR) / Path(filename)
        save_path = Path(filename)
        save_path = save_path.with_suffix('.hdf5')

        LOGGER.info('Writing %s', save_path)
        store = pd.HDFStore(save_path)
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, pd.DataFrame):
                store.put(var_name, var_val, format='fixed', complevel=9)
        store.get_storer('/samples_df').attrs.metadata = self.samples_df.attrs
        store.close()

        str_dt = h5py.special_dtype(vlen=str)
        with h5py.File(save_path, 'a') as fh:
            fh['sensitivity_method'] = [self.sensitivity_method]
            grp = fh.create_group("sensitivity_kwargs")
            for key, value in dict(self.sensitivity_kwargs).items():
                ds = grp.create_dataset(key, (1,), dtype=str_dt)
                ds[0] = str(value)
        return save_path


    def load_hdf5(self, filename):
        """
        Load a samples_df from .csv file

        Parameters
        ----------
        filename : str or pathlib.Path
            The filename with absolute or relative path.

        Returns
        -------
        samples_df : pandas.DataFrame
            The loaded samples_df
        """

        LOGGER.info('Reading %s', filename)
        store = pd.HDFStore(filename)
        for var_name in store.keys():
            setattr(self, var_name[1:], store.get(var_name))
        self.samples_df.attrs = store.get_storer('/samples_df').attrs.metadata
        store.close()
        with h5py.File(filename, 'r') as fh:
            self.sensitivity_method = fh.get('sensitivity_method')[0].decode('UTF-8')
            grp = fh["sensitivity_kwargs"]
            sens_kwargs = {
                key: u_hdf5.to_string(grp.get(key)[0])
                for key in grp.keys()
                }
            self.sensitivity_kwargs = tuple(sens_kwargs.items())

