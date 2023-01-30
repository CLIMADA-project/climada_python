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

__all__ = ['UncOutput', 'UncCostBenefitOutput', 'UncImpactOutput']

import logging
import datetime as dt

from itertools import zip_longest
from pathlib import Path


import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from climada import CONFIG

from climada.util.value_representation import value_to_monetary_unit as u_vtm
from climada.util.value_representation import convert_monetary_value as u_cmv
from climada.util import plot as u_plot
import climada.util.hdf5_handler as u_hdf5

LOGGER = logging.getLogger(__name__)

# Metrics that are multi-dimensional
METRICS_2D = ['eai_exp', 'at_event']

DATA_DIR = CONFIG.engine.uncertainty.local_data.user_data.dir()

FIG_W, FIG_H = 8, 5 #default figize width/heigh column/work multiplicators

MAP_CMAP = 'Dark2' #Default color map for the sensitivity map

#Table of recommended pairing between salib sampling and sensitivity methods
# NEEDS TO BE UPDATED REGULARLY!! https://salib.readthedocs.io/en/latest/api.html
SALIB_COMPATIBILITY = {
    'fast': ['fast_sampler'],
    'rbd_fast': ['latin'] ,
    'morris': ['morris'],
    'sobol' : ['saltelli'],
    'delta' : ['latin'],
    'dgsm' : ['fast_sampler', 'latin', 'morris', 'saltelli', 'latin', 'ff'],
    'ff' : ['ff'],
    }

plt.style.use('seaborn-white')
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
mpl.rcParams.update(params)


class UncOutput():
    """
    Class to store and plot uncertainty and sensitivity analysis output data

    This is the base class to store uncertainty and sensitivity outputs of an
    analysis  done on climada.engine.impact.Impact() or
    climada.engine.costbenefit.CostBenefit() object.

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
        Comon flattened dictionary of all the distr_dict of all input variables.
        It represents the distribution of all the uncertainty parameters.
    problem_sa : dict
        The description of the uncertainty variables and their
        distribution as used in SALib.
        https://salib.readthedocs.io/en/latest/basics.html.
    """

    _metadata = ['sampling_method', 'sampling_kwargs', 'sensitivity_method',
                 'sensitivity_kwargs']

    def __init__(self, samples_df, unit=None):
        """
        Initialize Uncertainty Data object.

        Parameters
        ----------
        samples_df : pandas.DataFrame
            input parameters samples
        unit : str, optional
            value unit
        """
        #Data
        self.samples_df = samples_df
        self.unit = unit

    def get_samples_df(self):
        return getattr(self, 'samples_df')

    def get_unc_df(self, metric_name):
        return getattr(self, f'{metric_name}_unc_df')

    def set_unc_df(self, metric_name, unc_df):
        setattr(self, f'{metric_name}_unc_df', unc_df)

    def get_sens_df(self, metric_name):
        return getattr(self, f'{metric_name}_sens_df')

    def set_sens_df(self, metric_name, sens_df):
        setattr(self, f'{metric_name}_sens_df', sens_df)

    def check_salib(self, sensitivity_method):
        """
        Checks whether the chosen sensitivity method and the sampling
        method used to generated self.samples_df
        respect the pairing recommendation by the SALib package.

        https://salib.readthedocs.io/en/latest/api.html

        Parameters
        ----------
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
        """
        Returns the sampling method used to generate self.samples_df

        Returns
        -------
        str :
            Sampling method name

        """
        return self.samples_df.attrs['sampling_method']

    @property
    def sampling_kwargs(self):
        """
        Returns the kwargs of the sampling method that generate self.samples_df

        Returns
        -------
        dict
            Dictionary of arguments for SALib sampling method

        """
        return self.samples_df.attrs['sampling_kwargs']

    @property
    def n_samples(self):
        """
        The effective number of samples

        Returns
        -------
        int :
            effective number of samples

        """
        return self.samples_df.shape[0]

    @property
    def param_labels(self):
        """
        Labels of all uncertainty input parameters.

        Returns
        -------
        list of str :
            Labels of all uncertainty input parameters.

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
        dict :
            Salib problem dictionary.

        """
        return {
            'num_vars' : len(self.param_labels),
            'names' : self.param_labels,
            'bounds' : [[0, 1]]*len(self.param_labels)
            }

    @property
    def uncertainty_metrics(self):
        """
        Retrieve all uncertainty output metrics names

        Returns
        -------
        unc_metric_list : [str]
            List of names of attributes containing metrics uncertainty values,
            without the trailing '_unc_df'

        """
        unc_metric_list = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                if not attr_value.empty and attr_name.endswith("_unc_df"):
                    unc_metric_list.append(attr_name[:-7])
        return unc_metric_list

    @property
    def sensitivity_metrics(self):
        """
        Retrieve all sensitivity output metrics names

        Returns
        -------
        sens_metric_list : [str]
            List of names of attributes containing metrics sensitivity values,
            without the trailing '_sens_df'

        """
        sens_metric_list = []
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                if not attr_value.empty and attr_name.endswith("_sens_df"):
                    sens_metric_list.append(attr_name[:-8])
        return sens_metric_list

    def get_uncertainty(self, metric_list=None):
        """
        Returns uncertainty dataframe with values for each sample

        Parameters
        ----------
        metric_list : [str], optional
            List of uncertainty metrics to consider.
            The default returns all uncertainty metrics at once.

        Returns
        -------
        pandas.DataFrame
            Joint dataframe of all uncertainty values for all metrics in
            the metric_list.

        See Also
        --------
        uncertainty_metrics: list of all available uncertainty metrics

        """
        if metric_list is None:
            metric_list = self.uncertainty_metrics
        try:
            unc_df = pd.concat(
                [self.get_unc_df(metric) for metric in metric_list],
                axis=1
                )
        except AttributeError:
            return pd.DataFrame([])
        return unc_df

    def get_sensitivity(self, salib_si, metric_list=None):
        """
        Returns sensitivity index

        E.g. For the sensitivity analysis method 'sobol', the choices
        are ['S1', 'ST'], for 'delta' the  choices are ['delta', 'S1'].

        For more information see the SAlib documentation:
        https://salib.readthedocs.io/en/latest/basics.html

        Parameters
        ----------
        salib_si : str
            Sensitivity index
        metric_list :[str], optional
            List of sensitivity metrics to consider.
            The default returns all sensitivity indices at once.

        Returns
        -------
        pandas.DataFrame
            Joint dataframe of the sensitvity indices for all metrics in
            the metric_list

        See Also
        --------
        sensitvity_metrics: list of all available sensitivity metrics

        """
        df_all = pd.DataFrame([])
        df_meta = pd.DataFrame([])

        if metric_list is None:
            metric_list = self.sensitivity_metrics
        for metric in metric_list:
            submetric_df = self.get_sens_df(metric)
            if not submetric_df.empty:
                submetric_df = submetric_df[submetric_df['si'] == salib_si]
                df_all = pd.concat(
                    [df_all, submetric_df.select_dtypes('number')],
                    axis=1
                    )
                if df_meta.empty:
                    df_meta = submetric_df.drop(
                        submetric_df.select_dtypes('number').columns, axis=1)
        return pd.concat([df_meta, df_all], axis=1).reset_index(drop=True)

    def get_largest_si(self, salib_si, metric_list=None, threshold=0.01):
        """
        Get largest si per metric

        Parameters
        ----------
        salib_si : str
            The name of the sensitivity index to plot.
        metric_list : list of strings, optional
            List of metrics to plot the sensitivity.
            Default is None.
        threshold : float
            The minimum value a sensitivity index must have to be considered
            as the largest. The default is 0.01.
        Returns
        -------
        max_si_df : pandas.dataframe
            Dataframe with the largest si and its value per metric

        """

        if metric_list is None:
            metric_list = self.sensitivity_metrics

        si_df = self.get_sensitivity(salib_si, metric_list)

        #get max index
        si_df_num = si_df.select_dtypes('number')
        si_df_num[si_df_num<threshold] = 0 #remove noise whenn all si are 0
        max_si_idx = si_df_num.idxmax().to_numpy()
        max_si_val = si_df_num.max().to_numpy()

        #get parameter of max index
        max_si_idx = np.nan_to_num(max_si_idx + 1).astype(int) # Set np.nan values to 0
        param = np.concatenate((['None'], si_df['param']))
        param_max_si = [param[idx] for idx in max_si_idx]
        param2 = np.concatenate((['None'], si_df['param2']))
        param2_max_si = [param2[idx] for idx in max_si_idx]

        max_si_df = pd.DataFrame(
            {'metric' : si_df_num.columns,
             'param' : param_max_si,
             'param2' : param2_max_si,
             'si' : max_si_val
             })

        return max_si_df

    def plot_sample(self, figsize=None):
        """
        Plot the sample distributions of the uncertainty input parameters

        For each uncertainty input variable, the sample distributions is shown
        in a separate axes.

        Parameters
        ---------
        figsize : tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:

            >>> nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
            >>> figsize = (ncols * FIG_W, nrows * FIG_H)

        Raises
        ------
        ValueError
            If no sample was computed the plot cannot be made.

        Returns
        -------
        axes: matplotlib.pyplot.axes
            The axis handle of the plot.

        """
        fontsize = 18

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
            ax.set_xlabel(label)
            ax.set_ylabel('Sample count')
            ax.tick_params(labelsize=fontsize)
            for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                item.set_fontsize(fontsize)

        plt.tight_layout()

        return axes

    def plot_uncertainty(self, metric_list=None, orig_list=None, figsize=None,
                         log=False, axes=None):
        """
        Plot the  uncertainty distribution

        For each risk metric, a separate axes is used to plot the uncertainty
        distribution of the output values obtained over the sampled
        input parameters.

        Parameters
        ----------
        metric_list : list[str], optional
            List of metrics to plot the distribution.
            The default is None.
        orig_list : list[float], optional
            List of the original (without uncertainty) values for each
            sub-metric of the mtrics in metric_list. The ordering is identical.
            The default is None.
        figsize : tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:
            nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
            figsize = (ncols * FIG_W, nrows * FIG_H)
        log : boolean, optional
            Use log10 scale for x axis. Default is False.
        axes : matplotlib.pyplot.axes, optional
            Axes handles to use for the plot. The default is None.

        Raises
        ------
        ValueError
            If no metric distribution was computed the plot cannot be made.

        Returns
        -------
        axes : matplotlib.pyplot.axes
            The axes handle of the plot.

        See Also
        --------
        uncertainty_metrics : list of all available uncertainty metrics

        """
        fontsize = 18 #default label fontsize

        if not self.uncertainty_metrics:
            raise ValueError("No uncertainty data present for these metrics. "+
                    "Please run an uncertainty analysis first.")

        if orig_list is None:
            orig_list = []

        if metric_list is None:
            metric_list = [
                metric
                for metric in self.uncertainty_metrics
                if metric not in METRICS_2D
                ]

        unc_df = self.get_uncertainty(metric_list)

        if unc_df.empty:
            raise AttributeError(f"The listed metrics {metric_list} either "
                 "do not exist or have no uncertainty values.")

        if log:
            unc_df_plt = unc_df.apply(np.log10).copy()
            unc_df_plt = unc_df_plt.replace([np.inf, -np.inf], np.nan)
        else:
            unc_df_plt = unc_df.copy()

        cols = unc_df_plt.columns
        nplots = len(cols)
        nrows, ncols = int(np.ceil(nplots / 2)), min(nplots, 2)
        if axes is None:
            if not figsize:
                figsize = (ncols * FIG_W, nrows * FIG_H)
            _fig, axes = plt.subplots(nrows = nrows,
                                     ncols = ncols,
                                     figsize = figsize)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])

        for ax, col, orig_val in zip_longest(flat_axes, cols, orig_list, fillvalue=None):
            if col is None:
                if ax is not None:
                    ax.remove()
                continue
            data, m_unit = u_vtm(unc_df_plt[col])
            data = pd.Series(data)
            if data.empty:
                ax.remove()
                continue
            data.hist(ax=ax, bins=30, density=True, histtype='bar',
                      color='lightsteelblue', edgecolor='black')
            try:
                data.plot.kde(ax=ax, color='darkblue', linewidth=4, label='')
            except np.linalg.LinAlgError:
                pass
            avg, std = data.mean(), data.std()
            _, ymax = ax.get_ylim()
            if log:
                avg_plot = np.log10(avg)
            else:
                avg_plot = avg
            ax.axvline(avg_plot, color='darkorange', linestyle='dashed', linewidth=2,
                    label="avg=%.2f%s" %(avg, m_unit))
            if orig_val is not None:
                if log:
                    orig_plot = np.log10(orig_val)
                else:
                    orig_plot = orig_val
                [orig_plot] = u_cmv(orig_plot, m_unit)
                ax.axvline(orig_plot, color='green', linestyle='dotted', linewidth=2,
                           label="orig=%.2f%s" %(orig_plot, m_unit))
            if log:
                std_m, std_p = np.log10(avg - std), np.log10(avg + std)
            else:
                std_m, std_p = avg - std, avg + std
            ax.plot([std_m, std_p],
                    [0.3 * ymax, 0.3 * ymax], color='black',
                    label="std=%.2f%s" %(std, m_unit))
            xlabel = col + ' [' + m_unit + ' ' + self.unit + '] '
            if log:
                ax.set_xlabel( xlabel + ' (log10 scale)')
            else:
                ax.set_xlabel(xlabel)
            ax.set_ylabel('density of samples')
            ax.legend(fontsize=fontsize-2)

            ax.tick_params(labelsize=fontsize)
            for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                item.set_fontsize(fontsize)
            ax.set_xlim([data.min(), data.max()])
        plt.tight_layout()

        return axes


    def plot_rp_uncertainty(self, orig_list=None, figsize=(16, 6), axes=None):
        """
        Plot the distribution of return period uncertainty

        Parameters
        ----------
        orig_list : list[float], optional
            List of the original (without uncertainty) values for each
            sub-metric of the mtrics in metric_list. The ordering is identical.
            The default is None.
        figsize : tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is (8, 6)
        axes: matplotlib.pyplot.axes, optional
            Axes handles to use for the plot. The default is None.

        Raises
        ------
        ValueError
            If no metric distribution was computed the plot cannot be made.

        Returns
        -------
        ax : matplotlib.pyplot.axes
            The axis handle of the plot.

        """

        try:
            unc_df = self.freq_curve_unc_df
        except AttributeError:
            unc_df = None
        if unc_df is None or unc_df.empty:
            raise ValueError("No return period uncertainty data present "
                    "Please run an uncertainty analysis with the desired "
                    "return period specified.")

        add_orig=True
        if orig_list is None:
            add_orig=False

        if  axes is  None:
            _fig, axes = plt.subplots(figsize=figsize, nrows=1, ncols=2)

        [min_l, max_l], m_unit = u_vtm([unc_df.min().min(), unc_df.max().max()], n_sig_dig=4)

        ax = axes[0]

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        for n, (_name, values) in enumerate(unc_df.iteritems()):
            values = u_cmv(values, m_unit, n_sig_dig=4)
            count, division = np.histogram(values, bins=100)
            count = count / count.max()
            losses = [(bin_i + bin_f )/2 for (bin_i, bin_f) in zip(division[:-1], division[1:])]
            ax.plot([min_l, max_l], [2*n, 2*n], color='k', alpha=0.5)
            ax.fill_between(losses, count + 2*n, 2*n)
            if add_orig:
                [orig_val] = u_cmv(orig_list[n], m_unit, n_sig_dig=4)
                ax.plot(
                    [orig_val, orig_val], [2*n, 2*(n+1)],
                    color=colors[n], linestyle='dotted', linewidth=2,
                    label="orig=%.2f%s" %(orig_val, m_unit)
                    )

        ax.set_xlim(min_l, max_l)
        ax.set_ylim(0, 2*unc_df.shape[1])
        ax.set_xlabel('Impact [%s %s]' %(m_unit, self.unit))
        ax.set_ylabel('Return period [years]')
        ax.set_yticks(np.arange(0, 2*unc_df.shape[1], 2))
        ax.set_yticklabels([s[2:] for s in unc_df.columns])
        ax.legend(loc='lower right')

        ax = axes[1]

        high = u_cmv(self.get_unc_df('freq_curve').quantile(0.95).values, m_unit, n_sig_dig=4)
        middle = u_cmv(self.get_unc_df('freq_curve').quantile(0.5).values, m_unit, n_sig_dig=4)
        low = u_cmv(self.get_unc_df('freq_curve').quantile(0.05).values, m_unit, n_sig_dig=4)

        x = [float(rp[2:]) for rp in unc_df.columns]
        ax.plot(x, high, linestyle='--', color = 'blue',
                alpha=0.5, label='0.95 percentile')
        ax.plot(x, middle, label='0.5 percentile')
        ax.plot(x, low, linestyle='dashdot', color='blue',
                alpha=0.5, label='0.05 percentile')
        ax.fill_between(x, low, high, alpha=0.2)
        if add_orig:
            ax.plot(x, u_cmv(orig_list, m_unit, n_sig_dig=4),
                    color='green', linestyle='dotted', label='orig')
        ax.set_xlabel('Return period [year]')
        ax.set_ylabel('Impact [' + m_unit + ' ' + self.unit + ']')
        ax.legend()

        return axes


    def plot_sensitivity(self, salib_si='S1', salib_si_conf='S1_conf',
                         metric_list=None, figsize=None, axes=None,
                         **kwargs):
        """
        Bar plot of a first order sensitivity index

        For each metric, the sensitivity indices are plotted in a
        separate axes.

        This requires that a senstivity analysis was already
        performed.

        E.g. For the sensitivity analysis method 'sobol', the choices
        are ['S1', 'ST'], for 'delta' the  choices are ['delta', 'S1'].

        Note that not all sensitivity indices have a confidence interval.

        For more information see the SAlib documentation:
        https://salib.readthedocs.io/en/latest/basics.html

        Parameters
        ----------
        salib_si : string, optional
            The first order (one value per metric output) sensitivity index
            to plot.
            The default is S1.
        salib_si_conf : string, optional
            The  confidence value for the first order sensitivity index
            to plot.
            The default is S1_conf.
        metric_list : list of strings, optional
            List of metrics to plot the sensitivity. If a metric is not found
            it is ignored.
        figsize : tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:

            >>> nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
            >>> figsize = (ncols * FIG_W, nrows * FIG_H)
        axes : matplotlib.pyplot.axes, optional
            Axes handles to use for the plot. The default is None.
        kwargs :
            Keyword arguments passed on to
            pandas.DataFrame.plot(kind='bar')

        Raises
        ------
        ValueError :
            If no sensitivity is available the plot cannot be made.

        Returns
        -------
        axes : matplotlib.pyplot.axes
            The axes handle of the plot.

        See Also
        --------
        sensitvity_metrics :
            list of all available sensitivity metrics

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
        if axes is None:
            if not figsize:
                figsize = (ncols * FIG_W, nrows * FIG_H)
            _fig, axes = plt.subplots(nrows = nrows,
                                     ncols = ncols,
                                     figsize = figsize)
        if nplots > 1:
            flat_axes = axes.flatten()
        else:
            flat_axes = np.array([axes])

        for ax, metric in zip(flat_axes, metric_list):
            df_S = self.get_sensitivity(salib_si, [metric]).select_dtypes('number')
            if df_S.empty:
                ax.set_xlabel('Input parameter')
                ax.remove()
                continue
            df_S_conf = self.get_sensitivity(salib_si_conf, [metric]).select_dtypes('number')
            if df_S_conf.empty:
                df_S.plot(ax=ax, kind='bar', **kwargs)
            df_S.plot(ax=ax, kind='bar', yerr=df_S_conf, **kwargs)
            ax.set_xticklabels(self.param_labels, rotation=0)
            ax.set_xlabel('Input parameter')
            ax.set_ylabel(salib_si)
        plt.tight_layout()

        return axes

    def plot_sensitivity_second_order(self, salib_si='S2', salib_si_conf='S2_conf',
                                      metric_list=None, figsize=None, axes=None,
                                      **kwargs):
        """
        Plot second order sensitivity indices as matrix.

        For each metric, the sensitivity indices are plotted in a
        separate axes.

        E.g. For the sensitivity analysis method 'sobol', the choices
        are ['S2', 'S2_conf'].

        Note that not all sensitivity indices have a confidence interval.

        For more information see the SAlib documentation:
        https://salib.readthedocs.io/en/latest/basics.html

        Parameters
        ----------
        salib_si : string, optional
            The second order sensitivity indexto plot.
            The default is S2.
        salib_si_conf : string, optional
            The  confidence value for thesensitivity index salib_si
            to plot.
            The default is S2_conf.
        metric_list : list of strings, optional
            List of metrics to plot the sensitivity. If a metric is not found
            it is ignored. Default is all 1D metrics.
        figsize : tuple(int or float, int or float), optional
            The figsize argument of matplotlib.pyplot.subplots()
            The default is derived from the total number of plots (nplots) as:

            >>> nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
            >>> figsize = (ncols * 5, nrows * 5)
        axes : matplotlib.pyplot.axes, optional
            Axes handles to use for the plot. The default is None.
        kwargs :
            Keyword arguments passed on to matplotlib.pyplot.imshow()

        Raises
        ------
        ValueError :
            If no sensitivity is available the plot cannot be made.

        Returns
        -------
        axes:  matplotlib.pyplot.axes
            The axes handle of the plot.

        See Also
        --------
        sensitvity_metrics :
            list of all available sensitivity metrics

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


        if 'cmap' not in kwargs.keys():
            kwargs['cmap'] = 'summer'

        #all the lowest level metrics (e.g. rp10) directly or as
        #submetrics of the metrics in metrics_list
        df_S = self.get_sensitivity(salib_si, metric_list).select_dtypes('number')
        df_S_conf = self.get_sensitivity(salib_si_conf, metric_list).select_dtypes('number')

        nplots = len(df_S.columns)
        nrows, ncols = int(np.ceil(nplots / 3)), min(nplots, 3)
        if axes is None:
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
            ax.imshow(s2_matrix, **kwargs)
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

    def plot_sensitivity_map(self, salib_si='S1', **kwargs):
        """
        Plot a map of the largest sensitivity index in each exposure point

        Requires the uncertainty distribution for eai_exp.

        Parameters
        ----------
        salib_si : str, optional
            The name of the sensitivity index to plot.
            The default is 'S1'.
        kwargs :
            Keyword arguments passed on to
            climada.util.plot.geo_scatter_categorical

        Raises
        ------
        ValueError :
            If no sensitivity data is found, raise error.

        Returns
        -------
        ax: matplotlib.pyplot.axes
            The axis handle of the plot.

        See Also
        --------
        climada.util.plot.geo_scatter_categorical :
            geographical plot for categorical variable

        """

        eai_max_si_df = self.get_largest_si(salib_si, metric_list=['eai_exp'])

        plot_val = eai_max_si_df['param']
        coord = np.array([self.coord_df.latitude, self.coord_df.longitude]).transpose()  # pylint: disable=no-member
        if 'var_name' not in kwargs:
            kwargs['var_name'] = 'Input parameter with largest ' + salib_si
        if 'title' not in kwargs:
            kwargs['title'] = ''
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (8,6)
        if 'cmap' not in kwargs:
            labels = np.unique(plot_val)
            n=np.where(labels=='None')[0]
            if len(n) > 0 :
                n = n[0]
                cmap = mpl.colors.ListedColormap(
                    plt.get_cmap(MAP_CMAP).colors[:len(labels)]
                    )
                colors = list(cmap.colors)
                colors[n] = tuple(np.repeat(0.93, 3))
                cmap.colors = tuple(colors)
                kwargs['cmap'] = cmap
        ax = u_plot.geo_scatter_categorical(
                plot_val, coord,
                **kwargs
                )

        return ax

    def to_hdf5(self, filename=None):
        """
        Save output to .hdf5

        Parameters
        ----------
        filename : str or pathlib.Path, optional
            The filename with absolute or relative path.
            The default name is "unc_output + datetime.now() + .hdf5" and
            the default path is taken from climada.config

        Returns
        -------
        save_path : pathlib.Path
            Path to the saved file

        """
        if filename is None:
            filename = "unc_output" + dt.datetime.now().strftime(
                                                            "%Y-%m-%d-%H%M%S"
                                                            )
            filename = Path(DATA_DIR) / Path(filename)
        save_path = Path(filename)
        save_path = save_path.with_suffix('.hdf5')

        LOGGER.info('Writing %s', save_path)
        store = pd.HDFStore(save_path, mode='w')
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, pd.DataFrame):
                store.put(var_name, var_val, format='fixed', complevel=9)
        store.get_storer('/samples_df').attrs.metadata = self.samples_df.attrs
        store.close()

        str_dt = h5py.special_dtype(vlen=str)
        with h5py.File(save_path, 'a') as fh:
            if getattr(self, 'unit'):
                fh['impact_unit'] = [self.unit]
            if hasattr(self, 'sensitivity_method'):
                if self.sensitivity_method:
                    fh['sensitivity_method'] = [self.sensitivity_method]
            if hasattr(self, 'sensitivity_kwargs'):
                if self.sensitivity_kwargs:
                    grp = fh.create_group("sensitivity_kwargs")
                    for key, value in dict(self.sensitivity_kwargs).items():
                        ds = grp.create_dataset(key, (1,), dtype=str_dt)
                        ds[0] = str(value)
        return save_path

    @staticmethod
    def from_hdf5(filename):
        """
        Load a uncertainty and uncertainty output data from .hdf5 file

        Parameters
        ----------
        filename : str or pathlib.Path
            The filename with absolute or relative path.

        Returns
        -------
        unc_output: climada.engine.uncertainty.unc_output.UncOutput
            Uncertainty and sensitivity data loaded from .hdf5 file.
        """
        filename = Path(filename).with_suffix('.hdf5')
        if not filename.exists():
            LOGGER.info('File not found')
            return None

        unc_data = UncOutput(pd.DataFrame())

        LOGGER.info('Reading %s', filename)
        store = pd.HDFStore(filename, mode='r')
        for var_name in store.keys():
            setattr(unc_data, var_name[1:], store.get(var_name))
        unc_data.samples_df.attrs = store.get_storer('/samples_df').attrs.metadata
        store.close()
        with h5py.File(filename, 'r') as fh:
            if 'impact_unit' in list(fh.keys()):
                unc_data.unit = fh.get('impact_unit')[0].decode('UTF-8')
            if 'sensitivity_method' in list(fh.keys()):
                unc_data.sensitivity_method = \
                    fh.get('sensitivity_method')[0].decode('UTF-8')
            if 'sensitivity_kwargs' in list(fh.keys()):
                grp = fh["sensitivity_kwargs"]
                sens_kwargs = {
                    key: u_hdf5.to_string(grp.get(key)[0])
                    for key in grp.keys()
                    }
                unc_data.sensitivity_kwargs = tuple(sens_kwargs.items())
        return unc_data


class UncImpactOutput(UncOutput):
    """Extension of UncOutput specific for CalcImpact, returned by the  uncertainty() method.
    """
    def __init__(self, samples_df, unit, aai_agg_unc_df, freq_curve_unc_df, eai_exp_unc_df,
                 at_event_unc_df, tot_value_unc_df, coord_df):
        """Constructor

        Uncertainty output values from impact.calc for each sample

        Parameters
        ----------
        samples_df : pandas.DataFrame
            input parameters samples
        unit : str
            value unit
        aai_agg_unc_df : pandas.DataFrame
            Each row contains the value of aai_aag for one sample (row of
            samples_df)
        freq_curve_unc_df : pandas.DataFrame
            Each row contains the values of the impact exceedence frequency
            curve for one sample (row of samples_df)
        eai_exp_unc_df : pandas.DataFrame
            Each row contains the values of eai_exp for one sample (row of
            samples_df)
        at_event_unc_df : pandas.DataFrame
            Each row contains the values of at_event for one sample (row of
            samples_df)
        tot_value_unc_df : pandas.DataFrame
            Each row contains the value of tot_value for one sample (row of
            samples_df)
        coord_df : pandas.DataFrame
            Coordinates of the exposure
        """
        super().__init__(samples_df, unit)
        self.aai_agg_unc_df = aai_agg_unc_df
        self.aai_agg_sens_df = None
        self.freq_curve_unc_df = freq_curve_unc_df
        self.freq_curve_sens_df = None
        self.eai_exp_unc_df = eai_exp_unc_df
        self.eai_exp_sens_df = None
        self.at_event_unc_df = at_event_unc_df
        self.at_event_sens_df = None
        self.tot_value_unc_df = tot_value_unc_df
        self.tot_value_sens_df = None
        self.coord_df = coord_df


class UncCostBenefitOutput(UncOutput):
    """Extension of UncOutput specific for CalcCostBenefit, returned by the uncertainty() method.
    """
    def __init__(self, samples_df, unit, imp_meas_present_unc_df, imp_meas_future_unc_df,
                 tot_climate_risk_unc_df, benefit_unc_df, cost_ben_ratio_unc_df,
                 cost_benefit_kwargs):
        """Constructor

        Uncertainty output values from cost_benefit.calc for each sample

        Parameters
        ----------
        samples_df : pandas.DataFrame
            input parameters samples
        unit : str
            value unit
        imp_meas_present_unc_df : pandas.DataFrame
            Each row contains the values of imp_meas_present for one sample
            (row of samples_df)
        imp_meas_future_unc_df : pandas.DataFrame
            Each row contains the values of imp_meas_future for one sample
            (row of samples_df)
        tot_climate_risk_unc_df : pandas.DataFrame
            Each row contains the values of tot_cliamte_risk for one sample
            (row of samples_df)
        benefit_unc_df : pandas.DataFrame
            Each row contains the values of benefit for one sample
            (row of samples_df)
        cost_ben_ratio_unc_df : pandas.DataFrame
            Each row contains the values of cost_ben_ratio for one sample
            (row of samples_df)
        cost_benefit_kwargs : pandas.DataFrame
            Each row contains the value of cost_benefit for one sample
            (row of samples_df)

        """
        super().__init__(samples_df, unit)
        self.imp_meas_present_unc_df= imp_meas_present_unc_df
        self.imp_meas_present_sens_df = None
        self.imp_meas_future_unc_df= imp_meas_future_unc_df
        self.imp_meas_future_sens_df = None
        self.tot_climate_risk_unc_df = tot_climate_risk_unc_df
        self.tot_climate_risk_sens_df = None
        self.benefit_unc_df = benefit_unc_df
        self.benefit_sens_df = None
        self.cost_ben_ratio_unc_df = cost_ben_ratio_unc_df
        self.cost_ben_ratio_sens_df = None
        self.cost_benefit_kwargs = cost_benefit_kwargs
