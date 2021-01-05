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

from SALib.sample import saltelli
from SALib.analyze import sobol

from climada.engine import Impact
from climada.entity import ImpactFuncSet
from climada.entity import Exposures
from climada.hazard import Hazard
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
        def unc_var_cat(m, n):
            exp = Litpop()
            exp.set_country('CHE', exponent=[m, n])
            return exp
        distr_dict = {
            m: sp.stats.randint(low=0, high=5),
            n: sp.stats.randint(low=0, high=5)
            }

    Continuous variable function: Impact function for TC
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

    """

    def __init__(self, unc_var, distr_dict):
        """
        Initialize UncVar

        Parameters
        ----------
        unc_var : function
            Variable defined as a function of the uncertainty parameters
        distr_dict : dict
            Dictionnary of the probability density distributions of the
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
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 16))
        for ax, (param_name, distr) in zip(ax, self.distr_dict.items()):
            x = np.linspace(distr.ppf(0.001), distr.ppf(0.999), 100)
            ax.plot(x, distr.pdf(x), label=param_name)
            ax.legend()
        return fig, ax

    def eval_unc_var(self, kwargs):
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


class UncSensitivity():
    """
    Sensitivity analysis class

    This class performs sensitivity analysis on the outputs of a
    climada.engine.impact.Impact() or climada.engine.costbenefit.CostBenefit()
    object.

    """

    def __init__(self, exp_unc, impf_unc, haz_unc, pool=None):
        """Initialize UncSensitivite

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

        if isinstance(exp_unc, Exposures):
            self.exp = UncVar(unc_var=lambda: exp_unc, distr_dict={})
        else:
            self.exp = exp_unc

        if isinstance(impf_unc, ImpactFuncSet):
            self.impf = UncVar(unc_var=lambda: impf_unc, distr_dict={})
        else:
            self.impf = impf_unc

        if isinstance(haz_unc, Hazard):
            self.haz = UncVar(unc_var=lambda: haz_unc, distr_dict={})
        else:
            self.haz = haz_unc

        self.params = pd.DataFrame()
        self.problem = {}
        self.aai_freq = pd.DataFrame()
        self.eai_exp = pd.DataFrame()
        self.at_event = pd.DataFrame()


    def plot_uncertainty(self):
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
        if self.aai_freq.empty:
            raise ValueError("No uncertainty data present. Please run "+
                    "a sensitivity analysis first.")

        log_aai_freq = self.aai_freq.apply(np.log10).copy()
        log_aai_freq = log_aai_freq.replace([np.inf, -np.inf], np.nan)
        cols = log_aai_freq.columns
        nplots = len(cols)
        nrows, ncols = int(nplots / 3) + 1, min(nplots, 3)
        fig, axes = plt.subplots(nrows = nrows,
                                 ncols = ncols,
                                 figsize=(20, ncols * 3.5),
                                 sharex=True,
                                 sharey=True)

        for ax, col in zip(axes.flatten(), cols):
            data = log_aai_freq[col]
            data.hist(ax=ax,  bins=100, density=True, histtype='step')
            avg = self.aai_freq[col].mean()
            std = self.aai_freq[col].std()
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
        Labels of all (exposure, impact function, hazard) uncertainty
        parameters.

        Returns
        -------
        list of strings
            Labels of all uncertainty parameters.

        """
        return self.exp.labels + self.haz.labels + self.impf.labels

    @property
    def distr_dict(self):
        """
        Dictionnary of all (exposure, imapct function, hazard) distributions.

        Returns
        -------
        distr_dict : dict( sp.stats objects )
            Dictionnary of all distributions.

        """

        distr_dict = dict(self.exp.distr_dict)
        distr_dict.update(self.haz.distr_dict)
        distr_dict.update(self.impf.distr_dict)
        return distr_dict

    def make_sobol_sample(self, N, calc_second_order):
        """
        Make a sobol sample for all parameters with their respective
        distributions.

        Parameters
        ----------
        N : int
            Number of samples as defined in SALib.sample.saltelli.sample().
        calc_second_order : boolean
            if True, calculate second-order sensitivities.

        Returns
        -------
        None.

        """

        self.n_samples = N
        sobol_uniform_sample = self._make_uniform_sobol_sample(
            calc_second_order=calc_second_order)
        df_params = pd.DataFrame(sobol_uniform_sample, columns=self.param_labels)
        for param in list(df_params):
            df_params[param] = df_params[param].apply(
                self.distr_dict[param].ppf
                )
        self.params = df_params



    def _make_uniform_sobol_sample(self, calc_second_order):
        """
        Make a uniform sobol sample for the defined model uncertainty parameters
        (self.param_labels)
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis

        Parameters
        ----------
        calc_second_order : boolean
            if True, calculate second-order sensitivities.

        Returns
        -------
        sobol_params : np.matrix
            Returns a NumPy matrix containing the sampled uncertainty parameters using
            Saltelli’s sampling scheme.

        """
        problem = {
            'num_vars' : len(self.param_labels),
            'names' : self.param_labels,
            'bounds' : [[0, 1]]*len(self.param_labels)
            }
        self.problem = problem
        sobol_params = saltelli.sample(problem, self.n_samples,
                                       calc_second_order=calc_second_order)
        return sobol_params


    def calc_impact_sobol_sensitivity(self,
                               N,
                               rp=None,
                               calc_eai_exp=False,
                               calc_at_event=False,
                               calc_second_order=True,
                               **kwargs):
        """
        Compute the sobol sensitivity indices using the SALib library:
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis

        Simple introduction:
        https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis

        Parameters
        ----------
        N : int
            Number of samples as defined in SALib.sample.saltelli.sample()
        rp : list(int), optional
            Return period in years for which sensitivity indices are computed.
            The default is [5, 10, 20, 50, 100, 250.
        calc_eai_exp : boolean, optional
            Toggle computation of the sensitivity for the impact at each
            centroid location. The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.
        calc_second_order : boolean, optional
            if True, calculate second-order sensitivities. The default is True.
        **kwargs :
            These parameters will be passed to SALib.analyze.sobol.analyze()
            The default is num_resamples=100, conf_level=0.95,
            print_to_console=False, parallel=False, n_processors=None,
            seed=None.

        Returns
        -------
        sobol_analysis : dict
            Dictionnary with keys the uncertainty parameter labels.
            For each uncertainty parameter, the item is another dictionnary
            with keys the sobol sensitivity indices ‘S1’, ‘S1_conf’, ‘ST’,
            and ‘ST_conf’. If calc_second_order is True, the dictionary also
            contains keys ‘S2’ and ‘S2_conf’.

        """

        if calc_eai_exp:
            raise NotImplementedError()

        if calc_at_event:
            raise NotImplementedError()

        if rp is None:
            rp =[5, 10, 20, 50, 100, 250]

        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event

        if self.params.empty:
            self.make_sobol_sample(N, calc_second_order=calc_second_order)
        if self.aai_freq.empty:
            self.calc_impact_distribution(rp=rp,
                                          calc_eai_exp=calc_eai_exp,
                                          calc_at_event=calc_at_event
                                          )

        sobol_analysis = {}
        for imp_out in self.aai_freq:
            Y = self.aai_freq[imp_out].to_numpy()
            si_sobol = sobol.analyze(self.problem, Y, **kwargs)
            sobol_analysis.update({imp_out: si_sobol})
            for si_list in si_sobol.values():
                if np.any(np.array(si_list)<0):
                    LOGGER.warning("There is at least one negative sobol " +
                        "index. Consider using more samples or using another "+
                        "sensitivity analysis method." +
                        "See https://github.com/SALib/SALib/issues/109")
                    continue

        self.sensitivity = sobol_analysis

        return sobol_analysis


    def calc_cost_benefit_sobol_sensitivity(self):
        raise NotImplementedError()


    def est_comp_time(self):
        """
        Estimate the computation time

        Returns
        -------
        None.

        """
        raise NotImplementedError()


    def calc_impact_distribution(self,
                                 rp=None,
                                 calc_eai_exp=False,
                                 calc_at_event=False,
                                 ):
        """
        Computes the impact for each of the parameters set defined in
        uncertainty.params.

        By default, the aggregated average annual impact
        (impact.aai_agg) and the excees impact at return periods (rp) is
        computed and stored in self.aai_freq. Optionally, the impact at
        each centroid location is computed (this may require a larger
        amount of memory if the number of centroids is large).

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

        aai_agg_list = []
        freq_curve_list = []
        if calc_eai_exp:
            eai_exp_list = []
        if calc_at_event:
            at_event_list = []

        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event

        if self.pool:
            chunksize = min(self.n_runs // self.pool.ncpus, 100)
            impact_metrics = self.pool.map(self._map_impact_eval,
                                           self.params.iterrows(),
                                           chunsize = chunksize)

            [aai_agg_list,
             freq_curve_list,
             eai_exp_list,
             at_event_list] = list(zip(*impact_metrics))

        else:

            impact_metrics = map(self._map_impact_eval, self.params.iterrows())

            [aai_agg_list,
             freq_curve_list,
             eai_exp_list,
             at_event_list] = list(zip(*impact_metrics))

        df_aai_freq = pd.DataFrame(freq_curve_list,
                                   columns=['rp' + str(n) for n in rp])
        df_aai_freq['aai_agg'] = aai_agg_list
        self.aai_freq = df_aai_freq

        if calc_eai_exp:
            df_eai_exp = pd.DataFrame(eai_exp_list)
            self.eai_exp = df_eai_exp

        if calc_at_event:
            df_at_event = pd.DataFrame(at_event_list)
            self.at_event = df_at_event


    def _map_impact_eval(self, param_sample):
        """
        Map to compute impact for all parameter samples in parrallel

        Parameters
        ----------
        param_sample : pd.DataFrame.iterrows()
            Generator of the parameter samples

        Returns
        -------
        list
            impact metrics list for all samples containing aai_agg, rp_curve,
            eai_exp (if self.calc_eai_exp=True), and at_event (if
            self.calc_at_event=True)

        """

        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        exp_params = param_sample[1][self.exp.labels].to_dict()
        haz_params = param_sample[1][self.haz.labels].to_dict()
        impf_params = param_sample[1][self.impf.labels].to_dict()

        exp = self.exp.eval_unc_var(exp_params)
        haz = self.haz.eval_unc_var(haz_params)
        impf = self.impf.eval_unc_var(impf_params)

        imp = Impact()
        imp.calc(exposures=exp, impact_funcs=impf, hazard=haz)

        rp_curve = imp.calc_freq_curve(self.rp).impact

        if self.calc_eai_exp:
            eai_exp = imp.eai_exp
        else:
            eai_exp = None

        if self.calc_at_event:
            at_event= imp.at_event
        else:
            at_event = None

        return [imp.aai_agg, rp_curve, eai_exp, at_event]


class UncRobustness():
    """
    Compute variance from multiplicative Gaussian noise
    """
    raise NotImplementedError()
