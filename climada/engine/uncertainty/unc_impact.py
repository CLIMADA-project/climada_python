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

Define Uncertainty Impact class
"""

__all__ = ['UncImpact']

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from climada.engine import Impact
from climada.engine.uncertainty.base import Uncertainty

LOGGER = logging.getLogger(__name__)

# Future planed features:
# Nice plots

class UncImpact(Uncertainty):
    """
    Impact uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() object.

    """

    def __init__(self, exp_unc, impf_unc, haz_unc):
        """Initialize UncImpact

        Parameters
        ----------
        exp_unc : climada.engine.uncertainty.UncVar or climada.entity.Exposure
            Exposure uncertainty variable or Exposure
        impf_unc : climada.engine.uncertainty.UncVar or climada.entity.ImpactFuncSet
            Impactfunction uncertainty variable or Impact function
        haz_unc : climada.engine.uncertainty.UncVar or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard

        """

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

        Uncertainty.__init__(self, unc_vars=unc_vars,
                             params=params, problem=problem, metrics=metrics)


    def calc_impact_distribution(self,
                                 rp=None,
                                 calc_eai_exp=False,
                                 calc_at_event=False,
                                 pool=None
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
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.

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
        if pool:
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
            chunksize = min(self.n_runs // pool.ncpus, 100)
            imp_metrics = pool.map(self._map_impact_eval,
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

    def plot_rp_distribution(self):
        """
        Plot the distribution of return period values.

        Parameters
        ----------

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


        df_values = self.metrics['freq_curve']

        fig, ax = plt.subplots()

        min_l, max_l = df_values.min().min(), df_values.max().max()

        for n, (name, values) in enumerate(df_values.iteritems()):
            count, division = np.histogram(values, bins=10)
            count = count / count.max()
            losses = [(bin_i + bin_f )/2 for (bin_i, bin_f) in zip(division[:-1], division[1:])]
            ax.plot([min_l, max_l], [2*n, 2*n], color='k', alpha=0.5)
            ax.fill_between(losses, count + 2*n, 2*n)

        ax.set_xlim(min_l, max_l)
        ax.set_ylim(0, 2*n)
        ax.set_xlabel('impact')
        ax.set_ylabel('return period [years]')
        ax.legend()
        ax.set_yticks(np.arange(0, 2*n, 2))
        ax.set_yticklabels(df_values.columns)

        return fig, ax
