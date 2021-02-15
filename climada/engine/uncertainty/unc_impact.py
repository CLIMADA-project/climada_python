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
import time

from climada.engine import Impact
from climada.engine.uncertainty.base import Uncertainty
from climada.util.config import CONFIG

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

        metrics = {'aai_agg': pd.DataFrame([]),
                   'freq_curve': pd.DataFrame([]),
                   'eai_exp': pd.DataFrame([]),
                   'at_event':  pd.DataFrame([])
                   }

        Uncertainty.__init__(self, unc_vars=unc_vars, metrics=metrics)


    def calc_distribution(self,
                            rp=None,
                            calc_eai_exp=False,
                            calc_at_event=False,
                            pool=None
                            ):
        """
        Computes the impact for each of the parameters set defined in
        uncertainty.samples.

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

        if self.sample.empty:
            raise ValueError("No sample was found. Please create one first"
                             "using UncImpact.make_sample(N)")

        if rp is None:
            rp=[5, 10, 20, 50, 100, 250]

        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event
        
        start = time.time()
        one_sample = self.sample.iloc[0:1].iterrows()
        imp_metrics = map(self._map_impact_calc, one_sample)
        [aai_agg_list, freq_curve_list,
         eai_exp_list, at_event_list] = list(zip(*imp_metrics))
        elapsed_time = (time.time() - start) 
        est_com_time = self.est_comp_time(elapsed_time, pool)
        LOGGER.info(f"\n\nEstimated computation time: {est_com_time}s\n")
        
        #Compute impact distributions
        if pool:
            LOGGER.info('Using %s CPUs.', pool.ncpus)
            chunksize = min(self.n_runs // pool.ncpus, 100)
            imp_metrics = pool.map(self._map_impact_calc,
                                           self.samples.iterrows(),
                                           chunsize = chunksize)

        else:
            imp_metrics = map(self._map_impact_calc, self.sample.iterrows())
        
        logger_imp = logging.getLogger('climada.engine.impact')
        logger_imp.setLevel('ERROR')
        logger_impf = logging.getLogger('climada.entity.impact_funcs')
        logger_impf.setLevel('ERROR')
        
        [aai_agg_list, freq_curve_list,
         eai_exp_list, at_event_list] = list(zip(*imp_metrics))
        
        logger_imp.setLevel(CONFIG.log_level.str())
        logger_impf.setLevel(CONFIG.log_level.str())

        # Assign computed impact distribution data to self
        self.metrics['aai_agg']  = pd.DataFrame(aai_agg_list,
                                                columns = ['aai_agg'])

        self.metrics['freq_curve'] = pd.DataFrame(freq_curve_list,
                                    columns=['rp' + str(n) for n in rp])
        self.metrics['eai_exp'] =  pd.DataFrame(eai_exp_list)
        self.metrics['at_event'] = pd.DataFrame(at_event_list)
        self.check()
        
        return None

    def _map_impact_calc(self, sample_iterrows):
        """
        Map to compute impact for all parameter samples in parrallel

        Parameters
        ----------
        sample_iterrows : pd.DataFrame.iterrows()
            Generator of the parameter samples

        Returns
        -------
         : list
            impact metrics list for all samples containing aai_agg, rp_curve,
            eai_exp (np.array([]) if self.calc_eai_exp=False) and at_event
            (np.array([]) if self.calc_at_event=False).

        """

        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        exp_samples = sample_iterrows[1][self.unc_vars['exp'].labels].to_dict()
        haz_samples = sample_iterrows[1][self.unc_vars['haz'].labels].to_dict()
        impf_samples = sample_iterrows[1][self.unc_vars['impf'].labels].to_dict()

        exp = self.unc_vars['exp'].evaluate(exp_samples)
        haz = self.unc_vars['haz'].evaluate(haz_samples)
        impf = self.unc_vars['impf'].evaluate(impf_samples)

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
        ax.set_yticks(np.arange(0, 2*n, 2))
        ax.set_yticklabels(df_values.columns)

        return fig, ax
