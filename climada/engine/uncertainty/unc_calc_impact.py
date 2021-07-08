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

Define Uncertainty Impact class
"""

__all__ = ['UncCalcImpact']

import logging
import time

import pandas as pd
import numpy as np

from climada.engine import Impact
from climada.engine.uncertainty import UncCalc, UncVar
from climada.util import log_level
from climada.util.config import setup_logging as u_setup_logging

LOGGER = logging.getLogger(__name__)
u_setup_logging()


class UncCalcImpact(UncCalc):
    """
    Impact uncertainty analysis class.

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() object.

    Attributes
    ----------
    rp : list(int)
        List of the chosen return periods.
    calc_eai_exp : bool
        Compute eai_exp or not
    calc_at_event : bool
        Compute eai_exp or not
    unc_vars : dict(UncVar)
        Dictonnary of the required uncertainty variables ['exp',
        'impf', 'haz'] and values are the corresponding UncVar.
    """

    def __init__(self, exp_unc_var, impf_unc_var, haz_unc_var):
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

        UncCalc.__init__(self)
        self.unc_var_names =('exp_unc_var', 'impf_unc_var', 'haz_unc_var')
        self.exp_unc_var =  UncVar.var_to_uncvar(exp_unc_var)
        self.impf_unc_var =  UncVar.var_to_uncvar(impf_unc_var)
        self.haz_unc_var =  UncVar.var_to_uncvar(haz_unc_var)
        self.metric_names = ('aai_agg', 'freq_curve', 'at_event',
                             'eai_exp', 'tot_value')


    def calc_uncertainty(self,
                         unc_data,
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

        This sets the attributes self.rp, self.calc_eai_exp,
        self.calc_at_event, self.metrics.

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

        if unc_data.samples_df.empty:
            raise ValueError("No sample was found. Please create one first"
                             "using UncImpact.make_sample(N)")

        if rp is None:
            rp=[5, 10, 20, 50, 100, 250]

        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event

        start = time.time()
        one_sample = unc_data.samples_df.iloc[0:1].iterrows()
        imp_metrics = map(self._map_impact_calc, one_sample)
        [aai_agg_list, freq_curve_list,
         eai_exp_list, at_event_list, tot_value_list] = list(zip(*imp_metrics))
        elapsed_time = (time.time() - start)
        est_com_time = self.est_comp_time(unc_data.n_samples, elapsed_time, pool)

        #Compute impact distributions
        with log_level(level='ERROR', name_prefix='climada'):
            if pool:
                LOGGER.info('Using %s CPUs.', pool.ncpus)
                chunksize = min(unc_data.n_samples // pool.ncpus, 100)
                imp_metrics = pool.map(self._map_impact_calc,
                                        unc_data.samples_df.iterrows(),
                                        chunsize = chunksize)

            else:
                imp_metrics = map(self._map_impact_calc,
                                  unc_data.samples_df.iterrows())

        #Perform the actual computation
        with log_level(level='ERROR', name_prefix='climada'):
            [aai_agg_list, freq_curve_list,
             eai_exp_list, at_event_list,
             tot_value_list] = list(zip(*imp_metrics))


        # Assign computed impact distribution data to self
        unc_data.aai_agg_unc_df  = pd.DataFrame(aai_agg_list,
                                                columns = ['aai_agg'])
        unc_data.freq_curve_unc_df = pd.DataFrame(freq_curve_list,
                                    columns=['rp' + str(n) for n in rp])
        unc_data.eai_exp_unc_df =  pd.DataFrame(eai_exp_list)
        unc_data.at_event_unc_df = pd.DataFrame(at_event_list)
        unc_data.tot_value_unc_df = pd.DataFrame(tot_value_list,
                                                 columns = ['tot_value'])


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
        exp_samples = sample_iterrows[1][self.exp_unc_var.labels].to_dict()
        impf_samples = sample_iterrows[1][self.impf_unc_var.labels].to_dict()
        haz_samples = sample_iterrows[1][self.haz_unc_var.labels].to_dict()

        exp = self.exp_unc_var.uncvar_func(**exp_samples)
        impf = self.impf_unc_var.uncvar_func(**impf_samples)
        haz = self.haz_unc_var.uncvar_func(**haz_samples)

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

        return [imp.aai_agg, freq_curve, eai_exp, at_event, imp.tot_value]



