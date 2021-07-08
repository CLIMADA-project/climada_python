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

Define Uncertainty Cost Benefit class
"""

__all__ = ['UncCalcCostBenefit']

import logging
import time

from functools import partial
import pandas as pd

from climada.engine.cost_benefit import CostBenefit
from climada.engine.uncertainty import UncCalc, UncVar
from climada.util import log_level
from climada.util.config import setup_logging as u_setup_logging

LOGGER = logging.getLogger(__name__)
u_setup_logging()

# Future planed features:
# - Add 'efc' (frequency curve) to UncCostBenenfit

class UncCalcCostBenefit(UncCalc):
    """
    Cost Benefit Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.costbenefit.CostBenefit().

    Attributes
    ----------

    """

    def __init__(self, haz_unc_var, ent_unc_var,
                 haz_fut_unc_var=None, ent_fut_unc_var=None):
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

        """

        UncCalc.__init__(self)
        self.unc_var_names =('haz_unc_var', 'ent_unc_var',
                             'haz_fut_unc_var', 'ent_fut_unc_var')
        self.haz_unc_var = UncVar.var_to_uncvar(haz_unc_var)
        self.ent_unc_var = UncVar.var_to_uncvar(ent_unc_var)
        self.haz_fut_unc_var = UncVar.var_to_uncvar(haz_fut_unc_var)
        self.ent_fut_unc_var = UncVar.var_to_uncvar(ent_fut_unc_var)
        self.metric_names = ('tot_climate_risk', 'benefit',
                             'cost_ben_ratio',
                             'imp_meas_present', 'imp_meas_future')



    def calc_uncertainty(self, unc_data, pool=None, **cost_benefit_kwargs):
        """
        Computes the cost benefit for each of the parameters set defined in
        uncertainty.samples.

        By default, imp_meas_present, imp_meas_future, tot_climate_risk,
        benefit, cost_ben_ratio are computed.

        This sets the attribute self.metrics.

        Parameters
        ----------
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.
        **kwargs : keyword arguments
            Any keyword arguments of climada.engine.CostBenefit.calc()
            EXCEPT: haz, ent, haz_fut, ent_fut

        """

        if unc_data.samples_df.empty:
            raise ValueError("No sample was found. Please create one first" +
                        "using UncImpact.make_sample(N)")

        LOGGER.info("The freq_curve is not saved. Please "
                    "change the risk_func if return period information "
                    "needed")

        start = time.time()
        one_sample = unc_data.samples_df.iloc[0:1].iterrows()
        cb_metrics = map(self._map_costben_calc, one_sample)
        [imp_meas_present,
         imp_meas_future,
         tot_climate_risk,
         benefit,
         cost_ben_ratio] = list(zip(*cb_metrics))
        elapsed_time = (time.time() - start)
        est_com_time = self.est_comp_time(unc_data.n_samples, elapsed_time, pool)

        #Compute impact distributions
        with log_level(level='ERROR', name_prefix='climada'):
            if pool:
                LOGGER.info('Using %s CPUs.', pool.ncpus)
                chunksize = min(unc_data.n_samples // pool.ncpus, 100)
                cb_metrics = pool.map(partial(self._map_costben_calc, **cost_benefit_kwargs),
                                               unc_data.samples_df.iterrows(),
                                               chunsize = chunksize)

            else:
                cb_metrics = map(partial(self._map_costben_calc, **cost_benefit_kwargs),
                                 unc_data.samples_df.iterrows())

        #Perform the actual computation
        with log_level(level='ERROR', name_prefix='climada'):
            [imp_meas_present,
             imp_meas_future,
             tot_climate_risk,
             benefit,
             cost_ben_ratio] = list(zip(*cb_metrics)) #Transpose list of list

        # Assign computed impact distribution data to self
        unc_data.tot_climate_risk_unc_df = \
            pd.DataFrame(tot_climate_risk, columns = ['tot_climate_risk'])

        unc_data.benefit_unc_df = pd.DataFrame(benefit)
        unc_data.cost_ben_ratio_unc_df = pd.DataFrame(cost_ben_ratio)

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
            setattr(unc_data, name + '_unc_df', df_imp_meas)
        cost_benefit_kwargs = {
            key: str(val)
            for key, val in cost_benefit_kwargs.items()}
        unc_data.cost_benefit_kwargs = tuple(cost_benefit_kwargs.items())

    def _map_costben_calc(self, param_sample, **kwargs):
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
        haz_samples = param_sample[1][self.haz_unc_var.labels].to_dict()
        ent_samples = param_sample[1][self.ent_unc_var.labels].to_dict()
        haz_fut_samples = param_sample[1][self.haz_fut_unc_var.labels].to_dict()
        ent_fut_samples = param_sample[1][self.ent_fut_unc_var.labels].to_dict()

        haz = self.haz_unc_var.uncvar_func(**haz_samples)
        ent = self.ent_unc_var.uncvar_func(**ent_samples)
        haz_fut = self.haz_fut_unc_var.uncvar_func(**haz_fut_samples)
        ent_fut = self.ent_fut_unc_var.uncvar_func(**ent_fut_samples)

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
