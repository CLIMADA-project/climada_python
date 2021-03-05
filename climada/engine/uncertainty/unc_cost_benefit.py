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

Define Uncertainty Cost Benefit class
"""

__all__ = ['UncCostBenefit']

import logging
import time
from functools import partial
import pandas as pd

from climada.engine.uncertainty.base import Uncertainty, UncVar
from climada.engine.cost_benefit import CostBenefit
from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

# Future planed features:
# - Add 'efc' (frequency curve) to UncCostBenenfit

class UncCostBenefit(Uncertainty):
    """
    Cost Benefit Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.costbenefit.CostBenefit() object using the SALib package.

    """

    def __init__(self, haz_unc, ent_unc, haz_fut_unc=None, ent_fut_unc=None):
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

        unc_vars = {
            'haz': UncVar.var_or_uncvar(haz_unc),
            'ent': UncVar.var_or_uncvar(ent_unc),
            'haz_fut': UncVar.var_or_uncvar(haz_fut_unc),
            'ent_fut': UncVar.var_or_uncvar(ent_fut_unc)
            }

        metrics =  {
            'tot_climate_risk': pd.DataFrame([]),
            'benefit': pd.DataFrame([]),
            'cost_ben_ratio': pd.DataFrame([]),
            'imp_meas_present': pd.DataFrame([]),
            'imp_meas_future': pd.DataFrame([])
            }

        Uncertainty.__init__(self, unc_vars=unc_vars, metrics=metrics)


    def calc_distribution(self, pool=None, **kwargs):
        """
        Computes the cost benefit for each of the parameters set defined in
        uncertainty.samples.

        By default, imp_meas_present, imp_meas_future, tot_climate_risk,
        benefit, cost_ben_ratio are computed.

        Parameters
        ----------
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.
        **kwargs : keyword arguments
            Any keyword arguments of climada.engine.CostBenefit.calc()
            EXCEPT: haz, ent, haz_fut, ent_fut

        Returns
        -------
        None.

        """

        if self.sample.empty:
            raise ValueError("No sample was found. Please create one first" + 
                        "using UncImpact.make_sample(N)")
            
        start = time.time()
        one_sample = self.sample.iloc[0:1].iterrows()
        cb_metrics = map(self._map_costben_calc, one_sample)
        [imp_meas_present,
         imp_meas_future,
         tot_climate_risk,
         benefit,
         cost_ben_ratio] = list(zip(*cb_metrics))
        elapsed_time = (time.time() - start) 
        est_com_time = self.est_comp_time(elapsed_time, pool)
        LOGGER.info(f"\n\nEstimated computation time: {est_com_time}s\n")

        #Compute impact distributions
        if pool:
            LOGGER.info('Using %s CPUs.', pool.ncpus)
            chunksize = min(self.n_samples // pool.ncpus, 100)
            cb_metrics = pool.map(partial(self._map_costben_calc, **kwargs),
                                           self.sample.iterrows(),
                                           chunsize = chunksize)

        else:
            cb_metrics = map(partial(self._map_costben_calc, **kwargs),
                             self.sample.iterrows())
        
        logger_cb = logging.getLogger('climada.engine.cost_benefit')
        logger_cb.setLevel('ERROR')
        logger_impf = logging.getLogger('climada.entity.impact_funcs')
        logger_impf.setLevel('ERROR')
        
        [imp_meas_present,
         imp_meas_future,
         tot_climate_risk,
         benefit,
         cost_ben_ratio] = list(zip(*cb_metrics)) #Transpose list of list
        
        logger_cb.setLevel(CONFIG.log_level.str())
        logger_impf.setLevel(CONFIG.log_level.str())

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
        self.check()

        return None


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
        haz_samples = param_sample[1][self.unc_vars['haz'].labels].to_dict()
        ent_samples = param_sample[1][self.unc_vars['ent'].labels].to_dict()
        haz_fut_samples = param_sample[1][self.unc_vars['haz_fut'].labels].to_dict()
        ent_fut_samples = param_sample[1][self.unc_vars['ent_fut'].labels].to_dict()

        haz = self.unc_vars['haz'].evaluate(haz_samples)
        ent = self.unc_vars['ent'].evaluate(ent_samples)
        haz_fut = self.unc_vars['haz_fut'].evaluate(haz_fut_samples)
        ent_fut = self.unc_vars['ent_fut'].evaluate(ent_fut_samples)

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

    