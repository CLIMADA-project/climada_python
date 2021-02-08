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
from functools import partial
import pandas as pd

from climada.engine.uncertainty.base import Uncertainty
from climada.engine.cost_benefit import CostBenefit

LOGGER = logging.getLogger(__name__)

# Future planed features:
# - Add 'efc' (frequency curve) to UncCostBenenfit

class UncCostBenefit(Uncertainty):
    """
    Cost Benefit Uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of a
    climada.engine.costbenefit.CostBenefit() object.

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

        unc_vars = {'haz': self._var_or_uncvar(haz_unc),
                         'ent': self._var_or_uncvar(ent_unc),
                         'haz_fut': self._var_or_uncvar(haz_fut_unc),
                         'ent_fut': self._var_or_uncvar(ent_fut_unc)
                         }

        params = pd.DataFrame()
        problem = {}
        metrics =  {'tot_climate_risk': None,
                    'benefit': None,
                    'cost_ben_ratio': None,
                    'imp_meas_present': None,
                    'imp_meas_future': None}

        Uncertainty.__init__(self, unc_vars=unc_vars,
                             params=params, problem=problem, metrics=metrics)


    def calc_cost_benefit_distribution(self, pool=None, **kwargs):
        """
        Computes the cost benefit for each of the parameters set defined in
        uncertainty.params.

        By default, imp_meas_present, imp_meas_future, tot_climate_risk,
        benefit, cost_ben_ratio are computed.

        Parameters
        ----------
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.
        **kwargs : keyword arguments
            These parameters are passed to
            climada.engine.CostBenefit.calc().

        Returns
        -------
        None.

        """
        

        if self.params.empty:
            LOGGER.info("No sample was found. Please create one first"
                        "using UncImpact.make_sample(N)")
            return None

        #Compute impact distributions
        if pool:
            LOGGER.info('Using %s CPUs.', pool.ncpus)
            chunksize = min(self.n_runs // pool.ncpus, 100)
            cb_metrics = pool.map(partial(self._map_costben_eval, **kwargs),
                                           self.params.iterrows(),
                                           chunsize = chunksize)

        else:
            cb_metrics = map(partial(self._map_costben_eval, **kwargs),
                             self.params.iterrows())

        [imp_meas_present,
         imp_meas_future,
         tot_climate_risk,
         benefit,
         cost_ben_ratio] = list(zip(*cb_metrics)) #Transpose list of list

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

        return None


    def _map_costben_eval(self, param_sample, **kwargs):
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
        haz_params = param_sample[1][self.unc_vars['haz'].labels].to_dict()
        ent_params = param_sample[1][self.unc_vars['ent'].labels].to_dict()
        haz_fut_params = param_sample[1][self.unc_vars['haz_fut'].labels].to_dict()
        ent_fut_params = param_sample[1][self.unc_vars['ent_fut'].labels].to_dict()

        haz = self.unc_vars['haz'].evaluate(haz_params)
        ent = self.unc_vars['ent'].evaluate(ent_params)
        haz_fut = self.unc_vars['haz_fut'].evaluate(haz_fut_params)
        ent_fut = self.unc_vars['ent_fut'].evaluate(ent_fut_params)

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



    def calc_cost_benefit_sensitivity(self,  method='sobol', **kwargs):
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

        #To import a submodule from a module 'from_list' is necessary
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
    