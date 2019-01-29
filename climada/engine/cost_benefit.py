"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define CostBenefit class.
"""

__all__ = ['CostBenefit', 'risk_aai_agg']

import logging
import numpy as np

from climada.engine.impact import Impact

LOGGER = logging.getLogger(__name__)

DEF_RP = np.array([1, 5, 10, 20, 25, 30, 35, 40, 45, 50, 75, 100, 125, 150, \
                   175, 200, 250, 300, 400, 500, 1000])
""" Default return periods used for impact exceedance frequency curve """

DEF_PRESENT_YEAR = 2016
""" Default present reference year """

DEF_FUTURE_YEAR = 2030
""" Default future reference year """

def risk_aai_agg(impact):
    """Risk measurement as average annual impact aggregated.

    Parameters:
        impact (Impact): an Impact instance
    """
    return impact.aai_agg

class CostBenefit():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes:
        impact_dict (dict): impact for regular case and every measure
        imp_time_depen (float): parameter describing the impact evolution in
            time

        present_year (int): present reference year
        future_year (int): future year
        tot_climate_risk (float): total climate risk without measures
        benefit (dict): benefit of each measure. Key: measure name, Value:
            float benefit
        cost_ben_ratio (dict): cost benefit ratio of each measure. Key: measure
            name, Value: float cost benefit ratio
        imp_meas_future (dict): impact of each measure at future or default.
            Key: measure name ('no measure' used for case without measure),
            Value: dict with:
                             'cost' (float): cost measure,
                             'risk' (float): risk measurement,
                             'efc'  (ImpactFreqCurve): impact exceedance freq
              (optionally)   'impact' (Impact): impact instance
        imp_meas_present (dict): impact of each measure at present.
            Key: measure name ('no measure' used for case without measure),
            Value: dict with:
                             'cost' (float): cost measure,
                             'risk' (float): risk measurement,
                             'efc'  (ImpactFreqCurve): impact exceedance freq
              (optionally)   'impact' (Impact): impact instance
    """

    def __init__(self):
        """ Initilization """
        self.present_year = DEF_PRESENT_YEAR
        self.future_year = DEF_FUTURE_YEAR

        self.tot_climate_risk = 0.0

        # dictionaries with key: measure name
        # value: measure benefit
        self.benefit = dict()
        # value: measure cost benefit
        self.cost_ben_ratio = dict()
        # 'no measure' key for impact without measures
        # values: dictionary with 'cost': cost measure,
        #                         'risk': risk measurement,
        #                         'efc': ImpactFreqCurve
        #          (optionally)   'impact': Impact
        self.imp_meas_future = dict()
        self.imp_meas_present = dict()

    def calc(self, hazard, entity, haz_future=None, ent_future=None, \
        risk_func=risk_aai_agg, imp_time_depen=1, save_imp=False):
        """Compute cost-benefit ratio for every measure provided current
        and future conditions.

        Parameters:
            hazard (Hazard): hazard
            entity (Entity): entity
            haz_future (Hazard): hazard in the future (future year provided at
                ent_future)
            ent_future (Exposures): entity in the future
            risk_func (func, optional): function describing risk measure given
                an Impact. Default: average annual impact (aggregated).
            imp_time_depen (float, optional): parameter which represent time
                evolution of impact. Default: 1 (linear).
            save_imp (bool, optional): activate if Impact of each measure is
                saved. Default: False.
        """
        # Present year given in entity. Future year in ent_future if provided.
        self.present_year = entity.exposures.ref_year

        if not haz_future and not ent_future:
            self._calc_impact_measures(hazard, entity.exposures, \
                entity.measures, entity.impact_funcs, 'future', \
                risk_func, save_imp)
        else:
            self._calc_impact_measures(hazard, entity.exposures, \
                entity.measures, entity.impact_funcs, 'present', \
                risk_func, save_imp)
            if haz_future and ent_future:
                self.future_year = ent_future.exposures.ref_year
                self._calc_impact_measures(haz_future, ent_future.exposures, \
                    ent_future.measures, ent_future.impact_funcs, 'future', \
                    risk_func, save_imp)
            elif haz_future:
                self._calc_impact_measures(haz_future, entity.exposures, \
                    entity.measures, entity.impact_funcs, 'future', risk_func,\
                    save_imp)
            else:
                self.future_year = ent_future.exposures.ref_year
                self._calc_impact_measures(hazard, ent_future.exposures, \
                    ent_future.measures, ent_future.impact_funcs, 'future', \
                    risk_func, save_imp)

        self._calc_cost_benefit(entity.disc_rates, imp_time_depen)

#    def calc_all_options(self, hazard, entity, haz_future, ent_future, \
#        risk_func=risk_aai_agg, \
#        imp_time_depen=1, \
#        save_imp=False):
#        """Compute cost benefit with respect future conditions. Intermediate
#        results of changes with and without change in hazard and entity are
#        returned. The returned data allows to plot the waterfall.
#
#        Parameters:
#            hazard (Hazard): hazard
#            entity (Entity): entity
#            haz_future (Hazard): hazard in the future (future year provided at
#                ent_future)
#            ent_future (Exposures): entity in the future
#            risk_func (func, optional): function describing risk measure given
#                an Impact. Default: average annual impact (aggregated).
#            imp_time_depen (float, optional): parameter which represent time
#                evolution of impact. Default: by configuration
#            save_imp (bool, optional): activate if Impact of each measure is
#                saved. Default: False.
#
#        Returns:
#            list [current risk, risk due to change in hazard, risk due to
#            change in entity, risk due to change in hazard and entity]
#        """
#        self.present_year = entity.exposures.ref_year
#        self.future_year = ent_future.exposures.ref_year
#
#        risk_evol = []
#
#        # current risk
#        self._calc_impact_measures(hazard, entity.exposures, \
#                entity.measures, entity.impact_funcs, 'future', \
#                risk_func, save_imp)
#        self._calc_cost_benefit(entity.disc_rates, imp_time_depen)
#        risk_evol.append(self.tot_climate_risk)
#
#        # set values to present and compute different futures
#        self.imp_meas_present = self.imp_meas_future
#        # climate change
#        self._calc_impact_measures(haz_future, entity.exposures, \
#            entity.measures, entity.impact_funcs, 'future', risk_func,\
#            save_imp)
#        self._calc_cost_benefit(entity.disc_rates, imp_time_depen)
#        risk_evol.append(self.tot_climate_risk)
#
#        # entity change
#        self._calc_impact_measures(hazard, ent_future.exposures, \
#            ent_future.measures, ent_future.impact_funcs, 'future', \
#            risk_func, save_imp)
#        self._calc_cost_benefit(entity.disc_rates, imp_time_depen)
#        risk_evol.append(self.tot_climate_risk)
#
#        # all change
#        self._calc_impact_measures(haz_future, ent_future.exposures, \
#            ent_future.measures, ent_future.impact_funcs, 'future', \
#            risk_func, save_imp)
#        self._calc_cost_benefit(entity.disc_rates, imp_time_depen)
#        risk_evol.append(self.tot_climate_risk)
#
#        return risk_evol

    def plot_cost_benefit(self):
        """ Plot cost-benefit graph. """
        raise NotImplementedError

    def plot_waterfall(self, risk_evol):
        """ Plot waterfall graph. """
        raise NotImplementedError

    def _calc_impact_measures(self, hazard, exposures, meas_set, imp_fun_set, \
        when='future', risk_func=risk_aai_agg, save_imp=False):
        """Compute impact of each measure and transform it to input risk
        measurement. Set reference year from exposures value.

        Parameters:
            hazard (Hazard): hazard.
            exposures (Exposures): exposures.
            meas_set (MeasureSet): set of measures.
            imp_fun_set (ImpactFuncSet): set of impact functions.
            when (str, optional): 'present' or 'future'. The conditions that
                are being considered.
            risk_func (function, optional): function used to transform impact
                to a risk measurement.
            save_imp (bool, optional): activate if Impact of each measure is
                saved. Default: False.
        """
        impact_meas = dict()

        # compute impact without measures
        imp_tmp = Impact()
        imp_tmp.calc(exposures, imp_fun_set, hazard)
        impact_meas['no measure'] = dict()
        impact_meas['no measure']['cost'] = 0.0
        impact_meas['no measure']['risk'] = risk_func(imp_tmp)
        impact_meas['no measure']['efc'] = imp_tmp.calc_freq_curve(DEF_RP)
        if save_imp:
            impact_meas['no measure']['impact'] = imp_tmp

        # compute impact for each measure
        for measure in meas_set.get_measure():
            new_exp, new_ifs, new_haz = measure.apply(exposures, imp_fun_set,
                                                      hazard)

            imp_tmp = Impact()
            imp_tmp.calc(new_exp, new_ifs, new_haz)
            impact_meas[measure.name] = dict()
            impact_meas[measure.name]['cost'] = measure.cost
            impact_meas[measure.name]['risk'] = risk_func(imp_tmp)
            impact_meas[measure.name]['efc'] = imp_tmp.calc_freq_curve(DEF_RP)
            if save_imp:
                impact_meas[measure.name]['impact'] = imp_tmp

        # if present reference provided save it
        if when == 'future':
            self.imp_meas_future = impact_meas
        else:
            self.imp_meas_present = impact_meas

    def _calc_cost_benefit(self, disc_rates, imp_time_depen=1):
        """Compute discounted impact from present year to future year

        Parameters:
            disc_rates (DiscRates): discount rates instance
            imp_time_depen (float, optional): parameter which represent time
                evolution of impact. Default: 1 (linear).
        """
        LOGGER.info('Computing cost benefit from years %s to %s.',
                    str(self.present_year), str(self.future_year))
        # TODO add risk transfer
        # TODO add premium
        n_years = self.future_year - self.present_year + 1
        if n_years <= 0:
            LOGGER.error('Wrong year range: %s - %s.', str(self.present_year),
                         str(self.future_year))
            raise ValueError

        if not self.imp_meas_future:
            LOGGER.error('Compute first _calc_impact_measures')
            raise ValueError

        if self.imp_meas_present:
            time_dep = np.arange(n_years)**imp_time_depen / \
                (n_years-1)**imp_time_depen
        else:
            time_dep = np.ones(n_years)

        # discounted cost benefit for each measure
        for meas_name, meas_val in self.imp_meas_future.items():
            if meas_name == 'no measure':
                continue
            meas_cost, meas_risk = meas_val['cost'], meas_val['risk']
            fut_benefit = self.imp_meas_future['no measure']['risk'] - meas_risk

            if self.imp_meas_present:
                pres_benefit = self.imp_meas_present['no measure']['risk'] - \
                    self.imp_meas_present[meas_name]['risk']
                meas_ben = pres_benefit + (fut_benefit-pres_benefit) * time_dep
            else:
                meas_ben = time_dep*fut_benefit

            meas_ben = disc_rates.net_present_value(self.present_year,
                                                    self.future_year, meas_ben)
            self.benefit[meas_name] = meas_ben
            self.cost_ben_ratio[meas_name] = meas_cost/meas_ben

        # npv of the full unaverted damages
        if self.imp_meas_present:
            pres_benefit = self.imp_meas_present['no measure']['risk']
            fut_benefit = self.imp_meas_future['no measure']['risk']
            tot_climate_risk = pres_benefit + \
                (fut_benefit-pres_benefit) * time_dep
            tot_climate_risk = disc_rates.net_present_value(self.present_year,\
                self.future_year, tot_climate_risk)
        else:
            tot_climate_risk = disc_rates.net_present_value(self.present_year,\
                self.future_year, time_dep * \
                self.imp_meas_future['no measure']['risk'])
        self.tot_climate_risk = tot_climate_risk
