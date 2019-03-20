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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tabulate import tabulate

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

    Returns:
        float
    """
    return impact.aai_agg

class CostBenefit():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes:
        present_year (int): present reference year
        future_year (int): future year

        tot_climate_risk (float): total climate risk without measures
        unit (str): unit used for impact

        color_rgb (dict): color code RGB for each measure.
            Key: measure name ('no measure' used for case without measure),
            Value: np.array

        benefit (dict): benefit of each measure. Key: measure name, Value:
            float benefit
        cost_ben_ratio (dict): cost benefit ratio of each measure. Key: measure
            name, Value: float cost benefit ratio

        imp_meas_future (dict): impact of each measure at future or default.
            Key: measure name ('no measure' used for case without measure),
            Value: dict with:
                             'cost' (float): cost measure,
                             'risk' (float): risk measurement,
                             'risk_transf' (float): annual expected risk transfer,
                             'efc'  (ImpactFreqCurve): impact exceedance freq
                (optional)   'impact' (Impact): impact instance
        imp_meas_present (dict): impact of each measure at present.
            Key: measure name ('no measure' used for case without measure),
            Value: dict with:
                             'cost' (float): cost measure,
                             'risk' (float): risk measurement,
                             'risk_transf' (float): annual expected risk transfer,
                             'efc'  (ImpactFreqCurve): impact exceedance freq
                (optional)   'impact' (Impact): impact instance
    """

    def __init__(self):
        """ Initilization """
        self.present_year = DEF_PRESENT_YEAR
        self.future_year = DEF_FUTURE_YEAR

        self.tot_climate_risk = 0.0
        self.unit = 'USD'

        # dictionaries with key: measure name
        # value: measure color_rgb
        self.color_rgb = dict()
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
        future_year=2050, risk_func=risk_aai_agg, imp_time_depen=1, save_imp=False):
        """Compute cost-benefit ratio for every measure provided current
        and future conditions. Present and future measures need to have the same
        name. The measures costs need to be discounted by the user.
        If present and future entity provided, only the costs of the measures
        of the future and the discount rates of the present will be used.

        Parameters:
            hazard (Hazard): hazard
            entity (Entity): entity
            haz_future (Hazard): hazard in the future (future year provided at
                ent_future)
            ent_future (Entity): entity in the future
            future_year (int): future year to consider if no ent_future provided
            risk_func (func, optional): function describing risk measure given
                an Impact. Default: average annual impact (aggregated).
            imp_time_depen (float, optional): parameter which represent time
                evolution of impact. Default: 1 (linear).
            save_imp (bool, optional): activate if Impact of each measure is
                saved. Default: False.
        """
        # Present year given in entity. Future year in ent_future if provided.
        self.present_year = entity.exposures.ref_year
        self.unit = entity.exposures.value_unit

        # save measure colors
        for meas in entity.measures.get_measure():
            self.color_rgb[meas.name] = meas.color_rgb

        if not haz_future and not ent_future:
            self.future_year = future_year
            self._calc_impact_measures(hazard, entity.exposures, \
                entity.measures, entity.impact_funcs, 'future', \
                risk_func, save_imp)
            self._calc_cost_benefit(entity.disc_rates)
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
                self.future_year = future_year
                self._calc_impact_measures(haz_future, entity.exposures, \
                    entity.measures, entity.impact_funcs, 'future', risk_func,\
                    save_imp)
            else:
                self.future_year = ent_future.exposures.ref_year
                self._calc_impact_measures(hazard, ent_future.exposures, \
                    ent_future.measures, ent_future.impact_funcs, 'future', \
                    risk_func, save_imp)
            self._calc_cost_benefit(entity.disc_rates, imp_time_depen)

        self._print_results()

    def plot_cost_benefit(self):
        """ Plot cost-benefit graph. Call after calc()

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        fig, axis = plt.subplots(1, 1, figsize=(12,10))
        norm_fact, norm_name = self._norm_values(self.tot_climate_risk)

        m_names = list(self.cost_ben_ratio.keys())
        m_cb = np.array([self.cost_ben_ratio[name] for name in m_names])
        sort_cb = np.argsort(m_cb)

        xmin = 0
        for meas_id in sort_cb:
            meas_n = m_names[meas_id]
            rect = Rectangle((xmin, 0), self.benefit[meas_n]/norm_fact, \
                1/self.cost_ben_ratio[meas_n], color=self.color_rgb[meas_n])
            axis.add_patch(rect)
            axis.text(xmin + (self.benefit[meas_n]/norm_fact)/2,
                      0.5, meas_n, horizontalalignment='center',
                      verticalalignment='bottom', rotation=90, fontsize=12)
            xmin += self.benefit[meas_n]/norm_fact
        axis.scatter(self.tot_climate_risk/norm_fact, 0, c='r', zorder=200, clip_on=False)
        axis.text(self.tot_climate_risk/norm_fact, 1.0, 'Tot risk', horizontalalignment='center',
                  verticalalignment='bottom', rotation=90, fontsize=12, color='r')

        text_pos = self.imp_meas_future['no measure']['risk']/norm_fact
        axis.scatter(text_pos, 0, c='r', zorder=200, clip_on=False)
        axis.text(text_pos, 1.0, 'AAI', horizontalalignment='center',
                  verticalalignment='bottom', rotation=90, fontsize=12, color='r')

        axis.set_xlim(0, np.array(list(self.benefit.values())).sum()/norm_fact)
        axis.set_ylim(0, int(1/self.cost_ben_ratio[m_names[sort_cb[0]]]) + 1)
        x_label = 'NPV averted damage over ' + str(self.future_year - self.present_year + 1) + \
                  ' years (' + self.unit + ' ' + norm_name + ')'
        axis.set_xlabel(x_label)
        axis.set_ylabel('Benefit/Cost ratio')
        return fig, axis

    def plot_event_view(self, return_per=(10, 25, 100)):
        """ Plot averted damages for return periods. Call after calc()

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if not self.imp_meas_future:
            LOGGER.error('Compute COstBenefit.calc() first')
            raise ValueError
        fig, axis = plt.subplots(1, 1)
        avert_rp = dict()
        ref_imp = np.interp(return_per,
                            self.imp_meas_future['no measure']['efc'].return_per,
                            self.imp_meas_future['no measure']['efc'].impact)
        for meas_name, meas_val in self.imp_meas_future.items():
            if meas_name == 'no measure':
                continue
            interp_imp = np.interp(return_per, meas_val['efc'].return_per,
                                   meas_val['efc'].impact)
            avert_rp[meas_name] = ref_imp - interp_imp

        m_names = list(self.cost_ben_ratio.keys())
        sort_cb = np.argsort(np.array([self.cost_ben_ratio[name] for name in m_names]))
        names_sort = [m_names[i] for i in sort_cb]
        color_sort = [self.color_rgb[name] for name in names_sort]
        for rp_i, _ in enumerate(return_per):
            val_i = [avert_rp[name][rp_i] for name in names_sort]
            cum_effect = np.cumsum(np.array([0] + val_i))
            for (eff, color) in zip(cum_effect[::-1][:-1], color_sort[::-1]):
                plt.bar(rp_i+1, eff, color=color)
            plt.bar(rp_i+1, ref_imp[rp_i], edgecolor='k', fc=(1, 0, 0, 0))
        axis.set_xlabel('Return Period')
        axis.set_ylabel('Impact ('+ self.unit + ')')
        plt.xticks(np.arange(len(return_per))+1, return_per)
        return fig, axis

    def plot_waterfall(self, hazard, entity, haz_future, ent_future,
                       risk_func=risk_aai_agg, imp_time_depen=1):
        """ Plot waterfall graph. Can be called before and after calc()

        Parameters:
            hazard (Hazard): hazard
            entity (Entity): entity
            haz_future (Hazard): hazard in the future (future year provided at
                ent_future)
            ent_future (Entity): entity in the future
            risk_func (func, optional): function describing risk measure given
                an Impact. Default: average annual impact (aggregated).
            imp_time_depen (float, optional): parameter which represent time
                evolution of impact. Default: 1 (linear).

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if ent_future.exposures.ref_year == entity.exposures.ref_year:
            LOGGER.error('Same reference years for future and present entities.')
            raise ValueError
        self.present_year = entity.exposures.ref_year
        self.future_year = ent_future.exposures.ref_year

        if not self.imp_meas_present:
            imp = Impact()
            imp.calc(entity.exposures, entity.impact_funcs, hazard)
            curr_risk = risk_func(imp)
        else:
            curr_risk = self.imp_meas_present['no measure']['risk']

        if not self.imp_meas_future:
            imp = Impact()
            imp.calc(ent_future.exposures, ent_future.impact_funcs, haz_future)
            fut_risk = risk_func(imp)
        else:
            fut_risk = self.imp_meas_future['no measure']['risk']

        fig, axis = plt.subplots(1, 1)
        norm_fact, norm_name = self._norm_values(curr_risk)

        # current situation
        risk_future = curr_risk
        time_dep = self._time_dependency_array()
        risk_curr = self._npv_unaverted_impact(risk_future, entity.disc_rates,
                                               time_dep)

        # changing future
        time_dep = self._time_dependency_array(imp_time_depen)
        # socio-economic dev
        imp = Impact()
        imp.calc(ent_future.exposures, ent_future.impact_funcs, hazard)
        risk_future = risk_func(imp)
        risk_dev = self._npv_unaverted_impact(risk_future, entity.disc_rates,
                                              time_dep, curr_risk)
        # socioecon + cc
        risk_future = fut_risk
        risk_tot = self._npv_unaverted_impact(risk_future, entity.disc_rates,
                                              time_dep, curr_risk)

        axis.bar(1, risk_curr/norm_fact)
        axis.text(1, risk_curr/norm_fact, str(int(round(risk_curr/norm_fact))), \
            horizontalalignment='center', verticalalignment='bottom', \
            fontsize=12, color='k')
        axis.bar(2, height=(risk_dev-risk_curr)/norm_fact, bottom=risk_curr/norm_fact)
        axis.text(2, risk_curr/norm_fact + (risk_dev-risk_curr)/norm_fact/2, \
            str(int(round((risk_dev-risk_curr)/norm_fact))), \
            horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
        axis.bar(3, height=(risk_tot-risk_dev)/norm_fact, bottom=risk_dev/norm_fact)
        axis.text(3, risk_dev/norm_fact + (risk_tot-risk_dev)/norm_fact/2, \
            str(int(round((risk_tot-risk_dev)/norm_fact))), \
            horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
        axis.bar(4, height=risk_tot/norm_fact)
        axis.text(4, risk_tot/norm_fact, str(int(round(risk_tot/norm_fact))), \
                  horizontalalignment='center', verticalalignment='bottom', \
                  fontsize=12, color='k')
        plt.xticks(np.arange(4)+1, ['Risk ' + str(self.present_year), \
            'Economic \ndevelopment', 'Climate \nchange', 'Risk ' + str(self.future_year)])
        axis.set_ylabel('Impact (' + self.unit + ' ' + norm_name + ')')

        return fig, axis

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
        impact_meas['no measure']['risk_transf'] = 0.0
        impact_meas['no measure']['efc'] = imp_tmp.calc_freq_curve(DEF_RP)
        if save_imp:
            impact_meas['no measure']['impact'] = imp_tmp

        # compute impact for each measure
        for measure in meas_set.get_measure():
            imp_tmp, risk_transf = measure.calc_impact(exposures, imp_fun_set, hazard)
            impact_meas[measure.name] = dict()
            impact_meas[measure.name]['cost'] = measure.cost
            impact_meas[measure.name]['risk'] = risk_func(imp_tmp)
            impact_meas[measure.name]['risk_transf'] = risk_transf
            impact_meas[measure.name]['efc'] = imp_tmp.calc_freq_curve(DEF_RP)
            if save_imp:
                impact_meas[measure.name]['impact'] = imp_tmp

        # if present reference provided save it
        if when == 'future':
            self.imp_meas_future = impact_meas
        else:
            self.imp_meas_present = impact_meas

    def _calc_cost_benefit(self, disc_rates, imp_time_depen=None):
        """Compute discounted impact from present year to future year

        Parameters:
            disc_rates (DiscRates): discount rates instance
            imp_time_depen (float, optional): parameter which represent time
                evolution of impact
        """
        LOGGER.info('Computing cost benefit from years %s to %s.',
                    str(self.present_year), str(self.future_year))

        if self.future_year - self.present_year + 1 <= 0:
            LOGGER.error('Wrong year range: %s - %s.', str(self.present_year),
                         str(self.future_year))
            raise ValueError

        if not self.imp_meas_future:
            LOGGER.error('Compute first _calc_impact_measures')
            raise ValueError

        time_dep = self._time_dependency_array(imp_time_depen)

        # discounted cost benefit for each measure and total climate risk
        for meas_name, meas_val in self.imp_meas_future.items():
            if meas_name == 'no measure':
                # npv of the full unaverted damages
                if self.imp_meas_present:
                    self.tot_climate_risk = self._npv_unaverted_impact(
                        self.imp_meas_future['no measure']['risk'], \
                        disc_rates, time_dep, self.imp_meas_present['no measure']['risk'])
                else:
                    self.tot_climate_risk = self._npv_unaverted_impact(
                        self.imp_meas_future['no measure']['risk'], \
                        disc_rates, time_dep)
                continue

            fut_benefit = self.imp_meas_future['no measure']['risk'] - meas_val['risk']
            fut_risk_tr = meas_val['risk_transf']
            if self.imp_meas_present:
                pres_benefit = self.imp_meas_present['no measure']['risk'] - \
                    self.imp_meas_present[meas_name]['risk']
                meas_ben = pres_benefit + (fut_benefit-pres_benefit) * time_dep

                pres_risk_tr = self.imp_meas_present[meas_name]['risk_transf']
                risk_tr = pres_risk_tr + (fut_risk_tr-pres_risk_tr) * time_dep
            else:
                meas_ben = time_dep*fut_benefit
                risk_tr = time_dep*fut_risk_tr

            # discount
            meas_ben = disc_rates.net_present_value(self.present_year,
                                                    self.future_year, meas_ben)
            risk_tr = disc_rates.net_present_value(self.present_year,
                                                   self.future_year, risk_tr)
            self.benefit[meas_name] = meas_ben
            self.cost_ben_ratio[meas_name] = (meas_val['cost']+risk_tr)/meas_ben

    def _time_dependency_array(self, imp_time_depen=None):
        """ Construct time dependency array. Each year contains a value in [0,1]
        representing the rate of damage difference achieved that year, according
        to the growth represented by parameter imp_time_depen.

        Parameters:
            imp_time_depen (float, optional): parameter which represent time
                evolution of impact. Time array is all ones if not provided
        """
        n_years = self.future_year - self.present_year + 1
        if imp_time_depen:
            time_dep = np.arange(n_years)**imp_time_depen / \
                (n_years-1)**imp_time_depen
        else:
            time_dep = np.ones(n_years)
        return time_dep

    def _npv_unaverted_impact(self, risk_future, disc_rates, time_dep,
                              risk_present=None):
        """ Net present value of total unaverted damages

        Parameters:
            risk_future (float): risk under future situation
            disc_rates (DiscRates): discount rates object
            time_dep (np.array): values in 0-1 indicating impact growth at each
                year
            risk_present (float): risk under current situation

        Returns:
            float
        """
        if risk_present:
            tot_climate_risk = risk_present + (risk_future-risk_present) * time_dep
            tot_climate_risk = disc_rates.net_present_value(self.present_year, \
                self.future_year, tot_climate_risk)
        else:
            tot_climate_risk = disc_rates.net_present_value(self.present_year, \
                self.future_year, time_dep * risk_future)
        return tot_climate_risk

    def _print_results(self):
        """ Print table with main results """
        norm_fact, norm_name = self._norm_values(np.array(list(self.benefit.values())).max())
        norm_name = '(' + self.unit + ' ' + norm_name + ')'

        table = []
        headers = ['Measure', 'Cost ' + norm_name, 'Benefit ' + norm_name, 'Benefit/Cost']
        for meas_name in self.benefit.keys():
            table.append([meas_name,
            self.cost_ben_ratio[meas_name]*self.benefit[meas_name]/norm_fact,
            self.benefit[meas_name]/norm_fact, 1/self.cost_ben_ratio[meas_name]])
        print()
        print(tabulate(table, headers, tablefmt="simple"))

        table = []
        table.append(['Total climate risk:',
                      self.tot_climate_risk/norm_fact, norm_name])
        table.append(['Average annual risk:',
                      self.imp_meas_future['no measure']['risk']/norm_fact, norm_name])
        table.append(['Residual damage:',
                      (self.tot_climate_risk -
                       np.array(list(self.benefit.values())).sum())/norm_fact, norm_name])
        print()
        print(tabulate(table, tablefmt="simple"))

    @staticmethod
    def _norm_values(value):
        """ Compute normalization value and name

        Parameters:
            value (float): value to normalize

        Returns:
            norm_fact, norm_name
        """
        norm_fact = 1.
        norm_name = ''
        if value/1.0e9 > 1:
            norm_fact = 1.0e9
            norm_name = 'bn'
        elif value/1.0e6 > 1:
            norm_fact = 1.0e6
            norm_name = 'm'
        elif value/1.0e3 > 1:
            norm_fact = 1.0e3
            norm_name = 'k'
        return norm_fact, norm_name
