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

Define CostBenefit class.
"""

__all__ = ['CostBenefit', 'risk_aai_agg', 'risk_rp_100', 'risk_rp_250']

import copy
import logging
from typing import Optional, Dict, Tuple, Union

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from tabulate import tabulate

from climada.engine.impact_calc import ImpactCalc
from climada.engine import Impact, ImpactFreqCurve

LOGGER = logging.getLogger(__name__)

DEF_PRESENT_YEAR = 2016
"""Default present reference year"""

DEF_FUTURE_YEAR = 2030
"""Default future reference year"""

NO_MEASURE = 'no measure'
"""Name of risk metrics when no measure is applied"""

def risk_aai_agg(impact):
    """Risk measurement as average annual impact aggregated.

    Parameters
    ----------
    impact : climada.engine.Impact
        an Impact instance

    Returns
    -------
    float
    """
    return impact.aai_agg

def risk_rp_100(impact):
    """Risk measurement as exceedance impact at 100 years return period.

    Parameters
    ----------
    impact : climada.engine.Impact
        an Impact instance

    Returns
    -------
    float
    """
    if impact.at_event.size > 0:
        efc = impact.calc_freq_curve([100])
        return efc.impact[0]
    return 0

def risk_rp_250(impact):
    """Risk measurement as exceedance impact at 250 years return period.

    Parameters
    ----------
    impact : climada.engine.Impact
        an Impact instance

    Returns
    -------
    float
    """
    if impact.at_event.size > 0:
        efc = impact.calc_freq_curve([250])
        return efc.impact[0]
    return 0

class CostBenefit():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes
    ----------
    present_year : int
        present reference year
    future_year : int
        future year
    tot_climate_risk : float
        total climate risk without measures
    unit : str
        unit used for impact
    color_rgb : dict
        color code RGB for each measure.
    Key : str
        measure name ('no measure' used for case without measure),
    Value : np.array
    benefit : dict
        benefit of each measure. Key: measure name, Value: float benefit
    cost_ben_ratio : dict
        cost benefit ratio of each measure. Key: measure
        name, Value: float cost benefit ratio
    imp_meas_future : dict
        impact of each measure at future or default.
        Key: measure name ('no measure' used for case without measure),
        Value: dict with:
        'cost' (tuple): (cost measure, cost factor insurance),
        'risk' (float): risk measurement,
        'risk_transf' (float): annual expected risk transfer,
        'efc'  (ImpactFreqCurve): impact exceedance freq (optional)
        'impact' (Impact): impact instance
    imp_meas_present : dict
        impact of each measure at present.
        Key: measure name ('no measure' used for case without measure),
        Value: dict with:
        'cost' (tuple): (cost measure, cost factor insurance),
        'risk' (float): risk measurement,
        'risk_transf' (float): annual expected risk transfer,
        'efc'  (ImpactFreqCurve): impact exceedance freq (optional)
        'impact' (Impact): impact instance
    """

    def __init__(
        self,
        present_year: int = DEF_PRESENT_YEAR,
        future_year: int = DEF_FUTURE_YEAR,
        tot_climate_risk: float = 0.0,
        unit: str = 'USD',
        color_rgb: Optional[Dict[str, np.ndarray]] = None,
        benefit: Optional[Dict[str, float]] = None,
        cost_ben_ratio: Optional[Dict[str, float]] = None,
        imp_meas_present: Optional[Dict[str,
            Union[float, Tuple[float, float], Impact, ImpactFreqCurve]]] = None,
        imp_meas_future: Optional[Dict[str,
            Union[float, Tuple[float, float], Impact, ImpactFreqCurve]]] = None,
    ):
        """Initilization"""
        self.present_year = present_year
        self.future_year = future_year
        self.tot_climate_risk = tot_climate_risk
        self.unit = unit

        # dictionaries with key: measure name
        # value: measure color_rgb
        self.color_rgb = color_rgb if color_rgb is not None else dict()
        # value: measure benefit
        self.benefit = color_rgb if color_rgb is not None else dict()
        # value: measure cost benefit
        self.cost_ben_ratio = cost_ben_ratio if cost_ben_ratio is not None else dict()
        self.benefit = benefit if benefit is not None else dict()

        # 'no measure' key for impact without measures
        # values: dictionary with 'cost': cost measure,
        #                         'risk': risk measurement,
        #                         'efc': ImpactFreqCurve
        #          (optionally)   'impact': Impact
        self.imp_meas_future = imp_meas_future if imp_meas_future is not None else dict()
        self.imp_meas_present = imp_meas_present if imp_meas_present is not None else dict()

    def calc(self, hazard, entity, haz_future=None, ent_future=None, future_year=None,
             risk_func=risk_aai_agg, imp_time_depen=None, save_imp=False, assign_centroids=True):
        """Compute cost-benefit ratio for every measure provided current
        and, optionally, future conditions. Present and future measures need
        to have the same name. The measures costs need to be discounted by the user.
        If future entity provided, only the costs of the measures
        of the future and the discount rates of the present will be used.

        Parameters
        ----------
        hazard : climada.Hazard
        entity : climada.entity
        haz_future : climada.Hazard, optional
            hazard in the future (future year provided at ent_future)
        ent_future : Entity, optional
            entity in the future. Default is None
        future_year : int, optional
            future year to consider if no ent_future. Default is None
            provided. The benefits are added from the entity.exposures.ref_year until
            ent_future.exposures.ref_year, or until future_year if no ent_future given.
            Default: entity.exposures.ref_year+1
        risk_func : func optional
            function describing risk measure to use
            to compute the annual benefit from the Impact.
            Default: average annual impact (aggregated).
        imp_time_depen : float, optional
            parameter which represents time
            evolution of impact (super- or sublinear). If None: all years
            count the same when there is no future hazard nor entity and 1
            (linear annual change) when there is future hazard or entity.
            Default is None.
        save_imp : bool, optional
            Default: False
        assign_centroids : bool, optional
            indicates whether centroids are assigned to the self.exposures object.
            Centroids assignment is an expensive operation; set this to ``False`` to save
            computation time if the exposures from ``ent`` and ``ent_fut`` have already
            centroids assigned for the respective hazards.
            Default: True
        True if Impact of each measure is saved. Default is False.
        """
        # Present year given in entity. Future year in ent_future if provided.
        self.present_year = entity.exposures.ref_year
        self.unit = entity.exposures.value_unit

        # save measure colors
        for meas in entity.measures.get_measure(hazard.tag.haz_type):
            self.color_rgb[meas.name] = meas.color_rgb
        self.color_rgb[NO_MEASURE] = colors.to_rgb('deepskyblue')

        if future_year is None and ent_future is None:
            future_year = entity.exposures.ref_year

        # assign centroids
        if assign_centroids:
            entity.exposures.assign_centroids(hazard, overwrite=True)
            if ent_future:
                ent_future.exposures.assign_centroids(
                    haz_future if haz_future else hazard, overwrite=True
                )

        if not haz_future and not ent_future:
            self.future_year = future_year
            self._calc_impact_measures(hazard, entity.exposures,
                                       entity.measures, entity.impact_funcs, 'future',
                                       risk_func, save_imp)
        else:
            if imp_time_depen is None:
                imp_time_depen = 1
            self._calc_impact_measures(hazard, entity.exposures,
                                       entity.measures, entity.impact_funcs, 'present',
                                       risk_func, save_imp)
            if haz_future and ent_future:
                self.future_year = ent_future.exposures.ref_year
                self._calc_impact_measures(haz_future, ent_future.exposures,
                                           ent_future.measures, ent_future.impact_funcs, 'future',
                                           risk_func, save_imp)
            elif haz_future:
                self.future_year = future_year
                self._calc_impact_measures(haz_future, entity.exposures,
                                           entity.measures, entity.impact_funcs, 'future',
                                           risk_func, save_imp)
            else:
                self.future_year = ent_future.exposures.ref_year
                self._calc_impact_measures(hazard, ent_future.exposures,
                                           ent_future.measures, ent_future.impact_funcs, 'future',
                                           risk_func, save_imp)

        self._calc_cost_benefit(entity.disc_rates, imp_time_depen)
        self._print_results()
        self._print_npv()

    def combine_measures(self, in_meas_names, new_name, new_color, disc_rates,
                         imp_time_depen=None, risk_func=risk_aai_agg):
        """Compute cost-benefit of the combination of measures previously
        computed by calc with save_imp=True. The benefits of the
        measures per event are added. To combine with risk transfer options use
        apply_risk_transfer.

        Parameters
        ----------
        in_meas_names : list(str)
        list with names of measures to combine
        new_name :  str
            name to give to the new resulting measure
            new_color (np.array): color code RGB for new measure, e.g.
            np.array([0.1, 0.1, 0.1])
        disc_rates : DiscRates
            discount rates instance
        imp_time_depen : float, optional
            parameter which represents time
            evolution of impact (super- or sublinear). If None: all years
            count the same when there is no future hazard nor entity and 1
            (linear annual change) when there is future hazard or entity.
            Default is None.
        risk_func : func, optional
            function describing risk measure given
            an Impact. Default: average annual impact (aggregated).

        Returns
        -------
        climada.CostBenefit
        """
        # pylint: disable=protected-access
        new_cb = CostBenefit(
            present_year=self.present_year,
            future_year=self.future_year,
            unit=self.unit,
            tot_climate_risk=self.tot_climate_risk,
            color_rgb=self.color_rgb,
            imp_meas_future=self.imp_meas_future,
        )
        new_cb.color_rgb[new_name] = new_color

        # compute impacts for imp_meas_future and imp_meas_present
        self._combine_imp_meas(new_cb, in_meas_names, new_name, risk_func, when='future')
        if self.imp_meas_present:
            new_cb.imp_meas_present[NO_MEASURE] = self.imp_meas_present[NO_MEASURE]
            if imp_time_depen is None:
                imp_time_depen = 1
            self._combine_imp_meas(new_cb, in_meas_names, new_name, risk_func, when='present')

        # cost-benefit computation: fill measure's benefit and cost_ben_ratio
        time_dep = new_cb._time_dependency_array(imp_time_depen)
        new_cb._cost_ben_one(new_name, new_cb.imp_meas_future[new_name], disc_rates,
                             time_dep)
        new_cb._print_results()
        new_cb._print_npv()
        return new_cb

    def apply_risk_transfer(self, meas_name, attachment, cover, disc_rates,
                            cost_fix=0, cost_factor=1, imp_time_depen=None,
                            risk_func=risk_aai_agg):
        """Applies risk transfer to given measure computed before with saved
        impact and compares it to when no measure is applied. Appended to
        dictionaries of measures.

        Parameters
        ----------
        meas_name : str
            name of measure where to apply risk transfer
        attachment : float
            risk transfer values attachment (deductible)
        cover : float
            risk transfer cover
        cost_fix : float
            fixed cost of implemented innsurance, e.g. transaction costs
        cost_factor : float, optional
            factor to which to multiply the insurance layer
            to compute its cost. Default is 1
        imp_time_depen : float, optional
            parameter which represents time
            evolution of impact (super- or sublinear). If None: all years
            count the same when there is no future hazard nor entity and 1
            (linear annual change) when there is future hazard or entity.
            Default is None.
        risk_func : func, optional
            function describing risk measure given
            an Impact. Default: average annual impact (aggregated).
        """
        m_transf_name = 'risk transfer (' + meas_name + ')'
        self.color_rgb[m_transf_name] = np.maximum(np.minimum(self.color_rgb[meas_name] -
                                                              np.ones(3) * 0.2, 1), 0)

        _, layer_no = self.imp_meas_future[NO_MEASURE]['impact']. \
            calc_risk_transfer(attachment, cover)
        layer_no = risk_func(layer_no)

        imp, layer = self.imp_meas_future[meas_name]['impact']. \
            calc_risk_transfer(attachment, cover)
        self.imp_meas_future[m_transf_name] = dict()
        self.imp_meas_future[m_transf_name]['risk_transf'] = risk_func(layer)
        self.imp_meas_future[m_transf_name]['impact'] = imp
        self.imp_meas_future[m_transf_name]['risk'] = risk_func(imp)
        self.imp_meas_future[m_transf_name]['cost'] = (cost_fix, cost_factor)
        self.imp_meas_future[m_transf_name]['efc'] = imp.calc_freq_curve()

        if self.imp_meas_present:
            if imp_time_depen is None:
                imp_time_depen = 1
            time_dep = self._time_dependency_array(imp_time_depen)
            _, pres_layer_no = self.imp_meas_present[NO_MEASURE]['impact']. \
                calc_risk_transfer(attachment, cover)
            pres_layer_no = risk_func(pres_layer_no)
            layer_no = pres_layer_no + (layer_no - pres_layer_no) * time_dep

            imp, layer = self.imp_meas_present[meas_name]['impact']. \
                calc_risk_transfer(attachment, cover)
            self.imp_meas_present[m_transf_name] = dict()
            self.imp_meas_present[m_transf_name]['risk_transf'] = risk_func(layer)
            self.imp_meas_present[m_transf_name]['impact'] = imp
            self.imp_meas_present[m_transf_name]['risk'] = risk_func(imp)
            self.imp_meas_present[m_transf_name]['cost'] = (cost_fix, cost_factor)
            self.imp_meas_present[m_transf_name]['efc'] = imp.calc_freq_curve()
        else:
            time_dep = self._time_dependency_array(imp_time_depen)
            layer_no = time_dep * layer_no

        self._cost_ben_one(m_transf_name, self.imp_meas_future[m_transf_name],
                           disc_rates, time_dep, ini_state=meas_name)

        # compare layer no measure
        layer_no = disc_rates.net_present_value(self.present_year,
                                                self.future_year, layer_no)
        layer = ((self.cost_ben_ratio[m_transf_name] * self.benefit[m_transf_name] - cost_fix)
                 / cost_factor)
        self._print_results()
        self._print_risk_transfer(layer, layer_no, cost_fix, cost_factor)
        self._print_npv()

    def remove_measure(self, meas_name):
        """Remove computed values of given measure

        Parameters
        ----------
        meas_name : str
            name of measure to remove
        """
        del self.color_rgb[meas_name]
        del self.benefit[meas_name]
        del self.cost_ben_ratio[meas_name]
        del self.imp_meas_future[meas_name]
        if self.imp_meas_present:
            del self.imp_meas_present[meas_name]

    def plot_cost_benefit(self, cb_list=None, axis=None, **kwargs):
        """Plot cost-benefit graph. Call after calc().

        Parameters
        ----------
        cb_list : list(CostBenefit), optional
            if other CostBenefit
            provided, overlay them all. Used for uncertainty visualization.
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
            arguments for Rectangle matplotlib, e.g. alpha=0.5
            (color is set by measures color attribute)

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        if cb_list:
            if 'alpha' not in kwargs:
                kwargs['alpha'] = 0.5
            cb_uncer = [self]
            cb_uncer.extend(cb_list)
            axis = self._plot_list_cost_ben(cb_uncer, axis, **kwargs)
            return axis

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 1.0
        axis = self._plot_list_cost_ben([self], axis, **kwargs)
        norm_fact, norm_name = _norm_values(self.tot_climate_risk + 0.01)

        text_pos = self.imp_meas_future[NO_MEASURE]['risk'] / norm_fact
        axis.scatter(text_pos, 0, c='r', zorder=200, clip_on=False)
        axis.text(text_pos, 0, '  AAI', horizontalalignment='center',
                  verticalalignment='bottom', rotation=90, fontsize=12, color='r')
        if abs(text_pos - self.tot_climate_risk / norm_fact) > 1:
            axis.scatter(self.tot_climate_risk / norm_fact, 0, c='r', zorder=200, clip_on=False)
            axis.text(self.tot_climate_risk / norm_fact, 0, '  Tot risk',
                      horizontalalignment='center', verticalalignment='bottom', rotation=90,
                      fontsize=12, color='r')

        axis.set_xlim(0, max(self.tot_climate_risk / norm_fact,
                             np.array(list(self.benefit.values())).sum() / norm_fact))
        axis.set_ylim(0, int(1 / np.nanmin(np.ma.masked_equal(np.array(list(
            self.cost_ben_ratio.values())), 0))) + 1)

        x_label = ('NPV averted damage over ' + str(self.future_year - self.present_year + 1)
                   + ' years (' + self.unit + ' ' + norm_name + ')')
        axis.set_xlabel(x_label)
        axis.set_ylabel('Benefit/Cost ratio')
        return axis

    def plot_event_view(self, return_per=(10, 25, 100), axis=None, **kwargs):
        """Plot averted damages for return periods. Call after calc().

        Parameters
        ----------
        return_per : list, optional
            years to visualize. Default 10, 25, 100
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
            arguments for bar matplotlib function, e.g. alpha=0.5
            (color is set by measures color attribute)

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        if not self.imp_meas_future:
            raise ValueError('Compute CostBenefit.calc() first')
        if not axis:
            _, axis = plt.subplots(1, 1)
        avert_rp = dict()
        for meas_name, meas_val in self.imp_meas_future.items():
            if meas_name == NO_MEASURE:
                continue
            interp_imp = np.interp(return_per, meas_val['efc'].return_per,
                                   meas_val['efc'].impact)
            # check if measure over no measure or combined with another measure
            try:
                ref_meas = meas_name[meas_name.index('(') + 1:meas_name.index(')')]
            except ValueError:
                ref_meas = NO_MEASURE
            ref_imp = np.interp(return_per,
                                self.imp_meas_future[ref_meas]['efc'].return_per,
                                self.imp_meas_future[ref_meas]['efc'].impact)
            avert_rp[meas_name] = ref_imp - interp_imp

        m_names = list(self.cost_ben_ratio.keys())
        sort_cb = np.argsort(np.array([self.cost_ben_ratio[name] for name in m_names]))
        names_sort = [m_names[i] for i in sort_cb]
        color_sort = [self.color_rgb[name] for name in names_sort]
        ref_imp = np.interp(return_per, self.imp_meas_future[NO_MEASURE]['efc'].return_per,
                            self.imp_meas_future[NO_MEASURE]['efc'].impact)
        for rp_i, _ in enumerate(return_per):
            val_i = [avert_rp[name][rp_i] for name in names_sort]
            cum_effect = np.cumsum(np.array([0] + val_i))
            for (eff, color) in zip(cum_effect[::-1][:-1], color_sort[::-1]):
                axis.bar(rp_i + 1, eff, color=color, **kwargs)
            axis.bar(rp_i + 1, ref_imp[rp_i], edgecolor='k', fc=(1, 0, 0, 0), zorder=100)
        axis.set_xlabel('Return Period (%s)' % str(self.future_year))
        axis.set_ylabel('Impact (' + self.unit + ')')
        axis.set_xticks(np.arange(len(return_per)) + 1)
        axis.set_xticklabels([str(per) for per in return_per])
        return axis

    @staticmethod
    def plot_waterfall(hazard, entity, haz_future, ent_future,
                       risk_func=risk_aai_agg, axis=None, **kwargs):
        """Plot waterfall graph at future with given risk metric. Can be called
        before and after calc().

        Parameters
        ----------
        hazard : climada.Hazard
        entity : climada.Entity
        haz_future : Hazard
            hazard in the future (future year provided at ent_future).
            ``haz_future`` is expected to have the same centroids as ``hazard``.
        ent_future : climada.Entity
            entity in the future
        risk_func : func, optional
            function describing risk measure given
            an Impact. Default: average annual impact (aggregated).
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
            arguments for bar matplotlib function, e.g. alpha=0.5

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        if ent_future.exposures.ref_year == entity.exposures.ref_year:
            raise ValueError('Same reference years for future and present entities.')
        present_year = entity.exposures.ref_year
        future_year = ent_future.exposures.ref_year

        imp = ImpactCalc(entity.exposures, entity.impact_funcs, hazard)\
              .impact(assign_centroids=hazard.centr_exp_col not in entity.exposures.gdf)
        curr_risk = risk_func(imp)

        imp = ImpactCalc(ent_future.exposures, ent_future.impact_funcs, haz_future)\
              .impact(assign_centroids=hazard.centr_exp_col not in ent_future.exposures.gdf)
        fut_risk = risk_func(imp)

        if not axis:
            _, axis = plt.subplots(1, 1)
        norm_fact, norm_name = _norm_values(curr_risk)

        # current situation
        LOGGER.info('Risk at {:d}: {:.3e}'.format(present_year, curr_risk))

        # changing future
        # socio-economic dev
        imp = ImpactCalc(ent_future.exposures, ent_future.impact_funcs, hazard)\
              .impact(assign_centroids=False)
        risk_dev = risk_func(imp)
        LOGGER.info('Risk with development at {:d}: {:.3e}'.format(future_year, risk_dev))

        # socioecon + cc
        LOGGER.info('Risk with development and climate change at {:d}: {:.3e}'.
                    format(future_year, fut_risk))

        axis.bar(1, curr_risk / norm_fact, **kwargs)
        axis.text(1, curr_risk / norm_fact, str(int(round(curr_risk / norm_fact))),
                  horizontalalignment='center', verticalalignment='bottom',
                  fontsize=12, color='k')
        axis.bar(2, height=(risk_dev - curr_risk) / norm_fact,
                 bottom=curr_risk / norm_fact, **kwargs)
        axis.text(2, curr_risk / norm_fact + (risk_dev - curr_risk) / norm_fact / 2,
                  str(int(round((risk_dev - curr_risk) / norm_fact))),
                  horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
        axis.bar(3, height=(fut_risk - risk_dev) / norm_fact,
                 bottom=risk_dev / norm_fact, **kwargs)
        axis.text(3, risk_dev / norm_fact + (fut_risk - risk_dev) / norm_fact / 2,
                  str(int(round((fut_risk - risk_dev) / norm_fact))),
                  horizontalalignment='center', verticalalignment='center', fontsize=12,
                  color='k')
        axis.bar(4, height=fut_risk / norm_fact, **kwargs)
        axis.text(4, fut_risk / norm_fact, str(int(round(fut_risk / norm_fact))),
                  horizontalalignment='center', verticalalignment='bottom',
                  fontsize=12, color='k')

        axis.set_xticks(np.arange(4) + 1)
        axis.set_xticklabels(['Risk ' + str(present_year),
                              'Economic \ndevelopment',
                              'Climate \nchange',
                              'Risk ' + str(future_year)])
        axis.set_ylabel('Impact (' + imp.unit + ' ' + norm_name + ')')
        axis.set_title('Risk at {:d} and {:d}'.format(present_year, future_year))
        return axis

    def plot_arrow_averted(self, axis, in_meas_names=None, accumulate=False, combine=False,
                           risk_func=risk_aai_agg, disc_rates=None, imp_time_depen=1, **kwargs):
        """Plot waterfall graph with accumulated values from present to future
        year. Call after calc() with save_imp=True.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot
            axis from plot_waterfall
            or plot_waterfall_accumulated where arrow will be added to last bar
        in_meas_names : list(str), optional
            list with names of measures to
            represented total averted damage. Default: all measures
        accumulate : bool, optional)
            accumulated averted damage (True) or averted
            damage in future (False). Default: False
        combine : bool, optional
            use combine_measures to compute total averted
            damage (True) or just add benefits (False). Default: False
        risk_func : func, optional
            function describing risk measure given
            an Impact used in combine_measures. Default: average annual impact (aggregated).
        disc_rates : DiscRates, optional
            discount rates used in combine_measures
        imp_time_depen : float, optional
            parameter which represent time
            evolution of impact used in combine_measures. Default: 1 (linear).
        kwargs : optional
            arguments for bar matplotlib function, e.g. alpha=0.5
        """
        if not in_meas_names:
            in_meas_names = list(self.benefit.keys())
        bars = [rect for rect in axis.get_children() if isinstance(rect, Rectangle)]

        if accumulate:
            tot_benefit = np.array([self.benefit[meas] for meas in in_meas_names]).sum()
            norm_fact = self.tot_climate_risk / bars[3].get_height()
        else:
            tot_benefit = np.array([risk_func(self.imp_meas_future[NO_MEASURE]['impact']) -
                                    risk_func(self.imp_meas_future[meas]['impact'])
                                    for meas in in_meas_names]).sum()
            norm_fact = (risk_func(self.imp_meas_future['no measure']['impact'])
                         / bars[3].get_height())
        if combine:
            try:
                LOGGER.info('Combining measures %s', in_meas_names)
                all_meas = self.combine_measures(in_meas_names, 'combine',
                                                 colors.to_rgba('black'), disc_rates,
                                                 imp_time_depen, risk_func)
            except KeyError:
                LOGGER.warning('Use calc() with save_imp=True to get a more accurate '
                               'approximation of total averted damage,')
            if accumulate:
                tot_benefit = all_meas.benefit['combine']
            else:
                tot_benefit = risk_func(all_meas.imp_meas_future[NO_MEASURE]['impact']) - \
                    risk_func(all_meas.imp_meas_future['combine']['impact'])

        self._plot_averted_arrow(axis, bars[3], tot_benefit, bars[3].get_height() * norm_fact,
                                 norm_fact, **kwargs)

    def plot_waterfall_accumulated(self, hazard, entity, ent_future,
                                   risk_func=risk_aai_agg, imp_time_depen=1,
                                   axis=None, **kwargs):
        """Plot waterfall graph with accumulated values from present to future
        year. Call after calc() with save_imp=True. Provide same inputs as in calc.

        Parameters
        ----------
        hazard : climada.Hazard
        entity : climada.Entity
        ent_future : climada.Entity
            entity in the future
        risk_func : func, optional
            function describing risk measure given an Impact.
            Default: average annual impact (aggregated).
        imp_time_depen : float, optional
            parameter which represent time
            evolution of impact used in combine_measures. Default: 1 (linear).
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
            arguments for bar matplotlib function, e.g. alpha=0.5

        Returns
        -------
            matplotlib.axes._subplots.AxesSubplot
        """
        if not self.imp_meas_future or not self.imp_meas_present:
            raise ValueError('Compute CostBenefit.calc() first')
        if ent_future.exposures.ref_year == entity.exposures.ref_year:
            raise ValueError('Same reference years for future and present entities.')

        self.present_year = entity.exposures.ref_year
        self.future_year = ent_future.exposures.ref_year

        # current situation
        curr_risk = self.imp_meas_present[NO_MEASURE]['risk']
        time_dep = self._time_dependency_array()
        risk_curr = self._npv_unaverted_impact(curr_risk, entity.disc_rates,
                                               time_dep)
        LOGGER.info('Current total risk at {:d}: {:.3e}'.format(self.future_year,
                                                                risk_curr))

        # changing future
        time_dep = self._time_dependency_array(imp_time_depen)
        # socio-economic dev
        imp = ImpactCalc(ent_future.exposures, ent_future.impact_funcs, hazard)\
              .impact(assign_centroids=False)
        risk_dev = self._npv_unaverted_impact(risk_func(imp), entity.disc_rates,
                                              time_dep, curr_risk)
        LOGGER.info('Total risk with development at {:d}: {:.3e}'.format(
            self.future_year, risk_dev))

        # socioecon + cc
        risk_tot = self._npv_unaverted_impact(self.imp_meas_future[NO_MEASURE]['risk'],
                                              entity.disc_rates, time_dep, curr_risk)
        LOGGER.info('Total risk with development and climate change at {:d}: {:.3e}'.
                    format(self.future_year, risk_tot))

        # plot
        if not axis:
            _, axis = plt.subplots(1, 1)
        norm_fact, norm_name = _norm_values(curr_risk)
        axis.bar(1, risk_curr / norm_fact, **kwargs)
        axis.text(1, risk_curr / norm_fact, str(int(round(risk_curr / norm_fact))),
                  horizontalalignment='center', verticalalignment='bottom',
                  fontsize=12, color='k')
        axis.bar(2, height=(risk_dev - risk_curr) / norm_fact,
                 bottom=risk_curr / norm_fact, **kwargs)
        axis.text(2, risk_curr / norm_fact + (risk_dev - risk_curr) / norm_fact / 2,
                  str(int(round((risk_dev - risk_curr) / norm_fact))),
                  horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
        axis.bar(3, height=(risk_tot - risk_dev) / norm_fact,
                 bottom=risk_dev / norm_fact, **kwargs)
        axis.text(3, risk_dev / norm_fact + (risk_tot - risk_dev) / norm_fact / 2,
                  str(int(round((risk_tot - risk_dev) / norm_fact))),
                  horizontalalignment='center', verticalalignment='center', fontsize=12, color='k')
        axis.bar(4, height=risk_tot / norm_fact, **kwargs)
        axis.text(4, risk_tot / norm_fact, str(int(round(risk_tot / norm_fact))),
                  horizontalalignment='center', verticalalignment='bottom',
                  fontsize=12, color='k')

        axis.set_xticks(np.arange(4) + 1)
        axis.set_xticklabels(['Risk ' + str(self.present_year),
                              'Economic \ndevelopment',
                              'Climate \nchange',
                              'Risk ' + str(self.future_year)])
        axis.set_ylabel('Impact (' + self.unit + ' ' + norm_name + ')')
        axis.set_title('Total accumulated impact from {:d} to {:d}'.format(
            self.present_year, self.future_year))
        return axis

    def _calc_impact_measures(self, hazard, exposures, meas_set, imp_fun_set,
                              when='future', risk_func=risk_aai_agg, save_imp=False):
        """Compute impact of each measure and transform it to input risk
        measurement. Set reference year from exposures value.

        Parameters
        ----------
        hazard : climada.Hazard
        exposures : climada.entity.Exposures
        meas_set : climada.MeasureSet
            set of measures.
        imp_fun_set : ImpactFuncSet
            set of impact functions.
        when : str, optional
            'present' or 'future'. The conditions that
            are being considered.
        risk_func : func, optional
            function describing risk measure given an Impact.
            Default: average annual impact (aggregated).
        save_imp : bool, optional
            activate if Impact of each measure is
            saved. Default: False.
        """
        impact_meas = dict()

        # compute impact without measures
        LOGGER.debug('%s impact with no measure.', when)
        imp_tmp = ImpactCalc(exposures, imp_fun_set, hazard).impact(assign_centroids=False)
        impact_meas[NO_MEASURE] = dict()
        impact_meas[NO_MEASURE]['cost'] = (0, 0)
        impact_meas[NO_MEASURE]['risk'] = risk_func(imp_tmp)
        impact_meas[NO_MEASURE]['risk_transf'] = 0.0
        impact_meas[NO_MEASURE]['efc'] = imp_tmp.calc_freq_curve()
        if save_imp:
            impact_meas[NO_MEASURE]['impact'] = imp_tmp

        # compute impact for each measure
        for measure in meas_set.get_measure(hazard.tag.haz_type):
            LOGGER.debug('%s impact of measure %s.', when, measure.name)
            imp_tmp, risk_transf = measure.calc_impact(exposures, imp_fun_set, hazard,
                                                       assign_centroids=False)
            impact_meas[measure.name] = dict()
            impact_meas[measure.name]['cost'] = (measure.cost, measure.risk_transf_cost_factor)
            impact_meas[measure.name]['risk'] = risk_func(imp_tmp)
            impact_meas[measure.name]['risk_transf'] = risk_func(risk_transf)
            impact_meas[measure.name]['efc'] = imp_tmp.calc_freq_curve()
            if save_imp:
                impact_meas[measure.name]['impact'] = imp_tmp

        # if present reference provided save it
        if when == 'future':
            self.imp_meas_future = impact_meas
        else:
            self.imp_meas_present = impact_meas

    def _calc_cost_benefit(self, disc_rates, imp_time_depen=None):
        """Compute discounted impact from present year to future year

        Parameters
        ----------
        disc_rates : DiscRates
            discount rates instance
        imp_time_depen : float, optional
            parameter which represent time evolution of impact
        """
        LOGGER.info('Computing cost benefit from years %s to %s.',
                    str(self.present_year), str(self.future_year))

        if self.future_year - self.present_year + 1 <= 0:
            raise ValueError('Wrong year range: %s - %s.'
                             % (str(self.present_year), str(self.future_year)))

        if not self.imp_meas_future:
            raise ValueError('Compute first _calc_impact_measures')

        time_dep = self._time_dependency_array(imp_time_depen)

        # discounted cost benefit for each measure and total climate risk
        for meas_name, meas_val in self.imp_meas_future.items():
            if meas_name == NO_MEASURE:
                # npv of the full unaverted damages
                if self.imp_meas_present:
                    self.tot_climate_risk = self._npv_unaverted_impact(
                        self.imp_meas_future[NO_MEASURE]['risk'],
                        disc_rates, time_dep, self.imp_meas_present[NO_MEASURE]['risk'])
                else:
                    self.tot_climate_risk = self._npv_unaverted_impact(
                        self.imp_meas_future[NO_MEASURE]['risk'],
                        disc_rates, time_dep)
                continue

            self._cost_ben_one(meas_name, meas_val, disc_rates, time_dep)

    def _cost_ben_one(self, meas_name, meas_val, disc_rates, time_dep,
                      ini_state=NO_MEASURE):
        """Compute cost and benefit for given measure with time dependency

        Parameters
        ----------
        meas_name : str
            name of measure
        meas_val : dict
            contains measure's cost, risk, efc, risk_trans and
            optionally impact at future
        disc_rates : DiscRates
            discount rates instance
        time_dep : np.array
            time dependency array
        ini_state : str, optional
            name of the measure to which to compute benefit.
            Default: 'no measure'
        """
        fut_benefit = self.imp_meas_future[ini_state]['risk'] - meas_val['risk']
        fut_risk_tr = meas_val['risk_transf']
        if self.imp_meas_present:
            pres_benefit = self.imp_meas_present[ini_state]['risk'] - \
                self.imp_meas_present[meas_name]['risk']
            meas_ben = pres_benefit + (fut_benefit - pres_benefit) * time_dep

            pres_risk_tr = self.imp_meas_present[meas_name]['risk_transf']
            risk_tr = pres_risk_tr + (fut_risk_tr - pres_risk_tr) * time_dep
        else:
            meas_ben = time_dep * fut_benefit
            risk_tr = time_dep * fut_risk_tr

        # discount
        meas_ben = disc_rates.net_present_value(self.present_year,
                                                self.future_year, meas_ben)
        risk_tr = disc_rates.net_present_value(self.present_year,
                                               self.future_year, risk_tr)
        self.benefit[meas_name] = meas_ben
        with np.errstate(divide='ignore'):
            self.cost_ben_ratio[meas_name] = (meas_val['cost'][0]
                                              + meas_val['cost'][1] * risk_tr) / meas_ben

    def _time_dependency_array(self, imp_time_depen=None):
        """Construct time dependency array. Each year contains a value in [0,1]
        representing the rate of damage difference achieved that year, according
        to the growth represented by parameter imp_time_depen.

        Parameters
        ----------
         imp_time_depen : float, optional
            parameter which represent time evolution of impact

        Returns
        -------
        np.array
        """
        n_years = self.future_year - self.present_year + 1
        if imp_time_depen:
            time_dep = np.arange(n_years)**imp_time_depen / \
                (n_years - 1)**imp_time_depen
        else:
            time_dep = np.ones(n_years)
        return time_dep

    def _npv_unaverted_impact(self, risk_future, disc_rates, time_dep,
                              risk_present=None):
        """Net present value of total unaverted damages

        Parameters
        ----------
        risk_future : float
            risk under future situation
        disc_rates : climada.DiscRates
            discount rates object
        time_dep : np.array
            values in 0-1 indicating impact growth at each year
        risk_present : float
            risk under current situation

        Returns
        -------
        float
        """
        if risk_present:
            tot_climate_risk = risk_present + (risk_future - risk_present) * time_dep
            tot_climate_risk = disc_rates.net_present_value(self.present_year,
                                                            self.future_year,
                                                            tot_climate_risk)
        else:
            tot_climate_risk = disc_rates.net_present_value(self.present_year,
                                                            self.future_year,
                                                            time_dep * risk_future)
        return tot_climate_risk

    def _combine_imp_meas(self, new_cb, in_meas_names, new_name, risk_func, when='future'):
        """Compute impacts combined measures assuming they are independent, i.e.
        their benefit can be added. Costs are also added. For the new measure
        the dictionary imp_meas_future if when='future' and imp_meas_present
        if when='present'.

        Parameters
        ----------
        in_meas_names : list(str)
            list with names of measures to combine
        new_name : str
            name to give to the new resulting measure
        risk_func : func, optional
            function describing risk measure given
            an Impact. Default: average annual impact (aggregated).
        when : str, optional
            'present' or 'future' making reference to which dictionary
            to fill (imp_meas_present or imp_meas_future respectively)
            default: 'future'
        """
        if when == 'future':
            imp_dict = self.imp_meas_future
            new_imp_dict = new_cb.imp_meas_future
        else:
            imp_dict = self.imp_meas_present
            new_imp_dict = new_cb.imp_meas_present

        sum_ben = np.sum([
            imp_dict[NO_MEASURE]['impact'].at_event - imp_dict[name]['impact'].at_event
            for name in in_meas_names
        ], axis=0)
        new_imp = copy.deepcopy(imp_dict[in_meas_names[0]]['impact'])
        new_imp.at_event = np.maximum(imp_dict[NO_MEASURE]['impact'].at_event
                                      - sum_ben, 0)

        new_imp.eai_exp = np.array([])
        new_imp.aai_agg = sum(new_imp.at_event * new_imp.frequency)

        new_imp_dict[new_name] = dict()
        new_imp_dict[new_name]['impact'] = new_imp
        new_imp_dict[new_name]['efc'] = new_imp.calc_freq_curve()
        new_imp_dict[new_name]['risk'] = risk_func(new_imp)
        new_imp_dict[new_name]['cost'] = (
            np.array([imp_dict[name]['cost'][0] for name in in_meas_names]).sum(),
            1)
        new_imp_dict[new_name]['risk_transf'] = 0

    def _print_results(self):
        """Print table with main results"""
        norm_fact, norm_name = _norm_values(np.array(list(self.benefit.values())).max())
        norm_name = '(' + self.unit + ' ' + norm_name + ')'

        table = []
        headers = ['Measure', 'Cost ' + norm_name, 'Benefit ' + norm_name, 'Benefit/Cost']
        for meas_name in self.benefit:
            if not np.isnan(self.cost_ben_ratio[meas_name]) and \
            not np.isinf(self.cost_ben_ratio[meas_name]):
                cost = self.cost_ben_ratio[meas_name] * self.benefit[meas_name] / norm_fact
            else:
                cost = self.imp_meas_future[meas_name]['cost'][0] / norm_fact
            table.append([meas_name, cost, self.benefit[meas_name] / norm_fact,
                          1 / self.cost_ben_ratio[meas_name]])
        print()
        print(tabulate(table, headers, tablefmt="simple"))

        table = []
        table.append(['Total climate risk:',
                      self.tot_climate_risk / norm_fact, norm_name])
        table.append(['Average annual risk:',
                      self.imp_meas_future[NO_MEASURE]['risk'] / norm_fact, norm_name])
        table.append(['Residual risk:',
                      (self.tot_climate_risk -
                       np.array(list(self.benefit.values())).sum()) / norm_fact, norm_name])
        print()
        print(tabulate(table, tablefmt="simple"))

    @staticmethod
    def _plot_list_cost_ben(cb_list, axis=None, **kwargs):
        """Overlay cost-benefit bars for every measure

        Parameters
        ----------
        cb_list : list
            list of CostBenefit instances with filled values
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
            arguments for Rectangle matplotlib, e.g. alpha=0.5
            (color is set by measures color attribute)

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5
        norm_fact = [_norm_values(cb_res.tot_climate_risk)[0] for cb_res in cb_list]
        norm_fact = np.array(norm_fact).mean()
        _, norm_name = _norm_values(norm_fact + 0.01)

        if not axis:
            _, axis = plt.subplots(1, 1)
        m_names = list(cb_list[0].cost_ben_ratio.keys())
        sort_cb = np.argsort(np.array([cb_list[0].cost_ben_ratio[name] for name in m_names]))
        xy_lim = [0, 0]
        for i_cb, cb_res in enumerate(cb_list):
            xmin = 0
            for meas_id in sort_cb:
                meas_n = m_names[meas_id]
                axis.add_patch(Rectangle((xmin, 0),
                                         cb_res.benefit[meas_n] / norm_fact,
                                         1 / cb_res.cost_ben_ratio[meas_n],
                                         color=cb_res.color_rgb[meas_n], **kwargs))

                if i_cb == 0:
                    axis.text(xmin + (cb_res.benefit[meas_n] / norm_fact) / 2,
                              0, '  ' + meas_n, horizontalalignment='center',
                              verticalalignment='bottom', rotation=90, fontsize=12)
                xmin += cb_res.benefit[meas_n] / norm_fact

            xy_lim[0] = max(xy_lim[0],
                            max(int(cb_res.tot_climate_risk / norm_fact),
                                np.array(list(cb_res.benefit.values())).sum() / norm_fact))
            try:
                with np.errstate(divide='ignore'):
                    xy_lim[1] = max(xy_lim[1], int(1 / cb_res.cost_ben_ratio[
                        m_names[sort_cb[0]]]) + 1)
            except (ValueError, OverflowError):
                xy_lim[1] = max(xy_lim[1],
                                int(1 / np.array(list(cb_res.cost_ben_ratio.values())).max()) + 1)

        axis.set_xlim(0, xy_lim[0])
        axis.set_ylim(0, xy_lim[1])
        axis.set_xlabel('NPV averted damage over ' +
                        str(cb_list[0].future_year - cb_list[0].present_year + 1) +
                        ' years (' + cb_list[0].unit + ' ' + norm_name + ')')
        axis.set_ylabel('Benefit/Cost ratio')
        return axis

    @staticmethod
    def _plot_averted_arrow(axis, bar_4, tot_benefit, risk_tot, norm_fact, **kwargs):
        """Plot arrow inn fourth bar of total averted damage by implementing
        all the measures.

        Parameters
        ----------
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        bar_4 : matplotlib.container.BarContainer
            bar where arrow is plotted
        tot_benefit : float
            arrow length
        risk_tot : float
            total risk
        norm_fact : float
            normalization factor
        kwargs : optional
            arguments for bar matplotlib function, e.g. alpha=0.5
        """
        bar_bottom, bar_top = bar_4.get_bbox().get_points()
        axis.text(bar_top[0] - (bar_top[0] - bar_bottom[0]) / 2, bar_top[1],
                  "Averted", ha="center", va="top", rotation=270, size=15)
        arrow_len = min(tot_benefit / norm_fact, risk_tot / norm_fact)

        if 'color' not in kwargs:
            kwargs['color'] = 'k'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.4
        if 'mutation_scale' not in kwargs:
            kwargs['mutation_scale'] = 100
        axis.add_patch(FancyArrowPatch(
            (bar_top[0] - (bar_top[0] - bar_bottom[0]) / 2, bar_top[1]),
            (bar_top[0] - (bar_top[0] - bar_bottom[0]) / 2, risk_tot / norm_fact - arrow_len),
            **kwargs))

    def _print_risk_transfer(self, layer, layer_no, cost_fix, cost_factor):
        """Print comparative of risk transfer with and without measure

        Parameters
        ----------
        layer : float
            expected insurance layer with measure
        layer_on : float
            expected insurance layer without measure
        """
        norm_fact, norm_name = _norm_values(np.array(list(self.benefit.values())).max())
        norm_name = '(' + self.unit + ' ' + norm_name + ')'
        headers = ['Risk transfer', 'Expected damage in \n insurance layer ' +
                   norm_name, 'Price ' + norm_name]
        table = [['without measure', layer_no / norm_fact,
                  (cost_fix + layer_no * cost_factor) / norm_fact],
                 ['with measure', layer / norm_fact,
                  (cost_fix + layer * cost_factor) / norm_fact]]
        print()
        print(tabulate(table, headers, tablefmt="simple"))
        print()

    @staticmethod
    def _print_npv():
        print('Net Present Values')

def _norm_values(value):
    """Compute normalization value and name

    Parameters
    ----------
    value : float
        value to normalize

    Returns :
    norm_fact: float
    norm_name: float
    """
    norm_fact = 1.
    norm_name = ''
    if value / 1.0e9 > 1:
        norm_fact = 1.0e9
        norm_name = 'bn'
    elif value / 1.0e6 > 1:
        norm_fact = 1.0e6
        norm_name = 'm'
    elif value / 1.0e3 > 1:
        norm_fact = 1.0e3
        norm_name = 'k'
    return norm_fact, norm_name
