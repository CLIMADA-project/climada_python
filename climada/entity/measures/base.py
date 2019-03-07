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

Define Measure class.
"""

__all__ = ['Measure']

import copy
import logging
import numpy as np
import pandas as pd

from climada.entity.exposures.base import Exposures, INDICATOR_IF, INDICATOR_CENTR
import climada.util.checker as check

LOGGER = logging.getLogger(__name__)

IF_ID_FACT = 1000
""" Factor internally used as id for impact functions when region selected."""

NULL_STR = 'nil'
""" String considered as no path in measures exposures_set and hazard_set or
no string in imp_fun_map"""

class Measure():
    """Contains the definition of one measure.

    Attributes:
        name (str): name of the action
        haz_type (str): related hazard type (peril), e.g. TC
        color_rgb (np.array): integer array of size 3. Gives color code of
            this measure in RGB
        cost (float): discounted cost (in same units as assets)
        hazard_set (str): file name of hazard to use (in h5 format)
        hazard_freq_cutoff (float): hazard frequency cutoff
        exposure_set (str): file name of exposure to use (in h5 format)
        imp_fun_map (str): change of impact function id of exposures, e.g. '1to3'
        hazard_inten_imp (tuple): parameter a and b of hazard intensity change
        mdd_impact (tuple): parameter a and b of the impact over the mean
            damage degree
        paa_impact (tuple): parameter a and b of the impact over the
            percentage of affected assets
        exp_region_id (int): region id of the selected exposures to consider ALL
            the previous parameters
        risk_transf_attach (float): risk transfer attachment
        risk_transf_cover (float): risk transfer cover
    """

    def __init__(self):
        """ Empty initialization."""
        self.name = ''
        self.haz_type = ''
        self.color_rgb = np.array([0, 0, 0])
        self.cost = 0

        # related to change in hazard
        self.hazard_set = NULL_STR
        self.hazard_freq_cutoff = 0

        # related to change in exposures
        self.exposures_set = NULL_STR
        self.imp_fun_map = NULL_STR # ids of impact functions to change e.g. 1to10

        # related to change in impact functions
        self.hazard_inten_imp = (1, 0) # parameter a and b
        self.mdd_impact = (1, 0) # parameter a and b
        self.paa_impact = (1, 0) # parameter a and b

        # related to change in region
        self.exp_region_id = 0

        # risk transfer
        self.risk_transf_attach = 0
        self.risk_transf_cover = 0

    def check(self):
        """ Check consistent instance data.

        Raises:
            ValueError
        """
        check.size(3, self.color_rgb, 'Measure.color_rgb')
        check.size(2, self.hazard_inten_imp, 'Measure.hazard_inten_imp')
        check.size(2, self.mdd_impact, 'Measure.mdd_impact')
        check.size(2, self.paa_impact, 'Measure.paa_impact')

    def calc_impact(self, exposures, imp_fun_set, hazard):
        """Apply measure and compute impact and risk transfer of measure
        implemented over inputs.

        Parameters:
            exposures (Exposures): exposures instance
            imp_fun_set (ImpactFuncSet): impact functions instance
            hazard (Hazard): hazard instance

        Returns:
            Impact, float
        """
        new_exp, new_ifs, new_haz = self.apply(exposures, imp_fun_set, hazard)
        return self._calc_impact(new_exp, new_ifs, new_haz)

    def apply(self, exposures, imp_fun_set, hazard):
        """Implement measure with all its defined parameters.

        Parameters:
            exposures (Exposures): exposures instance
            imp_fun_set (ImpactFuncSet): impact functions instance
            hazard (Hazard): hazard instance

        Returns:
            Exposures, ImpactFuncSet, Hazard
        """
        # chenge hazard
        new_haz = self._change_all_hazard(hazard)
        new_exp = self._change_all_exposures(exposures)
        # change exposures
        new_exp = self._change_exposures_if(new_exp)
        # change impact functions
        new_ifs = self._change_imp_func(imp_fun_set)
        # cutoff events whose damage happen with high frequency
        new_haz = self._cutoff_hazard_damage(new_exp, new_ifs, new_haz)

        # apply all previous changes only to the selected exposures
        new_exp, new_ifs, new_haz = self._filter_exposures(exposures, \
            imp_fun_set, hazard, new_exp, new_ifs, new_haz)

        return new_exp, new_ifs, new_haz

    def _calc_impact(self, new_exp, new_ifs, new_haz):
        """Compute impact and risk transfer of measure implemented over inputs.

        Parameters:
            new_exp (Exposures): exposures once measure applied
            new_ifs (ImpactFuncSet): impact functions once measure applied
            new_haz (Hazard): hazard once measure applied

        Returns:
            Impact, float
        """
        from climada.engine.impact import Impact
        imp = Impact()
        imp.calc(new_exp, new_ifs, new_haz)

        risk_transfer = 0
        if self.risk_transf_attach + self.risk_transf_cover > 0:
            imp_layer = np.minimum(np.maximum(imp.at_event - self.risk_transf_attach, 0),
                                   self.risk_transf_cover)
            risk_transfer = np.sum(imp_layer * imp.frequency)
            imp.at_event = np.maximum(imp.at_event -imp_layer, 0)
            imp.aai_agg = np.sum(imp.at_event * imp.frequency)
            # expected annual impact per exposure no longer valid
            imp.eai_exp = np.array([])

        return imp, risk_transfer

    def _change_all_hazard(self, hazard):
        """Change hazard to provided hazard_set.

        Parameters:
            hazard (Hazard): hazard instance

        Returns:
            Hazard
        """
        if self.hazard_set == NULL_STR:
            return hazard

        LOGGER.debug('Setting new hazard %s', self.hazard_set)
        from climada.hazard.base import Hazard
        new_haz = Hazard(hazard.tag.haz_type)
        new_haz.read_hdf5(self.hazard_set)
        new_haz.check()
        return new_haz

    def _change_all_exposures(self, exposures):
        """Change exposures to provided exposures_set.

        Parameters:
            exposures (Exposures): exposures instance

        Returns:
            Exposures
        """
        if self.exposures_set == NULL_STR:
            return exposures

        LOGGER.debug('Setting new exposures %s', self.exposures_set)
        new_exp = Exposures()
        new_exp.read_hdf5(self.exposures_set)
        new_exp.check()

        if exposures.shape[0] != new_exp.shape[0] or \
        not np.array_equal(exposures.latitude.values, new_exp.latitude.values) or \
        not np.array_equal(exposures.longitude.values, new_exp.longitude.values):
            LOGGER.warning('Exposures locations have changed.')

        return new_exp

    def _change_exposures_if(self, exposures):
        """ Change exposures impact functions ids according to imp_fun_map.

        Parameters:
            exposures (Exposures): exposures instance
        """
        if self.imp_fun_map == NULL_STR:
            return exposures

        LOGGER.debug('Setting new exposures impact functions%s', self.imp_fun_map)
        new_exp = copy.deepcopy(exposures)
        from_id = int(self.imp_fun_map[0:self.imp_fun_map.find('to')])
        to_id = int(self.imp_fun_map[self.imp_fun_map.find('to')+2:])
        exp_change = np.argwhere(new_exp[INDICATOR_IF+self.haz_type].values == from_id).\
            reshape(-1)
        new_exp[INDICATOR_IF+self.haz_type].values[exp_change] = to_id
        return new_exp

    def _change_imp_func(self, imp_set):
        """Apply measure to impact functions of the same hazard type.

        Parameters:
            imp_set (ImpactFuncSet): impact functions to be modified

        Returns:
            ImpactFuncSet
        """
        new_imp_set = copy.deepcopy(imp_set)
        for imp_fun in new_imp_set.get_func(self.haz_type):
            LOGGER.debug('Transforming impact functions.')
            imp_fun.intensity = np.maximum(imp_fun.intensity * \
                self.hazard_inten_imp[0] - self.hazard_inten_imp[1], 0.0)
            imp_fun.mdd = np.maximum(imp_fun.mdd * self.mdd_impact[0] + \
                self.mdd_impact[1], 0.0)
            imp_fun.paa = np.maximum(imp_fun.paa * self.paa_impact[0] + \
                self.paa_impact[1], 0.0)

        if not new_imp_set.size():
            LOGGER.info('No impact function of hazard %s found.', self.haz_type)

        return new_imp_set

    def _cutoff_hazard_damage(self, exposures, if_set, hazard):
        """Cutoff of hazard events which generate damage with a frequency higher
        than hazard_freq_cutoff.

        Parameters:
            exposures (Exposures): exposures instance
            imp_set (ImpactFuncSet): impact functions instance
            hazard (Hazard): hazard instance

        Returns:
            ImpactFuncSet
        """
        if self.hazard_freq_cutoff == 0:
            return hazard

        LOGGER.info('Cutting events whose damage have a frequency > %s.',
                    self.hazard_freq_cutoff)
        from climada.engine.impact import Impact
        imp = Impact()
        imp.calc(exposures, if_set, hazard)

        new_haz = copy.deepcopy(hazard)
        sort_idxs = np.argsort(imp.at_event)[::-1]
        exceed_freq = np.cumsum(imp.frequency[sort_idxs])
        cutoff = exceed_freq > self.hazard_freq_cutoff
        sel_haz = sort_idxs[cutoff]
        new_haz_inten = new_haz.intensity.tolil()
        new_haz_inten[sel_haz, :] = np.zeros(new_haz.intensity.shape[1])
        new_haz.intensity = new_haz_inten.tocsr()
        return new_haz

    def _filter_exposures(self, exposures, imp_set, hazard, new_exp, new_ifs,
                          new_haz):
        """ Incorporate changes of new elements to previous ones only for the
        selected exp_region_id. If exp_region_id is -1, all new changes
        will be accepted.

        Parameters:
            exposures (Exposures): old exposures instance
            imp_set (ImpactFuncSet): old impact functions instance
            hazard (Hazard): old hazard instance
            new_exp (Exposures): new exposures instance
            new_ifs (ImpactFuncSet): new impact functions instance
            new_haz (Hazard): new hazard instance

        Returns:
            Exposures, ImpactFuncSet, Hazard
        """
        if self.exp_region_id == 0:
            return new_exp, new_ifs, new_haz

        if exposures is new_exp:
            new_exp = copy.deepcopy(exposures)

        if hazard is new_haz:
            new_haz = copy.deepcopy(hazard)

        chg_reg = np.argwhere(self.exp_region_id == exposures.region_id.values).reshape(-1)
        no_chg_reg = np.argwhere(self.exp_region_id != exposures.region_id.values).reshape(-1)

        # provide new impact functions ids to changed impact functions
        fun_ids = list(new_ifs.get_func()[self.haz_type].keys())
        for key in fun_ids:
            new_ifs.get_func()[self.haz_type][key].id = key + IF_ID_FACT
            new_ifs.get_func()[self.haz_type][key + IF_ID_FACT] = \
                new_ifs.get_func()[self.haz_type][key]
        new_exp[INDICATOR_IF+self.haz_type] += IF_ID_FACT
        # collect old impact functions ase well (used by exposures)
        new_ifs.get_func()[self.haz_type].update(imp_set.get_func()[self.haz_type])

        # concatenate previous and new exposures
        new_exp = pd.concat([exposures.iloc[no_chg_reg], new_exp.iloc[chg_reg]])

        # put hazard intensities outside region to previous intensities
        exposures.assign_centroids(hazard)
        centr = exposures[INDICATOR_CENTR+self.haz_type].values[chg_reg]
        centr = np.delete(np.arange(hazard.intensity.shape[1]), np.unique(centr))
        new_haz_inten = new_haz.intensity.tolil()
        new_haz_inten[:, centr] = hazard.intensity[:, centr]
        new_haz.intensity = new_haz_inten.tocsr()

        return new_exp, new_ifs, new_haz
