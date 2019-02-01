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

from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
import climada.util.checker as check

LOGGER = logging.getLogger(__name__)

class Measure():
    """Contains the definition of one measure.

    Attributes:
        name (str): name of the action
        haz_type (str): related hazard type (peril), e.g. TC
        color_rgb (np.array): integer array of size 3. Gives color code of
            this measure in RGB
        cost (float): discounted cost (in same units as assets)
        hazard_set (str): file name of hazard to use
        hazard_freq_cutoff (float): hazard frequency cutoff
        exposure_set (str): file name of exposure to use
        exp_region_id (int): region id of the selected exposures to consider
        hazard_inten_imp (tuple): parameter a and b of hazard intensity change
        mdd_impact (tuple): parameter a and b of the impact over the mean
            damage degree
        paa_impact (tuple): parameter a and b of the impact over the
            percentage of affected assets
        imp_fun_map (str): change of impact function id, e.g. '1to3'
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
        self.hazard_set = ''
        self.hazard_freq_cutoff = 0

        # related to change in exposures
        self.exposures_set = ''
        self.exp_region_id = -1

        # related to change in impact functions
        self.hazard_inten_imp = (1, 0) # parameter a and b
        self.mdd_impact = (1, 0) # parameter a and b
        self.paa_impact = (1, 0) # parameter a and b
        self.imp_fun_map = ''

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

    def apply(self, exposures, imp_fun_set, hazard):
        """Implement measure with all its defined parameters.

        Parameters:
            exposures (Exposures): exposures instance
            imp_fun_set (ImpactFuncSet): impact functions instance
            hazard (Hazard): hazard instance

        Returns:
            new_haz, new_ifs, new_haz

        """
        new_haz = self._change_hazard(exposures, imp_fun_set, hazard)
        new_exp = self._change_exposures(exposures)
        new_ifs = self._change_imp_func(imp_fun_set)

        return new_exp, new_ifs, new_haz

    def _change_hazard(self, exposures, imp_fun_set, hazard):
        """Apply measure to hazard of the same type.

        Parameters:
            exposures (Exposures): exposures instance
            imp_fun_set (ImpactFuncSet): impact functions instance
            hazard (Hazard): hazard instance

        Returns:
            Hazard
        """
        if self.hazard_freq_cutoff == 0:
            return hazard

        # TODO implement hazard_set
        from climada.engine import Impact
        imp = Impact()
        imp.calc(exposures, imp_fun_set, hazard)

        new_haz = copy.deepcopy(hazard)
        sort_idxs = np.argsort(imp.at_event)[::-1]
        exceed_freq = np.cumsum(imp.frequency[sort_idxs])
        cutoff = exceed_freq > self.hazard_freq_cutoff
        sel_haz = sort_idxs[cutoff]
        new_haz.intensity[sel_haz, :] = np.zeros(new_haz.intensity.shape[1])
        return new_haz

    def _change_exposures(self, exposures):
        """Apply measure to exposures.

        Parameters:
            exposures (Exposures): exposures instance

        Returns:
            Exposures
        """
        # TODO: implement exposures_set, exp_region_id
        return exposures

    def _change_imp_func(self, imp_set):
        """Apply measure to impact functions of the same hazard type.

        Parameters:
            imp_set (ImpactFuncSet): impact functions to be modified

        Returns:
            ImpactFuncSet
        """
        # TODO implement imp_fun_map
        # all impact functions of one hazard
        new_imp_set = ImpactFuncSet()

        for imp_fun in imp_set.get_func(self.haz_type):
            new_if = copy.copy(imp_fun)
            new_if.intensity = np.maximum(new_if.intensity * \
                self.hazard_inten_imp[0] - self.hazard_inten_imp[1], 0.0)
            new_if.mdd = np.maximum(new_if.mdd * self.mdd_impact[0] + \
                self.mdd_impact[1], 0.0)
            new_if.paa = np.maximum(new_if.paa * self.paa_impact[0] + \
                self.paa_impact[1], 0.0)
            new_imp_set.add_func(new_if)

        if not new_imp_set.size():
            LOGGER.info('No impact function of hazard %s found.', self.haz_type)

        return new_imp_set
