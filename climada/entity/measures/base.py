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
        color_rgb (np.array): integer array of size 3. Gives color code of
            this measure in RGB
        cost (float): cost
        hazard_freq_cutoff (float): hazard frequency cutoff
        hazard_intensity (tuple): parameter a and b
        mdd_impact (tuple): parameter a and b of the impact over the mean
            damage (impact) degree
        paa_impact (tuple): parameter a and b of the impact over the
            percentage of affected assets (exposures)
        risk_transf_attach (float): risk transfer attach
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
        self.hazard_inten_imp = () # parameter a and b
        self.mdd_impact = () # parameter a and b
        self.paa_impact = () # parameter a and b
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

    def implement(self, exposures, imp_fun_set, hazard):
        """Implement measure with all its defined parameters."""
        new_haz = self.change_hazard(hazard)
        new_exp = self.change_exposures(exposures)
        new_ifs = self.change_imp_func(imp_fun_set)

        return new_exp, new_ifs, new_haz

    def change_hazard(self, hazard):
        """Apply measure to hazard of the same type.

        Parameters:
            hazard (Hazard): hazard instance

        Returns:
            Hazard
        """
        # TODO: implement
        return hazard

    def change_exposures(self, exposures):
        """Apply measure to exposures.

        Parameters:
            exposures (Exposures): exposures instance

        Returns:
            Exposures
        """
        # TODO: implement
        return exposures

    def change_imp_func(self, imp_set):
        """Apply measure to impact functions of the same hazard type.

        Parameters:
            imp_set (ImpactFuncSet): impact functions to be modified

        Returns:
            ImpactFuncSet
        """
        # all impact functions of one hazard??
        new_imp_set = ImpactFuncSet()

        for imp_fun in imp_set.get_func(self.haz_type):
            new_if = copy.copy(imp_fun)
            new_if.intensity = np.maximum(new_if.intensity * \
                self.hazard_inten_imp[0] - self.hazard_inten_imp[1], 0.0)
            new_if.mdd = np.maximum(new_if.mdd * self.mdd_impact[0] - \
                self.mdd_impact[1], 0.0)
            new_if.paa = np.maximum(new_if.paa * self.paa_impact[0] - \
                self.paa_impact[1], 0.0)
            new_imp_set.add_func(new_if)

        if not new_imp_set.size():
            LOGGER.info('No impact function of hazard %s found.', self.haz_type)

        return new_imp_set
