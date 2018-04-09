"""
Define Measure class.
"""

__all__ = ['Measure']

import logging
import numpy as np

import climada.util.checker as check

LOGGER = logging.getLogger(__name__)

class Measure(object):
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
        self.name = 'NA'
        self.color_rgb = np.array([0, 0, 0])
        self.cost = 0
        self.hazard_freq_cutoff = 0
#        self.hazard_event_set = 'NA'
        self.hazard_intensity = () # parameter a and b
        self.mdd_impact = () # parameter a and b
        self.paa_impact = () # parameter a and b
        self.risk_transf_attach = 0
        self.risk_transf_cover = 0

    def check(self):
        """ Check consistent instance data.

        Raises:
            ValueError
        """
        check.size(3, self.color_rgb, 'Measure.color_rgb')
        check.size(2, self.hazard_intensity, 'Measure.hazard_intensity')
        check.size(2, self.mdd_impact, 'Measure.mdd_impact')
        check.size(2, self.paa_impact, 'Measure.paa_impact')
