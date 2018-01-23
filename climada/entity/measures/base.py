"""
Define Measure class and Measures ABC.
"""

import numpy as np

from climada.entity.loader import Loader
import climada.util.checker as check
from climada.entity.tag import Tag

class Measures(Loader):
    """Contains measures of type Measures.

    Attributes
    ----------
        tag (Taf): information about the source data
        data (dict): dictionary of measures. Keys are the measures' id and
            values are instances of Measures.
    """

    def __init__(self, file_name=None, description=None):
        """Fill values from file, if provided.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError

        Examples
        --------
            >>> meas_1 = Measure()
            >>> meas_1.color_rgb = np.array([0.1529, 0.2510, 0.5451])
            >>> meas_1.hazard_intensity = (1, 0)
            >>> meas_1.mdd_impact = (1, 0)
            >>> meas_1.paa_impact = (1, 0)
            >>> meas = Measures()
            >>> meas.data.append(meas_1)
            >>> meas.tag.description = "my dummy measures."
            >>> meas.check()
            Fill measures with values and check consistency data.
        """
        self.tag = Tag(file_name, description)
        self.data = [] # [Measure()]

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def check(self):
        """ Override Loader check."""
        for meas in self.data:
            meas.check()

class Measure(object):
    """Contains the definition of one Measure.

    Attributes
    ----------
        name (str): name of the function
        color_rgb (np.array): integer array of size 3. Gives color code of
            this measure in RGB
        cost (float): cost
        hazard_freq_cutoff (float): hazard frequency cutoff
        hazard_event_set (str): hazard event set
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
        self.name = ""
        self.color_rgb = np.array([0, 0, 0])
        self.cost = 0
        self.hazard_freq_cutoff = 0
        self.hazard_event_set = 'NA'
        self.hazard_intensity = () # parameter a and b
        self.mdd_impact = () # parameter a and b
        self.paa_impact = () # parameter a and b
        self.risk_transf_attach = 0
        self.risk_transf_cover = 0

    def check(self):
        """ Check consistent instance data.

        Raises
        ------
            ValueError
        """
        check.size(3, self.color_rgb, 'Measure.color_rgb')
        check.size(2, self.hazard_intensity, 'Measure.hazard_intensity')
        check.size(2, self.mdd_impact, 'Measure.mdd_impact')
        check.size(2, self.paa_impact, 'Measure.paa_impact')
