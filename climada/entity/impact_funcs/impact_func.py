"""
Define ImpactFunc class.
"""

__all__ = ['ImpactFunc']

import logging
import numpy as np

import climada.util.checker as check
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

class ImpactFunc(object):
    """Contains the definition of one impact function.

    Attributes:
        haz_type (str): hazard type acronym (e.g. 'TC')
        id (int): id of the ImpactFunc (wrt vulnerabilities of same hazard)
        name (str): name of the ImpactFunc
        intensity_unit (str): unit of the intensity
        intensity (np.array): intensity values
        mdd (np.array): mean damage (impact) degree for each intensity (numbers
            in [0,1])
        paa (np.array): percentage of affected assets (exposures) for each
            intensity (numbers in [0,1])
    """
    def __init__(self):
        """ Empty initialization."""
        self.id = 'NA' # int expected
        self.name = ''
        self.intensity_unit = 'NA'
        self.haz_type = 'NA'
        # Followng values defined for each intensity value
        self.intensity = np.array([])
        self.mdd = np.array([])
        self.paa = np.array([])

    def interpolate(self, inten, attribute):
        """ Interpolate impact function to a given intensity.

        Parameters:
            inten (float or np.array): intensity, the x-coordinate of the
                interpolated values.
            attribute (str): defines the impact function attribute to
                interpolate. Possbile values: 'mdd' or 'paa'.

        Returns:
            np.array

        Raises:
            ValueError
        """
        if attribute == 'mdd':
            return np.interp(inten, self.intensity, self.mdd)
        elif attribute == 'paa':
            return np.interp(inten, self.intensity, self.paa)
        else:
            LOGGER.error("Attribute of the impact function not found: %s",\
                         attribute)
            raise ValueError

    def plot(self, graph=None):
        """Plot the impact functions MDD, MDR and PAA in one graph, where
        MDR = PAA * MDD.

        Parameters:
            graph (Graph2D, optional): graph where to add the plots
            show (bool, optional): bool to execute plt.show(). Default: True

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        if graph is None:
            graph = plot.Graph2D('', 1)
        title = '%s %s' % (self.haz_type, str(self.id))
        if self.name != str(self.id):
            title += ': %s' % self.name
        graph.add_subplot('Intensity (%s)' % self.intensity_unit, \
                         'Percentage (%)', title)
        graph.add_curve(self.intensity, self.mdd * 100, 'b', 'MDD')
        graph.add_curve(self.intensity, self.paa * 100, 'r', 'PAA')
        graph.add_curve(self.intensity, self.mdd * self.paa * 100, 'k--', \
                        'MDR')
        graph.set_x_lim(self.intensity)
        return graph.get_elems()

    def check(self):
        """ Check consistent instance data.

        Raises:
            ValueError
        """
        num_exp = len(self.intensity)
        check.size(num_exp, self.mdd, 'ImpactFunc.mdd')
        check.size(num_exp, self.paa, 'ImpactFunc.paa')
