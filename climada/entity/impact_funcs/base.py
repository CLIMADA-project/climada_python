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

Define ImpactFunc class.
"""

__all__ = ['ImpactFunc']

import logging
import numpy as np

import climada.util.checker as check
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

class ImpactFunc():
    """Contains the definition of one impact function.

    Attributes:
        haz_type (str): hazard type acronym (e.g. 'TC')
        id (int or str): id of the impact function. Exposures of the same type
            will refer to the same impact function id
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
        self.id = ''
        self.name = ''
        self.intensity_unit = ''
        self.haz_type = ''
        # Followng values defined for each intensity value
        self.intensity = np.array([])
        self.mdd = np.array([])
        self.paa = np.array([])

    def calc_mdr(self, inten):
        """ Interpolate impact function to a given intensity.

        Parameters:
            inten (float or np.array): intensity, the x-coordinate of the
                interpolated values.

        Returns:
            np.array
        """
#        return np.interp(inten, self.intensity, self.mdd * self.paa)
        return np.interp(inten, self.intensity, self.paa) * \
            np.interp(inten, self.intensity, self.mdd)

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
        graph.add_curve(self.intensity, self.mdd * 100, 'b', label='MDD')
        graph.add_curve(self.intensity, self.paa * 100, 'r', label='PAA')
        graph.add_curve(self.intensity, self.mdd * self.paa * 100, 'k--', \
                        label='MDR')
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
