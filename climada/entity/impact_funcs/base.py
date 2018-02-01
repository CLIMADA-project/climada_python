"""
Define Vulnerability class and ImpactFuncs.
"""

__all__ = ['Vulnerability', 'ImpactFuncs']

import numpy as np

from climada.entity.loader import Loader
import climada.util.checker as check
from climada.entity.tag import Tag
import climada.util.plot as plot

class ImpactFuncs(Loader):
    """Contains impact functions of type Vulnerability.

    Attributes
    ----------
        tag (Taf): information about the source data
        data (dict): dictionary of vulnerabilities. Keys are the
            vulnerabilities' id and values are instances of Vulnerability.
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
            >>> fun_1 = Vulnerability()
            >>> fun_1.id = 3
            >>> fun_1.intensity = np.array([0, 20])
            >>> fun_1.paa = np.array([0, 1])
            >>> fun_1.mdd = np.array([0, 0.5])
            >>> imp_fun = ImpactFuncs()
            >>> imp_fun.data['TC'] = {fun_1.id : fun_1}
            >>> imp_fun.check()
            Fill impact functions with values and check consistency data.
        """
        self.tag = Tag(file_name, description)
        self.data = {} # {hazard_id : {id:Vulnerability}}

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def check(self):
        """ Override Loader check."""
        for key_haz, fun in self.data.items():
            for key, val in fun.items():
                if key != val.id:
                    raise ValueError('Wrong Vulnerability.id: %s != %s' %\
                                     (key, val.id))
                if key_haz != val.haz_type:
                    raise ValueError('Wrong Vulnerability.haz_type: %s != %s'\
                                     % (key_haz, val.haz_type))
                val.check()

    def plot(self, haz_type=None, id_fun=None):
        """Plot impact functions of selected hazard (all if not provided) and
        selected function id (all if not provided).

        Parameters
        ----------
            haz_type (str, optional): hazard type
            id_fun (int, optional): id of the function
            show (bool, optional): bool to execute plt.show(). Default: True

        Returns
        -------
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        # Select all hazard types to plot
        if haz_type is not None:
            hazards = [haz_type]
        else:
            hazards = self.data.keys()
        # Count number of plots
        num_plts = 0
        for sel_haz in hazards:
            if id_fun is not None:
                num_plts += 1
            else:
                num_plts += len(self.data[sel_haz].keys())
        # Plot
        do_show = plot.SHOW
        plot.SHOW = False
        graph = plot.Graph2D('', num_plts)
        for sel_haz in hazards:
            if id_fun is not None:
                self.data[sel_haz][id_fun].plot(graph)
            else:
                for sel_id in self.data[sel_haz].keys():
                    self.data[sel_haz][sel_id].plot(graph)
        plot.SHOW = do_show
        plot.show()
        return graph.get_elems()

class Vulnerability(object):
    """Contains the definition of one Vulnerability (or impact function).

    Attributes
    ----------
        id (int): id of the function
        name (str): name of the function
        haz_type (str): hazard type
        intensity_unit (str): unit of the intensity
        intensity (np.array): intensity values
        mdd (np.array): mean damage (impact) degree for each intensity
        paa (np.array): percentage of affected assets (exposures) for each
            intensity
    """

    def __init__(self):
        """ Empty initialization."""
        self.id = 0
        self.name = ''
        self.intensity_unit = 'NA'
        self.haz_type = 'NA'
        # Followng values defined for each intensity value
        self.intensity = np.array([])
        self.mdd = np.array([])
        self.paa = np.array([])

    def interpolate(self, inten, attribute):
        """ Interpolate impact function to a given intensity.

        Parameters
        ----------
            inten (float or np.array): intensity, the x-coordinate of the
                interpolated values.
            attribute (str): defines the impact function attribute to
                interpolate. Possbile values: 'mdd' or 'paa'.

        Raises
        ------
            ValueError
        """
        if attribute == 'mdd':
            return np.interp(inten, self.intensity, self.mdd)
        elif attribute == 'paa':
            return np.interp(inten, self.intensity, self.paa)
        else:
            raise ValueError('Attribute of the impact function %s not found.'\
                             % (attribute))

    def plot(self, graph=None):
        """Plot the impact functions MDD, MDR and PAA in one graph.

        Parameters
        ----------
            graph (Graph2D, optional): graph where to add the plots
            show (bool, optional): bool to execute plt.show(). Default: True
        Returns
        -------
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        if graph is None:
            graph = plot.Graph2D('', 1)
        graph.add_subplot('Intensity (%s)' % self.intensity_unit, \
                         'Percentage (%)', \
                         '%s %s %s' % (self.haz_type, str(self.id), self.name))
        graph.add_curve(self.intensity, self.mdd * 100, 'b', 'MDD')
        graph.add_curve(self.intensity, self.paa * 100, 'r', 'PAA')
        graph.add_curve(self.intensity, self.mdd * self.paa * 100, 'k--', \
                        'MDR')
        graph.set_x_lim(self.intensity)
        plot.show()
        return graph.get_elems()

    def check(self):
        """ Check consistent instance data.

        Raises
        ------
            ValueError
        """
        num_exp = len(self.intensity)
        check.size(num_exp, self.mdd, 'Vulnerability.mdd')
        check.size(num_exp, self.paa, 'Vulnerability.paa')
