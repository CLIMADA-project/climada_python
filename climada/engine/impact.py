"""
Define Impact class.
"""

__all__ = ['ImpactFreqCurve', 'Impact']

import os
import warnings
import numpy as np

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard
import climada.util.plot as plot

class ImpactFreqCurve(object):
    """ Impact exceedence frequency curve.

    Attributes
    ----------
        return_per (np.array): return period
        impact (np.array): impact exceeding frequency
        unit (str): value unit used (given by exposures unit)
        label (str): string describing source data
    """
    def __init__(self):
        self.return_per = np.array([])
        self.impact = np.array([])
        self.unit = 'NA'
        self.label = ''

    def plot(self):
        """Plot impact frequency curve.

        Returns
        -------
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        graph = plot.Graph2D(self.label)
        graph.add_subplot('Return period (year)', 'Impact (%s)' % self.unit)
        graph.add_curve(self.return_per, self.impact, 'y')
        plot.show()
        return graph.get_elems()

    def plot_compare(self, ifc):
        """Plot current and input impact frequency curves in a figure.

        Returns
        -------
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        if self.unit != ifc.unit:
            warnings.warn("Comparing between two different units: %s and %s" %\
                         (self.unit, ifc.unit))
        graph = plot.Graph2D('', 2)
        graph.add_subplot('Return period (year)', 'Impact (%s)' % self.unit)
        graph.add_curve(self.return_per, self.impact, 'b', self.label)
        graph.add_curve(ifc.return_per, ifc.impact, 'r', ifc.label)
        plot.show()
        return graph.get_elems()

class Impact(object):
    """Impact for a given entity (exposures and impact functions) and hazard.

    Attributes
    ----------
        exposures_tag (Tag): information about the exposures
        impact_funcs_tag (Tag): information about the impact functions
        hazard_tag (TagHazard): information about the hazard
        at_exp (np.array): impact for each exposure
        at_event (np.array): impact for each hazard event
        frequency (np.arrray): frequency (pro event)
        tot_vale (float): total exposure value affected
        tot (float): total expected impact
        unit (str): value unit used (given by exposures unit)
    """

    def __init__(self):
        """ Empty initialization."""
        self.exposures_tag = Tag()
        self.impact_funcs_tag = Tag()
        self.hazard_tag = TagHazard()
        self.at_exp = np.array([])
        self.at_event = np.array([])
        self.frequency = np.array([])
        self.tot_value = 0
        self.tot = 0
        self.unit = 'NA'

    def calc_freq_curve(self):
        """Compute and plot impact frequency curve."""
        ifc = ImpactFreqCurve()
        # Sort descendingly the impacts per events
        sort_idxs = np.argsort(self.at_event)[::-1]
        # Calculate exceedence frequency
        exceed_freq = np.cumsum(self.frequency[sort_idxs])
        # Set return period and imact exceeding frequency
        ifc.return_per = 1/exceed_freq
        ifc.impact = self.at_event[sort_idxs]
        ifc.unit = self.unit
        ifc.label = os.path.basename(self.exposures_tag.file_name) + ' x ' +\
            os.path.basename(self.hazard_tag.file_name)
        return ifc

    def calc(self, exposures, impact_funcs, hazard):
        """Compute impact of an hazard to exposures.

        Parameters
        ----------
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncs): impact functions
            hazard (Hazard): hazard

        Examples
        --------
            >>> hazard = HazardMat('filename') # Set hazard
            >>> entity = Entity() # Load entity with default values
            >>> entity.exposures = ExposuresExcel('filename') # Set exposures
            >>> tc_impact = Impact()
            >>> tc_impact.calc(entity.exposures, entity.impact_functs, hazard)
            Use Entity class

            >>> hazard = HazardMat('filename') # Set hazard
            >>> funcs = ImpactFuncsExcel('filename') # Set impact functions
            >>> exposures = ExposuresExcel('filename') # Set exposures
            >>> tc_impact = Impact()
            >>> tc_impact.calc(exposures, funcs, hazard)
            Specify only exposures and impact functions
        """
        # 1. Assign centroids to each exposure if not done
        if exposures.assigned.size == 0:
            exposures.assign(hazard)

        # 2. Initialize values
        self.unit = exposures.value_unit
        self.frequency = hazard.frequency
        self.at_event = np.zeros(hazard.intensity.shape[0])
        self.at_exp = np.zeros(len(exposures.value))
        self.tot_value = 0
        self.exposures_tag = exposures.tag
        self.impact_funcs_tag = impact_funcs.tag
        self.hazard_tag = hazard.tag
        # Select exposures with positive value and assigned centroid
        exp_idx = np.where(np.logical_and(exposures.value > 0, \
                                          exposures.assigned >= 0))[0]
        # Warning if no exposures selected
        if exp_idx.size == 0:
            warnings.warn('No affected exposures.')

        # Get hazard type
        haz_type = hazard.tag.haz_type
        # Get damage functions for this hazard
        haz_imp = impact_funcs.get_vulner(haz_type)

        # 3. Loop over exposures according to their impact function
        # Loop over impact functions
        for imp_fun in haz_imp:
            # get indices of all the exposures with this impact function
            exp_iimp = np.where(exposures.impact_id[exp_idx] == imp_fun.id)[0]

            # loop over selected exposures
            for iexp in exp_iimp:
                # compute impact on exposure
                event_row, impact = self._one_exposure(iexp, exposures, \
                                                        hazard, imp_fun)

                # add values to impact impact
                self.at_event[event_row] += impact
                self.at_exp[iexp] += np.squeeze(sum(impact * hazard. \
                           frequency[event_row]))
                self.tot_value += exposures.value[iexp]

        self.tot = sum(self.at_event * hazard.frequency)

    @staticmethod
    def _one_exposure(iexp, exposures, hazard, imp_fun):
        """Impact to one exposures.

        Parameters
        ----------
            iexp (int): array index of the exposure computed
            exposures (Exposure): exposures
            hazard (Hazard): a hazard
            imp_fun (Vulnerability): a vulnerability

        Returns
        -------
            event_row (np.array): hazard' events indices affecting exposure
            impact (np.array: impact for each event in event_row
        """
        # get assigned centroid of this exposure
        icen = int(exposures.assigned[iexp])

        # get intensities for this centroid
        event_row = hazard.intensity[:, icen].nonzero()[0]
        inten_val = np.asarray(hazard.intensity[event_row, icen].todense()). \
                    squeeze()
        # get affected fraction for these events
        fract = np.squeeze(hazard.fraction[:, icen].toarray()[event_row])

        # get MDD and PAA for these intensities
        mdd = imp_fun.interpolate(inten_val, 'mdd')
        paa = imp_fun.interpolate(inten_val, 'paa') * fract

        # impact on this exposure
        impact = exposures.value[iexp] * mdd * paa
        if np.count_nonzero(impact) > 0:
            # TODO: if needed?
            if (exposures.deductible[iexp] > 0) or \
                (exposures.cover[iexp] < exposures.value[iexp]):
                impact = np.minimum(np.maximum(impact - \
                                               exposures.deductible[iexp] * \
                                               paa, 0), exposures.cover[iexp])
        return event_row, impact
