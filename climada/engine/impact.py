"""
Define Impact class.
"""

import warnings
import numpy as np

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard

class Impact(object):
    """Impact for a given entity (exposures and impact functions) and hazard.

    Attributes
    ----------
        exposures_tag (Tag): information about the exposures
        impact_funcs_tag (Tag): information about the impact functions
        hazard_tag (TagHazard): information about the hazard
        at_exp (np.array): impact for each exposure
        at_event (np.array): impact for each hazard event
        tot_vale (float): total exposure value affected
        tot (float): total expected impact
    """

    def __init__(self):
        """ Empty initialization."""
        self.exposures_tag = Tag()
        self.impact_funcs_tag = Tag()
        self.hazard_tag = TagHazard()
        self.at_exp = np.array([])
        self.at_event = np.array([])
        self.tot_value = 0
        self.tot = 0

    def calc(self, exposures, impact_funcs, hazard):
        """Compute impact of an hazard to exposures.

        Parameters
        ----------
            exposures (subclass Exposures): exposures
            impact_funcs (subclass ImpactFucs): vulnerability functions
            hazard (subclass Hazard): hazard

        Examples
        --------
            Use Entity class
            >>> hazard = HazardMat('filename') # Set hazard
            >>> entity = Entity() # Load entity with default values
            >>> entity.exposures = ExposuresExcel('filename') # Set exposures
            >>> tc_impact = Impact()
            >>> tc_impact.calc(entity.exposures, entity.impact_functs, hazard)

            Specify only exposures and impact functions
            >>> hazard = HazardMat('filename') # Set hazard
            >>> funcs = ImpactFuncsExcel('filename') # Set impact functions
            >>> exposures = ExposuresExcel('filename') # Set exposures
            >>> tc_impact = Impact()
            >>> tc_impact.calc(exposures, funcs, hazard)
        """
        # 1. Assign centroids to each exposure if not done
        if exposures.assigned.size == 0:
            exposures.assign(hazard)

        # 2. Initialize values
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
        haz_type = hazard.tag.type
        # Get damage functions for this hazard
        haz_imp = impact_funcs.data[haz_type]

        # 3. Loop over exposures according to their impact function
        # Loop over impact functions
        for fun_id, imp_fun in haz_imp.items():
            # get indices of all the exposures with this impact function
            exp_iimp = np.where(exposures.impact_id[exp_idx] == fun_id)[0]

            # loop over selected exposures
            for iexp in exp_iimp:
                # compute impact on exposure
                event_row, impact = self._one_exposure(iexp, exposures, \
                                                        hazard, imp_fun)

                # add values to impact impact
                self.at_event[event_row] += impact
                self.at_exp[iexp] += sum(impact * hazard. \
                           frequency[event_row].reshape(len(event_row),))
                self.tot_value += exposures.value[iexp]

        self.tot = sum(self.at_event * hazard.frequency)

    @staticmethod
    def _one_exposure(iexp, exposures, hazard, imp_fun):
        """Impact to one exposures.

        Parameters
        ----------
            iexp (int): array index of the exposure computed
            exposures (subclass Exposure): exposures
            hazard (subclass Hazard): a hazard
            imp_fun (ImpactFunc): one impact function

        Returns
        -------
            event_row (np.array): hazard' events indices affecting exposure
            impact (np.array: impact for each event in event_row
        """
        # get assigned centroid of this exposure
        icen = int(exposures.assigned[iexp])

        # get intensities for this centroid
        event_row = hazard.intensity.indices[hazard.intensity.indptr[icen]: \
                    hazard.intensity.indptr[icen+1]]
        inten_val = hazard.intensity.data[hazard.intensity.indptr[icen]: \
                    hazard.intensity.indptr[icen+1]]
        # get affected fraction for these events
        fract = hazard.fraction[:, icen].toarray()[event_row].reshape( \
                               len(event_row),)

        # get MDD and PAA for these intensities
        mdd = imp_fun.interpolate(inten_val, 'mdd')
        paa = imp_fun.interpolate(inten_val, 'paa')*fract

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
