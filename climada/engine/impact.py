"""
=====================
Impact
=====================

Impact class contains the event impact for a given exposures, damage functions
and hazard.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Nov 13 13:29:32 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import numpy as np

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard

class Impact(object):
    """ Contains the Impact variables and methods to compute them"""

    def __init__(self):
        # references to inputs
        self.exposures_tag = Tag()
        self.impact_funcs_tag = Tag()
        self.hazard_tag = TagHazard()
        # impact pro exposure
        self.at_exp = np.array([])
        # impact pro hazard event
        self.at_event = np.array([])
        # total exposure value affected
        self.tot_value = 0
        # total expecte impact
        self.tot = 0

    def calc(self, exposures, impact_funcs, hazard):
        """ Computes the impact for a given hazard, exposures and impact
        functions"""

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
                event_row, impact = self.one_exposure(iexp, exposures, \
                                                        hazard, imp_fun)

                # add values to impact impact
                self.at_event[event_row] += impact
                self.at_exp[iexp] += sum(impact * hazard. \
                           frequency[event_row].reshape(len(event_row),))
                self.tot_value += exposures.value[iexp]

        self.tot = sum(self.at_event * hazard.frequency)

    def one_exposure(self, iexp, exposures, hazard, imp_fun):
        """ Compute for one exposure the impact it receives due to the events
        which affect it.
        INPUTS:
            - iexp: array index of the exposure computed
            - exposures: derived Exposure class
            - hazard: derived Hazard class
            - imp_fun: ImpactFunc class related to the input hazard and
            exposure
        OUTPUTS:
            - event_row: row indices of the hazards events that affect the
            exposure
            - impact: impact at this exposure for each event in event_row
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
