"""
Define BushFire class.
"""

__all__ = ['BushFire']

import logging
import numpy as np
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'BF'
""" Hazard type acronym for Bush Fire """

class BushFire(Hazard):
    """Contains bush fire events.

    Attributes:
    """

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)

    def read_firms_csv(self, firms, centroids, description=''):
        """Read historical bush fires."""

        # distinguish one event in file

        # cut by centroids range
        sel_lines = np.zeros(())
        for event_lines in sel_lines:
            bf_haz = self._one_firms_csv(centroids)
            self.append(bf_haz)

#        self.tag.description = description
        raise NotImplementedError

    @staticmethod
    def _one_firms_csv(centroids):

        bf_haz = BushFire()

        # FILL these values
#        bf_haz.tag =
#        bf_haz.units =
#        # following values are defined for each event
#        bf_haz.centroids = centroids
#        bf_haz.event_id = np.array([], int)
#        bf_haz.frequency = np.array([])
#        bf_haz.event_name = list()
#        bf_haz.date = np.array([], int)
#        bf_haz.orig = np.array([], bool)
#        # following values are defined for each event and centroid
#        bf_haz.intensity = sparse.lil_matrix((1, centroids.id.size))
#        bf_haz.fraction = sparse.lil_matrix((1, centroids.id.size))

        return bf_haz

    def calc_probabilistic(self, n_ens):
        """ For every event, compute n_ens probabilistic events."""
        raise NotImplementedError
