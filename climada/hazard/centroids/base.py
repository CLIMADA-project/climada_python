"""
Define Centroids class.
"""

import numpy as np

from climada.entity.loader import Loader
import climada.util.checker as check
from climada.entity.tag import Tag

class Centroids(Loader):
    """Definition of the irregular grid."""

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
            This is an abstract class, it can't be instantiated.
        """
        self.tag = Tag()
        self.coord = np.array([])
        self.id = np.array([], np.int64)
        self.region_id = np.array([], np.int64)
        #self.mask = 0

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def check(self):
        """ Check if attributes are coherent."""
        num_exp = len(self.id)
        check.size(2, self.coord[0], 'Centroids.coord')
        check.size(num_exp, self.coord[:, 0], 'Centroids.coord')
        check.array_optional(num_exp, self.region_id, \
                                 'Centroids.region_id')
