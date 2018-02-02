"""
Define Centroids class.
"""

__all__ = ['Centroids']

import os
import pickle
import numpy as np

from climada.hazard.centroids.source_excel import read as read_excel
from climada.hazard.centroids.source_mat import read as read_mat
import climada.util.checker as check
from climada.entity.tag import Tag

class Centroids(object):
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
        """Check instance attributes.

        Raises
        ------
            ValueError
        """
        num_exp = len(self.id)
        check.size(2, self.coord[0], 'Centroids.coord')
        check.size(num_exp, self.coord[:, 0], 'Centroids.coord')
        check.array_optional(num_exp, self.region_id, \
                                 'Centroids.region_id')

    def load(self, file_name, description=None, out_file_name=None):
        """Read, check hazard (and its contained centroids) and save to pkl.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
            out_file_name (str, optional): output file name to save as pkl

        Raises
        ------
            ValueError
        """
        self.read(file_name, description)
        self.check()
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)

    def read(self, file_name, description=None):
        """ Read input file.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError, KeyError
        """
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            read_mat(self, file_name, description)
        elif (extension == '.xlsx') or (extension == '.xls'):
            read_excel(self, file_name, description)
        else:
            raise TypeError('Input file extension not supported: %s.' % \
                            extension)
