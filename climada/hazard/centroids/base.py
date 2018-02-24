"""
Define Centroids class.
"""

__all__ = ['Centroids']

import os
import warnings
from array import array
import numpy as np

from climada.hazard.centroids.source_excel import read as read_excel
from climada.hazard.centroids.source_mat import read as read_mat
import climada.util.checker as check
from climada.entity.tag import Tag

class Centroids(object):
    """Definition of the irregular grid through coordinates named centroids.
    
    Attributes
    ----------
        tag (Tag): information about the source
        coord (np.array): 2d array. Each row contains the coordinates for one
            centroid. The first column is for latitudes and the second for
            longitudes (in degrees)
        id (np.array): an id for each centroid
        region_id (np.array, optional): region id for each centroid
            (when defined) 
    """

    def __init__(self, file_name='', description=''):
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
        self.tag = Tag(file_name, description)
        self.coord = np.array([]).reshape((0, 2))
        self.id = np.array([], np.int64)
        self.region_id = np.array([], np.int64)
        #self.mask = 0

        # Load values from file_name if provided
        if file_name != '':
            self.load(file_name, description)

    def check(self):
        """Check instance attributes.

        Raises
        ------
            ValueError
        """
        num_exp = len(self.id)
        if np.unique(self.id).size != num_exp:
            raise ValueError('There are centroids with the same identifier.')
        check.shape(num_exp, 2, self.coord, 'Centroids.coord')
        check.array_optional(num_exp, self.region_id, \
                                 'Centroids.region_id')

    def load(self, file_name, description=''):
        """Read, check hazard (and its contained centroids).

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError
        """
        self.read(file_name, description)
        self.check()

    def read(self, file_name, description=''):
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

    def append(self, centroids):
        """Append input centroids coordinates to current. Id is perserved if 
        not present in current centroids. Otherwise, a new id is provided.
        Returns the array position of each appended centroid. 
        
        Parameters
        ----------
            centroids (Centroids): Centroids instance to append
        Returns
        -------
            array
        """
        self.check()
        centroids.check()

        if self.id.size == 0:
            self.__dict__ = centroids.__dict__.copy()
            return np.arange(centroids.id.size)

        self.tag.append(centroids.tag)

        # Check if region id need to be considered
        regions = True
        if (self.region_id.size == 0) | (centroids.region_id.size == 0):
            regions = False
            self.region_id = np.array([], np.int64)
            warnings.warn("Centroids.region_id is not going to be set.")

        new_pos = array('L')
        new_id = array('L')
        new_reg = array('l')
        new_lat = array('d')
        new_lon = array('d')
        max_id = int(np.max(self.id))
        for cnt, (centr, centr_id) \
        in enumerate(zip(centroids.coord, centroids.id)):
            found = np.where((centr == self.coord).all(axis=1))[0]
            if found.size > 0:
                new_pos.append(found[0])
                if (centr_id in self.id) and \
                (centr_id != self.id[found[0]]):
                    max_id += 1
                    self.id[found[0]] = max_id
                else:
                    self.id[found[0]] = centr_id
                    max_id = max(max_id, centr_id)
                if regions:
                    self.region_id[found[0]] = centroids.region_id[cnt]
            else:
                new_pos.append(self.coord.shape[0] + len(new_lat))
                new_lat.append(centr[0])
                new_lon.append(centr[1])
                if centr_id in self.id:
                    max_id += 1
                    new_id.append(max_id)
                else:
                    new_id.append(centr_id)
                    max_id = max(max_id, centr_id)
                if regions:
                    new_reg.append(centroids.region_id[cnt])

        self.coord = np.append(self.coord, np.transpose( \
                np.array([new_lat, new_lon])), axis=0)
        self.id = np.append(self.id, new_id).astype(int)
        if regions:
            self.region_id = np.append(self.region_id, new_reg)

        return new_pos
