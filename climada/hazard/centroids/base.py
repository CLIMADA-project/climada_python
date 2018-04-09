"""
Define Centroids class.
"""

__all__ = ['Centroids']

import logging
from array import array
import numpy as np

from climada.hazard.centroids.tag import Tag
from climada.hazard.centroids.source import read as read_source
import climada.util.checker as check
from climada.util.coordinates import Coordinates

LOGGER = logging.getLogger(__name__)

class Centroids(object):
    """Definition of the hazard coordinates.
    
    Attributes:
        tag (Tag): information about the source
        coord (Coordinates): Coordinates instance
        id (np.array): an id for each centroid
        region_id (np.array, optional): region id for each centroid
            (when defined) 
        dist_coast (np.array, optional): distance to coast   
        admin0_name (str, optional): admin0 country name
        admin0_iso3 (str, optional): admin0 ISO3 country name
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters:
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Raises:
            ValueError

        Examples:
            Fill centroids attributes by hand:
            
            >>> centr = Centroids()
            >>> centr.coord = IrregularGrid([[0,-1], [0, -2]])
            >>> ...
            
            Read centroids from file:
            
            >>> centr = Centroids(HAZ_DEMO_XLS, 'Centroids demo')
        """
        self.clear()
        if file_name != '':
            self.read_one(file_name, description)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self.coord = Coordinates()
        self.id = np.array([], int)
        self.region_id = np.array([], int)        
        self.dist_coast = np.array([], float)     
        self.admin0_name = ''
        self.admin0_iso3 = ''

    def check(self):
        """Check instance attributes.

        Raises:
            ValueError
        """
        num_exp = len(self.id)
        if np.unique(self.id).size != num_exp:
            LOGGER.error("There are centroids with the same identifier.")
            raise ValueError
        check.shape(num_exp, 2, self.coord, 'Centroids.coord')            
        check.array_optional(num_exp, self.region_id, \
                                 'Centroids.region_id')
        check.array_optional(num_exp, self.dist_coast, \
                                 'Centroids.dist_coast')

    def read_one(self, file_name, description='', var_names=None):
        """ Read input file.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises:
            TypeError, ValueError
        """
        read_source(self, file_name, description, var_names)
        LOGGER.info('Read file: %s', file_name)  

    def append(self, centroids):
        """Append input centroids coordinates to current. Id is perserved if 
        not present in current centroids. Otherwise, a new id is provided.
        Returns the array position of each appended centroid. 
        
        Parameters:
            centroids (Centroids): Centroids instance to append
            
        Returns:
            array
        """
        centroids.check()

        self.tag.append(centroids.tag)

        if self.id.size == 0:
            self.__dict__ = centroids.__dict__.copy()
            return np.arange(centroids.id.size)
        elif centroids.id.size == 0:
            return np.array([])

        # Check if region id need to be considered
        regions = True
        if (self.region_id.size == 0) | (centroids.region_id.size == 0):
            regions = False
            self.region_id = np.array([], int)
            LOGGER.warning("Centroids.region_id is not going to be set.")

        new_pos, new_id, new_reg, new_lat, new_lon = \
            self._append_one(centroids, regions)

        self.coord = np.append(self.coord, np.transpose( \
                np.array([new_lat, new_lon])), axis=0)
        self.id = np.append(self.id, new_id).astype(int)
        if regions:
            self.region_id = np.append(self.region_id, new_reg)

        return new_pos

    def _append_one(self, centroids, regions):
        """Append one by one centroid."""
        new_pos = array('l')
        new_id = array('L')
        new_reg = array('l')
        new_lat = array('d')
        new_lon = array('d')
        max_id = int(np.max(self.id))
        for cnt, (centr_id, centr) \
        in enumerate(zip(centroids.id, centroids.coord)):
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

        return new_pos, new_id, new_reg, new_lat, new_lon
    