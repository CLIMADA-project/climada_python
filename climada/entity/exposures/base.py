"""
Define Exposures ABC.
"""

import warnings
import numpy as np

from climada.entity.loader import Loader
import climada.util.auxiliar as aux
from climada.entity.tag import Tag
from climada.util.interpolation import Interpolator

class Exposures(Loader):
    """Contains the exposures values.

    Attributes
    ----------
        tag (Tag): information about the source data
        ref_year (int): reference year
        value_unit (str): unit of the exposures values
        id (np.array): an id for each exposure
        coord (np.array): 2d array. Each row contains the coordinates for one
            exposure. The first column is for latitudes and the second for
            longitudes
        value (np.array): a value for each exposure
        deductible (np.array): deductible value for each exposure
        cover (np.array): cover value for each exposure
        impact_id (np.array): impact function id corresponding to each
            exposure
        category_id (np.array, optional): category id for each exposure
            (when defined)
        region_id (np.array, optional): region id for each exposure
            (when defined)
        assigned (np.array, optional): for a given hazard, id of the
            centroid(s) affecting each exposure. This values are filled by
            the 'assign' method
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
            This is an abstract class, it can't be instantiated.
        """
        self.tag = Tag(file_name, description)
        self.ref_year = 0
        self.value_unit = 'NA'
        # Followng values defined for each exposure
        self.id = np.array([], np.int64)
        self.coord = np.array([]) # 2d array (n_exp x 2(lat,lon))
        self.value = np.array([])
        self.deductible = np.array([])
        self.cover = np.array([])
        self.impact_id = np.array([], np.int64)
        self.category_id = np.array([], np.int64)
        self.region_id = np.array([], np.int64)

        # Assignment of hazard centroids to each exposure
        # Computed in function 'assign'
        self.assigned = np.array([])

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description)

    def assign(self, hazard, method=Interpolator.method[0], \
               dist=Interpolator.dist_def[0], threshold=100):
        """Compute the hazard centroids ids affecting to each exposure.

        Parameters
        ----------
            hazard (subclass Hazard): one hazard
            method (str, optional): interpolation method, neareast neighbor by
                default. The different options are provided by the class
                attribute 'method' of the Interpolator class
            dist (str, optional): distance used, euclidian approximation by
                default. The different options are provided by the class
                attribute 'dist_def' of the Interpolator class
            threshold (float, optional): threshold distance in km between
                exposure coordinate and hazard's centroid. A warning is thrown
                when the threshold is exceeded. Default value: 100km.

        Raises
        ------
            ValueError
        """
        interp = Interpolator(threshold)
        self.assigned = interp.interpol_index(hazard.centroids.coord, \
                                              self.coord, method, dist)

    def geo_coverage(self):
        """Get geographic coverage of all the exposures together.

        Returns
        -------
            polygon of coordinates
        """
        # TODO

    def check(self):
        """ Override Loader check."""
        num_exp = len(self.id)
        aux.check_size(num_exp, self.value, 'exposures values')
        aux.check_size(num_exp, self.impact_id, 'exposures impact functions')
        aux.check_size(2, self.coord[0], 'exposures coordinates')
        aux.check_size(num_exp, self.coord[:, 0], 'exposures coordinates')

        if self.deductible.size == 0:
            warnings.warn('Empty exposures deductibles. Default values set.')
            self.deductible = np.zeros(num_exp)
        else:
            aux.check_size(num_exp, self.deductible, 'exposures deductibles')

        if self.cover.size == 0:
            warnings.warn('Empty exposures coverages. Default values set.')
            self.cover = self.value
        else:
            aux.check_size(num_exp, self.cover, 'exposures coverages')

        if self.category_id.size == 0:
            warnings.warn('Exposures categories not set.')
        else:
            aux.check_size(num_exp, self.category_id, 'exposures categories')

        if self.region_id.size == 0:
            warnings.warn('Exposures regions not set.')
        else:
            aux.check_size(num_exp, self.region_id, 'exposures regions')

        if self.assigned.size == 0:
            warnings.warn('Exposures assigned centroids not set.')
        else:
            aux.check_size(num_exp, self.assigned, \
                            'exposures assigned centroids')
