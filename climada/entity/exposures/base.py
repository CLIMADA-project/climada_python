"""
Define Exposures ABC.
"""

import numpy as np

from climada.entity.loader import Loader
import climada.util.auxiliar as aux
from climada.entity.tag import Tag
from climada.util.interpolation import Interpolator
from climada.util.config import config

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
            longitudes (in degrees)
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
            >>> exp_city = Exposures()
            >>> exp_city.coord = np.array([[40.1, 8], [40.2, 8], [40.3, 8]])
            >>> exp_city.value = np.array([5604, 123, 9005001])
            >>> exp_city.impact_id = np.array([1, 1, 1])
            >>> exp_city.id = np.array([11, 12, 13])
            >>> exp_city.check()
            Fill exposures with values and check consistency data.
        """
        # Optional variables
        self.tag = Tag(file_name, description)
        self.ref_year = config["present_ref_year"]
        self.value_unit = 'NA'
        # Following values defined for each exposure
        # Obligatory variables
        self.coord = np.array([], np.float64) # 2d array (n_exp x 2(lat,lon))
        self.value = np.array([], np.float64)
        self.impact_id = np.array([], np.int64)
        self.id = np.array([], np.int64)
        # Optional variables. Default values set in check if not filled.
        self.deductible = np.array([], np.float64)
        self.cover = np.array([], np.float64)
        # Optional variables. No default values set in check if not filled.
        self.category_id = np.array([], np.int64)
        self.region_id = np.array([], np.int64)
        self.assigned = np.array([], np.int64)

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
        self._check_obligatories(num_exp)
        self._check_optionals(num_exp)
        self._check_defaults(num_exp)

    def _check_obligatories(self, num_exp):
        """Check coherence obligatory variables."""
        aux.check_size(num_exp, self.value, 'Exposures.value')
        aux.check_size(num_exp, self.impact_id, 'Exposures.impact_id')
        aux.check_size(2, self.coord[0], 'Exposures.coord')
        aux.check_size(num_exp, self.coord[:, 0], 'Exposures.coord')

    def _check_defaults(self, num_exp):
        """Check coherence optional variables. Warn and set default values \
        if empty."""
        self.deductible = aux.check_array_default(num_exp, self.deductible, \
                                 'Exposures.deductible', np.zeros(num_exp))
        self.cover = aux.check_array_default(num_exp, self.cover, \
                                 'Exposures.cover', self.value)

    def _check_optionals(self, num_exp):
        """Check coherence optional variables. Warn if empty."""
        aux.check_array_optional(num_exp, self.category_id, \
                                 'Exposures.category_id')
        aux.check_array_optional(num_exp, self.region_id, \
                         'Exposures.region_id')
        aux.check_array_optional(num_exp, self.assigned, 'Exposures.assigned')
