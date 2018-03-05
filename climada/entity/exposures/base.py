"""
Define Exposures.
"""

__all__ = ['Exposures']

import os
import logging
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np

from climada.entity.exposures.source_mat import read as read_mat
from climada.entity.exposures.source_excel import read as read_excel
from climada.util.files_handler import to_str_list, get_file_names
import climada.util.checker as check
from climada.entity.tag import Tag
from climada.util.interpolation import Interpolator
from climada.util.config import CONFIG
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

class Exposures(object):
    """Defines exposures attributes and basic methods.

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
        impact_id (np.array): impact function id corresponding to each
            exposure
        deductible (np.array, default): deductible value for each exposure
        cover (np.array, default): cover value for each exposure
        category_id (np.array, optional): category id for each exposure
            (when defined)
        region_id (np.array, optional): region id for each exposure
            (when defined)
        assigned (np.array, optional): for a given hazard, id of the
            centroid(s) affecting each exposure. This values are filled by
            the 'assign' method
    """

    def __init__(self, file_name='', description='', var_names=None):
        """Fill values from file, if provided.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in 
                the file (default: DEF_VAR_NAME defined in the source modules)

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
            >>> exp_city = Exposures('Zurich.mat')
            Read exposures from Zurich.mat and checks consistency data.
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description, var_names)

    def clear(self):
        """Reinitialize attributes."""
        # Optional variables
        self.tag = Tag()
        self.ref_year = CONFIG["present_ref_year"]
        self.value_unit = 'NA'
        # Following values defined for each exposure
        # Obligatory variables
        self.coord = np.array([], np.float64).reshape(0, 2) # n_exp x (lat,lon)
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

    def check(self):
        """Check instance attributes.

        Raises
        ------
            ValueError
        """
        num_exp = len(self.id)
        if np.unique(self.id).size != num_exp:
            LOGGER.error("There are exposures with the same identifier.")
            raise ValueError
        self._check_obligatories(num_exp)
        self._check_optionals(num_exp)
        self._check_defaults(num_exp)

    def plot_value(self):
        """Plot exposures values binned over Earth's map."""
        return plot.geo_bin_from_array(self.coord, self.value, 'Value (%s)' % \
                                self.value_unit, \
                                os.path.splitext(os.path.basename( \
                                    self.tag.file_name))[0])

    def read(self, files, descriptions='', var_names=None):
        """Read and check exposures in parallel through files.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in 
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises
        ------
            ValueError
        """
        # Construct absolute path file names
        all_files = get_file_names(files)
        desc_list = to_str_list(len(all_files), descriptions, 'descriptions')
        var_list = to_str_list(len(all_files), var_names, 'var_names')
        self.clear()
        expo_part = Pool().map(self._read_one, all_files, desc_list, var_list)
        for expo, file in zip(expo_part, all_files):
            LOGGER.info('Read file: %s', file)    
            self.append(expo)

    def append(self, exposures):
        """Check and append variables of input Exposures to current Exposures.
        
        Parameters
        ----------
            exposures (Exposures): Exposures instance to append to current

        Raises
        ------
            ValueError
        """

        self._check_defaults(len(self.id))
        exposures.check()
        if self.id.size == 0:
            self.__dict__ = exposures.__dict__.copy()
            return
        
        self.tag.append(exposures.tag)
        if self.ref_year != exposures.ref_year:
            LOGGER.error("Append not possible. Different reference years.")
            raise ValueError
        if (self.value_unit == 'NA') and (exposures.value_unit != 'NA'):
            LOGGER.warning("Initial exposures does not have units.")
            self.value_unit = exposures.value_unit
        elif exposures.value_unit == 'NA':
            LOGGER.warning("Appended exposures does not have units.")
        elif self.value_unit != exposures.value_unit:
            LOGGER.error("Append not possible. Different units: %s != %s.", \
                             self.value_unit, exposures.value_unit)
            raise ValueError
        
        self.coord = np.append(self.coord, exposures.coord, axis=0)
        self.value = np.append(self.value, exposures.value)
        self.impact_id = np.append(self.impact_id, exposures.impact_id)
        self.id = np.append(self.id, exposures.id)
        self.deductible = np.append(self.deductible, exposures.deductible)
        self.cover = np.append(self.cover, exposures.cover)
        self.category_id = self._append_optional(self.category_id, \
                          exposures.category_id)
        self.region_id = self._append_optional(self.region_id, \
                        exposures.region_id)
        self.assigned = self._append_optional(self.assigned, \
                       exposures.assigned)
    
        # provide new ids to repeated ones
        _, indices = np.unique(self.id, return_index=True)
        new_id = np.max(self.id) + 1
        for dup_id in np.delete(np.arange(self.id.size), indices):
            self.id[dup_id] = new_id
            new_id += 1
            
    @staticmethod
    def _read_one(file_name, description='', var_names=None):
        """Read one file and fill attributes.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError

        Returns
        ------
            Exposures
        """
        exp = Exposures()
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            read_mat(exp, file_name, description, var_names)
        elif (extension == '.xlsx') or (extension == '.xls'):
            read_excel(exp, file_name, description, var_names)
        else:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        return exp
     
    @staticmethod
    def _append_optional(ini, to_add):
        """Append variable only if both are filled."""
        if (ini.size != 0) and (to_add.size != 0):
            ini = np.append(ini, to_add)    
        else:
            ini = np.array([], np.float64)
        return ini

    def _check_obligatories(self, num_exp):
        """Check coherence obligatory variables."""
        check.size(num_exp, self.value, 'Exposures.value')
        check.size(num_exp, self.impact_id, 'Exposures.impact_id')
        check.shape(num_exp, 2, self.coord, 'Exposures.coord')

    def _check_defaults(self, num_exp):
        """Check coherence optional variables. Warn and set default values \
        if empty."""
        self.deductible = check.array_default(num_exp, self.deductible, \
                                 'Exposures.deductible', np.zeros(num_exp))
        self.cover = check.array_default(num_exp, self.cover, \
                                 'Exposures.cover', self.value)

    def _check_optionals(self, num_exp):
        """Check coherence optional variables. Warn if empty."""
        check.array_optional(num_exp, self.category_id, \
                             'Exposures.category_id')
        check.array_optional(num_exp, self.region_id, \
                             'Exposures.region_id')
        check.array_optional(num_exp, self.assigned, 'Exposures.assigned')
