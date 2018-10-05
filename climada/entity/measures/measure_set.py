"""
Define MeasureSet class.
"""

__all__ = ['MeasureSet',
           'FILE_EXT']

import os
import copy
import logging

from climada.entity.measures.base import Measure
from climada.entity.measures.source import READ_SET
from climada.util.files_handler import to_list, get_file_names
from climada.entity.tag import Tag

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS'
           }
""" Supported files format to read from """

class MeasureSet():
    """Contains measures of type Measure. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (Tag): information about the source data
        _data (dict): cotains Measure classes. It's not suppossed to be
            directly accessed. Use the class methods instead.
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters:
            file_name (str or list(str), optional): absolute file name(s) or
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises:
            ValueError

        Examples:
            Fill MeasureSet with values and check consistency data:

            >>> act_1 = Measure()
            >>> act_1.name = 'Seawall'
            >>> act_1.color_rgb = np.array([0.1529, 0.2510, 0.5451])
            >>> act_1.hazard_intensity = (1, 0)
            >>> act_1.mdd_impact = (1, 0)
            >>> act_1.paa_impact = (1, 0)
            >>> meas = MeasureSet()
            >>> meas.add_Measure(act_1)
            >>> meas.tag.description = "my dummy MeasureSet."
            >>> meas.check()

            Read measures from file and checks consistency data:

            >>> meas = MeasureSet(ENT_TEMPLATE_XLS)
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self._data = dict() # {name: Measure()}

    def add_measure(self, meas):
        """Add an Measure.

        Parameters:
            meas (Measure): Measure instance

        Raises:
            ValueError
        """
        if not isinstance(meas, Measure):
            LOGGER.error("Input value is not of type Measure.")
            raise ValueError
        if not meas.name:
            LOGGER.error("Input Measure's name not set.")
            raise ValueError
        self._data[meas.name] = meas

    def remove_measure(self, name=None):
        """Remove Measure with provided name. Delete all Measures if no input
        name

        Parameters:
            name (str, optional): measure name

        Raises:
            ValueError
        """
        if name is not None:
            try:
                del self._data[name]
            except KeyError:
                LOGGER.warning('No Measure with name %s.', name)
        else:
            self._data = dict()

    def get_measure(self, name=None):
        """Get Measure with input name. Get all if no name provided.

        Parameters:
            name (str, optional): measure name

        Returns:
            list(Measure)
        """
        if name is not None:
            try:
                return self._data[name]
            except KeyError:
                return list()
        else:
            return list(self._data.values())

    def get_names(self):
        """Get all Measure names"""
        return list(self._data.keys())

    def num_measures(self):
        """Get number of measures contained """
        return len(self._data.keys())

    def check(self):
        """Check instance attributes.

        Raises:
            ValueError
        """
        for act_name, act in self._data.items():
            if (act_name != act.name) | (act.name == ''):
                raise ValueError('Wrong Measure.name: %s != %s' %\
                                (act_name, act.name))
            act.check()

    def read(self, files, descriptions='', var_names=None):
        """Read and check MeasureSet.

        Parameters:
            files (str or list(str)): absolute file name(s) or folder name
                containing the files to read
            descriptions (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises:
            ValueError
        """
        all_files = get_file_names(files)
        desc_list = to_list(len(all_files), descriptions, 'descriptions')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        for file, desc, var in zip(all_files, desc_list, var_list):
            self.append(self._read_one(file, desc, var))

    def append(self, meas):
        """Check and append measures of input MeasureSet to current MeasureSet.
        Overwrite Measure if same name.

        Parameters:
            meas (MeasureSet): MeasureSet instance to append

        Raises:
            ValueError
        """
        meas.check()
        if self.num_measures() == 0:
            self.__dict__ = meas.__dict__.copy()
            return

        self.tag.append(meas.tag)
        for measure in meas.get_measure():
            self.add_measure(measure)

    @staticmethod
    def get_sup_file_format():
        """ Get supported file extensions that can be read.

        Returns:
            list(str)
        """
        return list(FILE_EXT.keys())

    @staticmethod
    def get_def_file_var_names(src_format):
        """Get default variable names for given file format.

        Parameters:
            src_format (str): extension of the file, e.g. '.xls', '.mat'.

        Returns:
            dict: dictionary with variable names
        """
        try:
            if '.' not in src_format:
                src_format = '.' + src_format
            return copy.deepcopy(READ_SET[FILE_EXT[src_format]][0])
        except KeyError:
            LOGGER.error('File extension not supported: %s.', src_format)
            raise ValueError

    @staticmethod
    def _read_one(file_name, description='', var_names=None):
        """Read input file.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data
            var_names (dict), optional): name of the variables in the file

        Raises:
            ValueError

        Returns:
            MeasureSet
        """
        LOGGER.info('Reading file: %s', file_name)
        new_meas = MeasureSet()
        new_meas.tag = Tag(file_name, description)

        extension = os.path.splitext(file_name)[1]
        try:
            reader = READ_SET[FILE_EXT[extension]][1]
        except KeyError:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        reader(new_meas, file_name, var_names)

        return new_meas

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
