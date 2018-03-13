"""
Define MeasureSet class.
"""

__all__ = ['MeasureSet']

import logging
from pathos.multiprocessing import ProcessingPool as Pool

from climada.entity.measures.measure import Measure
from climada.entity.measures.source import read as read_source
from climada.util.files_handler import to_str_list, get_file_names
from climada.entity.tag import Tag

LOGGER = logging.getLogger(__name__)

class MeasureSet(object):
    """Contains MeasureSet of type MeasureSet.

    Attributes
    ----------
        tag (Taf): information about the source data
        _data (dict): cotains Measure classes. It's not suppossed to be
            directly accessed. Use the class methods instead.
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters
        ----------
            file_name (str or list(str), optional): absolute file name(s) or 
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises
        ------
            ValueError

        Examples
        --------
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
            Fill MeasureSet with values and check consistency data.
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
        
        Parameters
        ----------
            meas (Measure): Measure instance

        Raises
        ------
            ValueError
        """
        if not isinstance(meas, Measure):
            LOGGER.error("Input value is not of type Measure.")
            raise ValueError
        if meas.name == 'NA':
            LOGGER.error("Input Measure's name not set.")
            raise ValueError
        self._data[meas.name] = meas

    def remove_measure(self, name=None):
        """Remove Measure with provided name. Delete all Measures if no input
        name
        
        Parameters
        ----------
            name (str, optional): measure name

        Raises
        ------
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
        Parameters
        ----------
            name (str, optional): measure name

        Returns
        -------
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

        Raises
        ------
            ValueError
        """
        for act_name, act in self._data.items():
            if (act_name != act.name) | (act.name == 'NA'):
                raise ValueError('Wrong Measure.name: %s != %s' %\
                                (act_name, act.name))
            act.check()

    def read(self, files, descriptions='', var_names=None):
        """Read and check MeasureSet in parallel through files.

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
        all_files = get_file_names(files)
        desc_list = to_str_list(len(all_files), descriptions, 'descriptions')
        var_list = to_str_list(len(all_files), var_names, 'var_names')
        self.clear()       
        meas_part = Pool().map(self._read_one, all_files, desc_list, var_list)
        for meas, file in zip(meas_part, all_files):
            LOGGER.info('Read file: %s', file)    
            self.append(meas)
        
    def append(self, meas):
        """Check and append measures of input MeasureSet to current MeasureSet. 
        Overwrite Measure if same name.
        
        Parameters
        ----------
            meas (MeasureSet): MeasureSet instance to append

        Raises
        ------
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
    def _read_one(file_name, description, var_names):
        """Read input file.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str): description of the source data
            var_names (dict): name of the variables in the file (e.g. 
                      DEF_VAR_NAME defined in the source modules)

        Raises
        ------
            ValueError
            
        Returns
        ------
            MeasureSet
        """
        new_meas = MeasureSet()
        read_source(new_meas, file_name, description, var_names)
        return new_meas
