"""
Define Measures class.
"""

__all__ = ['Measures']

import os
import logging
from pathos.multiprocessing import ProcessingPool as Pool

from climada.entity.measures.action import Action
from climada.entity.measures.source_mat import read as read_mat
from climada.entity.measures.source_excel import read as read_excel
from climada.util.files_handler import to_str_list, get_file_names
from climada.entity.tag import Tag

LOGGER = logging.getLogger(__name__)

class Measures(object):
    """Contains measures of type Measures.

    Attributes
    ----------
        tag (Taf): information about the source data
        _data (dict): cotains Action classes. It's not suppossed to be
            directly accessed. Use the class methods instead.
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
            >>> act_1 = Action()
            >>> act_1.name = 'Seawall'
            >>> act_1.color_rgb = np.array([0.1529, 0.2510, 0.5451])
            >>> act_1.hazard_intensity = (1, 0)
            >>> act_1.mdd_impact = (1, 0)
            >>> act_1.paa_impact = (1, 0)
            >>> meas = Measures()
            >>> meas.add_action(act_1)
            >>> meas.tag.description = "my dummy measures."
            >>> meas.check()
            Fill measures with values and check consistency data.
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description, var_names)

    def clear(self):
        """Reinitialize attributes.""" 
        self.tag = Tag()
        self._data = dict() # {name: Action()}
        
    def add_action(self, action):
        """Add an Action.
        
        Parameters
        ----------
            action (Action): Action instance

        Raises
        ------
            ValueError
        """
        if not isinstance(action, Action):
            LOGGER.error("Input value is not of type Action.")
            raise ValueError
        if action.name == 'NA':
            LOGGER.error("Input Action's name not set.")
            raise ValueError
        self._data[action.name] = action

    def remove_action(self, name=None):
        """Remove Action with provided name. Delete all actions if no input
        name
        
        Parameters
        ----------
            action (Action): Action instance

        Raises
        ------
            ValueError
        """
        if name is not None:
            try:
                del self._data[name]
            except KeyError:
                LOGGER.warning('No Action with name %s.', name)
        else:
            self._data = dict()

    def get_action(self, name=None):
        """Get Action with input name. Get all if no name provided.
        Parameters
        ----------
            name (str, optional): action type

        Returns
        -------
            list(Action)
        """
        if name is not None:
            try:
                return self._data[name]
            except KeyError:
                return list()
        else:
            return list(self._data.values())

    def get_names(self):
        """Get all Action names"""
        return list(self._data.keys())

    def num_action(self):
        """Get number of actions contained """
        return len(self._data.keys())

    def check(self):
        """Check instance attributes.

        Raises
        ------
            ValueError
        """
        for act_name, act in self._data.items():
            if (act_name != act.name) | (act.name == 'NA'):
                raise ValueError('Wrong Action.name: %s != %s' %\
                                (act_name, act.name))
            act.check()

    def read(self, files, descriptions='', var_names=None):
        """Read and check measures in parallel through files.

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
        
    def append(self, measures):
        """Check and append actions of input Measures to current Measures. 
        Overwrite actions if same name.
        
        Parameters
        ----------
            measures (Measures): Measures instance to append

        Raises
        ------
            ValueError
        """
        measures.check()
        if self.num_action() == 0:
            self.__dict__ = measures.__dict__.copy()
            return
        
        self.tag.append(measures.tag)
        
        new_act = measures.get_action()
        for action in new_act:
            self.add_action(action)

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
            Measures
        """
        meas = Measures()
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            read_mat(meas, file_name, description, var_names)
        elif (extension == '.xlsx') or (extension == '.xls'):
            read_excel(meas, file_name, description, var_names)
        else:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        return meas
