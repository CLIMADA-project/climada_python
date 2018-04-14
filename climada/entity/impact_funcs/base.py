"""
Define ImpactFuncSet class.
"""

__all__ = ['ImpactFuncSet']

import logging
from pathos.multiprocessing import ProcessingPool as Pool

from climada.entity.impact_funcs.source import read as read_source
from climada.util.files_handler import to_list, get_file_names
from climada.entity.impact_funcs.impact_func import ImpactFunc
from climada.entity.tag import Tag
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

class ImpactFuncSet(object):
    """Contains impact functions of type ImpactFunc.

    Attributes:
        tag (Tag): information about the source data
        _data (dict): contains ImpactFunc classes. It's not suppossed to be
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
            Fill impact functions with values and check consistency data:

            >>> fun_1 = ImpactFunc()
            >>> fun_1.haz_type = 'TC'
            >>> fun_1.id = 3
            >>> fun_1.intensity = np.array([0, 20])
            >>> fun_1.paa = np.array([0, 1])
            >>> fun_1.mdd = np.array([0, 0.5])
            >>> imp_fun = ImpactFuncSet()
            >>> imp_fun.add_func(fun_1)
            >>> imp_fun.check()

            Read impact functions from file and checks consistency data.

            >>> imp_fun = ImpactFuncSet(ENT_TEST_XLS)
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self._data = dict() # {hazard_id : {id:ImpactFunc}}

    def add_func(self, func):
        """Add a ImpactFunc. Overwrite existing if same id.

        Parameters:
            func (ImpactFunc): ImpactFunc instance

        Raises:
            ValueError
        """
        if not isinstance(func, ImpactFunc):
            LOGGER.error("Input value is not of type ImpactFunc.")
            raise ValueError
        if func.haz_type == 'NA':
            LOGGER.error("Input ImpactFunc's hazard type not set.")
            raise ValueError
        if func.id == 'NA':
            LOGGER.error("Input ImpactFunc's id not set.")
            raise ValueError
        if func.haz_type not in self._data:
            self._data[func.haz_type] = dict()
        self._data[func.haz_type][func.id] = func

    def remove_func(self, haz_type=None, fun_id=None):
        """Remove vulenerability(ies) with provided hazard type and/or id.
        If no input provided, all impact functions are removed.

        Parameters:
            haz_type (str, optional): all impact functions with this hazard
            fun_id (int, optional): all impact functions with this id
        """
        if (haz_type is not None) and (fun_id is not None):
            try:
                del self._data[haz_type][fun_id]
            except KeyError:
                LOGGER.warning("No ImpactFunc with hazard %s and id %s.", \
                             haz_type, fun_id)
        elif haz_type is not None:
            try:
                del self._data[haz_type]
            except KeyError:
                LOGGER.warning("No ImpactFunc with hazard %s.", haz_type)
        elif fun_id is not None:
            haz_remove = self.get_hazard_types(fun_id)
            if not haz_remove:
                LOGGER.warning("No ImpactFunc with id %s.", fun_id)
            for vul_haz in haz_remove:
                del self._data[vul_haz][fun_id]
        else:
            self._data = dict()

    def get_hazard_types(self, fun_id=None):
        """Get impact functions hazard types contained for the id provided.
        Return all hazard types if no input id.

        Parameters:
            fun_id (int, optional): id of an impact function

        Returns:
            list
        """
        if fun_id is None:
            return list(self._data.keys())

        haz_types = []
        for vul_haz, vul_dict in self._data.items():
            if fun_id in vul_dict:
                haz_types.append(vul_haz)
        return haz_types

    def get_ids(self, haz_type=None):
        """Get impact functions ids contained for the hazard type provided.
        Return all ids for each hazard type if no input hazard type.

        Parameters:
            haz_type (str, optional): hazard type from which to obtain the ids

        Returns:
            list(ImpactFunc.id) (if haz_type provided),
            {ImpactFunc.haz_type : list(ImpactFunc.id)} (if no haz_type)
        """
        if haz_type is None:
            out_dict = dict()
            for vul_haz, vul_dict in self._data.items():
                out_dict[vul_haz] = list(vul_dict.keys())
            return out_dict
        else:
            try:
                return list(self._data[haz_type].keys())
            except KeyError:
                return list()

    def get_func(self, haz_type=None, fun_id=None):
        """Get ImpactFunc(s) of input hazard type and/or id.
        If no input provided, all impact functions are returned.

        Parameters:
            haz_type (str, optional): hazard type
            fun_id (int, optional): ImpactFunc id

        Returns:
            list(ImpactFunc) (if haz_type and/or fun_id)
            list(ImpactFunc) (if haz_type and/or fun_id)
            {ImpactFunc.haz_type : {ImpactFunc.id : ImpactFunc}}
                (if None)
        """
        if (haz_type is not None) and (fun_id is not None):
            try:
                return [self._data[haz_type][fun_id]]
            except KeyError:
                return list()
        elif haz_type is not None:
            try:
                return list(self._data[haz_type].values())
            except KeyError:
                return list()
        elif fun_id is not None:
            haz_return = self.get_hazard_types(fun_id)
            vul_return = []
            for vul_haz in haz_return:
                vul_return.append(self._data[vul_haz][fun_id])
            return vul_return
        else:
            return self._data

    def num_funcs(self, haz_type=None, fun_id=None):
        """Get number of impact functions contained with input hazard type and
        /or id. If no input provided, get total number of impact functions.

        Parameters:
            haz_type (str, optional): hazard type
            fun_id (int, optional): ImpactFunc id

        Returns:
            int
        """
        if (haz_type != None) or (fun_id != None):
            return len(self.get_func(haz_type, fun_id))

        return sum(len(vul_list) for vul_list in self.get_ids().values())

    def check(self):
        """Check instance attributes.

        Raises:
            ValueError
        """
        for key_haz, vul_dict in self._data.items():
            for fun_id, vul in vul_dict.items():
                if (fun_id != vul.id) | (fun_id == 'NA'):
                    LOGGER.error("Wrong ImpactFunc.id: %s != %s.", fun_id, \
                                 vul.id)
                    raise ValueError
                if (key_haz != vul.haz_type) | (key_haz == 'NA'):
                    LOGGER.error("Wrong ImpactFunc.haz_type: %s != %s.",\
                                 key_haz, vul.haz_type)
                    raise ValueError
                vul.check()

    def read(self, files, descriptions='', var_names=None):
        """Read and check impact functions in parallel through files.

        Parameters:
            file_name (str or list(str), optional): absolute file name(s) or
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises:
            ValueError
        """
        # Construct absolute path file names
        all_files = get_file_names(files)
        desc_list = to_list(len(all_files), descriptions, 'descriptions')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        imp_part = Pool().map(self._read_one, all_files, desc_list, var_list)
        for imp, file in zip(imp_part, all_files):
            LOGGER.info('Read file: %s', file)
            self.append(imp)

    def append(self, impact_funcs):
        """Check and append impact functions of input ImpactFuncSet to current
        ImpactFuncSet. Overwrite ImpactFunc if same id.

        Parameters:
            impact_funcs (ImpactFuncSet): ImpactFuncSet instance to append

        Raises:
            ValueError
        """
        impact_funcs.check()
        if self.num_funcs() == 0:
            self.__dict__ = impact_funcs.__dict__.copy()
            return

        self.tag.append(impact_funcs.tag)

        new_func = impact_funcs.get_func()
        for _, vul_dict in new_func.items():
            for _, vul in vul_dict.items():
                self.add_func(vul)

    def plot(self, haz_type=None, fun_id=None):
        """Plot impact functions of selected hazard (all if not provided) and
        selected function id (all if not provided).

        Parameters:
            haz_type (str, optional): hazard type
            fun_id (int, optional): id of the function

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        num_plts = self.num_funcs(haz_type, fun_id)
        # Select all hazard types to plot
        if haz_type is not None:
            hazards = [haz_type]
        else:
            hazards = self._data.keys()

        graph = plot.Graph2D('', num_plts)
        for sel_haz in hazards:
            if fun_id is not None:
                self._data[sel_haz][fun_id].plot(graph)
            else:
                for sel_id in self._data[sel_haz].keys():
                    self._data[sel_haz][sel_id].plot(graph)
        return graph.get_elems()

    @staticmethod
    def _read_one(file_name, description='', var_names=None):
        """Read input file.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data
            var_names (dict, optional): name of the variables in the file

        Raises:
            ValueError

        Returns:
            ImpactFuncSet
        """
        new_imp = ImpactFuncSet()
        read_source(new_imp, file_name, description, var_names)
        return new_imp

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
