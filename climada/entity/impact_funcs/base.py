"""
Define ImpactFuncs class.
"""

__all__ = ['ImpactFuncs']

import os
import logging
from pathos.multiprocessing import ProcessingPool as Pool

from climada.entity.impact_funcs.source_excel import read as read_excel
from climada.entity.impact_funcs.source_mat import read as read_mat
from climada.util.files_handler import to_str_list, get_file_names
from climada.entity.impact_funcs.vulnerability import Vulnerability
from climada.entity.tag import Tag
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

class ImpactFuncs(object):
    """Contains impact functions of type Vulnerability.

    Attributes
    ----------
        tag (Taf): information about the source data
        _data (dict): contains Vulnerability classes. It's not suppossed to be
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
            >>> fun_1 = Vulnerability()
            >>> fun_1.haz_type = 'TC'
            >>> fun_1.id = 3
            >>> fun_1.intensity = np.array([0, 20])
            >>> fun_1.paa = np.array([0, 1])
            >>> fun_1.mdd = np.array([0, 0.5])
            >>> imp_fun = ImpactFuncs()
            >>> imp_fun.add_vulner(fun_1)
            >>> imp_fun.check()
            Fill impact functions with values and check consistency data.
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description, var_names)

    def clear(self):
        """Reinitialize attributes."""        
        self.tag = Tag()
        self._data = dict() # {hazard_id : {id:Vulnerability}}

    def add_vulner(self, vulner):
        """Add a Vulnerability. Overwrite existing if same id.

        Parameters
        ----------
            vulner (Vulnerability): vulnerability instance

        Raises
        ------
            ValueError
        """
        if not isinstance(vulner, Vulnerability):
            LOGGER.error("Input value is not of type Vulnerability.")
            raise ValueError
        if vulner.haz_type == 'NA':
            LOGGER.error("Input Vulnerability's hazard type not set.")
            raise ValueError
        if vulner.id == 'NA':
            LOGGER.error("Input Vulnerability's id not set.")
            raise ValueError
        if vulner.haz_type not in self._data:
            self._data[vulner.haz_type] = dict()
        self._data[vulner.haz_type][vulner.id] = vulner

    def remove_vulner(self, haz_type=None, vul_id=None):
        """Remove vulenerability(ies) with provided hazard type and/or id.
        If no input provided, all vulnerabilities are removed.
        
        Parameters
        ----------
            haz_type (str, optional): all vulnerabilities with this hazard
            vul_id (int, optional): all vulnerabilities with this id
        """
        if (haz_type is not None) and (vul_id is not None):
            try:
                del self._data[haz_type][vul_id]
            except KeyError:
                LOGGER.warning("No Vulnerability with hazard %s and id %s.", \
                             haz_type, vul_id)
        elif haz_type is not None:
            try:
                del self._data[haz_type]
            except KeyError:
                LOGGER.warning("No Vulnerability with hazard %s.", haz_type)
        elif vul_id is not None:
            haz_remove = self.get_hazard_types(vul_id)
            if not haz_remove:
                LOGGER.warning("No Vulnerability with id %s.", vul_id)
            for vul_haz in haz_remove:
                del self._data[vul_haz][vul_id]
        else:
            self._data = dict()

    def get_hazard_types(self, vul_id=None):
        """Get vulnerabilities hazard types contained for the id provided.
        Return all hazard types if no input id.

        Returns
        -------
            list
        """
        if vul_id is None:
            return list(self._data.keys())

        haz_types = []
        for vul_haz, vul_dict in self._data.items():
            if vul_id in vul_dict:
                haz_types.append(vul_haz)
        return haz_types

    def get_ids(self, haz_type=None):
        """Get vulnerabilities ids contained for the hazard type provided.
        Return all ids for each hazard type if no input hazard type.

        Parameters
        ----------
            haz_type (str, optional): hazard type from which to obtain the ids

        Returns
        -------
            list(Vulnerability.id) (if haz_type provided),
            {Vulnerability.haz_type : list(Vulnerability.id)} (if no haz_type)
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

    def get_vulner(self, haz_type=None, vul_id=None):
        """Get Vulnerability(ies) of input hazard type and/or id.
        If no input provided, all vulnerabilities are returned.

        Parameters
        ----------
            haz_type (str, optional): hazard type
            vul_id (int, optional): vulnerability id

        Returns
        -------
            list(Vulnerability) (if haz_type and/or vul_id)
            list(Vulnerability) (if haz_type and/or vul_id)
            {Vulnerability.haz_type : {Vulnerability.id : Vulnerability}}
                (if None)
        """
        if (haz_type is not None) and (vul_id is not None):
            try:
                return [self._data[haz_type][vul_id]]
            except KeyError:
                return list()
        elif haz_type is not None:
            try:
                return list(self._data[haz_type].values())
            except KeyError:
                return list()
        elif vul_id is not None:
            haz_return = self.get_hazard_types(vul_id)
            vul_return = []
            for vul_haz in haz_return:
                vul_return.append(self._data[vul_haz][vul_id])
            return vul_return
        else:
            return self._data

    def num_vulner(self, haz_type=None, vul_id=None):
        """Get number of vulnerbilities contained with input hazard type and
        /or id. If no input provided, get total number of vulnerabilites.

        Parameters
        ----------
            haz_type (str, optional): hazard type
            vul_id (int, optional): vulnerability id

        Returns
        -------
            int      
        """
        if (haz_type != None) or (vul_id != None):
            return len(self.get_vulner(haz_type, vul_id))

        return sum(len(vul_list) for vul_list in self.get_ids().values())
        
    def check(self):
        """Check instance attributes.

        Raises
        ------
            ValueError
        """
        for key_haz, vul_dict in self._data.items():
            for vul_id, vul in vul_dict.items():
                if (vul_id != vul.id) | (vul_id == 'NA'):
                    LOGGER.error("Wrong Vulnerability.id: %s != %s.", vul_id, \
                                 vul.id)
                    raise ValueError
                if (key_haz != vul.haz_type) | (key_haz == 'NA'):
                    LOGGER.error("Wrong Vulnerability.haz_type: %s != %s.",\
                                 key_haz, vul.haz_type)
                    raise ValueError
                vul.check()

    def read(self, files, descriptions='', var_names=None):
        """Read and check impact functions in parallel through files.

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
        imp_part = Pool().map(self._read_one, all_files, desc_list, var_list)
        for imp, file in zip(imp_part, all_files):
            LOGGER.info('Read file: %s', file)    
            self.append(imp)

    def append(self, impact_funcs):
        """Check and append vulnerabilities of input ImpactFuncs to current
        ImpactFuncs. Overwrite vulnerability if same id.
        
        Parameters
        ----------
            impact_funcs (ImpactFuncs): ImpactFuncs instance to append

        Raises
        ------
            ValueError
        """
        impact_funcs.check()
        if self.num_vulner() == 0:
            self.__dict__ = impact_funcs.__dict__.copy()
            return
        
        self.tag.append(impact_funcs.tag)
        
        new_vulner = impact_funcs.get_vulner()
        for _, vul_dict in new_vulner.items():
            for _, vul in vul_dict.items():
                self.add_vulner(vul)

    def plot(self, haz_type=None, vul_id=None):
        """Plot impact functions of selected hazard (all if not provided) and
        selected function id (all if not provided).

        Parameters
        ----------
            haz_type (str, optional): hazard type
            vul_id (int, optional): id of the function

        Returns
        -------
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        num_plts = self.num_vulner(haz_type, vul_id)
        # Select all hazard types to plot
        if haz_type is not None:
            hazards = [haz_type]
        else:
            hazards = self._data.keys()

        # Plot
        do_show = plot.SHOW
        plot.SHOW = False
        graph = plot.Graph2D('', num_plts)
        for sel_haz in hazards:
            if vul_id is not None:
                self._data[sel_haz][vul_id].plot(graph)
            else:
                for sel_id in self._data[sel_haz].keys():
                    self._data[sel_haz][sel_id].plot(graph)
        plot.SHOW = do_show
        plot.show()
        return graph.get_elems()

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
            ImpactFuncs
        """
        new_imp = ImpactFuncs()
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            read_mat(new_imp, file_name, description, var_names)
        elif (extension == '.xlsx') or (extension == '.xls'):
            read_excel(new_imp, file_name, description, var_names)
        else:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        return new_imp
