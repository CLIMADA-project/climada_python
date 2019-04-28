"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define impact functions for river flood .
"""

__all__ = ['IFRiverFlood']

import logging
import os
import numpy as np
import pandas as pd
from os import walk
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity import ImpactFuncSet

LOGGER = logging.getLogger(__name__)

DEF_VAR_EXCEL = {'sheet_name': 'damagefunctions',
                 'col_name': {'func_id' : 'DamageFunID',
                              'inten' : 'Intensity',
                              'mdd' : 'MDD',
                              'paa' : 'PAA',
                              'mdr' : 'MDR',
                              'name' : 'name',
                              'peril' : 'peril_ID',
                              'unit' : 'Intensity_unit'
                             }
                }


class IFRiverFlood(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self, region = None):
        ImpactFunc.__init__(self)
        self.haz_type = 'RF'
        self.continent = region

    def read_excel(self, if_func_dir, file_name = None, var_names=DEF_VAR_EXCEL):
        """Read excel file following template and store variables.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): descriptionfrom climada.entity import ImpactFuncSet of the data
            var_names (dict, optional): name of the variables in the file
        """
        if file_name is None:
            if os.path.exists(if_func_dir):
                file_path = self._set_path_from_area(if_func_dir)
            else:
                LOGGER.error('Impact function directory does not exist')
                raise NameError
        else:
            self.continent = file_name.split('_')[0]
            file_path = os.path.join(if_func_dir, file_name)

        if not os.path.exists(file_path):
            LOGGER.error('File path or directory do not meet\
                         standard format')
            raise NameError
        try:
            dfr = pd.read_excel(file_path, var_names['sheet_name'])
        except KeyError:
            LOGGER.error('Sheet name not in file')
            raise NameError from KeyError
        try:
            self._fill_if(dfr, var_names)
        except KeyError:
            LOGGER.error('Column names do not meet standard format')

    def _set_path_from_area(self, if_func_dir, sector='residential'):
        """Provides paths for selected models to incorporate flood depth
        and fraction
        Parameters:
           if_func_dir (string): path of impact function directory
           sector (string): impact category of damage function 
        raises:
            AttributeError
        Returns:
            string
        """

        if not sector == residential:
            LOGGER.error('Damage categories other than residential\
                          are not yet implemented')
            raise NotImplementedError
        if self.continent is None:
            LOGGER.error('Region for impact function unknown')
            raise AttributeError
        path = "{}_FL_JRCdamagefunc_{}_PAA1.xls".format(self.continent, sector)
        file_path = os.path.join(if_func_dir, path)
        return file_path

    def _fill_if(self, dfr, var_names):
        """Fills impact function attributes with information from file
        Parameters:
           dfr (xls file): File that contains 
        raises:
            KeyError
        """
        self.id = dfr[var_names['col_name']['func_id']].values[0]
        self.name = dfr[var_names['col_name']['name']].values[0]
        self.intensity_unit = dfr[var_names['col_name']['unit']].values[0]
        # Followng values defined for each intensity value
        self.intensity = dfr[var_names['col_name']['inten']].values
        self.mdd = dfr[var_names['col_name']['mdd']].values
        self.paa = dfr[var_names['col_name']['paa']].values

    def set_id(self, new_id):
        """Assigns new if id"""
        self.id = new_id

    @staticmethod
    def flood_imp_func_set(flood_if_dir):
        """Builds impact function set for river flood, using standard files
        raises:
            NameError
        Returns:
            ImpactFunctionSet
        """
        if_files = []
        if_set = ImpactFuncSet()
        if not os.path.exists(flood_if_dir):
            LOGGER.error('Impact function directory does not exist')
            raise NameError
        for (dirpath, dirnames, filenames) in walk(flood_if_dir):
            if_files.extend(filenames)
        for i in range(len(if_files)):
            ifrf = IFRiverFlood()
            ifrf.read_excel(flood_if_dir, if_files[i])
            ifrf.set_id(i + 1)
            if_set.append(ifrf)
        return if_set