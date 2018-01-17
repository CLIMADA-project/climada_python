"""
Define ExposuresExcel class.
"""

from xlrd import XLRDError
import numpy as np
import pandas

from climada.entity.exposures.base import Exposures
from climada.entity.tag import Tag
from climada.util.config import config

class ExposuresExcel(Exposures):
    """Exposures class loaded from an excel file.

    Attributes
    ----------
        sheet_name (str): name of excel sheet containing the data
        col_names (dict): name of the table columns for each of the attributes
    """

    def __init__(self, file_name=None, description=None):
        """Extend Exposures __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> ExposuresExcel()
            Initializes empty attributes.
            >>> ExposuresExcel('filename')
            Loads data from the provided file.
            >>> ExposuresExcel('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sheet_name = {'exp': 'assets',
                           'name': 'names'
                          }
        self.col_names = {'lat' : 'Latitude',
                          'lon' : 'Longitude',
                          'val' : 'Value',
                          'ded' : 'Deductible',
                          'cov' : 'Cover',
                          'imp' : 'DamageFunID',
                          'cat' : 'Category_ID',
                          'reg' : 'Region_ID',
                          'uni' : 'Value unit',
                          'ass' : 'centroid_index',
                          'ref': 'reference_year',
                          'item' : 'Item'
                         }
        # Initialize
        Exposures.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name['exp'])
        # get variables
        self._read_obligatory(dfr)
        self._read_default(dfr)
        self._read_optional(dfr, file_name)

    def _read_obligatory(self, dfr):
        """Fill obligatory variables."""
        self.value = dfr[self.col_names['val']].values

        coord_cols = [self.col_names['lat'], self.col_names['lon']]
        self.coord = np.array(dfr[coord_cols])

        self.impact_id = dfr[self.col_names['imp']].values

        # set exposures id according to appearance order
        num_exp = len(dfr.index)
        self.id = np.linspace(self.id.size, self.id.size + \
                              num_exp - 1, num_exp, dtype=int)

    def _read_default(self, dfr):
        """Fill optional variables. Set default values."""
        # get the exposures deductibles as np.array float 64
        # if not provided set default zero values
        num_exp = len(dfr.index)
        self.deductible = self._parse_default(dfr, self.col_names['ded'], \
                                              np.zeros(num_exp))
        # get the exposures coverages as np.array float 64
        # if not provided set default exposure values
        self.cover = self._parse_default(dfr, self.col_names['cov'], \
                                         self.value)

    def _read_optional(self, dfr, file_name):
        """Fill optional parameters."""
        self.category_id = self._parse_optional(dfr, self.category_id, \
                                                self.col_names['cat'])
        self.region_id = self._parse_optional(dfr, self.region_id, \
                                              self.col_names['reg'])
        self.value_unit = self._parse_optional(dfr, self.value_unit, \
                                               self.col_names['uni'])
        if not isinstance(self.value_unit, str):
            # Check all exposures have the same unit
            if len(np.unique(self.value_unit)) is not 1:
                raise ValueError('Different value units provided for \
                                 exposures.')
            self.value_unit = self.value_unit[0]
        self.assigned = self._parse_optional(dfr, self.assigned, \
                                             self.col_names['ass'])

        # check if reference year given under "names" sheet
        # if not, set default present reference year
        self.ref_year = self._parse_ref_year(file_name)

    def _parse_ref_year(self, file_name):
        """Retrieve reference year provided in the other sheet, if given."""
        try:
            dfr = pandas.read_excel(file_name, self.sheet_name['name'])
            dfr.index = dfr[self.col_names['item']]
            ref_year = dfr.loc[self.col_names['ref']]['name']
        except (XLRDError, KeyError):
            ref_year = config['present_ref_year']
        return ref_year

    @staticmethod
    def _parse_optional(dfr, var, var_name):
        """Retrieve optional variable, leave its original value if fail."""
        try:
            var = dfr[var_name].values
        except KeyError:
            pass
        return var

    @staticmethod
    def _parse_default(dfr, var_name, def_val):
        """Retrieve optional variable, set default value if fail."""
        try:
            res = dfr[var_name].values
        except KeyError:
            res = def_val
        return res
