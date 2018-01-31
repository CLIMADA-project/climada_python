"""
Define ExposuresMat class.
"""

__all__ = ['ExposuresMat']

import numpy as np

from climada.entity.exposures.base import Exposures
from climada.entity.tag import Tag
import climada.util.hdf5_handler as hdf5

class ExposuresMat(Exposures):
    """Exposures class loaded from a mat file produced by climada.

    Attributes
    ----------
        sup_field_name (str): name of the enclosing variable, if present
        field_name (str): name of variable containing the data
        var (dict): name of the variables in field_name
    """

    def __init__(self, file_name=None, description=None):
        """Extend Exposures __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> ExposuresMat()
            Initializes empty attributes.
            >>> ExposuresMat('filename')
            Loads data from the provided file.
            >>> ExposuresMat('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sup_field_name = 'entity'
        self.field_name = 'assets'
        self.var = {'lat' : 'lat',
                    'lon' : 'lon',
                    'val' : 'Value',
                    'ded' : 'Deductible',
                    'cov' : 'Cover',
                    'imp' : 'DamageFunID',
                    'cat' : 'Category_ID',
                    'reg' : 'Region_ID',
                    'uni' : 'Value_unit',
                    'ass' : 'centroid_index',
                    'ref' : 'reference_year'
                   }
        # Initialize
        Exposures.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
       # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # Load mat data
        expo = hdf5.read(file_name)
        try:
            expo = expo[self.sup_field_name]
        except KeyError:
            pass
        expo = expo[self.field_name]

        # Fill variables
        self._read_obligatory(expo)
        self._read_default(expo)
        self._read_optional(expo, file_name)

    def _read_obligatory(self, expo):
        """Fill obligatory variables."""
        self.value = expo[self.var['val']]. \
        reshape(len(expo[self.var['val']]),)

        coord_lat = expo[self.var['lat']]
        coord_lon = expo[self.var['lon']]
        self.coord = np.concatenate((coord_lat, coord_lon), axis=1)

        self.impact_id = expo[self.var['imp']]. \
        reshape(len(expo[self.var['imp']]),)
        self.impact_id = self.impact_id.astype(int)

        # set exposures id according to appearance order
        num_exp = len(self.value)
        self.id = np.linspace(self.id.size, self.id.size + \
                              num_exp - 1, num_exp, dtype=int)

    def _read_default(self, expo):
        """Fill optional variables. Set default values."""
        num_exp = len(expo[self.var['val']])
        # get the exposures deductibles as np.array float 64
        # if not provided set default zero values
        self.deductible = self._parse_default(expo, self.var['ded'], \
                                               np.zeros(num_exp))
        # get the exposures coverages as np.array float 64
        # if not provided set default exposure values
        self.cover = self._parse_default(expo, self.var['cov'], self.value)

    def _read_optional(self, expo, file_name):
        """Fill optional parameters."""
        self.ref_year = self._parse_optional(expo, self.ref_year, \
                                             self.var['ref'])
        if not isinstance(self.ref_year, int):
            self.ref_year = int(self.ref_year[0])

        self.category_id = self._parse_optional(expo, self.category_id, \
                                                self.var['cat'])
        self.category_id = self.category_id.astype(int)
        self.region_id = self._parse_optional(expo, self.region_id, \
                                                self.var['reg'])
        self.region_id = self.region_id.astype(int)
        self.assigned = self._parse_optional(expo, self.assigned, \
                                                self.var['ass'])
        self.assigned = self.assigned.astype(int)
        try:
            self.value_unit = hdf5.get_str_from_ref(file_name, \
                                                  expo[self.var['uni']][0][0])
        except KeyError:
            pass

    @staticmethod
    def _parse_optional(hdf, var, var_name):
        """Retrieve optional variable, leave its original value if fail."""
        try:
            var = hdf[var_name].reshape(len(hdf[var_name]),)
        except KeyError:
            pass
        return var

    @staticmethod
    def _parse_default(hdf, var_name, def_val):
        """Retrieve optional variable, set default value if fail."""
        try:
            res = hdf[var_name].reshape(len(hdf[var_name]),)
        except KeyError:
            res = def_val
        return res
