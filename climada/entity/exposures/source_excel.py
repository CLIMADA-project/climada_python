"""
Define ExposuresExcel class.
"""

import warnings
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
        self.sheet_name = 'assets'
        self.col_names = {'lat' : 'Latitude',
                          'lon' : 'Longitude',
                          'value' : 'Value',
                          'ded' : 'Deductible',
                          'cov' : 'Cover',
                          'imp' : 'DamageFunID'
                         }
        # Initialize
        Exposures.__init__(self, file_name, description)

    def _read(self, file_name, description=None):
        """Override _read Exposures method."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # number of exposures
        num_exp = len(dfr.index)
        # get the exposures values as np.array float 64
        self.value = dfr[self.col_names['value']].values

        # get the exposures coordinates as 2dim np.array float 64
        coord_cols = [self.col_names['lat'], self.col_names['lon']]
        self.coord = np.array(dfr[coord_cols])

        # get the exposures deductibles as np.array float 64
        # if not provided set default zero values
        try:
            self.deductible = dfr[self.col_names['ded']].values
        except KeyError:
            self.deductible = np.zeros(len(self.value))
            warnings.warn('Column ' + self.col_names['ded'] + ' not found. '\
                          'Default zero values set for deductible.')

        # get the exposures coverages as np.array float 64
        # if not provided set default exposure values
        try:
            self.cover = dfr[self.col_names['cov']].values
        except KeyError:
            self.cover = self.value
            warnings.warn('Column ' + self.col_names['cov'] + ' not found. ' \
              'Cover set to exposures values.')

        # get the exposures impact function id as np.array int64
        self.impact_id = dfr[self.col_names['imp']].values

        # set exposures id according to appearance order
        self.id = np.linspace(self.id.size, self.id.size +
                              num_exp - 1, num_exp, dtype=int)

        # set default present reference year
        self.ref_year = config["present_ref_year"]

        # set dummy category and region id
        self.category_id = np.zeros(num_exp)
        self.region_id = np.zeros(num_exp)
