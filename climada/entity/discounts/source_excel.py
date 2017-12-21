"""
Define DiscountsExcel class.
"""

import pandas

from climada.entity.discounts.base import Discounts
from climada.entity.tag import Tag

class DiscountsExcel(Discounts):
    """Contains discount rates loaded from an excel file.
    
    Attributes
    ----------
        sheet_name (str): name of excel sheet containing the data
        col_names (dict): name of the table columns for each of the attributes
    """

    def __init__(self, file_name=None, description=None):
        """ Fill values from file, if provided.        

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> DiscountsExcel()
            Initializes empty attributes.
            >>> DiscountsExcel('filename')
            Loads data from the provided file.
            >>> DiscountsExcel('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sheet_name = 'discount'
        self.col_names = {'year' : 'year',
                          'disc' : 'discount_rate'
                         }
        # Initialize
        Discounts.__init__(self, file_name, description)

    def _read(self, file_name, description=None):
        """Read data from input file and stores input description."""
        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # get the discount rates years
        self.years = dfr[self.col_names['year']].values

        # get the discount rates for each year
        self.rates = dfr[self.col_names['disc']].values
