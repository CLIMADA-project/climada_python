"""
Define DiscountsMat class.
"""

__all__ = ['DiscountsMat']

import climada.util.hdf5_handler as hdf5
from climada.entity.discounts.base import Discounts
from climada.entity.tag import Tag

class DiscountsMat(Discounts):
    """Discounts class loaded from an excel file.

    Attributes
    ----------
        sup_field_name (str): name of the enclosing variable, if present
        field_name (str): name of variable containing the data
        var (dict): name of the variables in field_name
    """

    def __init__(self, file_name=None, description=None):
        """Extend Discounts __init__ method.

        Parameters
        ----------
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Examples
        --------
            >>> DiscountsMat()
            Initializes empty attributes.
            >>> DiscountsMat('filename')
            Loads data from the provided file.
            >>> DiscountsMat('filename', 'description of file')
            Loads data from the provided file and stores provided description.
        """
        self.sup_field_name = 'entity'
        self.field_name = 'discount'
        self.var = {'year' : 'year',
                    'disc' : 'discount_rate'
                   }
        # Initialize
        Discounts.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
       # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # Load mat data
        disc = hdf5.read(file_name)
        try:
            disc = disc[self.sup_field_name]
        except KeyError:
            pass
        disc = disc[self.field_name]

        # get the discount rates years
        self.years = disc[self.var['year']].\
        reshape(disc[self.var['year']].shape[1],)
        self.years = self.years.astype(int)

        # get the discount rates for each year
        self.rates = disc[self.var['disc']].\
        reshape(disc[self.var['disc']].shape[1],)
