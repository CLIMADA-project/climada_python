"""
=====================
source_excel module
=====================

Define DiscountsExcel class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  1 16:46:01 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import pandas

from climada.entity.discounts.base import Discounts
from climada.entity.tag import Tag

class DiscountsExcel(Discounts):
    """Class that loads the discount rates from an excel file"""

    def __init__(self, file_name=None, description=None):
        """ Define the name of the sheet nad columns names where the exposures
        are defined"""
        Discounts.__init__(self)
        # Define tha name of the sheet that is read
        self.sheet_name = 'discount'
        # Define the names of the columns that are read
        self.col_names = {'year' : 'year',
                          'disc' : 'discount_rate'
                         }

        # Initialize
        Discounts.__init__(self, file_name, description)

    def read(self, file_name, description=None):
        """ Virtual class. Needs to be defined for each child"""

        # append the file name and description into the instance class
        self.tag = Tag(file_name, description)

        # load Excel data
        dfr = pandas.read_excel(file_name, self.sheet_name)

        # get the discount rates years
        self.years = dfr[self.col_names['year']].values

        # get the discount rates for each year
        self.rates = dfr[self.col_names['disc']].values
