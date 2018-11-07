"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define functions to handle dates nad times in climada
"""
import logging
import datetime as dt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

def date_to_str(date):
    """ Compute date string in ISO format from input datetime ordinal int.

    Parameters:
        date (int): input datetime ordinal

    Returns:
        str
    """
    return dt.date.fromordinal(date).isoformat()

def str_to_date(date):
    """ Compute datetime ordinal int from input date string in ISO format.

    Parameters:
        date (str): idate string in ISO format, e.g. '2018-04-06'

    Returns:
        int
    """
    year, mounth, day = (int(val) for val in date.split('-'))
    return dt.date(year, mounth, day).toordinal()

def datetime64_to_ordinal(datetime):
    """ Converts from a numpy datetime64 object to an ordinal date.
    See https://stackoverflow.com/a/21916253 for the horrible details.

    Parameters:
        datetime (np.datetime64): date and time

    Returns:
        int
    """
    return pd.to_datetime(datetime.tolist()).toordinal()

def last_year(ordinal_vector):
    """ Extract first year from ordinal date """
    return dt.date.fromordinal(np.max(ordinal_vector)).year

def first_year(ordinal_vector):
    """ Extract first year from ordinal date """
    return dt.date.fromordinal(np.min(ordinal_vector)).year
