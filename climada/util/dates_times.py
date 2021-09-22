"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define functions to handle dates and times in climada
"""
import logging
import datetime as dt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

def date_to_str(date):
    """Compute date string in ISO format from input datetime ordinal int.
    Parameters
    ----------
    date : int or list or np.array
        input datetime ordinal

    Returns
    -------
    str or list(str)
    """
    try:
        date_int = int(date)
        return dt.date.fromordinal(date_int).isoformat()
    except TypeError:
        return [dt.date.fromordinal(i_date).isoformat() for i_date in date]


def str_to_date(date):
    """Compute datetime ordinal int from input date string in ISO format.
    Parameters
    ----------
    date : str or list
        idate string in ISO format, e.g. '2018-04-06'

    Returns
    -------
    int
    """
    if isinstance(date, str):
        year, mounth, day = (int(val) for val in date.split('-'))
        return dt.date(year, mounth, day).toordinal()

    all_date = []
    for i_date in date:
        year, mounth, day = (int(val) for val in i_date.split('-'))
        all_date.append(dt.date(year, mounth, day).toordinal())
    return all_date

def datetime64_to_ordinal(datetime):
    """Converts from a numpy datetime64 object to an ordinal date.
    See https://stackoverflow.com/a/21916253 for the horrible details.
    Parameters
    ----------
    datetime : np.datetime64, or list or np.array
        date and time

    Returns
    -------
    int
    """
    if isinstance(datetime, np.datetime64):
        return pd.to_datetime(datetime.tolist()).toordinal()

    return [pd.to_datetime(i_dt.tolist()).toordinal() for i_dt in datetime]

def last_year(ordinal_vector):
    """Extract first year from ordinal date

    Parameters
    ----------
    ordinal_vector : list or np.array
        input datetime ordinal

    Returns
    -------
    int
    """
    return dt.date.fromordinal(np.max(ordinal_vector)).year

def first_year(ordinal_vector):
    """Extract first year from ordinal date

    Parameters
    ----------
    ordinal_vector : list or np.array
        input datetime ordinal

    Returns
    -------
    int
    """
    return dt.date.fromordinal(np.min(ordinal_vector)).year
