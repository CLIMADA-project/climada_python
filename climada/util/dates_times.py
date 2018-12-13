"""
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
    if isinstance(datetime, np.datetime64):
        return pd.to_datetime(datetime.tolist()).toordinal()
    else:
        return np.array([date.toordinal() for date in pd.to_datetime(datetime)])
