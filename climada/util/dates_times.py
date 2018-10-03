"""
Define functions to handle dates nad times in climada
"""
import datetime as dt

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
