#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 19:21:42 2020

@author: ckropf
"""

import math
import numpy as np
import logging

LOGGER = logging.getLogger(__name__)

ABBREV = ['', 'K', 'M', 'Bn', 'Tn']

def sig_dig(x, n_sig_dig = 16):
    """
    Rounds x to n_sig_dig number of significant digits. 
    Examples: 1.234567 -> 1.2346, 123456.89 -> 123460.0

    Parameters
    ----------
    x : float
        number to be rounded
    n_sig_dig : int, optional
        Number of significant digits. The default is 16.

    Returns
    -------
    float
        Rounded number

    """
    num_of_digits = len(str(x).replace(".", ""))
    if n_sig_dig >= num_of_digits:
        return x
    n = math.floor(math.log10(abs(x)) + 1 - n_sig_dig)
    result = np.round(x * 10**(-n)) * 10**n
    return result

def sig_dig_list(iterable, n_sig_dig=16):
    """
    Vectorized form of sig_dig. Rounds a list of float to a number
    of significant digits

    Parameters
    ----------
    iterable : iter(float)
        iterable of numbers to be rounded
    n_sig_dig : int, optional
        Number of significant digits. The default is 16.


    Returns
    -------
    list
        list of rounded floats

    """
    return np.vectorize(sig_dig)(iterable, n_sig_dig)

def money_words(values, n_digits=None, abbreviations=None):
    """
    Converts values to closest common monetary unit (K, M Bn, Tn, ...)

    Parameters
    ----------
    values : list(float) or np.ndarray
        Values to be converted
    n_digits : int, optional
        Number of significant digits to return. The default is all digits
        are returned.
    abbreviations: string, optional
        Name of the abbreviations for the money 1000s counts
        (e.g., 'k', 'm', 'bn'). Default is ABBREV.

    Returns
    -------
    TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.

    """
    
    if abbreviations is None:
        abbreviations= ABBREV
    
    exponents = [math.log10(abs(val)) for val in values]
    max_exp = max(exponents)
    min_exp = min(exponents)
    
    avg_exp = math.floor((max_exp + min_exp) / 2) #rounded down
    
    name = ''
    try:
        idx = int(avg_exp/3)
        name = abbreviations[idx] 
        if idx == 0:
            mon_val = np.array(values),
            return (mon_val, name)
    except IndexError: 
        LOGGER.warning(f"The numbers are larger than 1000{abbreviations[-1]}")
        name = abbreviations[-1]
        largest_exp = (len(abbreviations)-1) * 3
        mon_val = values / 10**largest_exp
        return (mon_val , name)   
    mon_val = np.array(values) / 10**avg_exp
    
    return (mon_val, name)
    
