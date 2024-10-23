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

Created on Mon Nov 16 19:21:42 2020

@author: ckropf
"""

import decimal
import logging
import math

import numpy as np

LOGGER = logging.getLogger(__name__)

ABBREV = {1: "", 1000: "K", 1000000: "M", 1000000000: "Bn", 1000000000000: "Tn"}


def sig_dig(x, n_sig_dig=16):
    """
    Rounds x to n_sig_dig number of significant digits.
    0, inf, Nan are returned unchanged.

    Examples
    --------
        with n_sig_dig = 5:

        1.234567 -> 1.2346, 123456.89 -> 123460.0

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
    result = decimal.Decimal(str(np.round(x * 10 ** (-n)))) * decimal.Decimal(
        str(10**n)
    )
    return float(result)


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


def convert_monetary_value(values, abbrev, n_sig_dig=None):
    if isinstance(values, (int, float)):
        values = [values]

    thsder = list(ABBREV.keys())[list(ABBREV.values()).index(abbrev)]
    mon_val = np.array(values) / thsder
    if n_sig_dig is not None:
        mon_val = [sig_dig(val, n_sig_dig=n_sig_dig) for val in mon_val]

    return mon_val


def value_to_monetary_unit(values, n_sig_dig=None, abbreviations=None):
    """Converts list of values to closest common monetary unit.

    0, Nan and inf have not unit.

    Parameters
    ----------
    values : int or float, list(int or float) or np.ndarray(int or float)
        Values to be converted
    n_sig_dig : int, optional
        Number of significant digits to return.

        Examples: n_sig_di=5: 1.234567 -> 1.2346, 123456.89 -> 123460.0

        Default: all digits are returned.
    abbreviations: dict, optional
        Name of the abbreviations for the money 1000s counts

        Default:
        {
        1:'',
        1000: 'K',
        1000000: 'M',
        1000000000: 'Bn',
        1000000000000: 'Tn'
        }

    Returns
    -------
    mon_val : np.ndarray
        Array of values in monetary unit
    name : string
        Monetary unit

    Examples
    --------
    values = [1e6, 2*1e6, 4.5*1e7, 0, Nan, inf] ->
        [1, 2, 4.5, 0, Nan, inf]
        ['M']

    """
    if isinstance(values, (int, float)):
        values = [values]

    if abbreviations is None:
        abbreviations = ABBREV

    exponents = []
    for val in values:
        if math.isclose(val, 0) or not math.isfinite(val):
            continue
        exponents.append(math.log10(abs(val)))
    if not exponents:
        exponents = [0]
    max_exp = max(exponents)
    min_exp = min(exponents)

    avg_exp = math.floor((max_exp + min_exp) / 2)  # rounded down
    mil_exp = 3 * math.floor(avg_exp / 3)

    thsder = int(10**mil_exp)  # Remove negative exponents
    thsder = 1 if thsder < 1 else thsder

    try:
        name = abbreviations[thsder]
    except KeyError:
        LOGGER.warning(
            "Warning: The numbers are larger than %s", list(abbreviations.keys())[-1]
        )
        thsder, name = list(abbreviations.items())[-1]

    mon_val = np.array(values) / thsder

    if n_sig_dig is not None:
        mon_val = [sig_dig(val, n_sig_dig=n_sig_dig) for val in mon_val]

    return (mon_val, name)


def safe_divide(numerator, denominator, replace_with=np.nan):
    """
    Safely divide two arrays or scalars.

    This function handles division by zero and NaN values in the numerator or
    denominator on an element-wise basis. If the division results in infinity, NaN, or
    division by zero in any element, that particular result is replaced by the specified
    value.

    Parameters
    ----------
    numerator : np.ndarray or scalar
        The numerator for division.
    denominator : np.ndarray or scalar
        The denominator for division. Division by zero and NaN values are handled
        safely.
    replace_with : float, optional
        The value to use in place of division results that are infinity, NaN, or
        division by zero. By default, it is NaN.

    Returns
    -------
    np.ndarray or scalar
        The result of the division. If the division results in infinity, NaN, or
        division by zero in any element, it returns the value specified in
        ``replace_with`` for those elements.

    Notes
    -----
    The function uses numpy's ``true_divide`` for array-like inputs and handles both
    scalar and array-like inputs for the numerator and denominator. NaN values or
    division by zero in any element of the input will result in the `replace_with` value
    in the corresponding element of the output.

    Examples
    --------
    >>> safe_divide(1, 0)
    nan

    >>> safe_divide(1, 0, replace_with=0)
    0

    >>> safe_divide([1, 0, 3], [0, 0, 3])
    array([nan, nan,  1.])

    >>> safe_divide([4, 4], [1, 0])
    array([4., nan])

    >>> safe_divide([4, 4], [1, nan])
    array([ 4., nan])
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)

        # Check if the result is a scalar
        if np.isscalar(result):
            if not np.isfinite(result):
                return replace_with
        else:
            # Replace infinities, NaNs, and division by zeros in np.ndarray
            result[~np.isfinite(result)] = replace_with

    return result
