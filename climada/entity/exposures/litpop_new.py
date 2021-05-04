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

Define LitPop class.
"""

import numpy as np

import climada.util.coordinates as u_coord

def gridpoints_core_calc(data_arrays, offsets=None, exponents=None,
                         total_val_rescale=None):
    """
    Combines N dense numerical arrays by point-wise multipilcation and
    optionally rescales to new total value:
    (1) An offset (1 number per array) is added to all elements in
        the corresponding data array in data_arrays (optional).
    (2) Numbers in each array are taken to the power of the corresponding
        exponent (optional).
    (3) Arrays are multiplied element-wise.
    (4) if total_val_rescale is provided,
        results are normalized and re-scaled with total_val_rescale.
    (5) One array with results is returned.

    Parameters
    ----------
    data_arrays : list or array of numpy arrays containing numbers
        Data to be combined, i.e. list containing N (min. 1) arrays of same shape.
    total_val_rescale : float or int (optional)
        Total value for optional rescaling of resulting array. All values in result_array
        are skaled so that the sum is equal to total_val_rescale.
        The default (None) implies no rescaling (None).
    offsets: list or array containing N numbers >= 0 (optional)
        One numerical offset per array that is added (sum) to the
        corresponding array in data_arrays.
        The default (None) corresponds to np.zeros(N).
    exponents: list or array containing N numbers >= 0 (optional)
        One exponent per array used as power for the corresponding array.
        The default (None) corresponds to np.ones(N).

    Raises
        ------
        ValueError
            If input lists don't have the same number of elements.
            Or: If arrays in data_arrays do not have the same shape.

    Returns
    -------
    result_array : np.array of same shape as arrays in data_arrays
        Results from calculation described above.

    """
    # convert input data to arrays if proivided as lists
    # check integrity of data_array input (length and type):
    try:
        if isinstance(data_arrays[0], list):
            data_arrays[0] = np.array(data_arrays[0])
        for i_arr in np.arange(len(data_arrays)-1):
            if isinstance(data_arrays[i_arr+1], list):
                data_arrays[i_arr+1] = np.array(data_arrays[i_arr+1])
            if data_arrays[i_arr].shape != data_arrays[i_arr+1].shape:
                raise ValueError("Elements in data_arrays don't agree in shape.")
                
    except AttributeError as err:
            raise TypeError("data_arrays or contained elements have wrong type.") from err

    # if None, defaults for offsets and exponents are set:
    if offsets is None:
        offsets = np.zeros(len(data_arrays))
    if exponents is None:
        exponents = np.ones(len(data_arrays))
    if np.min(offsets) < 0:
        raise ValueError("offset values < 0 not are allowed.")
    if np.min(exponents) < 0:
        raise ValueError("exponents < 0 not are allowed.")
    # Steps 1-3: arrays are multiplied after application of offets and exponents:
    #       (arrays are converted to float to prevent ValueError:
    #       "Integers to negative integer powers are not allowed.")
    result_array = np.power(np.array(data_arrays[0]+offsets[0], dtype=float),
                            exponents[0])
    for i in np.arange(1, len(data_arrays)):
        result_array = np.multiply(result_array,
                                   np.power(np.array(data_arrays[i]+offsets[i], dtype=float),
                                            exponents[i])
                                   )

    # Steps 4+5: if total value for rescaling is provided, result_array is normalized and
    # scaled with this total value (total_val_rescale):
    if isinstance(total_val_rescale, float) or isinstance(total_val_rescale, int):
        return np.divide(result_array, result_array.sum()) * total_val_rescale
    elif total_val_rescale is not None:
        raise TypeError("total_val_rescale must be int or float.")

    return result_array