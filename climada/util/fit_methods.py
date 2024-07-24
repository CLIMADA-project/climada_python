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

define fit functionalities for (local) exceedance frequencies and return periods
"""


import numpy as np
from scipy import interpolate
import logging

from climada.util.value_representation import sig_dig_list

LOGGER = logging.getLogger(__name__)

def interpolate_ev(
        x_test, 
        x_train: np.array, 
        y_train: np.array, 
        method: str=None, 
        x_scale: str=None, 
        y_scale: str=None, 
        x_threshold: float = None, 
        y_threshold: float = None,
        y_asymptotic = np.nan,
        **kwargs
    ):
    """_summary_

    Args:
        x_test (_type_): x values (1-D array of x values to fit)
        x_train (np.array): x values (1-D array of x values to fit)
        y_train (np.array): y values (1-D array of x values to fit)
        method (str, optional): _description_. Defaults to None.
        x_scale (str, optional): _description_. Defaults to None.
        y_scale (str, optional): _description_. Defaults to None.
        x_threshold (float, optional): _description_. Defaults to None.
        y_threshold (float, optional): _description_. Defaults to None.
        y_asymptotic (float, optional): return value if x_test > x_train if 
            method is stepfunction or x_train.size < 2. Defaults to np.nan.
        **kwargs: further optional parameters for the fit methods

    Returns
    -------
    np.array
    """
    
    # check if inputs are valid
    if not method:
        method = 'interpolate'
    if not method in ['interpolate', 'stepfunction']:
        raise ValueError(f'Unknown method: {method}. Use "interpolate" or "stepfunction" instead')
    if method == 'stepfunction': # x_scale and y_scale unnecessary if fitting stepfunction
        x_scale, y_scale = None, None
    if x_train.shape != y_train.shape:
        raise ValueError(f'Incompatible shapes of input data, x_train {x_train.shape} and y_train {y_train.shape}. Should be the same')

    # transform input to float arrays
    x_test, x_train, y_train = np.array(x_test).astype(float), np.array(x_train).astype(float), np.array(y_train).astype(float)

    # cut x and y above threshold
    if x_threshold or x_threshold==0:
        x_th = np.asarray(x_train > x_threshold).squeeze()
        x_train = x_train[x_th]
        y_train = y_train[x_th]
    
    if y_threshold or y_threshold==0:
        y_th = np.asarray(y_train > y_threshold).squeeze()
        x_train = x_train[y_th]
        y_train = y_train[y_th]

    # return y_asymptotic if x_train and y_train empty
    if x_train.size == 0:
        return np.full_like(x_test, y_asymptotic)
    # if only one (x_train, y_train), return stepfunction with
    # y_train if x_test < x_train and y_asymtotic if x_test > x_train
    if x_train.size == 1:
        y_test = np.full_like(x_test, y_train[0])
        y_test[np.squeeze(x_test) > np.squeeze(x_train)] = y_asymptotic
        return y_test

    # adapt x and y scale
    if x_scale == 'log':
        x_train, x_test = np.log10(x_train), np.log10(x_test)
    if y_scale == 'log':
        y_train = np.log10(y_train)

    # calculate interpolation
    if method == 'interpolate':
        # warn if data is being extrapolated
        if (
            (('fill_value', 'extrapolate') in kwargs.items()) and 
            ((np.min(x_test) < np.min(x_train)) or (np.max(x_test) > np.max(x_train)))):
            LOGGER.warning('Data is being extrapolated.')
        # calculate fill values 
        elif 'fill_value' in kwargs.keys():
            if len(kwargs['fill_value']) == 2:
                if kwargs['fill_value'][0] == 'maximum':
                    kwargs['fill_value'] = (np.max(y_train), kwargs['fill_value'][1])
        interpolation = interpolate.interp1d(x_train, y_train, **kwargs)
        y_test = interpolation(x_test)
    
    # calculate stepfunction fit
    elif method == 'stepfunction':
        # find indeces of x_test if sorted into x_train
        if not all(sorted(x_train) == x_train):
            raise ValueError(f'Input array x_train must be sorted in ascending order.')
        indx = np.searchsorted(x_train, x_test)
        y_test = y_train[indx.clip(max = len(x_train) - 1)]
        y_test[indx == len(x_train)] = y_asymptotic

    # adapt output scale
    if y_scale == 'log':
        y_test = np.power(10., y_test)
    return y_test
    

def group_frequency(frequency, value, n_sig_dig=2):
    """util function to add frequencies for equal values

    Args:
        frequency (np.array): frequency corresponding to the values 
        value (np.array): value sorted in decreasing order
        n_sig_dig (int): number of significant digits for value when grouping frequency

    Returns:
        tuple: (frequency after aggregation, 
                unique value in cedreasing order)
    """
    frequency, value = np.array(frequency), np.array(value)
    if frequency.size == 0 and value.size == 0:
        return ([], [])
    
    if len(value) != len(np.unique(sig_dig_list(value, n_sig_dig=n_sig_dig))):
        #check ordering of value
        if not all(sorted(value) == value):
            raise ValueError(f'Value array must be sorted in ascending order.')
        # add frequency for equal value
        value, start_indices = np.unique(sig_dig_list(value, n_sig_dig=n_sig_dig), return_index=True)
        start_indices = np.insert(start_indices, len(value), len(frequency))
        frequency = np.array([
            sum(frequency[start_indices[i]:start_indices[i+1]])
            for i in range(len(value))
        ])
        return frequency, value
    return frequency, value
