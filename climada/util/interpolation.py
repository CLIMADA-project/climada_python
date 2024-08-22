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

Define interpolation and extrapolation functions for calculating (local) exceedance frequencies and return periods
"""


import logging

import numpy as np
from scipy import interpolate

from climada.util.value_representation import sig_dig_list

LOGGER = logging.getLogger(__name__)

def interpolate_ev(
        x_test,
        x_train,
        y_train,
        logx = False,
        logy = False,
        x_threshold = None,
        y_threshold = None,
        extrapolation = False,
        y_asymptotic = np.nan
    ):
    """
    Util function to interpolate (and extrapolate) training data (x_train, y_train)
    to new points x_test with several options (log scale, thresholds)

    Parameters:
    -------
        x_test : array_like
            1-D array of x-values for which training data should be interpolated
        x_train : array_like
            1-D array of x-values of training data
        y_train : array_like
            1-D array of y-values of training data
        logx : bool, optional
            If set to True, x_values are converted to log scale. Defaults to False.
        logy : bool, optional
            If set to True, y_values are converted to log scale. Defaults to False.
        x_threshold : float, optional
            Lower threshold to filter x_train. Defaults to None.
        y_threshold : float, optional
            Lower threshold to filter y_train. Defaults to None.
        extrapolation : bool, optional
            If set to True, values will be extrapolated. If set to False, x_test values
            smaller than x_train will be assigned y_train[0] (x_train must be sorted in
            ascending order), and x_test values larger than x_train will be assigned
            y_asymptotic. Defaults to False
        y_asymptotic : float, optional
            Return value and if extrapolation is True or x_train.size < 2, for x_test
            values larger than x_train. Defaults to np.nan.

    Returns
    -------
    np.array
        interpolated values y_test for the test points x_test
    """

    # preprocess interpolation data
    x_test, x_train, y_train = _preprocess_interpolation_data(
        x_test, x_train, y_train, logx, logy, x_threshold, y_threshold
    )

      # handle case of small training data sizes
    if x_train.size < 2:
        LOGGER.warning('Data is being extrapolated.')
        return _interpolate_small_input(x_test, x_train, y_train, logy, y_asymptotic)

    # calculate fill values
    if extrapolation:
        fill_value = 'extrapolate'
        if np.min(x_test) < np.min(x_train) or np.max(x_test) > np.max(x_train):
            LOGGER.warning('Data is being extrapolated.')
    else:
        if not all(sorted(x_train) == x_train):
            raise ValueError('x_train array must be sorted in ascending order.')
        fill_value = (y_train[0], np.log10(y_asymptotic) if logy else y_asymptotic) 

    interpolation = interpolate.interp1d(
        x_train, y_train, fill_value=fill_value, bounds_error=False)
    y_test = interpolation(x_test)

    # adapt output scale
    if logy:
        y_test = np.power(10., y_test)
    return y_test

def stepfunction_ev(
        x_test,
        x_train,
        y_train,
        x_threshold = None,
        y_threshold = None,
        y_asymptotic = np.nan
    ):
    """
    Util function to interpolate and extrapolate training data (x_train, y_train)
    to new points x_test using a step function

    Parameters:
    -------
        x_test : array_like
            1-D array of x-values for which training data should be interpolated
        x_train : array_like
            1-D array of x-values of training data
        y_train : array_like
            1-D array of y-values of training data
        x_threshold : float, optional
            Lower threshold to filter x_train. Defaults to None.
        y_threshold : float, optional
            Lower threshold to filter y_train. Defaults to None.
        y_asymptotic : float, optional
            Return value if x_test > x_train. Defaults to np.nan.

    Returns
    -------
    np.array
        interpolated values y_test for the test points x_test
    """

    # preprocess interpolation data
    x_test, x_train, y_train = _preprocess_interpolation_data(
        x_test, x_train, y_train, None, None, x_threshold, y_threshold
    )

    # handle case of small training data sizes
    if x_train.size < 2:
        return _interpolate_small_input(x_test, x_train, y_train, None, y_asymptotic)

    # find indices of x_test if sorted into x_train
    if not all(sorted(x_train) == x_train):
        raise ValueError('Input array x_train must be sorted in ascending order.')
    indx = np.searchsorted(x_train, x_test)
    y_test = y_train[indx.clip(max = len(x_train) - 1)]
    y_test[indx == len(x_train)] = y_asymptotic

    return y_test

def _preprocess_interpolation_data(
        x_test,
        x_train,
        y_train,
        logx,
        logy,
        x_threshold,
        y_threshold
    ):
    """
    helper function to preprocess interpolation training and test data by filtering data below
    thresholds and converting to log scale if required
    """

    if x_train.shape != y_train.shape:
        raise ValueError(f'Incompatible shapes of input data, x_train {x_train.shape} '
                         f'and y_train {y_train.shape}. Should be the same')

    # transform input to float arrays
    x_test, x_train, y_train = (np.array(x_test).astype(float),
                                np.array(x_train).astype(float),
                                np.array(y_train).astype(float))

    # cut x and y above threshold
    if x_threshold or x_threshold==0:
        x_th = np.asarray(x_train > x_threshold).squeeze()
        x_train = x_train[x_th]
        y_train = y_train[x_th]

    if y_threshold or y_threshold==0:
        y_th = np.asarray(y_train > y_threshold).squeeze()
        x_train = x_train[y_th]
        y_train = y_train[y_th]

    # convert to log scale
    if logx:
        x_train, x_test = np.log10(x_train), np.log10(x_test)
    if logy:
        y_train = np.log10(y_train)

    return (x_test, x_train, y_train)

def _interpolate_small_input(x_test, x_train, y_train, logy, y_asymptotic):
    """
    helper function to handle if interpolation data is small (empty or one point)
    """
    # return y_asymptotic if x_train and y_train empty
    if x_train.size == 0:
        return np.full_like(x_test, y_asymptotic)

    # reconvert logarithmic y_train to original y_train
    if logy:
        y_train = np.power(10., y_train)

    # if only one (x_train, y_train), return stepfunction with
    # y_train if x_test < x_train and y_asymtotic if x_test > x_train
    y_test = np.full_like(x_test, y_train[0])
    y_test[np.squeeze(x_test) > np.squeeze(x_train)] = y_asymptotic
    return y_test

def group_frequency(frequency, value, n_sig_dig=2):
    """
    Util function to aggregate (add) frequencies for equal values

    Parameters:
    ------
        frequency : array_like
            Frequency array
        value : array_like
            Value array in ascending order
        n_sig_dig : int
            number of significant digits for value when grouping frequency.
            Defaults to 2.

    Returns:
    ------
        tuple
            (frequency array after aggregation,
            unique value array in ascending order)
    """
    frequency, value = np.array(frequency), np.array(value)
    if frequency.size == 0 and value.size == 0:
        return ([], [])

    if len(value) != len(np.unique(sig_dig_list(value, n_sig_dig=n_sig_dig))):
        #check ordering of value
        if not all(sorted(value) == value):
            raise ValueError('Value array must be sorted in ascending order.')
        # add frequency for equal value
        value, start_indices = np.unique(
            sig_dig_list(value, n_sig_dig=n_sig_dig), return_index=True)
        start_indices = np.insert(start_indices, len(value), len(frequency))
        frequency = np.array([
            sum(frequency[start_indices[i]:start_indices[i+1]])
            for i in range(len(value))
        ])
    return frequency, value
