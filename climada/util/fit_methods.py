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

def calc_fit_interp(
        x_test, 
        x_train: np.array, 
        y_train: np.array, 
        method: str=None, 
        x_scale: str=None, 
        y_scale: str=None, 
        x_thres: float = None, 
        y_thres: float = None,
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
        x_thres (float, optional): _description_. Defaults to None.
        y_thres (float, optional): _description_. Defaults to None.
        **kwargs: further optional parameters for the fit methods

    Returns
    -------
    np.array
    """
    
    # check if inputs are valid
    if not method:
        method = 'interp'
    if not method in ['interp', 'fit', 'stepfunction']:
        raise ValueError(f'Unknown method: {method}. Use "interp", "fit", or "stepfunction" instead')
    if method == 'stepfunction': # x_scale and y_scale unnecessary if fitting stepfunction
        x_scale, y_scale = None, None
    if x_train.shape != y_train.shape:
        raise ValueError(f'Incompatible shapes of input data, x_train {x_train.shape} and y_train {y_train.shape}. Should be the same')
    
    # cut x and y above threshold
    if x_thres or x_thres==0:
        x_th = np.asarray(x_train > x_thres).squeeze()
        x_train = x_train[x_th]
        y_train = y_train[x_th]
    
    if y_thres or y_thres==0:
        y_th = np.asarray(y_train > y_thres).squeeze()
        x_train = x_train[y_th]
        y_train = y_train[y_th]

    # return zeros if x_train and y_train empty
    if x_train.size == 0:
        return np.zeros_like(x_test)
    # return y_train if only one (x_train, y_train) to fit
    if x_train.size == 1:
        return np.full_like(x_test, y_train[0])

    # adapt x and y scale
    if x_scale == 'log':
        x_train, x_test = np.log10(x_train), np.log10(x_test)
    if y_scale == 'log':
        y_train = np.log10(y_train)

    # calculate interpolation
    if method == 'interp':
        if not all(sorted(x_train) == x_train):
            raise ValueError(f'Input array x_train must be sorted in ascending order.')
        y_test = np.interp(x_test, x_train, y_train, **kwargs)

    # calculate linear fit
    elif method == 'fit':
        try: 
            #pol_coef = np.polyfit(x_train, y_train, deg=1) # old fit method (numpy recommends to replace it)
            pol_coef = np.polynomial.polynomial.Polynomial.fit(x_train, y_train, deg=1).convert().coef[::-1]
        except:
            raise ValueError(f"No linear fit possible.")
        y_test = np.polyval(pol_coef, x_test, **kwargs)
    
    # calculate stepfunction fit
    elif method == 'stepfunction':
        # find indeces of x_test if sorted into x_train
        if not all(sorted(x_train) == x_train):
            raise ValueError(f'Input array x_train must be sorted in ascending order.')
        indx = np.searchsorted(x_train, x_test, **kwargs)
        y_test = y_train[indx.clip(max = len(x_train) - 1)]
        y_test[indx == len(x_train)] = np.nan

    # adapt output scale
    if y_scale == 'log':
        y_test = np.power(10., y_test)
    
    return y_test
    

def group_frequency(freq, values):
    """util function to add frequencies for equal values

    Args:
        freq (np.array): frequencies corresponding to the values 
        values (np.array): values sorted in decreasing order

    Returns:
        tuple: (frequencies after aggregation, 
                unique values in cedreasing order)
    """
    if len(values) != len(np.unique(values)):
        #check ordering of values
        if not sorted(values, reverse=True) == values:
            raise ValueError(f'Value array must be sorted in decreasing order.')
        # add frequency for equal values
        values, start_indices = np.unique(values, return_index=True)
        start_indices = np.insert(start_indices, 0, len(freq))
        freq = np.array([
            sum(freq[start_indices[i+1]:start_indices[i]])
            for i in range(len(values))
        ])
        return freq[::-1], values[::-1]
    return freq, values