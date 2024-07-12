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
        method=None, 
        x_scale=None, 
        y_scale=None, 
        x_thres = None, 
        y_thres = None
    ):
    """Save variable with provided file name. Uses configuration save_dir folder
    if no absolute path provided.

    Parameters
    ----------
    x : np.array
        x values (1-D array of x values to fit)
    y : np.array
        y values (1-D array of y values to fir)
    method : object
        variable to save in pickle format

    Returns
    -------
    np.array
    """
    # check if inputs are valid
    if not method:
        method = 'interp'
    if not method in ['interp', 'fit', 'stepfunction']:
        raise ValueError(f'Unknown method: {method}. Use "interp", "fit", or "stepfunction" instead')
    if x_train.shape != y_train.shape:
        raise ValueError(f'Incompatible shapes of input data, x_train {x_train.shape} and y_train {y_train.shape}. Should be the same')
    
    # cut x and y above threshold
    if x_thres:
        x_th = np.asarray(x_train > x_thres).squeeze()
        x_train = x_train[x_th]
        y_train = y_train[x_th]
    
    if y_thres:
        y_th = np.asarray(y_train > y_thres).squeeze()
        x_train = x_train[y_th]
        y_train = y_train[y_th]


    # adapt x and y scale
    if x_scale == 'log':
        x_train, x_test = np.log10(x_train), np.log10(x_test)
    if y_scale == 'log':
        y_train = np.log10(y_train)


    # calculate interpolation
    if method == 'interp':
        if not np.all(x_train[:-1] <= x_train[1:]):
            raise ValueError(f'Input array x_train must be sorted.')
        y_test = np.interp(x_test, x_train, y_train)

    # calculate linear fit
    elif method == 'fit':
        try: 
            pol_coef = np.polyfit(x_train, y_train, deg=1)
        except ValueError:
            pol_coef = np.polyfit(x_train, y_train, deg=0)
        y_test = np.polyval(pol_coef, x_test)
    
    # calculate stepfunction fit
    elif method == 'stepfunction':
        # find indeces of x_test if sorted into x_train
        if not np.all(x_train[:-1] <= x_train[1:]):
            raise ValueError(f'Input array x_train must be sorted.')
        indx = np.searchsorted(x_train, x_test)
        y_test = y_train[indx.clip(max = len(x_train) - 1)]
        y_test[indx == len(x_train)] = np.nan

    # adapt output scale
    if y_scale == 'log':
        y_test = np.exp(y_test)
    
    return y_test
    

