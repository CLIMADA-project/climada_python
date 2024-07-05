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

def calc_fit_interp(x_test, x_train, y_train, method=None, x_scale=None, y_scale=None):
    """Save variable with provided file name. Uses configuration save_dir folder
    if no absolute path provided.

    Parameters
    ----------
    x : np.array
        x values 
    y : np.array
        y values
    method : object
        variable to save in pickle format

    Returns
    -------
    np.array
    """
    if not method:
        method = 'interp'
    if not method in ['interp', 'fit']:
        raise ValueError(f'Unknown method: {method}. Use "interp" or "fit" instead')
    if x_train.shape != y_train.shape:
        raise ValueError(f'Incompatible shapes of input data, x_train {x_train.shape} and y_train {y_train.shape}. Should be the same')
    if len(x_train.shape) == 1:
        x_train, y_train = x_train.reshape(-1,1), y_train.reshape(-1,1)
    
    
    # adapt x and y scale
    if x_scale == 'log':
        x_train, x_test = np.log10(x_train), np.log10(x_test)
    if y_scale == 'log':
        y_train = np.log10(y_train)

    # calculate fit/interpolation
    y_test = np.zeros((x_test.shape[0], x_train.shape[1]))
    for i in range(x_train.shape[1]):
        if method == 'interp':
            y_test[:, i] = np.interp(x_test, x_train[:, i], y_train[:, i])
        elif method == 'fit':
            try: 
                pol_coef = np.polyfit(x_train[:, i], y_train[:, i], deg=1)
            except ValueError:
                pol_coef = np.polyfit(x_train[:, i], y_train[:, i], deg=0)
            y_test[:, i] = np.polyval(pol_coef, x_test)

    # adapt output scale
    if y_scale == 'log':
        y_test = np.exp(y_test)
    
    return np.squeeze(y_test)

def local_fit_interp(x_test, x_train, y_train, method=None, x_scale=None, y_scale=None):

    return    

    

