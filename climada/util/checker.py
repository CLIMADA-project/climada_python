"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

module containing functions to check variables properties.
"""

__all__ = [
    'size',
    'shape',
    'array_optional',
    'array_default'
]

import logging
import numpy as np
import scipy.sparse as sparse

LOGGER = logging.getLogger(__name__)

def check_oligatories(var_dict, var_obl, name_prefix, n_size, n_row, n_col):
    """Check size of obligatory variables.

    Paraemters:
        var_dict (dict): __dict__ class attribute
        var_obl (set): name of the obligatory variables
        name_prefix (str): name to add in the error log, e.g. the class name
        n_size (int): size expected from arrays and lists
        n_row (int): number of rows expected in 2D arrays
        n_col (int): number of columns expected in 2D arrays

    Raises:
        ValueError
    """
    for var_name, var_val in var_dict.items():
        if var_name in var_obl:
            if (isinstance(var_val, np.ndarray) and var_val.ndim == 1) \
               or isinstance(var_val, list):
                size(n_size, var_val, name_prefix + var_name)
            elif (isinstance(var_val, np.ndarray) and var_val.ndim == 2):
                shape(n_row, n_col, var_val, name_prefix + var_name)
            elif isinstance(var_val, (np.ndarray, sparse.csr.csr_matrix)) and var_val.ndim == 2:
                shape(n_row, n_col, var_val, name_prefix + var_name)

def check_optionals(var_dict, var_opt, name_prefix, n_size):
    """Check size of obligatory variables.

    Paraemters:
        var_dict (dict): __dict__ class attribute
        var_opt (set): name of the ooptional variables
        name_prefix (str): name to add in the error log, e.g. the class name
        n_size (int): size expected from arrays and lists

    Raises:
        ValueError
    """
    for var_name, var_val in var_dict.items():
        if var_name in var_opt:
            if isinstance(var_val, (np.ndarray, list)):
                array_optional(n_size, var_val, name_prefix + var_name)

def empty_optional(var, var_name):
    """Check if a data structure is empty."""
    if not var:
        LOGGER.debug("%s not set. ", var_name)

def size(exp_len, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises:
            ValueError
    """
    try:
        if exp_len != len(var):
            LOGGER.error("Invalid %s size: %s != %s.", var_name, exp_len,
                         len(var))
            raise ValueError
    except TypeError:
        LOGGER.error("%s has wrong dimensions.", var_name)
        raise ValueError

def shape(exp_row, exp_col, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises:
            ValueError
    """
    try:
        if exp_row != var.shape[0]:
            LOGGER.error("Invalid %s row size: %s != %s.", var_name, exp_row,
                         var.shape[0])
            raise ValueError
        if exp_col != var.shape[1]:
            LOGGER.error("Invalid %s column size: %s != %s.", var_name,
                         exp_col, var.shape[1])
            raise ValueError
    except TypeError:
        LOGGER.error("%s has wrong dimensions.", var_name)
        raise ValueError


def array_optional(exp_len, var, var_name):
    """Check if array has right size. Warn if array empty. Call check_size.

        Parameters:
            exp_len (str): expected array size
            var (np.array): numpy array to check
            var_name (str): name of the variable. Used in error/warning msg

        Raises:
            ValueError
    """
    if len(var) == 0 and exp_len > 0:
        LOGGER.debug("%s not set. ", var_name)
    else:
        size(exp_len, var, var_name)

def array_default(exp_len, var, var_name, def_val):
    """Check array has right size. Set default value if empty. Call check_size.

        Parameters:
            exp_len (str): expected array size
            var (np.array): numpy array to check
            var_name (str): name of the variable. Used in error/warning msg
            def_val (np.array): nump array used as default value

        Raises:
            ValueError

        Returns:
            Filled array
    """
    res = var
    if len(var) == 0 and exp_len > 0:
        LOGGER.debug("%s not set. Default values set.", var_name)
        res = def_val
    else:
        size(exp_len, var, var_name)
    return res
