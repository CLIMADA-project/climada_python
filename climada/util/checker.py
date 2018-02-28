"""
module containing functions to check variables properties.
"""

__all__ = ['size',
           'shape',
           'array_optional',
           'array_default'
          ]

import logging

LOGGER = logging.getLogger(__name__)

def size(exp_len, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises
        ------
            ValueError
    """
    try:        
        if exp_len != len(var):
            LOGGER.error("Invalid %s size: %s != %s.", var_name, exp_len, \
                         len(var))
            raise ValueError
    except TypeError:
        LOGGER.error("%s has wrong dimensions.", var_name)
        raise ValueError

def shape(exp_row, exp_col, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises
        ------
            ValueError
    """
    try:
        if exp_row != var.shape[0]:
            LOGGER.error("Invalid %s row size: %s != %s.", var_name, exp_row,\
                         var.shape[0])
            raise ValueError
        if exp_col != var.shape[1]:
            LOGGER.error("Invalid %s column size: %s != %s.", var_name, \
                         exp_col, var.shape[1])
            raise ValueError
    except TypeError:
        LOGGER.error("%s has wrong dimensions.", var_name)
        raise ValueError


def array_optional(exp_len, var, var_name):
    """Check if array has right size. Warn if array empty. Call check_size.

        Parameters
        ----------
            exp_len (str): expected array size
            var (np.array): numpy array to check
            var_name (str): name of the variable. Used in error/warning msg

        Raises
        ------
            ValueError
    """
    if len(var) == 0:
        LOGGER.warning("%s not set. ", var_name)
    else:
        size(exp_len, var, var_name)

def array_default(exp_len, var, var_name, def_val):
    """Check array has right size. Set default value if empty. Call check_size.

        Parameters
        ----------
            exp_len (str): expected array size
            var (np.array): numpy array to check
            var_name (str): name of the variable. Used in error/warning msg
            def_val (np.array): nump array used as default value

        Raises
        ------
            ValueError

        Returns
        ------
            Filled array
    """
    res = var
    if len(var) == 0:
        LOGGER.warning("%s not set. Default values set.", var_name)
        res = def_val
    else:
        size(exp_len, var, var_name)
    return res
