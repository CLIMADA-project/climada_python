"""
auxiliary module
"""

import warnings

def check_size(exp_len, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises
        ------
            ValueError
    """
    try:
        if exp_len != len(var):
            raise ValueError('Invalid %s size: %s != %s' %
                             (var_name, str(exp_len), str(len(var))))
    except TypeError:
        raise ValueError('%s has wrong dimensions.' % var_name)

def check_shape(exp_row, exp_col, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises
        ------
            ValueError
    """
    try:
        if exp_row != var.shape[0]:
            raise ValueError('Invalid %s row size: %s != %s' %
                             (var_name, str(exp_row), str(var.shape[0])))
        if exp_col != var.shape[1]:
            raise ValueError('Invalid %s column size: %s != %s' %
                             (var_name, str(exp_col), str(var.shape[1])))
    except TypeError:
        raise ValueError('%s has wrong dimensions.' % var_name)


def check_array_optional(size, var, var_name):
    """Check if array has right size. Warn if array empty. Call check_size.

        Parameters
        ----------
            size (str): expected array size
            var (np.array): numpy array to check
            var_name (str): name of the variable. Used in error/warning msg

        Raises
        ------
            ValueError
    """
    if var.size == 0:
         warnings.warn("%s not set. " % var_name)
    else:
         check_size(size, var, var_name)

def check_array_default(size, var, var_name, def_val):
    """Check array has right size. Set default value if empty. Call check_size.
    
        Parameters
        ----------
            size (str): expected array size
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
    if var.size == 0:
        warnings.warn("%s not set. Default values set." % var_name)
        res = def_val
    else:
        check_size(size, var, var_name)
    return res
