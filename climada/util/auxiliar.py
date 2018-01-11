"""
auxiliary module
"""

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
