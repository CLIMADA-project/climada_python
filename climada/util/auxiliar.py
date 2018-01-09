"""
auxiliary module
"""

def check_size(exp_len, var, var_name):
    """Check if the length of a variable is the expected one.

        Raises
        ------
            ValueError
    """
    if exp_len != len(var):
        raise ValueError('Invalid %s size: %s != %s' %
                         (var_name, str(exp_len), str(len(var))))
