"""
Functions to deal with files.
"""

__all__ = ['to_str_list',
           'get_file_names'
          ]

import os
import logging

def to_str_list(num_exp, values, val_name):
    """Check size and transform to list if necessary. If size is one, build
    a list with num_exp repeated values.
    
    Parameters
    ----------
        num_exp (int): number of expect list elements
        values (object or list(object)): values to check and transform
        val_name (str): name of the variable values

    Returns
    -------
        list
    """
    val_list = list()
    if isinstance(values, list):
        if len(values) == num_exp:
            val_list = values
        elif len(values) == 1:
            val_list = list()
            val_list += num_exp * [values[0]]
        else:
            logger = logging.getLogger(__name__)
            logger.error('Provide one or %s %s.', num_exp, val_name)
    else:
        val_list += num_exp * [values]
    return val_list

def get_file_names(file_name):
    """Return list of files contained.
    
    Parameters
    ----------
        file_name (str or list(str)): file name, or list of file names or name
            of the folder containing the files

    Returns
    -------
        list
    """
    file_list = list()
    if isinstance(file_name, list):
        for file in file_name:
            _process_one_file_name(file, file_list)
    else:
        _process_one_file_name(file_name, file_list)
    return file_list

def _process_one_file_name(name, file_list):
    """Apend to input list the file contained in name"""
    if os.path.splitext(name)[1] == '': 
        tmp_files = os.listdir(name)
        # append only files, not folders
        for file in tmp_files:
            if os.path.splitext(file)[1] != '': 
                file_list.append(file)
    else:
        file_list.append(name)
