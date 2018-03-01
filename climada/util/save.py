"""
define save functionalities
"""

import os
import pickle
import logging

from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

def save(out_file_name, var):
    """Save variable with provided file name in configurable directory.
            
    Parameters
    ----------
        out_file_name (str): file name (relative to configured save_dir)
        var (object): variable to save in pickle format
    
    """
    abs_path = os.path.abspath(os.path.join(CONFIG['local_data']['save_dir'], \
                                            out_file_name))
    try:
        with open(abs_path, 'wb') as file:
            pickle.dump(var, file)
    except FileNotFoundError:
        folder_path = os.path.abspath(os.path.join(abs_path, os.pardir))
        LOGGER.error('Folder not found: %s', folder_path)
        raise FileNotFoundError
