"""
define save functionalities
"""

import os
import pickle
import logging

from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

def save(out_file_name, var):
    """Save variable with provided file name.
            
    Parameters:
        out_file_name (str): file name (absolute path or relative to configured
            save_dir)
        var (object): variable to save in pickle format
    
    """
    abs_path = out_file_name
    if not os.path.isabs(abs_path):
        abs_path = os.path.abspath(os.path.join( \
                CONFIG['local_data']['save_dir'], out_file_name))
    folder_path = os.path.abspath(os.path.join(abs_path, os.pardir))
    try:
        # Generate folder if it doesn't exists
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            LOGGER.info('Created folder %s.', folder_path)
        with open(abs_path, 'wb') as file:
            pickle.dump(var, file)
            LOGGER.info('Written file %s', abs_path)
    except FileNotFoundError:
        LOGGER.error('Folder not found: %s', folder_path)
        raise FileNotFoundError
