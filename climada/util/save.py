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

define save functionalities
"""

__all__ = ['save',
           'load']

import os
import pickle
import logging

from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

def save(out_file_name, var):
    """Save variable with provided file name. Uses configuration save_dir folder
    if no absolute path provided.

    Parameters:
        out_file_name (str): file name (absolute path or relative to configured
            save_dir)
        var (object): variable to save in pickle format
    """
    abs_path = out_file_name
    if not os.path.isabs(abs_path):
        abs_path = os.path.abspath(os.path.join(
            CONFIG['local_data']['save_dir'], out_file_name))
    folder_path = os.path.abspath(os.path.join(abs_path, os.pardir))
    try:
        # Generate folder if it doesn't exists
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
            LOGGER.info('Created folder %s.', folder_path)
        with open(abs_path, 'wb') as file:
            pickle.dump(var, file, pickle.HIGHEST_PROTOCOL)
            LOGGER.info('Written file %s', abs_path)
    except FileNotFoundError:
        LOGGER.error('Folder not found: %s', folder_path)
        raise FileNotFoundError
    except OSError:
        LOGGER.error('Data is probably too big. Try splitting it.')
        raise ValueError

def load(in_file_name):
    """Load variable contained in file. Uses configuration save_dir folder
    if no absolute path provided.

    Parameters:
        in_file_name (str)

    Returns:
        object
    """
    abs_path = in_file_name
    if not os.path.isabs(abs_path):
        abs_path = os.path.abspath(os.path.join(
            CONFIG['local_data']['save_dir'], in_file_name))
    with open(abs_path, 'rb') as file:
        data = pickle.load(file)
    return data
