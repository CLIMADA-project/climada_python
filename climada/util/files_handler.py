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

Functions to deal with files.
"""

__all__ = [
    'to_list',
    'get_file_names',
]

import os
import glob
import logging
import math
import urllib
import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

class DownloadProgressBar(tqdm):
    """Class to use progress bar during dowloading"""
    def update_to(self, blocks=1, bsize=1, tsize=None):
        """Update progress bar

        Parameters:
            blocks (int, otional): Number of blocks transferred so far [default: 1].
            bsize  (int, otional): Size of each block (in tqdm units) [default: 1].
            tsize  (int, otional): Total size (in tqdm units). If [default: None]
                remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)

def download_file(url):
    """Download file from url in current folder and provide absolute file path
    and name.

    Parameters:
        url (str): url containing data to download

    Returns:
        str

    Raises:
        ValueError
    """
    try:
        req_file = requests.get(url, stream=True)
    except IOError:
        LOGGER.error('Connection error: check your internet connection.')
        raise IOError
    if req_file.status_code == 404:
        LOGGER.error('Error loading page %s.', url)
        raise ValueError
    if req_file.status_code == 503:
        LOGGER.error('Service Unavailable: %s.', url)
        raise ValueError
    total_size = int(req_file.headers.get('content-length', 0))
    block_size = 1024
    file_name = url.split('/')[-1]
    file_abs_name = os.path.abspath(os.path.join(os.getcwd(), file_name))
    LOGGER.info('Downloading file %s', file_abs_name)
    with open(file_name, 'wb') as file:
        for data in tqdm(req_file.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB', unit_scale=True):
            file.write(data)
    return file_abs_name

def download_ftp(url, file_name):
    """Download file from ftp in current folder.

    Parameters:
        url (str): url containing data to download
        file_name (str): name of the file to dowload

    Raises:
        ValueError
    """
    LOGGER.info('Downloading file %s', file_name)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1,
                                 desc=url.split('/')[-1]) as prog_bar:
            urllib.request.urlretrieve(url, file_name, reporthook=prog_bar.update_to)
    except Exception as exc:
        raise ValueError(f'{exc.__class__} - "{exc}": failed to retrieve {url} into {file_name}')

def to_list(num_exp, values, val_name):
    """Check size and transform to list if necessary. If size is one, build
    a list with num_exp repeated values.

    Parameters:
        num_exp (int): number of expect list elements
        values (object or list(object)): values to check and transform
        val_name (str): name of the variable values

    Returns:
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
    """Return list of files contained. Supports globbing.

    Parameters:
        file_name (str or list(str)): Either a single string or a list of
            strings that are either
                - a file path
                - or the path of the folder containing the files
                - or a globbing pattern.

    Returns:
        list
    """
    file_list = list()
    if isinstance(file_name, list):
        for file in file_name:
            _process_one_file_name(file, file_list)
    else:
        _process_one_file_name(file_name, file_list)
    return file_list

def get_extension(file_name):
    """Get file without extension and its extension (e.g. ".nc", ".grd.gz").

    Parameters:
        file_name (str): file name (with or without path)

    Returns:
        str, str
    """
    file_pth, file_ext = os.path.splitext(file_name)
    file_pth_bis, file_ext_bis = os.path.splitext(file_pth)
    if file_ext_bis and file_ext_bis[0] == '.':
        return file_pth_bis, file_ext_bis + file_ext
    return file_pth, file_ext

def _process_one_file_name(name, file_list):
    """Apend to input list the file contained in name
        Tries globbing if name is neither dir nor file.
    """
    if os.path.isdir(name):
        tmp_files = glob.glob(os.path.join(name, '*'))
        for file in tmp_files:
            if os.path.isfile(file):
                file_list.append(file)
    if os.path.isfile(name):
        file_list.append(name)
    else:
        tmp_files = sorted(glob.glob(name))
        for file in tmp_files:
            if os.path.isfile(file):
                file_list.append(file)
