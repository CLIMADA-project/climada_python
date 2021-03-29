"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Functions to deal with files.
"""

__all__ = [
    'to_list',
    'get_file_names',
]

import glob
import logging
import math
import urllib
from pathlib import Path

import requests
from tqdm import tqdm

from climada.util.config import CONFIG

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


def download_file(url, download_dir=None, overwrite=True):
    """Download file from url to given target folder and provide full path of the downloaded file.

    Parameters
    ----------
    url : str
        url containing data to download
    download_dir : Path or str, optional
        the parent directory of the eventually downloaded file
    overwrite : bool, optional
        whether or not an alredy existing file at the target location should be overwritten,
        by default True

    Returns
    -------
    str
        the full path to the eventually downloaded file
    """
    file_name = url.split('/')[-1]
    if file_name.strip() == '':
        raise ValueError(f"cannot download {url} as a file")
    download_path = CONFIG.local_data.save_dir.dir() if download_dir is None else Path(download_dir)
    file_path = download_path.absolute().joinpath(file_name)
    if file_path.exists():
        if not file_path.is_file() or not overwrite:
            raise FileExistsError(f"cannot download to {file_path}")

    try:
        req_file = requests.get(url, stream=True)
    except IOError as ioe:
        LOGGER.error('Connection error: check url and internet connection.')
        raise ioe
    if req_file.status_code < 200 or req_file.status_code > 299:
        LOGGER.error('Error loading page %s.', url)
        raise ValueError(f'Error loading page {url}\n'
                         + f' Status: {req_file.status_code}\n'
                         + f' Content: {req_file.content}')

    total_size = int(req_file.headers.get('content-length', 0))
    block_size = 1024

    LOGGER.info('Downloading %s to file %s', url, file_path)
    with file_path.open('wb') as file:
        for data in tqdm(req_file.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB', unit_scale=True):
            file.write(data)

    return str(file_path)


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
        raise ValueError(
            f'{exc.__class__} - "{exc}": failed to retrieve {url} into {file_name}'
        ) from exc


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
        list(str)
    """
    file_list = list()

    for pattern in file_name if isinstance(file_name, list) else [file_name]:
        try:
            if Path(pattern).is_file():
                file_list.append(str(pattern))
            elif Path(pattern).is_dir():
                file_list.extend([
                    str(fil) for fil in Path(pattern).iterdir() if fil.is_file()
                ])
            else:  # glob pattern
                file_list.extend([
                    str(Path(fil)) for fil in glob.glob(pattern)
                ])
        except OSError:
            file_list.extend([
                str(Path(fil)) for fil in glob.glob(pattern)
            ])

    return file_list


def get_extension(file_name):
    """Get file without extension and its extension (e.g. ".nc", ".grd.gz").

    Parameters:
        file_name (str): file name (with or without path)

    Returns:
        str, str
    """
    file_path = Path(file_name)
    cuts = file_path.name.split('.')
    return str(file_path.parent.joinpath(cuts[0])), "".join(file_path.suffixes)
