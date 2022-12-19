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

init test
"""

from climada.util.api_client import Client


def get_test_file(ds_name, file_format=None):
    """convenience method for downloading test files from the CLIMADA Data API.
    By convention they have a status 'test_dataset', their name is unique and they contain a single
    file. These conventions are used to identify and download the file to the
    CLIMADA data folder.

    Parameters
    ----------
    ds_name : str
        name of the dataset
    file_format : str, optional
        file format to look for within the datasets files.
        Any format is accepted if it is ``None``.
        Default is ``None``.
    Returns
    -------
    pathlib.Path
        the path to the downloaded file
    """
    client = Client()
    test_ds = client.get_dataset_info(name=ds_name, status='test_dataset')
    _, files = client.download_dataset(test_ds)
    [test_file] = [fil for fil in files if fil.name in [
        dsf.file_name 
        for dsf in test_ds.files 
        if file_format is None or dsf.file_format == file_format]]
    return test_file
