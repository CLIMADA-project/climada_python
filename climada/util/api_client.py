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

Data API client
"""
import json
from urllib.parse import quote, unquote

import requests

from climada import CONFIG

class AmbiguousResult(Exception):
    """Custom Exception for Non-Unique Query Result"""


class NoResult(Exception):
    """Custom Exception for No Query Result"""

class Client():
    """Python wrapper around REST calls to the CLIMADA data API server.
    """
    def __init__(self):
        """Constructor of Client.
        
        Data API host and chunk_size (for download) are configurable values.
        Default values are 'climada.ethz.ch' and 8096 respectively.
        """
        self.headers = {"accept": "application/json"}
        self.host = CONFIG.data_api.host.str().rstrip("/")
        self.chunk_size = CONFIG.data_api.chunk_size.int()

    @staticmethod
    def _passes(cds, parameters):
        if parameters:
            obj_parameters = cds['parameters']
            for key, val in parameters.items():
                if val != obj_parameters.get(key, ''):
                    return False
        return True

    @staticmethod
    def _request_200(url, **kwargs):
        page = requests.get(url, **kwargs)
        if page.status_code == 200:
            return json.loads(page.content.decode())
        raise NoResult(page.content.decode())

    def get_datasets(self, data_type=None, name=None, version=None, properties=None, status='active'):
        """Find all datasets matching the given parameters.

        Parameters
        ----------
        data_type : str, optional
            data_type of the dataset, e.g., 'litpop' or 'draught'
        name : str, optional
            the name of the dataset
        version : str, optional
            the version of the dataset
        properties : dict, optional
            search parameters for dataset properties, by default None
        status : str, optional
            valid values are 'preliminary', 'active', 'expired', and 'test_dataset',
            by default 'active'

        Returns
        -------
        list
            each item representing a dataset as a dictionary
        """
        url = f'{self.host}/rest/datasets'
        params = {
            'data_type': data_type,
            'name': name,
            'version': version,
            'status': '' if status is None else status,
        }.update(properties)

        page = requests.get(url, params=params)
        jarr = json.loads(page.content.decode())

        if name:
            jarr = [jo for jo in jarr if jo['name'] == name]
        if version:
            jarr = [jo for jo in jarr if jo['version'] == version]

        return jarr

    def get_dataset(self, data_type=None, name=None, version=None, properties=None):
        """Find the one (active) dataset that matches the given parameters.

        Parameters
        ----------
        data_type : str, optional
            data_type of the dataset, e.g., 'litpop' or 'draught'
        name : str, optional
            the name of the dataset
        version : str, optional
            the version of the dataset
        properties : dict, optional
            search parameters for dataset properties, by default None

        Returns
        -------
        dict
            the dataset json object, as returned from the api server

        Raises
        ------
        AmbiguousResult
            when there is more than one dataset matching the search parameters
        NoResult
            when there is no dataset matching the search parameters
        """
        jarr = self.get_datasets(data_type=data_type, name=name, version=version,
                                 properties=properties, status='')
        jarr = [jo for jo in jarr if Client._passes(jo, properties)]
        if len(jarr) > 1:
            raise AmbiguousResult(f"there are several datasets meeting the requirements: {jarr}")
        if len(jarr) < 1:
            raise NoResult("there is no dataset meeting the requirements")
        return jarr[0]

    def get_dataset_by_uuid(self, uuid):
        """[summary]

        Parameters
        ----------
        uuid : [type]
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        NoResult
            [description]
        """
        url = f'{self.host}/rest/dataset/{uuid}'
        return Client._request_200(url)

    def get_data_types(self, data_type_group=None):
        """[summary]

        Parameters
        ----------
        data_type_group : [type], optional
            [description], by default None

        Returns
        -------
        [type]
            [description]
        """
        url = f'{self.host}/rest/data_types'
        params = {'data_type_group': data_type_group} \
            if data_type_group else {}
        return Client._request_200(url, params=params)

    def get_data_type(self, data_type):
        """[summary]

        Parameters
        ----------
        data_type : str
            [description]

        Returns
        -------
        [type]
            [description]

        Raises
        ------
        NoResult
            [description]
        """
        url = f'{self.host}/rest/data_type/{quote(data_type)}'
        return Client._request_200(url)

    def download(self, url, path, replace=False):
        """Downloads a file from the given url to a specified location.

        Parameters
        ----------
        url : str
            the link to the file to be downloaded
        path : Path
            download path, if it's a directory the original file name is kept
        replace : bool, optional
            flag to indicate whether a present file with the same name should
            be replaced

        Returns
        -------
        Path
            Path to the downloaded file

        Raises
        ------
        FileExistsError
            in case there is already a file present at the given location
            and replace is False
        """
        if path.is_dir():
            path /= unquote(url.split('/')[-1])
        if path.is_file() and not replace:
            raise FileExistsError(path)
        with requests.get(url, stream=True) as stream:
            stream.raise_for_status()
            with open(path, 'wb') as dump:
                for chunk in stream.iter_content(chunk_size=self.chunk_size):
                    dump.write(chunk)
        return path
