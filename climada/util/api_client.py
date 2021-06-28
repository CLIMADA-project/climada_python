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
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from urllib.parse import quote, unquote
import time

from peewee import CharField, DateTimeField, IntegrityError, Model, SqliteDatabase
import requests


from climada import CONFIG

LOGGER = logging.getLogger(__name__)

class AmbiguousResult(Exception):
    """Custom Exception for Non-Unique Query Result"""


class NoResult(Exception):
    """Custom Exception for No Query Result"""


class Download(Model):
    """Database entry keeping track of downloaded files from the CLIMADA data API"""
    url = CharField()
    path = CharField()
    startdownload = DateTimeField()
    enddownload = DateTimeField(null=True)

    class Meta:
        """SQL database and table definition."""
        database = SqliteDatabase(CONFIG.data_api.cache_db.str())
        indexes = (
            (('url', 'path'), True),
        )

    @staticmethod
    def checksize(local_path, filinfo):
        return True

    @staticmethod
    def checkhash(local_path, fileinfo):
        return True

    class Failed(Exception):
        """The download failed for some reason."""


@dataclass
class DataTypeInfo():
    """data_type data from CLIMADA data API."""
    data_type:str
    data_type_group:str
    description:str


@dataclass
class DatasetInfo():
    """dataset data from CLIMADA data API."""
    uuid:str
    data_type:DataTypeInfo
    name:str
    version:str
    status:str
    properties:dict
    files:list  # of FileInfo
    doi:str
    description:str
    activation_date:str
    expiration_date:str


@dataclass
class FileInfo():
    """file data from CLIMADA data API."""
    url:str
    file_name:str
    file_format:str
    file_size:int
    check_sum:str


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
    def _request_200(url, **kwargs):
        """Helper method, triaging successfull and failing requests.

        Raises
        ------
        NoResult
            if the response status code is different from 200
        """
        page = requests.get(url, **kwargs)
        if page.status_code == 200:
            return json.loads(page.content.decode())
        raise NoResult(page.content.decode())

    def get_datasets(self, data_type=None, name=None, version=None, properties=None,
                     status='active'):
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
        }
        params.update(properties if properties else dict())
        datasets = [DatasetInfo(**ds) for ds in Client._request_200(url, params=params)]
        for dataset in datasets:
            dataset.data_type = DataTypeInfo(**dataset.data_type)
            dataset.files = [FileInfo(**filo) for filo in dataset.files]
        return datasets

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
        return [DataTypeInfo(**jobj) for jobj in Client._request_200(url, params=params)]

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
        return DataTypeInfo(**Client._request_200(url))

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

    def _tracked_download(self, remote_url, local_path):
        if local_path.is_dir():
            raise Exception("tracked download requires a path to a file not a directory")
        path_as_str = str(local_path.absolute())
        try:
            dlf = Download.create(url=remote_url, path=path_as_str, startdownload=datetime.utcnow())
        except IntegrityError:
            return Download.get((Download.url==remote_url)
                              & (Download.path==path_as_str)).enddownload
        try:
            self.download(url=remote_url, path=path_as_str, replace=True)
            dlf.enddownload = datetime.utcnow()
            dlf.save()
        except Exception:
            dlf.delete_instance()
            raise
        return Download.get((Download.url==remote_url) & (Download.path==path_as_str)).enddownload

    def cached_download(self, local_path, fileinfo, check=Download.checksize, retries=3):
        """Download a file if it is not already present at the target destination.

        Parameters
        ----------
        local_path : Path
            target destination
        fileinfo : FileInfo
            file object as retrieved from the data api
        check : function, optional
            how to check download success, by default Download.checksize
        retries : int, optional
            how many times one should retry in case of failure, by default 3

        Raises
        ------
        Exception
            when number of retries was exceeded or when a download is already running
        """
        try:
            downloaded = self._tracked_download(remote_url=fileinfo.url, local_path=local_path)
            if not downloaded:
                raise Exception("Download seems to be in progress, please try again later"
                                " or remove cache entry by calling purge_cache the database!")
            check(local_path, fileinfo)
        except Download.Failed as dle:
            if retries > 0:
                LOGGER.warning("Download failed, retrying...")
                time.sleep(6/retries)
                self.cached_download(local_path=local_path, fileinfo=fileinfo, check=check,
                                     retries=retries - 1)
            else:
                raise Exception("Download failed, won't retry") from dle

    def purge_cache(self, local_path, fileinfo):
        """Removes entry from the sqlite database that keeps track of files downloaded by
        `cached_download`. This may be necessary in case a previous attempt has failed
        in an uncontroled way (power outage or the like).

        Parameters
        ----------
        local_path : Path
            target destination
        fileinfo : FileInfo
            file object as retrieved from the data api
        """
        dlf = Download.get((Download.url==fileinfo.url)
                         & (Download.path==str(local_path.absolute())))
        dlf.delete_instance()
