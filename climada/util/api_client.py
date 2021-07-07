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
from pathlib import Path
from urllib.parse import quote, unquote
import time

from peewee import CharField, DateTimeField, IntegrityError, Model, SqliteDatabase
import requests


from climada import CONFIG
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

DB = SqliteDatabase(Path(CONFIG.data_api.cache_db.str()).expanduser())


class Download(Model):
    """Database entry keeping track of downloaded files from the CLIMADA data API"""
    url = CharField()
    path = CharField(unique=True)
    startdownload = DateTimeField()
    enddownload = DateTimeField(null=True)

    class Meta:
        """SQL database and table definition."""
        database = DB

    class Failed(Exception):
        """The download failed for some reason."""


DB.connect()
DB.create_tables([Download])


@dataclass
class FileInfo():
    """file data from CLIMADA data API."""
    url:str
    file_name:str
    file_format:str
    file_size:int
    check_sum:str


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

    @staticmethod
    def from_json(jsono):
        """creates a DatasetInfo object from the json object returned by the
        CLIMADA data api server.

        Parameters
        ----------
        jsono : dict

        Returns
        -------
        DatasetInfo
        """
        dataset = DatasetInfo(**jsono)
        dataset.data_type = DataTypeInfo(**dataset.data_type)
        dataset.files = [FileInfo(**filo) for filo in dataset.files]
        return dataset


def checksize(local_path, fileinfo):
    """Checks sanity of downloaded file simply by comparing actual and registered size.

    Parameters
    ----------
    local_path : Path
        the downloaded file
    filinfo : FileInfo
        file information from CLIMADA data API

    Raises
    ------
    Download.Failed
        if the file is not what it's supposed to be
    """
    if not local_path.is_file():
        raise Download.Failed(f"{str(local_path)} is not a file")
    if local_path.stat().st_size != fileinfo.file_size:
        raise Download.Failed(f"{str(local_path)} has the wrong size:"
            f"{local_path.stat().st_size} instead of {fileinfo.file_size}")


def checkhash(local_path, fileinfo):
    """Checks sanity of downloaded file by comparing actual and registered check sum.

    Parameters
    ----------
    local_path : Path
        the downloaded file
    filinfo : FileInfo
        file information from CLIMADA data API

    Raises
    ------
    Download.Failed
        if the file is not what it's supposed to be
    """
    raise NotImplementedError("sanity check by hash sum needs to be implemented yet")


class Client():
    """Python wrapper around REST calls to the CLIMADA data API server.
    """
    MAX_WAITING_PERIOD = 6

    class AmbiguousResult(Exception):
        """Custom Exception for Non-Unique Query Result"""

    class NoResult(Exception):
        """Custom Exception for No Query Result"""

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

        Returns
        -------
        dict
            loaded from the json object of a successful request.

        Raises
        ------
        NoResult
            if the response status code is different from 200
        """
        page = requests.get(url, **kwargs)
        if page.status_code == 200:
            return json.loads(page.content.decode())
        raise Client.NoResult(page.content.decode())

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
        list of DatasetInfo
        """
        url = f'{self.host}/rest/datasets'
        params = {
            'data_type': data_type,
            'name': name,
            'version': version,
            'status': '' if status is None else status,
        }
        params.update(properties if properties else dict())
        return [DatasetInfo.from_json(ds) for ds in Client._request_200(url, params=params)]

    def get_dataset(self, data_type=None, name=None, version=None, properties=None,
                    status=None):
        """Find the one dataset that matches the given parameters.

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
            by default None

        Returns
        -------
        DatasetInfo

        Raises
        ------
        AmbiguousResult
            when there is more than one dataset matching the search parameters
        NoResult
            when there is no dataset matching the search parameters
        """
        jarr = self.get_datasets(data_type=data_type, name=name, version=version,
                                 properties=properties, status=status)
        if len(jarr) > 1:
            raise Client.AmbiguousResult("there are several datasets meeting the requirements:"
                                        f" {jarr}")
        if len(jarr) < 1:
            raise Client.NoResult("there is no dataset meeting the requirements")
        return jarr[0]

    def get_dataset_by_uuid(self, uuid):
        """Returns the data from 'https://climada/rest/dataset/{uuid}' as DatasetInfo object.

        Parameters
        ----------
        uuid : str
            the universal unique identifier of the dataset

        Returns
        -------
        DatasetInfo

        Raises
        ------
        NoResult
            if the uuid is not valid
        """
        url = f'{self.host}/rest/dataset/{uuid}'
        return DatasetInfo.from_json(Client._request_200(url))

    def get_data_types(self, data_type_group=None):
        """Returns all data types from the climada data API
        belonging to a given data type group.

        Parameters
        ----------
        data_type_group : str, optional
            name of the data type group, by default None

        Returns
        -------
        list of DataTypeInfo
        """
        url = f'{self.host}/rest/data_types'
        params = {'data_type_group': data_type_group} \
            if data_type_group else {}
        return [DataTypeInfo(**jobj) for jobj in Client._request_200(url, params=params)]

    def get_data_type(self, data_type):
        """Returns the data type from the climada data API
        with a given name.

        Parameters
        ----------
        data_type : str
            data type name

        Returns
        -------
        DataTypeInfo

        Raises
        ------
        NoResult
            if there is no such data type registered
        """
        url = f'{self.host}/rest/data_type/{quote(data_type)}'
        return DataTypeInfo(**Client._request_200(url))

    def _download(self, url, path, replace=False):
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
        except IntegrityError as ierr:
            dlf = Download.get(Download.path==path_as_str)
            if dlf.url != remote_url:
                raise Exception("this file has been downloaded from another url, "
                    "please purge the entry from data base before trying again") from ierr
            return dlf
        try:
            self._download(url=remote_url, path=local_path, replace=True)
            dlf.enddownload = datetime.utcnow()
            dlf.save()
        except Exception:
            dlf.delete_instance()
            raise
        return Download.get(Download.path==path_as_str)

    def download_file(self, local_path, fileinfo, check=checksize, retries=3):
        """Download a file if it is not already present at the target destination.

        Parameters
        ----------
        local_path : Path
            target destination,
            if it is a directory the original filename (fileinfo.filen_name) is kept
        fileinfo : FileInfo
            file object as retrieved from the data api
        check : function, optional
            how to check download success, by default checksize
        retries : int, optional
            how many times one should retry in case of failure, by default 3

        Raises
        ------
        Exception
            when number of retries was exceeded or when a download is already running
        """
        try:
            if local_path.is_dir():
                local_path /= fileinfo.file_name
            downloaded = self._tracked_download(remote_url=fileinfo.url, local_path=local_path)
            if not downloaded.enddownload:
                raise Download.Failed("Download seems to be in progress, please try again later"
                    " or remove cache entry by calling purge_cache the database!")
            try:
                check(local_path, fileinfo)
            except Download.Failed as dlf:
                local_path.unlink(missing_ok=True)
                self.purge_cache(local_path)
                raise dlf
            return local_path
        except Download.Failed as dle:
            if retries < 1:
                raise dle
            LOGGER.warning("Download failed: %s, retrying...", dle)
            time.sleep(self.MAX_WAITING_PERIOD/retries)
            return self.download_file(local_path=local_path, fileinfo=fileinfo, check=check,
                                      retries=retries - 1)

    def download_dataset(self, dataset, target_dir=SYSTEM_DIR, organize_path=True,
                         check=checksize):
        """Download all files from a given dataset to a given directory.

        Parameters
        ----------
        dataset : DatasetInfo
            the dataset
        target_dir : Path, optional
            target directory for download, by default `climada.util.constants.SYSTEM_DIR`
        organize_path: bool, optional
            if set to True the files will end up in subdirectories of target_dir:
            [target_dir]/[data_type_group]/[data_type]/[name]/[version]
            by default True
        check : function, optional
            how to check download success for each file, by default Download.checksize

        Raises
        ------
        Exception
            when one of the files cannot be downloaded
        """
        if not target_dir.is_dir():
            raise ValueError(f"{target_dir} is not a directory")

        if organize_path:
            if dataset.data_type.data_type_group:
                target_dir /= dataset.data_type.data_type_group
            if dataset.data_type.data_type_group != dataset.data_type.data_type:
                target_dir /= dataset.data_type.data_type
            target_dir /= dataset.name
            if dataset.version:
                target_dir /= dataset.version
            target_dir.mkdir(exist_ok=True, parents=True)

        return [
            self.download_file(local_path=target_dir, fileinfo=dsfile, check=check)
            for dsfile in dataset.files
        ]

    def purge_cache(self, local_path):
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
        dlf = Download.get(Download.path==str(local_path.absolute()))
        dlf.delete_instance()
