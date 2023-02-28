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
import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit
import time

import pandas as pd
from peewee import CharField, DateTimeField, IntegrityError, Model, SqliteDatabase
import requests
import pycountry

from climada import CONFIG
from climada.entity import Exposures
from climada.hazard import Hazard, Centroids
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

DB = SqliteDatabase(Path(CONFIG.data_api.cache_db.str()).expanduser())

HAZ_TYPES = [ht.str() for ht in CONFIG.data_api.supported_hazard_types.list()]
EXP_TYPES = [et.str() for et in CONFIG.data_api.supported_exposures_types.list()]


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
    uuid:str
    url:str
    file_name:str
    file_format:str
    file_size:int
    check_sum:str


@dataclass
class DataTypeInfo():
    """data type meta data from CLIMADA data API."""
    data_type:str
    data_type_group:str
    status: str
    description:str
    properties:list  # of dict


@dataclass
class DataTypeShortInfo():
    """data type name and group from CLIMADA data API."""
    data_type:str
    data_type_group:str


@dataclass
class DatasetInfo():
    """dataset data from CLIMADA data API."""
    uuid:str
    data_type:DataTypeShortInfo
    name:str
    version:str
    status:str
    properties:dict
    files:list  # of FileInfo
    doi:str
    description:str
    license: str
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
        dataset.data_type = DataTypeShortInfo(data_type=dataset.data_type['data_type'],
                                              data_type_group=dataset.data_type['data_type_group'])
        dataset.files = [FileInfo(uuid=dataset.uuid, **filo) for filo in dataset.files]
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


class Cacher():
    """Utility class handling cached results from http requests,
    to enable the API Client working in offline mode.
    """
    def __init__(self, cache_enabled):
        """Constructor of Cacher.

        Parameters
        ----------
        cache_enabled : bool, None
            Default: None, in this case the value is taken from CONFIG.data_api.cache_enabled.
        """
        self.enabled = (CONFIG.data_api.cache_enabled.bool()
                        if cache_enabled is None else cache_enabled)
        self.cachedir = CONFIG.data_api.cache_dir.dir() if self.enabled else None

    @staticmethod
    def _make_key(*args, **kwargs):
        as_text = '\t'.join(
            [str(a) for a in args] +
            [f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]
        )
        print(as_text)
        md5h = hashlib.md5()
        md5h.update(as_text.encode())
        return md5h.hexdigest()

    def store(self, result, *args, **kwargs):
        """stores the result from a API call to a local file.

        The name of the file is the md5 hash of a string created from the call's arguments, the
        content of the file is the call's result in json format.

        Parameters
        ----------
        result : dict
            will be written in json format to the cached result file
        *args : list of str
        **kwargs : list of dict of (str,str)
        """
        _key = Cacher._make_key(*args, **kwargs)
        try:
            with Path(self.cachedir, _key).open('w', encoding='utf-8') as flp:
                json.dump(result, flp)
        except (OSError, ValueError):
            pass

    def fetch(self, *args, **kwargs):
        """reloads the result from a API call from a local file, created by the corresponding call
        of `self.store`.

        If no call with exactly the same arguments has been made in the past, the result is None.

        Parameters
        ----------
        *args : list of str
        **kwargs : list of dict of (str,str)

        Returns
        -------
        dict or None
        """
        _key = Cacher._make_key(*args, **kwargs)
        try:
            with Path(self.cachedir, _key).open(encoding='utf-8') as flp:
                return json.load(flp)
        except (OSError, ValueError):
            return None


class Client():
    """Python wrapper around REST calls to the CLIMADA data API server.
    """
    MAX_WAITING_PERIOD = 6
    UNLIMITED = 100000
    DOWNLOAD_TIMEOUT = 3600
    QUERY_TIMEOUT = 300

    class AmbiguousResult(Exception):
        """Custom Exception for Non-Unique Query Result"""

    class NoResult(Exception):
        """Custom Exception for No Query Result"""

    class NoConnection(Exception):
        """To be raised if there is no internet connection and no cached result."""

    def _online(self) -> bool:
        """Check if this client can connect to the target URL"""
        # Use just the base location
        parse_result = urlsplit(self.url)
        query_url = urlunsplit((parse_result.scheme, parse_result.netloc, "", "", ""))

        try:
            # NOTE: 'timeout' might not work as intended, depending on OS and network status
            return requests.head(query_url, timeout=1).status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def __init__(self, cache_enabled=None):
        """Constructor of Client.

        Data API host and chunk_size (for download) are configurable values.
        Default values are 'climada.ethz.ch' and 8096 respectively.

        Parameters
        ----------
        cache_enabled : bool, optional
            This flag controls whether the api calls of this client are going to be cached to the
            local file system (location defined by CONFIG.data_api.cache_dir).
            If set to true, the client can reload the results from the cache in case there is no
            internet connection and thus work in offline mode.
            Default: None, in this case the value is taken from CONFIG.data_api.cache_enabled.
        """
        self.headers = {"accept": "application/json"}
        self.url = CONFIG.data_api.url.str().rstrip("/")
        self.chunk_size = CONFIG.data_api.chunk_size.int()
        self.cache = Cacher(cache_enabled)
        self.online = self._online()

    def _request_200(self, url, params=None):
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
        # pylint: disable=no-else-return

        if params is None:
            params = dict()

        if self.online:
            page = requests.get(url, params=params, timeout=Client.QUERY_TIMEOUT)
            if page.status_code != 200:
                raise Client.NoResult(page.content.decode())
            result = json.loads(page.content.decode())
            if self.cache.enabled:
                self.cache.store(result, url, **params)
            return result

        else:  # try to restore previous results from an identical request
            if not self.cache.enabled:
                raise Client.NoConnection("there is no internet connection and the client does"
                                          " not cache results.")
            cached_result = self.cache.fetch(url, **params)
            if not cached_result:
                raise Client.NoConnection("there is no internet connection and the client has not"
                                          " found any cached result for this request.")
            LOGGER.warning("there is no internet connection but the client has stored the results"
                           " of this very request sometime in the past.")
            return cached_result


    @staticmethod
    def _divide_straight_from_multi(properties):
        straights, multis = dict(), dict()
        for k, _v in properties.items():
            if isinstance(_v, str):
                straights[k] = _v
            elif isinstance(_v, list):
                multis[k] = _v
            else:
                raise ValueError("properties must be a string or a list of strings")
        return straights, multis

    @staticmethod
    def _filter_datasets(datasets, multi_props):
        pdf = pd.DataFrame([ds.properties for ds in datasets])
        for prop, selection in multi_props.items():
            pdf = pdf[pdf[prop].isin(selection)]
        return [datasets[i] for i in pdf.index]

    def list_dataset_infos(self, data_type=None, name=None, version=None, properties=None,
                     status='active'):
        """Find all datasets matching the given parameters.

        Parameters
        ----------
        data_type : str, optional
            data_type of the dataset, e.g., 'litpop' or 'draught'
        name : str, optional
            the name of the dataset
        version : str, optional
            the version of the dataset, 'any' for all versions, 'newest' or None for the newest
            version meeting the requirements
            Default: None
        properties : dict, optional
            search parameters for dataset properties, by default None
            any property has a string for key and can be a string or a list of strings for value
        status : str, optional
            valid values are 'preliminary', 'active', 'expired', 'test_dataset' and None
            by default 'active'

        Returns
        -------
        list of DatasetInfo
        """
        url = f'{self.url}/dataset'
        params = {
            'data_type': data_type,
            'name': name,
            'version': version,
            'status': '' if status is None else status,
            'limit': Client.UNLIMITED,
        }

        if properties:
            straight_props, multi_props = self._divide_straight_from_multi(properties)
        else:
            straight_props, multi_props = None, None

        if straight_props:
            params.update(straight_props)

        datasets = [DatasetInfo.from_json(ds) for ds in self._request_200(url, params=params)]

        if datasets and multi_props:
            return self._filter_datasets(datasets, multi_props)
        return datasets

    def get_dataset_info(self, data_type=None, name=None, version=None, properties=None,
                    status='active'):
        """Find the one dataset that matches the given parameters.

        Parameters
        ----------
        data_type : str, optional
            data_type of the dataset, e.g., 'litpop' or 'draught'
        name : str, optional
            the name of the dataset
        version : str, optional
            the version of the dataset
            Default: newest version meeting the requirements
        properties : dict, optional
            search parameters for dataset properties, by default None
            any property has a string for key and can be a string or a list of strings for value
        status : str, optional
            valid values are 'preliminary', 'active', 'expired', 'test_dataset', None
            by default 'active'

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
        jarr = self.list_dataset_infos(data_type=data_type, name=name, version=version,
                                 properties=properties, status=status)
        if len(jarr) > 1:
            shown = 10
            endofmessage = '' if len(jarr) <= shown else f'\nand {len(jarr)-shown} more'
            datasetlist = ',\n* '.join(str(jarr[i]) for i in range(min(shown, len(jarr))))
            raise Client.AmbiguousResult(f"there are {len(jarr)} datasets meeting the requirements:"
                                         f"\n* {datasetlist}{endofmessage}.")
        if len(jarr) < 1:
            data_info = self.list_dataset_infos(data_type)
            properties = self.get_property_values(data_info)
            raise Client.NoResult("there is no dataset meeting the requirements, the following"
                                  f" property values are available for {data_type}: {properties}")
        return jarr[0]

    def get_dataset_info_by_uuid(self, uuid):
        """Returns the data from 'https://climada.ethz.ch/data-api/v1/dataset/{uuid}' as
        DatasetInfo object.

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
        url = f'{self.url}/dataset/{uuid}'
        return DatasetInfo.from_json(self._request_200(url))

    def list_data_type_infos(self, data_type_group=None):
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
        url = f'{self.url}/data_type'
        params = {'data_type_group': data_type_group} \
            if data_type_group else {}
        return [DataTypeInfo(**jobj) for jobj in self._request_200(url, params=params)]

    def get_data_type_info(self, data_type):
        """Returns the metadata of the data type with the given name from the climada data API.

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
        url = f'{self.url}/data_type/{quote(data_type)}'
        return DataTypeInfo(**self._request_200(url))

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
        with requests.get(url, stream=True, timeout=Client.DOWNLOAD_TIMEOUT) as stream:
            stream.raise_for_status()
            with open(path, 'wb') as dump:
                for chunk in stream.iter_content(chunk_size=self.chunk_size):
                    dump.write(chunk)
        return path

    def _tracked_download(self, remote_url, local_path):
        if local_path.is_dir():
            raise ValueError("tracked download requires a path to a file not a directory")
        path_as_str = str(local_path.absolute())
        try:
            dlf = Download.create(url=remote_url,
                                  path=path_as_str,
                                  startdownload=datetime.utcnow())
        except IntegrityError as ierr:
            dlf = Download.get(Download.path==path_as_str)  # path is the table's one unique column
            if not Path(path_as_str).is_file():  # in case the file has been removed
                dlf.delete_instance()  # delete entry from database
                return self._tracked_download(remote_url, local_path)  # and try again
            if dlf.url != remote_url:
                raise RuntimeError(f"this file ({path_as_str}) has been downloaded from another"
                                f" url ({dlf.url}), possibly because it belongs to a dataset with"
                                " a recent version update. Please remove the file or purge the"
                                " entry from data base before trying again") from ierr
            return dlf
        try:
            self._download(url=remote_url, path=local_path, replace=True)
            dlf.enddownload = datetime.utcnow()
            dlf.save()
        except Exception:
            dlf.delete_instance()
            raise
        return Download.get(Download.path==path_as_str)

    def _download_file(self, local_path, fileinfo, check=checksize, retries=3):
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

        Returns
        -------
        Path
            the path to the downloaded file

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
                                      " or remove cache entry by calling"
                                      f" `Client.purge_cache(Path('{local_path}'))`!")
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
            time.sleep(Client.MAX_WAITING_PERIOD/retries)
            return self._download_file(local_path=local_path, fileinfo=fileinfo, check=check,
                                       retries=retries - 1)

    def download_dataset(self, dataset, target_dir=SYSTEM_DIR, organize_path=True):
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

        Returns
        -------
        download_dir : Path
            the path to the directory containing the downloaded files,
            will be created if organize_path is True
        downloaded_files : list of Path
            the downloaded files themselves

        Raises
        ------
        Exception
            when one of the files cannot be downloaded
        """
        if not target_dir.is_dir():
            raise ValueError(f"{target_dir} is not a directory")

        if organize_path:
            target_dir = self._organize_path(dataset, target_dir)

        return target_dir, [
            self._download_file(local_path=target_dir, fileinfo=dsfile)
            for dsfile in dataset.files
        ]

    @staticmethod
    def _organize_path(dataset, target_dir):
        if dataset.data_type.data_type_group:
            target_dir /= dataset.data_type.data_type_group
        if dataset.data_type.data_type_group != dataset.data_type.data_type:
            target_dir /= dataset.data_type.data_type
        target_dir /= dataset.name
        if dataset.version:
            target_dir /= dataset.version
        target_dir.mkdir(exist_ok=True, parents=True)
        return target_dir

    @staticmethod
    def purge_cache(local_path):
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

    @staticmethod
    def _multi_version(datasets):
        ddf = pd.DataFrame(datasets)
        gdf = ddf.groupby('name').agg({'version': 'nunique'})
        return list(gdf[gdf.version > 1].index)

    def get_hazard(self, hazard_type, name=None, version=None, properties=None,
                   status='active', dump_dir=SYSTEM_DIR):
        """Queries the data api for hazard datasets of the given type, downloads associated
        hdf5 files and turns them into a climada.hazard.Hazard object.

        Parameters
        ----------
        hazard_type : str
            Type of climada hazard.
        name : str, optional
            the name of the dataset
        version : str, optional
            the version of the dataset
            Default: newest version meeting the requirements
        properties : dict, optional
            search parameters for dataset properties, by default None
            any property has a string for key and can be a string or a list of strings for value
        status : str, optional
            valid values are 'preliminary', 'active', 'expired', 'test_dataset', None
            by default 'active'
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR
            If the directory is the SYSTEM_DIR (as configured in
            climada.conf, i.g. ~/climada/data), the eventual target directory is organized into
            dump_dir > hazard_type > dataset name > version

        Returns
        -------
        climada.hazard.Hazard
            The combined hazard object
        """
        if not hazard_type in HAZ_TYPES:
            raise ValueError("Valid hazard types are a subset of CLIMADA hazard types."
                             f" Currently these types are supported: {HAZ_TYPES}")
        dataset = self.get_dataset_info(data_type=hazard_type, name=name, version=version,
                                        properties=properties, status=status)
        return self.to_hazard(dataset, dump_dir)

    def to_hazard(self, dataset, dump_dir=SYSTEM_DIR):
        """Downloads hdf5 files belonging to the given datasets reads them into Hazards and
        concatenates them into a single climada.Hazard object.

        Parameters
        ----------
        dataset : DatasetInfo
            Dataset to download and read into climada.Hazard object.
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR (as configured in
            climada.conf, i.g. ~/climada/data).
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > hazard_type > dataset name > version

        Returns
        -------
        climada.hazard.Hazard
            The combined hazard object
        """
        target_dir = self._organize_path(dataset, dump_dir) \
                     if dump_dir == SYSTEM_DIR else dump_dir
        hazard_list = [
            Hazard.from_hdf5(self._download_file(target_dir, dsf))
            for dsf in dataset.files
            if dsf.file_format == 'hdf5'
        ]
        if not hazard_list:
            raise ValueError("no hdf5 files found in dataset")
        if len(hazard_list) == 1:
            return hazard_list[0]
        hazard_concat = Hazard()
        hazard_concat = hazard_concat.concat(hazard_list)
        hazard_concat.sanitize_event_ids()
        hazard_concat.check()
        return hazard_concat

    def get_exposures(self, exposures_type, name=None, version=None, properties=None,
                   status='active', dump_dir=SYSTEM_DIR):
        """Queries the data api for exposures datasets of the given type, downloads associated
        hdf5 files and turns them into a climada.entity.exposures.Exposures object.

        Parameters
        ----------
        exposures_type : str
            Type of climada exposures.
        name : str, optional
            the name of the dataset
        version : str, optional
            the version of the dataset
            Default: newest version meeting the requirements
        properties : dict, optional
            search parameters for dataset properties, by default None
            any property has a string for key and can be a string or a list of strings for value
        status : str, optional
            valid values are 'preliminary', 'active', 'expired', 'test_dataset', None
            by default 'active'
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > hazard_type > dataset name > version

        Returns
        -------
        climada.entity.exposures.Exposures
            The combined exposures object
        """
        if not exposures_type in EXP_TYPES:
            raise ValueError("Valid exposures types are a subset of CLIMADA exposures types."
                             f" Currently these types are supported: {EXP_TYPES}")
        dataset = self.get_dataset_info(data_type=exposures_type, name=name, version=version,
                                        properties=properties, status=status)
        return self.to_exposures(dataset, dump_dir)

    def to_exposures(self, dataset, dump_dir=SYSTEM_DIR):
        """Downloads hdf5 files belonging to the given datasets reads them into Exposures and
        concatenates them into a single climada.Exposures object.

        Parameters
        ----------
        dataset : DatasetInfo
            Dataset to download and read into climada.Exposures objects.
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR (as configured in
            climada.conf, i.g. ~/climada/data).
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > exposures_type > dataset name > version

        Returns
        -------
        climada.entity.exposures.Exposures
            The combined exposures object
        """
        target_dir = self._organize_path(dataset, dump_dir) \
                     if dump_dir == SYSTEM_DIR else dump_dir
        exposures_list = [
            Exposures.from_hdf5(self._download_file(target_dir, dsf))
            for dsf in dataset.files
            if dsf.file_format == 'hdf5'
        ]
        if not exposures_list:
            raise ValueError("no hdf5 files found in dataset")
        if len(exposures_list) == 1:
            return exposures_list[0]
        exposures_concat = Exposures()
        exposures_concat = exposures_concat.concat(exposures_list)
        exposures_concat.check()
        return exposures_concat

    def get_litpop(self, country=None, exponents=(1,1), version=None, dump_dir=SYSTEM_DIR):
        """Get a LitPop ``Exposures`` instance on a 150arcsec grid with the default parameters:
        exponents = (1,1) and fin_mode = 'pc'.

        Parameters
        ----------
        country : str, optional
            Country name or iso3 codes for which to create the LitPop object.
            For creating a LitPop object over multiple countries, use ``get_litpop`` individually
            and concatenate using ``LitPop.concat``, see Examples.
            If country is None a global LitPop instance is created. Defaut is None.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop. To get
            nightlights^3 without population count: (3, 0).
            To use population count alone: (0, 1).
            Default: (1, 1)
        version : str, optional
            the version of the dataset
            Default: newest version meeting the requirements
        dump_dir : str
            directory where the files should be downoladed. Default: SYSTEM_DIR

        Returns
        -------
        climada.entity.exposures.Exposures
            default litpop Exposures object

        Examples
        --------
        Combined default LitPop object for Austria and Switzerland:

        >>> client = Client()
        >>> litpop_aut = client.get_litpop("AUT")
        >>> litpop_che = client.get_litpop("CHE")
        >>> litpop_comb = LitPop.concat([litpop_aut, litpop_che])
        """
        properties = {
            'exponents': "".join(['(',str(exponents[0]),',',str(exponents[1]),')'])}
        if country is None:
            properties['spatial_coverage'] = 'global'
        elif isinstance(country, str):
            properties['country_name'] = pycountry.countries.lookup(country).name
        elif isinstance(country, list):
            if len(set(country)) > 1:
                raise ValueError("``get_litpop`` can only query single countries. Download the"
                                 " data for multiple countries individually and concatenate the"
                                 " objects using ``LitPop.concat``")
            properties['country_name'] = [pycountry.countries.lookup(c).name for c in country]
        else:
            raise ValueError("country must be string")
        return self.get_exposures(exposures_type='litpop', properties=properties, version=version,
                                  dump_dir=dump_dir)

    def get_centroids(self, res_arcsec_land=150, res_arcsec_ocean=1800,
                      extent=(-180, 180, -60, 60), country=None, version=None,
                      dump_dir=SYSTEM_DIR):
        """Get centroids from teh API

        Parameters
        ----------
        res_land_arcsec : int
            resolution for land centroids in arcsec. Default is 150
        res_ocean_arcsec : int
            resolution for ocean centroids in arcsec. Default is 1800
        country : str
            country name, numeric code or iso code based on pycountry. Default is None (global).
        extent : tuple
            Format (min_lon, max_lon, min_lat, max_lat) tuple.
            If min_lon > lon_max, the extend crosses the antimeridian and is
            [lon_max, 180] + [-180, lon_min]
            Borders are inclusive. Default is (-180, 180, -60, 60).
        version : str, optional
            the version of the dataset
            Default: newest version meeting the requirements
        dump_dir : str
            directory where the files should be downoladed. Default: SYSTEM_DIR
        Returns
        -------
        climada.hazard.centroids.Centroids
            Centroids from the api
        """

        properties = {
            'res_arcsec_land': str(res_arcsec_land),
            'res_arcsec_ocean': str(res_arcsec_ocean),
            'extent': '(-180, 180, -90, 90)'
        }
        dataset = self.get_dataset_info('centroids', version=version, properties=properties)
        target_dir = self._organize_path(dataset, dump_dir) \
                     if dump_dir == SYSTEM_DIR else dump_dir
        centroids = Centroids.from_hdf5(self._download_file(target_dir, dataset.files[0]))
        if country:
            reg_id = pycountry.countries.lookup(country).numeric
            centroids = centroids.select(reg_id=int(reg_id), extent=extent)
        if extent:
            centroids = centroids.select(extent=extent)

        return centroids

    @staticmethod
    def get_property_values(dataset_infos, known_property_values=None,
                            exclude_properties=None):
        """Returns a dictionnary of possible values for properties of a data type, optionally given
        known property values.

        Parameters
        ----------
        dataset_infos : list of DataSetInfo
            as returned by list_dataset_infos
        known_properties_value : dict, optional
            dict {'property':'value1, 'property2':'value2'}, to provide only a subset of property
            values that can be combined with the given properties.
        exclude_properties: list of str, optional
            properties in this list will be excluded from the resulting dictionary, e.g., because
            they are strictly metadata and don't provide any information essential to the dataset.
            Default: 'creation_date', 'climada_version'

        Returns
        -------
        dict
            of possibles property values
        """
        if exclude_properties is None:
            exclude_properties = ['date_creation', 'climada_version']

        ppdf = pd.DataFrame([ds.properties for ds in dataset_infos])
        if known_property_values:
            for key, val in known_property_values.items():
                ppdf = ppdf[ppdf[key] == val]

        property_values = dict()
        for col in ppdf.columns:
            if col in exclude_properties:
                continue
            valar = ppdf[col].dropna().drop_duplicates().values
            if valar.size:
                property_values[col] = list(valar)
        return property_values

    @staticmethod
    def into_datasets_df(dataset_infos):
        """Convenience function providing a DataFrame of datasets with properties.

        Parameters
        ----------
        dataset_infos : list of DatasetInfo
             as returned by list_dataset_infos

        Returns
        -------
        pandas.DataFrame
            of datasets with properties as found in query by arguments
        """
        dsdf = pd.DataFrame(dataset_infos)
        ppdf = pd.DataFrame([ds.properties for ds in dataset_infos])
        dtdf = pd.DataFrame([pd.Series(dt) for dt in dsdf.data_type])

        return dtdf.loc[:, [c for c in dtdf.columns
                            if c not in ['description', 'properties']]].join(
               dsdf.loc[:, [c for c in dsdf.columns
                            if c not in ['data_type', 'properties', 'files']]]).join(
               ppdf)

    @staticmethod
    def into_files_df(dataset_infos):
        """Convenience function providing a DataFrame of files aligned with the input datasets.

        Parameters
        ----------
        datasets : list of DatasetInfo
            as returned by list_dataset_infos

        Returns
        -------
        pandas.DataFrame
            of the files' informations including dataset informations
        """
        return Client.into_datasets_df(dataset_infos) \
            .merge(pd.DataFrame([dsfile for ds in dataset_infos for dsfile in ds.files]))
