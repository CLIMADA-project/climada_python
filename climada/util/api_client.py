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
import os.path
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from urllib.parse import quote, unquote
import time

import numpy as np
from peewee import CharField, DateTimeField, IntegrityError, Model, SqliteDatabase
import requests


from climada import CONFIG
from climada.entity import Exposures
from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

DB = SqliteDatabase(Path(CONFIG.data_api.cache_db.str()).expanduser())

HAZ_TYPES = ['river_flood', 'tropical_cyclone', 'storm_europe']
EXP_TYPES = ['litpop', 'crop_production']


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
    """data_type data from CLIMADA data API."""
    data_type:str
    data_type_group:str
    description:str
    properties:list = None


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
        dataset.data_type = DataTypeInfo(**dataset.data_type)
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
            if dataset.data_type.data_type_group:
                target_dir /= dataset.data_type.data_type_group
            if dataset.data_type.data_type_group != dataset.data_type.data_type:
                target_dir /= dataset.data_type.data_type
            target_dir /= dataset.name
            if dataset.version:
                target_dir /= dataset.version
            target_dir.mkdir(exist_ok=True, parents=True)

        return target_dir, [
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

    def get_hazard(self, hazard_type=None, data_dir=SYSTEM_DIR):
        """Provides options to chose a hazard dataset, saves the file locally and opens it as a hazard.
        Several countries can be given, creating a hazard combining the single countries

        Parameters
        ----------
        hazard_type : str
            Type of climada Hazard. If None, options of available hazard types are given
        data_dir : str
            directory where the files should be downoladed. Default: SYSTEM_DIR
        """
        if not hazard_type:
            while True:
                hazard_type = input("The following data types are available: "
                                    + ", ".join(HAZ_TYPES) + ". Which one do you want to get?")
                if hazard_type in HAZ_TYPES:
                    break
                else:
                    LOGGER.error('Please give a valid value from the list provided.')
        datasets = self._get_data(hazard_type)

        hazard_list = []
        for dataset in datasets:
            if os.path.isfile(os.path.join(data_dir, dataset.files[0].file_name)):
                LOGGER.info('The file already exists and it was not downloaded again.')
            else:
                self.download_file(data_dir, dataset.files[0])
            hazard = Hazard()
            hazard.read_hdf5(os.path.join(data_dir, dataset.files[0].file_name))
            hazard_list.append(hazard)
        hazard_concat = Hazard()
        hazard_concat = hazard_concat.concat(hazard_list)
        return hazard_concat

    def get_exposures(self, exposures_type=None, data_dir=SYSTEM_DIR):
        """Provides options to chose a exposures dataset, saves the file locally and opens it as a climada Exposures.
        Several countries can be given, creating an exposure combining the single countries
        Parameters
        ----------
        exposures_type : str
            Type of climada Exposures. If None, options of available exposures types are given
        data_dir : str
            directory where the files should be downoladed. Default: SYSTEM_DIR
        """
        if not exposures_type:
            while True:
                exposures_type = input("The following exposures types are available: "
                                       + ", ".join(EXP_TYPES) + ". Which one would you like to get?")
                if exposures_type in EXP_TYPES:
                    break
                else:
                    LOGGER.error('Please give a valid value from the list provided.')
        datasets = self._get_data(exposures_type)
        exposures_list = []
        for dataset in datasets:
            self.download_file(data_dir, dataset.files[0])
            if os.path.isfile(os.path.join(data_dir, dataset.files[0].file_name)):
                LOGGER.info('The file already exists and it was not downloaded again.')
            exposures = Exposures()
            exposures.read_hdf5(os.path.join(data_dir, dataset.files[0].file_name))
            exposures_list.append(exposures)
        exposures_concat = Exposures()
        exposures_concat = exposures_concat.concat(exposures_list)
        return exposures_concat

    def get_litpot_default(self, country=None, data_dir=SYSTEM_DIR):
        """Get a LitPop instance on a 150arcsec grid with the default parameters: exponents:(1,1) and fin_mode='pc'.
          Parameters
          ----------
          country : list
              List of country for which to create the LitPop object. If None is given, a global LitPop instance
              is created
          data_dir : str
              directory where the files should be downoladed. Default: SYSTEM_DIR
          """
        if not country:
            datasets = [self.get_datasets(data_type='litpop', properties={'exponents': '(1,1)', 'fin_mode': 'pc'})]
        if country:
            country = list(country)
            datasets = [self.get_datasets(data_type='litpop', properties={'exponents': '(1,1)', 'fin_mode': 'pc',
                                                                             'country_iso3alpha': c}) for c in country]
        exposures_list = []
        for dataset in datasets:
            if os.path.isfile(os.path.join(data_dir, dataset.files[0])):
                LOGGER.info('The file already exists and it was not downloaded again.')
            self.download_file(data_dir, dataset.files[0])
            exposures = Exposures()
            exposures.read_hdf5(os.path.join(data_dir, dataset.files[0].file_name))
            exposures_list.append(exposures)
        exposures_concat = Exposures()
        exposures_concat = exposures_concat.concat(exposures_list)
        return exposures_concat

    def _get_data(self, type):
        datasets = self.get_datasets(data_type=type)
        properties_keys = np.unique([dataset.properties.keys() for dataset in datasets])
        properties_keys = set(properties_keys[0]).intersection(*properties_keys)
        user_properties_input = {}
        user_properties_input = self._select_properties(datasets, properties_keys, user_properties_input)
        datasets = self.get_datasets(data_type=type, properties=user_properties_input)
        properties_keys2 = set(np.unique([dataset.properties.keys() for dataset in datasets])[0]) - properties_keys
        user_properties_input.update(self._select_properties(datasets, properties_keys2, user_properties_input))
        try: # make a list of properties to get several datasets in case several countries are given
            user_properties_input['country_name']
            datasets_properties = []
            for country in user_properties_input['country_name']:
                properties = user_properties_input.copy()
                properties['country_name'] = country
                datasets_properties.append(properties)

        except:
            datasets_properties = [user_properties_input]
        datasets = [self.get_dataset(data_type=type, properties=properties) for properties in datasets_properties]
        return datasets

    def _select_properties(self, datasets, properties_keys, user_properties_input):
        for property_key in properties_keys:
            property_values = list(np.unique([dataset.properties[property_key] for dataset in datasets]))
            if property_key == 'date_creation' or  property_key == 'climada_version' \
                    or property_key == 'country_iso3alpha':
                continue
            if len(property_values) <= 1:
                continue
            while True:
                if property_key == 'country_name':
                    user_properties_input[property_key] = input(
                        "The following " + property_key + " are available: "
                        + ", ".join(property_values) + ". Which one(s) would you like to get? You can also provide "
                                                       "a list of countries separated by a coma").split(',')
                else:
                    user_properties_input[property_key] = [input(
                        "The following " + property_key + " are available: "
                        + ", ".join(property_values) + ". Which one would you like to get?")]

                if set(user_properties_input[property_key]).issubset(property_values):
                    break
                else:
                    LOGGER.error('Please give a valid value from the list provided.')
        return user_properties_input