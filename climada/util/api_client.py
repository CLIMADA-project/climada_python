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

import pandas as pd
from peewee import CharField, DateTimeField, IntegrityError, Model, SqliteDatabase
import requests
import pycountry

from climada import CONFIG
from climada.entity import Exposures
from climada.hazard import Hazard
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

DB = SqliteDatabase(Path(CONFIG.data_api.cache_db.str()).expanduser())

HAZ_TYPES = [ht.str() for ht in CONFIG.data_api.supported_hazard_types.list()]
EXP_TYPES = [et.str() for et in CONFIG.data_api.supported_exposures_types.list()]
MUTUAL_PROPS = [ms.str() for ms in CONFIG.data_api.mutual_properties.list()]


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
            any property has a string for key and can be a string or a list of strings for value
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

        if properties:
            straight_props, multi_props = self._divide_straight_from_multi(properties)
        else:
            straight_props, multi_props = None, None

        if straight_props:
            params.update(straight_props)

        datasets = [DatasetInfo.from_json(ds) for ds in Client._request_200(url, params=params)]

        if multi_props:
            return self._filter_datasets(datasets, multi_props)
        return datasets

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
            any property has a string for key and can be a string or a list of strings for value
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
                    f" or remove cache entry by calling `purge_cache(Path('{local_path}'))`!")
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
            target_dir = self._organize_path(dataset, target_dir)

        return target_dir, [
            self.download_file(local_path=target_dir, fileinfo=dsfile, check=check)
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

    @staticmethod
    def _multi_selection(datasets):
        pdf = pd.DataFrame([ds.properties for ds in datasets]).nunique()
        return list(pdf[pdf > 1].index)

    @staticmethod
    def _check_datasets_for_concatenation(datasets, max_datasets):
        if not datasets:
            raise ValueError("no datasets found meeting the requirements")
        if 0 < max_datasets < len(datasets):
            raise ValueError(f"There are {len(datasets)} datasets matching the query"
                             f" and the limit is set to {max_datasets}.\n"
                             "You can force concatenation of multiple datasets by increasing"
                             " max_datasets or setting it to a value <= 0 (no limit).\n"
                             "Attention! For hazards, concatenation of datasets is currently done by"
                             " event, which may lead to event duplication and thus biased data.\n"
                             "In a future release this limitation will be overcome.")
        not_supported = [msd for msd in Client._multi_selection(datasets)
                         if msd in MUTUAL_PROPS]
        if not_supported:
            raise ValueError("Cannot combine datasets, there are distinct values for these"
                             f" properties in your selection: {not_supported}")
        ambiguous_ds_names = Client._multi_version(datasets)
        if ambiguous_ds_names:
            raise ValueError("There are datasets with multiple versions in your selection:"
                             f" {ambiguous_ds_names}")

    def get_hazard(self, hazard_type, dump_dir=SYSTEM_DIR, max_datasets=1, **kwargs):
        """Queries the data api for hazard datasets of the given type, downloads associated
        hdf5 files and turns them into a climada.hazard.Hazard object.

        Parameters
        ----------
        hazard_type : str
            Type of climada hazard.
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > hazard_type > dataset name > version
        max_datasets : int, optional
            Download limit for datasets. If a query matches is matched by more datasets than this
            number, a ValueError is raised. Setting it to 0 or a negative number inactivates the
            limit. Default is 10.
        **kwargs :
            additional parameters passed on to get_datasets

        Returns
        -------
        climada.hazard.Hazard
            The combined hazard object
        """
        if 'data_type' in kwargs:
            raise ValueError("data_type is already given as hazard_type")
        if not hazard_type in HAZ_TYPES:
            raise ValueError("Valid hazard types are a subset of CLIMADA hazard types."
                             f" Currently these types are supported: {HAZ_TYPES}")
        datasets = self.get_datasets(data_type=hazard_type, **kwargs)

        self._check_datasets_for_concatenation(datasets, max_datasets)

        return self.to_hazard(datasets, dump_dir)

    def to_hazard(self, datasets, dump_dir=SYSTEM_DIR):
        """Downloads hdf5 files belonging to the given datasets reads them into Hazards and
        concatenates them into a single climada.Hazard object.

        Parameters
        ----------
        datasets : list of DatasetInfo
            Datasets to download and read into climada.Hazard objects.
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > hazard_type > dataset name > version

        Returns
        -------
        climada.hazard.Hazard
            The combined hazard object
        """
        hazard_list = []
        for dataset in datasets:
            target_dir = self._organize_path(dataset, dump_dir) \
                         if dump_dir == SYSTEM_DIR else dump_dir
            for dsf in dataset.files:
                if dsf.file_format == 'hdf5':
                    hazard_file = self.download_file(target_dir, dsf)
                    hazard = Hazard()
                    hazard.read_hdf5(hazard_file)
                    hazard_list.append(hazard)
        if not hazard_list:
            raise ValueError("no hazard files found in datasets")

        hazard_concat = Hazard()
        hazard_concat = hazard_concat.concat(hazard_list)
        hazard_concat.sanitize_event_ids()
        hazard.check()
        return hazard_concat

    def get_exposures(self, exposures_type, dump_dir=SYSTEM_DIR, max_datasets=10, **kwargs):
        """Queries the data api for exposures datasets of the given type, downloads associated
        hdf5 files and turns them into a climada.entity.exposures.Exposures object.

        Parameters
        ----------
        hazard_type : str
            Type of climada exposures.
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > hazard_type > dataset name > version
        max_datasets : int, optional
            Download limit for datasets. If a query matches is matched by more datasets than this
            number, a ValueError is raised. Setting it to 0 or a negative number inactivates the
            limit. Default is 10.
        **kwargs :
            additional parameters passed on to `Client.get_datasets`

        Returns
        -------
        climada.entity.exposures.Exposures
            The combined exposures object
        """
        if 'data_type' in kwargs:
            raise ValueError("data_type is already given as hazard_type")
        if not exposures_type in EXP_TYPES:
            raise ValueError("Valid exposures types are a subset of CLIMADA exposures types."
                             f" Currently these types are supported: {EXP_TYPES}")
        datasets = self.get_datasets(data_type=exposures_type, **kwargs)

        self._check_datasets_for_concatenation(datasets, max_datasets)

        return self.to_exposures(datasets, dump_dir)

    def to_exposures(self, datasets, dump_dir=SYSTEM_DIR):
        """Downloads hdf5 files belonging to the given datasets reads them into Exposures and
        concatenates them into a single climada.Exposures object.

        Parameters
        ----------
        datasets : list of DatasetInfo
            Datasets to download and read into climada.Exposures objects.
        dump_dir : str, optional
            Directory where the files should be downoladed. Default: SYSTEM_DIR
            If the directory is the SYSTEM_DIR, the eventual target directory is organized into
            dump_dir > exposures_type > dataset name > version

        Returns
        -------
        climada.entity.exposures.Exposures
            The combined exposures object
        """
        exposures_list = []
        for dataset in datasets:
            target_dir = self._organize_path(dataset, dump_dir) \
                         if dump_dir == SYSTEM_DIR else dump_dir
            for dsf in dataset.files:
                if dsf.file_format == 'hdf5':
                    exposures_file = self.download_file(target_dir, dsf)
                    exposures = Exposures.from_hdf5(exposures_file)
                    exposures_list.append(exposures)
        if not exposures_list:
            raise ValueError("no exposures files found in datasets")

        exposures_concat = Exposures()
        exposures_concat = exposures_concat.concat(exposures_list)
        exposures_concat.check()
        return exposures_concat

    def get_litpop_default(self, country=None, dump_dir=SYSTEM_DIR):
        """Get a LitPop instance on a 150arcsec grid with the default parameters:
        exponents = (1,1) and fin_mode = 'pc'.

        Parameters
        ----------
        country : str or list, optional
            List of country name or iso3 codes for which to create the LitPop object.
            If None is given, a global LitPop instance is created. Defaut is None
        dump_dir : str
            directory where the files should be downoladed. Default: SYSTEM_DIR

        Returns
        -------
        climada.entity.exposures.Exposures
            default litpop Exposures object
        """
        properties = {
            'exponents': '(1,1)',
            'fin_mode': 'pc'
        }
        if country is None:
            properties['geographical_scale'] = 'global'
        elif isinstance(country, str):
            properties['country_name'] = pycountry.countries.lookup(country).name
        elif isinstance(country, list):
            properties['country_name'] = [pycountry.countries.lookup(c).name for c in country]
        else:
            raise ValueError("country must be string or list of strings")
        return self.get_exposures(exposures_type='litpop', dump_dir=dump_dir, properties=properties)

    @staticmethod
    def into_datasets_df(datasets):
        """Convenience function providing a DataFrame of datasets with properties.

        Parameters
        ----------
        datasets : list of DatasetInfo
            e.g., return of get_datasets

        Returns
        -------
        pandas.DataFrame
            of datasets with properties as found in query by arguments
        """
        dsdf = pd.DataFrame(datasets)
        ppdf = pd.DataFrame([ds.properties for ds in datasets])
        dtdf = pd.DataFrame([pd.Series(dt) for dt in dsdf.data_type])

        return dtdf.loc[:, [c for c in dtdf.columns if c not in ['description', 'properties']]].join(
               dsdf.loc[:, [c for c in dsdf.columns if c not in ['data_type', 'properties', 'files']]]).join(
               ppdf)

    @staticmethod
    def into_files_df(datasets):
        """Convenience function providing a DataFrame of files aligned with the input datasets.

        Parameters
        ----------
        datasets : list of DatasetInfo
            e.g., return of get_datasets

        Returns
        -------
        pandas.DataFrame
            of the files' informations including dataset informations
        """
        return Client.into_datasets_df(datasets) \
            .merge(pd.DataFrame([dsfile for ds in datasets for dsfile in ds.files]))
