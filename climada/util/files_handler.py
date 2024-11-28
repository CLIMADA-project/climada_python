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
    "to_list",
    "get_file_names",
]

import glob
import hashlib
import logging
import math
import time
import urllib
from datetime import datetime, timezone
from pathlib import Path

import requests
from peewee import (
    AutoField,
    CharField,
    DateTimeField,
    IntegerField,
    IntegrityError,
    Model,
    SqliteDatabase,
)
from playhouse.migrate import SqliteMigrator, migrate
from tqdm import tqdm

from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

DB = SqliteDatabase(Path(CONFIG.data_api.cache_db.str()).expanduser())


class Download(Model):
    """Database entry keeping track of downloaded files from the CLIMADA data API"""

    # If you do not specify a primary key, Peewee will automatically create an auto-incrementing
    # primary key named “id”.
    url = CharField()
    path = CharField(unique=True)
    startdownload = DateTimeField()
    enddownload = DateTimeField(null=True)
    timestamp = IntegerField(null=True)
    filesize = IntegerField(null=True)
    checksum = CharField(null=True)

    class Meta:
        """SQL database and table definition."""

        database = DB

    class Failed(Exception):
        """The download failed for some reason."""


def init_DB():
    DB.connect()
    DB.create_tables([Download])
    try:
        migrator = SqliteMigrator(DB)
        migrate(
            migrator.add_column("download", "checksum", CharField(null=True)),
            migrator.add_column("download", "timestamp", IntegerField(null=True)),
            migrator.add_column("download", "filesize", IntegerField(null=True)),
        )
    except Exception as exc:
        print(exc)
    DB.close()


init_DB()

HASH_FUNCS = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
}


def file_checksum(filename, hash_func):
    hf = HASH_FUNCS[hash_func]()
    ba = bytearray(128 * 1024)
    mv = memoryview(ba)
    with open(filename, "rb", buffering=0) as f:
        for n in iter(lambda: f.readinto(mv), 0):
            hf.update(mv[:n])
    hashsum = hf.hexdigest()
    return f"{hash_func}:{hashsum}"


class Downloader:

    MAX_WAITING_PERIOD = 6
    DOWNLOAD_TIMEOUT = 3600

    def __init__(self, checksum=None):
        """Downloader

        Parameters
        ----------
        checksum : {'md5', 'sha1}, optional
            if not None, the checksum of downloaded files are being recorded.
            Default: None
        """
        self.chunk_size = CONFIG.data_api.chunk_size.int()
        self.checksum = checksum

    def _tracked_download(self, remote_url, local_path):
        if local_path.is_dir():
            raise ValueError(
                "tracked download requires a path to a file not a directory"
            )
        path_as_str = str(local_path.absolute())
        try:
            dlf = Download.create(
                url=remote_url, path=path_as_str, startdownload=datetime.utcnow()
            )
        except IntegrityError as ierr:
            dlf = Download.get(
                Download.path == path_as_str
            )  # path is the table's one unique column
            if not Path(path_as_str).is_file():  # in case the file has been removed
                dlf.delete_instance()  # delete entry from database
                return self._tracked_download(remote_url, local_path)  # and try again
            if dlf.url != remote_url:
                raise RuntimeError(
                    f"this file ({path_as_str}) has been downloaded from another url ({dlf.url}),"
                    " possibly because it belongs to a dataset with a recent version update."
                    " Please remove the file or purge the entry from data base before trying again"
                ) from ierr
            return dlf
        try:
            # the actual download from url, using requests.get method
            with requests.get(
                remote_url, stream=True, timeout=Downloader.DOWNLOAD_TIMEOUT
            ) as stream:
                stream.raise_for_status()
                with open(local_path, "wb") as dump:
                    for chunk in stream.iter_content(chunk_size=self.chunk_size):
                        dump.write(chunk)
            # update the db entry
            dlf.enddownload = datetime.now(timezone.utc)
            dlf.timestamp = int(local_path.stat().st_mtime)
            dlf.filesize = local_path.stat().st_size
            if self.checksum:
                dlf.checksum = file_checksum(local_path, self.checksum)
            dlf.save()
        except Exception:
            dlf.delete_instance()
            raise
        return Download.get(Download.path == path_as_str)

    def download(self, url, target_dir, file_name, integrity_check=None):
        """Download a file if it is not already present at the target destination.

        Parameters
        ----------
        local_path : Path
            target destination,
            if it is a directory the original filename (fileinfo.filen_name) is kept
        fileinfo : FileInfo
            file object as retrieved from the data api
        integrity_check : function(path) -> (), optional
            the method that is used to check the integrity of the already downloaded file.
            expected to raise a `Download.Failed` exception if it wasn't successfull.
            Default: `Downloader.file_unchanged_since`.

        Returns
        -------
        Path
            the path to the downloaded file

        Raises
        ------
        Exception
            when number of retries was exceeded or when a download is already running
        """
        return self._download_file(
            url=url,
            target_dir=target_dir,
            file_name=file_name,
            check_method=integrity_check or self.file_unchanged_since,
            retries=3,
        )

    def _download_file(self, url, target_dir, file_name, check_method, retries):
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
            local_path = Path(target_dir, file_name)
            downloaded = self._tracked_download(remote_url=url, local_path=local_path)
            if not downloaded.enddownload:
                raise Download.Failed(
                    f"A download of {url} via the climada.util.files_handler.Downloader has been"
                    " requested before. Either it is still in progress or the process got"
                    " interrupted. In the former case just wait until the download has finished"
                    " and try again, in the latter run `Downloader.purge_cache_db(Path('"
                    f"{local_path}'))` from Python. If unsure, check your internet connection,"
                    " wait for as long as it takes to download a file of the given size and try"
                    " again. If the problem persists, purge the cache db with said call."
                )
            try:
                check_method(local_path)
            except Download.Failed as dlf:
                local_path.unlink(missing_ok=True)
                self.purge_cache_db(local_path)
                raise dlf
            return local_path
        except Download.Failed as dle:
            if retries < 1:
                raise dle
            LOGGER.warning("Download failed: %s, retrying...", dle)
            time.sleep(Downloader.MAX_WAITING_PERIOD / retries)
            return self._download_file(
                url=url,
                target_dir=target_dir,
                file_name=file_name,
                check_method=check_method,
                retries=retries - 1,
            )

    @staticmethod
    def purge_cache_db(local_path):
        """Removes entry from the sqlite database that keeps track of files downloaded by
        `_tracked_download`. This may be necessary in case a previous attempt has failed
        in an uncontroled way (power outage or the like).

        Parameters
        ----------
        local_path : Path
            target destination
        fileinfo : FileInfo
            file object as retrieved from the data api
        """
        dlf = Download.get(Download.path == str(local_path.absolute()))
        dlf.delete_instance()

    def file_unchanged_since(self, local_path):
        """default method for checking file integrity
        basically checks whether the file has changed since the download

        Parameters
        ----------
        local_path : Path
            the file whose integrity is being scrutinized

        Raises
        ------
        Download.Failed
            if the file is not there or doesn't have the same size anymore
            or the same timestamp, or the same checksum in case checksums are recorded
        """
        if not local_path.is_file():
            raise Download.Failed(f"{str(local_path)} is not a file")
        dlf = Download.get(Download.path == str(local_path.absolute()))
        if local_path.stat().st_size != dlf.filesize:
            raise Download.Failed(
                f"{str(local_path)} has the wrong size:"
                f"{local_path.stat().st_size} instead of {dlf.filesize}"
            )
        if self.checksum:
            csm, hsh = dlf.checksum.split(":")
            found = file_checksum(local_path, csm)
            if hsh != found:
                raise Download.Failed(
                    f"{str(local_path)} has changed, checksums differ: "
                    f"{csm}:{hsh}, {csm}:{found}"
                )
        else:  # if the checksum is still the same, there is no point comparing timestamps!
            # even if it has changed, the file has _certainly_ just been touched.
            if int(local_path.stat().st_mtime) != dlf.timestamp:
                expected = datetime.fromtimestamp(dlf.timestamp).isoformat()
                found = datetime.fromtimestamp(
                    int(local_path.stat().st_mtime)
                ).isoformat()
                raise Download.Failed(
                    f"{str(local_path)} has been modified: timestamp changed from"
                    f"{expected} to {found}"
                )


class DownloadProgressBar(tqdm):
    """Class to use progress bar during dowloading"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """Update progress bar

        Parameters
        ----------
        blocks : int, optional
            Number of blocks transferred so far [default: 1].
        bsize : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize : int, optional
            Total size (in tqdm units). If [default: None]
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
        default: local_data.save_dir as defined in climada.conf
    overwrite : bool, optional
        whether or not an already existing file at the target location should be overwritten,
        by default True

    Returns
    -------
    str
        the full path to the eventually downloaded file
    """
    file_name = url.split("/")[-1]
    if file_name.strip() == "":
        raise ValueError(f"cannot download {url} as a file")
    download_path = (
        CONFIG.local_data.save_dir.dir() if download_dir is None else Path(download_dir)
    )
    file_path = download_path.absolute().joinpath(file_name)
    if file_path.exists():
        if not file_path.is_file() or not overwrite:
            raise FileExistsError(f"cannot download to {file_path}")

    try:
        req_file = requests.get(url, stream=True)
    except IOError as ioe:
        raise type(ioe)("Check URL and internet connection: " + str(ioe)) from ioe
    if req_file.status_code < 200 or req_file.status_code > 299:
        raise ValueError(
            f"Error loading page {url}\n"
            f" Status: {req_file.status_code}\n"
            f" Content: {req_file.content}"
        )

    total_size = int(req_file.headers.get("content-length", 0))
    block_size = 1024

    LOGGER.info("Downloading %s to file %s", url, file_path)
    with file_path.open("wb") as file:
        for data in tqdm(
            req_file.iter_content(block_size),
            total=math.ceil(total_size // block_size),
            unit="KB",
            unit_scale=True,
        ):
            file.write(data)

    return str(file_path)


def download_ftp(url, file_name):
    """Download file from ftp in current folder.

    Parameters
    ----------
    url : str
        url containing data to download
    file_name : str
        name of the file to dowload

    Raises
    ------
    ValueError
    """
    LOGGER.info("Downloading file %s", file_name)
    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as prog_bar:
            urllib.request.urlretrieve(url, file_name, reporthook=prog_bar.update_to)
    except Exception as exc:
        raise ValueError(
            f'{exc.__class__} - "{exc}": failed to retrieve {url} into {file_name}'
        ) from exc


def to_list(num_exp, values, val_name):
    """Check size and transform to list if necessary. If size is one, build
    a list with num_exp repeated values.

    Parameters
    ----------
    num_exp : int
        expected number of list elements
    values : object or list(object)
        values to check and transform
    val_name : str
        name of the variable values

    Returns
    -------
    list
    """
    if not isinstance(values, list):
        return num_exp * [values]
    if len(values) == num_exp:
        return values
    if len(values) == 1:
        return num_exp * [values[0]]
    raise ValueError(f"Provide one or {num_exp} {val_name}.")


def get_file_names(file_name):
    """Return list of files contained. Supports globbing.

    Parameters
    ----------
    file_name : str or list(str)
        Either a single string or a list of
        strings that are either
        - a file path
        - or the path of the folder containing the files
        - or a globbing pattern.

    Returns
    -------
    list(str)
    """
    pattern_list = file_name if isinstance(file_name, list) else [file_name]
    pattern_list = [Path(pattern) for pattern in pattern_list]

    file_list = []
    for pattern in pattern_list:
        if pattern.is_file():
            file_list.append(str(pattern))
        elif pattern.is_dir():
            extension = [str(fil) for fil in pattern.iterdir() if fil.is_file()]
            if not extension:
                raise ValueError(f'there are no files in directory "{pattern}"')
            file_list.extend(extension)
        else:  # glob pattern
            extension = [fil for fil in glob.glob(str(pattern)) if Path(fil).is_file()]
            if not extension:
                raise ValueError(f'cannot find the file "{pattern}"')
            file_list.extend(extension)
    return file_list
