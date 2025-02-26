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
    BooleanField,
    CharField,
    DateTimeField,
    IntegerField,
    Model,
    SqliteDatabase,
)
from playhouse.migrate import SqliteMigrator, migrate
from tqdm import tqdm

from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)


HASH_FUNCS = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
}


def file_checksum(file_name, hash_method):  # pylint: disable=invalid-name
    """Utiliity function calculating a checksum md5 or sha1 of a file.

    Parameters
    ----------
    file_name : str
        path to the file
    hash_method : {'md5', 'sha'}
        hash method

    Returns:
    --------
    str : formatted string, e.g. "md5:66358e7c618a1bafc2e4f04518cb4263"
    """
    hash_func = HASH_FUNCS[hash_method]()
    bytar = bytearray(128 * 1024)
    memv = memoryview(bytar)
    with open(file_name, "rb", buffering=0) as rbf:
        for chsize in iter(lambda: rbf.readinto(memv), 0):
            hash_func.update(memv[:chsize])
    hashsum = hash_func.hexdigest()
    return f"{hash_method}:{hashsum}"


class Download(Model):
    """Database entry keeping track of downloaded files from the CLIMADA data API"""

    # Since no primary key is specified, Peewee automatically creates an auto-incrementing
    # primary key named “id”.
    url = CharField()
    path = CharField(unique=True)
    startdownload = DateTimeField()

    # set after the download has been successfull
    enddownload = DateTimeField(null=True)
    timestamp = IntegerField(null=True)
    filesize = IntegerField(null=True)
    checksum = CharField(null=True)

    # set after a download failed or didn't match expected size or hecksum
    failed = BooleanField(default=False)
    # actually necessary to distinguish cases where a download is in progress and one should
    # just wait from cases where a previous download has failed. The former raises an error in
    # `Downloader.download()` the latter cleans up the database and starts another attempt

    def check(self, filesize=None, checksum=None):
        """Returns self if the Download object meets the expected file size and/or checksum

        Parameters
        ----------
        filesize : int, optional
            expected file size, by default None
        checksum : str, optional
            expected checksum,
            format: [md5|sha]:<checksum>, e.g., 'md5:18bd3d39c0671a8d6811fe6b99e140d2',
            by default None

        Returns
        -------
        Download or None
        """
        if filesize and filesize != self.filesize:
            LOGGER.warning(
                "the downloaded file does not have the expected file size: %d instead of %d",
                self.filesize,
                filesize,
            )
            return None
        if checksum:
            refsum = self.checksum or file_checksum(self.path, checksum.split(":"))
            if checksum != refsum:
                LOGGER.warning(
                    "the downloaded file does not have the expected file checksum:"
                    " %s instead of %s",
                    refsum,
                    checksum,
                )
                return None
        return self


class DownloadFailed(Exception):
    """The download failed for some reason."""


class Downloader:
    """Manages files that are downloaded from an url. The goal is to avoid downloading when
    the requested file has already been downloaded before.
    To do this, downloads are tracked in a sqlite database, where they are identified
    by the (local) path of the downloded file.

    Beside the target path and the source url, download start and end times are stored,
    file size, timestamp of the local file, and - if requested - a hashsum.
    Entries of failed download attempts are kept too and marked with a 'failed' flag.

    Files that have changed in size or timestamp (or optionally checksum) since they were
    downloaded first will be replaced with the next download.

    If a download from the same url to the same target path hast been successful before and
    size and timestamp are still the same, a subsequent download is skipped and the local file
    is taken instead.
    """

    class Check:
        """Collection of flags for checking whether a file that has been downloaded before is
        still valid
        """

        # just the flags that are used to check whether an old local file is still valid
        SYNC = 1
        """the local file will always be replaced"""
        TIMESTAMP = 2
        """the timestamps must be equal"""
        SIZE = 4
        """the sizes must be equal"""
        CHECKSUM = 8
        """the checksums must be equal"""

    MAX_WAITING_PERIOD = 6.0
    """Sets the limit in seconds for the downloader to wait for another download in progress to
    finish."""

    DOWNLOAD_TIMEOUT = 3600
    """How long the Downloader waits for another process to finish the download before giving up"""

    RETRIES = 3
    """The number of download attempts that are made before giving up"""

    def __init__(
        self,
        downloads_db: Path = None,
        checksum: str = None,
    ):
        """Constructor

        Parameters
        ----------
        downloads_db : Path
            Path to the sqlite db where the download records are stored.
            Default: the path configured as CONFIG.data_api.cache_db
        checksum : {'md5', 'sha1}, optional
            if not None, the checksum of downloaded files are being recorded,
            otherwise the calculation of checksums is skipped and thus cpu time saved.
            Default: None

        Raises
        ------
        DownloadFailed
            in case the file couldn't be downloaded
        """
        self.DB = SqliteDatabase(  # pylint: disable=invalid-name
            downloads_db or Path(CONFIG.data_api.cache_db.str()).expanduser()
        )
        self.chunk_size = CONFIG.data_api.chunk_size.int()
        self.checksum = checksum
        self._init_database()

    def __exit__(self, exc_type, exc_value, traceback):
        self.DB.close()

    def _init_database(self):
        """sets up the database if need be"""
        try:
            with Download.bind_ctx(self.DB):
                self.DB.create_tables([Download])

                # migration
                # 1: 2025-0?-??, PR #10??, feature/public_tracked_download,
                #    additional columns checksum timestamp and filesize
                present_names = [x.name for x in self.DB.get_columns("download")]
                migrator = SqliteMigrator(self.DB)
                migrate(
                    *[
                        migrator.add_column("download", col_name, col_field(null=True))
                        for (col_name, col_field) in [
                            ["failed", BooleanField],
                            ["filesize", IntegerField],
                            ["timestamp", IntegerField],
                            ["checksum", CharField],
                        ]
                        if col_name not in present_names
                    ]
                )
        finally:
            self.DB.close()

    def download(
        self,
        url,
        target_dir,
        file_name=None,
        size=None,
        checksum=None,
        integrity_check=None,
    ):
        """Download a file if it is not already present at the target destination.

        Parameters
        ----------
        url : str
            the url of the file to be downloaded
        target_dir : str | Path
            target destination, must be the path to a directory
        file_name : str, optional
            the target file name
            If `None`, the name will be derived from the url
        size : int, optional
            the expected file size.
            if set, this method will fail unless the actual size matches it.
        checksum : str, optional
            the expected checksum.
            if set, this method will fail unless the actual checksum matches it.
        integrity_check : int [Check.SIZE, Check.CHECKSUM, Check.TIMESTAMP], optional
            file properties of already downloaded files that are compared to recorded properties
            in case they match, another download is skipped and the existing file returned
            Default: `Check.FLAG_SIZE & Check.FLAG_TIMESTAMP`.

        Returns
        -------
        Path
            the path to the downloaded file

        Raises
        ------
        DownloadFailed
            in case the file couldn't be downloaded
            when number of retries was exceeded or when a download is already running
        """
        target_dir = Path(target_dir)
        file_name = file_name or Path(url).name
        if integrity_check is None:
            integrity_check = Downloader.Check.SIZE + Downloader.Check.TIMESTAMP

        download = self._previous_download(
            url=url,
            local_path=Path(target_dir, file_name),
            checkflags=integrity_check,
        )
        if not download or not download.check(
            filesize=size,
            checksum=":".join([self.checksum, checksum]) if checksum else None,
        ):
            download = self._actual_download(
                url=url,
                target_dir=target_dir,
                file_name=file_name,
                size=size,
                checksum=checksum,
            )
        return Path(download.path)

    def _get_download(self, local_path: Path):
        """get download by path when the download db is not bound"""
        try:
            with Download.bind_ctx(self.DB):
                return Download.get_or_none(Download.path == str(local_path.absolute()))
        finally:
            self.DB.close()

    def _previous_download(
        self, url: str, local_path: Path, checkflags: int
    ) -> Download:
        """Looks up the database for a previous download.
        If it is found, the file is checked for changes.
        """
        # look up sqlite database
        download = self._get_download(local_path)

        # no entry means this is the first time (or the former entry was deleted)
        if download is None:
            return None

        # there is a db entry but no file
        if not Path(download.path).exists():
            return None

        if download.url != url:
            raise DownloadFailed(
                f"this file ({str(local_path)}) has been downloaded from another url before"
                f" ({download.url}). Please remove the file or purge the entry from data base"
                f" ({self.DB.database}) if this is the correct url/target combination,"
                " then try again."
            )

        if download.failed:
            return None

        # in case a download is already in progress, weit for MAX_WAITING_PERIOD, than raise
        # DownloadFailed exception
        while not download.enddownload:
            wait = 2
            for _ in range(self.MAX_WAITING_PERIOD / wait):
                time.sleep(wait)
                download = self._get_download(local_path)
                if download.enddownload:
                    break
            if not download.enddownload:
                raise DownloadFailed(
                    f"A download of {url} via the climada.util.files_handler.Downloader has"
                    " been requested before. Either it is still in progress or the process"
                    " got interrupted. In the former case just wait until the download has"
                    " finished and try again, in the latter run"
                    f" `Downloader.remove_download_record(Path('{str(local_path)}'))` from Python."
                    " If unsure, check your internet connection, wait for as long as it takes"
                    " to download a file of the given size and try again."
                    " If the problem persists, purge the cache db with said call."
                )

        # return the path if all is good
        if self._integrity_check(
            download=download, local_path=local_path, checkflags=checkflags
        ):
            return download

        # the file must have been successfully downloaded but afterwards manipulated.
        # the record in the data base is corrupt. it will be removed in the `_actual_download``.
        return None

    def _actual_download(
        self,
        url,
        target_dir,
        file_name,
        size,
        checksum,
    ) -> Download:
        """Download a file if it is not already present at the target destination.

        Parameters
        ----------
        url : str
            the url of the file to be download
        target_dir : str | Path
            target destination, must be(have like) a directory
        file_name : str
            the target file name
            If `None`, the name will be derived from the url
        size : int, optional
            if set, the downloaded file's size must eventually match it
        checksum : int, optional
            if set, the downloaded file's checksum must eventually match it

        Returns
        -------
        Download
            the database entry of a successful download

        Raises
        ------
        DownloadFailed
            if the download was not successful
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        local_path = Path(target_dir, file_name)
        if local_path.is_dir():
            raise DownloadFailed(
                f"There is a directory at the target location {str(local_path)}"
            )
        try:
            with Download.bind_ctx(self.DB):
                for _ in range(self.RETRIES):
                    local_path.unlink(missing_ok=True)
                    download = self._tracked_download(
                        remote_url=url, local_path=local_path
                    )
                    if not download:
                        continue
                    if not download.check(filesize=size, checksum=checksum):
                        Path(download.path).unlink()
                        download.failed = True
                        download.save()
                        continue
                    if not download.enddownload:
                        raise RuntimeError("unexpected state, must be caused by a bug")
                    return download
        finally:
            self.DB.close()
        raise DownloadFailed(
            f"could not actually download {url}, giving up after {self.RETRIES} times"
        )

    def _tracked_download(self, remote_url, local_path: Path) -> Download:
        """Creates or replaces an entry in the download db.

        Parameters
        ----------
        remote_url : str
            the url pointing to the file to be downloaded
        local_path : Path
            target destination,
            if it is a directory the original filename (fileinfo.filen_name) is kept

        Returns
        -------
        Download or None
        """
        path_as_str = str(local_path.absolute())

        download = Download.get_or_none(Download.path == path_as_str)
        if download:  # i.g. this means that the attempt just before failed
            download.delete_instance()
        download = Download.create(
            url=remote_url, path=path_as_str, startdownload=datetime.now(timezone.utc)
        )
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
            download.enddownload = datetime.now(timezone.utc)
            download.timestamp = int(local_path.stat().st_mtime)
            download.filesize = local_path.stat().st_size
            if self.checksum:
                download.checksum = file_checksum(local_path, self.checksum)
            download.save()

        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("Download from %s has failed with %s", remote_url, exc)
            download.failed = True
            download.save()
            return None

        return Download.get(Download.path == path_as_str)

    def remove_download_record(self, local_path: Path):
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
        try:
            with Download.bind_ctx(self.DB):
                download = Download.get(Download.path == str(local_path.absolute()))
                download.delete_instance()
        finally:
            self.DB.close()  # on Linux this call is unnecessary. On Windows, though,
            # for, e.g,, peewee 3.17.1, it is required to release the sqlite db file.

    def purge_cache(self, keep_urls: list = None, purge_dir: Path = None):
        """Removes all entries from the download database and the corresponding files from the file
        system except those in the keep_urls list

        Parameters
        ----------
        keep_urls: list of str, optional
            entries with matching url won't be deleted
        purge_dir: Path, optional
            if set, all empty directories beneath this path will be removed if any of their descents
            has been downloaded and got removed by this purge.
        """
        keep_urls = keep_urls or []

        parents = set()
        try:
            with Download.bind_ctx(self.DB):
                for download in Download.select():
                    if download.url not in keep_urls:
                        Path(download.path).unlink(missing_ok=True)
                        download.delete_instance()
                        if purge_dir and Path(purge_dir) in Path(download.path).parents:
                            parents.add(Path(download.path).parent)
        finally:
            self.DB.close()

        while parents:
            ancestors = parents
            parents = set()
            for parent in ancestors:
                try:
                    parent.rmdir()
                    parents.add(parent.parent)
                except OSError:  # raised if the directory isn't empty
                    pass

    def _integrity_check(self, download, local_path, checkflags):
        """default method for checking file integrity
        basically checks whether the file is there and hasn't changed since the download

        Parameters
        ----------
        download : Download
            record from the database
        local_path : Path
            the file whose integrity is being scrutinized
        checkflags :

        Raises
        ------
        DownloadFailed
            if the file
            - is not there
            - actual size and recorded size differ
            - actual timestamp and recorded timestamp differ
            - checksum is recordedn and actual checksum and recorded checksum differ
        """
        # downlaod anew if SYNC is flagged
        if checkflags & Downloader.Check.SYNC:
            return False

        if not local_path.exists():
            return False

        # raise Exception if path exists but isn't a file
        if not local_path.is_file():
            raise ValueError(f"{local_path} must be a file")

        if checkflags & Downloader.Check.SIZE:
            if local_path.stat().st_size != download.filesize:
                LOGGER.warning(
                    "%s has wrong size: %d instead of %d",
                    str(local_path),
                    local_path.stat().st_size,
                    download.filesize,
                )
                return False

        if checkflags & Downloader.Check.TIMESTAMP:
            if int(local_path.stat().st_mtime) != download.timestamp:
                expected = datetime.fromtimestamp(download.timestamp).isoformat()
                found = datetime.fromtimestamp(
                    int(local_path.stat().st_mtime)
                ).isoformat()
                LOGGER.warning(
                    "%s has wrong timestamp: %s instead of %s",
                    str(local_path),
                    found,
                    expected,
                )
                return False

        if checkflags & Downloader.Check.CHECKSUM:
            try:
                csm, hsh = download.checksum.split(":")
            except ValueError as exc:
                raise ValueError(
                    f"cannot check integrity of {local_path} based on hash sum,"
                    " field missing in database"
                ) from exc
            found = file_checksum(local_path, csm)
            if hsh != found:
                LOGGER.warning(
                    "%s has wrong %s checksum: %s instead of %s",
                    str(local_path),
                    csm,
                    hsh,
                    found,
                )
                return False

        return True


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
