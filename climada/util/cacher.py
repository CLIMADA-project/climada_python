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

module containing functions to check variables properties.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from peewee import CharField, DateTimeField, Model, SqliteDatabase


class CachedResults(Model):
    """Database entry keeping track of results"""

    # Since no primary key is specified, Peewee automatically creates an auto-incrementing
    # primary key named “id”.
    arg_key = CharField(unique=True)
    result = CharField()
    cached = DateTimeField()


LOGGER = logging.getLogger(__name__)


class Cacher:
    """Utility class handling cached results from http requests,
    e.g., to enable the API Client working in offline mode.
    """

    def __init__(self, enabled=True, sqlite=None, cachedir=None):
        """Constructor of Cacher.

        Parameters
        ----------
        cache_enabled : bool, None
            Default: None, in this case the value is taken from CONFIG.data_api.cache_enabled.
        """
        self.enabled = enabled

        self.cachedir = cachedir and Path(cachedir)
        if self.cachedir:
            self.cachedir.mkdir(parents=True, exist_ok=True)

        sqlite = sqlite and Path(sqlite)
        if sqlite:
            Path(sqlite).parent.mkdir(parents=True, exist_ok=True)
            self.sqlite = SqliteDatabase(sqlite)
            self._initsqlite()
        else:
            self.sqlite = None

    def _initsqlite(self):
        try:
            with CachedResults.bind_ctx(self.sqlite):
                self.sqlite.create_tables([CachedResults])
        finally:
            self.sqlite.close()

    @staticmethod
    def _make_key(*args, **kwargs):
        as_text = "\t".join(
            [str(a) for a in args] + [f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]
        )
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
            with Path(self.cachedir, _key).open("w", encoding="utf-8") as flp:
                json.dump(result, flp)
        except (OSError, ValueError, TypeError) as e:
            pass
        if self.sqlite:
            try:
                with CachedResults.bind_ctx(self.sqlite):
                    CachedResults.create(
                        arg_key=_key,
                        result=str(result),
                        cached=datetime.now(timezone.utc),
                    )
            finally:
                self.sqlite.close()

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
        if self.sqlite:
            try:
                with CachedResults.bind_ctx(self.sqlite):
                    found = CachedResults.get_or_none(CachedResults.arg_key == _key)
                    if found:
                        return found.result
            finally:
                self.sqlite.close()
        try:
            with Path(self.cachedir, _key).open(encoding="utf-8") as flp:
                return json.load(flp)
        except (OSError, ValueError, TypeError):
            return None


def _insert_function(func):
    try:
        fclas = func.__self__.__class__.__name__
        return func.__module__, fclas, func.__name__
    except AttributeError:
        return func.__module__, func.__name__


def cached(cacher: Cacher):

    def wrapped(func):

        def _(*args, **kwargs):
            cached_result = cacher.fetch(*_insert_function(func), *args, **kwargs)
            if cached_result:
                return cached_result
            result = func(*args, **kwargs)
            if isinstance(result, str):
                cacher.store(result, *_insert_function(func), *args, **kwargs)
            return result

        return _

    return wrapped
