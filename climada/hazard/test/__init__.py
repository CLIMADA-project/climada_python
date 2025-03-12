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

import shutil

from climada.hazard.tc_tracks import IBTRACS_FILE, IBTRACS_URL
from climada.util.api_client import Client
from climada.util.constants import SYSTEM_DIR
from climada.util.files_handler import download_ftp


def download_ibtracs():
    """This makes sure a IBTrACS.ALL.v04r01.nc file is present in SYSTEM_DIR
    First, downloading from the original sources is attempted. If that fails an old version
    is downloaded from the CLIMADA Data API
    """
    if SYSTEM_DIR.joinpath(IBTRACS_FILE).is_file():
        return  # Nothing to do

    try:
        download_ftp(f"{IBTRACS_URL}/{IBTRACS_FILE}", IBTRACS_FILE)
        shutil.move(IBTRACS_FILE, SYSTEM_DIR)

    except (
        ValueError
    ):  # plan b: download an old version of that file from the climada api
        client = Client()
        dsinfo = client.get_dataset_info(
            name="IBTrACS", version="v04r01", status="external"
        )
        [fileinfo] = [
            fi for fi in dsinfo.files if fi.file_name == "IBTrACS.ALL.v04r01.nc"
        ]
        client._download_file(local_path=SYSTEM_DIR, fileinfo=fileinfo)
