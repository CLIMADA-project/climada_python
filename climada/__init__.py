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

climada init
"""
from shutil import copyfile
from pathlib import Path

from .util.config import CONFIG, setup_logging
from .util.constants import *


__all__ = ['init']


REPO_SYSTEM_DATA = 'data/system'
SYSTEM_DATA_FILES = [
    ISIMIP_GPWV3_NATID_150AS,
    GLB_CENTROIDS_MAT,
    ENT_TEMPLATE_XLS,
    HAZ_TEMPLATE_XLS,
    RIVER_FLOOD_REGIONS_CSV,
    NATEARTH_CENTROIDS[150],
    NATEARTH_CENTROIDS[360],
    SYSTEM_DIR.joinpath('WEALTH2GDP_factors_CRI_2016.csv'),
    SYSTEM_DIR.joinpath('GDP_TWN_IMF_WEO_data.csv'),
]
REPO_DEMO_DATA = 'data/demo'
DEMO_DATA_FILES = [
    HAZ_DEMO_FL,
    HAZ_DEMO_FLDDPH,
    HAZ_DEMO_FLDFRC,
    HAZ_DEMO_MAT,
    DEMO_DIR.joinpath('demo_emdat_impact_data_2020.csv')
] + WS_DEMO_NC


def setup_climada_data():
    for src_dir, path_list in [(REPO_SYSTEM_DATA, SYSTEM_DATA_FILES),
                               (REPO_DEMO_DATA, DEMO_DATA_FILES)]:
        for path in path_list:
            if not path.exists():
                src = Path(__file__).parent.parent.joinpath(src_dir, path.name)
                copyfile(src, path)


def init():
    setup_climada_data()
    setup_logging(CONFIG.log_level.str())


init()