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

Import data from Global Population of the World (GPW) datasets
"""
import logging
import subprocess

import rasterio
import numpy as np

from climada.util.constants import SYSTEM_DIR

logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

FILENAME_GPW = 'gpw_v4_population_count_rev%02i_%04i_30_sec.tif'
DIRNAME_GPW = 'gpw-v4-population-count-rev%02i_%04i_30_sec_tif'
GPW_VERSIONS = [11, 10, 12, 13]
# FILENAME_GPW1 = '_30_sec.tif'
YEARS_AVAILABLE = np.array([2000, 2005, 2010, 2015, 2020])
# BUFFER_VAL = -340282306073709652508363335590014353408
# Hard coded value which is used for NANs in original GPW data


def load_gpw_pop_shape(geometry, reference_year, gpw_version=11, data_dir=None,
                       layer=0): # TODO: manually tested but no tests exist yet
    """Read gridded population data from GPW TIFF
    and crop to given shape(s).
    
    Parameters
    ----------
    geometry : shape(s) to crop data to in degree lon/lat.
        for example shapely.geometry.Polygon object or
        from polygon(s) defined in a (country) shapefile.
    reference_year : int
        target year for data extraction
    gpw_version : int (optional)
        Version number of GPW population data.
        The default is 11
    data_dir : Path (optional)
        Path to directory with GPW data.
        The default is SYSTEM_DIR.
    layer : int (optional)
        relevant data layer in input TIFF file to return.
        The default is 0 and should not be changed without understanding the
        different data layers in the given TIFF file.

    Returns
    -------
    pop_data : 2D numpy array
        contains extracted population count data per grid point in shape
        first dimension is lat, second dimension is lon.
    meta : dict
        contains meta data per array, including "transform" with 
        meta data on coordinates.
    global_transform : Affine instance
        contains six numbers, providing transform info for global GWP grid.
        global_transform is required for resampling on a globally consistent grid
    """
    # check whether GPW input file exists and get file path
    file_path = get_gpw_file_path(gpw_version, reference_year, data_dir=data_dir)

    # open TIFF and extract cropped data from input file:
    src = rasterio.open(file_path)
    global_transform = src.transform
    pop_data, out_transform = rasterio.mask.mask(src, [geometry], crop=True,
                                                  nodata=0)

    # extract and update meta data for cropped data and close src:
    meta = src.meta
    meta.update({"driver": "GTiff",
                 "height": pop_data.shape[1],
                 "width": pop_data.shape[2],
                 "transform": out_transform})
    src.close()
    return pop_data[layer,:,:], meta, global_transform

def get_gpw_file_path(gpw_version, reference_year, data_dir=None):
    """Check available GPW population data versions and year closest to
    reference_year and return full path to TIFF file.

    Parameters
    ----------
    gpw_version : int (optional)
        Version number of population data.
        The default is 11
    reference_year : int (optional)
        Data year is selected as close to reference_year as possible.
        The default is 2020.
    data_dir : pathlib.Path (optional)
        Absolute path where files are stored. Default: SYSTEM_DIR

    Raises
    ------
    FileExistsError

    Returns
    -------
    pathlib.Path : path to input file with population data
    """
    if data_dir is None:
        data_dir = SYSTEM_DIR
    # find closest year to reference_year with data available:
    year = YEARS_AVAILABLE[np.abs(YEARS_AVAILABLE - reference_year).argmin()]
    if year != reference_year:
        LOGGER.warning('Reference year: %i. Using nearest available year for GPW population data: %i',
                    reference_year, year)

    # check if file is available for given or alternative other GPW version,
    # if available, return full path to file:
    for ver in [gpw_version] + GPW_VERSIONS:
        file_path = data_dir / (FILENAME_GPW % (ver, year))
        if file_path.is_file():
            LOGGER.info('GPW Version v4.%2i', ver)
            return file_path
        else:
            file_path = data_dir / (DIRNAME_GPW % (ver, year)) / (FILENAME_GPW % (ver, year))
            if file_path.is_file():
                LOGGER.info('GPW Version v4.%2i', ver)
                return file_path

    # if no inoput file was found, FileExistsError is raised
    if SYSTEM_DIR.joinpath('GPW_help.pdf').is_file():
        subprocess.Popen([str(SYSTEM_DIR.joinpath('GPW_help.pdf'))], shell=True)
        raise FileExistsError(f'The file {file_path} could not '
                              + 'be found. Please download the file '
                              + 'first or choose a different folder. '
                              + 'Instructions on how to download the '
                              + 'file has been openend in your PDF '
                              + 'viewer.')
    else:
        raise FileExistsError(f'The file {file_path} could not '
                              + 'be found. Please download the file '
                              + 'first or choose a different folder. '
                              + 'The data can be downloaded from '
                              + 'http://sedac.ciesin.columbia.edu/'
                              + 'data/collection/gpw-v4/sets/browse')
    return None
