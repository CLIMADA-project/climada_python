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

import rasterio
import numpy as np

from climada.util.constants import SYSTEM_DIR
from climada import CONFIG

LOGGER = logging.getLogger(__name__)


def load_gpw_pop_shape(geometry, reference_year, gpw_version,
                       data_dir=SYSTEM_DIR, layer=0, verbose=True):
    """Read gridded population data from TIFF and crop to given shape(s).

    Note: A (free) NASA Earthdata login is necessary to download the data.
    Data can be downloaded e.g. for gpw_version=11 and year 2015 from
    https://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/
    gpw-v4-population-count-rev11/gpw-v4-population-count-rev11_2015_30_sec_tif.zip

    Parameters
    ----------
    geometry : shape(s) to crop data to in degree lon/lat.
        for example shapely.geometry.(Multi)Polygon or shapefile.Shape
        from polygon(s) defined in a (country) shapefile.
    reference_year : int
        target year for data extraction
    gpw_version : int
        Version number of GPW population data, i.e. 11 for v4.11.
        The default is CONFIG.exposures.litpop.gpw_population.gpw_version.int()
    data_dir : Path, optional
        Path to data directory holding GPW data folders.
        The default is SYSTEM_DIR.
    layer : int, optional
        relevant data layer in input TIFF file to return.
        The default is 0 and should not be changed without understanding the
        different data layers in the given TIFF file.
    verbose : bool, optional
        Enable verbose logging about the used GPW version and reference year. Default: True.


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
    file_path = get_gpw_file_path(gpw_version, reference_year, data_dir=data_dir, verbose=verbose)

    # open TIFF and extract cropped data from input file:
    with rasterio.open(file_path, 'r') as src:
        global_transform = src.transform
        pop_data, out_transform = rasterio.mask.mask(src, [geometry], crop=True, nodata=0)

        # extract and update meta data for cropped data and close src:
        meta = src.meta
        meta.update({
            "driver": "GTiff",
            "height": pop_data.shape[1],
            "width": pop_data.shape[2],
            "transform": out_transform,
        })
    return pop_data[layer,:,:], meta, global_transform

def get_gpw_file_path(gpw_version, reference_year, data_dir=None, verbose=True):
    """Check available GPW population data versions and year closest to
    `reference_year` and return full path to TIFF file.

    Parameters
    ----------
    gpw_version : int (optional)
        Version number of GPW population data, i.e. 11 for v4.11.
    reference_year : int (optional)
        Data year is selected as close to reference_year as possible.
        The default is 2020.
    data_dir : pathlib.Path (optional)
        Absolute path where files are stored. Default: SYSTEM_DIR
    verbose : bool, optional
        Enable verbose logging about the used GPW version and reference year. Default: True.

    Raises
    ------
    FileExistsError

    Returns
    -------
    pathlib.Path : path to input file with population data
    """
    if data_dir is None:
        data_dir = SYSTEM_DIR

    # get years available in GPW data from CONFIG and convert to array:
    years_available = np.array([
        year.int() for year in CONFIG.exposures.litpop.gpw_population.years_available.list()
    ])

    # find closest year to reference_year with data available:
    year = years_available[np.abs(years_available - reference_year).argmin()]
    if verbose and year != reference_year:
        LOGGER.warning('Reference year: %i. Using nearest available year for GPW data: %i',
                       reference_year, year)

    # check if file is available for given GPW version, construct GPW file path from CONFIG:
    # if available, return full path to file:
    gpw_dirname = CONFIG.exposures.litpop.gpw_population.dirname_gpw.str() % (gpw_version, year)
    gpw_filename = CONFIG.exposures.litpop.gpw_population.filename_gpw.str() % (gpw_version, year)
    for file_path in [data_dir / gpw_filename, data_dir / gpw_dirname / gpw_filename]:
        if file_path.is_file():
            if verbose:
                LOGGER.info('GPW Version v4.%2i', gpw_version)
            return file_path

    # if the file was not found, an exception is raised with instructions on how to obtain it
    sedac_url = "http://sedac.ciesin.columbia.edu"
    sedac_browse_url = f"{sedac_url}/data/collection/gpw-v4/sets/browse"
    sedac_file_url = (
        f"{sedac_url}/downloads/data/gpw-v4/gpw-v4-population-count-rev{gpw_version}/"
        f"{gpw_dirname}.zip"
    )
    raise FileNotFoundError(
        f'The file {file_path} could not be found. Please download the file first or choose a'
        f' different folder. The data can be downloaded from {sedac_browse_url}, e.g.,'
        f' {sedac_file_url} (Free NASA Earthdata login required).'
    )
