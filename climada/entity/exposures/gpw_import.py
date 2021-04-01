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

from osgeo import gdal
import pandas as pd
from scipy import ndimage as nd
import numpy as np

from climada.util.constants import SYSTEM_DIR
from climada.entity.exposures import litpop as LitPop
logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

FILENAME_GPW = 'gpw_v4_population_count_rev%02i_%04i_30_sec.tif'
GPW_VERSIONS = [11, 10, 12, 13]
# FILENAME_GPW1 = '_30_sec.tif'
YEARS_AVAILABLE = np.array([2000, 2005, 2010, 2015, 2020])
BUFFER_VAL = -340282306073709652508363335590014353408
# Hard coded value which is used for NANs in original GPW data



def _gpw_bbox_cutter(gpw_data, bbox, resolution, arr1_shape=[17400, 43200]):
    """Crops the imported GPW data to the bounding box to reduce memory foot
        print after it has been resized to desired resolution.

    Optional parameters:
        gpw_data (array): Imported GPW data in gridded format
        bbox (array 4x1): Bounding box to which the data is cropped.
        resolution (int): The resolution in arcsec to which the data has
            been resized.

    Returns:
        gpw_data (array): Cropped GPW data
    """

    """gpw data is 17400 rows x 43200 cols in dimension (from 85 N to 60 S in
    latitude, full longitudinal range). Hence, the bounding box can easily be
    converted to the according indices in the gpw data"""
    steps_p_res = 3600 / resolution
    zoom = 30 / resolution
    col_min, row_min, col_max, row_max =\
        LitPop._litpop_coords_in_glb_grid(bbox, resolution)

    # accomodate to fact that not the whole grid is present in the v.10 dataset:
    if arr1_shape[0] == 17400:
        row_min, row_max = int(row_min - 5 * steps_p_res), \
            int(row_max - 5 * steps_p_res)

    rows_gpw = arr1_shape[0]
    cols_gpw = arr1_shape[1]
    if col_max < (cols_gpw / zoom) - 1:
        col_max = col_max + 1
    if row_max < (rows_gpw / zoom) - 1:
        row_max = row_max + 1
    gpw_data = gpw_data[:, col_min:col_max]

    if row_min >= 0 and row_min < (rows_gpw / zoom) and row_max >= 0 \
       and row_max < (rows_gpw / zoom):
        gpw_data = gpw_data[row_min:row_max, :]
    elif row_min < 0 and row_max >= 0 and row_max < (rows_gpw / zoom):
        np.concatenate(np.zeros((abs(row_min), gpw_data.shape[1])),
                       gpw_data[0:row_max, :])
    elif row_min < 0 and row_max < 0:
        gpw_data = np.zeros((row_max - row_min, col_max - col_min))
    elif row_min < 0 and row_max >= (rows_gpw / zoom):
        np.concatenate(np.zeros((abs(row_min), gpw_data.shape[1])), gpw_data,
                       np.zeros((row_max - (rows_gpw / zoom) + 1, gpw_data.shape[1])))
    elif row_min >= (rows_gpw / zoom):
        gpw_data = np.zeros((row_max - row_min, col_max - col_min))
    return gpw_data

def check_bounding_box(coord_list):
    """Check if a bounding box is valid.
    Parameters:
        coord_list (4x1 array): bounding box to be checked.
    OUTPUT:
        isCorrectType (boolean): True if bounding box is valid, false otehrwise
    """
    is_correct_type = True
    if coord_list.size != 4:
        is_correct_type = False
        return is_correct_type
    min_lat, min_lon, max_lat, max_lon = (coord_list[0], coord_list[1],
                                          coord_list[2], coord_list[3])
    assert max_lat < min_lat, "Maximum latitude cannot be smaller than minimum latitude."
    assert max_lon < min_lon, "Maximum longitude cannot be smaller than minimum longitude."
    assert min_lat < -90, "Minimum latitude cannot be smaller than -90."
    assert min_lon < -180, "Minimum longitude cannot be smaller than -180."
    assert max_lat > 90, "Maximum latitude cannot be larger than 90."
    assert max_lon > 180, "Maximum longitude cannot be larger than 180."
    return is_correct_type

def get_box_gpw(**parameters):
    """Reads data from GPW GeoTiff file and cuts out the data along a chosen
        bounding box.

    Parameters
    ----------
    gpw_path : pathlib.Path
        Absolute path where files are stored. Default: SYSTEM_DIR
    resolution : int
        The resolution in arcsec in which the data output is created.
    country_cut_mode : int
        Defines how the country is cut out: If 0, the country is only cut out
        with a bounding box. If 1, the country is cut out along it's borders
        Default: 0.
        #TODO: Unimplemented
    cut_bbox : array-like, shape (1,4)
        Bounding box (ESRI type) to be cut out.
        The layout of the bounding box corresponds to the bounding box of
        the ESRI shape files and is as follows:
        [minimum longitude, minimum latitude, maximum longitude, maxmimum latitude]
        If country_cut_mode = 1, the cut_bbox is overwritten/ignored.
    return_coords : int
        Determines whether latitude and longitude are delievered along with gpw
        data (0) or only gpw_data is returned. Default: 0.
    add_one : boolean
        Determine whether the integer one is added to all cells to eliminate
        zero pixels. Default: 0.
        #TODO: Unimplemented
    reference_year : int
        reference year, available years are:
        2000, 2005, 2010, 2015 (default), 2020

    Returns
    -------
    tile_temp : pandas.arrays.SparseArray
        GPW data
    lon : list
        List with longitudinal infomation on the GPW data. Same
        dimensionality as tile_temp (only returned if return_coords is 1).
    lat : list
        list with latitudinal infomation on the GPW data. Same
        dimensionality as tile_temp (only returned if return_coords is 1).
    """
    resolution = parameters.get('resolution', 30)
    cut_bbox = parameters.get('cut_bbox')
#    country_cut_mode = parameters.get('country_cut_mode', 0)
    return_coords = parameters.get('return_coords', 0)
    reference_year = parameters.get('reference_year', 2015)
    year = YEARS_AVAILABLE[np.abs(YEARS_AVAILABLE - reference_year).argmin()]

    if year != reference_year:
        LOGGER.info('Reference year: %i. Using nearest available year for GWP population data: %i',
                    reference_year, year)
    if (cut_bbox is None) & (return_coords == 0):
    # If we don't have any bbox by now and we need one, we just use the global
        cut_bbox = np.array((-180, -90, 180, 90))
    zoom_factor = 30 / resolution  # Orignal resolution is arc-seconds
    file_exists = False
    for ver in GPW_VERSIONS:
        gpw_path = parameters.get('gpw_path', SYSTEM_DIR)
        fpath = gpw_path.joinpath(FILENAME_GPW % (ver, year))
        if fpath.is_file():
            file_exists = True
            LOGGER.info('GPW Version v4.%2i', ver)
            break

    try:
        if not file_exists:
            if SYSTEM_DIR.joinpath('GPW_help.pdf').is_file():
                subprocess.Popen([str(SYSTEM_DIR.joinpath('GPW_help.pdf'))], shell=True)
                raise FileExistsError(f'The file {fpath} could not '
                                      + 'be found. Please download the file '
                                      + 'first or choose a different folder. '
                                      + 'Instructions on how to download the '
                                      + 'file has been openend in your PDF '
                                      + 'viewer.')
            else:
                raise FileExistsError(f'The file {fpath} could not '
                                      + 'be found. Please download the file '
                                      + 'first or choose a different folder. '
                                      + 'The data can be downloaded from '
                                      + 'http://sedac.ciesin.columbia.edu/'
                                      + 'data/collection/gpw-v4/sets/browse')
        LOGGER.debug('Importing %s', str(fpath))
        gpw_file = gdal.Open(str(fpath))
        band1 = gpw_file.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        del band1, gpw_file
        arr1[arr1 < 0] = 0
        if arr1.shape != (17400, 43200) and arr1.shape != (21600, 43200):
            LOGGER.warning('GPW data dimensions mismatch. Actual dimensions: %s x %s',
                           arr1.shape[0], arr1.shape[1])
            LOGGER.warning('Expected dimensions: 17400x43200 or 21600x43200.')
        if zoom_factor != 1:
            total_population = arr1.sum()
            tile_temp = nd.zoom(arr1, zoom_factor, order=1)
            # normalize interpolated gridded population count to keep total population stable:
            tile_temp = tile_temp * (total_population / tile_temp.sum())
        else:
            tile_temp = arr1
        if tile_temp.ndim == 2:
            if cut_bbox is not None:
                tile_temp = _gpw_bbox_cutter(tile_temp, cut_bbox, resolution,
                                             arr1_shape=arr1.shape)
        else:
            LOGGER.error('Error: Matrix has an invalid number of dimensions \
                         (more than 2). Could not continue operation.')
            raise TypeError
        tile_temp = pd.arrays.SparseArray(
            tile_temp.reshape((tile_temp.size,), order='F'),
            fill_value=0)
        del arr1
        if return_coords == 1:
            lon = tuple((cut_bbox[0], 1 / (3600 / resolution)))
            lat = tuple((cut_bbox[1], 1 / (3600 / resolution)))
            return tile_temp, lon, lat

        return tile_temp

    except:
        LOGGER.error('Importing the GPW population density file failed.')
        raise
