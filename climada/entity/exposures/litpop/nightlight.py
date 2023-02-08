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

Define nightlight reader and cutting functions.
"""
import glob
import shutil
import tarfile
import gzip
import pickle
import logging
from pathlib import Path
import rasterio

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from osgeo import gdal
from PIL import Image
from shapefile import Shape

from climada.util import ureg
from climada.util.constants import SYSTEM_DIR
from climada.util.files_handler import download_file
from climada.util.save import save
from climada import CONFIG

Image.MAX_IMAGE_PIXELS = 1e9

LOGGER = logging.getLogger(__name__)

NOAA_RESOLUTION_DEG = (30 * ureg.arc_second).to(ureg.deg).magnitude
"""NOAA nightlights coordinates resolution in degrees."""

NASA_RESOLUTION_DEG = (15 * ureg.arc_second).to(ureg.deg).magnitude
"""NASA nightlights coordinates resolution in degrees."""

NASA_TILE_SIZE = (21600, 21600)
"""NASA nightlights tile resolution."""

NOAA_BORDER = (-180, -65, 180, 75)
"""NOAA nightlights border (min_lon, min_lat, max_lon, max_lat)"""

BM_FILENAMES = ['BlackMarble_%i_A1_geo_gray.tif',
                'BlackMarble_%i_A2_geo_gray.tif',
                'BlackMarble_%i_B1_geo_gray.tif',
                'BlackMarble_%i_B2_geo_gray.tif',
                'BlackMarble_%i_C1_geo_gray.tif',
                'BlackMarble_%i_C2_geo_gray.tif',
                'BlackMarble_%i_D1_geo_gray.tif',
                'BlackMarble_%i_D2_geo_gray.tif'
               ]
"""Nightlight NASA files which generate the whole earth when put together."""

def load_nasa_nl_shape(geometry, year, data_dir=SYSTEM_DIR, dtype='float32'):
    """Read nightlight data from NASA BlackMarble tiles
    cropped to given shape(s) and combine arrays from each tile.

    1) check and download required blackmarble files
    2) read and crop data from each file required in a bounding box around
       the given `geometry`.
    3) combine data from all input files into one array. this array then
       contains all data in the geographic bounding box around `geometry`.
    4) return array with nightlight data

    Parameters
    ----------
    geometry : shape(s) to crop data to in degree lon/lat.
        for example shapely.geometry.(Multi)Polygon or shapefile.Shape.
        from polygon defined in a shapefile. The object should have
        attribute 'bounds' or 'points'
    year : int
        target year for nightlight data, e.g. 2016.
        Closest availble year is selected.
    data_dir : Path (optional)
        Path to directory with BlackMarble data.
        The default is SYSTEM_DIR.
    dtype : dtype
        data type for output default 'float32', required for LitPop,
        choose 'int8' for integer.

    Returns
    -------
    results_array : numpy array
        extracted and combined nightlight data for bounding box around shape
    meta : dict
        rasterio meta data for results_array
    """
    if isinstance(geometry, Shape):
        bounds = geometry.bbox
    else:
        bounds = geometry.bounds

    # get years available in BlackMarble data from CONFIG and convert to array:
    years_available = [year.int() for year in \
                       CONFIG.exposures.litpop.nightlights.blackmarble_years.list()
                       ]
    # get year closest to year with BlackMarble data available:
    year = min(years_available, key=lambda x: abs(x - year))
    # determin black marble tiles with coordinates containing the bounds:
    req_files = get_required_nl_files(bounds)
    # check wether required files exist locally:
    files_exist = check_nl_local_file_exists(required_files=req_files,
                                             check_path=data_dir, year=year)
    # download data that is missing:
    download_nl_files(req_files, files_exist, data_dir, year)
    # convert `req_files` to sorted list of indices:
    req_files = np.where(req_files ==1)[0]
    # init empty lists for tiles depending on position in global grid:
    results_array_north = list() # tiles A1, B1, C1, D1 (Nothern Hemisphere)
    results_array_south = list() # tiles A2, B2, C2, D2 (Southern Hemisphere)

    # loop through required files, load and crop data for each:
    for idx, i_file in enumerate(req_files):
        # read cropped data from  source file (src) to np.ndarray:
        out_image, meta_tmp = load_nasa_nl_shape_single_tile(geometry,
                                        data_dir / (BM_FILENAMES[i_file] %(year)))
        # sort indicies to northenr and southern hemisphere:
        if i_file in [0,2,4,6]: # indicies of northern hemisphere files
            results_array_north.append(out_image)
        elif i_file in [1,3,5,7]: # indicies of southern hemisphere files
            results_array_south.append(out_image)

        # from first (top left) of tiles, meta is initiated, incl. origin:
        if idx == 0:
            meta = meta_tmp
            # set correct CRS from local tile's CRS to global WGS 84:
            meta.update({"crs": rasterio.crs.CRS.from_epsg(4326),
                         "dtype": dtype})
            if len(req_files) == 1: # only one tile required:
                return np.array(out_image, dtype=dtype), meta
    # Else, combine data from multiple input files (BlackMarble tiles) -
    # concatenate arrays from west to east and from north to south:
    del out_image
    if results_array_north: # northern hemisphere west to east
        results_array_north = np.concatenate(results_array_north, axis=1)
    if results_array_south: # southern hemisphere west to east
        results_array_south = np.concatenate(results_array_south, axis=1)
    if isinstance(results_array_north, np.ndarray) and isinstance(results_array_south, np.ndarray):
        # north to south if both hemispheres are involved
        results_array_north = np.concatenate([results_array_north, results_array_south], axis=0)
    elif isinstance(results_array_south, np.ndarray): # only southern hemisphere
        results_array_north = results_array_south
    del results_array_south

    # update number of elements per axis in meta dictionary:
    meta.update({"height": results_array_north.shape[0],
                 "width": results_array_north.shape[1],
                 "dtype": dtype})
    return np.array(results_array_north, dtype=dtype), meta

def get_required_nl_files(bounds):
    """Determines which of the satellite pictures are necessary for
        a certain bounding box (e.g. country)

    Parameters
    ----------
    bounds : 1x4 tuple
        bounding box from shape (min_lon, min_lat, max_lon, max_lat).

    Raises
    ------
    ValueError
        invalid `bounds`

    Returns
    -------
    req_files : numpy array
        Array indicating the required files for the current operation with a
        boolean value (1: file is required, 0: file is not required).
    """
    # check if bounds are valid:
    if (np.size(bounds) != 4) or (bounds[0] > bounds[2]) or (bounds[1] > bounds[3]):
        raise ValueError('Invalid bounds supplied. `bounds` must be tuple'+
                         ' with (min_lon, min_lat, max_lon, max_lat).')
    min_lon, min_lat, max_lon, max_lat = bounds

    # longitude first. The width of all tiles is 90 degrees
    tile_width = 90
    req_files = np.zeros(np.count_nonzero(BM_FILENAMES),)

    # determine the staring tile
    first_tile_lon = min(np.floor((min_lon - (-180)) / tile_width), 3)  # "normalise" to zero
    last_tile_lon = min(np.floor((max_lon - (-180)) / tile_width), 3)

    # Now latitude. The height of all tiles is the same as the height.
    # Note that for this analysis returns an index which follows from North to South oritentation.
    first_tile_lat = min(np.floor(-(min_lat - (90)) / tile_width), 1)
    last_tile_lat = min(np.floor(-(max_lat - 90) / tile_width), 1)

    for i_lon in range(0, int(len(req_files) / 2)):
        if first_tile_lon <= i_lon <= last_tile_lon:
            if first_tile_lat == 0 or last_tile_lat == 0:
                req_files[((i_lon)) * 2] = 1
            if first_tile_lat == 1 or last_tile_lat == 1:
                req_files[((i_lon)) * 2 + 1] = 1
        else:
            continue
    return req_files

def check_nl_local_file_exists(required_files=None, check_path=SYSTEM_DIR,
                               year=2016):
    """Checks if BM Satellite files are avaialbe and returns a vector
    denoting the missing files.

    Parameters
    ----------
    required_files : numpy array, optional
        boolean array of dimension (8,) with which
        some files can be skipped. Only files with value 1 are checked,
        with value zero are skipped.
        The default is np.ones(len(BM_FILENAMES),)
    check_path : str or Path
        absolute path where files are stored.
        Default: SYSTEM_DIR
    year : int
        year of the image, e.g. 2016

    Returns
    -------
    files_exist : numpy array
        Boolean array that denotes if the required files exist.
    """
    if required_files is None:
        required_files = np.ones(len(BM_FILENAMES),)
    if np.size(required_files) < np.count_nonzero(BM_FILENAMES):
        required_files = np.ones(np.count_nonzero(BM_FILENAMES),)
        LOGGER.warning('The parameter \'required_files\' was too short and '
                       'is ignored.')
    if isinstance(check_path, str):
        check_path = Path(check_path)
    if not check_path.is_dir():
        raise ValueError(f'The given path does not exist: {check_path}')
    files_exist = np.zeros(np.count_nonzero(BM_FILENAMES),)
    for num_check, name_check in enumerate(BM_FILENAMES):
        if required_files[num_check] == 0:
            continue
        curr_file = check_path.joinpath(name_check %(year))
        if curr_file.is_file():
            files_exist[num_check] = 1

    if sum(files_exist) == sum(required_files):
        LOGGER.debug('Found all required satellite data (%s files) in folder %s',
                     int(sum(required_files)), check_path)
    elif sum(files_exist) == 0:
        LOGGER.info('No satellite files found locally in %s', check_path)
    else:
        LOGGER.debug('Not all satellite files available. '
                     'Found %d out of %d required files in %s',
                     int(sum(files_exist)), int(sum(required_files)), check_path)

    return files_exist

def download_nl_files(req_files=np.ones(len(BM_FILENAMES),),
                      files_exist=np.zeros(len(BM_FILENAMES),),
                      dwnl_path=SYSTEM_DIR, year=2016):
    """Attempts to download nightlight files from NASA webpage.

    Parameters
    ----------
    req_files : numpy array, optional
        Boolean array which indicates the files required (0-> skip, 1-> download).
            The default is np.ones(len(BM_FILENAMES),).
    files_exist : numpy array, optional
        Boolean array which indicates if the files already
            exist locally and should not be downloaded (0-> download, 1-> skip).
            The default is np.zeros(len(BM_FILENAMES),).
    dwnl_path : str or path, optional
        Download directory path. The default is SYSTEM_DIR.
    year : int, optional
        Data year to be downloaded. The default is 2016.

    Raises
    ------
    ValueError
    RuntimeError

    Returns
    -------
    dwnl_path : str or path
        Download directory path.
    """

    if (len(req_files) != len(files_exist)) or (len(req_files) != len(BM_FILENAMES)):
        raise ValueError('The given arguments are invalid. req_files and '
                         'files_exist must both be as long as there are files to download'
                         ' (' + str(len(BM_FILENAMES)) + ').')
    if not Path(dwnl_path).is_dir():
        raise ValueError(f'The folder {dwnl_path} does not exist. Operation aborted.')
    if np.all(req_files == files_exist):
        LOGGER.debug('All required files already exist. No downloads necessary.')
        return dwnl_path
    try:
        for num_files in range(0, np.count_nonzero(BM_FILENAMES)):
            if req_files[num_files] == 0 or files_exist[num_files] == 1:
                continue # file already available or not required
            path_check = False
            # loop through different possible URLs defined in CONFIG:
            for url in CONFIG.exposures.litpop.nightlights.nasa_sites.list():
                try: # control for ValueError due to wrong URL
                    curr_file = url.str() + BM_FILENAMES[num_files] %(year)
                    LOGGER.info('Attempting to download file from %s', curr_file)
                    path_check = download_file(curr_file, download_dir=dwnl_path)
                    break # leave loop if sucessful
                except ValueError as err:
                    value_err = err
            if path_check: # download succesful
                continue
            raise ValueError("Download failed, check URLs in " +
                             "CONFIG.exposures.litpop.nightlights.nasa_sites! \n Last " +
                             "error message: \n" + value_err.args[0])

    except Exception as exc:
        raise RuntimeError('Download failed. Please check the network '
            'connection and whether filenames are still valid.') from exc
    return dwnl_path

def load_nasa_nl_shape_single_tile(geometry, path, layer=0):
    """Read nightlight data from single NASA BlackMarble tile and crop to given shape.

    Parameters
    ----------
    geometry : shape or geometry object
        shape(s) to crop data to in degree lon/lat. for example
        shapely.geometry.Polygon object or from polygon defined in a shapefile.
    path : Path or str
        full path to BlackMarble tif (including filename)
    layer : int, optional
        TIFF-layer to be returned. The default is 0.
        BlackMarble usually comes with 3 layers.

    Returns
    -------
    out_image[layer,:,:] : 2D numpy ndarray
        2d array with data cropped to bounding box of shape
    meta : dict
        rasterio meta
    """
    # open tif source file with raterio:
    with rasterio.open(path, 'r') as src:
        # read cropped data from  source file (src) to np.ndarray:
        out_image, transform = rasterio.mask.mask(src, [geometry], crop=True)
        meta = src.meta
        meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": transform})
    return out_image[layer,:,:], meta

def load_nightlight_nasa(bounds, req_files, year):
    """Get nightlight from NASA repository that contain input boundary.

    Note: Legacy for BlackMarble, not required for litpop module

    Parameters
    ----------
    bounds : tuple
        min_lon, min_lat, max_lon, max_lat
    req_files : np.array
        array with flags for NASA files needed
    year : int
        nightlight year

    Returns
    -------
    nightlight : sparse.csr_matrix
    coord_nl : np.array
    """
    # TODO: argument req_files is not used in this function

    coord_min = np.array([-90, -180]) + NASA_RESOLUTION_DEG / 2
    coord_h = np.full((2,), NASA_RESOLUTION_DEG)

    min_lon, min_lat, max_lon, max_lat = bounds
    bounds_mat = np.array([[min_lat, min_lon], [max_lat, max_lon]])
    global_idx = (bounds_mat - coord_min[None]) / coord_h[None]
    global_idx[0, :] = np.floor(global_idx[0, :])
    global_idx[1, :] = np.ceil(global_idx[1, :])
    tile_size = np.array(NASA_TILE_SIZE)

    nightlight = []
    for idx, fname in enumerate(BM_FILENAMES):
        tile_coord = np.array([1 - idx % 2, idx // 2])
        extent = global_idx - (tile_coord * tile_size)[None]
        if np.any(extent[1, :] < 0) or np.any(extent[0, :] >= NASA_TILE_SIZE):
            # this tile does not intersect the specified bounds
            continue
        extent = np.int64(np.clip(extent, 0, tile_size[None] - 1))
        # pylint: disable=unsubscriptable-object
        im_nl, _ = read_bm_file(SYSTEM_DIR, fname %(year))
        im_nl = np.flipud(im_nl)
        im_nl = sparse.csc.csc_matrix(im_nl)
        im_nl = im_nl[extent[0, 0]:extent[1, 0] + 1, extent[0, 1]:extent[1, 1] + 1]
        nightlight.append((tile_coord, im_nl))

    tile_coords = np.array([n[0] for n in nightlight])
    shape = tile_coords.max(axis=0) - tile_coords.min(axis=0) + 1
    nightlight = np.array([n[1] for n in nightlight]).reshape(shape, order='F')
    nightlight = sparse.bmat(np.flipud(nightlight), format='csr')

    coord_nl = np.vstack([coord_min, coord_h]).T
    coord_nl[:, 0] += global_idx[0, :] * coord_h[:]

    return nightlight, coord_nl


def read_bm_file(bm_path, filename):
    """Reads a single NASA BlackMarble GeoTiff and returns the data. Run all required checks first.

    Note: Legacy for BlackMarble, not required for litpop module

    Parameters
    ----------
    bm_path : str
        absolute path where files are stored.
    filename : str
        filename of the file to be read.

    Returns
    -------
    arr1 : array
        Raw BM data
    curr_file : gdal GeoTiff File
        Additional info from which coordinates can be calculated.
    """
    path = Path(bm_path, filename)
    LOGGER.debug('Importing%s.', path)
    if not path.exists():
        raise FileNotFoundError('Invalid path: check that the path to BlackMarble file is correct.')
    curr_file = gdal.Open(str(path))
    arr1 = curr_file.GetRasterBand(1).ReadAsArray()
    return arr1, curr_file

def unzip_tif_to_py(file_gz):
    """Unzip image file, read it, flip the x axis, save values as pickle
    and remove tif.

    Parameters
    ----------
    file_gz : str
        file fith .gz format to unzip

    Returns
    -------
    fname : str
        file_name of unzipped file
    nightlight : sparse.csr_matrix
    """
    LOGGER.info("Unzipping file %s.", file_gz)
    file_name = Path(Path(file_gz).stem)
    with gzip.open(file_gz, 'rb') as f_in:
        with file_name.open('wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    nightlight = sparse.csc_matrix(plt.imread(file_name))
    # flip X axis
    nightlight.indices = -nightlight.indices + nightlight.shape[0] - 1
    nightlight = nightlight.tocsr()
    file_name.unlink()
    file_path = SYSTEM_DIR.joinpath(file_name.stem + ".p")
    save(file_path, nightlight)

    return file_name, nightlight

def untar_noaa_stable_nightlight(f_tar_ini):
    """Move input tar file to SYSTEM_DIR and extract stable light file.
    Returns absolute path of stable light file in format tif.gz.

    Parameters
    ----------
    f_tar_ini : str
        absolute path of file

    Returns
    -------
    f_tif_gz : str
        path of stable light file
    """
    # move to SYSTEM_DIR
    f_tar_dest = SYSTEM_DIR.joinpath(Path(f_tar_ini).name)
    shutil.move(f_tar_ini, f_tar_dest)
    # extract stable_lights.avg_vis.tif
    with tarfile.open(f_tar_ini) as tar_file:
        extract_name = [name for name in tar_file.getnames()
                        if name.endswith('stable_lights.avg_vis.tif.gz')]
        if len(extract_name) == 0:
            raise ValueError('No stable light intensities for selected year and satellite '
                            f'in file {f_tar_ini}')
        if len(extract_name) > 1:
            LOGGER.warning('found more than one potential intensity file in %s %s',
                           f_tar_ini, extract_name)
        tar_file.extract(extract_name[0], SYSTEM_DIR)
    return SYSTEM_DIR.joinpath(extract_name[0])


def load_nightlight_noaa(ref_year=2013, sat_name=None):
    """Get nightlight luminosites. Nightlight matrix, lat and lon ordered
    such that nightlight[1][0] corresponds to lat[1], lon[0] point (the image
    has been flipped).

    Parameters
    ----------
    ref_year : int, optional
        reference year. The default is 2013.
    sat_name : str, optional
        satellite provider (e.g. 'F10', 'F18', ...)

    Returns
    -------
    nightlight : sparse.csr_matrix
    coord_nl : np.array
    fn_light : str
    """
    # NOAA's URL used to retrieve nightlight satellite images:
    noaa_url = CONFIG.exposures.litpop.nightlights.noaa_url.str()
    if sat_name is None:
        fn_light = str(SYSTEM_DIR.joinpath('*' +
                             str(ref_year) + '*.stable_lights.avg_vis'))
    else:
        fn_light = str(SYSTEM_DIR.joinpath(sat_name +
                             str(ref_year) + '*.stable_lights.avg_vis'))
    # check if file exists in SYSTEM_DIR, download if not
    if glob.glob(fn_light + ".p"):
        fn_light = glob.glob(fn_light + ".p")[0]
        with open(fn_light, 'rb') as f_nl:
            nightlight = pickle.load(f_nl)
    elif glob.glob(fn_light + ".tif.gz"):
        fn_light = glob.glob(fn_light + ".tif.gz")[0]
        fn_light, nightlight = unzip_tif_to_py(fn_light)
    else:
        # iterate over all satellites if no satellite name provided
        if sat_name is None:
            ini_pre, end_pre = 18, 9
            for pre_i in np.arange(ini_pre, end_pre, -1):
                url = noaa_url + 'F' + str(pre_i) + str(ref_year) + '.v4.tar'
                try:
                    file_down = download_file(url, download_dir=SYSTEM_DIR)
                    break
                except ValueError:
                    pass
            if 'file_down' not in locals():
                raise ValueError(f'Nightlight for reference year {ref_year} not available. '
                                 'Try a different year.')
        else:
            url = noaa_url + sat_name + str(ref_year) + '.v4.tar'
            try:
                file_down = download_file(url, download_dir=SYSTEM_DIR)
            except ValueError as err:
                raise ValueError(f'Nightlight intensities for year {ref_year} and satellite'
                                 f' {sat_name} do not exist.') from err
        fn_light = untar_noaa_stable_nightlight(file_down)
        fn_light, nightlight = unzip_tif_to_py(fn_light)

    # first point and step
    coord_nl = np.empty((2, 2))
    coord_nl[0, :] = [NOAA_BORDER[1], NOAA_RESOLUTION_DEG]
    coord_nl[1, :] = [NOAA_BORDER[0], NOAA_RESOLUTION_DEG]

    return nightlight, coord_nl, fn_light
