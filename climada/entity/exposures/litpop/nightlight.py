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
import rasterio
from pathlib import Path
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from PIL import Image
from shapefile import Shape

from climada.util import ureg
from climada.util.constants import SYSTEM_DIR
from climada.util.files_handler import download_file
from climada.util.save import save

Image.MAX_IMAGE_PIXELS = 1e9

LOGGER = logging.getLogger(__name__)

NOAA_SITE = "https://ngdc.noaa.gov/eog/data/web_data/v4composites/"
"""NOAA's URL used to retrieve nightlight satellite images."""

NOAA_RESOLUTION_DEG = (30 * ureg.arc_second).to(ureg.deg).magnitude
"""NOAA nightlights coordinates resolution in degrees."""

NASA_RESOLUTION_DEG = (15 * ureg.arc_second).to(ureg.deg).magnitude
"""NASA nightlights coordinates resolution in degrees."""

NASA_TILE_SIZE = (21600, 21600)
"""NASA nightlights tile resolution."""

NOAA_BORDER = (-180, -65, 180, 75)
"""NOAA nightlights border (min_lon, min_lat, max_lon, max_lat)"""

NASA_SITE = 'https://www.nasa.gov/specials/blackmarble/*/tiles/georeferrenced/'
"""NASA nightlight web url."""

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

BM_YEARS = [2016, 2012] # list of available years with data, please update.

def get_required_nl_files(bounds, *coords):
    """Determines which of the satellite pictures are necessary for
        a certain bounding box (e.g. country)

    Parameters:
        either:
            bounds (1x4 tuple): bounding box from shape (min_lon, min_lat,
                 max_lon, max_lat)
        or:
            min_lon (float): (=min_lon) Western-most point in decimal degrees
            min_lat (float): Southern-most point in decimal degrees
            max_lon (float): Eastern-most point in decimal degrees
            max_lat (float): Northern-most point in decimal degrees

    Returns:
        req_files (array): Array indicating the required files for the current
            operation with a Boolean value (1: file is required, 0: file is not
            required).
    """
    try:
        if not coords:
            # check if bbox is valid
            if (np.size(bounds) != 4) or (bounds[0] > bounds[2]) \
            or (bounds[1] > bounds[3]):
                raise ValueError('Invalid bounding box supplied.')
            else:
                min_lon, min_lat, max_lon, max_lat = bounds
        else:
            if (len(coords) != 3) or (not coords[1] > bounds) \
            or (not coords[2] > coords[0]):
                raise ValueError('Invalid coordinates supplied.')
            else:
                min_lon = bounds
                min_lat, max_lon, max_lat = coords
    except Exception as exc:
        raise ValueError('Invalid coordinates supplied. Please either '
                         ' deliver a bounding box or the coordinates defining the '
                         ' bounding box separately.') from exc

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
        if first_tile_lon <= i_lon and last_tile_lon >= i_lon:
            if first_tile_lat == 0 or last_tile_lat == 0:
                req_files[((i_lon)) * 2] = 1
            if first_tile_lat == 1 or last_tile_lat == 1:
                req_files[((i_lon)) * 2 + 1] = 1
        else:
            continue
    return req_files


def check_nl_local_file_exists(required_files=np.ones(len(BM_FILENAMES),),
                               check_path=SYSTEM_DIR, year=2016):
    """Checks if BM Satellite files are avaialbe and returns a vector
    denoting the missing files.

    Parameters:
        check_path (str or Path): absolute path where files are stored.
            Default: SYSTEM_DIR
        required_files (array): Boolean array of dimension (8,) with which
            some files can be skipped. Only files with value 1 are checked,
            with value zero are skipped
        year (int): year of the image

    Returns:
        files_exist (array): Denotes if the all required files exist
            (Boolean values)
    """
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

    return (files_exist, check_path)

def download_nl_files(req_files=np.ones(len(BM_FILENAMES),),
                      files_exist=np.zeros(len(BM_FILENAMES),),
                      dwnl_path=SYSTEM_DIR, year=2016):
    """Attempts to download nightlight files from NASA webpage.

    Parameters:
        req_files (array): Boolean array which indicates the files
            required for the current operation (0-> skip, 1-> download).
            Can be obtained by check_required_nightlight_files
        files_exists (array): Boolean array which indicates if the files already
            exist locally and should not be downloaded (0-> download, 1-> skip).
            Can be obtained by function check_nightlight_local_file_exists
        dwnl_path (str):

    Returns:
        path_str (Path): Path to download directory.
    """
    if (len(req_files) != len(files_exist)) or (len(req_files) != len(BM_FILENAMES)):
        raise ValueError('The given arguments are invalid. req_files and '
                         'files_exist must both be as long as there are files to download'
                         ' (' + str(len(BM_FILENAMES)) + ').')
    if not Path(dwnl_path).is_dir():
        raise ValueError(f'The folder {dwnl_path} does not exist. Operation aborted.')
    if np.all(req_files == files_exist):
        LOGGER.debug('All required files already exist. '
                     'No downloads necessary.')
        return dwnl_path
    try:
        for num_files in range(0, np.count_nonzero(BM_FILENAMES)):
            if req_files[num_files] == 0:
                continue
            else:
                if files_exist[num_files] == 1:
                    continue
                else:
                    curr_file = NASA_SITE + BM_FILENAMES[num_files] %(2016)
                    LOGGER.info('Attempting to download file from %s',
                                curr_file)
                    download_file(curr_file, download_dir=dwnl_path)
    except Exception as exc:
        raise RuntimeError('Download failed. Please check the network '
            'connection and whether filenames are still valid.') from exc
    return dwnl_path

def load_nasa_nl_shape_single_tile(geometry, path, layer=0): # TODO: manually tested but no tests exist yet
    """Read nightlight data from single NASA BlackMarble tile
    and crop to given shape.

    Parameters
    ----------
    geometry : shape(s) to crop data to in degree lon/lat.
        for example shapely.geometry.Polygon(s) object or
        from polygon defined in a shapefile.
    path : Path or str
        full path to BlackMarble tif (including filename)
    layer : int
        TIF-layer to be returned. The default is 0.
        BlackMarble usually comes with 3 layers.

    Returns
    -------
    out_image[layer,:,:] : 2D numpy ndarray with cropped data
    meta : dict containing meta data 
    """
    # open tif source file with raterio:
    src = rasterio.open(path)
    # read cropped data from  source file (src) to np.ndarray:
    out_image, transform = rasterio.mask.mask(src, [geometry], crop=True)
    meta = src.meta
    meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": transform})
    src.close()
    return out_image[layer,:,:], meta

# TODO: LitPop 2.0
def load_nasa_nl_shape(geometry, reference_year, data_dir=None, dtype=None):
    """Read nightlight data from NASA BlackMarble tiles
    cropped to given shape(s) and combine.
    
    Parameters
    ----------
    geometry : shape(s) to crop data to in degree lon/lat.
        for example shapely.geometry.(Multi)Polygon or shapefile.Shape.
        from polygon defined in a shapefile. The object should have
        attribute 'bounds' or 'points'
    reference_year : int
        target year for nightlight data, e.g. 2016.
    data_dir : Path (optional)
        Path to directory with BlackMarble data.
        The default is SYSTEM_DIR.
    dtype : dtype
        data type for output default 'float32', required for LitPop,
        choose 'int8' for integer.

    Returns
    -------
    results_array : list containing numpy array
    meta : list containing meta data per array
    """
    if data_dir is None:
        data_dir = SYSTEM_DIR
    if dtype is None:
        dtype = 'float32'
    if isinstance(geometry, Shape):
        bounds = geometry.bbox
    else:
        bounds = geometry.bounds

    # get closest available year from reference_year:
    year = min(BM_YEARS, key=lambda x: abs(x - reference_year))
    # determin black marble tiles with coordinates containing the bounds:
    req_files = get_required_nl_files(bounds)
    # check wether required files exist locally:
    check_nl_local_file_exists(required_files=req_files, check_path=data_dir,
                               year=year)
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
    if idx == 0: # only 1 tile required:
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

def unzip_tif_to_py(file_gz):
    """Unzip image file, read it, flip the x axis, save values as pickle
    and remove tif.

    Parameters:
        file_gz (str): file fith .gz format to unzip

    Returns:
        str (file_name of unzipped file)
        sparse.csr_matrix (nightlight)
    """
    LOGGER.info("Unzipping file %s.", file_gz)
    file_name = Path(Path(file_gz).stem)
    with gzip.open(file_gz, 'rb') as f_in:
        with file_name.open('wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    nightlight = sparse.csc.csc_matrix(plt.imread(file_name))
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

    Parameters:
        f_tar_ini (str): absolute path of file

    Returns:
        f_tif_gz (str)
    """
    # move to SYSTEM_DIR
    f_tar_dest = SYSTEM_DIR.joinpath(Path(f_tar_ini).name)
    shutil.move(f_tar_ini, f_tar_dest)
    # extract stable_lights.avg_vis.tif
    tar_file = tarfile.open(f_tar_ini)
    extract_name = [name for name in tar_file.getnames()
                    if name.endswith('stable_lights.avg_vis.tif.gz')]
    if len(extract_name) == 0:
        raise ValueError('No stable light intensities for selected year and satellite '
                         f'in file {f_tar_ini}')
    if len(extract_name) > 1:
        LOGGER.warning('found more than one potential intensity file in %s %s', f_tar_ini, extract_name)
    try:
        tar_file.extract(extract_name[0], SYSTEM_DIR)
    except tarfile.TarError as err:
        raise
    finally:
        tar_file.close()
    f_tif_gz = SYSTEM_DIR.joinpath(extract_name[0])

    return f_tif_gz

def load_nightlight_noaa(ref_year=2013, sat_name=None):
    """Get nightlight luminosites. Nightlight matrix, lat and lon ordered
    such that nightlight[1][0] corresponds to lat[1], lon[0] point (the image
    has been flipped).

    Parameters:
        ref_year (int): reference year
        sat_name (str, optional): satellite provider (e.g. 'F10', 'F18', ...)

    Returns:
        nightlight (sparse.csr_matrix), coord_nl (np.array),
        fn_light (str)
    """
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
                url = NOAA_SITE + 'F' + str(pre_i) + str(ref_year) + '.v4.tar'
                try:
                    file_down = download_file(url, download_dir=SYSTEM_DIR)
                    break
                except ValueError:
                    pass
            if 'file_down' not in locals():
                raise ValueError(f'Nightlight for reference year {ref_year} not available. '
                                 'Try a different year.')
        else:
            url = NOAA_SITE + sat_name + str(ref_year) + '.v4.tar'
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
