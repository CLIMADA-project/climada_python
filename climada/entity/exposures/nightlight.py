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

Define nightlight reader and cutting functions.
"""
from os import path, getcwd, chdir, remove
import glob
import shutil
import tarfile
import re
import gzip
import pickle
import logging
import math
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from PIL import Image
from pint import UnitRegistry

from climada.util.constants import SYSTEM_DIR
from climada.util.files_handler import download_file
from climada.util.save import save

Image.MAX_IMAGE_PIXELS = 1e9

LOGGER = logging.getLogger(__name__)

NOAA_SITE = "https://ngdc.noaa.gov/eog/data/web_data/v4composites/"
""" NOAA's URL used to retrieve nightlight satellite images. """

NOAA_RESOLUTION_DEG = (30*UnitRegistry().arc_second).to(UnitRegistry().deg). \
                       magnitude
""" NOAA nightlights coordinates resolution in degrees. """

NASA_RESOLUTION_DEG = (15*UnitRegistry().arc_second).to(UnitRegistry().deg). \
                       magnitude
""" NASA nightlights coordinates resolution in degrees. """

NOAA_BORDER = (-180, -65, 180, 75)
""" NOAA nightlights border (min_lon, min_lat, max_lon, max_lat) """

NASA_SITE = 'https://www.nasa.gov/specials/blackmarble/*/tiles/georeferrenced/'
"""NASA nightlight web url."""

BM_FILENAMES = ['BlackMarble_*_A1_geo_gray.tif',
                'BlackMarble_*_A2_geo_gray.tif',
                'BlackMarble_*_B1_geo_gray.tif',
                'BlackMarble_*_B2_geo_gray.tif',
                'BlackMarble_*_C1_geo_gray.tif',
                'BlackMarble_*_C2_geo_gray.tif',
                'BlackMarble_*_D1_geo_gray.tif',
                'BlackMarble_*_D2_geo_gray.tif'
               ]
"""Nightlight NASA files which generate the whole earth when put together."""

def check_required_nl_files(bbox, *coords):
    """ Determines which of the satellite pictures are neccessary for
        a certain bounding box (e.g. country)

    Parameters:
        either:
            bbox (1x4 tuple): bounding box from shape (min_lon, min_lat,
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
            #check if bbox is valid
            if (np.size(bbox) != 4) or (bbox[0] > bbox[2]) \
            or (bbox[1] > bbox[3]):
                LOGGER.error('Invalid bounding box supplied.')
                raise ValueError
            else:
                min_lon, min_lat, max_lon, max_lat = bbox
        else:
            if (len(coords) != 3) or (not coords[1] > bbox) \
            or (not coords[2] > coords[0]):
                LOGGER.error('Invalid coordinates supplied.')
                raise ValueError
            else:
                min_lon = bbox
                min_lat, max_lon, max_lat = coords
    except:
        raise ValueError('Invalid coordinates supplied. Please either ' + \
            ' deliver a bounding box or the coordinates defining the ' + \
            ' bounding box separately.')

    # longitude first. The width of all tiles is 90 degrees
    tile_width = 90
    req_files = np.zeros(np.count_nonzero(BM_FILENAMES),)

    # determine the staring tile
    first_tile_lon = min(np.floor((min_lon-(-180))/tile_width), 3) #"normalise" to zero
    last_tile_lon = min(np.floor((max_lon-(-180))/tile_width), 3)

    # Now latitude. The height of all tiles is the same as the height.
    # Note that for this analysis returns an index which follows from North to South oritentation.
    first_tile_lat = min(np.floor(-(min_lat-(90))/tile_width), 1)
    last_tile_lat = min(np.floor(-(max_lat-90)/tile_width), 1)

    for i_lon in range(0, int(len(req_files)/2)):
        if first_tile_lon <= i_lon and last_tile_lon >= i_lon:
            if first_tile_lat == 0 or last_tile_lat == 0:
                req_files[((i_lon))*2] = 1
            if first_tile_lat == 1 or last_tile_lat == 1:
                req_files[((i_lon))*2 + 1] = 1
        else:
            continue
    return req_files


def check_nl_local_file_exists(required_files=np.ones(len(BM_FILENAMES),),
                               check_path=SYSTEM_DIR, year=2016):
    """ Checks if BM Satellite files are avaialbe and returns a vector
    denoting the missing files.

    Parameters:
        check_path (str): absolute path where files are stored.
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
        LOGGER.warning('The parameter \'required_files\' was too short and '+ \
                       'is ignored.')
    if not path.exists(check_path):
        check_path = SYSTEM_DIR
        LOGGER.warning('The given path does not exist and is ignored. %s' + \
                       ' is checked instead.', SYSTEM_DIR)
    files_exist = np.zeros(np.count_nonzero(BM_FILENAMES),)
    for num_check, name_check in enumerate(BM_FILENAMES):
        if required_files[num_check] == 0:
            continue
        curr_file = path.join(check_path, name_check)
        curr_file = curr_file.replace('*', str(year))
        if path.isfile(curr_file):
            files_exist[num_check] = 1

    if sum(files_exist) == sum(required_files):
        LOGGER.debug('Found all required satellite data (' +
                     str(int(sum(required_files))) + ' files) in folder ' +
                     check_path)
    elif sum(files_exist) == 0:
        LOGGER.info('No satellite files found locally in %s', check_path)
    else:
        LOGGER.debug('Not all satellite files available. Found ' +
                     str(int(sum(files_exist))) + ' out of ' +
                     str(int(sum(required_files))) + ' required files in ' +
                     check_path)

    return (files_exist, check_path)

def download_nl_files(req_files=np.ones(len(BM_FILENAMES),), \
    files_exist=np.zeros(len(BM_FILENAMES),), dwnl_path=SYSTEM_DIR, year=2016):
    """ Attempts to download nightlight files from NASA webpage.

    Parameters:
        req_files (array): Boolean array which indicates the files
            required for the current operation (0-> skip, 1-> download).
            Can be obtained by check_required_nightlight_files
        files_exists (array): Boolean array which indicates if the files already
            exist locally and should not be downloaded (0-> download, 1-> skip).
            Can be obtained by function check_nightlight_local_file_exists
        dwnl_path (str):

    Returns:
        path_str (str): Absolute path to file storage.
    """
    if (len(req_files) != len(files_exist)) or \
        (len(req_files) != len(BM_FILENAMES)):
        raise ValueError('The given arguments are invalid. req_files and ' + \
            'files_exist must both be as long as there are files to download'+\
            ' (' + str(len(BM_FILENAMES)) + ').')
    if not path.exists(dwnl_path):
        dwnl_path = SYSTEM_DIR
        if not path.exists(dwnl_path):
            raise ValueError('The folder does not exist. Operation aborted.')
        else:
            LOGGER.warning('The given folder does not exist using the ' + \
                'Climada data directory instead.')
    if np.all(req_files == files_exist):
        LOGGER.debug('All required files already exist. ' +
                     'No downloads neccessary.')
        return None
    try:
        curr_wd = getcwd()
        chdir(dwnl_path)
        for num_files in range(0, np.count_nonzero(BM_FILENAMES)):
            if req_files[num_files] == 0:
                continue
            else:
                if files_exist[num_files] == 1:
                    continue
                else:
                    curr_file = NASA_SITE + BM_FILENAMES[num_files]
                    curr_file = curr_file.replace('*', str(year))
                    LOGGER.info('Attempting to download file from %s',
                                curr_file)
                    path_dwn = download_file(curr_file)
                    path_str = path.dirname(path_dwn)
    except:
        chdir(curr_wd)
        raise RuntimeError('Download failed. Please check the network ' + \
            'connection and whether filenames are still valid.')
    return path_str

def load_nightlight_nasa(bounds, req_files, year):
    """ Get nightlight from NASA repository that contain input boundary.

    Parameters:
        bounds (tuple): min_lon, min_lat, max_lon, max_lat
        req_files (np.array): array with flags for NASA files needed
        year (int): nightlight year

    Returns:
        nightlight (sparse.csr_matrix), coord_nl (np.array)
    """
    coord_nl = np.empty((2, 2))
    coord_nl[0, :] = [-90+NASA_RESOLUTION_DEG/2, NASA_RESOLUTION_DEG]
    coord_nl[1, :] = [-180+NASA_RESOLUTION_DEG/2, NASA_RESOLUTION_DEG]

    in_lat = math.floor((bounds[1] - coord_nl[0, 0])/coord_nl[0, 1]), \
             math.ceil((bounds[3] - coord_nl[0, 0])/coord_nl[0, 1])
    # Upper (0) or lower (1) latitude range for min and max latitude
    in_lat_nb = (math.floor(in_lat[0]/21600)+1)%2, \
                (math.floor(in_lat[1]/21600)+1)%2

    in_lon = math.floor((bounds[0] - coord_nl[1, 0])/coord_nl[1, 1]), \
             math.ceil((bounds[2] - coord_nl[1, 0])/coord_nl[1, 1])
    # 0, 1, 2, 3 longitude range for min and max longitude
    in_lon_nb = math.floor(in_lon[0]/21600), math.floor(in_lon[1]/21600)

    nightlight = sparse.lil.lil_matrix([])
    idx_info = [0, -1, False] # idx, prev_idx and row added flag
    for idx, file in enumerate(BM_FILENAMES):
        idx_info[0] = idx
        if not req_files[idx]:
            continue

        with Image.open(path.join(SYSTEM_DIR, file.replace('*', str(year)))) \
        as im_nl:
            cut_nl_nasa(im_nl.getchannel(0), idx_info, nightlight, in_lat, in_lon,
                        in_lat_nb, in_lon_nb)

        idx_info[1] = idx

    coord_nl[0, 0] = coord_nl[0, 0] + in_lat[0]*coord_nl[0, 1]
    coord_nl[1, 0] = coord_nl[1, 0] + in_lon[0]*coord_nl[1, 1]

    return nightlight.tocsr(), coord_nl

def cut_nl_nasa(aux_nl, idx_info, nightlight, in_lat, in_lon, in_lat_nb,
                in_lon_nb):
    """Cut nasa's nightlight image piece (1-8) to bounds and append to final
    matrix.

    Parameters:
        aux_nl (PIL.Image): nasa's nightlight part (1-8)
        idx_info (list): idx (0-7), prev_idx (0-7) and row_added flag (bool).
        nightlight (sprse.lil_matrix): matrix with nightlight that is expanded
        in_lat (tuple): min and max latitude indexes in the whole nasa's image
        in_lon (tuple): min and max longitude indexes in the whole nasa's image
        in_lat_nb (tuple): for min and max latitude, range where they belong
            to: upper (0) or lower (1) row of nasa's images.
        on_lon_nb (tuple): for min and max longitude, range where they belong
            to: 0, 1, 2 or 3 column of nasa's images.
    """
    idx, prev_idx, row_added = idx_info

    aux_nl = sparse.csc.csc_matrix(aux_nl)
    # flip X axis
    aux_nl.indices = -aux_nl.indices + aux_nl.shape[0] - 1

    aux_bnd = []
    # in min lon
    if int(idx/2) % 4 == in_lon_nb[0]:
        aux_bnd.append(int(in_lon[0] - (int(idx/2)%4)*21600))
    else:
        aux_bnd.append(0)

    # in min lat
    if idx % 2 == in_lat_nb[0]:
        aux_bnd.append(in_lat[0] - ((idx+1)%2)*21600)
    else:
        aux_bnd.append(0)

    # in max lon
    if int(idx/2) % 4 == in_lon_nb[1]:
        aux_bnd.append(int(in_lon[1] - (int(idx/2)%4)*21600) + 1)
    else:
        aux_bnd.append(21600)

    # in max lat
    if idx % 2 == in_lat_nb[1]:
        aux_bnd.append(in_lat[1] - ((idx+1)%2)*21600 + 1)
    else:
        aux_bnd.append(21600)

    if prev_idx == -1:
        nightlight.resize((aux_bnd[3]-aux_bnd[1], aux_bnd[2]-aux_bnd[0]))
        nightlight[:, :] = aux_nl[aux_bnd[1]:aux_bnd[3], aux_bnd[0]:aux_bnd[2]]
    elif idx%2 == prev_idx%2 or prev_idx%2 == 1:
        # append horizontally in first rows e.g 0->2 or 1->2
        nightlight.resize((nightlight.shape[0],
                           nightlight.shape[1] + aux_bnd[2]-aux_bnd[0]))
        nightlight[-aux_bnd[3]+aux_bnd[1]:, -aux_bnd[2]+aux_bnd[0]:] = \
            aux_nl[aux_bnd[1]:aux_bnd[3], aux_bnd[0]:aux_bnd[2]]
    else:
        # append vertically in firsts rows and columns e.g 0->1 or 2->3
        if not row_added:
            old_shape = nightlight.shape
            nightlight.resize((old_shape[0] + aux_bnd[3] - aux_bnd[1],
                               old_shape[1]))
            nightlight[-old_shape[0]:, :] = nightlight[:old_shape[0], :]
            idx_info[2] = True
        nightlight[:aux_bnd[3]-aux_bnd[1], -aux_bnd[2]+aux_bnd[0]:] = \
            aux_nl[aux_bnd[1]:aux_bnd[3], aux_bnd[0]:aux_bnd[2]]

def unzip_tif_to_py(file_gz):
    """ Unzip image file, read it, flip the x axis, save values as pickle
    and remove tif.

    Parameters:
        file_gz (str): file fith .gz format to unzip

    Returns:
        str (file_name of unzipped file)
        sparse.csr_matrix (nightlight)
    """
    LOGGER.info("Unzipping file %s.", file_gz)
    file_name = path.splitext(file_gz)[0]
    with gzip.open(file_gz, 'rb') as f_in:
        with open(file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    nightlight = sparse.csc.csc_matrix(plt.imread(file_name))
    # flip X axis
    nightlight.indices = -nightlight.indices + nightlight.shape[0] - 1
    nightlight = nightlight.tocsr()
    remove(file_name)
    file_name = path.splitext(file_name)[0] + ".p"
    save(file_name, nightlight)

    return file_name, nightlight

def untar_noaa_stable_nightlight(f_tar_ini):
    """ Move input tar file to SYSTEM_DIR and extract stable light file.
    Returns absolute path of stable light file in format tif.gz.

    Parameters:
        f_tar_ini (str): absolute path of file

    Returns:
        f_tif_gz (str)
    """
    # move to SYSTEM_DIR
    f_tar_dest = path.abspath(path.join(SYSTEM_DIR,
                                        path.basename(f_tar_ini)))
    shutil.move(f_tar_ini, f_tar_dest)
    # extract stable_lights.avg_vis.tif
    chdir(SYSTEM_DIR)
    tar_file = tarfile.open(f_tar_dest)
    file_contents = tar_file.getnames()
    extract_name = path.splitext(path.basename(f_tar_dest))[0] + \
        '.*stable_lights.avg_vis.tif.gz'
    regex = re.compile(extract_name)
    try:
        extract_name = list(filter(regex.match, file_contents))[0]
    except IndexError:
        LOGGER.error('No stable light intensities for selected year and '
                     'satellite in file %s', f_tar_dest)
        raise ValueError
    try:
        tar_file.extract(extract_name)
    except tarfile.TarError as err:
        LOGGER.error(str(err))
        raise err
    finally:
        tar_file.close()
    remove(f_tar_dest)
    f_tif_gz = path.join(path.abspath(SYSTEM_DIR), extract_name)

    return f_tif_gz

def load_nightlight_noaa(ref_year=2013, sat_name=None):
    """ Get nightlight luminosites. Nightlight matrix, lat and lon ordered
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
        fn_light = path.join(path.abspath(SYSTEM_DIR), '*' + \
            str(ref_year) + '*.stable_lights.avg_vis')
    else:
        fn_light = path.join(path.abspath(SYSTEM_DIR), sat_name + \
            str(ref_year) + '*.stable_lights.avg_vis')
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
                    file_down = download_file(url)
                    break
                except ValueError:
                    pass
            if 'file_down' not in locals():
                LOGGER.error('Nightlight for reference year %s not available. '
                             'Try an other year.', ref_year)
                raise ValueError
        else:
            url = NOAA_SITE + sat_name + str(ref_year) + '.v4.tar'
            try:
                file_down = download_file(url)
            except ValueError:
                LOGGER.error('Nightlight intensities for year %s and satellite'
                             ' %s do not exist.', ref_year, sat_name)
                raise ValueError
        fn_light = untar_noaa_stable_nightlight(file_down)
        fn_light, nightlight = unzip_tif_to_py(fn_light)

    # first point and step
    coord_nl = np.empty((2, 2))
    coord_nl[0, :] = [NOAA_BORDER[1], NOAA_RESOLUTION_DEG]
    coord_nl[1, :] = [NOAA_BORDER[0], NOAA_RESOLUTION_DEG]

    return nightlight, coord_nl, fn_light
