# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:25:07 2018

@author: Dario
"""
from os import path, getcwd, chdir
import logging
import glob
import numpy as np

from climada.util.constants import SYSTEM_DIR
from climada.util.files_handler import download_file

LOGGER = logging.getLogger(__name__)

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

def check_required_nightlight_files(bbox, *coords):
    """ Determines which of the satellite pictures are neccessary for
        a certain bounding box (e.g. country)

    Parameters:
        either:
            bbox (1x4 tuple): bounding box from shape (min_lon, min_lat, max_lon, max_lat)
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
            if (np.size(bbox) != 4) or (bbox[0] > bbox[2]) or (bbox[1] > bbox[3]):
                LOGGER.error('Invalid bounding box supplied.')
                raise ValueError
            else:
                min_lon, min_lat, max_lon, max_lat = bbox
        else:
            if (len(coords) != 3) or (not coords[1] > bbox) or (not coords[2] > coords[0]):
                LOGGER.error('Invalid coordinates supplied.')
                raise ValueError
            else:
                min_lon = bbox
                min_lat, max_lon, max_lat = coords
    except:
        raise ValueError('Invalid coordinates supplied. Please either deliver \
                         a bounding box or the coordinates defining the bounding box\
                         separately.')

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


def check_nightlight_local_file_exists(required_files=np.ones(np.count_nonzero(BM_FILENAMES),),\
                                  check_path=SYSTEM_DIR):
    """ Checks if BM Satellite files are avaialbe and returns a vector denoting the missing files.

    Parameters:
        check_path (str): absolute path where files are stored. Default: SYSTEM_DIR
        required_files (array): Boolean array of dimension (8,) with which some files can
            be skipped. Only files with value 1 are checked, with value zero are skipped.

    Returns:
        files_exist (array): Denotes if the all required files exist (Boolean values)
    """
    if np.size(required_files) < np.count_nonzero(BM_FILENAMES):
        required_files = np.ones(np.count_nonzero(BM_FILENAMES),)
        LOGGER.warning('The parameter \'required_files\' was too short and is ignored.')
    if not path.exists(check_path):
        check_path = SYSTEM_DIR
        LOGGER.warning('The given path does not exist and is ignored. ' +\
                       SYSTEM_DIR + ' is checked instead.')
    files_exist = np.zeros(np.count_nonzero(BM_FILENAMES),)
    for num_check, name_check in enumerate(BM_FILENAMES):
        if required_files[num_check] == 0:
            continue
        curr_file = path.join(check_path, name_check)
        if glob.glob(curr_file):
            files_exist[num_check] = 1

    if sum(files_exist) == sum(required_files):
        LOGGER.info('Found all required satellite data (' + str(int(sum(required_files))) \
                                                        + ' files) in folder ' + check_path)
    elif sum(files_exist) == 0:
        LOGGER.info('No satellite files found locally in %s', check_path)
    else:
        LOGGER.info('Not all satellite files available. Found ' + str(int(sum(files_exist))) \
                    + ' out of ' + str(int(sum(required_files))) \
                    + ' required files in ' + check_path)

    return (files_exist, check_path)


def download_nightlight_files(req_files=np.ones(np.count_nonzero(BM_FILENAMES),), \
                      files_exist=np.zeros(np.count_nonzero(BM_FILENAMES),), 
                      dwnl_path=SYSTEM_DIR,
                      year=2016):
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
    if (len(req_files) != len(files_exist)) or (len(req_files) != len(BM_FILENAMES)):
        raise ValueError('The given arguments are invalid. req_files and files_exist '\
                         'must both be as long as there are files to download ('\
                                                + str(len(BM_FILENAMES)) + ').')
    if not path.exists(dwnl_path):
        dwnl_path = SYSTEM_DIR
        if not path.exists(dwnl_path):
            raise ValueError('The folder does not exist. Operation aborted.')
        else:
            LOGGER.warn('The given folder does not exist using the Climada data \
                        directory instead.')
    if np.all(req_files == files_exist):
        LOGGER.info('All required files already exist. No downloads neccessary.')
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
                    curr_file.replace('*', str(year))
                    LOGGER.info('Attempting to download file from %s', curr_file)
                    path_dwn = download_file(curr_file)
                    path_str = path.dirname(path_dwn)
    except:
        chdir(curr_wd)
        raise RuntimeError('Download failed. Please check the network\
                                           connection and whether filenames are still valid.')
    return path_str
