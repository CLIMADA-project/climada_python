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

Functions to download weather forecasts of the ICON weather forecast model 
from the German Weather Service DWD (opendata.dwd.de)
"""

__all__ = [
    'download_icon_grib',
    'delete_icon_grib',
    'download_icon_centroids_file',
]


import logging

import urllib
from pathlib import Path
import numpy as np

import requests

import bz2

from climada.util.config import CONFIG
from climada.util.files_handler import download_file

LOGGER = logging.getLogger(__name__)



def download_icon_grib(run_date,
                        model_name='icon-eu-eps',
                        parameter_name='vmax_10m',
                        max_lead_time=None,
                        download_dir=None):
    """download the gribfiles of a weather forecast run for a certain 
    weather parameter from opendata.dwd.de/weather/nwp/. 
    
    Parameters:
        run_date (datetime): The starting timepoint of the forecast run 
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        parameter_name (str): the name of the meteorological parameter 
            written as it appears in the folder structure in 
            opendata.dwd.de/weather/nwp/
        max_lead_time (int): number of hours for which files should be 
            downloaded, will default to maximum available data
        download_dir: (str or Path): directory where the downloaded files
            should be saved in
    
    Returns:
        file_names (list): a list of filenames that link to all just 
            downloaded or available files from the forecast run, defined by
            the input parameters
    """
    
    LOGGER.info(('Downloading icon grib files of model ' +
                 model_name + ' for parameter ' + parameter_name +
                 ' with starting date ' + run_date.strftime('%Y%m%d%H') +
                 '.'))
    
    url, file_name, lead_times = _create_icon_grib_name(run_date,
                                                        model_name,
                                                        parameter_name,
                                                        max_lead_time)
    
    #download all files
    file_names = []
    for lead_i in lead_times:
        file_name_i = file_name.format(lead_i=lead_i)
        try:
            full_path_name_i = download_file(url + file_name_i,
                                             download_dir=download_dir)
        except ValueError as err:
            LOGGER.error('Error while downloading %s.', file_name_i)
            request = requests.get(url)
            if request.status_code == 200:
                raise FileNotFoundError(('Error while downloading %s.', file_name_i)) from err
            else:
                raise err
        file_names.append(full_path_name_i)            
        
    return file_names
        
    
 
def delete_icon_grib(run_date,
                     model_name='icon-eu-eps',
                     parameter_name='vmax_10m',
                     max_lead_time=None,
                     download_dir=None):
    """delete the downloaded gribfiles of a weather forecast run for a 
    certain weather parameter from opendata.dwd.de/weather/nwp/. 
    
    Parameters:
        run_date (datetime): The starting timepoint of the forecast run 
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        parameter_name (str): the name of the meteorological parameter 
            written as it appears in the folder structure in 
            opendata.dwd.de/weather/nwp/
        max_lead_time (int): number of hours for which files should be 
            deleted, will default to maximum available data
        download_dir (str or Path): directory where the downloaded files
            are stored at the moment
    """    
    
    url, file_name, lead_times = _create_icon_grib_name(run_date,
                                                        model_name,
                                                        parameter_name,
                                                        max_lead_time)
    download_path = CONFIG.local_data.save_dir.dir() if download_dir is None else Path(download_dir)
    #delete all files
    for lead_i in lead_times:
        file_name_i = file_name.format(lead_i=lead_i)
        full_path_name_i = download_path.absolute().joinpath(file_name_i)
        if full_path_name_i.exists():
          full_path_name_i.unlink()
        else:
          LOGGER.warning('File %s does not exist and could not be deleted.',
                         full_path_name_i)


def _create_icon_grib_name(run_date,
                           model_name='icon-eu-eps',
                           parameter_name='vmax_10m',
                           max_lead_time=None):
    """create all parameters to download or delete gribfiles of a weather 
    forecast run for a certain weather parameter from 
    opendata.dwd.de/weather/nwp/. 
    
    Parameters:
        run_date (datetime): The starting timepoint of the forecast run 
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        parameter_name (str): the name of the meteorological parameter 
            written as it appears in the folder structure in 
            opendata.dwd.de/weather/nwp/
        max_lead_time (int): number of hours for which files should be 
            selected, will default to maximum available data
    
    Returns:
        url (str): url where the gribfiles are stored on opendata.dwd.de
        file_name (str): filenames of gribfiles (lead_time missing)
        lead_times (np.array): array of integers representing the leadtimes
            in hours, which are available for download
    """    
    # define defaults of the url for each model and parameter combination
    if (model_name=='icon-eu-eps') & (parameter_name=='vmax_10m'):
        file_extension = '_europe_icosahedral_single-level_'
        max_lead_time_default = 120 # maximum available data
        lead_times = np.concatenate((np.arange(1,49),
                                     np.arange(51,73,3),
                                     np.arange(78,121,6)
                                     ))
    else:
        LOGGER.error(('Download for model ' + model_name + 
                      ' and parameter ' + parameter_name + 
                      ' is not yet implemented. Please define ' +
                      'the default values in the code first.'))
        raise ValueError
        
    # create the url for download
    url = ('https://opendata.dwd.de/weather/nwp/' +
           model_name +
           '/grib/' +
           run_date.strftime('%H') + 
           '/' + 
           parameter_name +
           '/')
    file_name = (model_name +
                 file_extension +
                 run_date.strftime('%Y%m%d%H') +
                 '_' +
                 '{lead_i:03}' +
                 '_' +
                 parameter_name +
                 '.grib2.bz2')
    
    
    # define the leadtimes
    if  not max_lead_time:
        max_lead_time = max_lead_time_default
    elif max_lead_time > max_lead_time_default:
        LOGGER.warning(('Parameter max_lead_time ' +
                        str(max_lead_time) + ' is bigger than maximum ' +
                        'available files. max_lead_time is adjusted to ' +
                        str(max_lead_time_default)))
        max_lead_time = max_lead_time_default
    lead_times = lead_times[lead_times<=max_lead_time]
    
    return url, file_name, lead_times


def download_icon_centroids_file(model_name='icon-eu-eps',
                                 download_dir = None):
    """ create centroids based on netcdf files provided by dwd, links 
    found here: 
    https://www.dwd.de/DE/leistungen/opendata/neuigkeiten/opendata_dez2018_02.html
    https://www.dwd.de/DE/leistungen/opendata/neuigkeiten/opendata_aug2020_01.html
            
    Parameters:
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        download_dir (str or Path): directory where the downloaded files
            should be saved in
    Returns:
        file_name (str): absolute path and filename of the downloaded 
            and decompressed netcdf file
    """
    
    # define url and filename
    url = 'https://opendata.dwd.de/weather/lib/cdo/'
    if model_name=='icon-eu-eps':
        file_name = 'icon_grid_0028_R02B07_N02.nc.bz2'
    elif model_name=='icon-eu':
        file_name = 'icon_grid_0024_R02B06_G.nc.bz2'
    elif model_name=='icon-d2-eps' or model_name=='icon-d2':
        file_name = 'icon_grid_0047_R19B07_L.nc.bz2'
    else:
        LOGGER.error(('Creation of centroids for the icon model ' +
                      model_name + 'is not implemented. Please define ' +
                      'the default values in the code first.'))
        raise ValueError
    download_path = CONFIG.local_data.save_dir.dir() if download_dir is None else Path(download_dir)
    bz2_pathfile = download_path.absolute().joinpath(file_name)
    nc_pathfile = bz2_pathfile.with_suffix('')
    
    # download and unzip file
    if not nc_pathfile.exists():
        if not bz2_pathfile.exists():
            try:
                download_file(url + file_name,
                                             download_dir=download_path)
            except ValueError as err:
                LOGGER.error('Error while downloading %s.', url + file_name)
                raise err
        with open(bz2_pathfile, 'rb') as source, open(nc_pathfile, 'wb') as dest:
            dest.write(bz2.decompress(source.read()))
        bz2_pathfile.unlink()
    
    return str(nc_pathfile)
