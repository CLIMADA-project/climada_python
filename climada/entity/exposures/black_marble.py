"""
Define BlackMarble class.
"""

__all__ = ['CountryBlackMarble']

import os
import glob
import tarfile
import shutil
import logging
import re
import gzip
import matplotlib.pyplot as plt
import numpy as np
import pycountry
from pint import UnitRegistry
from PIL import Image

from climada.entity.exposures.base import Exposures
from climada.util.files_handler import download_file
from climada.util.constants import DATA_DIR

LOGGER = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1e9

NOAA_SITE = 'https://ngdc.noaa.gov/eog/data/web_data/v4composites/'
""" URL used to retrieve nightlight satellite images. """

NOAA_RESOLUTION_DEG = (30*UnitRegistry().arc_second).to(UnitRegistry().deg). \
                       magnitude
""" Default coordinates resolution in degrees. """

MIN_LAT = -65
""" Minimum latitude """

MAX_LAT = 75
""" Maximum latitude """

MIN_LON = -180
""" Minimum longitude """

MAX_LON = 180
""" Maximum longitude """

class CountryBlackMarble(Exposures):
    """Defines exposures from night light intensity and GDP.
    """

    def __init__(self):
        """ Empty initializer. """
        Exposures.__init__(self)

    def clear(self):
        """Clear and reinitialize all data."""
        super(CountryBlackMarble, self).clear()
        self.admin0_name = ''
        self.admin0_iso3 = ''

    def set(self, country, ref_year=2013, resolution=1):
        """ Model using values of ref_year.

        Parameters:
            ref_year (int): reference year
        """
        count_iso = country_iso(country)
        # nightlight intensity for selected country with 30 arcsec resolution
#        nightlight, lat, lon, fn_light = load_nightlight(ref_year)
#        nightlight, lat, lon = cut_nightlight_country(nightlight, lat, lon,
#                                                      count_iso)
#        nightlight = transform_nightlight(nightlight, lat, lon)
#        nightlight, lat, lon = resample_resolution(nightlight, lat, lon,
#                                                   resolution)
#        # obtain GDP data in selected country
#        gdp_count = load_gdp(count_iso, ref_year)
#        # Interpolate GDP over nightlight map
#        self.coord, self.value = interpol_light_gdp(nightlight, lat, lon,
#                                                    gdp_count)

        # set other variables
        self.ref_year = ref_year
        self.tag.description = str(ref_year)
#        self.tag.file_name = fn_light
        self.value_unit = 'USD'
        self.impact_id = np.ones(self.value.size, int)
        self.id = np.arange(1, self.value.size + 1)
        self.admin0_name = country
        self.admin0_iso3 = count_iso

    def fill_centroids(self):
        """ From coordinates information, generate Centroids instance."""

def country_iso(country_name):
    """ Compute country ISO alpha_3 name from country name.

    Parameters:
        country (str): country name

    Returns:
        country_iso (str): ISO alpha_3 country name
    """
    countries = {}
    for country in pycountry.countries:
        countries[country.name] = country.alpha_3
    return countries.get(country_name)

def untar_stable_nightlight(f_tar_ini):
    """ Move input tar file to DATA_DIR and extract stable light file.
    Returns absolute path of stable light file in format tif.gz.

    Parameters:
        f_tar_ini (str): absolute path of file

    Returns:
        f_tif_gz (str)
    """
    # move to DATA_DIR
    f_tar_dest = os.path.abspath(os.path.join(DATA_DIR,
                                              os.path.basename(f_tar_ini)))
    shutil.move(f_tar_ini, f_tar_dest)
    # extract stable_lights.avg_vis.tif
    os.chdir(DATA_DIR)
    tar_file = tarfile.open(f_tar_dest)
    file_contents = tar_file.getnames()
    extract_name = os.path.splitext(os.path.basename(f_tar_dest))[0] + \
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
    os.remove(f_tar_dest)
    f_tif_gz = os.path.join(os.path.abspath(DATA_DIR), extract_name)
    return f_tif_gz

def load_nightlight(ref_year=2013, sat_name=None):
    """ Get nightlight luminosites.

    Parameters:
        ref_year (int): reference year
        sat_name (str, optional): satellite provider (e.g. 'F10', 'F18', ...)

    Returns:
        nightlight (np.array), lat (np.array), lon (np.array), fn_light (str)
    """
    # check if file exists in DATA_DIR, download if not
    if sat_name is None:
        fn_light = os.path.join(os.path.abspath(DATA_DIR), '*' + str(ref_year)
                                + '*.stable_lights.avg_vis.tif.gz')
    else:
        fn_light = os.path.join(os.path.abspath(DATA_DIR), sat_name + \
                   str(ref_year) + '*.stable_lights.avg_vis.tif.gz')
    # check if file exists, download if not
    if glob.glob(fn_light):
        fn_light = glob.glob(fn_light)[0]
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
        fn_light = untar_stable_nightlight(file_down)
        LOGGER.info('New file stored %s', fn_light)

    fn_tif = os.path.splitext(fn_light)[0]
    with gzip.open(fn_light, 'rb') as f_in:
        with open(fn_tif, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    nightlight = plt.imread(fn_tif)
    os.remove(fn_tif)

    # The products are 30 arc second grids, spanning -180 to 180 degrees
    # longitude and -65 to 75 degrees latitude.
    lat = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT,
                      nightlight.shape[0])
    lon = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON,
                      nightlight.shape[1])

    return nightlight, lat, lon, fn_light

def cut_nightlight_country():
    """ Get nightlight luminosites.

    Parameters:
        country_iso (str): ISO alpha_3 country name
        lat (np.array), lon (np.array)
        ref_year (int): reference year

    Returns:
        nightlight_country ()
    """
#    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
#                                         name='admin_0_countries')
#    shp = shapereader.Reader(shp_file)
#    cntry_geom = []
#    for info in shp.records():
#        if info.attributes['ADM0_A3'] == country_iso:
#            cntry_geom = info.geometry
#    if not cntry_geom:
#        LOGGER.error('Country %s does not exist.', country_iso)
#        raise ValueError
#
#    for lat_p, lon_p in zip(lat, lon):
#        Point((lat_p, lon_p))
#    return nightlight_cut, lat_cut, lon_cut
    raise NotImplementedError

def transform_nightlight():
    """ Transform nightlight intensity using a polynomial transformation."""
    return NotImplementedError

def resample_resolution():
    """ Change resolution."""
    raise NotImplementedError

def load_gdp():
    """ Get GDP data. Exactly GDP * (income_group + 1)

    Parameters:
        country_iso (str): ISO alpha_3 country name
        ref_year (int): reference year

    Returns:
        gdp_count (float)
    """
    raise NotImplementedError

def interpol_light_gdp():
    """ Interpolate GDP data over input nightlight intensities.

    Parameters:
        light_count (): nightlight map
        gdp_count (float): GDP value to interpolate

    Returns:
        coord (np.array), value (np.array)
    """
    raise NotImplementedError
