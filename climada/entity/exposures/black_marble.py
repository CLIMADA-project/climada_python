"""
Define BlackMarble class.
"""

__all__ = ['BlackMarble']

import os
import glob
import tarfile
import shutil
import logging
import re
import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from pint import UnitRegistry
from PIL import Image
from cartopy.io import shapereader

from climada.entity.exposures.base import Exposures
from climada.util.files_handler import download_file
from climada.util.constants import SYSTEM_DIR
from climada.util.config import CONFIG
from climada.util.save import save

LOGGER = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1e9

NOAA_SITE = CONFIG['entity']['black_marble']['nightlight_url']
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

class BlackMarble(Exposures):
    """Defines exposures from night light intensity and GDP.
    """

    def __init__(self):
        """ Empty initializer. """
        Exposures.__init__(self)

    def clear(self):
        """Clear and reinitialize all data."""
        super(BlackMarble, self).clear()
        self.admin0_name = ''
        self.admin0_iso3 = ''

    def set_countries(self, countries, ref_year=2013, resolution=1):
        """ Model using values of ref_year.

        Parameters:
            ref_year (int): reference year
        """
        country_isos = country_iso(countries)
        # nightlight intensity for selected country with 30 arcsec resolution
        nightlight, nl_lat, nl_lon, fn_light = load_nightlight(ref_year)
        in_lat, in_lon = cut_nightlight_country(country_isos, nl_lat, nl_lon)
#        nl_cntry = transform_nightlight(nightlight, in_lat, in_lon)
#        nl_cntry, lat_cntry, lon_cnrty = resample_resolution(nl_lat, nl_lon,
#            in_lat, in_lon, nl_cntry, resolution)
#        # obtain GDP data in selected country
#        gdp_cntry = load_gdp(count_iso, ref_year)
#        # Interpolate GDP over nightlight map
#        self.coord, self.value = interpol_light_gdp(nl_cntry, lat_cntry, lon_cnrty,
#                                                    gdp_cntry)

        # set other variables
        self.ref_year = ref_year
        self.tag.description = str(ref_year)
        self.tag.file_name = fn_light
        self.value_unit = 'USD'
        self.impact_id = np.ones(self.value.size, int)
        self.id = np.arange(1, self.value.size + 1)
        self.admin0_name = countries
        self.admin0_iso3 = country_isos

    def set_region(self, centroids, ref_year=2013, resolution=1):
        """ Model a specific region given by centroids."""
        # TODO: accept input centroids as well
        raise NotImplementedError

    def fill_centroids(self):
        """ From coordinates information, generate Centroids instance."""
        # add sea in lower resolution
        raise NotImplementedError

def country_iso(country_name):
    """ Compute country ISO alpha_3 name from country name.

    Parameters:
        country (str): country name

    Returns:
        country_iso (str): ISO alpha_3 country name
    """
    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_0_countries')
    shp = shapereader.Reader(shp_file)

    tit_name = country_name.title()
    countries = {}
    options = []
    for info in shp.records():
        std_name = info.attributes['ADMIN'].title()
        if tit_name in std_name:
            options.append(std_name)
            std_name = tit_name
        countries[std_name] = info.attributes['ADM0_A3']
    iso_val = countries.get(tit_name)
    if iso_val is None:
        LOGGER.error('Country %s not found. Possible options: %s', tit_name,
                     countries.keys())
        raise ValueError
    elif len(options) > 1:
        LOGGER.info('More than one possible country: %s.', str(options))
        LOGGER.info('Considered country: %s, with ISO: %s.', options.pop(),
                    iso_val)
    return iso_val

def untar_stable_nightlight(f_tar_ini):
    """ Move input tar file to SYSTEM_DIR and extract stable light file.
    Returns absolute path of stable light file in format tif.gz.

    Parameters:
        f_tar_ini (str): absolute path of file

    Returns:
        f_tif_gz (str)
    """
    # move to SYSTEM_DIR
    f_tar_dest = os.path.abspath(os.path.join(SYSTEM_DIR,
                                              os.path.basename(f_tar_ini)))
    shutil.move(f_tar_ini, f_tar_dest)
    # extract stable_lights.avg_vis.tif
    os.chdir(SYSTEM_DIR)
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
    f_tif_gz = os.path.join(os.path.abspath(SYSTEM_DIR), extract_name)

    return f_tif_gz

def _unzip_tif_to_py(file_gz):
    """ Unzip image file, read it, save values as pickle and remove tif.

    Parameters:
        file_gz (str): file fith .gz format to unzip

    Returns:
        str (file_name of unzipped file)
        sparse.csr_matrix (nightlight)
    """
    LOGGER.info("Unzipping file %s.", file_gz)
    file_name = os.path.splitext(file_gz)[0]
    with gzip.open(file_gz, 'rb') as f_in:
        with open(file_name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    nightlight = sparse.csr.csr_matrix(plt.imread(file_name))
    os.remove(file_name)
    file_name = os.path.splitext(file_name)[0] + ".p"
    save(file_name, nightlight)

    return file_name, nightlight

def load_nightlight(ref_year=2013, sat_name=None):
    """ Get nightlight luminosites.

    Parameters:
        ref_year (int): reference year
        sat_name (str, optional): satellite provider (e.g. 'F10', 'F18', ...)

    Returns:
        nightlight (sparse.csr_matrix), lat (np.array), lon (np.array),
        fn_light (str)
    """
    if sat_name is None:
        fn_light = os.path.join(os.path.abspath(SYSTEM_DIR), '*' + \
                            str(ref_year) + '*.stable_lights.avg_vis')
    else:
        fn_light = os.path.join(os.path.abspath(SYSTEM_DIR), sat_name + \
                   str(ref_year) + '*.stable_lights.avg_vis')
    # check if file exists in SYSTEM_DIR, download if not
    if glob.glob(fn_light + ".p"):
        fn_light = glob.glob(fn_light + ".p")[0]
        with open(fn_light, 'rb') as f_nl:
            nightlight = pickle.load(f_nl)
    elif glob.glob(fn_light + ".tif.gz"):
        fn_light = glob.glob(fn_light + ".tif.gz")[0]
        fn_light, nightlight = _unzip_tif_to_py(fn_light)
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
        fn_light, nightlight = _unzip_tif_to_py(fn_light)

    # The products are 30 arc second grids, spanning -180 to 180 degrees
    # longitude and -65 to 75 degrees latitude.
    lat = np.linspace(MIN_LAT + NOAA_RESOLUTION_DEG, MAX_LAT,
                      nightlight.shape[0])
    lon = np.linspace(MIN_LON + NOAA_RESOLUTION_DEG, MAX_LON,
                      nightlight.shape[1])

    return nightlight, lat, lon, fn_light

def cut_nightlight_country(country_isos, nl_lat, nl_lon):
    """Get nightlight indexes for every country.

    Parameters:
        country_isos (list(str)): ISO alpha_3 country names
        nl_lat (np.array): latitudes of earth's nightlight
        nl_lon (np.array): longitudes of earth's nightlight

    Returns:
        in_lat (dict(np.array(int))), in_lon (dict(np.array(int)))
    """
    shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                         name='admin_0_countries')
    shp = shapereader.Reader(shp_file)
    all_geom = {}
    for info in shp.records():
        if info.attributes['ADM0_A3'] in country_isos:
            all_geom[info.attributes['ADM0_A3']] = info.geometry
    if not all_geom:
        LOGGER.error('Countries %s do not exist.', country_isos)
        raise ValueError

    in_lat, in_lon = {}, {}
    for cntry_iso, cntry_geom in all_geom.items():
        in_lon[cntry_iso] = np.argwhere(np.logical_and( \
            nl_lon >= cntry_geom.bounds[0], nl_lon <= cntry_geom.bounds[2]))
        in_lat[cntry_iso] = np.argwhere(np.logical_and( \
            nl_lat >= cntry_geom.bounds[1], nl_lat <= cntry_geom.bounds[3]))

    return in_lat, in_lon

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
