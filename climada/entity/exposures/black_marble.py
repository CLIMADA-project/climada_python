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
import pandas as pd
import requests
import shapely.vectorized
from pint import UnitRegistry
from PIL import Image
from cartopy.io import shapereader

from climada.entity.exposures.base import Exposures
from climada.util.files_handler import download_file
from climada.util.constants import SYSTEM_DIR
from climada.util.config import CONFIG
from climada.util.save import save
from climada.util.coordinates import GridPoints

# solve version problem in pandas-datareader-0.6.0. see:
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-
# importerror-cannot-import-name-is-list-like
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

LOGGER = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1e9

NOAA_SITE = CONFIG['entity']['black_marble']['nl_noaa_url']
""" NOAA's URL used to retrieve nightlight satellite images. """

WORLD_BANK_INC_GRP = \
"http://databank.worldbank.org/data/download/site-content/OGHIST.xls"
""" Income group historical data from World bank."""

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

DEF_RES_KM = 1
""" Default approximate resolution in km."""

INCOME_GRP_WB_TABLE = {'L' : 1, # low income
                       'LM': 2, # lower middle income
                       'UM': 3, # upper middle income
                       'H' : 4, # high income
                       '..': np.nan # no data
                      }
""" Meaning of values of world banks' historical table on income groups. """

INCOME_GRP_NE_TABLE = {5: 1, # Low income
                       4: 2, # Lower middle income
                       3: 3, # Upper middle income
                       2: 4, # High income: nonOECD
                       1: 4  # High income: OECD
                      }
""" Meaning of values of natural earth's income groups. """

class BlackMarble(Exposures):
    """Defines exposures from night light intensity and GDP.
    """

    def __init__(self):
        """ Empty initializer. """
        Exposures.__init__(self)

    def set_countries(self, countries, \
        ref_year=CONFIG['entity']['present_ref_year'], res_km=DEF_RES_KM, \
        **kwargs):
        """ Model countries using values at reference year.

        Parameters:
            countries (list): list of country names (admin0)
            ref_year (int): reference year
            res_km (float): approx resolution in km
            kwargs (optional): 'gdp' and 'inc_grp' dictionaries with keys the
                country ISO_alpha3 code. If provided, these are used
        """
        shp_file = shapereader.natural_earth(resolution='10m', \
            category='cultural', name='admin_0_countries')
        shp_file = shapereader.Reader(shp_file)
        country_info = country_iso_geom(countries, shp_file)
        # nightlight intensity for selected country with 30 arcsec resolution
        nightlight, lat_nl, lon_nl, fn_nl = load_nightlight_noaa(ref_year)
        fill_econ_indicators(ref_year, country_info, shp_file, **kwargs)

        # TODO parallize per thread?
        for cntry in country_info.values():
            LOGGER.info('Processing country %s', cntry[1])
            self.append(self._set_one_country(cntry, res_km, nightlight,
                                              lon_nl, lat_nl, fn_nl))

    def set_region(self, centroids, ref_year=2013, resolution=1):
        """ Model a specific region given by centroids."""
        # TODO: accept input centroids as well
        raise NotImplementedError

    def fill_centroids(self):
        """ From coordinates information, generate Centroids instance."""
        # add sea in lower resolution
        raise NotImplementedError

    @staticmethod
    def _set_one_country(country_info, res_km, nightlight, lon_nl, lat_nl,
                         fn_nl):
        """ Model one country.

        Parameters:
            country_info (lsit): [cntry_id, cnytry_name, cntry_geometry,
                ref_year, gdp, income_group]
            res_km (float): approx resolution in km
            nightlight (np.array): nightlight in 30arcsec ~ 1km resolution.
                Row latitudes, col longitudes
            lat_nl (np.array): latitudes of nightlight matrix (its first dim)
            lon_nl (np.array): longitudes of nightlight matrix (its second dim)
            fn_light (str): file name of considered nightlight with path
        """
        exp_bkmrb = BlackMarble()

        in_lat, in_lon, lat_mgrid, lon_mgrid, on_land = _process_land( \
            exp_bkmrb, country_info[2], country_info[4], country_info[5], \
            nightlight, lat_nl, lon_nl)

        _resample_land(exp_bkmrb, res_km, lat_nl, lon_nl, in_lat, in_lon,
                       on_land)
        _add_surroundings(exp_bkmrb, lat_mgrid, lon_mgrid, on_land)

        exp_bkmrb.id = np.arange(1, exp_bkmrb.value.size+1)
        exp_bkmrb.region_id = np.ones(exp_bkmrb.value.shape) * country_info[0]
        exp_bkmrb.impact_id = np.ones(exp_bkmrb.value.size, int)
        exp_bkmrb.ref_year = country_info[3]
        exp_bkmrb.tag.description = ("{} {:d} GDP: {:.3e} income group: {:d}"+\
            "\n").format(country_info[1], country_info[3], \
            country_info[4], country_info[5])
        exp_bkmrb.tag.file_name = fn_nl
        exp_bkmrb.value_unit = 'USD'

        return exp_bkmrb

def country_iso_geom(country_list, shp_file):
    """ Compute country ISO alpha_3 name from country name.

    Parameters:
        country_list (list(str)): list with country names
        shp_file (cartopy.io.shapereader.Reader): shape file

    Returns:
        country_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
    """
    countries_shp = {}
    list_records = list(shp_file.records())
    for info_idx, info in enumerate(list_records):
        std_name = info.attributes['ADMIN'].title()
        countries_shp[std_name] = info_idx

    country_info = dict()
    for country_id, country_name in enumerate(country_list):
        country_tit = country_name.title()
        country_idx = countries_shp.get(country_tit)
        if country_idx is None:
            options = [country_opt for country_opt in countries_shp
                       if country_tit in country_opt]
            if not options:
                options = list(countries_shp.keys())
            LOGGER.error('Country %s not found. Possible options: %s',
                         country_name, options)
            raise ValueError
        country_info[list_records[country_idx].attributes['ADM0_A3']] = \
            [country_id+1, country_tit, list_records[country_idx].geometry]
    return country_info

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

def load_nightlight_noaa(ref_year=2013, sat_name=None):
    """ Get nightlight luminosites. Nightlight matrix, lat and lon ordered
    such that nightlight[1][0] corresponds to lat[1], lon[0] point (the image
    has been flipped).

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

def fill_econ_indicators(ref_year, country_info, shp_file, **kwargs):
    """ Get GDP and income group per country in reference year, or it closest
    one. Source: world bank. Natural earth repository used when missing data.
    Modifies country info with values [country id, country name,
    country geometry, ref_year, gdp, income_group].

    Parameters:
        ref_year (int): reference year
        country_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        kwargs (optional): 'gdp' and 'inc_grp' dictionaries with keys the
            country ISO_alpha3 code. If provided, these are used
    """
    for cntry_iso, cntry_val in country_info.items():
        cntry_val.append(ref_year)

    if 'gdp' not in kwargs:
        _get_gdp(country_info, ref_year, shp_file)
    else:
        for cntry_iso, cntry_val in country_info.items():
            cntry_val.append(kwargs['gdp'][cntry_iso])

    if 'inc_grp' not in kwargs:
        _get_income_group(country_info, ref_year, shp_file)
    else:
        for cntry_iso, cntry_val in country_info.items():
            cntry_val.append(kwargs['inc_grp'][cntry_iso])

def _get_income_group(country_info, ref_year, shp_file):
    """ Append country's income group from World Bank's data at a given year, or
    closest year value. If no data, get the natural earth's approximation.

    Parameters:
        country_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        ref_year (int): reference year
        shp_file (cartopy.io.shapereader.Reader): shape file with INCOME_GRP
            attribute for every country.
    """
    # check if file with income groups exists in SYSTEM_DIR, download if not
    fn_ig = os.path.join(os.path.abspath(SYSTEM_DIR), 'OGHIST.xls')
    if not glob.glob(fn_ig):
        file_down = download_file(WORLD_BANK_INC_GRP)
        shutil.move(file_down, fn_ig)
    dfr_wb = pd.read_excel(fn_ig, 'Country Analytical History', skiprows=5)
    dfr_wb = dfr_wb.drop(dfr_wb.index[0:5]).set_index('Unnamed: 0')
    dfr_wb = dfr_wb.replace(INCOME_GRP_WB_TABLE.keys(),
                            INCOME_GRP_WB_TABLE.values())
    for cntry_iso, cntry_val in country_info.items():
        try:
            cntry_dfr = dfr_wb.loc[cntry_iso]
            # select closest not nan value to ref_year
            close_inc = cntry_dfr.iloc[np.abs( \
                    np.array(cntry_dfr.index[1:])-ref_year).argsort()+1].dropna()
            close_inc_val = int(close_inc.iloc[0])
            LOGGER.info('Income group %s %s: %s.', cntry_iso,
                        close_inc.index[0], close_inc_val)

        except (KeyError, IndexError):
            # take value from natural earth repository
            list_records = list(shp_file.records())
            for info in list_records:
                if info.attributes['ADM0_A3'] == cntry_iso:
                    close_inc = info.attributes['INCOME_GRP']
            try:
                close_inc_val = INCOME_GRP_NE_TABLE[int(close_inc[0])]
            except (KeyError, IndexError):
                LOGGER.error("No income group for country %s found.", cntry_iso)
                raise ValueError
            LOGGER.info('Income group %s: %s.', cntry_iso, close_inc_val)

        cntry_val.append(close_inc_val)

def _get_gdp(country_info, ref_year, shp_file):
    """ Append country's GDP from World Bank's data at a given year, or
    closest year value. If no data, get the natural earth's approximation.

    Parameters:
        country_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        ref_year (int): reference year
        shp_file (cartopy.io.shapereader.Reader): shape file with INCOME_GRP
            attribute for every country.
    """
    wb_gdp_ind = 'NY.GDP.MKTP.CD'
    for cntry_iso, cntry_val in country_info.items():
        try:
            cntry_gdp = wb.download(indicator=wb_gdp_ind, country=cntry_iso,
                                    start=1960, end=2030)
            years = np.array([int(year) \
                for year in cntry_gdp.index.get_level_values('year')])
            close_gdp = cntry_gdp.iloc[np.abs(years-ref_year).argsort()].dropna()
            close_gdp_val = float(close_gdp.iloc[0].values)
            LOGGER.info("GDP {} {:d}: {:.3e}".format(cntry_iso, \
                        int(close_gdp.iloc[0].name[1]), close_gdp_val))

        except (ValueError, IndexError):
            list_records = list(shp_file.records())
            for info in list_records:
                if info.attributes['ADM0_A3'] == cntry_iso:
                    close_gdp_val = info.attributes['GDP_MD_EST']
            if close_gdp_val == -99.0:
                LOGGER.error("No GDP for country %s found.", cntry_iso)
                raise ValueError
            close_gdp_val *= 1e6
            LOGGER.info("GDP {}: {:.3e}".format(cntry_iso, close_gdp_val))

        except requests.exceptions.ConnectionError:
            LOGGER.error('Connection error: check your internet connection.')
            raise ConnectionError

        cntry_val.append(close_gdp_val)

def _unzip_tif_to_py(file_gz):
    """ Unzip image file, read it, flip the x axis, save values as pickle
    and remove tif.

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
    nightlight = sparse.csc.csc_matrix(plt.imread(file_name))
    # flip X axis
    nightlight.indices = -nightlight.indices + nightlight.shape[0] - 1
    nightlight = nightlight.tocsr()
    os.remove(file_name)
    file_name = os.path.splitext(file_name)[0] + ".p"
    save(file_name, nightlight)

    return file_name, nightlight

def _process_land(exp, geom, gdp, income, nightlight, lat_nl, lon_nl):
    """Model land exposures from nightlight intensities and normalized
    to GDP * (income_group + 1).

    Parameters:
        exp(BlackMarble): BlackMarble instance
        geom (shapely.geometry): geometry of the region to consider
        gdp (float): GDP to interpolate in the region
        income (float): index to weight exposures in the region
        nightlight (sparse.csr_matrix): nichtlight values
        lat_nl (np.array): latitudes of nightlight matrix (its first dim)
        lon_nl (np.array): longitudes of nightlight matrix (its second dim)

    Returns:
        in_lat (tuple) indexes of latitude range in nightlight,
        in_lon (tuple) indexes of longitude range in nightlight,
        lat_mgrid (np.array) meshgrid with latitudes in in_lat range,
        lat_mgrid (np.array) meshgrid with longitudes in in_lat range,
        on_land (np.array) matrix with onland flags in in_lat, in_lon range
    """
    in_lat = np.argwhere(np.logical_and(lat_nl >= geom.bounds[1],
                                        lat_nl <= geom.bounds[3]))
    in_lat = (max(0, in_lat[0][0] - 1),
              min(in_lat[-1][0] + 1, lat_nl.size - 1))
    in_lon = np.argwhere(np.logical_and(lon_nl >= geom.bounds[0],
                                        lon_nl <= geom.bounds[2]))
    in_lon = (max(0, in_lon[0][0] - 1),
              min(in_lon[-1][0] + 1, lon_nl.size - 1))
    lat_mgrid, lon_mgrid = np.mgrid[
        lat_nl[in_lat[0]]:lat_nl[in_lat[1]]:complex(0, in_lat[1]-in_lat[0]+1),
        lon_nl[in_lon[0]]:lon_nl[in_lon[1]]:complex(0, in_lon[1]-in_lon[0]+1)]
    on_land = shapely.vectorized.contains(geom, lon_mgrid, lat_mgrid)

    exp.value = np.power(np.asarray(nightlight[in_lat[0]:in_lat[-1]+1, :] \
        [:, in_lon[0]:in_lon[-1]+1][on_land]).ravel(), 3)
    exp.value = exp.value/exp.value.sum()*gdp*(income+1)
    exp.coord = np.empty((exp.value.size, 2))
    exp.coord[:, 0] = lat_mgrid[on_land]
    exp.coord[:, 1] = lon_mgrid[on_land]

    return in_lat, in_lon, lat_mgrid, lon_mgrid, on_land

def _resample_land(exp, res_km, lat_nl, lon_nl, in_lat, in_lon, on_land):
    """ Resample exposures values to input resolution.

    Parameters:
        exp(BlackMarble): BlackMarble instance
        res_km (float): wished resolution in km
        lat_nl (np.array): latitudes of nightlight matrix (its first dim)
        lon_nl (np.array): longitudes of nightlight matrix (its second dim)
        in_lat (tuple): indexes of latitude range in nightlight
        in_lon (tuple): indexes of longitude range in nightlight
        on_land (np.array): matrix with onland flags in in_lat, in_lon range
    """
    if res_km == DEF_RES_KM:
        return
    if res_km > DEF_RES_KM:
        amp_fact = int(res_km/DEF_RES_KM)
        LOGGER.info('Generating resolution of approx %s km', \
                    amp_fact*DEF_RES_KM)

        new_grid = np.meshgrid(lon_nl[in_lon[0]:in_lon[-1]+1:amp_fact],
                               lat_nl[in_lat[0]:in_lat[-1]+1:amp_fact])
        new_onland = on_land[::amp_fact, ::amp_fact]

        new_grid = GridPoints(np.array(
            [new_grid[1][new_onland].ravel(),
             new_grid[0][new_onland].ravel()])).transpose()

        assigned = new_grid.resample_nn(exp.coord, threshold=5*res_km)
        value_new = np.zeros((new_grid.shape[0],))
        for coord_idx, coord_ass in enumerate(assigned):
            value_new[coord_ass] += exp.value[coord_idx]

        exp.coord = new_grid
        exp.value = value_new
    else:
        # bilinear interpolation
        raise NotImplementedError

def _add_surroundings(exp, lat_mgrid, lon_mgrid, on_land):
    """ Add surroundings of country in a resolution of 50 km.

    Parameters:
        exp(BlackMarble): BlackMarble instance
        lat_mgrid (np.array) meshgrid with latitudes in in_lat range,
        lat_mgrid (np.array) meshgrid with longitudes in in_lat range,
        on_land (np.array) matrix with onland flags in in_lat, in_lon range
    """
    surr_lat = lat_mgrid[np.logical_not(on_land)].ravel()[::50]
    surr_lon = lon_mgrid[np.logical_not(on_land)][::50]
    exp.value = np.append(exp.value, surr_lat*0)
    exp.coord = np.array([np.append(exp.coord.lat, surr_lat), \
        np.append(exp.coord.lon, surr_lon)]).transpose()
