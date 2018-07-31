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
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy import ndimage
import pandas as pd
import requests
import shapely.vectorized
from pint import UnitRegistry
from PIL import Image
from cartopy.io import shapereader

from climada.entity.exposures.base import Exposures
from climada.util.files_handler import download_file
from climada.util.constants import SYSTEM_DIR, ONE_LAT_KM
from climada.util.config import CONFIG
from climada.util.save import save
from climada.util.coordinates import coord_on_land
from . import nightlight as nl_utils

# solve version problem in pandas-datareader-0.6.0. see:
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-
# importerror-cannot-import-name-is-list-like
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

LOGGER = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1e9

NOAA_SITE = "https://ngdc.noaa.gov/eog/data/web_data/v4composites/"
""" NOAA's URL used to retrieve nightlight satellite images. """

WORLD_BANK_INC_GRP = \
"http://databank.worldbank.org/data/download/site-content/OGHIST.xls"
""" Income group historical data from World bank."""

NOAA_RESOLUTION_DEG = (30*UnitRegistry().arc_second).to(UnitRegistry().deg). \
                       magnitude
""" NOAA nightlights coordinates resolution in degrees. """

NASA_RESOLUTION_DEG = (15*UnitRegistry().arc_second).to(UnitRegistry().deg). \
                       magnitude
""" NASA nightlights coordinates resolution in degrees. """

NOAA_BORDER = (-180, -65, 180, 75)
""" NOAA nightlights border (min_lon, min_lat, max_lon, max_lat) """

DEF_RES_NOAA_KM = 1
""" Default approximate resolution for NOAA NGDC nightlights in km."""

DEF_RES_NASA_KM = 0.5
""" Default approximate resolution for NASA's nightlights in km."""

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
    """Defines exposures from night light intensity, GDP and income group.
    """

    def __init__(self):
        """ Empty initializer. """
        Exposures.__init__(self)

    def set_countries(self, countries, \
        ref_year=CONFIG['entity']['present_ref_year'], res_km=1, \
        sea_res=(0, 1), from_hr=None, **kwargs):
        """ Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list): list of country names (admin0)
            ref_year (int, optional): reference year. Default: present_ref_year
                in configuration.
            res_km (float, optional): approx resolution in km. Default: 1km.
            sea_res (tuple, optional): (sea_coast_km, sea_res_km), where first
                parameter is distance from coast to fill with water and second
                parameter is resolution between sea points
            from_hr (bool, optional)
            kwargs (optional): 'gdp' and 'inc_grp' dictionaries with keys the
                country ISO_alpha3 code. If provided, these are used
        """
        shp_file = shapereader.natural_earth(resolution='10m',
                                             category='cultural',
                                             name='admin_0_countries')
        shp_file = shapereader.Reader(shp_file)
        country_info = country_iso_geom(countries, shp_file)
        fill_econ_indicators(ref_year, country_info, shp_file, **kwargs)

        nightlight, coord_nl, fn_nl, res_fact = get_nightlight(ref_year,\
            country_info, res_km, from_hr)

        # TODO parallize per thread?
        for cntry in country_info.values():
            LOGGER.info('Processing country %s.', cntry[1])
            self.append(self._set_one_country(cntry, nightlight, coord_nl,
                                              fn_nl, res_fact, res_km))

        add_sea(self, sea_res)

    def set_region(self, centroids, ref_year=2013, resolution=1):
        """ Model a specific region given by centroids."""
        # TODO: accept input centroids as well
        raise NotImplementedError

    def fill_centroids(self):
        """ From coordinates information, generate Centroids instance."""
        # add sea in lower resolution
        raise NotImplementedError

    @staticmethod
    def _set_one_country(country_info, nightlight, coord_nl, fn_nl, res_fact,
                         res_km):
        """ Model one country.

        Parameters:
            country_info (lsit): [cntry_id, cnytry_name, cntry_geometry,
                ref_year, gdp, income_group]
            nightlight (np.array): nightlight in 30arcsec ~ 1km resolution.
                Row latitudes, col longitudes
            coord_nl (np.array): nightlight coordinates: [[min_lat, lat_step],
                [min_lon, lon_step]]
            fn_nl (str): file name of considered nightlight with path
            res_fact (float): resampling factor
            res_km (float): wished resolution in km
        """
        exp_bkmrb = BlackMarble()

        _process_land(exp_bkmrb, country_info[2], nightlight, coord_nl,
                      res_fact, res_km)

        _set_econ_indicators(exp_bkmrb, country_info[4], country_info[5])

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
    """ Get country ISO alpha_3, country id (defined as country appearance
    order in natural earth shape file) and country's geometry.

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
    for country_name in country_list:
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
            [country_idx+1, country_tit, list_records[country_idx].geometry]
    return country_info

def get_nightlight(ref_year, country_info, res_km, from_hr=None):
    """ Obtain nightlight from different sources depending on reference year.
    Compute resolution factor used at resampling depending on source.

    Parameters:
        ref_year (int): reference year
        country_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        res_km (float): approx resolution in km.
        from_hr (bool, optional):
    Returns:
        nightlight (sparse.csr_matrix), coord_nl (np.array), fn_nl (str),
        res_fact (float)
    """
    if from_hr is None and ref_year > 2013:
        from_hr = True
    elif from_hr is None and ref_year <= 2013:
        from_hr = False

    if from_hr:
        nl_year = ref_year
        if ref_year > 2013:
            nl_year = 2016
        else:
            nl_year = 2012
        LOGGER.info("Nightlights from NASA's earth observatory for " + \
                    "year %s.", nl_year)
        res_fact = DEF_RES_NASA_KM/res_km
        geom = [info[2] for info in country_info.values()]
        geom = shapely.ops.cascaded_union(geom)
        req_files = nl_utils.check_required_nightlight_files(geom.bounds)
        files_exist, _ = nl_utils.check_nightlight_local_file_exists(req_files)
        nl_utils.download_nightlight_files(req_files, files_exist,
                                           SYSTEM_DIR, nl_year)

        nightlight, coord_nl = load_nightlight_nasa(geom.bounds,
                                                    req_files, nl_year)
        fn_nl = [file.replace('*', str(nl_year)) for idx, file \
                 in enumerate(nl_utils.BM_FILENAMES) if req_files[idx]]
    else:
        nl_year = ref_year
        if ref_year < 1992:
            nl_year = 1992
        elif ref_year > 2013:
            nl_year = 2013
        LOGGER.info("Nightlights from NOAA's earth observation group "+ \
                    "for year %s.", nl_year)
        res_fact = DEF_RES_NOAA_KM/res_km
        # nightlight intensity with 30 arcsec resolution
        nightlight, coord_nl, fn_nl = load_nightlight_noaa(nl_year)

    return nightlight, coord_nl, fn_nl, res_fact

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
        nightlight (sparse.csr_matrix), coord_nl (np.array),
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

    # first point and step
    coord_nl = np.empty((2, 2))
    coord_nl[0, :] = [NOAA_BORDER[1], NOAA_RESOLUTION_DEG]
    coord_nl[1, :] = [NOAA_BORDER[0], NOAA_RESOLUTION_DEG]

    return nightlight, coord_nl, fn_light

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


def load_nightlight_nasa(bounds, req_files, year):
    """ Get nightlight from NASA repository that contain input boundary.

    Parameters:
        bounds (tuple): nmin_lon, min_lat, max_lon, max_lat
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

    prev_idx = -1
    for idx, file in enumerate(nl_utils.BM_FILENAMES):
        if not req_files[idx]:
            continue

        aux_nl = sparse.csc.csc_matrix(plt.imread(os.path.join(
            SYSTEM_DIR, file.replace('*', str(year))))[:, :, 0])
        # flip X axis
        aux_nl.indices = -aux_nl.indices + aux_nl.shape[0] - 1
        aux_nl = aux_nl.tolil()

        aux_bnd = []
        if idx/2 % 4 == in_lon_nb[0]:
            aux_bnd.append(int(in_lon[0] - (idx/2%4)*21600))
        else:
            aux_bnd.append(0)

        if idx % 2 == in_lat_nb[0]:
            aux_bnd.append(in_lat[0] - ((idx+1)%2)*21600)
        else:
            aux_bnd.append(0)

        if idx/2 % 4 == in_lon_nb[1]:
            aux_bnd.append(int(in_lon[1] - (idx/2%4)*21600) + 1)
        else:
            aux_bnd.append(21600)

        if idx % 2 == in_lat_nb[1]:
            aux_bnd.append(in_lat[1] - ((idx+1)%2)*21600 + 1)
        else:
            aux_bnd.append(21600)

        if prev_idx == -1:
            nightlight = sparse.lil.lil_matrix((aux_bnd[3]-aux_bnd[1],
                                                aux_bnd[2]-aux_bnd[0]))
            nightlight = aux_nl[aux_bnd[1]:aux_bnd[3], aux_bnd[0]:aux_bnd[2]]
        elif idx%2 == prev_idx%2:
            nightlight = sparse.hstack([nightlight, \
                aux_nl[aux_bnd[1]:aux_bnd[3], aux_bnd[0]:aux_bnd[2]]])
        else:
            nightlight = sparse.vstack([nightlight, \
                aux_nl[aux_bnd[1]:aux_bnd[3], aux_bnd[0]:aux_bnd[2]]])

        prev_idx = idx

    coord_nl[0, 0] = coord_nl[0, 0] + in_lat[0]*coord_nl[0, 1]
    coord_nl[1, 0] = coord_nl[1, 0] + in_lon[0]*coord_nl[1, 1]

    return nightlight.tocsr(), coord_nl

def _get_income_group(country_info, ref_year, shp_file):
    """ Append country's income group from World Bank's data at a given year,
    or closest year value. If no data, get the natural earth's approximation.

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
                LOGGER.error("No income group for country %s found.",
                             cntry_iso)
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
            close_gdp = cntry_gdp.iloc[ \
                np.abs(years-ref_year).argsort()].dropna()
            close_gdp_val = float(close_gdp.iloc[0].values)
            LOGGER.info("GDP {} {:d}: {:.3e}.".format(cntry_iso, \
                        int(close_gdp.iloc[0].name[1]), close_gdp_val))

        except (ValueError, IndexError):
            list_records = list(shp_file.records())
            for info in list_records:
                if info.attributes['ADM0_A3'] == cntry_iso:
                    close_gdp_val = info.attributes['GDP_MD_EST']
                    close_gdp_year = int(info.attributes['GDP_YEAR'])
            if close_gdp_val == -99.0:
                LOGGER.error("No GDP for country %s found.", cntry_iso)
                raise ValueError
            close_gdp_val *= 1e6
            LOGGER.info("GDP {} {:d}: {:.3e}.".format(cntry_iso, \
                        close_gdp_year, close_gdp_val))

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

def _process_land(exp, geom, nightlight, coord_nl, res_fact, res_km):
    """Model land exposures from nightlight intensities and normalized
    to GDP * (income_group + 1).

    Parameters:
        exp(BlackMarble): BlackMarble instance
        geom (shapely.geometry): geometry of the region to consider
        nightlight (sparse.csr_matrix): nichtlight values
        coord_nl (np.array): nightlight coordinates: [[min_lat, lat_step],
            [min_lon, lon_step]]
        res_fact (float): resampling factor
        res_km (float): wished resolution in km
    """
    in_lat = math.floor((geom.bounds[1] - coord_nl[0, 0])/coord_nl[0, 1]), \
             math.ceil((geom.bounds[3] - coord_nl[0, 0])/coord_nl[0, 1])
    in_lon = math.floor((geom.bounds[0] - coord_nl[1, 0])/coord_nl[1, 1]), \
             math.ceil((geom.bounds[2] - coord_nl[1, 0])/coord_nl[1, 1])

    LOGGER.info('Generating resolution of approx %s km.', res_km)
    nightlight_res = ndimage.zoom(nightlight[in_lat[0]:in_lat[-1]+1, :] \
        [:, in_lon[0]:in_lon[-1]+1].todense(), res_fact)
    lat_res, lon_res = np.mgrid[
        coord_nl[0, 0] + in_lat[0]*coord_nl[0, 1]:
        coord_nl[0, 0] + in_lat[1]*coord_nl[0, 1]:
        complex(0, nightlight_res.shape[0]),
        coord_nl[1, 0] + in_lon[0]*coord_nl[1, 1]:
        coord_nl[1, 0] + in_lon[1]*coord_nl[1, 1]:
        complex(0, nightlight_res.shape[1])]

    on_land = shapely.vectorized.contains(geom, lon_res, lat_res)

    exp.value = nightlight_res[on_land].ravel()
    exp.coord = np.empty((exp.value.size, 2))
    exp.coord[:, 0] = lat_res[on_land]
    exp.coord[:, 1] = lon_res[on_land]

def _set_econ_indicators(exp, gdp, inc_grp):
    """Model land exposures from nightlight intensities and normalized
    to GDP * (income_group + 1)

    Parameters:
        exp(BlackMarble): BlackMarble instance with value = nightlight
        gdp (float): GDP to interpolate in the region
        inc_grp (float): index to weight exposures in the region
    """
    exp.value = np.power(exp.value, 3)
    exp.value = exp.value/exp.value.sum()* gdp * (inc_grp+1)

def add_sea(exp, sea_res):
    """ Add sea to geometry's surroundings with given resolution.

    Parameters:
        exp(BlackMarble): BlackMarble instance
        sea_res (tuple): (sea_coast_km, sea_res_km), where first parameter
            is distance from coast to fill with water and second parameter
            is resolution between sea points
    """
    if sea_res[0] == 0:
        return

    LOGGER.info("Adding sea at %s km resolution and %s km distance from " + \
        "coast.", sea_res[1], sea_res[0])

    sea_res = (sea_res[0]/ONE_LAT_KM, sea_res[1]/ONE_LAT_KM)

    min_lat = max(-90, float(np.min(exp.coord.lat)) - sea_res[0])
    max_lat = min(90, float(np.max(exp.coord.lat)) + sea_res[0])
    min_lon = max(-180, float(np.min(exp.coord.lon)) - sea_res[0])
    max_lon = min(180, float(np.max(exp.coord.lon)) + sea_res[0])

    lat_arr = np.arange(min_lat, max_lat+sea_res[1], sea_res[1])
    lon_arr = np.arange(min_lon, max_lon+sea_res[1], sea_res[1])

    lon_mgrid, lat_mgrid = np.meshgrid(lon_arr, lat_arr)
    lon_mgrid, lat_mgrid = lon_mgrid.ravel(), lat_mgrid.ravel()
    on_land = np.logical_not(coord_on_land(lat_mgrid, lon_mgrid))

    exp.coord = np.array([np.append(exp.coord.lat, lat_mgrid[on_land]), \
        np.append(exp.coord.lon, lon_mgrid[on_land])]).transpose()
    exp.value = np.append(exp.value, lat_mgrid[on_land]*0)
    exp.id = np.arange(1, exp.value.size+1)
    exp.region_id = np.append(exp.region_id, lat_mgrid[on_land]*0-1)
    exp.impact_id = np.ones(exp.value.size, int)
    exp.deductible = np.zeros(exp.value.size)
    exp.cover = exp.value.copy()
