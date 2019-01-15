"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Finance functionalities.
"""
__all__ = ['net_present_value', 'income_group', 'gdp']

import os
import glob
import shutil
import logging
import requests
import warnings
import numpy as np
import pandas as pd
from cartopy.io import shapereader

from climada.util.files_handler import download_file
from climada.util.constants import SYSTEM_DIR

# solve version problem in pandas-datareader-0.6.0. see:
# https://stackoverflow.com/questions/50394873/import-pandas-datareader-gives-
# importerror-cannot-import-name-is-list-like
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import wb

LOGGER = logging.getLogger(__name__)

WORLD_BANK_INC_GRP = \
"http://databank.worldbank.org/data/download/site-content/OGHIST.xls"
""" Income group historical data from World bank."""

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

FILE_GWP_WEALTH2GDP_FACTORS = 'WEALTH2GDP_factors_CRI_2016.csv'
""" File with wealth-to-GDP factors from the
Credit Suisse's Global Wealth Report 2017 (household wealth)"""

def _nat_earth_shp(resolution='10m', category='cultural',
                   name='admin_0_countries'):
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category=category, name=name)
    return shapereader.Reader(shp_file)

def net_present_value(years, disc_rates, val_years):
    """Compute net present value.

    Parameters:
        years (np.array): array with the sequence of years to consider.
        disc_rates (np.array): discount rate for every year in years.
        val_years (np.array): chash flow at each year.

    Returns:
        float
    """
    if years.size != disc_rates.size or years.size != val_years.size:
        LOGGER.error('Wrong input sizes %s, %s, %s.', years.size,
                     disc_rates.size, val_years.size)
        raise ValueError

    npv = val_years[-1]
    for val, disc in zip(val_years[-2::-1], disc_rates[-2::-1]):
        npv = val + npv/(1+disc)

    return npv

def income_group(cntry_iso, ref_year, shp_file=None):
    """ Get country's income group from World Bank's data at a given year,
    or closest year value. If no data, get the natural earth's approximation.

    Parameters:
        cntry_iso (str): key = ISO alpha_3 country
        ref_year (int): reference year
        shp_file (cartopy.io.shapereader.Reader, optional): shape file with
            INCOME_GRP attribute for every country. Load Natural Earth admin0
            if not provided.
    """
    try:
        close_year, close_val = world_bank(cntry_iso, ref_year, 'INC_GRP')
    except (KeyError, IndexError):
        # take value from natural earth repository
        close_year, close_val = nat_earth_adm0(cntry_iso, 'INCOME_GRP',
                                               shp_file=shp_file)
    finally:
        LOGGER.info('Income group %s %s: %s.', cntry_iso, close_year, close_val)

    return close_year, close_val

def gdp(cntry_iso, ref_year, shp_file=None):
    """ Get country's GDP from World Bank's data at a given year, or
    closest year value. If no data, get the natural earth's approximation.

    Parameters:
        cntry_iso (str): key = ISO alpha_3 country
        ref_year (int): reference year
        shp_file (cartopy.io.shapereader.Reader, optional): shape file with
            INCOME_GRP attribute for every country. Load Natural Earth admin0
            if not provided.

    Returns:
        float
    """
    try:
        close_year, close_val = world_bank(cntry_iso, ref_year, 'NY.GDP.MKTP.CD')
    except (ValueError, IndexError, requests.exceptions.ConnectionError) \
    as err:
        if isinstance(err, requests.exceptions.ConnectionError):
            LOGGER.warning('Internet connection failed while retrieving GDPs.')
        close_year, close_val = nat_earth_adm0(cntry_iso, 'GDP_MD_EST',
                                               'GDP_YEAR', shp_file)
    finally:
        LOGGER.info("GDP {} {:d}: {:.3e}.".format(cntry_iso, close_year,
                                                  close_val))

    return close_year, close_val

def world_bank(cntry_iso, ref_year, info_ind):
    """ Get country's GDP from World Bank's data at a given year, or
    closest year value. If no data, get the natural earth's approximation.

    Parameters:
        cntry_iso (str): key = ISO alpha_3 country
        ref_year (int): reference year
        info_ind (str): indicator of World Bank, e.g. 'NY.GDP.MKTP.CD'. If
            'INC_GRP', historical income groups from excel file used.

    Returns:
        int, float

    Raises:
        IOError, KeyError, IndexError
    """
    if info_ind != 'INC_GRP':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cntry_gdp = wb.download(indicator=info_ind, \
                country=cntry_iso, start=1960, end=2030)
        years = np.array([int(year) for year in cntry_gdp.index.get_level_values('year')])
        sort_years = np.abs(years-ref_year).argsort()
        close_val = cntry_gdp.iloc[sort_years].dropna()
        close_year = int(close_val.iloc[0].name[1])
        close_val = float(close_val.iloc[0].values)
    else: # income group level
        fn_ig = os.path.join(os.path.abspath(SYSTEM_DIR), 'OGHIST.xls')
        dfr_wb = pd.DataFrame()
        try:
            if not glob.glob(fn_ig):
                file_down = download_file(WORLD_BANK_INC_GRP)
                shutil.move(file_down, fn_ig)
            dfr_wb = pd.read_excel(fn_ig, 'Country Analytical History', skiprows=5)
            dfr_wb = dfr_wb.drop(dfr_wb.index[0:5]).set_index('Unnamed: 0')
            dfr_wb = dfr_wb.replace(INCOME_GRP_WB_TABLE.keys(),
                                    INCOME_GRP_WB_TABLE.values())
        except (IOError, requests.exceptions.ConnectionError) as err:
            LOGGER.error('Internet connection failed while downloading ' +
                         'historical income groups.')
            raise err

        cntry_dfr = dfr_wb.loc[cntry_iso]
        close_val = cntry_dfr.iloc[np.abs( \
            np.array(cntry_dfr.index[1:])-ref_year).argsort()+1].dropna()
        close_year = close_val.index[0]
        close_val = int(close_val.iloc[0])

    return close_year, close_val

def nat_earth_adm0(cntry_iso, info_name, year_name=None, shp_file=None):
    """ Get country's parameter from natural earth's admin0 shape file.

    Parameters:
        cntry_iso (str): key = ISO alpha_3 country
        info_name (str): attribute to get, e.g. 'GDP_MD_EST', 'INCOME_GRP'.
        year_name (str, optional): year name of the info_name in shape file,
            e.g. 'GDP_YEAR'
        shp_file (cartopy.io.shapereader.Reader, optional): shape file with
            INCOME_GRP attribute for every country. Load Natural Earth admin0
            if not provided.

    Returns:
        int, float

    Raises:
        ValueError
    """
    if not shp_file:
        shp_file = _nat_earth_shp('10m', 'cultural', 'admin_0_countries')

    close_val = 0
    close_year = 0
    for info in shp_file.records():
        if info.attributes['ADM0_A3'] == cntry_iso:
            close_val = info.attributes[info_name]
            if year_name:
                close_year = int(info.attributes[year_name])
            break

    if not close_val:
        LOGGER.error("No GDP for country %s found.", cntry_iso)
        raise ValueError

    if info_name == 'GDP_MD_EST':
        close_val *= 1e6
    elif info_name == 'INCOME_GRP':
        close_val = INCOME_GRP_NE_TABLE.get(int(close_val[0]))

    return close_year, close_val

def wealth2gdp(cntry_iso, non_financial = True, ref_year=2016, file_name = FILE_GWP_WEALTH2GDP_FACTORS):
    """ Get country's wealth-to-GDP factor from the
        Credit Suisse's Global Wealth Report 2017 (household wealth).
        Missing value: returns NaN.
        Parameters:
            cntry_iso (str): key = ISO alpha_3 country
            non_financial (boolean): use non-financial wealth (True)
                                     use total wealth (False) 
            ref_year (int): reference year
        Returns:
            float
    """
    fname = os.path.join(SYSTEM_DIR, file_name)
    factors_all_countries = pd.read_csv(fname, sep=',', index_col=None, \
                     header=0, encoding='ISO-8859-1')
    if ref_year != 2016:
        LOGGER.warning('Reference year for the factor to convert GDP to '\
            + 'wealth was set to 2016 because other years have not '\
            + 'been implemented yet.')
        ref_year = 2016
    if non_financial:
        val = factors_all_countries\
            [factors_all_countries.country_iso3 == cntry_iso]\
            ['NFW-to-GDP-ratio'].values[0]
    else:
        val = factors_all_countries\
            [factors_all_countries.country_iso3 == cntry_iso]\
            ['TW-to-GDP-ratio'].values[0]

    val = np.around(val,5)
    return ref_year, val