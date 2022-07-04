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

Finance functionalities.
"""
__all__ = ['net_present_value', 'income_group', 'gdp']

import shutil
import logging
import warnings
import zipfile

from pathlib import Path

import requests
import numpy as np
import pandas as pd
from pandas_datareader import wb
from cartopy.io import shapereader

from climada.util.files_handler import download_file
from climada.util.constants import SYSTEM_DIR


LOGGER = logging.getLogger(__name__)

WORLD_BANK_WEALTH_ACC = \
"https://databank.worldbank.org/data/download/Wealth-Accounts_CSV.zip"
"""Wealth historical data (1995, 2000, 2005, 2010, 2014) from World Bank (ZIP).
    https://datacatalog.worldbank.org/dataset/wealth-accounting
    Includes variable Produced Capital (NW.PCA.TO)"""

FILE_WORLD_BANK_WEALTH_ACC = "Wealth-AccountsData.csv"

WORLD_BANK_INC_GRP = \
"http://databank.worldbank.org/data/download/site-content/OGHIST.xls"
"""Income group historical data from World bank."""

INCOME_GRP_WB_TABLE = {'L': 1,  # low income
                       'LM': 2,  # lower middle income
                       'UM': 3,  # upper middle income
                       'H': 4,  # high income
                       '..': np.nan  # no data
                      }
"""Meaning of values of world banks' historical table on income groups."""

INCOME_GRP_NE_TABLE = {5: 1,  # Low income
                       4: 2,  # Lower middle income
                       3: 3,  # Upper middle income
                       2: 4,  # High income: nonOECD
                       1: 4  # High income: OECD
                      }
"""Meaning of values of natural earth's income groups."""

FILE_GWP_WEALTH2GDP_FACTORS = 'WEALTH2GDP_factors_CRI_2016.csv'
"""File with wealth-to-GDP factors from the
Credit Suisse's Global Wealth Report 2017 (household wealth)"""

def _nat_earth_shp(resolution='10m', category='cultural',
                   name='admin_0_countries'):
    shp_file = shapereader.natural_earth(resolution=resolution,
                                         category=category, name=name)
    return shapereader.Reader(shp_file)

def net_present_value(years, disc_rates, val_years):
    """Compute net present value.

    Parameters
    ----------
    years : np.array
        array with the sequence of years to consider.
    disc_rates : np.array
        discount rate for every year in years.
    val_years : np.array
        chash flow at each year.

    Returns
    -------
    float
    """
    if years.size != disc_rates.size or years.size != val_years.size:
        raise ValueError(f'Wrong input sizes {years.size}, {disc_rates.size}, {val_years.size}.')

    npv = val_years[-1]
    for val, disc in zip(val_years[-2::-1], disc_rates[-2::-1]):
        npv = val + npv / (1 + disc)

    return npv

def income_group(cntry_iso, ref_year, shp_file=None):
    """Get country's income group from World Bank's data at a given year,
    or closest year value. If no data, get the natural earth's approximation.

    Parameters
    ----------
    cntry_iso : str
        key = ISO alpha_3 country
    ref_year : int
        reference year
    shp_file : cartopy.io.shapereader.Reader, optional
        shape file with
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

def gdp(cntry_iso, ref_year, shp_file=None, per_capita=False):
    """Get country's (current value) GDP from World Bank's data at a given year, or
    closest year value. If no data, get the natural earth's approximation.

    Parameters
    ----------
    cntry_iso : str
        key = ISO alpha_3 country
    ref_year : int
        reference year
    shp_file : cartopy.io.shapereader.Reader, optional
        shape file with
        INCOME_GRP attribute for every country. Load Natural Earth admin0
        if not provided.
    per_capita : boolean, optional
        If True, GDP is returned per capita

    Returns
    -------
    float
    """
    if cntry_iso == 'TWN':
        LOGGER.warning('GDP data for TWN is not provided by World Bank. \
                       Instead, IMF data is returned here.')
        close_year, close_val = _gdp_twn(ref_year, per_capita=per_capita)
        return close_year, close_val
    try:
        if per_capita:
            close_year, close_val = world_bank(cntry_iso, ref_year, 'NY.GDP.PCAP.CD')
        else:
            close_year, close_val = world_bank(cntry_iso, ref_year, 'NY.GDP.MKTP.CD')
    except (ValueError, IndexError, requests.exceptions.ConnectionError) as err:
        if isinstance(err, requests.exceptions.ConnectionError):
            LOGGER.warning('Internet connection failed while retrieving GDPs.')
        close_year, close_val = nat_earth_adm0(cntry_iso, 'GDP_MD', 'GDP_YEAR', shp_file)
    LOGGER.info("GDP {} {:d}: {:.3e}.".format(cntry_iso, close_year, close_val))

    return close_year, close_val

def world_bank(cntry_iso, ref_year, info_ind):
    """Get country's GDP from World Bank's data at a given year, or
    closest year value. If no data, get the natural earth's approximation.

    Parameters
    ----------
    cntry_iso : str
        key = ISO alpha_3 country
    ref_year : int
        reference year
    info_ind : str
        indicator of World Bank, e.g. 'NY.GDP.MKTP.CD'. If
        'INC_GRP', historical income groups from excel file used.

    Returns
    -------
    int, float

    Raises
    ------
    IOError, KeyError, IndexError
    """
    if info_ind != 'INC_GRP':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cntry_gdp = wb.download(indicator=info_ind, country=cntry_iso, start=1960, end=2030)
        years = np.array([int(year) for year in cntry_gdp.index.get_level_values('year')])
        sort_years = np.abs(years - ref_year).argsort()
        close_val = cntry_gdp.iloc[sort_years].dropna()
        close_year = int(close_val.iloc[0].name[1])
        close_val = float(close_val.iloc[0].values)
    else:  # income group level
        fn_ig = SYSTEM_DIR.joinpath('OGHIST.xls')
        dfr_wb = pd.DataFrame()
        try:
            if not fn_ig.is_file():
                file_down = download_file(WORLD_BANK_INC_GRP)
                shutil.move(file_down, fn_ig)
            dfr_wb = pd.read_excel(fn_ig, 'Country Analytical History', skiprows=5)
            dfr_wb = dfr_wb.drop(dfr_wb.index[0:5]).set_index('Unnamed: 0')
            dfr_wb = dfr_wb.replace(INCOME_GRP_WB_TABLE.keys(),
                                    INCOME_GRP_WB_TABLE.values())
        except (IOError, requests.exceptions.ConnectionError) as err:
            raise type(err)('Internet connection failed while downloading '
                            'historical income groups: ' + str(err)) from err

        cntry_dfr = dfr_wb.loc[cntry_iso]
        close_val = cntry_dfr.iloc[np.abs(
            np.array(cntry_dfr.index[1:]) - ref_year).argsort() + 1].dropna()
        close_year = close_val.index[0]
        close_val = int(close_val.iloc[0])

    return close_year, close_val

def nat_earth_adm0(cntry_iso, info_name, year_name=None, shp_file=None):
    """Get country's parameter from natural earth's admin0 shape file.

    Parameters
    ----------
    cntry_iso : str
        key = ISO alpha_3 country
    info_name : str
        attribute to get, e.g. 'GDP_MD', 'INCOME_GRP'.
    year_name : str, optional
        year name of the info_name in shape file,
        e.g. 'GDP_YEAR'
    shp_file : cartopy.io.shapereader.Reader, optional
        shape file with
        INCOME_GRP attribute for every country. Load Natural Earth admin0
        if not provided.

    Returns
    -------
    int, float

    Raises
    ------
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
        raise ValueError("No GDP for country %s found." % cntry_iso)

    # the variable name changed in Natural Earth v5.0.0
    if info_name in ['GDP_MD', 'GDP_MD_EST']:
        close_val *= 1e6
    elif info_name == 'INCOME_GRP':
        close_val = INCOME_GRP_NE_TABLE.get(int(close_val[0]))

    return close_year, close_val

def wealth2gdp(cntry_iso, non_financial=True, ref_year=2016,
               file_name=FILE_GWP_WEALTH2GDP_FACTORS):
    """Get country's wealth-to-GDP factor from the
        Credit Suisse's Global Wealth Report 2017 (household wealth).
        Missing value: returns NaN.

    Parameters
    ----------
    cntry_iso : str
        key = ISO alpha_3 country
    non_financial : boolean
        use non-financial wealth (True)
        use total wealth (False)
    ref_year : int
        reference year

    Returns
    -------
    float
    """
    fname = SYSTEM_DIR.joinpath(file_name)
    factors_all_countries = pd.read_csv(fname, sep=',', index_col=None,
                                        header=0, encoding='ISO-8859-1')
    if ref_year != 2016:
        LOGGER.warning('Reference year for the factor to convert GDP to '
                       'wealth was set to 2016 because other years have not '
                       'been implemented yet.')
        ref_year = 2016
    if non_financial:
        try:
            val = factors_all_countries[
                factors_all_countries.country_iso3 == cntry_iso]['NFW-to-GDP-ratio'].values[0]
        except (AttributeError, KeyError, IndexError):
            LOGGER.warning('No data for country, using mean factor.')
            val = factors_all_countries["NFW-to-GDP-ratio"].mean()
    else:
        try:
            val = factors_all_countries[
                factors_all_countries.country_iso3 == cntry_iso]['TW-to-GDP-ratio'].values[0]
        except (AttributeError, KeyError, IndexError):
            LOGGER.warning('No data for country, using mean factor.')
            val = factors_all_countries["TW-to-GDP-ratio"].mean()
    val = np.around(val, 5)
    return ref_year, val

def world_bank_wealth_account(cntry_iso, ref_year, variable_name="NW.PCA.TO",
                              no_land=True):
    """
    Download and unzip wealth accounting historical data (1995, 2000, 2005, 2010, 2014)
    from World Bank (https://datacatalog.worldbank.org/dataset/wealth-accounting).
    Return requested variable for a country (cntry_iso) and a year (ref_year).

    Parameters
    ----------
    cntry_iso : str
        ISO3-code of country, i.e. "CHN" for China
    ref_year : int
        reference year
        - available in data: 1995, 2000, 2005, 2010, 2014
        - other years between 1995 and 2014 are interpolated
        - for years outside range, indicator is scaled
          proportionally to GDP
    variable_name : str
        select one variable, i.e.:
        'NW.PCA.TO': Produced capital stock of country
            incl. manufactured or built assets such as machinery,
            equipment, and physical structures
            and value of built-up urban land (24% mark-up)
        'NW.PCA.PC': Produced capital stock per capita
            incl. manufactured or built assets such as machinery,
            equipment, and physical structures
            and value of built-up urban land (24% mark-up)
        'NW.NCA.TO': Total natural capital of country. Natural capital
            includes the valuation of fossil fuel energy (oil, gas,
            hard and soft coal) and minerals (bauxite, copper, gold,
            iron ore, lead, nickel, phosphate, silver, tin, and zinc),
            agricultural land (cropland and pastureland),
            forests (timber and some nontimber forest products), and
            protected areas.
        'NW.TOW.TO': Total wealth of country.
        Note: Values are measured at market exchange rates in constant 2014 US dollars,
            using a country-specific GDP deflator.
    no_land : boolean
        If True, return produced capital without built-up land value
        (applies to 'NW.PCA.*' only). Default: True.
    """
    try:
        data_file = SYSTEM_DIR.joinpath(FILE_WORLD_BANK_WEALTH_ACC)
        if not data_file.is_file():
            data_file = SYSTEM_DIR.joinpath('Wealth-Accounts_CSV', FILE_WORLD_BANK_WEALTH_ACC)
        if not data_file.is_file():
            if not SYSTEM_DIR.joinpath('Wealth-Accounts_CSV').is_dir():
                SYSTEM_DIR.joinpath('Wealth-Accounts_CSV').mkdir()
            file_down = download_file(WORLD_BANK_WEALTH_ACC)
            zip_ref = zipfile.ZipFile(file_down, 'r')
            zip_ref.extractall(SYSTEM_DIR.joinpath('Wealth-Accounts_CSV'))
            zip_ref.close()
            Path(file_down).unlink()
            LOGGER.debug('Download and unzip complete. Unzipping %s', str(data_file))

        data_wealth = pd.read_csv(data_file, sep=',', index_col=None, header=0)
    except Exception as err:
        raise type(err)(
            'Downloading World Bank Wealth Accounting Data failed: ' + str(err)) from err

    data_wealth = data_wealth[data_wealth['Country Code'].str.contains(cntry_iso)
                              & data_wealth['Indicator Code'].str.contains(variable_name)
                             ].loc[:, '1995':'2014']
    years = list(map(int, list(data_wealth)))
    if data_wealth.size == 0 and 'NW.PCA.TO' in variable_name:  # if country is not found in data
        LOGGER.warning('No data available for country. Using non-financial wealth instead')
        gdp_year, gdp_val = gdp(cntry_iso, ref_year)
        fac = wealth2gdp(cntry_iso)[1]
        return gdp_year, np.around((fac * gdp_val), 1), 0
    if ref_year in years:  # indicator for reference year is available directly
        result = data_wealth.loc[:, str(ref_year)].values[0]
    elif np.min(years) < ref_year < np.max(years):  # interpolate
        result = np.interp(ref_year, years, data_wealth.values[0, :])
    elif ref_year < np.min(years):  # scale proportionally to GDP
        gdp_year, gdp0_val = gdp(cntry_iso, np.min(years))
        gdp_year, gdp_val = gdp(cntry_iso, ref_year)
        result = data_wealth.values[0, 0] * gdp_val / gdp0_val
        ref_year = gdp_year
    else:
        gdp_year, gdp0_val = gdp(cntry_iso, np.max(years))
        gdp_year, gdp_val = gdp(cntry_iso, ref_year)
        result = data_wealth.values[0, -1] * gdp_val / gdp0_val
        ref_year = gdp_year
    if 'NW.PCA.' in variable_name and no_land:
        # remove value of built-up land from produced capital
        result = result / 1.24
    return ref_year, np.around(result, 1), 1

def _gdp_twn(ref_year, per_capita=False):
    """returns GDP for TWN (Republic of China / Taiwan Province of China) based
    on a CSV sheet downloaded from the
    International Monetary Fund (IMF).
    The reason for this special treatment is the
    lack of GDP data for TWN in the World Bank data

    Data Source:
        https://www.imf.org/external/pubs/ft/weo/2019/02/weodata/index.aspx
        https://www.imf.org/external/pubs/ft/weo/2019/02/weodata/weorept.aspx?sy=1980&ey=2024&scsm=1&ssd=1&sic=1&sort=country&ds=.&br=1&pr1.x=42&pr1.y=10&c=528&s=NGDPD%2CNGDP_D%2CNGDPDPC&grp=0&a=
        (saved as CSV with name GDP_TWN_IMF_WEO_data in SYSTEM_DIR)

    Parameters
    ----------
    ref_year : int
        reference year, i.e. the year for which a GDP value is required
    per_capita : boolean
        return GDP per capita? Default False.

    Returns
    -------
    float
    """
    fname = 'GDP_TWN_IMF_WEO_data.csv'
    if not SYSTEM_DIR.joinpath(fname).is_file():
        raise FileNotFoundError(f'File {fname} not found in SYSTEM_DIR')
    if per_capita:
        var_name = 'Gross domestic product per capita, current prices'
    else:
        var_name = 'Gross domestic product, current prices'
    if ref_year < 1980:
        close_year = 1980
    elif ref_year > 2024:
        close_year = 2024
    else:
        close_year = ref_year
    data = pd.read_csv(SYSTEM_DIR.joinpath('GDP_TWN_IMF_WEO_data.csv'),
                       index_col=None, header=0)
    close_val = data.loc[data['Subject Descriptor'] == var_name, str(close_year)].values[0]
    close_val = float(close_val.replace(',', ''))
    if not per_capita:
        close_val = close_val * 1e9
    return close_year, close_val
