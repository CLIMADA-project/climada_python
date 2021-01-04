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

Define constants.
"""

__all__ = ['SYSTEM_DIR',
           'DEMO_DIR',
           'ENT_DEMO_TODAY',
           'ENT_DEMO_FUTURE',
           'HAZ_DEMO_MAT',
           'HAZ_DEMO_FL',
           'HAZ_DEMO_FLDDPH',
           'HAZ_DEMO_FLDFRC',
           'ENT_TEMPLATE_XLS',
           'HAZ_TEMPLATE_XLS',
           'ONE_LAT_KM',
           'EARTH_RADIUS_KM',
           'GLB_CENTROIDS_MAT',
           'GLB_CENTROIDS_NC',
           'ISIMIP_GPWV3_NATID_150AS',
           'NATEARTH_CENTROIDS',
           'DEMO_GDP2ASSET',
           'RIVER_FLOOD_REGIONS_CSV',
           'TC_ANDREW_FL',
           'HAZ_DEMO_H5',
           'EXP_DEMO_H5',
           'WS_DEMO_NC']

# pylint: disable=unused-import
# without importing numpy ahead of fiona the debugger may run into an error
import numpy
from fiona.crs import from_epsg

from .config import CONFIG

SYSTEM_DIR = CONFIG.local_data.system.dir(create=False)
"""Folder containing the data used internally"""

DEMO_DIR = CONFIG.local_data.demo.dir(create=False)
"""Folder containing the data used for tutorials"""

ISIMIP_GPWV3_NATID_150AS = SYSTEM_DIR.joinpath('NatID_grid_0150as.nc')
"""
Compressed version of National Identifier Grid in 150 arc-seconds from
ISIMIP project, based on GPWv3. Location in ISIMIP repository:

`ISIMIP2a/InputData/landuse_humaninfluences/population/ID_GRID/Nat_id_grid_ISIMIP.nc`

More references:

* https://www.isimip.org/gettingstarted/input-data-bias-correction/details/13/
* https://sedac.ciesin.columbia.edu/data/set/gpw-v3-national-identifier-grid
"""

GLB_CENTROIDS_NC = ISIMIP_GPWV3_NATID_150AS
"""For backwards compatibility, it remains available under its old name."""

GLB_CENTROIDS_MAT = SYSTEM_DIR.joinpath('GLB_NatID_grid_0360as_adv_2.mat')
"""Global centroids"""

NATEARTH_CENTROIDS = {
    150: SYSTEM_DIR.joinpath('NatEarth_Centroids_150as.hdf5'),
    360: SYSTEM_DIR.joinpath('NatEarth_Centroids_360as.hdf5'),
}
"""
Global centroids at XXX arc-seconds resolution,
including region ids from Natural Earth. The 360 AS file includes distance to
coast from NASA.
"""

ENT_TEMPLATE_XLS = SYSTEM_DIR.joinpath('entity_template.xlsx')
"""Entity template in xls format."""

HAZ_TEMPLATE_XLS = SYSTEM_DIR.joinpath('hazard_template.xlsx')
"""Hazard template in xls format."""

RIVER_FLOOD_REGIONS_CSV = SYSTEM_DIR.joinpath('NatRegIDs.csv')
"""Look-up table for river flood module"""

HAZ_DEMO_FL = DEMO_DIR.joinpath('SC22000_VE__M1.grd.gz')
"""Raster file of flood over Venezuela. Model from GAR2015"""

HAZ_DEMO_FLDDPH = DEMO_DIR.joinpath('flddph_2000_DEMO.nc')
"""NetCDF4 Flood depth from isimip simulations"""

HAZ_DEMO_FLDFRC = DEMO_DIR.joinpath('fldfrc_2000_DEMO.nc')
"""NetCDF4 Flood fraction from isimip simulations"""

HAZ_DEMO_MAT = DEMO_DIR.joinpath('atl_prob_nonames.mat')
"""
Hazard demo from climada in MATLAB: hurricanes from 1851 to 2011 over Florida with 100 centroids.
"""

HAZ_DEMO_H5 = DEMO_DIR.joinpath('tc_fl_1990_2004.h5')
"""
Hazard demo in hdf5 format: IBTrACS from 1990 to 2004 over Florida with 2500 centroids.
"""

DEMO_GDP2ASSET = DEMO_DIR.joinpath('gdp2asset_CHE_exposure.nc')
"""Exposure demo file for GDP2Asset"""

WS_DEMO_NC = [DEMO_DIR.joinpath('fp_lothar_crop-test.nc'),
              DEMO_DIR.joinpath('fp_xynthia_crop-test.nc')]
"""
Winter storm in Europe files. These test files have been generated using
the netCDF kitchen sink:

>>> ncks -d latitude,50.5,54.0 -d longitude,3.0,7.5 ./file_in.nc ./file_out.nc
"""


ENT_DEMO_TODAY = DEMO_DIR.joinpath('demo_today.xlsx')
"""Entity demo present in xslx format."""

ENT_DEMO_FUTURE = DEMO_DIR.joinpath('demo_future_TEST.xlsx')
"""Entity demo future in xslx format."""

EXP_DEMO_H5 = DEMO_DIR.joinpath('exp_demo_today.h5')
"""Exposures over Florida"""


TC_ANDREW_FL = DEMO_DIR.joinpath('ibtracs_global_intp-None_1992230N11325.csv')
"""Tropical cyclone Andrew in Florida"""


ISIMIP_NATID_TO_ISO = [
    '', 'ABW', 'AFG', 'AGO', 'AIA', 'ALB', 'AND', 'ANT', 'ARE', 'ARG', 'ARM',
    'ASM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR',
    'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BMU', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN',
    'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COK',
    'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CYM', 'CYP', 'CZE', 'DEU', 'DJI', 'DMA',
    'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESP', 'EST', 'ETH', 'FIN', 'FJI',
    'FLK', 'FRA', 'FRO', 'FSM', 'GAB', 'GBR', 'GEO', 'GGY', 'GHA', 'GIB', 'GIN',
    'GLP', 'GMB', 'GNB', 'GNQ', 'GRC', 'GRD', 'GTM', 'GUF', 'GUM', 'GUY', 'HKG',
    'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IMN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL',
    'ISR', 'ITA', 'JAM', 'JEY', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR',
    'KNA', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LIE', 'LKA', 'LSO',
    'LTU', 'LUX', 'LVA', 'MAC', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MHL',
    'MKD', 'MLI', 'MLT', 'MMR', 'MNG', 'MNP', 'MOZ', 'MRT', 'MSR', 'MTQ', 'MUS',
    'MWI', 'MYS', 'MYT', 'NAM', 'NCL', 'NER', 'NFK', 'NGA', 'NIC', 'NIU', 'NLD',
    'NOR', 'NPL', 'NRU', 'NZL', 'OMN', 'PAK', 'PAN', 'PCN', 'PER', 'PHL', 'PLW',
    'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'PYF', 'QAT', 'REU', 'ROU',
    'RUS', 'RWA', 'SAU', 'SCG', 'SDN', 'SEN', 'SGP', 'SHN', 'SJM', 'SLB', 'SLE',
    'SLV', 'SMR', 'SOM', 'SPM', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SYC',
    'SYR', 'TCA', 'TCD', 'TGO', 'THA', 'TJK', 'TKL', 'TKM', 'TLS', 'TON', 'TTO',
    'TUN', 'TUR', 'TUV', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'USA', 'UZB', 'VCT',
    'VEN', 'VGB', 'VIR', 'VNM', 'VUT', 'WLF', 'WSM', 'YEM', 'ZAF', 'ZMB', 'ZWE',
]
"""ISO3166 alpha-3 codes of countries used in ISIMIP_GPWV3_NATID_150AS"""


ONE_LAT_KM = 111.12
"""Mean one latitude (in degrees) to km"""

EARTH_RADIUS_KM = 6371
"""Earth radius in km"""

DEF_EPSG = 4326
"""Default EPSG code"""

DEF_CRS = from_epsg(DEF_EPSG)
"""Default coordinate reference system WGS 84"""
