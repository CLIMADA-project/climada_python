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

__all__ = ['SOURCE_DIR',
           'DATA_DIR',
           'SYSTEM_DIR',
           'HAZ_DEMO_MAT',
           'HAZ_DEMO_FLDDPH',
           'HAZ_DEMO_FLDFRC',
           'ENT_TEMPLATE_XLS',
           'ONE_LAT_KM',
           'EARTH_RADIUS_KM',
           'GLB_CENTROIDS_MAT',
           'GLB_CENTROIDS_NC',
           'DEMO_GDP2ASSET',
           'CONVERTER',
           'FLOOD_IF_DIR',
           'NAT_REG_ID',
           'TC_ANDREW_FL',
           'HAZ_DEMO_H5',
           'EXP_DEMO_H5']

import os
from fiona.crs import from_epsg

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir))
""" climada directory """

DATA_DIR = os.path.abspath(os.path.join(SOURCE_DIR, os.pardir, 'data'))
""" Folder containing the data """

SYSTEM_DIR = os.path.abspath(os.path.join(DATA_DIR, 'system'))
""" Folder containing the data used internally """


GLB_CENTROIDS_NC = os.path.join(SYSTEM_DIR, 'NatID_grid_0150as.nc')
""" Global centroids nc."""

GLB_CENTROIDS_MAT = os.path.join(SYSTEM_DIR, 'GLB_NatID_grid_0360as_adv_2.mat')
""" Global centroids."""

ENT_TEMPLATE_XLS = os.path.join(SYSTEM_DIR, 'entity_template.xlsx')
""" Entity template in xls format."""

HAZ_TEMPLATE_XLS = os.path.join(SYSTEM_DIR, 'hazard_template.xlsx')
""" Hazard template in xls format."""
NAT_REG_ID = os.path.join(SYSTEM_DIR, 'NatRegIDs.csv')
""" Look-up table ISO3 codes"""


HAZ_DEMO_FL = os.path.join(DATA_DIR, 'demo', 'SC22000_VE__M1.grd.gz')
""" Raster file of flood over Venezuela. Model from GAR2015"""


HAZ_DEMO_FLDDPH = os.path.join(DATA_DIR, 'demo', 'flddph_WaterGAP2_miroc5_historical_flopros_gev_picontrol_2000_0.1.nc')
""" NetCDF4 Flood depth from isimip simulations"""


HAZ_DEMO_FLDFRC = os.path.join(DATA_DIR, 'demo', 'fldfrc_WaterGAP2_miroc5_historical_flopros_gev_picontrol_2000_0.1.nc')
""" NetCDF4 Flood fraction from isimip simulations"""

HAZ_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob.mat')
""" Hazard demo from climada in MATLAB: hurricanes from 1851 to 2011 over Florida with 100 centroids."""

HAZ_DEMO_H5 = os.path.join(DATA_DIR, 'demo', 'tc_fl_1975_2011.h5')
""" Hazard demo in h5 format: ibtracs from 1975 to 2011 over Florida with 2500 centroids."""

DEMO_GDP2ASSET = os.path.join(DATA_DIR, 'demo', 'gdp2asset_demo_exposure.nc')
"""Exposure demo file for GDP2Asset"""

CONVERTER = os.path.join(SYSTEM_DIR, 'GDP2Asset_converter_2.5arcmin.nc')
"""GDP2Asset coverter for GDP2Asset exposures"""

FLOOD_IF_DIR = os.path.join(SYSTEM_DIR, 'FloodImpactFnc')
"""Flood impct function directory"""

ENT_DEMO_TODAY = os.path.join(DATA_DIR, 'demo', 'demo_today.xlsx')
""" Entity demo present in xslx format."""

ENT_DEMO_FUTURE = os.path.join(DATA_DIR, 'demo', 'demo_future_TEST.xlsx')
""" Entity demo future in xslx format."""

EXP_DEMO_H5 = os.path.join(DATA_DIR, 'demo', 'exp_demo_today.h5')
""" Exposures over Florida """


TC_ANDREW_FL = os.path.join(DATA_DIR, 'demo',
                            'ibtracs_global_intp-None_1992230N11325.csv')
""" Tropical cyclone Andrew in Florida """


ONE_LAT_KM = 111.12
""" Mean one latitude (in degrees) to km """

EARTH_RADIUS_KM = 6371
""" Earth radius in km """

DEF_EPSG = 4326
""" Default EPSG code """

DEF_CRS = from_epsg(DEF_EPSG)
""" Default coordinate reference system WGS 84 """
