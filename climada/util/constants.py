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
           'ENT_TEMPLATE_XLS',
           'ONE_LAT_KM',
           'EARTH_RADIUS_KM',
           'GLB_CENTROIDS_MAT',
           'GLB_CENTROIDS_NC',
           'NAT_REG_ID',
           'TC_ANDREW_FL',
           'HAZ_DEMO_H5',
           'EXP_DEMO_H5']

import os

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

"""TODO: Set real demo file"""
HAZ_DEMO_FLDPH = os.path.join(DATA_DIR, 'demo', 'test_dph')
""" NetCDF4 Flood depth from isimip simulations"""

"""TODO: Set real demo file"""
HAZ_DEMO_FLFRC = os.path.join(DATA_DIR, 'demo', 'test_frc')
""" NetCDF4 Flood fraction from isimip simulations"""

HAZ_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob.mat')
""" Hazard demo from climada in MATLAB: hurricanes from 1851 to 2011 over Florida with 100 centroids."""

HAZ_DEMO_H5 = os.path.join(DATA_DIR, 'demo', 'tc_fl_1975_2011.h5')
""" Hazard demo in h5 format: ibtracs from 1975 to 2011 over Florida with 2500 centroids."""


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
