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
           'TC_ANDREW_FL',
           'HAZ_DEMO_H5',
           'EXP_DEMO_H5',
           'WS_DEMO_NC']

import os
from fiona.crs import from_epsg

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir))
""" climada directory """

DATA_DIR = os.path.abspath(os.path.join(SOURCE_DIR, os.pardir, 'data'))
""" Folder containing the data """

SYSTEM_DIR = os.path.abspath(os.path.join(DATA_DIR, 'system'))
""" Folder containing the data used internally """




GLB_CENTROIDS_MAT = os.path.join(SYSTEM_DIR, 'GLB_NatID_grid_0360as_adv_2.mat')
""" Global centroids."""

ENT_TEMPLATE_XLS = os.path.join(SYSTEM_DIR, 'entity_template.xlsx')
""" Entity template in xls format."""

HAZ_TEMPLATE_XLS = os.path.join(SYSTEM_DIR, 'hazard_template.xlsx')
""" Hazard template in xls format."""


HAZ_DEMO_FL = os.path.join(DATA_DIR, 'demo', 'SC22000_VE__M1.grd.gz')
""" Raster file of flood over Venezuela. Model from GAR2015"""

HAZ_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob.mat')
""" Hazard demo from climada in MATLAB: hurricanes from 1851 to 2011 over Florida with 100 centroids."""

HAZ_DEMO_H5 = os.path.join(DATA_DIR, 'demo', 'tc_fl_1975_2011.h5')
""" Hazard demo in h5 format: ibtracs from 1975 to 2011 over Florida with 2500 centroids."""

WS_DEMO_NC = [os.path.join(DATA_DIR, 'demo', 'fp_lothar_crop-test.nc'),
              os.path.join(DATA_DIR, 'demo', 'fp_xynthia_crop-test.nc')]
""" Winter storm in Europe files. These test files have been generated using
the netCDF kitchen sink: ncks -d latitude,50.5,54.0 -d longitude,3.0,7.5
./file_in.nc ./file_out.nc """


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
