"""
Define constants.
"""

__all__ = ['SOURCE_DIR',
           'DATA_DIR',
           'HAZ_TEST_XLS',
           'HAZ_DEMO_MAT',
           'HAZ_TEST_MAT',
           'ENT_TEST_XLS',
           'ENT_TEMPLATE_XLS',
           'ENT_DEMO_MAT',
           'ONE_LAT_KM',
           'EARTH_RADIUS',
           'CENTR_TEST_BRB',
           'GLB_CENTROIDS_MAT']

import os

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), \
                                           os.pardir))
""" climada directory """

DATA_DIR = os.path.abspath(os.path.join(SOURCE_DIR, 'test', 'data'))
""" Folder containing the data """

HAZ_TEST_XLS = os.path.join(DATA_DIR, 'test', 'Excel_hazard.xlsx')
""" Hazard demo in xls format."""

HAZ_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob.mat')
""" Hazard demo in mat format."""

HAZ_TEST_MAT = os.path.join(DATA_DIR, 'test', 'atl_prob_no_name.mat')
""" Hazard for unit tests in mat format."""

ENT_TEST_XLS = os.path.join(DATA_DIR, 'test', 'demo_today.xlsx')
""" Entity demo in xls format."""

ENT_TEMPLATE_XLS = os.path.join(DATA_DIR, 'entity_template.xlsx')
""" Entity template in xls format."""

ENT_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'demo_today.mat')
""" Entity demo in mat format."""

GLB_CENTROIDS_MAT = os.path.join(DATA_DIR, 'GLB_NatID_grid_0360as_adv_2.mat')
""" Global centroids."""

CENTR_TEST_BRB = os.path.join(DATA_DIR, 'test', 'centr_brb_test.mat')
""" Global centroids."""

ONE_LAT_KM = 111.12
""" Mean one latitude in km """

EARTH_RADIUS = 6371
""" Earth radius in km """
