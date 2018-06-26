"""
Define constants.
"""

__all__ = ['SOURCE_DIR',
           'DATA_DIR',
           'HAZ_DEMO_MAT',
           'ENT_TEMPLATE_XLS',
           'ENT_DEMO_MAT',
           'ONE_LAT_KM',
           'EARTH_RADIUS',
           'GLB_CENTROIDS_MAT',
           'ENT_FL_MAT',
           'TC_ANDREW_FL']

import os

SOURCE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir))
""" climada directory """

DATA_DIR = os.path.abspath(os.path.join(SOURCE_DIR, os.pardir, 'data'))
""" Folder containing the data """

HAZ_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'atl_prob.mat')
""" Hazard demo in mat format."""

ENT_TEMPLATE_XLS = os.path.join(DATA_DIR, 'entity_template.xlsx')
""" Entity template in xls format."""

ENT_DEMO_MAT = os.path.join(DATA_DIR, 'demo', 'demo_today.mat')
""" Entity demo in mat format."""

ENT_FL_MAT = os.path.join(DATA_DIR, 'demo',
                          'USA_UnitedStates_Florida_entity.mat')
""" Entity for Florida """

TC_ANDREW_FL = os.path.join(DATA_DIR, 'demo',
                            'ibtracs_global_intp-None_1992230N11325.csv')
""" Tropical cyclone Andrew in Florida """

GLB_CENTROIDS_MAT = os.path.join(DATA_DIR, 'GLB_NatID_grid_0360as_adv_2.mat')
""" Global centroids."""

ONE_LAT_KM = 111.12
""" Mean one latitude in km """

EARTH_RADIUS = 6371
""" Earth radius in km """
