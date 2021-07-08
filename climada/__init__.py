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

climada init
"""
from shutil import copyfile
from pathlib import Path

from .util.config import CONFIG, setup_logging
from .util.constants import *


__all__ = ['init']

GSDP_DIR = SYSTEM_DIR.joinpath('GSDP')

REPO_DATA = {
    'data/system': [
        ISIMIP_GPWV3_NATID_150AS,
        GLB_CENTROIDS_MAT,
        ENT_TEMPLATE_XLS,
        HAZ_TEMPLATE_XLS,
        RIVER_FLOOD_REGIONS_CSV,
        NATEARTH_CENTROIDS[150],
        NATEARTH_CENTROIDS[360],
        SYSTEM_DIR.joinpath('GDP2Asset_converter_2.5arcmin.nc'),
        SYSTEM_DIR.joinpath('WEALTH2GDP_factors_CRI_2016.csv'),
        SYSTEM_DIR.joinpath('GDP_TWN_IMF_WEO_data.csv'),
        SYSTEM_DIR.joinpath('FAOSTAT_data_country_codes.csv'),
        SYSTEM_DIR.joinpath('rcp_db.xls'),
        SYSTEM_DIR.joinpath('tc_impf_cal_v01_TDR1.0.csv'),
        SYSTEM_DIR.joinpath('tc_impf_cal_v01_EDR.csv'),
        SYSTEM_DIR.joinpath('tc_impf_cal_v01_RMSF.csv'),
    ],
    'data/system/GSDP': [
        GSDP_DIR.joinpath(f'{cc}_GSDP.xls')
        for cc in ['AUS', 'BRA', 'CAN', 'CHE', 'CHN', 'DEU', 'FRA', 'IDN', 'IND', 'JPN', 'MEX',
                   'TUR', 'USA', 'ZAF']
    ],
    'data/demo': [
        ENT_DEMO_TODAY,
        ENT_DEMO_FUTURE,
        EXP_DEMO_H5,
        HAZ_DEMO_FL,
        HAZ_DEMO_FLDDPH,
        HAZ_DEMO_FLDFRC,
        HAZ_DEMO_MAT,
        HAZ_DEMO_H5,
        TC_ANDREW_FL,
        DEMO_GDP2ASSET,
        DEMO_DIR.joinpath('demo_emdat_impact_data_2020.csv'),
        DEMO_DIR.joinpath('histsoc_landuse-15crops_annual_FR_DE_DEMO_2001_2005.nc'),
        DEMO_DIR.joinpath('hist_mean_mai-firr_1976-2005_DE_FR.hdf5'),
        DEMO_DIR.joinpath('crop_production_demo_data_yields_CHE.nc4'),
        DEMO_DIR.joinpath('crop_production_demo_data_cultivated_area_CHE.nc4'),
        DEMO_DIR.joinpath('FAOSTAT_data_producer_prices.csv'),
        DEMO_DIR.joinpath('FAOSTAT_data_production_quantity.csv'),
        DEMO_DIR.joinpath('lpjml_ipsl-cm5a-lr_ewembi_historical_2005soc_co2_yield-whe-noirr_annual_FR_DE_DEMO_1861_2005.nc'),
        DEMO_DIR.joinpath('h08_gfdl-esm2m_ewembi_historical_histsoc_co2_dis_global_daily_DEMO_FR_2001_2003.nc'),
        DEMO_DIR.joinpath('h08_gfdl-esm2m_ewembi_historical_histsoc_co2_dis_global_daily_DEMO_FR_2004_2005.nc'),
        DEMO_DIR.joinpath('gepic_gfdl-esm2m_ewembi_historical_2005soc_co2_yield-whe-noirr_global_DEMO_TJANJIN_annual_1861_2005.nc'),
        DEMO_DIR.joinpath('pepic_miroc5_ewembi_historical_2005soc_co2_yield-whe-firr_global_annual_DEMO_TJANJIN_1861_2005.nc'),
        DEMO_DIR.joinpath('pepic_miroc5_ewembi_historical_2005soc_co2_yield-whe-noirr_global_annual_DEMO_TJANJIN_1861_2005.nc'),
        DEMO_DIR.joinpath('WS_ERA40_sample.mat'),
        DEMO_DIR.joinpath('WS_Europe.xls'),
        DEMO_DIR.joinpath('Portugal_firms_June_2017.csv'),
        DEMO_DIR.joinpath('Portugal_firms_2016_17_18_MODIS.csv'),
    ] + WS_DEMO_NC
}


def setup_climada_data(reload=False):

    for dirpath in [DEMO_DIR, SYSTEM_DIR, GSDP_DIR]:
        dirpath.mkdir(parents=True, exist_ok=True)

    for src_dir, path_list in REPO_DATA.items():
        for path in path_list:
            if not path.exists() or reload:
                src = Path(__file__).parent.parent.joinpath(src_dir, path.name)
                copyfile(src, path)


def init():
    setup_climada_data()
    setup_logging(CONFIG.log_level.str())


init()
