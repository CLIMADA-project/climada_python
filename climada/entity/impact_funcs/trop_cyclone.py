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

Define impact functions for tropical cyclnes .
"""

__all__ = ['IFTropCyclone']

import logging
import numpy as np
import pandas as pd

from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

class IFTropCyclone(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'TC'

    def set_emanuel_usa(self, if_id=1, intensity=np.arange(0, 121, 5),
                        v_thresh=25.7, v_half=74.7, scale=1.0):
        """Using the formula of Emanuele 2011.

        Parameters:
            if_id (int, optional): impact function id. Default: 1
            intensity (np.array, optional): intensity array in m/s. Default:
                5 m/s step array from 0 to 120m/s
            v_thresh (float, optional): first shape parameter, wind speed in
                m/s below which there is no damage. Default: 25.7(Emanuel 2011)
            v_half (float, optional): second shape parameter, wind speed in m/s
                at which 50% of max. damage is expected. Default:
                v_threshold + 49 m/s (mean value of Sealy & Strobl 2017)
            scale (float, optional): scale parameter, linear scaling of MDD.
                0<=scale<=1. Default: 1.0

        Raises:
            ValueError
        """
        if v_half <= v_thresh:
            LOGGER.error('Shape parameters out of range: v_half <= v_thresh.')
            raise ValueError
        if v_thresh < 0 or v_half < 0:
            LOGGER.error('Negative shape parameter.')
            raise ValueError
        if scale > 1 or scale <= 0:
            LOGGER.error('Scale parameter out of range.')
            raise ValueError

        self.name = 'Emanuel 2011'
        self.id = if_id
        self.intensity_unit = 'm/s'
        self.intensity = intensity
        self.paa = np.ones(intensity.shape)
        v_temp = (self.intensity - v_thresh) / (v_half - v_thresh)
        v_temp[v_temp < 0] = 0
        self.mdd = v_temp**3 / (1 + v_temp**3)
        self.mdd *= scale

class IFSTropCyclone(ImpactFuncSet):
    """Impact function set (IFS) for tropical cyclones."""

    def __init__(self):
        ImpactFuncSet.__init__(self)

    def set_calibrated_regional_IFs(self, calibration_approach='TDR', q=.5,
                                    input_file_path=None, version=1):
        """ initiate TC wind impact functions based on Eberenz et al. (2020)

        Optional Parameters:
                calibration_approach (str):
                    'TDR' (default): Total damage ratio (TDR) optimization with
                        TDR=1.0 (simulated damage = reported damage from EM-DAT)
                    'TDR1.5' : Total damage ratio (TDR) optimization with
                        TDR=1.5 (simulated damage = 1.5*reported damage from EM-DAT)
                    'RMSF': Root-mean-squared fraction (RMSF) optimization
                    'EDR': quantile from individually fitted v_half per event,
                        i.e. v_half fitted to get EDR=1.0 for each event
                q (float): quantile between 0 and 1.0 to select
                    (EDR only, default=0.5, i.e. median v_half)
                input_file_path (str or DataFrame): full path to calibration
                    result file to be used instead of default file in repository
                    (expert users only)

        Returns:
            v_half (dict): IF slope parameter v_half per region¨

        Raises:
            ValueError
        """
        calibration_approach = calibration_approach.upper()
        if calibration_approach not in ['TDR', 'TDR1.0', 'TDR1.5', 'RMSF', 'EDR']:
            LOGGER.error('calibration_approach is invalid')
            raise ValueError
        if 'EDR' in calibration_approach and (q < 0. or q > 1.):
            LOGGER.error('Quantile q out of range [0, 1]')
            raise ValueError
        if calibration_approach == 'TDR':
            calibration_approach = 'TDR1.0'
        # load calibration results depending on approach:
        if isinstance(input_file_path, str):
            df_calib_results = pd.read_csv(input_file_path,
                                           encoding="ISO-8859-1", header=0)
        elif isinstance(input_file_path, pd.DataFrame):
            df_calib_results = input_file_path
        else:
            df_calib_results = pd.read_csv(
                SYSTEM_DIR.joinpath(
                             'tc_if_cal_v%02.0f_%s.csv' % (version, calibration_approach)),
                encoding="ISO-8859-1", header=0)

        # define regions and parameters:
        v_0 = 25.7  # v_threshold based on Emanuel (2011)
        scale = 1.0

        regions_short = ['NA1', 'NA2', 'NI', 'OC', 'SI', 'WP1', 'WP2', 'WP3', 'WP4']
        regions_long = dict()
        regions_long[regions_short[0]] = 'Caribbean and Mexico (NA1)'
        regions_long[regions_short[1]] = 'USA and Canada (NA2)'
        regions_long[regions_short[2]] = 'North Indian (NI)'
        regions_long[regions_short[3]] = 'Oceania (OC)'
        regions_long[regions_short[4]] = 'South Indian (SI)'
        regions_long[regions_short[5]] = 'South East Asia (WP1)'
        regions_long[regions_short[6]] = 'Philippines (WP2)'
        regions_long[regions_short[7]] = 'China Mainland (WP3)'
        regions_long[regions_short[8]] = 'North West Pacific (WP4)'
        regions_long['all'] = 'Global'
        regions_long['GLB'] = 'Global'
        regions_long['ROW'] = 'Global'

        # loop over calibration regions (column cal_region2 in df):
        reg_v_half = dict()
        for idx, region in enumerate(regions_short):
            df_reg = df_calib_results.loc[df_calib_results.cal_region2 == region]
            df_reg = df_reg.reset_index(drop=True)
            reg_v_half[region] = np.round(df_reg['v_half'].quantile(q=q), 5)
        # rest of the world (ROW), calibrated by all data:
        regions_short = regions_short + ['ROW']
        if calibration_approach == 'EDR':
            reg_v_half[regions_short[-1]] = np.round(df_calib_results['v_half'].quantile(q=q), 5)
        else:
            df_reg = df_calib_results.loc[df_calib_results.cal_region2 == 'GLB']
            df_reg = df_reg.reset_index(drop=True)
            reg_v_half[regions_short[-1]] = np.round(df_reg['v_half'].values[0], 5)

        for idx, region in enumerate(regions_short):
            if_tc = IFTropCyclone()
            if_tc.set_emanuel_usa(if_id=int(idx + 1), v_thresh=v_0, v_half=reg_v_half[region],
                                  scale=scale)
            if_tc.name = regions_long[region]
            self.append(if_tc)
        return reg_v_half

    @staticmethod
    def get_countries_per_region(region=None):
        """Returns dictionaries with numerical and alphabetical ISO3 codes
        of all countries associated to a calibration region.
        Only contains countries that were affected by tropical cyclones
        between 1980 and 2017 according to EM-DAT.

        Optional Parameters:
            region (str): regional abbreviation (default='all'),
                either 'NA1', 'NA2', 'NI', 'OC', 'SI', 'WP1', 'WP2',
                        'WP3', 'WP4', or 'all'.

        Returns:
            [0] region_name (dict or str): long name per region
            [1] if_id (dict or int): impact function ID per region
            [2] iso3n (dict or list): numerical ISO3codes (=region_id) per region
            [3] iso3a (dict or list): numerical ISO3codes (=region_id) per region
        """
        if not region:
            region = 'all'
        iso3n = {'NA1': [660, 28, 32, 533, 44, 52, 84, 60, 68, 132, 136,
                         152, 170, 188, 192, 212, 214, 218, 222, 238, 254,
                         308, 312, 320, 328, 332, 340, 388, 474, 484, 500,
                         558, 591, 600, 604, 630, 654, 659, 662, 670, 534,
                         740, 780, 796, 858, 862, 92, 850],
                 'NA2': [124, 840],
                 'NI': [4, 51, 31, 48, 50, 64, 262, 232,
                        231, 268, 356, 364, 368, 376, 400, 398, 414, 417,
                        422, 462, 496, 104, 524, 512, 586, 634, 682, 706,
                        144, 760, 762, 795, 800, 784, 860, 887],
                 'OC': [16, 36, 184, 242, 258, 316, 296, 584, 583, 520,
                        540, 554, 570, 574, 580, 585, 598, 612, 882, 90,
                        626, 772, 776, 798, 548, 876],
                 'SI': [174, 180, 748, 450, 454, 466, 480, 508, 710, 834,
                        716],
                 'WP1': [116, 360, 418, 458, 764, 704],
                 'WP2': [608],
                 'WP3': [156],
                 'WP4': [344, 392, 410, 446, 158],
                 'ROW': [8, 12, 20, 24, 10, 40, 112, 56, 204, 535, 70, 72,
                         74, 76, 86, 96, 100, 854, 108, 120, 140, 148, 162,
                         166, 178, 191, 531, 196, 203, 384, 208, 818, 226,
                         233, 234, 246, 250, 260, 266, 270, 276, 288, 292,
                         300, 304, 831, 324, 624, 334, 336, 348, 352, 372,
                         833, 380, 832, 404, 408, 983, 428, 426, 430, 434,
                         438, 440, 442, 470, 478, 175, 498, 492, 499, 504,
                         516, 528, 562, 566, 807, 578, 275, 616, 620, 642,
                         643, 646, 638, 652, 663, 666, 674, 678, 686, 688,
                         690, 694, 702, 703, 705, 239, 728, 724, 729, 744,
                         752, 756, 768, 788, 792, 804, 826, 581, 732, 894,
                         248]}
        iso3a = {'NA1': ['AIA', 'ATG', 'ARG', 'ABW', 'BHS', 'BRB', 'BLZ', 'BMU',
                         'BOL', 'CPV', 'CYM', 'CHL', 'COL', 'CRI', 'CUB', 'DMA',
                         'DOM', 'ECU', 'SLV', 'FLK', 'GUF', 'GRD', 'GLP', 'GTM',
                         'GUY', 'HTI', 'HND', 'JAM', 'MTQ', 'MEX', 'MSR', 'NIC',
                         'PAN', 'PRY', 'PER', 'PRI', 'SHN', 'KNA', 'LCA', 'VCT',
                         'SXM', 'SUR', 'TTO', 'TCA', 'URY', 'VEN', 'VGB', 'VIR'],
                 'NA2': ['CAN', 'USA'],
                 'NI': ['AFG', 'ARM', 'AZE', 'BHR', 'BGD', 'BTN', 'DJI', 'ERI',
                        'ETH', 'GEO', 'IND', 'IRN', 'IRQ', 'ISR', 'JOR', 'KAZ',
                        'KWT', 'KGZ', 'LBN', 'MDV', 'MNG', 'MMR', 'NPL', 'OMN',
                        'PAK', 'QAT', 'SAU', 'SOM', 'LKA', 'SYR', 'TJK', 'TKM',
                        'UGA', 'ARE', 'UZB', 'YEM'],
                 'OC': ['ASM', 'AUS', 'COK', 'FJI', 'PYF', 'GUM', 'KIR', 'MHL',
                        'FSM', 'NRU', 'NCL', 'NZL', 'NIU', 'NFK', 'MNP', 'PLW',
                        'PNG', 'PCN', 'WSM', 'SLB', 'TLS', 'TKL', 'TON', 'TUV',
                        'VUT', 'WLF'],
                 'SI': ['COM', 'COD', 'SWZ', 'MDG', 'MWI', 'MLI', 'MUS', 'MOZ',
                        'ZAF', 'TZA', 'ZWE'],
                 'WP1': ['KHM', 'IDN', 'LAO', 'MYS', 'THA', 'VNM'],
                 'WP2': ['PHL'],
                 'WP3': ['CHN'],
                 'WP4': ['HKG', 'JPN', 'KOR', 'MAC', 'TWN'],
                 'ROW': ['ALB', 'DZA', 'AND', 'AGO', 'ATA', 'AUT', 'BLR', 'BEL',
                         'BEN', 'BES', 'BIH', 'BWA', 'BVT', 'BRA', 'IOT', 'BRN',
                         'BGR', 'BFA', 'BDI', 'CMR', 'CAF', 'TCD', 'CXR', 'CCK',
                         'COG', 'HRV', 'CUW', 'CYP', 'CZE', 'CIV', 'DNK', 'EGY',
                         'GNQ', 'EST', 'FRO', 'FIN', 'FRA', 'ATF', 'GAB', 'GMB',
                         'DEU', 'GHA', 'GIB', 'GRC', 'GRL', 'GGY', 'GIN', 'GNB',
                         'HMD', 'VAT', 'HUN', 'ISL', 'IRL', 'IMN', 'ITA', 'JEY',
                         'KEN', 'PRK', 'XKX', 'LVA', 'LSO', 'LBR', 'LBY', 'LIE',
                         'LTU', 'LUX', 'MLT', 'MRT', 'MYT', 'MDA', 'MCO', 'MNE',
                         'MAR', 'NAM', 'NLD', 'NER', 'NGA', 'MKD', 'NOR', 'PSE',
                         'POL', 'PRT', 'ROU', 'RUS', 'RWA', 'REU', 'BLM', 'MAF',
                         'SPM', 'SMR', 'STP', 'SEN', 'SRB', 'SYC', 'SLE', 'SGP',
                         'SVK', 'SVN', 'SGS', 'SSD', 'ESP', 'SDN', 'SJM', 'SWE',
                         'CHE', 'TGO', 'TUN', 'TUR', 'UKR', 'GBR', 'UMI', 'ESH',
                         'ZMB', 'ALA']}
        if_id = {'NA1': 1, 'NA2': 2, 'NI': 3, 'OC': 4, 'SI': 5,
                 'WP1': 6, 'WP2': 7, 'WP3': 8, 'WP4': 9, 'ROW': 10}
        region_name = dict()
        region_name['NA1'] = 'Caribbean and Mexico'
        region_name['NA2'] = 'USA and Canada'
        region_name['NI'] = 'North Indian'
        region_name['OC'] = 'Oceania'
        region_name['SI'] = 'South Indian'
        region_name['WP1'] = 'South East Asia'
        region_name['WP2'] = 'Philippines'
        region_name['WP3'] = 'China Mainland'
        region_name['WP4'] = 'North West Pacific'

        if region == 'all':
            return region_name, if_id, iso3n, iso3a

        return region_name[region], if_id[region], iso3n[region], iso3a[region]
