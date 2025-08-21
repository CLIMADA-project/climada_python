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

Define impact functions for tropical cyclnes .
"""

__all__ = ["ImpfTropCyclone", "ImpfSetTropCyclone", "IFTropCyclone"]

import logging
from enum import Enum

import numpy as np
import pandas as pd
from deprecation import deprecated

from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.util import coordinates
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)


class CountryCode(Enum):
    """
    Enum class that links ISO country codes (both ISO3A and ISO3N) to specific regions and
    associated impact function IDs.

    Attributes
    ----------
        ISO3A: dict
             A mapping of region names to lists of 3-letter ISO country codes (iso3a).
        ISO3N: dict
             A mapping of region names to lists of numeric ISO country codes (iso3n).
        IMPF_ID: dict
            A mapping of region names to corresponding impact function IDs.
        REGION_NAME: dict
            A mapping of region names to their descriptive names.
    """

    ALPHA3 = {
        "NA1": [
            "AIA",
            "ATG",
            "ARG",
            "ABW",
            "BHS",
            "BRB",
            "BLZ",
            "BMU",
            "BOL",
            "CPV",
            "CYM",
            "CHL",
            "COL",
            "CRI",
            "CUB",
            "DMA",
            "DOM",
            "ECU",
            "SLV",
            "FLK",
            "GUF",
            "GRD",
            "GLP",
            "GTM",
            "GUY",
            "HTI",
            "HND",
            "JAM",
            "MTQ",
            "MEX",
            "MSR",
            "NIC",
            "PAN",
            "PRY",
            "PER",
            "PRI",
            "SHN",
            "KNA",
            "LCA",
            "VCT",
            "SXM",
            "SUR",
            "TTO",
            "TCA",
            "URY",
            "VEN",
            "VGB",
            "VIR",
        ],
        "NA2": ["CAN", "USA"],
        "NI": [
            "AFG",
            "ARM",
            "AZE",
            "BHR",
            "BGD",
            "BTN",
            "DJI",
            "ERI",
            "ETH",
            "GEO",
            "IND",
            "IRN",
            "IRQ",
            "ISR",
            "JOR",
            "KAZ",
            "KWT",
            "KGZ",
            "LBN",
            "MDV",
            "MNG",
            "MMR",
            "NPL",
            "OMN",
            "PAK",
            "QAT",
            "SAU",
            "SOM",
            "LKA",
            "SYR",
            "TJK",
            "TKM",
            "UGA",
            "ARE",
            "UZB",
            "YEM",
        ],
        "OC": [
            "ASM",
            "AUS",
            "COK",
            "FJI",
            "PYF",
            "GUM",
            "KIR",
            "MHL",
            "FSM",
            "NRU",
            "NCL",
            "NZL",
            "NIU",
            "NFK",
            "MNP",
            "PLW",
            "PNG",
            "PCN",
            "WSM",
            "SLB",
            "TLS",
            "TKL",
            "TON",
            "TUV",
            "VUT",
            "WLF",
        ],
        "SI": [
            "COM",
            "COD",
            "SWZ",
            "MDG",
            "MWI",
            "MLI",
            "MUS",
            "MOZ",
            "ZAF",
            "TZA",
            "ZWE",
        ],
        "WP1": ["KHM", "IDN", "LAO", "MYS", "THA", "VNM"],
        "WP2": ["PHL"],
        "WP3": ["CHN"],
        "WP4": ["HKG", "JPN", "KOR", "MAC", "TWN"],
        "ROW": [
            "ALB",
            "DZA",
            "AND",
            "AGO",
            "ATA",
            "AUT",
            "BLR",
            "BEL",
            "BEN",
            "BES",
            "BIH",
            "BWA",
            "BVT",
            "BRA",
            "IOT",
            "BRN",
            "BGR",
            "BFA",
            "BDI",
            "CMR",
            "CAF",
            "TCD",
            "CXR",
            "CCK",
            "COG",
            "HRV",
            "CUW",
            "CYP",
            "CZE",
            "CIV",
            "DNK",
            "EGY",
            "GNQ",
            "EST",
            "FRO",
            "FIN",
            "FRA",
            "ATF",
            "GAB",
            "GMB",
            "DEU",
            "GHA",
            "GIB",
            "GRC",
            "GRL",
            "GGY",
            "GIN",
            "GNB",
            "HMD",
            "VAT",
            "HUN",
            "ISL",
            "IRL",
            "IMN",
            "ITA",
            "JEY",
            "KEN",
            "PRK",
            "XKX",
            "LVA",
            "LSO",
            "LBR",
            "LBY",
            "LIE",
            "LTU",
            "LUX",
            "MLT",
            "MRT",
            "MYT",
            "MDA",
            "MCO",
            "MNE",
            "MAR",
            "NAM",
            "NLD",
            "NER",
            "NGA",
            "MKD",
            "NOR",
            "PSE",
            "POL",
            "PRT",
            "ROU",
            "RUS",
            "RWA",
            "REU",
            "BLM",
            "MAF",
            "SPM",
            "SMR",
            "STP",
            "SEN",
            "SRB",
            "SYC",
            "SLE",
            "SGP",
            "SVK",
            "SVN",
            "SGS",
            "SSD",
            "ESP",
            "SDN",
            "SJM",
            "SWE",
            "CHE",
            "TGO",
            "TUN",
            "TUR",
            "UKR",
            "GBR",
            "UMI",
            "ESH",
            "ZMB",
            "ALA",
        ],
    }
    # fmt: on
    IMPF_ID = {
        "NA1": 1,
        "NA2": 2,
        "NI": 3,
        "OC": 4,
        "SI": 5,
        "WP1": 6,
        "WP2": 7,
        "WP3": 8,
        "WP4": 9,
        "ROW": 10,
    }

    REGION_NAME = {
        "NA1": "Caribbean and Mexico",
        "NA2": "USA and Canada",
        "NI": "North Indian",
        "OC": "Oceania",
        "SI": "South Indian",
        "WP1": "South East Asia",
        "WP2": "Philippines",
        "WP3": "China Mainland",
        "WP4": "North West Pacific",
        "ROW": "Rest of The World",
    }


class ImpfTropCyclone(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = "TC"

    def set_emanuel_usa(self, *args, **kwargs):
        """This function is deprecated, use from_emanuel_usa() instead."""
        LOGGER.warning(
            "The use of ImpfTropCyclone.set_emanuel_usa is deprecated."
            "Use ImpfTropCyclone.from_emanuel_usa instead."
        )
        self.__dict__ = ImpfTropCyclone.from_emanuel_usa(*args, **kwargs).__dict__

    @classmethod
    def from_emanuel_usa(
        cls,
        impf_id=1,
        intensity=np.arange(0, 121, 5),
        v_thresh=25.7,
        v_half=74.7,
        scale=1.0,
    ):
        """
        Init TC impact function using the formula of Kerry Emanuel, 2011:
        'Global Warming Effects on U.S. Hurricane Damage',
        https://doi.org/10.1175/WCAS-D-11-00007.1

        Parameters
        ----------
        impf_id : int, optional
            impact function id. Default: 1
        intensity : np.array, optional
            intensity array in m/s. Default:
            5 m/s step array from 0 to 120m/s
        v_thresh : float, optional
            first shape parameter, wind speed in
            m/s below which there is no damage. Default: 25.7(Emanuel 2011)
        v_half : float, optional
            second shape parameter, wind speed in m/s
            at which 50% of max. damage is expected. Default:
            v_threshold + 49 m/s (mean value of Sealy & Strobl 2017)
        scale : float, optional
            scale parameter, linear scaling of MDD.
            0<=scale<=1. Default: 1.0

        Raises
        ------
        ValueError

        Returns
        -------
        impf : ImpfTropCyclone
            TC impact function instance based on formula by Emanuel (2011)
        """
        if v_half <= v_thresh:
            raise ValueError("Shape parameters out of range: v_half <= v_thresh.")
        if v_thresh < 0 or v_half < 0:
            raise ValueError("Negative shape parameter.")
        if scale > 1 or scale <= 0:
            raise ValueError("Scale parameter out of range.")

        impf = cls()
        impf.name = "Emanuel 2011"
        impf.id = impf_id
        impf.intensity_unit = "m/s"
        impf.intensity = intensity
        impf.paa = np.ones(intensity.shape)
        v_temp = (impf.intensity - v_thresh) / (v_half - v_thresh)
        v_temp[v_temp < 0] = 0
        impf.mdd = v_temp**3 / (1 + v_temp**3)
        impf.mdd *= scale
        return impf


class ImpfSetTropCyclone(ImpactFuncSet):
    """Impact function set (ImpfS) for tropical cyclones."""

    def __init__(self):
        ImpactFuncSet.__init__(self)

    def set_calibrated_regional_ImpfSet(self, *args, **kwargs):
        """This function is deprecated, use from_calibrated_regional_ImpfSet() instead."""
        LOGGER.warning(
            "ImpfSetTropCyclone.set_calibrated_regional_ImpfSet is deprecated."
            "Use ImpfSetTropCyclone.from_calibrated_regional_ImpfSet instead."
        )
        self.__dict__ = ImpfSetTropCyclone.from_calibrated_regional_ImpfSet(
            *args, **kwargs
        ).__dict__
        return ImpfSetTropCyclone.calibrated_regional_vhalf(*args, **kwargs)

    @classmethod
    def from_calibrated_regional_ImpfSet(
        cls, calibration_approach="TDR", q=0.5, input_file_path=None, version=1
    ):
        """Calibrated regional TC wind impact functions

        Based on Eberenz et al. 2021: https://doi.org/10.5194/nhess-21-393-2021

        Parameters
        ----------
        calibration_approach : str, optional
            The following values are supported:

            'TDR' (default)
                Total damage ratio (TDR) optimization with
                TDR=1.0 (simulated damage = reported damage from EM-DAT)
            'TDR1.5'
                Total damage ratio (TDR) optimization with
                TDR=1.5 (simulated damage = 1.5*reported damage from EM-DAT)
            'RMSF'
                Root-mean-squared fraction (RMSF) optimization
            'EDR'
                quantile from individually fitted v_half per event,
                i.e. v_half fitted to get EDR=1.0 for each event
        q : float, optional
            Quantile between 0 and 1.0 to select (EDR only).
            Default: 0.5, i.e. median v_half
        input_file_path : str or DataFrame, optional
            full path to calibration
            result file to be used instead of default file in repository
            (expert users only)

        Returns
        -------
        impf_set : ImpfSetTropCyclone
            TC Impact Function Set based on Eberenz et al, 2021.
        """
        reg_v_half = ImpfSetTropCyclone.calibrated_regional_vhalf(
            calibration_approach=calibration_approach,
            q=q,
            input_file_path=input_file_path,
            version=version,
        )

        # define regions and parameters:
        v_0 = 25.7  # v_threshold based on Emanuel (2011)
        scale = 1.0

        # init impact function set
        impf_set = cls()
        for idx, region in enumerate(reg_v_half.keys()):
            impf_tc = ImpfTropCyclone.from_emanuel_usa(
                impf_id=int(idx + 1),
                v_thresh=v_0,
                v_half=reg_v_half[region],
                scale=scale,
            )
            impf_tc.name = CountryCode.REGION_NAME.value[region]
            impf_set.append(impf_tc)
        return impf_set

    @staticmethod
    def calibrated_regional_vhalf(
        calibration_approach="TDR", q=0.5, input_file_path=None, version=1
    ):
        """Calibrated TC wind impact function slope parameter v_half per region

        Based on Eberenz et al., 2021: https://doi.org/10.5194/nhess-21-393-2021

        Parameters
        ----------
        calibration_approach : str, optional
            The following values are supported:

            'TDR' (default)
                Total damage ratio (TDR) optimization with
                TDR=1.0 (simulated damage = reported damage from EM-DAT)
            'TDR1.5'
                Total damage ratio (TDR) optimization with
                TDR=1.5 (simulated damage = 1.5*reported damage from EM-DAT)
            'RMSF'
                Root-mean-squared fraction (RMSF) optimization
            'EDR'
                quantile from individually fitted v_half per event,
                i.e. v_half fitted to get EDR=1.0 for each event
        q : float, optional
            Quantile between 0 and 1.0 to select (EDR only).
            Default: 0.5, i.e. median v_half
        input_file_path : str or DataFrame, optional
            full path to calibration
            result file to be used instead of default file in repository
            (expert users only)

        Raises
        ------
        ValueError

        Returns
        -------
        v_half : dict
            TC impact function slope parameter v_half per region
        """
        calibration_approach = calibration_approach.upper()
        if calibration_approach not in ["TDR", "TDR1.0", "TDR1.5", "RMSF", "EDR"]:
            raise ValueError("calibration_approach is invalid")
        if "EDR" in calibration_approach and (q < 0.0 or q > 1.0):
            raise ValueError("Quantile q out of range [0, 1]")
        if calibration_approach == "TDR":
            calibration_approach = "TDR1.0"
        # load calibration results depending on approach:
        if isinstance(input_file_path, str):
            df_calib_results = pd.read_csv(
                input_file_path, encoding="ISO-8859-1", header=0
            )
        elif isinstance(input_file_path, pd.DataFrame):
            df_calib_results = input_file_path
        else:
            df_calib_results = pd.read_csv(
                SYSTEM_DIR.joinpath(
                    "tc_impf_cal_v%02.0f_%s.csv" % (version, calibration_approach)
                ),
                encoding="ISO-8859-1",
                header=0,
            )

        regions_short = list(CountryCode.REGION_NAME.value.keys())[
            :-1
        ]  # removing the last item ROW

        # loop over calibration regions (column cal_region2 in df):
        reg_v_half = dict()
        for region in regions_short:
            df_reg = df_calib_results.loc[df_calib_results.cal_region2 == region]
            df_reg = df_reg.reset_index(drop=True)
            reg_v_half[region] = np.round(df_reg["v_half"].quantile(q=q), 5)
        # rest of the world (ROW), calibrated by all data:
        regions_short = regions_short + ["ROW"]
        if calibration_approach == "EDR":
            reg_v_half[regions_short[-1]] = np.round(
                df_calib_results["v_half"].quantile(q=q), 5
            )
        else:
            df_reg = df_calib_results.loc[df_calib_results.cal_region2 == "GLB"]
            df_reg = df_reg.reset_index(drop=True)
            reg_v_half[regions_short[-1]] = np.round(df_reg["v_half"].values[0], 5)
        return reg_v_half

    @staticmethod
    def get_countries_per_region(region=None):
        """Returns dictionaries with numerical (numeric) and alphabetical (alpha3) ISO3 codes
        of all countries associated to a calibration region.
        Only contains countries that were affected by tropical cyclones
        between 1980 and 2017 according to EM-DAT.

        Parameters
        ----------
        region : str
            regional abbreviation (default='all'),
            either 'NA1', 'NA2', 'NI', 'OC', 'SI', 'WP1', 'WP2',
            'WP3', 'WP4', or 'all'.

        Returns
        -------
        region_name : dict or str
            long name per region
        impf_id : dict or int
            impact function ID per region
        numeric : dict or list
            numerical ISO3codes (=region_id) per region
        alpha3 : dict or list
            numerical ISO3codes (=region_id) per region
        """
        if not region:
            region = "all"

        if region == "all":
            return (
                CountryCode.REGION_NAME.value,
                CountryCode.IMPF_ID.value,
                coordinates.country_to_iso(
                    CountryCode.ALPHA3.value, representation="numeric"
                ),
                CountryCode.ALPHA3.value,
            )

        return (
            CountryCode.REGION_NAME.value[region],
            CountryCode.IMPF_ID.value[region],
            coordinates.country_to_iso(
                CountryCode.ALPHA3.value[region], representation="numeric"
            ),
            CountryCode.ALPHA3.value[region],
        )

    @staticmethod
    def get_impf_id_regions_per_countries(countries: list = None) -> tuple:
        """Return the impact function id and the region corresponding to a list of countries,
        or a single country.

        Parameters:
        -----------
        countries : list
            List containing the ISO codes of the country, which should be either
            in string format if the code is "ISO 3166-1 alpha-3" abbreviated as "alpha3", or an integer
            if the code is in "ISO 3166-1 numeric" abbreviated as "numeric", which is a three-digit country code,
            the numeric version of "ISO 3166-1 alpha-3". For example, the "alpha3" code of Switzerland is
            "CHE" and the "numeric" is 756.

        Returns:
        --------
        impf_ids : list
            List of impact function ids matching the countries.
        regions_ids : list
            List of the region ids. Regions are a container of countries as defined in:
            https://nhess.copernicus.org/articles/21/393/2021/nhess-21-393-2021.pdf, and implemented
            in the CountryCode Enum Class. Example: "NA1", "NA2", ...
        regions_names : list
            List of the regions names. Example: "Caribbean and Mexico", "USA and Canada", ...
        """

        # Find region
        regions_ids = []
        for country in countries:

            if isinstance(country, int):
                country = coordinates.country_to_iso(
                    country, representation="alpha3", fillvalue=None
                )

            for region_id, countr_in_region_id in CountryCode.ALPHA3.value.items():
                if country in countr_in_region_id:
                    regions_ids.append(region_id)

        # Find impact function id
        impf_ids = [CountryCode.IMPF_ID.value[region] for region in regions_ids]
        regions_name = [CountryCode.REGION_NAME.value[region] for region in regions_ids]

        return impf_ids, regions_ids, regions_name


@deprecated(
    details="The class name IFTropCyclone is deprecated and won't be supported in a future "
    + "version. Use ImpfTropCyclone instead"
)
class IFTropCyclone(ImpfTropCyclone):
    """Is ImpfTropCyclone now"""
