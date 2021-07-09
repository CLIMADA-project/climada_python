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

Define the SupplyChain class.
"""

__all__ = ['SupplyChain']

import logging
import datetime as dt
from tqdm import tqdm
import numpy as np
import pandas as pd

from climada import CONFIG
from climada.util import files_handler as u_fh
import climada.util.coordinates as u_coord
from climada.engine import Impact
from climada.entity.exposures.base import Exposures

LOGGER = logging.getLogger(__name__)
WIOD_FILE_LINK = CONFIG.engine.supplychain.resources.wiod16.str()
"""Link to the 2016 release of the WIOD tables."""

WIOD_DIRECTORY = CONFIG.engine.supplychain.local_data.wiod.dir()
"""Directory where WIOD tables are downloaded into."""

class SupplyChain():
    """SupplyChain class.

    The SupplyChain class provides methods for loading Multi-Regional Input-Output
    Tables (MRIOT) and computing direct, indirect and total impacts.

    Attributes
    ----------
    mriot_data : np.array
        The input-output table data.
    mriot_reg_names : np.array
        Names of regions considered in the input-output table.
    sectors : np.array
        Sectors considered in the input-output table.
    total_prod : np.array
        Countries' total production.
    mriot_type : str
        Type of the adopted input-output table.
    reg_pos : dict
        Regions' positions within the input-output table and impact arrays.
    reg_dir_imp : list
        Regions undergoing direct impacts.
    years : np.array
        Years of the considered hazard events for which impact is calculated.
    direct_impact : np.array
        Direct impact array.
    direct_aai_agg : np.array
        Average annual direct impact array.
    indirect_impact : np.array
        Indirect impact array.
    indirect_aai_agg : np.array
        Average annual indirect impact array.
    total_impact : np.array
        Total impact array.
    total_aai_agg : np.array
        Average annual total impact array.
    io_data : dict
        Dictionary with the coefficients, inverse and risk_structure matrixes and
        the selected input-output modeling approach.
    """

    def __init__(self):
        """Initialize SupplyChain."""
        self.mriot_data = np.array([], dtype='f')
        self.mriot_reg_names = np.array([], dtype='str')
        self.sectors = np.array([], dtype='str')
        self.total_prod = np.array([], dtype='f')
        self.mriot_type = ''
        self.reg_pos = {}
        self.years = np.array([], dtype='f')
        self.direct_impact = np.array([], dtype='f')
        self.direct_aai_agg = np.array([], dtype='f')
        self.indirect_impact = np.array([], dtype='f')
        self.indirect_aai_agg = np.array([], dtype='f')
        self.total_impact = np.array([], dtype='f')
        self.total_aai_agg = np.array([], dtype='f')
        self.io_data = {}

    def read_wiod16(self, year=2014, range_rows=(5,2469),
                    range_cols=(4,2468), col_iso3=2,
                    col_sectors=1):
        """Read multi-regional input-output tables of the 2016 release of the
        WIOD project: http://www.wiod.org/database/wiots16

        Parameters
        ----------
        year : int
            Year of WIOD table to use. Valid years go from 2000 to 2014.
            Default year is 2014.
        range_rows : tuple
            initial and end positions of data along rows. Default is (5,2469).
        range_cols : tuple
            initial and end positions of data along columns. Default is (4,2468).
        col_iso3 : int
            column with countries names in ISO3 codes. Default is 2.
        col_sectors : int
            column with sector names. Default is 1.
        References
        ----------
        [1] Timmer, M. P., Dietzenbacher, E., Los, B., Stehrer, R. and de Vries, G. J.
        (2015), "An Illustrated User Guide to the World Input–Output Database: the Case
        of Global Automotive Production", Review of International Economics., 23: 575–605

        """

        file_name = 'WIOT{}_Nov16_ROW.xlsb'.format(year)
        file_loc = WIOD_DIRECTORY / file_name

        if not file_loc in WIOD_DIRECTORY.iterdir():
            download_link = WIOD_FILE_LINK + file_name
            u_fh.download_file(download_link, download_dir=WIOD_DIRECTORY)
            LOGGER.info('Downloading WIOD table for year %s', year)
        mriot = pd.read_excel(file_loc, engine='pyxlsb')

        start_row, end_row = range_rows
        start_col, end_col = range_cols

        self.sectors = mriot.iloc[start_row:end_row, col_sectors].unique()
        self.mriot_reg_names = mriot.iloc[start_row:end_row, col_iso3].unique()
        self.mriot_data = mriot.iloc[start_row:end_row,
                                     start_col:end_col].values
        self.total_prod = mriot.iloc[start_row:end_row, -1].values
        self.reg_pos = {
            name: range(len(self.sectors)*i, len(self.sectors)*(i+1))
            for i, name in enumerate(self.mriot_reg_names)
            }
        self.mriot_type = 'WIOD'

    def calc_sector_direct_impact(self, hazard, exposure, imp_fun_set,
                                  selected_subsec="service"):
        """Calculate direct impacts.

        Parameters:
        ----------
        hazard : Hazard
            Hazard object for impact calculation.
        exposure : Exposures
            Exposures object for impact calculation. For WIOD tables, exposure.region_id
            must be country names following ISO3 codes.
        imp_fun_set : ImpactFuncSet
            Set of impact functions.
        selected_subsec : str or list
            Positions of the selected sectors. These positions can be either
            defined by the user by passing a list of values, or by using built-in
            sectors' aggregations for the WIOD data passing a string with possible
            values being "service", "manufacturing", "agriculture" or "mining".
            Default is "service".

        """

        if isinstance(selected_subsec, str):
            built_in_subsec_pos = {'service': range(26, 56),
                                   'manufacturing': range(4, 23),
                                   'agriculture': range(0, 1),
                                   'mining': range(3, 4)}
            selected_subsec = built_in_subsec_pos[selected_subsec]

        dates = [
            dt.datetime.strptime(date, "%Y-%m-%d")
            for date in hazard.get_event_date()
            ]
        self.years = np.unique([date.year for date in dates])

        unique_exp_regid = exposure.gdf.region_id.unique()
        self.direct_impact = np.zeros(shape=(len(self.years),
                                             len(self.mriot_reg_names)*len(self.sectors)))

        self.reg_dir_imp = []
        for exp_regid in unique_exp_regid:
            reg_exp = Exposures(exposure.gdf[exposure.gdf.region_id == exp_regid])
            reg_exp.check()

            # Normalize exposure
            total_reg_value = reg_exp.gdf['value'].sum()
            reg_exp.gdf['value'] /= total_reg_value

            # Calc impact for country
            imp = Impact()
            imp.calc(reg_exp, imp_fun_set, hazard)
            imp_year_set = np.array(list(imp.calc_impact_year_set(imp).values()))

            mriot_reg_name = self._map_exp_to_mriot(exp_regid, self.mriot_type)

            self.reg_dir_imp.append(mriot_reg_name)

            subsec_reg_pos = np.array(selected_subsec) + self.reg_pos[mriot_reg_name][0]
            subsec_reg_prod = self.mriot_data[subsec_reg_pos].sum(axis=1)

            imp_year_set = np.repeat(imp_year_set, len(selected_subsec)
                                     ).reshape(len(self.years),
                                               len(selected_subsec))
            direct_impact_reg = np.multiply(imp_year_set, subsec_reg_prod)

            # Sum needed below in case of many ROWs, which are aggregated into
            # one country as per WIOD table.
            self.direct_impact[:, subsec_reg_pos] += direct_impact_reg.astype(np.float32)

        # average impact across years
        self.direct_aai_agg = self.direct_impact.mean(axis=0)

    def calc_indirect_impact(self, io_approach='ghosh'):
        """Calculate indirect impacts according to the specified input-output
        appraoch. This function needs to be run after calc_sector_direct_impact.

        Parameters
        ----------
        io_approach : str
            The adopted input-output modeling approach. Possible approaches
            are 'leontief', 'ghosh' and 'eeioa'. Default is 'gosh'.

        References
        ----------
        [1] W. W. Leontief, Output, employment, consumption, and investment,
        The Quarterly Journal of Economics 58, 1944.
        [2] Ghosh, A., Input-Output Approach in an Allocation System,
        Economica, New Series, 25, no. 97: 58-64. doi:10.2307/2550694, 1958.
        [3] Kitzes, J., An Introduction to Environmentally-Extended Input-Output
        Analysis, Resources, 2, 489-503; doi:10.3390/resources2040489, 2013.
        """

        io_switch = {'leontief': self._leontief_calc, 'ghosh': self._ghosh_calc,
                     'eeioa': self._eeioa_calc}

        # Compute coefficients based on selected IO approach
        coefficients = np.zeros_like(self.mriot_data, dtype=np.float32)
        if io_approach in ['leontief', 'eeioa']:
            for col_i, col in enumerate(self.mriot_data.T):
                if self.total_prod[col_i] > 0:
                    coefficients[:, col_i] = np.divide(col, self.total_prod[col_i])
                else:
                    coefficients[:, col_i] = 0
        else:
            for row_i, row in enumerate(self.mriot_data):
                if self.total_prod[row_i] > 0:
                    coefficients[row_i, :] = np.divide(row, self.total_prod[row_i])
                else:
                    coefficients[row_i, :] = 0

        inverse = np.linalg.inv(np.identity(len(self.mriot_data)) - coefficients)
        inverse = inverse.astype(np.float32)

        # Calculate indirect impacts
        self.indirect_impact = np.zeros_like(self.direct_impact, dtype=np.float32)
        risk_structure = np.zeros(np.shape(self.mriot_data) + (len(self.years),),
                                  dtype=np.float32)

        # Loop over years indices:
        for year_i, _ in enumerate(tqdm(self.years)):
            direct_impact_yearly = self.direct_impact[year_i, :]

            direct_intensity = np.zeros_like(direct_impact_yearly)
            for idx, (impact, production) in enumerate(zip(direct_impact_yearly,
                                                           self.total_prod)):
                if production > 0:
                    direct_intensity[idx] = impact/production
                else:
                    direct_intensity[idx] = 0

            # Calculate risk structure based on selected IO approach
            risk_structure = io_switch[io_approach](direct_intensity, inverse,
                                                    risk_structure, year_i)
            # Total indirect risk per sector/country-combination:
            self.indirect_impact[year_i, :] = np.nansum(
                risk_structure[:, :, year_i], axis=0)

        self.indirect_aai_agg = self.indirect_impact.mean(axis=0)

        self.io_data = {}
        self.io_data.update({'coefficients': coefficients, 'inverse': inverse,
                             'risk_structure' : risk_structure,
                             'io_approach' : io_approach})

    def calc_total_impact(self):
        """Calculate total impacts summing direct and indirect impacts."""
        self.total_impact = self.indirect_impact + self.direct_impact
        self.total_aai_agg = self.total_impact.mean(axis=0)

    def _map_exp_to_mriot(self, exp_regid, mriot_type):
        """
        Map regions names in exposure into Input-output regions names.
        exp_regid must be according to ISO 3166 numeric country codes.
        """

        if mriot_type == 'WIOD':
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha3")
            idx_country = np.where(self.mriot_reg_names == mriot_reg_name)[0]

            if not idx_country.size > 0.:
                mriot_reg_name = 'ROW'

        elif mriot_type == '':
            mriot_reg_name = exp_regid

        return mriot_reg_name

    def _leontief_calc(self, direct_intensity, inverse, risk_structure, year_i):
        """Calculate the risk_structure based on the Leontief approach."""
        demand = self.total_prod - np.nansum(self.mriot_data, axis=1)
        degr_demand = direct_intensity*demand
        for idx, row in enumerate(inverse):
            risk_structure[:, idx, year_i] = row * degr_demand
        return risk_structure

    def _ghosh_calc(self, direct_intensity, inverse, risk_structure, year_i):
        """Calculate the risk_structure based on the Ghosh approach."""
        value_added = self.total_prod - np.nansum(self.mriot_data, axis=0)
        degr_value_added = np.maximum(direct_intensity*value_added,\
                                      np.zeros_like(value_added))
        for idx, col in enumerate(inverse.T):
           # Here, we iterate across columns of inverse (hence transpose used).
            risk_structure[:, idx, year_i] = degr_value_added * col
        return risk_structure

    def _eeioa_calc(self, direct_intensity, inverse, risk_structure, year_i):
        """Calculate the risk_structure based on the EEIOA approach."""
        for idx, col in enumerate(inverse.T):
            risk_structure[:, idx, year_i] = (direct_intensity * col) * self.total_prod[idx]
        return risk_structure
