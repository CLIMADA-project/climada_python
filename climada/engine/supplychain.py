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

Define the SupplyChain class.
"""

__all__ = ['SupplyChain']

import os
import datetime as dt
from tqdm import tqdm
import numpy as np
import pandas as pd
from iso3166 import countries_by_alpha3 as ctry_iso3
from iso3166 import countries_by_numeric as ctry_ids
from climada.engine import Impact
from climada.util.constants import DATA_DIR, SYSTEM_DIR
from climada.entity.exposures.base import Exposures

SUP_DATA_DIR = os.path.join(DATA_DIR, 'supplychain')
WIOD_FILE = 'WIOT2014_Nov16_ROW.xlsx'

class SupplyChain():
    """SupplyChain definition. Provides methods for the entire supplychain-risk
    workflow and attributes holding the workflow's data and results.

    Attributes:
        mriot_data (np.array): 2-dim np.array of floats representing the data of
            a full multi-regional input-output table (mriot).
        countries (np.array): 1-dim np.array of strings containing the full list
            of countries represented in the mriot, corresponding to the columns/
            rows of mriot_data. For these countries risk calculations can be made.
        countries_iso3 (np.array): similar to .countries, but containing the
            countries' respective iso3-codes.
        sectors (np.array): 1-dim np.array of strings containing the full
            list of sectors represented in the mriot, corresponding to the columns/
            rows of mriot_data. For these sectors risk calculations can be made.
        total_prod (np.array): 1-dim arrays of floats representing the total
            production value of each country/sector-pair, i.e. each sector's
            total production per country.
        n_countries (int): number of countries represented in the used mriot.
            Equals the number of unique entries in the countries list.
        n_sectors (int): number of sectors represented in the used mriot.
            Equals the number of unique entries in the sectors list.
        mriot_type (str): short string describing the mriot used for analysis.
        values_pos (np.array): n-dim arrays where n is the number of countries
            being modeled. It containes positions of the mriot values of each 
            considered country.
        
    Attributes storing results of risk calculations:
        years (np.array): 1-dim np.array containing all years for which impact
            calculations where made (in yyyy format).
        direct_impact (np.array): 2-dim np.array containing an impact-YEAR-set
            with direct impact per year on each country/sector-pair.
        direct_aai_agg (np.array): 1-dim np.array containing the average annual
            direct impact for each country/sector-pair.
        indirect_impact (np.array): 2-dim np.array containing an impact-YEAR-set
            with indirect impact per year on each country/sector-pair.
        indirect_aai_agg (np.array): 1-dim np.array containing the average annual
            indirect impact for each country/sector-pair.
        total_impact (np.array): 2-dim array containing an impact-year-set with
            total (i.e. sum direct+indirect) impact per year on each 
            country/sector-pair.
        total_aai_agg (np.array): 1-dim np.array containing the average annual
            total impact for each country/sector-pair.
        io_data (dict): dictionary with four key:value-pairs:
            coefficients (np.array): 2-dim np.array containing the technical or
                allocation coefficient matrix, depending on employed io approach.
            inverse (np.array): 2-dim np.array containing Leontief or Ghosh
                inverse matrix, depending on employed io approach.
            io_approach (str): string informing about which io approach was
                used in calculation of indirect risk.
            risk_structure (np.array): 3-dim np.array containing for each year
                the risk relations between all sector/country-pairs.
            For further theoretical background, see documentation.
        """
    def __init__(self):
        """Initialization"""
        self.mriot_data = np.array([], dtype='f')
        self.countries = np.array([], dtype='str')
        self.countries_iso3 = np.array([], dtype='str')
        self.sectors = np.array([], dtype='str')
        self.total_prod = np.array([], dtype='f')
        self.n_countries = 0
        self.n_sectors = 0
        self.mriot_type = 'None'
    
    def read_wiot(self, file_path=SYSTEM_DIR, file_name=WIOD_FILE):
        """Read multi-regional input-output table of the WIOD project. 
        See www.wiod.org and the following paper: Timmer, M. P., Dietzenbacher, 
        E., Los, B., Stehrer, R. and de Vries, G. J. (2015), "An Illustrated 
        User Guide to the World Input–Output Database: the Case of Global 
        Automotive Production", Review of International Economics., 23: 575–605
        
        The function currently support the WIOT 2014, 2016 release.
        Direct link to file:
            http://www.wiod.org/protected3/data16/wiot_ROW/WIOT2014_Nov16_ROW.xlsb
        Currently, the file needs to be downloaded and as .xlsx after download.
        Future versions will allow the automatic download of the WIOT 2014 as 
        as other IO tables.
        
        Parameters:
            file_path (str): path to the folder where the wiod table is stored.
            file_name (str): name of the wiod table file.
        """
        
        row_st,row_end=(5,2469)
        col_iso3=2 
        col_sect,row_sect_end=(1,61)
        col_data_st,col_data_end=(4,2468)
        
        file_path_name = os.path.join(file_path, file_name)
        # TODO: if not file, download file
        # TODO: download multiple wiot tables (different years) and so adjust
        # self.mriot_data, self.total_prod structure accordingly
        mriot = pd.read_excel(file_path_name)

        sectors = mriot.iloc[row_st:row_sect_end, col_sect].values
        countries_iso3 = np.unique(mriot.iloc[row_st:row_end, col_iso3])
        # move Rest Of World (ROW) at the end of the array as countries 
        # are in chronological order
        idx_row = np.where(countries_iso3 == 'ROW')[0][0]
        countries_iso3 = np.hstack([countries_iso3[:idx_row], 
                                    countries_iso3[idx_row+1:], np.array('ROW')])

        mriot_data = mriot.iloc[row_st:row_end, 
                                col_data_st:col_data_end].values
        total_prod = mriot.iloc[row_st:row_end, -1].values
        
        countries = []
        for iso3 in countries_iso3:
            try:
                countries.append(ctry_iso3[iso3][0])
            except KeyError:
                countries.append('Rest of World')

        n_countries = len(countries_iso3)
        n_sectors = len(sectors)
        
        self.mriot_data = mriot_data
        self.countries = np.array(countries, dtype=str)
        self.countries_iso3 = countries_iso3
        self.sectors = sectors
        self.total_prod = total_prod
        self.n_countries = n_countries
        self.n_sectors = n_sectors
        self.mriot_type = 'wiod'

    def calc_sector_direct_impact(self, hazard, exposure, imp_fun_set, 
                                  sector_type=None, sec_subsec=None):
        """Calculate for each country/sector-combination the direct impact per year.
        I.e. compute one year impact set for each country/sector combination. Returns
        the notion of a supplychain year impact set, which is a dataframe with size
        (n years) * ((n countries)*(n sectors)).

        Parameters:
            hazard (Hazard): Hazard object for impact calculation.
            exposure (Exposures): Exposures object for impact calculation.
            imp_fun_set (ImpactFuncSet): Set of impact functions.
            sector_type (str): It allows setting the start and end positions of 
                CLIMADA-default sectors in the mriot table. Possible values are
                "service", "manufacturing", "agriculture" and "mining".
            sec_subsec (list): List with starting and ending positions of sector
                in the mriot table. To be provided if sector_type is None.
        """

        if not sec_subsec:
            built_in_sec_subsec = {'service': [26, 56],
                                   'manufacturing': [4, 23],
                                   'agriculture': [0, 1],
                                   'mining': [3, 4]}
            
            sec_subsec = built_in_sec_subsec[sector_type]

        # Positions of subsectors in table
        init_pos, end_pos = sec_subsec

        dates = [dt.datetime.strptime(date, "%Y-%m-%d")\
                      for date in hazard.get_event_date()]
        years = np.unique([date.year for date in dates])
        
        # Keep original order of countries
        _, cntry_idx = np.unique(exposure.region_id, return_index=True)
        unique_regid_same_order = exposure.region_id[np.sort(cntry_idx)]
        
        n_years = len(years)
        n_subsecs = end_pos - init_pos
        direct_impact = np.zeros(shape=(n_years, 
                                        self.n_countries*self.n_sectors))
        values_pos = {}
        
        for cntry in unique_regid_same_order:
            cntyr_exp = Exposures(exposure[exposure.region_id == cntry])
            cntyr_exp.check()
            
            # Normalize exposure
            total_ctry_value = cntyr_exp.loc[:, 'value'].sum()
            cntyr_exp.loc[:, 'value'] = cntyr_exp.value.div(total_ctry_value)
            
            # Calc impact for country
            imp = Impact()
            imp.calc(cntyr_exp, imp_fun_set, hazard)
            imp_year_set = np.array(list(imp.calc_impact_year_set(imp).values()))

            # Total production of country
            cntry_iso3 = ctry_ids.get(str(cntry)).alpha3
            idx_country = np.where(self.countries_iso3 == cntry_iso3)[0]
            
            if idx_country.size > 0.:
                step_in_table = idx_country[0]*self.n_sectors
            else:
                step_in_table = (self.n_countries-1)*self.n_sectors
                cntry_iso3 = 'ROW'
                
            values_pos_cntry = range(step_in_table+init_pos,
                                     step_in_table+end_pos)
            subsec_cntry_prod = self.mriot_data[values_pos_cntry].sum(axis=1)
            values_pos[cntry_iso3] = values_pos_cntry
            
            imp_year_set = np.repeat(imp_year_set,
                                     n_subsecs).reshape(n_years, n_subsecs)
            direct_impact_cntry = np.multiply(imp_year_set, subsec_cntry_prod)
            
            # Sum needed below in case of many ROWs, which need be aggregated. 
            # Another option is to keep them separate, but it requires changing 
            # all functions below as impact matrix will no longer have 
            # dim = (n_years, self.n_countries*self.n_sectors) but 
            # dim = (n_years, (self.n_countries+#ROWs_countries-1)*self.n_sectors)
            direct_impact[:, values_pos_cntry] += direct_impact_cntry.astype(np.float32)
        
        self.direct_impact = direct_impact
        # average impact across years
        self.direct_aai_agg = self.direct_impact.mean(axis=0)
        self.years = years
        self.values_pos = values_pos

    def calc_indirect_impact(self, io_approach='ghosh'):
        """Estimate indirect impact based on direct impact using input-output (IO)
        methodology. There are three IO approaches to choose from (see Parameters).
            [1] Standard Input-Output (IO) Model;
                W. W. Leontief, Output, employment, consumption, and investment,
                The Quarterly Journal of Economics 58 (2) 290?314, 1944
            [2] Ghosh Model;
                Ghosh, A., Input-Output Approach in an Allocation System,
                Economica, New Series, 25, no. 97: 58-64. doi:10.2307/2550694, 1958
            [3] Environmentally Extended Input-Output Analysis (EEIOA);
                Kitzes, J., An Introduction to Environmentally-Extended Input-Output Analysis,
                Resources 2013, 2, 489-503; doi:10.3390/resources2040489, 2013
        Parameters:
            io_approach (str): string specifying which IO approach the user would
                like to use. Either 'leontief', 'ghosh' (default) or 'eeioa'.
        """
        
        io_switch = {'leontief': self._leontief_calc, 'ghosh': self._ghosh_calc, 
                     'eeioa': self._eeioa_calc}
        io_data = {}
        
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
        io_data['coefficients'] = coefficients

        inverse = np.linalg.inv(np.identity(len(self.mriot_data)) - coefficients)
        inverse = inverse.astype(np.float32)
        
        # Calculate indirect impacts
        indirect_impact = np.zeros_like(self.direct_impact, dtype=np.float32)
        risk_structure = np.zeros(np.shape(self.mriot_data) + (len(self.years),), 
                                  dtype=np.float32)
        
        # Loop over years indices:
        for year_i, year in enumerate(tqdm(self.years)):
            direct_impact_yearly = self.direct_impact[year_i, :]

            direct_intensity = np.zeros_like(direct_impact_yearly)
            for idx, (impact, production) in enumerate(zip(direct_impact_yearly, 
                                                           self.total_prod)):
                if production > 0:
                    direct_intensity[idx] = impact/production
                else:
                    direct_intensity[idx] = 0
                    
            # Calculate risk structure based on selected IO approach
            risk_structure = io_switch[io_approach](io_data['coefficients'], \
                                  direct_intensity, inverse, risk_structure, year_i)
            # Total indirect risk per sector/country-combination:
            indirect_impact[year_i, :] = np.nansum(risk_structure[:, :, year_i], axis=1)

        io_data['inverse'] = inverse
        io_data['risk_structure'] = risk_structure
        io_data['io_approach'] = io_approach
        self.io_data = io_data
        self.indirect_impact = indirect_impact
        self.indirect_aai_agg = self.indirect_impact.mean(axis=0)
    
    def calc_total_impact(self):
        """Calculates the total impact and total average annual impact on each 
        country/sector """
        self.total_impact = self.indirect_impact + self.direct_impact
        self.total_aai_agg = self.total_impact.mean(axis=0)

    def _leontief_calc(self, coefficients, direct_intensity, 
                       inverse, risk_structure, year_i):
        """It calculates the risk_structure based on the Leontief approach"""
        demand = self.total_prod - np.nansum(self.mriot_data, axis=1)
        degr_demand = direct_intensity*demand
        for idx, row in enumerate(inverse):
            risk_structure[:, idx, year_i] = row * degr_demand
        return risk_structure

    def _ghosh_calc(self, coefficients, direct_intensity, inverse, 
                    risk_structure, year_i):
        """It calculates the risk_structure based on the Ghosh approach"""
        value_added = self.total_prod - np.nansum(self.mriot_data, axis=0)
        degr_value_added = np.maximum(direct_intensity*value_added,\
                                      np.zeros_like(value_added))
        for idx, col in enumerate(inverse.T):
           # Here, we iterate across columns of inverse (hence transpose used).
            risk_structure[:, idx, year_i] = degr_value_added * col
        return risk_structure

    def _eeioa_calc(self, coefficients, direct_intensity, inverse, 
                    risk_structure, year_i):
        """It calculates the risk_structure based on the EEIOA approach"""
        
        for idx, col in enumerate(inverse.T):
            risk_structure[:, idx, year_i] = (direct_intensity * col) * self.total_prod[idx]
        return risk_structure