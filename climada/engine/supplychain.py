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

Define the SupplyChain class, which provides functionality to calculate
industry-sector-specific direct risk and resultant supply chain risk due to sector-
sector-interdependencies. For methodological details on the notion of supply chain
(or: "indirect") risk and its calculation, see [...].
"""

import os
import glob
#import pdb   #Just for debugging
import logging
import datetime as dt
from tqdm import tqdm
import numpy as np
import pandas as pd
from iso3166 import countries_by_alpha3 as ctry_iso3
from iso3166 import countries_by_numeric as ctry_ids
#from iso3166 import countries_by_name as ctry_names
from cartopy.io import shapereader

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures
from climada.entity.exposures.spam_agrar import SpamAgrar
from climada.entity.exposures.litpop import LitPop
from climada.entity.exposures.mining_exp import MiningExp
from climada.entity.exposures.manufacturing_exp import ManufacturingExp
from climada.entity.exposures.utilities_exp import UtilitiesExp
from climada.entity.exposures.forest_exp import ForestExp
from climada.hazard import TropCyclone, TCTracks
from climada.hazard.centroids import Centroids
from climada.entity.impact_funcs import ImpactFuncSet, IFTropCyclone
from climada.engine import Impact
from climada.util.constants import DATA_DIR, GLB_CENTROIDS_MAT

LOGGER = logging.getLogger(__name__)

SUP_DATA_DIR = os.path.join(DATA_DIR, 'supplychain')
WIOD_FILE = 'WIOT2014_Nov16_ROW.xlsx'

class SupplyChain():
    """SupplyChain definition. Provides methods for the entire supplychain-risk
    workflow and attributes holding the workflow's data and results.

    Attributes:
        mriot_data (np.array): 2-dim np.array of floats representing the data of
            a full multi-regional input-output table (mriot).
        aggregated_mriot (dict): a dictionary with three key:value-pairs:
                countries (np.array): aggregated country labels.
                sectors (np.array): aggregated sectors labels.
                aggregation_info (dict): stores information on which subsectors
                    (values; list of strings) were aggregated into which main
                    sectors (keys; str). See private method _aggregated_mriot for more on
                    the notion of "aggregation".
        countries (np.array): 1-dim np.array of strings containing the full list
            of countries represented in the mriot, corresponding to the columns/
            rows of mriot_data. For these countries risk calculations can be made.
        countries_iso3 (np.array): similar to .countries, but containing the
            countries' respective iso3-codes.
        sectors (np.array): 1-dim np.array of strings containing the full
            list of sectors represented in the mriot, corresponding to the columns/
            rows of mriot_data. For these sectors risk calculations can be made.
        main_sectors (np.array): 1-dim np.array of strings containing for the full
            list of sectors represented in the mriot its corresponding climada-
            defined main sectors. This mapping is provided by the user prior to
            reading the data.
        total_prod (np.array): 1-dim arrays of floats representing the total
            production value of each country/sector-pair, i.e. each sector's
            total production per country.
        n_countries (int): number of countries represented in the used mriot.
            Equals the number of unique entries in the countries list.
        n_sectors (int): number of sectors represented in the used mriot.
            Equals the number of unique entries in the sectors list.
        mriot_type (str): short string describing the mriot used for analysis. 
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
        """Empty initialization: all required attributes are set to empty
        target data structures. To start with default mrio table, call method
        read_wiod next.
        Parameters:
            None
        """
        self.mriot_data = np.array([], dtype='f')
        self.countries_iso3 = np.array([], dtype='str')
        self.countries = np.array([], dtype='str')
        self.sectors = np.array([], dtype='str')
        self.main_sectors = np.array([], dtype='str')
        self.total_prod = np.array([], dtype='f')
        self.n_countries = 0
        self.n_sectors = 0
        self.mriot_type = 'None'

    def read_wiod(self, file_path=SUP_DATA_DIR, file_name=WIOD_FILE,\
                  rows=2464, cols='E:CPX', tot_prod_col='CYK'):
        """Function to read multi-regional input-output table of the WIOD
        project. See www.wiod.org and the following paper:
            Timmer, M. P., Dietzenbacher, E., Los, B., Stehrer, R. and de Vries, G. J. (2015),
            "An Illustrated User Guide to the World Input–Output Database:
            the Case of Global Automotive Production",
            Review of International Economics., 23: 575–605
        Function was tested with the WIOT 2014 of the 2016 release.
        Direct link to file:
            http://www.wiod.org/protected3/data16/wiot_ROW/WIOT2014_Nov16_ROW.xlsb
        
        IMPORTANT: you need to re-save the file as .xlsx after download (standard
             excel format). The uncommon .xlsb is not supported.
        
        The function also fills the class attribute "aggregated_mriot".
        
        Parameters:
            file_path (str): string specifying the path to the FOLDER in which 
                the wiod table file is stored.
            file_name (str): string specifying the name of the wiod table file.
            rows: number of rows to read. Only change if you know what you're doing.
            cols: column-range containing the main mriot data. Only change if you know what you're doing.
            tot_prod_col: column containing total production values. Only change if you know what you're doing.
        Returns:
            None
        SUGGESTED IMPROVEMENTS:
            Implement automatic download of file if wrong or no file is provided
            by user.
            
            Due to time constraints, there is too much hard-coded functionality
            which makes the function inflexible with regard to possible future 
            changes to the source file.
        """
        try:
            countries_iso3 = np.array(pd.read_excel(os.path.join(file_path, file_name),\
                            sheet_name = '2014', usecols='C', skiprows=5,\
                            nrows=rows), dtype=str)
        except FileNotFoundError:
            LOGGER.error("We couldn't find the required wiod table file in the "\
                         "given location. Please download file and/or check "\
                         "whether correct path and name was given.")
            raise FileNotFoundError("Couldn't find required file.")
            #TO-DO: IMPLEMENT AUTOMATIC DOWNLOAD OF FILE
        #Number of countries:          
        n_countries = len(np.unique(countries_iso3))

        # Next, we read the sectors. Here, we rely on proper data preparation
        # by the user (separate sheet with proper mappig of wiod to climada sectors):
        sectors = np.array(pd.read_excel(os.path.join(file_path, file_name),\
                            sheet_name = 'climada_sectors', usecols='A,B',\
                            nrows=57, header=None), dtype=str)
        ### COME BACK TO THE ABOVE AS FOR WRONG USER INPUT TURNS OUT QUITE MESSY
        ### (BAD EXCEPTION HANDLING).

        # More sanity checks on user's data preparation:
        if not sectors[0,0] == 'sector_name' or \
           not sectors[0, 1] == 'main_sector_name':
               raise IOError("Please make sure to prepare data in correct format. " \
                             "The column headers are not named correctly. " \
                             "Refer to documentation for help.")
        main_sectors = sectors[1:,1].copy()
        sectors = sectors[1:,0]
        # Sanity check:
        #    The arrays "sectors" and "main_sectors" are of same length
        #    if data was properly prepared:
        if not len(sectors) == len(main_sectors):
            raise IOError("Please make sure to prepare data in correct format. " \
                             "There must be the same number of sectors and " \
                             "main sectors, which is not the case. " \
                             "Refer to documentation for help.")
        # To represent full list of data points in data set, we need to replicate
        # ("tile") the list of sectors n times, with n = n_countries:
        sectors = np.tile(sectors, n_countries)
        main_sectors = np.tile(main_sectors, n_countries)

        # Finally, we read the actual data of commodity exchanges between
        # countries and sectors as well as total prodcution of each country/sector-combination:
        
        mriot_data = np.array(pd.read_excel(os.path.join(file_path, file_name),\
                            sheet_name = '2014', usecols=cols, skiprows=5,\
                            nrows=rows), dtype=np.float32)
        total_prod = np.array(pd.read_excel(os.path.join(file_path, file_name),\
                            sheet_name = '2014', usecols=tot_prod_col, skiprows=5,\
                            nrows=rows), dtype=np.float32)
        
        self.countries_iso3 = countries_iso3[:,0]
        # We construct the "countries" array based on the iso3 codes
        # using the iso3166 package:
        countries = []
        for iso3 in self.countries_iso3:
            try:
                countries.append(ctry_iso3[iso3][0])
            except KeyError:
                countries.append('Rest of World')
        self.countries = np.array(countries, dtype=str)
        self.sectors = sectors
        self.main_sectors = main_sectors
        self.total_prod = total_prod[:,0]
        self.n_countries = n_countries
        self.n_sectors = len(set(self.sectors))
        self.mriot_type = 'wiod'
        
        self.mriot_data = mriot_data
        
        #Finally, aggregate labels and add to class attribute "aggregated_mriot":
        self._aggregate_labels()

    def default_exposures(self): 
        """Creates default Exposures for use in the supplychain context and saves
        them as hdf5 files to the data/supplychain/exposures folder. 
        
        NOTE: Calculation of all default exposures may take up to 1 hour. Refer to 
        documentation for other ways to obtain the required exposure files.

        Six default exposures are implemented; in the climada context these are 
        called the six "mainsectors": 
                agriculture
                forestry_fishing
                utilities
                services
                mining_quarrying
                manufacturing
        Returns:
            None; default

        SUGGESTED IMPROVEMENTS:
        Add possibility to pass argument "plot" which creates plots 
            (world maps with value distribution) of the created exposures (True) 
            or not (False)).
        """
        main_sectors = ['agriculture', 'forestry_fishing', 'utilities', \
                'services', 'mining_quarrying', 'manufacturing']

        file_names = {}
        file_exists = []
        for name in main_sectors:
            file_names[name] = 'GLB_' + name + '_XXX'
            if os.path.isfile(os.path.join(SUP_DATA_DIR, file_names[name])):
                file_exists.append(name)
        if not 'agriculture' in file_exists:
            self._create_agri_expo(file_names)
            
    ### !!! For now (February 2020) utilities, forestry and mining exposure
    ### classes all actually create manufacturing exposures as placeholders
    ### for testing purposes (until the respective classes are ready).
        if not 'utilities' in file_exists:
            self._create_utilities_expo(file_names)
        if not 'forestry_fishing' in file_exists:
            self._create_forest_expo(file_names)
        if not 'manufacturing' in file_exists:
            self._create_manu_expo(file_names)
        if not 'mining_quarrying' in file_exists:
            self._create_mining_expo(file_names)
        
    # We deal differently with the services exposures since default is already
    # per-country. The check for existing files is done in the static method 
    # _create_services_expo() directly.
        LOGGER.info("Calculation of services exposures may take up to "\
                           "45 minutes. Refer to documentation for other ways to "\
                           "obtain the required exposure files.")
        self._create_services_expo(file_names) 
        
        if file_exists: #Checks whether list "file_exists" ist empty or not.
            files_string = str(file_exists)
            LOGGER.info("For the following sectors no new exposures were " \
                "created as they seem to exist already (remove them from the supplychain "\
                "data folder if you want to create new ones): " + files_string)           
        
    def prepare_exposures(self, files_source=SUP_DATA_DIR, files_target=SUP_DATA_DIR,\
                          remove_restofw_ctries=True):
        """Loads existing global exposures and prepares them for use in supply
        chain context. Preparing means: splitting global exposures into
        per-country exposures, assigning 'RoW' (Rest of World) to countries not
        represented in the mriot that is used, normalizing each exposure with
        the country's total value and saving each in the required format.

        Optional parameter:
            files_source (str): path to folder where supplychain-ready exposusure
                files are located (as created by .default_exposures method).
            files_target (str): path to folder in which resulting exposures are
                are to be stored.
            remove_restofw_ctries (bool): only used for testing (see TestSupplyChain class).
                Only change if you know what you are doing.
        Returns:
            None

        SUGGESTED IMPROVEMENTS: 
        MAYBE ADD FLAG WHICH SHOWS WETHER NORMALIZATION HAS TAKEN
        PLACE ALREADY TO AVOID NORMALIZING TWICE (IF STANDARD WORKFLOW IS ADHERED
        TO SHOULD BE NO PROBLEM).
        """
        main_sectors = ['agriculture', 'forestry_fishing', 'utilities', \
                'services', 'mining_quarrying', 'manufacturing']
        mriot_ctries = np.unique(self.countries_iso3)
        has_restofw = False # Initialize as False. Will be set to True if we
                        # work with exposures in countries that are not
                        # covered by the mriot we use. These countries
                        # will be integrated into a Rest of World (ROW)
                        # exposure.
        ctry_ids['000'] = ('NA', 'NA', 'NA', 'NA', 'NA')
        for name in tqdm(main_sectors):
            if name != 'services':
                file_name = 'GLB_' + name + '_XXX'
                exp = Exposures()
                exp.read_hdf5(os.path.join(files_source, file_name))
                ctries = exp.loc[:, 'region_id'].unique()
                ctries[ctries == -99] = 000
                restofw_exp = Exposures() # Empty df to hold all exposures for countries
                                      # that are not covered in the mriot table.
                #pdb.set_trace()
                for ctry in ctries:
                    ctry_code = '{0:0>3}'.format(ctry)
                    country_iso3 = ctry_ids[ctry_code][2]
                    exp_new = exp.copy()
                    file_name = os.path.join(files_target, country_iso3 + '_'\
                                             + name + '_XXX')
                    if  country_iso3 in mriot_ctries:
                        if file_name not in glob.glob(os.path.join(files_target,\
                                            '*'+country_iso3+'*')):
                            exp_new = Exposures(exp_new[exp_new.loc[:, 'region_id'] == ctry],\
                                                crs=exp.crs)
                            total_ctry_value = exp_new.loc[:, 'value'].sum()
                            exp_new.loc[:, 'value'] = exp_new.loc[:, 'value'].div(total_ctry_value)
                            exp_new = exp_new.reset_index(drop=True)
                            LOGGER.info('Saving ' + name + ' ____  ' + country_iso3)
                            exp_new.loc[:, 'region_id'] = country_iso3
                            exp_new.write_hdf5(file_name)
                        else:
                            LOGGER.info('Exposure '+file_name+' already exists.')
                    else: # Values belong to Rest of World (ROW)
                        has_restofw = True # Are there countries  that are not covered by the mriot?
                        exp_new = Exposures(exp_new[exp_new.loc[:, 'region_id'] == ctry], crs=exp.crs)
                        restofw_exp = restofw_exp.append(exp_new)
                if has_restofw:
                    total_ctry_value = restofw_exp.loc[:, 'value'].sum()
                    restofw_exp.loc[:, 'value'] = restofw_exp.loc[:, 'value'].div(total_ctry_value)
                    restofw_exp = restofw_exp.reset_index(drop=True) 
                    LOGGER.info('Saving ' + name + ' ____  ' + 'Rest of World (ROW)')
                    restofw_exp.loc[:, 'region_id'] = 'ROW'
                    save_name = os.path.join(files_target, 'ROW_' + name + '_XXX')
                    exp_new.write_hdf5(save_name)
            # Services are already per-country exposures from the start (LitPop),
            # hence only need normalization and no split (but do consider Rest of World!).
            else:
                restofw_exp = Exposures()
                for fname in glob.glob(os.path.join(files_source, '*services*')):   
                    exp = Exposures()
                    exp.read_hdf5(fname)
                    country_iso3 = exp.loc[0, 'region_id']
                    country_iso3 = '{0:0>3}'.format(country_iso3)
                    if country_iso3 == 'ROW':
                        continue ## Not very pythonic!!
                    #Note about "continue" statement: this is to account for the case
                    #where a ROW_services_XXX exposure already exists in the data folder
                    #from a possible previous work with the model. Since ROW is not
                    #an ISO3 code, it would cause an error in the following line. Also,
                    #a ROW_services_XXX exposure is only created by the prepare_exposures
                    #method, hence we can leave it as is to aboiv multiple normalizations
                    #of the values.
                    country_iso3 = ctry_ids[country_iso3][2]
                    if country_iso3 in mriot_ctries:
                        total_ctry_value = exp.loc[:, 'value'].sum()
                        exp.loc[:, 'value'] = exp.loc[:, 'value'].div(total_ctry_value)
                        savename = os.path.join(files_target, country_iso3+\
                                                         '_services_'+'XXX')
                        exp.write_hdf5(savename)
                    else:
                        has_restofw = True
                        restofw_exp = restofw_exp.append(exp)
                        if remove_restofw_ctries:   
                            os.remove(fname) # Delete the original exposure file for
                                             # this country as its values will be integrated
                                             # into this sector's RoW exposure file.
                if has_restofw:
                    total_ctry_value = restofw_exp.loc[:, 'value'].sum()
                    restofw_exp.loc[:, 'value'] = restofw_exp.loc[:, 'value'].div(total_ctry_value)
                    restofw_exp = restofw_exp.reset_index(drop=True)
                    LOGGER.info('Saving ' + name + ' ____  ' + 'Rest of World (ROW)')
                    restofw_exp.loc[:, 'region_id'] = 'ROW'
                    save_name = os.path.join(files_target, 'ROW_' + name + '_XXX')
                    restofw_exp.write_hdf5(save_name)
        del ctry_ids['000'] # Reset to original state after import.
    #%%
        
    def create_default_haz(self, save_haz=True, file_path=SUP_DATA_DIR,\
                           file_name='sup_ib_hazard_default'):
        """Creates default hazard for use in supplychain workflow.
        Default hazard is of subtype TropicalCyclone and based on IB tracks
        database.
        
        NOTE: Calculation of default hazard may take up to 1 hour. Refer to 
        documentation for other ways to obtain a valid hazard.
        
        Prameters:
            save_haz (bool): If true (default), resulting hazard is saved as hdf5 file in
                folder given by file_path.
            file_path (str): Path to folder to which resulting hazard is saved
                (if save_haz=True).
            file_name (str): file name which is used to check whether hazard
                already exists. Only change if you know what you're doing. Mainly 
                used for testing."""
        try:
            haz = TropCyclone()
            haz.read_hdf5(os.path.join(file_path,\
                    file_name))
            LOGGER.info('Default hazard already exists. No new hazard created.')
            return haz
        except OSError:
            LOGGER.info("Couldn't find default hazard file. A new default hazard "\
                        "is created, which can take up to 30 minutes. Refer to "\
                           "documentation for other ways to obtain a valid hazard.")        
            tracks = TCTracks()
            tracks.read_ibtracs_netcdf() 
            
            haz = TropCyclone()
            centr = Centroids()
            centr.read_mat(GLB_CENTROIDS_MAT)
            haz.set_from_tracks(tracks, centr)
            haz.check()
            if save_haz:
                haz.write_hdf5(os.path.join(file_path, \
                       'sup_ib_hazard_default'))
            return haz
  
        
    def calc_direct_impact(self, haz, hazard_type='TC', exp_source_path=SUP_DATA_DIR, \
                           imp_target_path=SUP_DATA_DIR):
        """Calculate for each country/sector-combination the direct impact per year.
        I.e. compute one year impact set for each country/sector combination. Returns
        the notion of a supplychain year impact set, which is a dataframe with size
        (n years) * ((n countries)*(n sectors)).

        Parameters:
            haz (Hazard obj): Hazard object to work with in impact calculation.
                    Careful: supplychain worflow cannot proceed without valid hazard. 
                    Currently only works with objects of subtype TropicalCyclone.
                    If unsure, first generate hazard with sup.create_default_hazard().
            hazard_type (str): Of which type is the provided hazard? 
                    Note: Currently only Tropical cyclone ('TC'; default) possible.
            exp_source_path (str): path to folder in which the employed exposures
                are stored. Only change if you know what you're doing.
            imp_target_path (str): path to folder to which resulting impact arrays
                are stored.
        SUGGESTED IMPROVEMENTS:
            Add multi-hazard capabilities/flexibility to deal with other hazard
            types than tropical cyclones and multiple hazards at once. The main
            calculations inside the if-statement ('hazard_type==TC') should be
            easy to generalize as only the impact functions and naming conventions
            are TC-specific.
            
            A better way to deal with missing exposure files needs to be found. 
            Currently, 0 is set as impact for missing exposures so that calculation
            workflow can continue, but this obviously distorts the results. A list
            of missing exposures is collected. The solution to this issue needs to
            be solved at the exposure side though, by making sure that default exposures
            generated above do all cover all countries represented in WIOD.
        """
        # 1. Load default TC hazard
        # 2. Loop through all CTRY_mainsector_XXX files, loading each exposure.
        # 3. With each exposure, calculate a year impact set,
        #    using default TC hazard with Emanuel's impact function
        #    (note difference for NWP).
        ### Future: implement multi-hazard capabilities. 
           
        if hazard_type == 'TC':
            funcs = ImpactFuncSet()
            func = IFTropCyclone()
            unique_ctries, ind = np.unique(self.countries_iso3, return_index=True)
            unique_ctries = unique_ctries[np.argsort(ind)]
            unique_mainsectors, ind = np.unique(self.main_sectors, return_index=True)
            unique_mainsectors = unique_mainsectors[np.argsort(ind)] #reset order
            unique_subsectors, ind = np.unique(self.sectors, return_index=True)
            unique_subsectors = unique_subsectors[np.argsort(ind)]#reset order
            dates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in haz.get_event_date()]
            years = np.unique([date.year for date in dates])
            year_range = len(years)
            mainsector_impact = np.zeros((year_range, np.size(unique_ctries)*np.size(unique_mainsectors)))
            nwp_ctries = ['JPN', 'VNM', 'MYS', 'PHL', 'CHN', 'KOR', 'TWN'] # Ugly hard-coded; these ctries belong to Northwest-Pacific and use differend impact function parameters than the rest.
            missing_exp = [] #List to collect exposures which are missing.
            for ctry_i, ctry in enumerate(tqdm(unique_ctries)):
                LOGGER.info("Working on mainsectors of country: "+ctry+".")
                if ctry in nwp_ctries:
                    func.set_emanuel_usa(v_thresh=25, v_half=61, scale=0.08)
                else:
                    func.set_emanuel_usa(v_thresh=25, v_half=61, scale=0.64)
                funcs.append(func)
                funcs.check()
                for ms_i, msect in enumerate(unique_mainsectors):
                    exp = Exposures()
                    try:
                        exp.read_hdf5(os.path.join(exp_source_path, ctry + '_' + msect + '_XXX'))
                        exp.check()
                    except KeyError: #If required exposure file does not exist. See function header for comment.
                        missing_exp.append(ctry+'/'+msect)
                        mainsector_impact[:, ms_i + np.size(unique_mainsectors)*ctry_i] = 0
                        continue ## Not very pythonic!!!
                    exp.rename(index=str, columns={'if_DR': 'if_TC', 'if_': 'if_TC'}, inplace=True)
                        # .rename does not throw error if column name non-existent.
                    imp = Impact()
                    imp.calc(exp, funcs, haz)
                    imp_year_set = imp.calc_impact_year_set(imp) # Returns dictionary with years as keys.
                    mainsector_impact[:, ms_i + np.size(unique_mainsectors)*ctry_i] = list(imp_year_set.values())                      
                np.save(os.path.join(imp_target_path, hazard_type+'_mainsector_impact.npy'), mainsector_impact)
                
            subsector_impact = np.zeros((year_range, self.n_sectors*self.n_countries))
            for mainsector in unique_mainsectors:
                for subsector in self.aggregated_mriot['aggregation_info'][mainsector]:
                    sub_mask = np.transpose(self.sectors == subsector)
                    main_mask = np.transpose(self.aggregated_mriot['sectors'] == mainsector)
                    subsector_impact[:, np.nonzero(sub_mask)[0]] = mainsector_impact[:, np.nonzero(main_mask)[0]]
                    np.save(os.path.join(imp_target_path, hazard_type+'_subsector_impact.npy'), subsector_impact)
           
            # Impact in absolute terms according to each subsector's contribution
            # to mainsector production: multiply with each subsector's total production:
            subsector_production = self.mriot_data.sum(axis=1)
            direct_impact = np.multiply(subsector_impact, subsector_production)
            self.direct_impact = direct_impact.astype(np.float32)
            self.direct_aai_agg = self.direct_impact.mean(axis=0)
            self.years = years
            
            #Print list of exposures that were missing (if any):
            if missing_exp: #List empty or not
                missing_string = str(missing_exp)
                LOGGER.info("For the following country-sector-combinations no exposures "\
                    "were found (0 was assumed as impact value so that calculations "\
                    "can proceed; note that this might distort the analysis): " + missing_string)             
            
        else: 
            raise ValueError("Currently, only hazard_type 'TC' (tropical cyclone) is supported.")
            LOGGER.error("Currently, only hazard_type 'TC' (tropical cyclone) is supported.")

        ## QUITE SLOW, especially for the large services exposures!



    def read_direct_impact(self, haz, hazard_type='TC', fnamepart='_subsector_impact.npy'):
        """If direct impact was calculated before, use this method to read
        data. Saves calculation time. Existing impact file must be in the
        supplychain 'data' folder.

        Parameters:
            haz (Hazard obj): needed to identify for which years impact has been
                calculated. Is then added as attribute to SupplyChain instance
                (done in function calc_direct_impact() if normal workflow is followed).
            hazard_type (str): identifier of hazard on which existing impact file is based.
                Used to derive file name of to-be-loaded file.
            fnamepart (str): name (without hazard prefix) of file containing the
                impact data in case it does not follow naming conventions of
                method calc_direct_impact. Note that by default impact on subsector
                level is read. Only read mainsector data if you know what you
                are doing.
        """
        try:
            self.direct_impact = np.load(os.path.join(SUP_DATA_DIR, hazard_type+fnamepart))
        except FileNotFoundError:
            LOGGER.error("We couldn't find the given file. Please make sure to" \
                         " provide a correct file name (or use default) and store" \
                         " the file in the supplychain>data folder. If no impact has" \
                         " been calculated yet, use 'method calc_direct_impact' instead.")
            raise FileNotFoundError
        self.direct_aai_agg = self.direct_impact.mean(axis=0)
        
        dates = [dt.datetime.strptime(date, "%Y-%m-%d") for date in haz.get_event_date()]
        years = np.unique([date.year for date in dates])
        self.years = years

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
        if io_approach not in ['leontief', 'ghosh', 'eeio']:
            LOGGER.warning('Wrong paramter provided. Using default value ' \
                           '(Ghosh approach) instead.')
            io_approach = 'ghosh'            
        # Compute technical coefficient or allocation coefficient matrix (for
        # leontief or ghosh approach, respectively):
        io_data = {}

        coefficients = np.zeros_like(self.mriot_data, dtype=np.float32)
        if io_approach in ['leontief', 'eeio']:
            for col_i, col in enumerate(self.mriot_data.T): # Loop through columns
                if self.total_prod[col_i] > 0:
                    coefficients[:, col_i] = np.divide(col, self.total_prod[col_i])
                else:
                    coefficients[:, col_i] = 0
        else:
            for row_i, row in enumerate(self.mriot_data): # Loop through rows
                if self.total_prod[row_i] > 0:
                    coefficients[row_i, :] = np.divide(row, self.total_prod[row_i])
                else:
                    coefficients[row_i, :] = 0
        io_data['coefficients'] = coefficients

        inverse = np.linalg.inv(np.identity(len(self.mriot_data)) - coefficients)
        inverse = inverse.astype(np.float32)
        # Above: either Leontief or Ghosh inverse, depending on coefficients used.
        # Calculation is equivalent.
        indirect_impact = np.zeros_like(self.direct_impact, dtype=np.float32)
        risk_structure = np.zeros(np.shape(self.mriot_data) + (len(self.years), ), dtype=np.float32) #Tuple concatenation.

        # The following calculations are done per year; i.e. within a large
        # encompassing loop through year indices:
        for year_i, year in enumerate(tqdm(self.years)):
            direct_impact_yearly = self.direct_impact[year_i, :]

            direct_intensity = np.zeros_like(direct_impact_yearly)
            for idx, (impact, production) in enumerate(zip(direct_impact_yearly, self.total_prod)):
                if production > 0:
                    direct_intensity[idx] = impact/production
                else:
                    direct_intensity[idx] = 0
            # Now calculate risk structure, switching IO approach depending on provided parameter:
            io_switch = {'leontief': self._leontief_calc, 'ghosh': self._ghosh_calc, 'eeio': self._eeio_calc}
            risk_structure = io_switch[io_approach](io_data['coefficients'], \
                                  direct_intensity, inverse, risk_structure, year_i)
            # Finally, sum columns to obtain total indirect risk per sector/country-combination:
            indirect_impact[year_i, :] = np.nansum(risk_structure[:, :, year_i], axis=0)

        # Set up results:
        io_data['inverse'] = inverse
        io_data['risk_structure'] = risk_structure
        io_data['io_approach'] = io_approach
        self.io_data = io_data
        self.indirect_impact = indirect_impact
        self.indirect_aai_agg = self.indirect_impact.mean(axis=0)
    
    def calc_total_impact(self):
        """Calculates the total impact on each country/sector-pairs. Total impact
        for any given country/sector-pair is the sum of direct and indirect
        impact of the respective pair. The resulting 2-dim array is andded as
        an attribute to the SupplyChain instance. It is analogous to the attributes
        'direct_impact' and 'indirect_impact'. 
        
        Additionally, total average annual impact is computed and added as attribute
        'total_aai_agg' to the SupplyChain instance (analogous to 'direct_aai_agg' 
        and 'indirect_aai_agg'. """
        self.total_impact = self.indirect_impact + self.direct_impact
        self.total_aai_agg = self.total_impact.mean(axis=0)

    def _aggregate_labels(self):
        """Method to aggregate the countries and sectors attributes of a
        SupplyChain instance. "Aggregating" means that all subsectors are aggregated
        into their corresponding main sectors for which the SupplChain class offers
        default exposures: agriculture, forestry_fishing, mining_quarrying, manufacturing,
        utilities supply and services. The countries attribute is shortened
        accordingly, so that cross-indexing from countries to sectors will still
        yield matching results.

        Parameters:
            None
        Returns:
            None.
        A new attribute "aggregated_mriot" is added to the target SupplyChain instance:
        a dictionary with four key:values-pairs:
            countries (np.array): aggregated country labels.
            countries_iso3 (np.array): aggregated country iso3 labels
            sectors (np.array): aggregated sectors labels.
            aggregation_info (dict): stores information on which subsectors
                (values; list of strings) were aggregated into which main
                sectors (keys; str).

        """
        unique_main_sectors, ind = np.unique(self.main_sectors, return_index=True)
        # To keep original order of values:
        unique_main_sectors = unique_main_sectors[np.argsort(ind)]
        # Construct aggregation_info dict:
        agg_info = {}
        main_sectors_temp = self.main_sectors[0:self.n_sectors]
        sub_sectors_temp = self.sectors[0:self.n_sectors]
        for msect in unique_main_sectors:
            agg_info[str(msect)] = sub_sectors_temp[main_sectors_temp == msect]
        ### !!! For now, dict is constructed in alphabetical order of main sectors.
        ### This should not matter. In case it turns out it does, get back and adapt
        ### so that original main sector order is preserved!
        aggregated_mriot = {'sectors': np.tile(unique_main_sectors, self.n_countries)}
        aggregated_mriot['aggregation_info'] = agg_info
        unique_countries, ind = np.unique(self.countries, return_index=True)
        unique_countries = unique_countries[np.argsort(ind)]
        aggregated_mriot['countries'] = np.repeat(unique_countries, len(unique_main_sectors))
        unique_countries_iso3, ind = np.unique(self.countries_iso3, return_index=True)
        unique_countries_iso3 = unique_countries_iso3[np.argsort(ind)]
        aggregated_mriot['countries_iso3'] = np.repeat(unique_countries_iso3,\
                        len(unique_main_sectors))

        self.aggregated_mriot = aggregated_mriot

    def _leontief_calc(self, coefficients, direct_intensity, inverse, risk_structure, year_i):
        """Only used by calc_indirect_impact to declutter fuction; calculates
        the risk_structure based on the Leontief IO approach (see documentation
        for more details). Compare with equivalent internal functions (below) to calculate
        risk structure based on Ghosh or EEIO approach, respectively.
        """
        demand = self.total_prod - np.nansum(self.mriot_data, axis=1)
        degr_demand = direct_intensity*demand
        for idx, row in enumerate(inverse):
            risk_structure[:, idx, year_i] = row * degr_demand
        return risk_structure

    def _ghosh_calc(self, coefficients, direct_intensity, inverse, risk_structure, year_i):
        value_added = self.total_prod - np.nansum(self.mriot_data, axis=0)
        degr_value_added = np.maximum(direct_intensity*value_added,\
                                      np.zeros_like(value_added)) # Force non-negative value-added.
        for idx, col in enumerate(inverse.T):
           # Here, we iterate across columns of inverse (hence transpose used).
            risk_structure[:, idx, year_i] = degr_value_added * col
        return risk_structure

    def _eeio_calc(self, coefficients, direct_intensity, inverse, risk_structure, year_i):
        for idx, col in enumerate(inverse.T): # Also across columns of inverse (see above).
            risk_structure[:, idx, year_i] = (direct_intensity * col) * self.total_prod[idx]
        return risk_structure
    #%%
    @staticmethod
    def _create_agri_expo(file_names):
        agrar = SpamAgrar()
        agrar.init_spam_agrar()
        agrar.write_hdf5(os.path.join(SUP_DATA_DIR, \
                                 file_names['agriculture']))
    @staticmethod
    def _create_manu_expo(file_names):
        manu = ManufacturingExp()
        manu.init_manu_exp(assign_centroids=True)
        manu.tag = Tag()
        manu.tag.description = "Default manufacturing exposure for the supplychain "\
            "workflow. Exposure is based on: Greenhouse gas & Air pollution Interactions "\
            "and Synergies (GAINS) model, International Institute for Applied "\
            "Systems Analysis (IIASA), 2015, 'ECLIPSE V5a global emission fields'"
        manu.tag.file_name = "See class 'ManufacturingExp' definition for more details on "\
            "sorce file."
        manu.write_hdf5(os.path.join(SUP_DATA_DIR, \
                                file_names['manufacturing']))
    @staticmethod
    def _create_mining_expo(file_names):
        mining = MiningExp()
        mining.init_mining_exp(assign_centroids=True)
        mining.tag = Tag()
        mining.tag.description = "Default mining exposure for the supplychain "\
            "workflow. Approximating global exposure of the mining and quarrying "\
            "industry sector (approximated using global distribution of mines "\
            "according to the US Geological Survey)."
        mining.tag.file_name = "See class 'MiningExp' definition for more details on "\
            "sorce file."
        mining.write_hdf5(os.path.join(SUP_DATA_DIR, \
                             file_names['mining_quarrying']))
    @staticmethod
    def _create_forest_expo(file_names):
        forest = ForestExp()
        forest.init_forest_exp(assign_centroids=True)
        forest.tag = Tag()
        forest.tag.description = "Default mining exposure for the supplychain "\
            "workflow. Approximating global forest exposure based on: "\
            "SA and Université Catholique de Louvain, 2015, "\
            "Land Cover Map 2015, Version 2.0."
        forest.tag.file_name = "See class 'ForestExp' definition for more details on "\
            "sorce file."
        forest.write_hdf5(os.path.join(SUP_DATA_DIR,\
                             file_names['forestry_fishing']))
    @staticmethod
    def _create_utilities_expo(file_names):
        util = UtilitiesExp()
        util.init_utilities_exp(assign_centroids=True)
        util.tag = Tag()
        util.tag.description = "Default utilities exposure for the supplychain "\
            "workflow. Approximating global exposure of the utilities sector "\
            "using global distribution of power plants)."
        util.tag.file_name = "See class 'UtilitiesExp' definition for more details on "\
            "sorce file."
        util.write_hdf5(os.path.join(SUP_DATA_DIR, \
                             file_names['utilities']))
        
    @staticmethod
    def _create_services_expo(file_names):
        # Get list of all countries (ISO3) represented in BlackMarble:
        invalid_countries = ['PSE','SSD','ESH','SXM','LIE','GIB','VAT','ABW','SPM',\
                             'PCN','MHL','LCA','DMA','UMI','MSR','KNA','BLM','AIA',\
                                 'BMU','HMD','SHN','STP','GGY','ALA','IOT','NFK',\
                                     'COK','WLF','TUV','MDV','NRU','ASM','MNP','BHR','MAC']
        shp_file = shapereader.natural_earth(resolution='10m', category='cultural',
                                     name='admin_0_countries')
        shp_file = shapereader.Reader(shp_file)
        list_records = list(shp_file.records())
        all_countries = []
        failed_countries = [] #To catch countries where LitPop fails
        file_exists = [] #To catch countries where file exists already. 
                         #We don't compute these again.
        for entry in list_records:
            all_countries.append((entry.attributes['ISO_A3'].title()).upper())
            if all_countries[-1] == '-99':
                all_countries.pop()
        for i, ctry in enumerate(all_countries):
            if (not os.path.isfile(os.path.join(SUP_DATA_DIR, ctry + '_services_XXX'))) \
                and (ctry not in invalid_countries): 
                lit = LitPop()
                try:
                    lit.set_country(ctry, res_km=20)
                    lit.write_hdf5(os.path.join(SUP_DATA_DIR, \
                        ctry + '_services_XXX')) 
                except (ValueError, UnboundLocalError, OverflowError):
                         # Note: LitPop fails for some of the countries. Since 
                         # these are all niche countries, we for now just skip
                         # them as they are insignificant for our purpose.
                    failed_countries.append(ctry)
            else:
                file_exists.append(ctry)
        if file_exists: #Checks whether list "file_exists" ist empty or not.
            files_string = str(file_exists)
            LOGGER.info("For the following countries no new SERVICES exposures were " \
                "created as they seem to exist already (remove them from the supplychain "\
                "data folder if you want to create new ones): " + files_string)  
        if failed_countries: #Checks whether list "failed_countries" ist empty or not.
            files_string = str(failed_countries)
            LOGGER.info("We couldn't create exposures for the following countries; " \
                        "these should all be insignificant countries (microstates) " \
                        "that do not affect the overall analysis: " + files_string)
        
            
            
    # def _create_services_expo():
    #     if not glob.glob(os.path.join(SYSTEM_DIR, '*LitPop*')):
    #         LOGGER.error('To load default services exposures, the LitPop ' \
    #                      'Exposures data needs to be downlaoded. Please refer ' \
    #                      'to documentation for details on requirements.')
    #         raise IOError('To load default services exposures, the LitPop ' \
    #                      'Exposures data needs to be downlaoded. Please refer ' \
    #                      'to documentation for details on requirements.')
    #     for fname in tqdm(glob.glob(os.path.join(SYSTEM_DIR, '*LitPop*', '*.csv'))):
    #         exp = pd.read_csv(fname)
    #         country_id = int(exp['region_id'][0])
    #         # Add leading zeros to region_id for use in iso3166 package:
    #         country_id = '{0:0>3}'.format(country_id)
    #         country_iso3 = ctry_ids[str(country_id)][2] # Get iso3 code
    #         exp['region_id'] = country_iso3
    #         exp = Exposures(exp)
    #         exp.ref_year = 2015
    #         exp.value_unit = 'USD'
    #         exp.tag = Tag()
    #         exp.tag.description = ("LitPop exposure for country '" + country_iso3 + \
    #                                "' as proxy for services sector.")
    #         exp.check()
    #         exp.write_hdf5(os.path.join(SUP_DATA_DIR, \
    #                         country_iso3 + '_services_XXX'))
