#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:53:44 2018

@author: eberenzs

Creates Agriculture Exposure Set (Entity) based on SPAM
(Global Spatially-Disaggregated Crop Production Statistics Data for 2005
Version 3.2 )
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DHXBJX

Todos:
    - integration tests
    - unit tests
"""

import os
import logging
import zipfile
import pandas as pd
import numpy as np
from climada import SYSTEM_DIR
from climada.entity.exposures.base import Exposures
from climada.util.files_handler import download_file

logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

FILENAME_SPAM = 'spam2005V3r2_global'
FILENAME_CELL5M = 'cell5m_allockey_xy.csv'
FILENAME_PERMALINKS = 'spam2005V3r2_download_permalinks.csv'
BUFFER_VAL = -340282306073709652508363335590014353408
# Hard coded value which is used for NANs in original data

class SpamAgrar(Exposures):
    """Defines exposures from
    """

    def __init__(self):
        """ Empty initializer. """
        Exposures.__init__(self)

    def clear(self):
        """ Appending the base class clear attribute to also delete attributes
            which are only used here.
        """
        Exposures.clear(self)
        try:
            del self.country_data
        except AttributeError:
            pass

    def init_spam_agrar(self, **parameters):
        """ initiates agriculture exposure from SPAM data:

            https://dataverse.harvard.edu/
            dataset.xhtml?persistentId=doi:10.7910/DVN/DHXBJX

        Optional parameters:
            data_path (str): absolute path where files are stored.
                Default: SYSTEM_DIR

            country (str): Three letter country code of country to be cut out.
                No default (global)
            name_adm1 (str): Name of admin1 (e.g. Federal State) to be cut out.
                No default
            name_adm2 (str): Name of admin2 to be cut out.
                No default

            region_id (int): region_id to be assigned to exposure. Default=1

            spam_variable (str): select one agricultural variable:
                'A'		physical area
                'H'		harvested area
                'P'		production
                'Y'		yield
                'V_agg'	value of production, aggregated to all crops,
                                 food and non-food (default)
                 Warning: for A, H, P and Y, currently all crops are summed up

            spam_technology (str): select one agricultural technology type:
                'TA'	   all technologies together, ie complete crop (default)
                'TI'   irrigated portion of crop
                'TH'   rainfed high inputs portion of crop
                'TL'   rainfed low inputs portion of crop
                'TS'   rainfed subsistence portion of crop
                'TR'   rainfed portion of crop (= TA - TI, or TH + TL + TS)
                ! different impact_ids are assigned to each technology (1-6)

            result_mode (int): Determines how many aditional data are saved:
                1: only basics (lat, lon, total value)
                2: like 1 + name of country
                3: like 2 + name of admin1


        Returns:
        """
        data_p = parameters.get('data_path', SYSTEM_DIR)
        spam_t = parameters.get('spam_technology', 'TA')
        spam_v = parameters.get('spam_variable', 'V_agg')
        adm0 = parameters.get('country')
        adm1 = parameters.get('name_adm1')
        adm2 = parameters.get('name_adm2')
        reg_id = parameters.get('region_id', 1)
        result_m = parameters.get('result_mode', 2)
        # fname_short = FILENAME_SPAM+'_'+ spam_var  + '_' + spam_tech + '.csv'

        # read data from CSV:
        data = self._read_spam_file(data_path=data_p, spam_technology=spam_t, \
                                  spam_variable=spam_v, result_mode=1)

        # extract country or admin level (if provided)
        data, region = self._spam_set_country(data, country_adm0=adm0, \
                                       name_adm1=adm1, name_adm2=adm2)

        # sort by alloc_key to make extraction of lat / lon easier:
        data.sort_values(by=['alloc_key'])

        lat, lon = self._spam_get_coordinates(data.loc[:, 'alloc_key'], \
                                             data_path=data_p)
        if result_m == 2 or result_m == 4:
            self.country = data.loc[:, 'iso3'].values
            self.name_adm1 = data.loc[:, 'name_adm1'].values

        if spam_v == 'V_agg': # total only (column 7)
            i_1 = 7
            i_2 = 8
        else:
            i_1 = 7 # get sum over all crops (columns 7 to 48)
            i_2 = 49
        self.value = data.iloc[:, i_1:i_2].sum(axis=1)
        self.coord = np.empty((self.value.size, 2))
        self.coord[:, 0] = lat.values
        self.coord[:, 1] = lon.values
        LOGGER.info('Lat. range: {:+.3f} to {:+.3f}.'.format(\
                    np.min(self.coord[:, 0]), np.max(self.coord[:, 0])))
        LOGGER.info('Lon. range: {:+.3f} to {:+.3f}.'.format(\
                    np.min(self.coord[:, 1]), np.max(self.coord[:, 1])))
        self.id = np.arange(1, self.value.size+1)
        self.region_id = np.ones(self.value.size, int)
        if reg_id > 1:
            self.region_id = reg_id*self.region_id

        # assign different damage function ID per technology type:
        if spam_t == 'TA':
            self.impact_id = np.ones(self.value.size, int)
            self.comment = 'TA: all technologies together, ie complete crop'
        elif spam_t == 'TI':
            self.impact_id = np.ones(self.value.size, int)+1
            self.comment = 'TI: irrigated portion of crop'
        elif spam_t == 'TH':
            self.impact_id = np.ones(self.value.size, int)+2
            self.comment = 'TH: rainfed high inputs portion of crop'
        elif spam_t == 'TL':
            self.impact_id = np.ones(self.value.size, int)+3
            self.comment = 'TL: rainfed low inputs portion of crop'
        elif spam_t == 'TS':
            self.impact_id = np.ones(self.value.size, int)+4
            self.comment = 'TS: rainfed subsistence portion of crop'
        elif spam_t == 'TR':
            self.impact_id = np.ones(self.value.size, int)+5
            self.comment = 'TI: rainfed portion of crop (= TA - TI)'
        else:
            self.impact_id = np.ones(self.value.size, int)

        self.ref_year = 2005
        self.tag.description = ("SPAM agrar exposure for technology "\
            + spam_t + " and variable " + spam_v)
        self.tag.file_name = (FILENAME_SPAM+'_'+ spam_v\
                              + '_' + spam_t + '.csv')
#        self.tag.shape = cntry_info[2]
        #self.tag.country = cntry_info[1]
        if spam_v == 'A' or spam_v == 'H':
            self.value_unit = 'Ha'
        elif spam_v == 'Y':
            self.value_unit = 'kg/Ha'
        elif spam_v == 'P':
            self.value_unit = 'mt'
        else:
            self.value_unit = 'USD'


        LOGGER.info('Total {} {} {}: {:.1f} {}.'.format(\
                    spam_v, spam_t, region, self.value.sum(), self.value_unit))



    def _read_spam_file(self, **parameters):
        """ Reads data from SPAM CSV file and cuts out the data for the
            according country, admin1, or admin2 (if requested).

        Optional parameters:
            data_path (str): absolute path where files are stored. Default: SYSTEM_DIR

            spam_variable (str): select one agricultural variable:
                'A'		physical area
                'H'		harvested area
                'P'		production
                'Y'		yield
                'V_agg'	value of production, aggregated to all crops,
                                 food and non-food (default)

            spam_technology (str): select one agricultural technology type:
                'TA'	   all technologies together, ie complete crop (default)
                'TI'   irrigated portion of crop
                'TH'   rainfed high inputs portion of crop
                'TL'   rainfed low inputs portion of crop
                'TS'   rainfed subsistence portion of crop
                'TR'   rainfed portion of crop (= TA - TI, or TH + TL + TS)

            result_mode (int): Determines whether latitude and longitude are
                delievered along with gpw data (0) or only gpw_data is returned (1)
                Default = 1.

        Returns:
            tile_temp (pandas SparseArray): GPW data
            lon (list): list with longitudinal infomation on the GPW data. Same
                dimensionality as tile_temp (only returned if result_mode=1)
            lat (list): list with latitudinal infomation on the GPW data. Same
                dimensionality as tile_temp (only returned if result_mode=1)
        """
        data_path = parameters.get('data_path', SYSTEM_DIR)
        spam_tech = parameters.get('spam_technology', 'TA')
        spam_var = parameters.get('spam_variable', 'V_agg')
        # result_mode = parameters.get('result_mode',1)
        fname_short = FILENAME_SPAM+'_'+ spam_var  + '_' + spam_tech + '.csv'

        try:
            fname = os.path.join(data_path, fname_short)
            if not os.path.isfile(fname):
                try:
                    self._spam_download_csv(data_path=data_path,\
                                            spam_variable=spam_var)
                except:
                    raise FileExistsError('The file ' + str(fname)\
                                + ' could not '\
                                + 'be found. Please download the file '\
                                + 'first or choose a different folder. '\
                                + 'The data can be downloaded from '\
                                + 'https://dataverse.harvard.edu/'\
                                + 'dataset.xhtml?persistentId=doi:'\
                                + '10.7910/DVN/DHXBJX')
            LOGGER.debug('Importing ' + str(fname_short))

            data = pd.read_csv(fname, sep=',', index_col=None, header=0, \
                             encoding='ISO-8859-1')

        except:
            LOGGER.error('Importing the SPAM agriculturer file failed. '\
                         + 'Operation aborted.')
            raise
        # remove data points with zero crop production: (works only for TA)
        # data = data[data.vp_crop_a != 0]

        return data

    def _spam_get_coordinates(self, alloc_key_array, data_path=SYSTEM_DIR):
        """mapping from cell5m to lat/lon:"""

        # load concordance_data:

        try:
            fname = os.path.join(data_path, FILENAME_CELL5M)

            if not os.path.isfile(fname):
                try:
                    self._spam_download_csv(data_path=data_path,\
                                            spam_variable='cell5m')
                except:
                    raise FileExistsError('The file ' + str(fname)\
                                + ' could not '\
                                + 'be found. Please download the file '\
                                + 'first or choose a different folder. '\
                                + 'The data can be downloaded from '\
                                + 'https://dataverse.harvard.edu/'\
                                + 'dataset.xhtml?persistentId=doi:'\
                                + '10.7910/DVN/DHXBJX')
            # LOGGER.debug('Inporting ' + str(fname))

            concordance_data = pd.read_csv(fname, sep=',', index_col=None, \
                                           header=0, encoding='ISO-8859-1')

            concordance_data = concordance_data\
                [concordance_data['alloc_key'].isin(alloc_key_array)]

            concordance_data.sort_values(by=['alloc_key'])

            lat = concordance_data.loc[:, 'y']
            lon = concordance_data.loc[:, 'x']

        except:
            LOGGER.error('Importing the SPAM cell5m mapping file failed. '\
                         + 'Operation aborted.')
            raise
        return lat, lon

    @staticmethod
    def _spam_set_country(data, **parameters):
        """
        restrict data to given country (admin0) or admin1/ admin2.

        Input:
            data: dataframe from _read_spam_file()

        Optional parameters:

            country_adm0 (str): Three letter country code of country to be cut out.
                No default (global)
            name_adm1 (str): Name of admin1 (e.g. Federal State) to be cut out.
                No default
            name_adm2 (str): Name of admin2 to be cut out.
                No default
        """
        adm0 = parameters.get('country_adm0')
        adm1 = parameters.get('name_adm1')
        adm2 = parameters.get('name_adm2')
        signifier = ''
        if not adm0 is None:
            if data[data.iso3 == adm0].empty:
                data = data[data.name_cntr == adm0]
            else:
                data = data[data.iso3 == adm0]
            if data.empty:
                LOGGER.error('Country not found in data: ' + str(adm0))
            else:
                signifier = signifier + adm0
        if not adm1 is None:
            data = data[data.name_adm1 == adm1]
            if data.empty:
                LOGGER.error('Admin1 not found in data: ' + str(adm1))
            else:
                signifier = signifier + ' ' + adm1
        if not adm2 is None:
            data = data[data.name_adm2 == adm2]
            if data.empty:
                LOGGER.error('Admin2 not found in data: ' + str(adm2))
            else:
                signifier = signifier + ' ' + adm2

        if signifier == '':
            signifier = 'global'

        return data, signifier

    @staticmethod
    def _spam_download_csv(data_path=SYSTEM_DIR, spam_variable='V_agg'):
        """
        Download and unzip CSV files from https://dataverse.harvard.edu/file

        Inputs:
            data_path (str): absolute path where files are to be stored.
                                Default: SYSTEM_DIR

            spam_variable (str): select one variable:
                'A'		physical area
                'H'		harvested area
                'P'		production
                'Y'		yield
                'V_agg'	value of production, aggregated to all crops,
                                 food and non-food (default)
                'cell5m' concordance_data to retrieve lat / lon
        """
        try:
            fname = os.path.join(data_path, FILENAME_PERMALINKS)
            if not os.path.isfile(fname):
                url1 = 'https://dataverse.harvard.edu/api/access/datafile/:'\
                        + 'persistentId?persistentId=doi:10.7910/DVN/DHXBJX/'
                permalinks = pd.DataFrame(columns=['A', 'H', \
                        'P', 'Y', 'V_agg', 'cell5m'])
                permalinks.loc[0, 'A'] = url1 + 'FS1JO8'
                permalinks.loc[0, 'H'] = url1 + 'M727TX'
                permalinks.loc[0, 'P'] = url1 + 'HPUWVA'
                permalinks.loc[0, 'Y'] = url1 + 'RTGSQA'
                permalinks.loc[0, 'V_agg'] = url1 + 'UG0N7K'
                permalinks.loc[0, 'cell5m'] = url1 + 'H2D3LI'
            else:
                permalinks = pd.read_csv(fname, sep=',', index_col=None, \
                                         header=0)
                LOGGER.debug('Importing ' + str(fname))

            # go to data directory:
            os.chdir(data_path)
            path_dwn = download_file(permalinks.loc[0, spam_variable])

            LOGGER.debug('Download complete. Unzipping ' + str(path_dwn))
            zip_ref = zipfile.ZipFile(path_dwn, 'r')
            zip_ref.extractall(data_path)
            zip_ref.close()
            os.remove(path_dwn)
        except:
            LOGGER.error('Downloading SPAM data failed. '\
                             + 'Operation aborted.')
            raise
            