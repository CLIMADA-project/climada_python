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

Defines AgricultureExp class, a subclass of Exposures.

"""

import os
import logging
import zipfile
import pandas as pd
import numpy as np
from iso3166 import countries as iso_cntry

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, INDICATOR_IF
from climada.util.files_handler import download_file
from climada.util.constants import SYSTEM_DIR

logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'CP'
"""Default hazard type used in impact functions id."""

FILENAME_SPAM = 'spam2005V3r2_global'
"""Perfix of all SPAM related datasets names"""

FILENAME_CELL5M = 'cell5m_allockey_xy.csv'
"""Concordance table of the primary key in SPAM data and its centroid coordinates"""

#FILENAME_PERMALINKS = 'spam2005V3r2_download_permalinks.csv'

BUFFER_VAL = -340282306073709652508363335590014353408
"""Hard coded value which is used for NANs in original data"""

# TODO: review this class in light of the new strucutre of SupplyChain, i.e.
# no more global data needed

class AgricultureExp(Exposures):
    """Defines agriculture exposures from SPAM (Spatial Production Allocation Model)
    retrieved by the Global Spatially-Disaggregated Crop Production Statistics Data 
    for 2005 Version 3.2:
    
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DHXBJX

    """

    @property
    def _constructor(self):
        return AgricultureExp

    def init_agri_exp(self, **parameters):
        """Optional parameters:
        data_path (str): absolute path where files are stored.
            Default: SYSTEM_DIR
        country (str): country code (ISO 3).
            No default (global)
        name_adm1 (str): Name of admin1 (e.g. Federal State).
            No default
        name_adm2 (str): Name of admin2.
            No default
        spam_variable (str): select one agricultural variable:
            'A'	physical area
            'H'	harvested area
            'P'	production
            'Y'	yield
            'V_agg'	value of production, aggregated to all crops, food and 
                non-food (default)
            spam_technology (str): select one agricultural technology type:
                'TA' all technologies together, i.e. complete crop (default)
                'TI' irrigated portion of crop
                'TH' rainfed high inputs portion of crop
                'TL' rainfed low inputs portion of crop
                'TS' rainfed subsistence portion of crop
                'TR' rainfed portion of crop (= TA - TI, or TH + TL + TS)

            save_name_adm1 (bool): wether saving name of admin1

            haz_type (str): hazard type abbreviation, e.g.
                'DR' for Drought or
                'CP' for CropPotential (is this even an hazard?)

        Returns:
        """
        data_p = parameters.get('data_path', SYSTEM_DIR)
        spam_t = parameters.get('spam_technology', 'TA')
        spam_v = parameters.get('spam_variable', 'V_agg')
        adm0 = parameters.get('country')
        adm1 = parameters.get('name_adm1')
        adm2 = parameters.get('name_adm2')
        save_adm1 = parameters.get('save_name_adm1', False)
        haz_type = parameters.get('haz_type', DEF_HAZ_TYPE)
        
        # read exposure files
        data = self._read_spam_file(data_path=data_p, spam_technology=spam_t,
                                    spam_variable=spam_v, result_mode=1)
        # clip per administrative area of interest
        data, region = self._spam_set_country(data, adm0=adm0, adm1=adm1, adm2=adm2)

        # sort by alloc_key to make extraction of lat / lon easier:
        data = data.sort_values(by=['alloc_key'])
        lat, lon = self._spam_get_coordinates(data.loc[:, 'alloc_key'],
                                              data_path=data_p)
        if save_adm1:
            self.name_adm1 = data.loc[:, 'name_adm1'].values

        if spam_v == 'V_agg':
            # only take total
            i_1 = 7
            i_2 = 8
        else:
            # get sum over all crops
            i_1 = 7
            i_2 = 49
        self['value'] = data.iloc[:, i_1:i_2].sum(axis=1).values
        self['latitude'] = lat.values
        self['longitude'] = lon.values
        LOGGER.info('Lat. range: {:+.3f} to {:+.3f}.'.format(
            np.min(self.latitude), np.max(self.latitude)))
        LOGGER.info('Lon. range: {:+.3f} to {:+.3f}.'.format(
            np.min(self.longitude), np.max(self.longitude)))

        # set region_id (numeric ISO3):
        country_id = data.loc[:, 'iso3']
        if country_id.unique().size == 1:
            region_id = np.ones(self.value.size, int)\
                * int(iso_cntry.get(country_id.iloc[0]).numeric)
        else:
            region_id = np.zeros(self.value.size, int)
            for i in range(0, self.value.size):
                region_id[i] = int(iso_cntry.get(country_id.iloc[i]).numeric)
        self['region_id'] = region_id
        self.ref_year = 2005
        self.tag = Tag()
        self.tag.description = ("SPAM agrar exposure for variable "
                                + spam_v + " and technology " + spam_t)

        # if impact id variation iiv = 1, assign different damage function ID
        # per technology type.
        self._set_if(spam_t, haz_type)

        self.tag.file_name = (FILENAME_SPAM + '_' + spam_v + '_' + spam_t + '.csv')

        if spam_v in ('A', 'H'):
            self.value_unit = 'Ha'
        elif spam_v == 'Y':
            self.value_unit = 'kg/Ha'
        elif spam_v == 'P':
            self.value_unit = 'mt'
        else:
            self.value_unit = 'USD'

        LOGGER.info('Total {} {} {}: {:.1f} {}.'.format(
            spam_v, spam_t, region, self.value.sum(), self.value_unit))
        self.check()

    def _set_if(self, spam_t, haz_type):
        """Set impact function id depending on technology."""
        # hazard type drought is default.
        iiv = 0
        
        # FIXME: what is this? it is setting all impact functions as series of ones?
        
        if spam_t == 'TA':
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int)
            self.tag.description = self.tag.description + '. '\
            + 'all technologies together, ie complete crop'
        elif spam_t == 'TI':
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int) + 1 * iiv
            self.tag.description = self.tag.description + '. '\
            + 'irrigated portion of crop'
        elif spam_t == 'TH':
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int) + 2 * iiv
            self.tag.description = self.tag.description + '. '\
            + 'rainfed high inputs portion of crop'
        elif spam_t == 'TL':
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int) + 3 * iiv
            self.tag.description = self.tag.description + '. '\
            + 'rainfed low inputs portion of crop'
        elif spam_t == 'TS':
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int) + 4 * iiv
            self.tag.description = self.tag.description + '. '\
            + 'rainfed subsistence portion of crop'
        elif spam_t == 'TR':
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int) + 5 * iiv
            self.tag.description = self.tag.description + '. '\
            + 'rainfed portion of crop (= TA - TI)'
        else:
            self[INDICATOR_IF + haz_type] = np.ones(self.value.size, int)
        self.set_geometry_points()

    def _read_spam_file(self, data_path, spam_technology, spam_variable):
        """Reads data from SPAM CSV file.

        Returns:
            data: pd.DataFrame with all data for selected country / region
        """

        fname_short = FILENAME_SPAM + '_' + spam_variable + '_' + \
                        + spam_technology + '.csv'

        try:
            fname = os.path.join(data_path, fname_short)
            if not os.path.isfile(fname):
                try:
                    self._spam_download_csv(data_path=data_path,
                                            spam_variable=spam_variable)
                # FIXME: why this exception if there is already one inside
                # _spam_download_csv
                except:
                    raise FileExistsError(
                        'The file {} could not be found'.format(fname))
                    
            LOGGER.debug('Importing %s', str(fname_short))
            data = pd.read_csv(fname, sep=',', index_col=None, header=0, 
                               encoding='ISO-8859-1')
            
        # FIXME: what is this exception for? fname should be there and if not
        # it is downloaded
        except:
            LOGGER.error('Could not import the SPAM agriculturer file failed. '
                         'Operation aborted.')
            raise

        return data

    def _spam_get_coordinates(self, alloc_key_array, data_path=SYSTEM_DIR):
        """mapping from cell5m to lat/lon:"""

        try:
            fname = os.path.join(data_path, FILENAME_CELL5M)

            if not os.path.isfile(fname):
                try:
                    self._spam_download_csv(data_path=data_path,
                                            spam_variable='cell5m')
                    
                # FIXME: there already is an exception in _spam_download_csv
                except:
                    raise FileExistsError('The file ' + str(fname) + ' could not '
                                          + 'be found. Please download the file '
                                          + 'first or choose a different folder. '
                                          + 'The data can be downloaded from '
                                          + 'https://dataverse.harvard.edu/'
                                          + 'dataset.xhtml?persistentId=doi:'
                                          + '10.7910/DVN/DHXBJX')
            # LOGGER.debug('Inporting %s', str(fname))

            concordance_data = pd.read_csv(fname, sep=',', index_col=None,
                                           header=0, encoding='ISO-8859-1')

            concordance_data = concordance_data[
                concordance_data['alloc_key'].isin(alloc_key_array)]

            concordance_data = concordance_data.sort_values(by=['alloc_key'])

            lat = concordance_data.loc[:, 'y']
            lon = concordance_data.loc[:, 'x']

        # FIXME: why so many exceptions
        except:
            LOGGER.error('Importing the SPAM cell5m mapping file failed. '
                         'Operation aborted.')
            raise
        return lat, lon

    @staticmethod
    def _spam_set_country(data, adm0=None, adm1=None, adm2=None):
        """
        restrict data to given country (admin0) or admin1/admin2.

        Input:
            data: dataframe from _read_spam_file()
        """
 
        signifier = ''
        if adm0 is not None:
            if data[data.iso3 == adm0].empty:
                if data[data.name_cntr == adm0].empty:
                    LOGGER.warning('Country name not found in data: %s',
                                   str(adm0) + '. Try passing the ISO3-code instead.')
                else:
                    data = data[data.name_cntr == adm0]
                    signifier = signifier + adm0
            else:
                data = data[data.iso3 == adm0]
                signifier = signifier + adm0

        if adm1 is not None:
            if data[data.name_adm1 == adm1].empty:
                LOGGER.warning('Admin1 not found in data: %s', str(adm1))
            else:
                data = data[data.name_adm1 == adm1]
                signifier = signifier + ' ' + adm1
        if adm2 is not None:
            if data[data.name_adm2 == adm2].empty:
                LOGGER.warning('Admin2 not found in data: %s', str(adm2))
            else:
                data = data[data.name_adm2 == adm2]
                signifier = signifier + ' ' + adm2

        if signifier == '':
            signifier = 'global'

        return data, signifier

    @staticmethod
    def _spam_download_csv(data_path=SYSTEM_DIR, spam_variable='V_agg'):
        """
        Download and unzip CSV files

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
        
        var_download_code = {'A': 'FS1JO8', 'H': 'M727TX', 'P': 'HPUWVA', 
                             'Y': 'RTGSQA', 'V_agg': 'UG0N7K', 'cell5m': 'H2D3LI'}

        orig_url = 'https://dataverse.harvard.edu/api/access/datafile/:'\
                    + 'persistentId?persistentId=doi:10.7910/DVN/DHXBJX/'

        download_url = orig_url + var_download_code[spam_variable]   
        os.chdir(data_path)
        
        # download data
        try:
            path_dwn = download_file(download_url)
            LOGGER.debug('Download complete. Unzipping %s', str(path_dwn))
        except:
            LOGGER.error('Downloading SPAM data failed.')
            raise
            
        # unzip data
        zip_ref = zipfile.ZipFile(path_dwn, 'r')
        zip_ref.extractall(data_path)
        zip_ref.close()
        os.remove(path_dwn)    
