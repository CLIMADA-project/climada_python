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

Defines MiningExp class, subclass of Exposures.

"""
import logging
import pandas as pd
import zipfile
import os
from climada.entity.exposures.base import Exposures
from climada.entity.tag import Tag
from climada.util.files_handler import download_file
from climada.util.coordinates import get_country_code
from climada.util.constants import SYSTEM_DIR

LOGGER = logging.getLogger(__name__)

US_MINES_FILE = 'mineplant.csv'
" Location of file for US mines "

ROW_MINES_FILE = 'minfac.csv'
" Location of file for Rest of World mines "

US_MINES_url = 'https://mrdata.usgs.gov/mineplant/mineplant-csv.zip'
" Url to download file for US mines "

ROW_MINES_url = 'https://mrdata.usgs.gov/mineral-operations/minfac-csv.zip'
" Url to download file for Rest of World mines "

minedata_dict = {'US': [US_MINES_FILE, US_MINES_url, [6,7]], 
                 'ROW':[ROW_MINES_FILE, ROW_MINES_url, [10,11]]}

# TODO: review this class in light of the new strucutre of SupplyChain, i.e.
# no more global data needed

def save_and_unzip(url, data_folder):
    os.chdir(data_folder)
    try:
        path_dwn = download_file(url)
        LOGGER.debug('Download complete. Unzipping %s', str(path_dwn))
    except:
        LOGGER.error('Downloading mining data failed.')
        raise
            
    zip_ref = zipfile.ZipFile(path_dwn, 'r')
    zip_ref.extractall(data_folder)
    zip_ref.close()
    os.remove(path_dwn)

class MiningExp(Exposures):
    """Approximates global manufacturing exposures.
    Source of underlying data: 
        
    U.S. Geological Survey (2005b). Active Mines and Mineral Processing Plants 
    in the United States in 2003 [Data file]: Geological Survey (U.S.), 
    Reston, Virginia. https://mrdata.usgs.gov/mineplant/
    and
    U.S. Geological Survey (2010). Mineral operations outside the United States 
    [Data file]: Geological Survey (U.S.), Reston, Virginia. 
    https://mrdata.usgs.gov/mineral-operations/
    
    """

    def init_mining_exp(self):              
        df_files = []
        for key in minedata_dict.keys():
            file = minedata_dict[key][0]
            
            if file not in os.listdir(SYSTEM_DIR):
                url = minedata_dict[key][1]
                save_and_unzip(url, SYSTEM_DIR)
                
                mining_file = pd.read_csv(SYSTEM_DIR+'\\'+file, 
                                          usecols=minedata_dict[key][2])
                mining_file.columns = ['latitude', 'longitude']
                df_files.append(mining_file)
            else:
                mining_file = pd.read_csv(SYSTEM_DIR+'\\'+file, 
                                          usecols=minedata_dict[key][2])
                mining_file.columns = ['latitude', 'longitude']
                df_files.append(mining_file)
        
        df_files = pd.concat(df_files)
        self['latitude'] = df_files.latitude.values 
        self['longitude'] = df_files.longitude.values
        self.set_geometry_points()
        self['region_id'] = get_country_code(self.latitude.values, 
                                             self.longitude.values)
        self['value'] = 1.
        self.ref_year = 2005
        self.tag = Tag()
        self.tag.description = 'Global exposures of the mining '\
                               'industry sector'        
        self.check()