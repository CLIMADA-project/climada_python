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
"""


import logging
import os
from os import listdir
from os.path import isfile, join
import math
import numpy as np
import xarray as xr
import pandas as pd
from iso3166 import countries as iso_cntry
from climada.entity.exposures.base import Exposures
from climada.entity.tag import Tag
import climada.util.coordinates as co





logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'CP'
""" Default hazard type used in impact functions id."""

BBOX = np.array([-180, -85, 180, 85]) # [Lon min, lat min, lon max, lat max]
AREA_CELL = (1.853*5)**2*100 #[ha]: 1' = 1.853km;  1km^2 = 100ha


CL_MODEL = ['gfdl-esm2m',
            'hadgem2-es',
            'ipsl-cm5a-lr',
            'miroc5'
            ]

FN_STR_VAR = 'landuse-15crops_annual'

SCENARIO = ['1860soc',
            'histsoc',
            '2005soc',
            'rcp26soc',
            'rcp60soc',
            '2100rcp26soc']

YEARCHUNKS = dict()
#two types of 1860soc (1661-1860 & 1661-2299)
YEARCHUNKS[SCENARIO[0]] = dict()
YEARCHUNKS[SCENARIO[0]] = {'startyear' : 1661, 'endyear': 1860, 'duration': 199}
YEARCHUNKS[SCENARIO[1]] = dict()
YEARCHUNKS[SCENARIO[1]] = {'startyear' : 1861, 'endyear': 2005, 'duration': 145}
YEARCHUNKS[SCENARIO[2]] = dict()
YEARCHUNKS[SCENARIO[2]] = {'startyear' : 2006, 'endyear': 2299, 'duration': 194}
YEARCHUNKS[SCENARIO[3]] = dict()
YEARCHUNKS[SCENARIO[3]] = {'startyear' : 2006, 'endyear': 2099, 'duration': 94}
YEARCHUNKS[SCENARIO[4]] = dict()
YEARCHUNKS[SCENARIO[4]] = {'startyear' : 2006, 'endyear': 2099, 'duration': 94}
YEARCHUNKS[SCENARIO[5]] = dict()
YEARCHUNKS[SCENARIO[5]] = {'startyear' : 2100, 'endyear': 2299, 'duration': 200}

CROP = ['mai',
        'ric',
        'whe',
        'soy'
       ]

CROP_NAME = dict()
CROP_NAME[CROP[0]] = {'input': 'maize', 'fao' : 'Maize', 'print': 'Maize'}
CROP_NAME[CROP[1]] = {'input': 'rice', 'fao' : 'Rice, paddy', 'print': 'Rice'}
CROP_NAME[CROP[2]] = {'input': 'temperate_cereals', 'fao' : 'Wheat', 'print': 'Wheat'}
CROP_NAME[CROP[3]] = {'input': 'oil_crops_soybean', 'fao' : 'Soybeans', 'print': 'Soybeans'}


IRR = ['combined', 'rainfed', 'irrigated']

#HIST_MEAN = default hist_mean: hist_mean+str(crop)
FAO_FILE = "FAOSTAT_data_5-8-2020.csv"
FAO_FILE2 = "FAOSTAT_data_6-3-2020.csv"
YEARS_FAO = np.array([1990, 2010])

class CropyieldIsimip(Exposures):
    """Defines agriculture exposures from ISIMIP input data and FAO crop price data
    """
    @property
    def _constructor(self):
        return CropyieldIsimip

    def set_from_single_run(self, input_dir=None, filename=None, hist_mean=None, bbox=BBOX, \
                            yearrange=None, cl_model=CL_MODEL[0], scenario=SCENARIO[1], \
                            crop=CROP[0], irr=IRR[0], unit='USD', fn_str_var=FN_STR_VAR):

        """Wrapper to fill exposure from nc_dis file from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            filename (string): name of the file to use
            historic mean (array): historic mean yield per centroid
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set
                f.i. (1990, 2010)
            scenario (string): climate change and socio economic scenario
                f.i. 'histsoc' or 'rcp60soc'
            cl_model (string): abbrev. climate model (only when landuse data
            is future projection)
                f.i. 'gfdl-esm2m' etc.
            crop (string): crop type
                f.i. 'mai', 'ric', 'whe', 'soy'
            irr (string): irrigation type
                f.i 'rainfed', 'irrigated' or 'combined'= rainfed+irrigated
            unit (string): unit of the exposure (per year)
                f.i 'USD' or 't'
            fn_str_var (string): FileName STRing depending on VARiable and
                ISIMIP simuation round
        """


        if filename is None:
            yearchunk = YEARCHUNKS[scenario]
            if scenario == 'histsoc':
                string = '%s_%s_%s_%s.nc'
                filename = os.path.join(input_dir, string % (scenario, fn_str_var, \
                                                         str(yearchunk['startyear']), \
                                                         str(yearchunk['endyear'])))
            else:
                string = '%s_%s_%s_%s_%s.nc'
                filename = os.path.join(input_dir, string % (scenario, cl_model, fn_str_var, \
                                                         str(yearchunk['startyear']), \
                                                         str(yearchunk['endyear'])))
        else:
            items = filename.split('_')
            if 'histsoc' in filename:
                cl_model = None
                scenario = items[0]
#            elif '1860soc' in filename:
#                cl_model = None
#                scenario = items[2]
#                if '1860' in filename:
            else:
                cl_model = items[0]
                scenario = items[1]
            yearchunk = YEARCHUNKS[scenario]
            filename = input_dir+filename
            
        data_set = xr.open_dataset(filename, decode_times=False)
        data = data_set.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]))

        if yearrange is None:
            yearrange = np.array([yearchunk['startyear'], yearchunk['endyear']])

        time_idx = np.array([int(yearrange[0]-yearchunk['startyear']), \
                             int(yearrange[1]-yearchunk['startyear'])])

        if irr == 'combined':
            area_crop = (getattr(data, (CROP_NAME[crop])['input']+'_'+IRR[1])[\
                         time_idx[0]:time_idx[1], :, :].mean(dim='time')*AREA_CELL).values + \
                         (getattr(data, (CROP_NAME[crop])['input']+'_'+IRR[2])[\
                          time_idx[0]:time_idx[1], :, :].mean(dim='time')*AREA_CELL).values
        else:
            area_crop = (getattr(data, (CROP_NAME[crop])['input']+'_'+irr)[\
                         time_idx[0]:time_idx[1], :, :].mean(dim='time')*AREA_CELL).values

        area_crop = np.nan_to_num(area_crop).flatten()
        self['value'] = np.squeeze(area_crop*hist_mean)

        lon, lat = np.meshgrid(data.lon.values, data.lat.values)
        self['latitude'] = lat.flatten()
        self['longitude'] = lon.flatten()
        self['region_id'] = co.get_country_code(self.latitude, self.longitude)

        self.tag = Tag()
        self.tag.description = ("Crop yield ISIMIP " + (CROP_NAME[crop])['print'] + ' ' + \
                                irr + ' ' + str(yearrange[0]) + '-' + str(yearrange[1]))
        self.value_unit = 't / y'


        if unit == 'USD':
            dir_fao = input_dir+'FAO/'
            self.set_to_usd(dir_fao, crop=crop)

        self.check()

        return self

    def set_mean_of_several_models(self, input_dir=None, hist_mean=None, bbox=BBOX, \
                                   yearrange=None, cl_model=None, scenario=None, \
                                   crop=CROP[0], irr=IRR[0], unit='USD', fn_str_var=FN_STR_VAR):
        """Wrapper to fill exposure from several nc_dis files from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            historic mean (array): historic mean yield per centroid
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set, f.i. (1990, 2010)
            scenario (string): climate change and socio economic scenario
                f.i. 'histsoc' or 'rcp60soc'
            cl_model (string): abbrev. climate model (only when landuse data
            is future projection)
                f.i. 'gfdl-esm2m' etc.
            crop (string): crop type
                f.i. 'mai', 'ric', 'whe', 'soy'
            irr (string): irrigation type
                f.i 'rainfed', 'irrigated' or 'combined'= rainfed+irrigated
            unit (string): unit of the exposure (per year)
                f.i 'USD' or 't'
            fn_str_var (string): FileName STRing depending on VARiable and
                ISIMIP simuation round
        """

        filenames = [f for f in listdir(input_dir) if (isfile(join(input_dir, f))) if not \
                     f.startswith('.')]

        filenames2 = list()
        for name in filenames:
            if cl_model is not None and scenario is not None:
                if cl_model in name or scenario in name:
                    filenames2.append(name)
            elif cl_model is not None and scenario is None:
                if cl_model in name:
                    filenames2.append(name)
            elif cl_model is None and scenario is not None:
                if scenario in name:
                    filenames2.append(name)
            else:
                filenames2 = filenames


        self.set_from_single_run(input_dir, filename=filenames2[0], hist_mean=hist_mean, \
                                 bbox=bbox, yearrange=yearrange, crop=crop, irr=irr, \
                                 unit=unit, fn_str_var=fn_str_var)

        combined_exp = np.zeros([self.value.size, len(filenames2)])
        combined_exp[:, 0] = self.value

        for j in range(1, len(filenames2)):
            self.set_from_single_run(input_dir, filename=filenames2[j], hist_mean=hist_mean, \
                                     bbox=bbox, yearrange=yearrange, crop=crop, irr=irr, unit=unit)
            combined_exp[:, j] = self.value

        self['value'] = np.mean(combined_exp, 1)

        self.check()

        return self

    def set_to_usd(self, dir_fao=None, yearrange=YEARS_FAO, crop=CROP[0]):
        #to do: check api availability?; default yearrange for single year (e.g. 5a)
        #to do now: price for a specific year
        """ Calculates the exposure in USD using country and year specific data published
        by the FAO.

        Parameters:
            dir_fao (string): directory containing the FAO pricing data
            yearrange (int tuple): year range for prices set, f.i. (1990, 2010)
            crop (str): crop type
                f.i. 'temperate_cereals', 'maize', 'oil_crops_soybean' or 'rice'
        """
        fao = pd.read_csv(dir_fao + FAO_FILE)
        fao_countries = pd.read_csv(dir_fao + FAO_FILE2)

        fao_area = getattr(fao, 'Area Code').values
        fao_crops = fao.Item.values
        fao_year = fao.Year.values
        fao_price = fao.Value.values

        fao_code = getattr(fao_countries, 'Country Code').values
        fao_iso = getattr(fao_countries, 'ISO3 Code').values

        fao_country = list()
        for item in range(len(fao_area)):
            idx = (np.where(fao_area[item] == fao_code)[0])[0]
            fao_country.append(fao_iso[idx])

        iso3alpha = list()
        for item in range(len(self.region_id)):
            if self.region_id[item] == 0 or self.region_id[item] == -99:
                iso3alpha.append('No country')
            else:
                iso3alpha.append(iso_cntry.get(self.region_id[item]).alpha3)

        list_countries = np.unique(iso3alpha)

        crop_name_fao = (CROP_NAME[crop])['fao']
        area_price = np.zeros(self.value.size)

        for item in range(len(list_countries)):
            country = list_countries[item]
            if country != 'No country':
                idx_price = np.where((np.asarray(fao_country) == country) & \
                                     (np.asarray(fao_crops) == crop_name_fao) & \
                                     (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
                price = np.mean(fao_price[idx_price])
                if math.isnan(price) or price == 0:
                    idx_price = np.where((np.asarray(fao_crops) == crop_name_fao) & \
                                     (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
                    price = np.mean(fao_price[idx_price])
                idx_country = np.where(np.asarray(iso3alpha) == country)[0]
                area_price[idx_country] = self.value[idx_country]*price

        self['value'] = area_price
        self.value_unit = 'USD / y'

        self.check()

        return self
