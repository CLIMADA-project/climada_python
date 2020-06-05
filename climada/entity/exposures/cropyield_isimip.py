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
            '2100rcp26soc',
            'rcp60soc']

YEARCHUNKS = dict()
YEARCHUNKS[SCENARIO[0]] = dict()
YEARCHUNKS[SCENARIO[0]] = {'startyear' : 1861, 'endyear': 2005, 'duration': 145}
YEARCHUNKS[SCENARIO[1]] = dict()
YEARCHUNKS[SCENARIO[1]] = {'startyear' : 1861, 'endyear': 2005, 'duration': 145}
YEARCHUNKS[SCENARIO[4]] = dict()
YEARCHUNKS[SCENARIO[4]] = {'startyear' : 2100, 'endyear': 2299, 'duration': 200}


CROP = ['maize', 'rice', 'temperate_cereals', 'oil_crops_soybean']
CROP_NAME_FAO = dict()
CROP_NAME_FAO[CROP[0]] = {'crop' : 'Maize'}
CROP_NAME_FAO[CROP[1]] = {'crop' : 'Rice, paddy'}
CROP_NAME_FAO[CROP[2]] = {'crop' : 'Wheat'}
CROP_NAME_FAO[CROP[3]] = {'crop' : 'Soybeans'}


IRR = ['combined', 'rainfed', 'irrigated']

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
                            yearrange=None, scenario=SCENARIO[1], cl_model=CL_MODEL[0], \
                            fn_str_var=FN_STR_VAR, crop=CROP[0], irr=IRR[0], unit='USD'):

        """Wrapper to fill exposure from nc_dis file from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            historic mean (array): historic mean yield per centroid
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set, f.i. (1990, 2010)
            cl_model (str): abbrev. climate model (only when input_dir is selected and landuse data
            is future projection)
                f.i. 'gfdl-esm2m' etc.
            scenario (str): climate change and socio economic scenario
                f.i. 'histsoc' or 'rcp60soc'
            crop (str): crop type (only when input_dir is selected)
                f.i. 'temperate_cereals', 'maize', 'oil_crops_soybean' or 'rice'
            irr (str): irrigation type (only when input_dir is selected)
                f.i 'rainfed', 'irrigated' or 'combined'= rainfed+irrigated
            fn_str_var (str): FileName STRing depending on VARiable and
                ISIMIP simuation round
        raises:
            NameError
        """

        yearchunk = YEARCHUNKS[scenario]
        if filename is None:
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
            filename = input_dir+filename

        data_set = xr.open_dataset(filename, decode_times=False)
        data = data_set.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]))

        if yearrange is None:
            yearrange = np.array([yearchunk['startyear'], yearchunk['endyear']])

        time_idx = np.array([int(yearrange[0]-yearchunk['startyear']), \
                             int(yearrange[1]-yearchunk['startyear'])])

        if irr == 'combined':
            area_crop = (getattr(data, crop+'_'+IRR[1])[time_idx[0]:time_idx[1], :, :].mean(\
                         dim='time')*AREA_CELL).values + \
                         (getattr(data, crop+'_'+IRR[2])[time_idx[0]:time_idx[1], :, :].mean(\
                          dim='time')*AREA_CELL).values
        else:
            area_crop = (getattr(data, crop+'_'+irr)[time_idx[0]:time_idx[1], :, :].mean(\
                         dim='time')*AREA_CELL).values

        #vereinfachen?
        #area_crop = sparse.csr_matrix(np.nan_to_num(area_crop).flatten())
#        self['value'] = np.squeeze(np.asarray(sparse.csr_matrix(area_crop).multiply(\
#            sparse.csr_matrix(hist_mean)).todense()))
        area_crop = np.nan_to_num(area_crop).flatten()
        self['value'] = np.squeeze(area_crop*hist_mean)

        lon, lat = np.meshgrid(data.lon.values, data.lat.values)
        self['latitude'] = lat.flatten()
        self['longitude'] = lon.flatten()
        self['region_id'] = co.get_country_code(self.latitude, self.longitude)

        self.tag = Tag()
        self.tag.description = ("Crop yield ISIMIP " + crop + ' ' + irr + ' ' + \
                                str(yearrange[0]) + '-' + str(yearrange[1]))
        self.value_unit = 't / y'


        if unit == 'USD':
            dir_fao = input_dir+'FAO/'
            self.set_to_usd(dir_fao, crop=crop)

        self.check()

        return self

    def set_mean_of_several_models(self, input_dir=None, hist_mean=None, bbox=BBOX, yearrange=None):
        """Wrapper to fill exposure from several nc_dis files from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            historic mean (array): historic mean yield per centroid
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set, f.i. (1990, 2010)
        raises:
            NameError
        """

        filenames = [f for f in listdir(input_dir) if (isfile(join(input_dir, f))) if not \
                     f.startswith('.')]

        self.set_from_single_run(input_dir, filename=filenames[0], hist_mean=hist_mean, \
                                 bbox=bbox, yearrange=yearrange)

        combined_exp = np.zeros([self.value.size, len(filenames)])
        combined_exp[:, 0] = self.value

        for j in range(1, len(filenames)):
            self.set_from_single_run(input_dir, filename=filenames[j], hist_mean=hist_mean, \
                                     bbox=bbox, yearrange=yearrange)
            combined_exp[:, j] = self.value

        self['value'] = np.mean(combined_exp, 1)

        self.check()

        return self

    def set_to_usd(self, dir_fao=None, yearrange=YEARS_FAO, crop=CROP[0]):
        #to do: check api availability?; default yearrange bei einzelnem Jahr (fünf Jahre Umfeld)
        #to do now: price for a specific year;
            #what if there is no price for that crop and country? (use worldwide price?)
        """ Calculates the exposure in USD using country and year specific data published
        by the FAO.

        Parameters:
            yearrange (int tuple): year range for prices set, f.i. (1990, 2010)
            crop (str): crop type (only when input_dir is selected)
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
        for item, value in enumerate(fao_area):
            idx = (np.where(fao_area[item] == fao_code)[0])[0]
            fao_country.append(fao_iso[idx])

        iso3alpha = list()
        for item, value in  enumerate(self.region_id):
            if self.region_id[item] == 0 or self.region_id[item] == -99:
                iso3alpha.append('No country')
            else:
                iso3alpha.append(iso_cntry.get(self.region_id[item]).alpha3)

        list_countries = np.unique(iso3alpha)

        crop_name_fao = (CROP_NAME_FAO[crop])['crop']
        area_price = np.zeros(self.value.size)


        for item, value in enumerate(list_countries):
            country = list_countries[item]
            if country != 'No country':
                idx_price = np.where((np.asarray(fao_country) == country) & \
                                     (np.asarray(fao_crops) == crop_name_fao) & \
                                     (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
                price = np.mean(fao_price[idx_price])
                idx_country = np.where(np.asarray(iso3alpha) == country)[0]
                area_price[idx_country] = self.value[idx_country]*price

        self['value'] = area_price
        self.value_unit = 'USD / y'

        self.check()

        return self
