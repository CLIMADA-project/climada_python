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
from os.path import isfile, isdir, join
import math
import numpy as np
import xarray as xr
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from iso3166 import countries as iso_cntry
from climada.entity.exposures.base import Exposures
from climada.entity.tag import Tag
import climada.util.coordinates as co
from climada.util.constants import DATA_DIR, DEF_CRS
from climada.util.coordinates import pts_to_raster_meta, get_resolution


logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'CP'
"""Default hazard type used in impact functions id."""

BBOX = np.array([-180, -85, 180, 85])  # [Lon min, lat min, lon max, lat max]

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
# two types of 1860soc (1661-2299 not implemented)
YEARCHUNKS[SCENARIO[0]] = dict()
YEARCHUNKS[SCENARIO[0]] = {'yearrange': np.array([1800, 1860]), 'startyear': 1661, 'endyear': 1860}
YEARCHUNKS[SCENARIO[1]] = dict()
YEARCHUNKS[SCENARIO[1]] = {'yearrange': np.array([1976, 2005]), 'startyear': 1861, 'endyear': 2005}
YEARCHUNKS[SCENARIO[2]] = dict()
YEARCHUNKS[SCENARIO[2]] = {'yearrange': np.array([2006, 2099]), 'startyear': 2006, 'endyear': 2299}
YEARCHUNKS[SCENARIO[3]] = dict()
YEARCHUNKS[SCENARIO[3]] = {'yearrange': np.array([2006, 2099]), 'startyear': 2006, 'endyear': 2099}
YEARCHUNKS[SCENARIO[4]] = dict()
YEARCHUNKS[SCENARIO[4]] = {'yearrange': np.array([2006, 2099]), 'startyear': 2006, 'endyear': 2099}
YEARCHUNKS[SCENARIO[5]] = dict()
YEARCHUNKS[SCENARIO[5]] = {'yearrange': np.array([2100, 2299]), 'startyear': 2100, 'endyear': 2299}

YEARS_FAO = np.array([2000, 2018])

CROP = ['mai',
        'ric',
        'whe',
        'soy'
       ]

CROP_NAME = dict()
CROP_NAME[CROP[0]] = {'input': 'maize', 'fao': 'Maize', 'print': 'Maize'}
CROP_NAME[CROP[1]] = {'input': 'rice', 'fao': 'Rice, paddy', 'print': 'Rice'}
CROP_NAME[CROP[2]] = {'input': 'temperate_cereals', 'fao': 'Wheat', 'print': 'Wheat'}
CROP_NAME[CROP[3]] = {'input': 'oil_crops_soybean', 'fao': 'Soybeans', 'print': 'Soybeans'}

IRR = ['combined', 'noirr', 'firr']

IRR_NAME = dict()
IRR_NAME[IRR[0]] = {'name': 'combined'}
IRR_NAME[IRR[1]] = {'name': 'rainfed'}
IRR_NAME[IRR[2]] = {'name': 'irrigated'}

# default:
#   deposit the landuse files in the directory: climada_python/data/ISIMIP_crop/Input/Exposure
#   deposit the FAO files in the directory: climada_python/data/ISIMIP_crop/Input/Exposure/FAO
# The FAO files need to be downloaded and renamed
#   FAO_FILE: contains producer prices per crop, country and year
#               (http://www.fao.org/faostat/en/#data/PP)
#   FAO_FILE2: contains production quantity per crop, country and year
#               (http://www.fao.org/faostat/en/#data/QC)
INPUT_DIR = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Input', 'Exposure')
FAO_FILE = "FAOSTAT_data_producer_prices.csv"
FAO_FILE2 = "FAOSTAT_data_production_quantity.csv"

# default output directory: climada_python/data/ISIMIP_crop/Output/Exposure
# by default the hist_mean files created by climada_python/hazard/crop_potential are saved in
# climada_python/data/ISIMIP_crop/Output/hist_mean/
HIST_MEAN_PATH = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Output', 'Hist_mean')
OUTPUT_DIR = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Output')


class CropyieldIsimip(Exposures):
    """Defines agriculture exposures from ISIMIP input data and
    FAO crop price data"""

    _metadata = Exposures._metadata + ['crop']

    @property
    def _constructor(self):
        return CropyieldIsimip

    def set_from_single_run(self, input_dir=INPUT_DIR, filename=None, hist_mean=HIST_MEAN_PATH,
                            bbox=BBOX, yearrange=(YEARCHUNKS[SCENARIO[1]])['yearrange'],
                            cl_model=None, scenario=SCENARIO[1], crop=CROP[0], irr=IRR[0],
                            unit='USD', fn_str_var=FN_STR_VAR):

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
                f.i 'firr' (full irrigation), 'noirr' (no irrigation) or 'combined'= firr+noirr
            unit (string): unit of the exposure (per year)
                f.i 'USD' or 't'
            fn_str_var (string): FileName STRing depending on VARiable and
                ISIMIP simuation round
        """

        # The filename is set or other variables (cl_model, scenario) are extracted of the
        # specified filename
        if filename is None:
            yearchunk = YEARCHUNKS[scenario]
            # if scenario == 'histsoc' or scenario == '1860soc':
            if scenario in ('histsoc', '1860soc'):
                string = '%s_%s_%s_%s.nc'
                filename = os.path.join(input_dir, string % (scenario, fn_str_var,
                                                             str(yearchunk['startyear']),
                                                             str(yearchunk['endyear'])))
            else:
                string = '%s_%s_%s_%s_%s.nc'
                filename = os.path.join(input_dir, string % (scenario, cl_model, fn_str_var,
                                                             str(yearchunk['startyear']),
                                                             str(yearchunk['endyear'])))
        elif scenario == 'flexible':
            items = filename.split('_')
            yearchunk = dict()
            yearchunk = {'yearrange': np.array([int(items[6]), int(items[7].split('.')[0])]),
                         'startyear': int(items[6]), 'endyear': int(items[7].split('.')[0])}
            filename = os.path.join(input_dir, filename)
        else:
            items = filename.split('_')
            if 'histsoc' or '1860soc' in filename:
                cl_model = None
                scenario = items[0]
            else:
                cl_model = items[0]
                scenario = items[1]
            yearchunk = YEARCHUNKS[scenario]
            filename = os.path.join(input_dir, filename)

        # Dataset is opened and data within the bbox extends is extracted
        data_set = xr.open_dataset(filename, decode_times=False)
        data = data_set.sel(lon=slice(bbox[0], bbox[2]), lat=slice(bbox[3], bbox[1]))

        # The latitude and longitude are set; the region_id is determined
        lon, lat = np.meshgrid(data.lon.values, data.lat.values)
        self['latitude'] = lat.flatten()
        self['longitude'] = lon.flatten()
        self['region_id'] = co.get_country_code(self.latitude, self.longitude)

        # The indeces of the yearrange to be extracted are determined
        time_idx = np.array([int(yearrange[0] - yearchunk['startyear']),
                             int(yearrange[1] - yearchunk['startyear'])])

        # The area covered by a grid cell is calculated depending on the latitude
        # 1 degree = 111.12km (at the equator); resolution data: 0.5 degree;
        # longitudal distance in km = 111.12*0.5*cos(lat);
        # latitudal distance in km = 111.12*0.5;
        # area = longitudal distance * latitudal distance;
        # 1km2 = 100ha
        area = (111.12 * 0.5)**2 * np.cos(np.deg2rad(lat)) * 100

        # The area covered by a crop is calculated as the product of the fraction and
        # the grid cell size
        if irr == 'combined':
            area_crop = (
                (getattr(data,
                         (CROP_NAME[crop])['input'] + '_' + (IRR_NAME[IRR[1]])['name'])[
                            time_idx[0]:time_idx[1], :, :].mean(dim='time') * area
                ).values
                + (getattr(data,
                           (CROP_NAME[crop])['input'] + '_' + (IRR_NAME[IRR[2]])['name'])[
                               time_idx[0]:time_idx[1], :, :].mean(dim='time') * area
                ).values)
        else:
            area_crop = (getattr(data, (CROP_NAME[crop])['input'] + '_' + (IRR_NAME[irr])['name'])[
                         time_idx[0]:time_idx[1], :, :].mean(dim='time') * area).values

        area_crop = np.nan_to_num(area_crop).flatten()

        # The historic mean, its latitude and longitude are set
        if isdir(hist_mean):
        # The adequate file from the directory (depending on crop and irrigation) is extracted
        # and the variables hist_mean, lat_mean and lon_mean are set accordingly
            if irr != 'combined':
                filename = os.path.join(hist_mean, 'hist_mean_' + crop + '-' + irr + '_' +
                                        str(yearrange[0]) + '-' + str(yearrange[1]) + '.hdf5')
                hist_mean = (h5py.File(filename, 'r'))['mean'][()]
            else:
                filename = os.path.join(hist_mean, 'hist_mean_' + crop + '-' + IRR[1]
                                        + '_' + str(yearrange[0]) + '-'
                                        + str(yearrange[1]) + '.hdf5')
                filename2 = os.path.join(hist_mean, 'hist_mean_' + crop + '-' + IRR[2]
                                         + '_' + str(yearrange[0]) + '-'
                                         + str(yearrange[1]) + '.hdf5')
                hist_mean = ((h5py.File(filename, 'r'))['mean'][()] +
                             (h5py.File(filename2, 'r'))['mean'][()]) / 2
            lat_mean = (h5py.File(filename, 'r'))['lat'][()]
            lon_mean = (h5py.File(filename, 'r'))['lon'][()]
        elif isfile(os.path.join(input_dir, hist_mean)):
        # Hist_mean, lat_mean and lon_mean are extracted from the given file
            hist_mean_file = h5py.File(os.path.join(input_dir, hist_mean), 'r')
            hist_mean = hist_mean_file['mean'][()]
            lat_mean = hist_mean_file['lat'][()]
            lon_mean = hist_mean_file['lon'][()]
        else:
        # Hist_mean as returned by the hazard crop_potential is used (array format) with same
        # bbox extensions as the exposure
            lat_mean = self.latitude.values

        # The bbox is cut out of the hist_mean data file if needed
        if len(lat_mean) != len(self.latitude.values):
            idx_mean = np.zeros(len(self.latitude.values), dtype=int)
            for i in range(len(self.latitude.values)):
                idx_mean[i] = np.where(
                    (lat_mean == self.latitude.values[i])
                    & (lon_mean == self.longitude.values[i])
                )[0][0]
        else:
            idx_mean = np.arange(0, len(lat_mean))

        # The exposure [t/y] is computed per grid cell as the product of the area covered
        # by a crop [ha] and its yield [t/ha/y]
        self['value'] = np.squeeze(area_crop * hist_mean[idx_mean])

        self.tag = Tag()
        self.tag.description = ("Crop yield ISIMIP " + (CROP_NAME[crop])['print'] + ' ' +
                                irr + ' ' + str(yearrange[0]) + '-' + str(yearrange[1]))
        self.value_unit = 't / y'
        self.crop = crop
        self.ref_year = yearrange
        self.crs = DEF_CRS
        try:
            rows, cols, ras_trans = pts_to_raster_meta(
                (self.longitude.min(), self.latitude.min(),
                 self.longitude.max(), self.latitude.max()),
                get_resolution(self.longitude, self.latitude))
            self.meta = {
                'width': cols,
                'height': rows,
                'crs': self.crs,
                'transform': ras_trans,
            }
        except ValueError:
            LOGGER.warning('Could not write attribute meta, because exposure'
                           ' has only 1 data point')
            self.meta = {}

        # Method set_to_usd() is called to compute the exposure in USD/y (per centroid)
        # the exposure in t/y is saved as 'value_tonnes'
        if unit == 'USD':
            self['value_tonnes'] = self['value']
            self.set_to_usd(input_dir=input_dir)
        self.check()

        return self

    def set_mean_of_several_models(self, input_dir=INPUT_DIR, hist_mean=HIST_MEAN_PATH, bbox=BBOX,
                                   yearrange=(YEARCHUNKS[SCENARIO[1]])['yearrange'],
                                   cl_model=None, scenario=None, crop=CROP[0], irr=IRR[0],
                                   unit='USD', fn_str_var=FN_STR_VAR):
        """Wrapper to fill exposure from several nc_dis files from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            historic mean (array): historic mean yield per centroid
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set, f.i. (1976, 2005)
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

        filenames = [f for f in listdir(input_dir) if (isfile(join(input_dir, f))) if not
                     f.startswith('.') if 'nc' in f]

        # If only files with a certain scenario and or cl_model shall be considered, they
        # are extracted from the original list of files
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

        # The first exposure is calculate to determine its size
        # and initialize the combined exposure
        self.set_from_single_run(input_dir, filename=filenames2[0], hist_mean=hist_mean,
                                 bbox=bbox, yearrange=yearrange, crop=crop, irr=irr,
                                 unit=unit, fn_str_var=fn_str_var)

        combined_exp = np.zeros([self.value.size, len(filenames2)])
        combined_exp[:, 0] = self.value

        # The calculations are repeated for all exposures
        for j in range(1, len(filenames2)):
            self.set_from_single_run(input_dir, filename=filenames2[j], hist_mean=hist_mean,
                                     bbox=bbox, yearrange=yearrange, crop=crop, irr=irr, unit=unit)
            combined_exp[:, j] = self.value

        self['value'] = np.mean(combined_exp, 1)
        self['crop'] = crop

        self.check()

        return self

    def set_to_usd(self, input_dir=INPUT_DIR, yearrange=YEARS_FAO):
        # to do: check api availability?; default yearrange for single year (e.g. 5a)
        """Calculates the exposure in USD using country and year specific data published
        by the FAO.

        Parameters:
            input_dir (string): directory containing the input (FAO pricing) data
            yearrange (array): year range for prices, f.i. (2000, 2018)
                can also be set to a single year
            crop (str): crop type
                f.i. 'mai', 'ric', 'whe', 'soy'
        """

        # account for the case of only specifying one year as yearrange
        if len(yearrange) == 1:
            yearrange = np.array([yearrange[0], yearrange[0]])

        # open both FAO files and extract needed variables
        # FAO_FILE: contains producer prices per crop, country and year
        fao = pd.read_csv(os.path.join(input_dir, FAO_FILE))
        fao_area = getattr(fao, 'Area Code').values
        fao_crops = fao.Item.values
        fao_year = fao.Year.values
        fao_price = fao.Value.values

        fao_country = co.country_faocode2iso(fao_area)

        # create a list of the countries contained in the exposure
        iso3alpha = list()
        for item, _ in enumerate(self.region_id):
            if (self.region_id[item] == 0) or (self.region_id[item] == -99):
                iso3alpha.append('No country')
            elif (self.region_id[item] == 902) or (self.region_id[item] == 910) or \
            (self.region_id[item] == 914) or (self.region_id[item] == 915):
                iso3alpha.append('Other country')
            else:
                iso3alpha.append(iso_cntry.get(self.region_id[item]).alpha3)
        list_countries = np.unique(iso3alpha)

        # iterate over all countries that are covered in the exposure, extract the according price
        # and calculate the produced yield in USD/y
        area_price = np.zeros(self.value.size)
        for item, _ in enumerate(list_countries):
            country = list_countries[item]
            if country != 'No country':
                if country == 'Other country':
                    price = 0
                else:
                    idx_price = np.where((np.asarray(fao_country) == country) &
                                         (np.asarray(fao_crops) == (CROP_NAME[self.crop])['fao']) &
                                         (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
                    price = np.mean(fao_price[idx_price])
                # if no price can be determined for a specific yearrange and country, the world
                # average for that crop (in the specified yearrange) is used
                if math.isnan(price) or price == 0:
                    idx_price = np.where((np.asarray(fao_crops) == (CROP_NAME[self.crop])['fao']) &
                                         (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
                    price = np.mean(fao_price[idx_price])
                idx_country = np.where(np.asarray(iso3alpha) == country)[0]
                area_price[idx_country] = self.value[idx_country] * price

        self['value'] = area_price
        self.value_unit = 'USD / y'

        self.check()

        return self

    def aggregate_countries(self):
        """Aggregate exposure data by country.

        Returns:
            list_countries (list): country codes
            country_values (array): aggregated exposure value

        """

        list_countries = np.unique(self.region_id)
        country_values = np.zeros(len(list_countries))
        for i, iso_nr in enumerate(list_countries):
            country_values[i] = self.loc[self.region_id == iso_nr].value.sum()

        return list_countries, country_values

def init_full_exposure_set(input_dir=INPUT_DIR, filename=None, hist_mean_dir=HIST_MEAN_PATH,
                           output_dir=OUTPUT_DIR, bbox=BBOX,
                           yearrange=(YEARCHUNKS[SCENARIO[1]])['yearrange'], unit='t',
                           returns='filename_list'):
    """Generates CropyieldIsimip exposure sets for all files contained in the
    input directory and saves them as hdf5 files in the output directory

        Parameters:
        input_dir (string): path to input data directory
        filename (string): if not specified differently, the file
            'histsoc_landuse-15crops_annual_1861_2005.nc' will be used
        output_dir (string): path to output data directory
        bbox (list of four floats): bounding box:
            [lon min, lat min, lon max, lat max]
        yearrange (array): year range for hazard set, f.i. (1976, 2005)
        unit (str): unit in which to return exposure (t/y or USD/y)
        returns (str): returned output
        'filename_list': returns list of filenames only, else returns also list of data

    """

    filenames = [f for f in listdir(hist_mean_dir) if (isfile(join(hist_mean_dir, f))) if not
                 f.startswith('.')]

    # generate output directory if it does not exist yet
    if not os.path.exists(os.path.join(output_dir, 'Exposure')):
        os.mkdir(os.path.join(output_dir, 'Exposure'))

    # create exposures for all crop-irrigation combinations and save them
    filename_list = list()
    output_list = list()
    for i, _ in enumerate(filenames):
        item = filenames[i].split('_')
        cropyield = CropyieldIsimip()
        cropyield.set_from_single_run(input_dir=input_dir, filename=filename,
                                      hist_mean=hist_mean_dir, bbox=bbox,
                                      yearrange=yearrange, crop=((item[2]).split('-'))[0],
                                      irr=((item[2]).split('-'))[1], unit=unit)
        filename_saveto = ('cropyield_isimip_' + ((item[2]).split('-'))[0] + '-'
                           + (((item[2]).split('-'))[1]).split('.')[0] + '_'
                           + str(yearrange[0]) + '-' + str(yearrange[1]) + '.hdf5')
        filename_list.append(filename_saveto)
        output_list.append(cropyield)
        cropyield.write_hdf5(os.path.join(output_dir, 'Exposure', filename_saveto))

    if returns == 'filename_list':
        return filename_list
    return filename_list, output_list

def normalize_with_fao_cropyield(exp_firr, exp_noirr, input_dir=INPUT_DIR,
                                 yearrange=np.array([2008, 2018]),
                                 unit='t', returns='all'):
    """Normalize the given exposures countrywise with the mean cropyield production quantity
    documented by the FAO.

        Parameters:
        exp_firr (cropyield_isimip): exposure under full irrigation
        exp_noirr (cropyield_isimip): exposure under no irrigation
        input_dir (str): directory containing exposure input data
        yearrange (array): the mean yield in this year range is used to normalize the exposure
            data (default 2008-2018)
        unit (str): unit in which to return exposure (t/y or USD/y)
        returns (str): returned output
            'all': country list, ratio = FAO/ISIMIP, normalized exposures, yield per country
            as documented by the FAO and calculated by the ISIMIP dataset
            else: country list, ratio = FAO/ISIMIP, normalized exposures

    """

    # use the exposure in t/y to normalize with FAO yield values
    if (exp_firr.value_unit == 'USD / y') and (exp_noirr.value_unit == 'USD / y'):
        exp_firr.value = exp_firr.value_tonnes
        exp_noirr.value = exp_noirr.value_tonnes
    elif exp_firr.value_unit == 'USD / y':
        exp_firr.value = exp_firr.value_tonnes
    elif exp_noirr.value_unit == 'USD / y':
        exp_noirr.value = exp_noirr.value_tonnes

    country_list, countries_firr = exp_firr.aggregate_countries()
    country_list, countries_noirr = exp_noirr.aggregate_countries()

    exp_totyield = countries_firr + countries_noirr

    fao = pd.read_csv(os.path.join(input_dir, FAO_FILE2))
    fao_crops = fao.Item.values
    fao_year = fao.Year.values
    fao_values = fao.Value.values
    fao_code = getattr(fao, 'Area Code').values

    fao_country = co.country_iso2faocode(country_list)

    fao_yield = np.zeros(len(country_list))
    ratio = np.ones(len(country_list))
    exp_firr_norm = CropyieldIsimip()
    exp_firr_norm = exp_firr
    exp_noirr_norm = CropyieldIsimip()
    exp_noirr_norm = exp_noirr

    for country, iso_nr in enumerate(country_list):
        idx = np.where((np.asarray(fao_code) == fao_country[country])
                       & (np.asarray(fao_crops) == (CROP_NAME[exp_firr.crop])['fao'])
                       & (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
        if len(idx) >= 1:
            fao_yield[country] = np.mean(fao_values[idx])

        # if a country has no values in the exposure (e.g. Cyprus) the FAO average value is used
        if exp_totyield[country] == 0:
            exp_totyield[country] = fao_yield[country]
        # if a country has no fao value, the ratio is left being 1
        elif fao_yield[country] != np.nan and fao_yield[country] != 0:
            ratio[country] = fao_yield[country] / exp_totyield[country]

        exp_firr_norm.value[exp_firr.region_id == iso_nr] = ratio[country] * \
        exp_firr.value[exp_firr.region_id == iso_nr]
        exp_noirr_norm.value[exp_firr.region_id == iso_nr] = ratio[country] * \
        exp_noirr.value[exp_noirr.region_id == iso_nr]

        if unit == 'USD':
            exp_noirr['value_tonnes'] = exp_noirr['value']
            exp_noirr.set_to_usd(input_dir=input_dir)
            exp_firr['value_tonnes'] = exp_firr['value']
            exp_firr.set_to_usd(input_dir=input_dir)

    if returns == 'all':
        return country_list, ratio, exp_firr_norm, exp_noirr_norm, fao_yield, exp_totyield
    return country_list, ratio, exp_firr_norm, exp_noirr_norm

def normalize_several_exp(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                          yearrange=np.array([2008, 2018]),
                          unit='t', returns='all'):
    """

        Parameters:
        input_dir (str): directory containing exposure input data
        output_dir (str): directory containing exposure datasets (output of exposure creation)
        yearrange (array): the mean yield in this year range is used to normalize the exposure
            data (default 2008-2018)
        unit (str): unit in which to return exposure (t/y or USD/y)
        returns (str): returned output
            'all': lists containing data for each exposure file. Lists: crops, country list,
            ratio = FAO/ISIMIP, normalized exposures, yield per country as documented by the
            FAO and calculated by the ISIMIP dataset
            else: lists containing data for each exposure file. Lists: crops, country list,
            ratio = FAO/ISIMIP, normalized exposures
        Returns:

    """
    filenames_exp = [f for f in listdir(os.path.join(output_dir, 'Exposure')) if
                     (isfile(join(os.path.join(output_dir, 'Exposure'), f))) if not
                     f.startswith('.') if 'firr' in f]

    crop_list = list()
    countries_list = list()
    ratio_list = list()
    exp_firr_norm = list()
    exp_noirr_norm = list()
    fao_yield_list = list()
    exp_totyield_list = list()

    for crop, _ in enumerate(filenames_exp):
        items_exp = filenames_exp[crop].split('_')
        exp_noirr = CropyieldIsimip()
        exp_noirr.read_hdf5(os.path.join(output_dir, 'Exposure', filenames_exp[crop]))

        filename_firr = items_exp[0] + '_' + items_exp[1] + '_' + items_exp[2].split('-')[0] +\
        '-' + 'noirr' + '_' + items_exp[3]
        exp_firr = CropyieldIsimip()
        exp_firr.read_hdf5(os.path.join(output_dir, 'Exposure', filename_firr))

        if returns == 'all':
            countries, ratio, exp_firr2, exp_noirr2, fao_yield, \
            exp_totyield = normalize_with_fao_cropyield(exp_firr, exp_noirr, input_dir=input_dir,
                                                        yearrange=yearrange, unit=unit)
            fao_yield_list.append(fao_yield)
            exp_totyield_list.append(exp_totyield)
        else:
            countries, ratio, exp_firr2, \
            exp_noirr2 = normalize_with_fao_cropyield(exp_firr, exp_noirr,
                                                      input_dir=input_dir,
                                                      yearrange=yearrange, unit=unit,
                                                      returns='reduced')


        crop_list.append(items_exp[2].split('-')[0])
        countries_list.append(countries)
        ratio_list.append(ratio)
        exp_firr_norm.append(exp_firr2)
        exp_noirr_norm.append(exp_noirr2)

    if returns == 'all':
        return crop_list, countries_list, ratio_list, exp_firr_norm, exp_noirr_norm, \
                fao_yield_list, exp_totyield_list
    return crop_list, countries_list, ratio_list, exp_firr_norm, exp_noirr_norm

def semilogplot_ratio(crop, countries, ratio, output_dir=OUTPUT_DIR, save=True):
    """Plot ratio = FAO/ISIMIP against country codes.

        Parameters:
        crop (str): crop to plot
        countries (list): country codes of countries to plot
        ratio (array): ratio = FAO/ISIMIP yield data of countries to plot
        output_dir (str): directory to save figure
        save (boolean): True saves figure, else figure is not saved
        Returns:

    """
    fig = plt.figure()
    axes = plt.gca()
    axes.scatter(countries[ratio != 1], ratio[ratio != 1])
    axes.set_yscale('log')
    axes.set_ylabel('Ratio= FAO / ISIMIP')
    axes.set_xlabel('ISO3 country code')
    axes.set_ylim(np.nanmin(ratio), np.nanmax(ratio))
    plt.title(crop)

    if save:
        if not os.path.exists(os.path.join(output_dir, 'Exposure_norm_plots')):
            os.mkdir(os.path.join(output_dir, 'Exposure_norm_plots'))
        plt.savefig(os.path.join(output_dir, 'Exposure_norm_plots',
                                 'fig_ratio_norm_' + crop))

    return fig, axes
