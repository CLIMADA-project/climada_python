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

Agriculture exposures from ISIMIP and FAO.
"""


import logging
import math
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
import h5py
from matplotlib import pyplot as plt

from climada.entity.tag import Tag
import climada.util.coordinates as u_coord
from climada import CONFIG
from .base import Exposures, INDICATOR_IMPF

logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'RC'
"""Default hazard type used in impact functions id."""

BBOX = (-180, -85, 180, 85)  # [Lon min, lat min, lon max, lat max]
""""Default geographical bounding box of the total global agricultural land extent"""

#ISIMIP input data specific global variables
YEARCHUNKS = dict()
"""start and end years per ISIMIP version and senario as in ISIMIP-filenames
of landuse data containing harvest area per crop"""
# two types of 1860soc (1661-2299 not implemented)
YEARCHUNKS['ISIMIP2'] = dict()
YEARCHUNKS['ISIMIP2']['1860soc'] = {'yearrange': (1800, 1860), 'startyear': 1661, 'endyear': 1860}
YEARCHUNKS['ISIMIP2']['histsoc'] = {'yearrange': (1976, 2005), 'startyear': 1861, 'endyear': 2005}
YEARCHUNKS['ISIMIP2']['2005soc'] = {'yearrange': (2006, 2099), 'startyear': 2006, 'endyear': 2299}
YEARCHUNKS['ISIMIP2']['rcp26soc'] = {'yearrange': (2006, 2099), 'startyear': 2006, 'endyear': 2099}
YEARCHUNKS['ISIMIP2']['rcp60soc'] = {'yearrange': (2006, 2099), 'startyear': 2006, 'endyear': 2099}
YEARCHUNKS['ISIMIP2']['2100rcp26soc'] = {'yearrange': (2100, 2299), 'startyear': 2100,
                                         'endyear': 2299}
YEARCHUNKS['ISIMIP3'] = dict()
YEARCHUNKS['ISIMIP3']['histsoc'] = {'yearrange': (1983, 2013), 'startyear': 1850, 'endyear': 2014}
YEARCHUNKS['ISIMIP3']['2015soc'] = {'yearrange': (1983, 2013), 'startyear': 1850, 'endyear': 2014}

FN_STR_VAR = 'landuse-15crops_annual'
"""fix filename part in input data"""

CROP_NAME = dict()
"""mapping of crop names"""
CROP_NAME['mai'] = {'input': 'maize', 'fao': 'Maize', 'print': 'Maize'}
CROP_NAME['ric'] = {'input': 'rice', 'fao': 'Rice, paddy', 'print': 'Rice'}
CROP_NAME['whe'] = {'input': 'temperate_cereals', 'fao': 'Wheat', 'print': 'Wheat'}
CROP_NAME['soy'] = {'input': 'oil_crops_soybean', 'fao': 'Soybeans', 'print': 'Soybeans'}

CROP_NAME['ri1'] = {'input': 'rice', 'fao': 'Rice, paddy', 'print': 'Rice 1st season'}
CROP_NAME['ri2'] = {'input': 'rice', 'fao': 'Rice, paddy', 'print': 'Rice 2nd season'}
CROP_NAME['swh'] = {'input': 'temperate_cereals', 'fao': 'Wheat', 'print': 'Spring Wheat'}
CROP_NAME['wwh'] = {'input': 'temperate_cereals', 'fao': 'Wheat', 'print': 'Winter Wheat'}

"""mapping of irrigation parameter long names"""
IRR_NAME = {'combined': {'name': 'combined'},
            'noirr': {'name': 'rainfed'},
            'firr': {'name': 'irrigated'},
            }

"""Conversion factor weight [tons] to nutritional value [kcal].
Based on Mueller et al. (2021), https://doi.org/10.1088/1748-9326/abd8fc :

"For the aggregation of different crops, we compute total calories, assuming
net water contents of 12% for maize, spring and winter wheat, 13% for rice and
9% for soybean, according to Wirsenius (2000) and caloric contents of the
“as purchased” biomass (i.e. including the water content) of 3.56kcal/g for maize,
2.8kcal/g for rice, 3.35kcal/g for soybean and of 3.34kcal/g for spring and
winter wheat, following FAO (2001).” (Müller et al., 2021)

Version 1: conversion factors for crop biomass "as purchased",
    here applied as default for FAO-normalized production:
    Production [kcal] = Production [t] * KCAL_PER_TON [kcal/t]
"""

KCAL_PER_TON = dict()
KCAL_PER_TON['biomass'] = {'mai': 3.56e6,
                           'ric': 2.80e6,
                           'soy': 3.35e6,
                           'whe': 3.34e6,
                           }
"""
Version 2: conversion factors for crop dry matter as simulated by most crop models,
    here applied as default for raw ISIMIP model yields and derived production values:
    Yield [kcal] = Yield [t] * KCAL_PER_TON [kcal/t] / (1-net_water_content_fraction)
"""
KCAL_PER_TON['drymatter'] = {'mai': 3.56e6 / (1-.12),
                             'ric': 2.80e6 / (1-.13),
                             'soy': 3.35e6 / (1-.09),
                             'whe': 3.34e6 / (1-.12),
                             }

# Default folder structure for ISIMIP data:
#   deposit the landuse and FAO files in the directory:
#   {CONFIG.exposures.crop_production.local_data}/Input/Exposure
# The FAO files need to be downloaded and renamed
#   FAO_FILE: contains producer prices per crop, country and year
#               (http://www.fao.org/faostat/en/#data/PP)
#   FAO_FILE2: contains production quantity per crop, country and year
#               (http://www.fao.org/faostat/en/#data/QC)
DATA_DIR = CONFIG.exposures.crop_production.local_data.dir()
INPUT_DIR = DATA_DIR.joinpath('Input', 'Exposure')
FAO_FILE = "FAOSTAT_data_producer_prices.csv"
FAO_FILE2 = "FAOSTAT_data_production_quantity.csv"

YEARS_FAO = (2008, 2018)
"""Default years from FAO used (data file contains values for 1991-2018)"""

# default output directory: climada_python/data/ISIMIP_crop/Output/Exposure
# by default the hist_mean files created by climada_python/hazard/crop_potential are saved in
# climada_python/data/ISIMIP_crop/Output/hist_mean/
HIST_MEAN_PATH = DATA_DIR.joinpath('Output', 'Hist_mean')
OUTPUT_DIR = DATA_DIR.joinpath('Output')


class CropProduction(Exposures):
    """Defines agriculture exposures from ISIMIP input data and FAO crop data

    geopandas GeoDataFrame with metadata and columns (pd.Series) defined in
    Attributes and Exposures.

    Attributes:
        crop (str): crop typee.g., 'mai', 'ric', 'whe', 'soy'
    """

    _metadata = Exposures._metadata + ['crop']

    def set_from_isimip_netcdf(self, input_dir=None, filename=None, hist_mean=None,
                               bbox=None, yearrange=None, cl_model=None, scenario=None,
                               crop=None, irr=None, isimip_version=None,
                               unit=None, fn_str_var=None):

        """Wrapper to fill exposure from NetCDF file from ISIMIP. Requires historical
        mean relative cropyield module as additional input.
        Optional Parameters:
            input_dir (Path or str): path to input data directory,
                default: {CONFIG.exposures.crop_production.local_data}/Input/Exposure
            filename (string): name of the landuse data file to use,
                e.g. "histsoc_landuse-15crops_annual_1861_2005.nc""
            hist_mean (str or array): historic mean crop yield per centroid (or path)
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set
               e.g., (1990, 2010)
            scenario (string): climate change and socio economic scenario
               e.g., '1860soc', 'histsoc', '2005soc', 'rcp26soc','rcp60soc','2100rcp26soc'
            cl_model (string): abbrev. climate model (only for future projections of lu data)
               e.g., 'gfdl-esm2m', 'hadgem2-es', 'ipsl-cm5a-lr','miroc5'
            crop (string): crop type
               e.g., 'mai', 'ric', 'whe', 'soy'
            irr (string): irrigation type, default: 'combined'
                f.i 'firr' (full irrigation), 'noirr' (no irrigation) or 'combined'= firr+noirr
            isimip_version(str): 'ISIMIP2' (default) or 'ISIMIP3'
            unit (string): unit of the exposure (per year)
                f.i 't/y' (default), 'USD/y', or 'kcal/y'
            fn_str_var (string): FileName STRing depending on VARiable and
                ISIMIP simuation round

        Returns:
            Exposure
        """
        # parameters not provided in method call are set to default values:
        if irr is None:
            irr = 'combined'
        if not bbox:
            bbox = BBOX
        if not input_dir:
            input_dir = INPUT_DIR
        input_dir = Path(input_dir)
        if hist_mean is None:
            hist_mean = HIST_MEAN_PATH
        if isinstance(hist_mean, str):
            hist_mean = Path(hist_mean)
        if not fn_str_var:
            fn_str_var = FN_STR_VAR
        if (not isimip_version) or (isimip_version in ('ISIMIP2a', 'ISIMIP2b')):
            isimip_version = 'ISIMIP2'
        elif isimip_version in ('ISIMIP3a', 'ISIMIP3b'):
            isimip_version = 'ISIMIP3'
        if (not scenario) or (scenario in ('historical', 'hist')):
            scenario = 'histsoc'
        if yearrange is None:
            yearrange = YEARCHUNKS[isimip_version][scenario]['yearrange']
        if not unit:
            unit = 't/y'

        if isinstance(filename, Path): # if Path, extract pure filename as string
            if  filename.is_file() and filename.parent.is_dir():
                LOGGER.info('input_dir is reset from %s to %s', input_dir, filename.parent)
                input_dir = filename.parent
            filename = filename.parts[-1]

        # The filename is set or other variables (cl_model, scenario) are extracted of the
        # specified filename
        if filename is None:
            yearchunk = YEARCHUNKS[isimip_version][scenario]
            # if scenario == 'histsoc' or scenario == '1860soc':
            if scenario in ('histsoc', '1860soc'):
                string = '{}_{}_{}_{}.nc'
                filepath = Path(input_dir, string.format(scenario, fn_str_var,
                                                         yearchunk['startyear'],
                                                         yearchunk['endyear']))
            else:
                string = '{}_{}_{}_{}_{}.nc'
                filepath = Path(input_dir, string.format(scenario, cl_model, fn_str_var,
                                                         yearchunk['startyear'],
                                                         yearchunk['endyear']))
        elif scenario == 'flexible':
            _, _, _, _, _, _, startyear, endyearnc = filename.split('_')
            endyear = endyearnc.split('.')[0]
            yearchunk = dict()
            yearchunk = {'yearrange': (int(startyear), int(endyear)),
                         'startyear': int(startyear), 'endyear': int(endyear)}
            filepath = Path(input_dir, filename)
        else:
            scenario, *_ = filename.split('_')
            yearchunk = YEARCHUNKS[isimip_version][scenario]
            filepath = Path(input_dir, filename)

        # Dataset is opened and data within the bbox extends is extracted
        data_set = xr.open_dataset(filepath, decode_times=False)
        [lonmin, latmin, lonmax, latmax] = bbox
        data = data_set.sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin))

        # The latitude and longitude are set; the region_id is determined
        lon, lat = np.meshgrid(data.lon.values, data.lat.values)
        self.gdf['latitude'] = lat.flatten()
        self.gdf['longitude'] = lon.flatten()
        self.gdf['region_id'] = u_coord.get_country_code(self.gdf.latitude, self.gdf.longitude)

        # The indeces of the yearrange to be extracted are determined
        time_idx = (int(yearrange[0] - yearchunk['startyear']),
                    int(yearrange[1] - yearchunk['startyear']))

        # The area covered by a grid cell is calculated depending on the latitude
        area = u_coord.get_gridcellarea(lat, resolution=0.5)

        # The area covered by a crop is calculated as the product of the fraction and
        # the grid cell size
        if irr == 'combined':
            irr_types = ['firr', 'noirr']
        else:
            irr_types = [irr]
        area_crop = dict()
        for irr_var in irr_types:
            area_crop[irr_var] = (
                getattr(
                    data, (CROP_NAME[crop])['input']+'_'+ (IRR_NAME[irr_var])['name']
                )[time_idx[0]:time_idx[1], :, :].mean(dim='time')*area
            ).values
            area_crop[irr_var] = np.nan_to_num(area_crop[irr_var]).flatten()

        # set historic mean, its latitude, and longitude:
        hist_mean_dict = dict()
        # if hist_mean is given as np.ndarray or dict,
        # code assumes it contains hist_mean as returned by relative_cropyield
        # however structured in dictionary as hist_mean_dict, with same
        # bbox extensions as the exposure:
        if isinstance(hist_mean, dict):
            if not ('firr' in hist_mean.keys() or 'noirr' in hist_mean.keys()):
                # as a dict hist_mean, needs to contain key 'firr' or 'noirr';
                # if irr=='combined', both 'firr' and 'noirr' are required.
                raise ValueError(f'Invalid hist_mean provided: {hist_mean}')
            hist_mean_dict = hist_mean
            lat_mean = self.gdf.latitude.values
        elif isinstance(hist_mean, np.ndarray) or isinstance(hist_mean, list):
            hist_mean_dict[irr_types[0]] = np.array(hist_mean)
            lat_mean = self.gdf.latitude.values
        elif Path(hist_mean).is_dir(): # else if hist_mean is given as path to directory
        # The adequate file from the directory (depending on crop and irrigation) is extracted
        # and the variables hist_mean, lat_mean and lon_mean are set accordingly
            for irr_var in irr_types:
                filename = str(Path(hist_mean, 'hist_mean_%s-%s_%i-%i.hdf5' %(
                    crop, irr_var, yearrange[0], yearrange[1])))
                hist_mean_dict[irr_var] = (h5py.File(filename, 'r'))['mean'][()]
            lat_mean = (h5py.File(filename, 'r'))['lat'][()]
            lon_mean = (h5py.File(filename, 'r'))['lon'][()]
        elif Path(input_dir, hist_mean).is_file(): # file in input_dir
        # Hist_mean, lat_mean and lon_mean are extracted from the given file
            if len(irr_types) > 1:
                raise ValueError("For irr=='combined', hist_mean cannot be a single file.")
            hist_mean = h5py.File(str(Path(input_dir, hist_mean)), 'r')
            hist_mean_dict[irr_types[0]] = hist_mean['mean'][()]
            lat_mean = hist_mean['lat'][()]
            lon_mean = hist_mean['lon'][()]
        elif hist_mean.is_file(): # fall back: complete file path
        # Hist_mean, lat_mean and lon_mean are extracted from the given file
            if len(irr_types) > 1:
                raise ValueError("For irr=='combined', hist_mean can not be single file.")
            hist_mean = h5py.File(str(Path(input_dir, hist_mean)), 'r')
            hist_mean_dict[irr_types[0]] = hist_mean['mean'][()]
            lat_mean = hist_mean['lat'][()]
            lon_mean = hist_mean['lon'][()]
        else:
            raise ValueError(f"Invalid hist_mean provided: {hist_mean}")

        # The bbox is cut out of the hist_mean data file if needed
        if len(lat_mean) != len(self.gdf.latitude.values):
            idx_mean = np.zeros(len(self.gdf.latitude.values), dtype=int)
            for i in range(len(self.gdf.latitude.values)):
                idx_mean[i] = np.where(
                    (lat_mean == self.gdf.latitude.values[i])
                    & (lon_mean == self.gdf.longitude.values[i])
                )[0][0]
        else:
            idx_mean = np.arange(0, len(lat_mean))

        # The exposure [t/y] is computed per grid cell as the product of the area covered
        # by a crop [ha] and its yield [t/ha/y]
        self.gdf['value'] = np.squeeze(area_crop[irr_types[0]] * \
                                   hist_mean_dict[irr_types[0]][idx_mean])
        self.gdf['value'] = np.nan_to_num(self.gdf.value) # replace NaN by 0.0
        for irr_val in irr_types[1:]: # add other irrigation types if irr=='combined'
            value_tmp = np.squeeze(area_crop[irr_val]*hist_mean_dict[irr_val][idx_mean])
            value_tmp = np.nan_to_num(value_tmp) # replace NaN by 0.0
            self.gdf['value'] += value_tmp
        self.tag = Tag()

        self.tag.description = ("Crop production exposure from ISIMIP " +
                                (CROP_NAME[crop])['print'] + ' ' +
                                irr + ' ' + str(yearrange[0]) + '-' + str(yearrange[-1]))
        self.value_unit = 't/y' # input unit, will be reset below if required by user
        self.crop = crop
        self.ref_year = yearrange
        try:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (self.gdf.longitude.min(), self.gdf.latitude.min(),
                 self.gdf.longitude.max(), self.gdf.latitude.max()),
                u_coord.get_resolution(self.gdf.longitude, self.gdf.latitude))
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

        if 'USD' in unit:
            # set_value_to_usd() is called to compute the exposure in USD/y (country specific)
            self.set_value_to_usd(input_dir=input_dir)
        elif 'kcal' in unit:
            # set_value_to_kcal() is called to compute the exposure in kcal/y
            # here, biomass=False because most crop models provide yield weight
            # for dry matter, not biomass:
            self.set_value_to_kcal(biomass=False)
        self.check()
        return self

    def set_from_area_and_yield_nc4(self, crop_type, layer_yield, layer_area,
                                    filename_yield, filename_area, var_yield,
                                    var_area, bbox=BBOX, input_dir=INPUT_DIR):
        
        """
        Set crop_production exposure from cultivated area [ha] and
        yield [t/ha/year] provided in two netcdf files with the same grid.

        Both input files need to be netcdf format and come with dimensions
        'lon', 'lat' and 'crop'. The information which crop type is saved in which
        crop layer in each input files needs to be provided manually via
        the parameters 'layer_*'.

        A convenience wrapper around this expert method is provided with
        set_from_spam_ray_mirca().

        Parameters
        ----------
        crop_type : str
            Crop type, e.g. 'mai' for maize, or 'ric', 'whe', 'soy', etc.
        layer_yield : int
            crop layer in yield input data set. Index typically starts with 1.
        layer_area : int
            crop layer in area input data set. Index typically starts with 1.
        filename_yield : str
            Name of netcdf-file containing gridded yield data.
            Requires coordinates 'lon', 'lat', and 'crop'.
        filename_area : str
            Name of netcdf-file containing gridded cultivated area.
            Requires coordinates 'lon', 'lat', and 'crop'.
        var_yield : str
             variable name to be extracted from yield file, e.g. 'yield.rf',
             'yield.ir', 'yield.tot', or depending on netcdf structure.
        var_area : str
             variable name to be extracted from area file,
             e.g. 'cultivated area rainfed', 'cultivated area irrigated',
             'cultivated area all', or depending on netcdf structure.
        bbox (tuple of four floats): bounding box:
             bounding box to be extracted: (lon min, lat min, lon max, lat max).
             The default is (-180, -85, 180, 85).
        input_dir : Path, optional
             directory where input data is found. The default is
             {CONFIG.exposures.crop_production.local_data}/Input/Exposure.
        """
        if isinstance(input_dir, str):
            input_dir = Path(input_dir)
        [lonmin, latmin, lonmax, latmax] = bbox

        # extract yield data to xarray.DataArray:
        data_set_tmp = xr.open_dataset(input_dir / filename_yield, decode_times=False)
        data_yield = data_set_tmp.sel(lon=slice(lonmin, lonmax),
                                      lat=slice(latmax, latmin),
                                      crop=layer_yield
                                      )[var_yield]
        # extract cultivated area data to xarray.DataArray:
        data_set_tmp = xr.open_dataset(input_dir / filename_area, decode_times=False)
        data_area = data_set_tmp.sel(lon=slice(lonmin, lonmax),
                                     lat=slice(latmax, latmin),
                                     crop=layer_area
                                     )[var_area]
        del data_set_tmp

        # The latitude and longitude are set; region_id is determined
        lon, lat = np.meshgrid(data_area.lon.values, data_area.lat.values)

        # initiate coordinates and values in GeoDatFrame:
        self.gdf['latitude'] = lat.flatten()
        self.gdf['longitude'] = lon.flatten()
        self.gdf['region_id'] = u_coord.get_country_code(self.gdf.latitude,
                                                         self.gdf.longitude)
        self.gdf[INDICATOR_IMPF + DEF_HAZ_TYPE] = 1
        self.gdf[INDICATOR_IMPF] = 1
        # calc annual crop production, [t/y] = [ha] * [t/ha/y]:
        self.gdf['value'] = np.multiply(data_area.values, data_yield.values).flatten()

        self.crop = crop_type
        self.tag = Tag()
        self.tag.description = ("Annual crop production from " + var_area +
                                " and " + var_yield + " for " + self.crop +
                                " from files " + filename_area + " and " +
                                filename_yield)
        self.value_unit = 't/y'
        try:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (self.gdf.longitude.min(), self.gdf.latitude.min(),
                 self.gdf.longitude.max(), self.gdf.latitude.max()),
                u_coord.get_resolution(self.gdf.longitude, self.gdf.latitude))
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


    def set_from_spam_ray_mirca(self, crop_type, irrigation_type='all',
                                bbox=BBOX, input_dir=INPUT_DIR):
        """
        Wrapper method around set_from_area_and_yield_nc4().

        Set crop_production exposure from cultivated area [ha] and
        yield [t/ha/year] provided in default input files.
        The default input files are based on the public yield data from
        SPAM2005 with gaps filled based on Ray et.al (2012); and cultivated area
        from MIRCA2000, both as post-processed by Jägermeyr et al. 2020; See
        https://doi.org/10.1073/pnas.1919049117 for more information and cite
        when using this data for publication.

        Parameters
        ----------
        crop_type : str
            Crop type, e.g. 'mai' for maize, or 'ric', 'whe', 'soy', etc.
        irrigation_type : str, optional
            irrigation type to be extracted, the options are:
            'all' : total crop production, i.e. irrigated + rainfed
            'firr' : fully irrigated
            'noirr' : not irrigated, i.e., rainfed
            The default is 'all'
        bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
        input_dir : Path, optional
            directory where input data is found. The default is
            {CONFIG.exposures.crop_production.local_data}/Input/Exposure.
        """
        filename_yield = 'spam_ray_yields.nc4'
        filename_area = 'cultivated_area_MIRCA_GGCMI.nc4'

        # crop layers and variable names in default input files:
        layers_yield = {'mai': 1, 'whe': 2, 'soy': 4, 'ric': 3}
        layers_area = {'mai': 1, 'whe': 2, 'soy': 3, 'ric': 4}
        # Note: layer numbers fo rice and soybean differ between input files.
        varnames_yield = {'noirr': 'yield.rf',
                         'firr': 'yield.ir',
                         'all': 'yield.tot'}
        varnames_area = {'noirr': 'cultivated area rainfed',
                         'firr': 'cultivated area irrigated',
                         'all': 'cultivated area all'}

        # set exposure from netcdf files:
        self.set_from_area_and_yield_nc4(crop_type, layers_yield[crop_type],
                                         layers_area[crop_type],
                                         filename_yield, filename_area,
                                         varnames_yield[irrigation_type],
                                         varnames_area[irrigation_type],
                                         bbox=bbox, input_dir=input_dir)

    def set_mean_of_several_isimip_models(self, input_dir=None, hist_mean=None, bbox=None,
                                          yearrange=None, cl_model=None, scenario=None,
                                          crop=None, irr=None, isimip_version=None,
                                          unit=None, fn_str_var=None):
        """Wrapper to fill exposure from several NetCDF files with crop yield data
        from ISIMIP.

        Optional Parameters:
            input_dir (string): path to input data directory
            historic mean (array): historic mean crop production per centroid
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for exposure set,e.g., (1976, 2005)
            scenario (string): climate change and socio economic scenario
               e.g., 'histsoc' or 'rcp60soc'
            cl_model (string): abbrev. climate model (only when landuse data
            is future projection)
               e.g., 'gfdl-esm2m' etc.
            crop (string): crop type
               e.g., 'mai', 'ric', 'whe', 'soy'
            irr (string): irrigation type
                f.i 'rainfed', 'irrigated' or 'combined'= rainfed+irrigated
            isimip_version(str): 'ISIMIP2' (default) or 'ISIMIP3'
            unit (string): unit of the exposure (per year)
                f.i 't/y' (default), 'USD/y', or 'kcal/y'
            fn_str_var (string): FileName STRing depending on VARiable and
                ISIMIP simuation round
        Returns:
            Exposure
        """
        if not bbox:
            bbox = BBOX
        if (not isimip_version) or (isimip_version in ('ISIMIP2a', 'ISIMIP2b')):
            isimip_version = 'ISIMIP2'
        elif isimip_version in ('ISIMIP3a', 'ISIMIP3b'):
            isimip_version = 'ISIMIP3'
        if not input_dir:
            input_dir = INPUT_DIR
        input_dir = Path(input_dir)
        if not hist_mean:
            hist_mean = HIST_MEAN_PATH
        if yearrange is None:
            yearrange = YEARCHUNKS[isimip_version]['histsoc']['yearrange']
        if not unit:
            unit = 't/y'
        if not fn_str_var:
            fn_str_var = FN_STR_VAR
        filenames = dict()
        filenames['all'] = [f for f in Path(input_dir).iterdir()
                            if f.is_file() and 'nc' in f.name and not f.name.startswith('.')]

        # If only files with a certain scenario and or cl_model shall be considered, they
        # are extracted from the original list of files
        filenames['subset'] = list()
        for name in filenames['all']:
            if cl_model is not None and scenario is not None:
                if cl_model in name or scenario in name:
                    filenames['subset'].append(name)
            elif cl_model is not None and scenario is None:
                if cl_model in name:
                    filenames['subset'].append(name)
            elif cl_model is None and scenario is not None:
                if scenario in name:
                    filenames['subset'].append(name)
            else:
                filenames['subset'] = filenames['all']

        # The first exposure is calculate to determine its size
        # and initialize the combined exposure
        self.set_from_isimip_netcdf(input_dir, filename=filenames['subset'][0],
                                    hist_mean=hist_mean, bbox=bbox, yearrange=yearrange,
                                    crop=crop, irr=irr, isimip_version=isimip_version,
                                    unit=unit, fn_str_var=fn_str_var)

        combined_exp = np.zeros([self.gdf.value.size, len(filenames['subset'])])
        combined_exp[:, 0] = self.gdf.value

        # The calculations are repeated for all remaining exposures (starting from index 1 as
        # the first exposure has been saved in combined_exp[:, 0])
        for j, fn in enumerate(filenames['subset'][1:]):
            self.set_from_isimip_netcdf(input_dir, filename=fn, hist_mean=hist_mean,
                                        bbox=bbox, yearrange=yearrange,
                                        crop=crop, irr=irr, unit=unit,
                                        isimip_version=isimip_version)
            combined_exp[:, j+1] = self.gdf.value

        self.gdf.value = np.mean(combined_exp, 1)
        self.gdf['crop'] = crop

        self.check()

        return self

    def set_value_to_kcal(self, biomass=True):
        """Converts the exposure value from tonnes to kcalper year using
        conversion factor per crop type.

        Optional Parameter:
            biomass (bool): if true, KCAL_PER_TON['biomass'] is used (default,
                for FAO normalized crop production). If False, KCAL_PER_TON['drymatter']
                is used (best for crop model output in dry matter, default for
                raw crop model output)

        Returns:
            Exposure with unit kcal/y
        """
        if self.value_unit != 't/y':
            LOGGER.warning('self.unit is not t/y.')
        self.gdf['tonnes_per_year'] = self.gdf['value'].values
        if biomass:
            self.gdf.value *= KCAL_PER_TON['biomass'][self.crop]
        else:
            self.gdf.value *= KCAL_PER_TON['drymatter'][self.crop]

        self.value_unit = 'kcal/y'
        return self

    def set_value_to_usd(self, input_dir=None, yearrange=None):
        # to do: check api availability?; default yearrange for single year (e.g. 5a)
        """Calculates the exposure in USD using country and year specific data published
        by the FAO.

        Optional Parameters:
            input_dir (Path or str): directory containing the input (FAO pricing) data,
                default: {CONFIG.exposures.crop_production.local_data}/Input/Exposure
            yearrange (array): year range for prices, can also be set to a single year
                Default is set to the arbitrary time range (2000, 2018)
                The data is available for the years 1991-2018
            crop (str): crop type
               e.g., 'mai', 'ric', 'whe', 'soy'

        Returns:
            Exposure
        """
        if not input_dir:
            input_dir = INPUT_DIR
        input_dir = Path(input_dir)
        if yearrange is None:
            yearrange = YEARS_FAO
        # the exposure in t/y is saved as 'tonnes_per_year'
        self.gdf['tonnes_per_year'] = self.gdf['value'].values

        # account for the case of only specifying one year as yearrange
        if len(yearrange) == 1:
            yearrange = (yearrange[0], yearrange[0])

        # open both FAO files and extract needed variables
        # FAO_FILE: contains producer prices per crop, country and year
        fao = dict()
        fao['file'] = pd.read_csv(input_dir / FAO_FILE)
        fao['crops'] = fao['file'].Item.values
        fao['year'] = fao['file'].Year.values
        fao['price'] = fao['file'].Value.values

        fao_country = u_coord.country_faocode2iso(getattr(fao['file'], 'Area Code').values)

        # create a list of the countries contained in the exposure
        iso3alpha = list()
        self.gdf.region_id[self.gdf.region_id == -99] = 0
        iso3alpha = np.asarray(u_coord.country_to_iso(
            self.gdf.region_id, representation="alpha3", fillvalue='Other country'), dtype=object)
        iso3alpha[iso3alpha == ""] = 'No country'
        list_countries = np.unique(iso3alpha)

        # iterate over all countries that are covered in the exposure, extract the according price
        # and calculate the crop production in USD/y
        area_price = np.zeros(self.gdf.value.size)
        for country in list_countries:
            [idx_country] = (iso3alpha == country).nonzero()
            if country == 'Other country':
                price = 0
                area_price[idx_country] = self.gdf.value[idx_country] * price
            elif country != 'No country' and country != 'Other country':
                idx_price = np.where((np.asarray(fao_country) == country) &
                                     (np.asarray(fao['crops']) == \
                                     (CROP_NAME[self.crop])['fao']) &
                                     (fao['year'] >= yearrange[0]) &
                                     (fao['year'] <= yearrange[1]))
                price = np.mean(fao['price'][idx_price])
                # if no price can be determined for a specific yearrange and country, the world
                # average for that crop (in the specified yearrange) is used
                if math.isnan(price) or price == 0:
                    idx_price = np.where((np.asarray(fao['crops']) == \
                                          (CROP_NAME[self.crop])['fao']) &
                                         (fao['year'] >= yearrange[0]) &
                                         (fao['year'] <= yearrange[1]))
                    price = np.mean(fao['price'][idx_price])
                area_price[idx_country] = self.gdf.value[idx_country] * price


        self.gdf['value'] = area_price
        self.value_unit = 'USD/y'
        self.check()
        return self

    def aggregate_countries(self):
        """Aggregate exposure data by country.

        Returns:
            list_countries (list): country codes (numerical ISO3)
            country_values (array): aggregated exposure value
        """

        list_countries = np.unique(self.gdf.region_id)
        country_values = np.zeros(len(list_countries))
        for i, iso_nr in enumerate(list_countries):
            country_values[i] = self.gdf.loc[self.gdf.region_id == iso_nr].value.sum()

        return list_countries, country_values

def init_full_exp_set_isimip(input_dir=None, filename=None, hist_mean_dir=None,
                             output_dir=None, bbox=None, yearrange=None, unit=None,
                             isimip_version=None, return_data=False):
    """Generates CropProduction instances (exposure sets) for all files found in the
        input directory and saves them as hdf5 files in the output directory.
        Exposures are aggregated per crop and irrigation type.

        Parameters:
        input_dir (str or Path): path to input data directory,
            default: {CONFIG.exposures.crop_production.local_data}/Input/Exposure
        filename (string): if not specified differently, the file
            'histsoc_landuse-15crops_annual_1861_2005.nc' will be used
        output_dir (string): path to output data directory
        bbox (list of four floats): bounding box:
            [lon min, lat min, lon max, lat max]
        yearrange (array): year range for hazard set, e.g., (1976, 2005)
        isimip_version(str): 'ISIMIP2' (default) or 'ISIMIP3'
        unit (str): unit in which to return exposure (e.g., t/y or USD/y)
        return_data (boolean): returned output
            False: returns list of filenames only, True: returns also list of data

    Returns:
        filename_list (list): all filenames of saved initiated exposure files
        output_list (list): list containing all inisiated Exposure instances
    """
    if not bbox:
        bbox = BBOX
    if (not isimip_version) or (isimip_version in ('ISIMIP2a', 'ISIMIP2b')):
        isimip_version = 'ISIMIP2'
    elif isimip_version in ('ISIMIP3a', 'ISIMIP3b'):
        isimip_version = 'ISIMIP3'
    if not input_dir:
        input_dir = INPUT_DIR
    input_dir = Path(input_dir)
    if not hist_mean_dir:
        hist_mean_dir = HIST_MEAN_PATH
    hist_mean_dir = Path(hist_mean_dir)
    if not output_dir:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    if yearrange is None:
        yearrange = YEARCHUNKS[isimip_version]['histsoc']['yearrange']
    if not unit:
        unit = 't/y'

    filenames = [f.name for f in hist_mean_dir.iterdir()
                 if f.is_file() and not f.name.startswith('.')]

    # generate output directory if it does not exist yet
    target_dir = output_dir / 'Exposure'
    target_dir.mkdir(exist_ok=True)

    # create exposures for all crop-irrigation combinations and save them
    filename_list = list()
    output_list = list()
    for file in filenames:
        _, _, crop_irr, *_ = file.split('_')
        crop, irr = crop_irr.split('-')
        crop_production = CropProduction()
        crop_production.set_from_isimip_netcdf(input_dir=input_dir, filename=filename,
                                               hist_mean=hist_mean_dir, bbox=bbox,
                                               isimip_version=isimip_version,
                                               yearrange=yearrange, crop=crop, irr=irr, unit=unit)
        filename_expo = ('crop_production_' + crop + '-'+ irr + '_'
                         + str(yearrange[0]) + '-' + str(yearrange[1]) + '.hdf5')
        filename_list.append(filename_expo)
        crop_production.write_hdf5(str(Path(target_dir, filename_expo)))
        if return_data:
            output_list.append(crop_production)

    return filename_list, output_list

def normalize_with_fao_cp(exp_firr, exp_noirr, input_dir=None,
                          yearrange=None, unit=None, return_data=True):
    """Normalize (i.e., bias correct) the given exposures countrywise with the mean
    crop production quantity documented by the FAO.
    Refer to the beginning of the script for guidance on where to download the
    required crop production data from FAO.Stat.

    Parameters:
        exp_firr (crop_production): exposure under full irrigation
        exp_noirr (crop_production): exposure under no irrigation

    Optional Parameters:
        input_dir (Path or str): directory containing exposure input data,
            default: {CONFIG.exposures.crop_production.local_data}/Input/Exposure
        yearrange (array): the mean crop production in this year range is used to normalize
            the exposure data
            Default is set to the arbitrary time range (2008, 2018)
            The data is available for the years 1961-2018
        unit (str): unit in which to return exposure (t/y or USD/y)
        return_data (boolean): returned output
            True: returns country list, ratio = FAO/ISIMIP, normalized exposures, crop production
            per country as documented by the FAO and calculated by the ISIMIP dataset
            False: country list, ratio = FAO/ISIMIP, normalized exposures

    Returns:
        country_list (list): List of country codes (numerical ISO3)
        ratio (list): List of ratio of FAO crop production and aggregated exposure
            for each country
        exp_firr_norm (CropProduction): Normalized CropProduction (full irrigation)
        exp_noirr_norm (CropProduction): Normalized CropProduction (no irrigation)

    Returns (optional):
        fao_crop_production (list): FAO crop production value per country
        exp_tot_production(list): Exposure crop production value per country
            (before normalization)
    """
    if not input_dir:
        input_dir = INPUT_DIR
    input_dir = Path(input_dir)
    if yearrange is None:
        yearrange = YEARS_FAO
    if not unit:
        unit = 't/y'
    # if the exposure unit is USD/y or kcal/y, temporarily reset the exposure to t/y
    # (stored in tonnes_per_year) in order to normalize with FAO crop production
    # values and then apply set_to_XXX() for the normalized exposure to restore the
    # initial exposure unit
    if exp_firr.value_unit == 'USD/y' or 'kcal' in exp_firr.value_unit:
        exp_firr.gdf.value = exp_firr.tonnes_per_year
    if exp_noirr.value_unit == 'USD/y' or 'kcal' in exp_noirr.value_unit:
        exp_noirr.gdf.value = exp_noirr.tonnes_per_year

    country_list, countries_firr = exp_firr.aggregate_countries()
    country_list, countries_noirr = exp_noirr.aggregate_countries()

    exp_tot_production = countries_firr + countries_noirr

    fao = pd.read_csv(input_dir / FAO_FILE2)
    fao_crops = fao.Item.values
    fao_year = fao.Year.values
    fao_values = fao.Value.values
    fao_code = getattr(fao, 'Area Code').values

    fao_country = u_coord.country_iso2faocode(country_list)

    fao_crop_production = np.zeros(len(country_list))
    ratio = np.ones(len(country_list))
    exp_firr_norm = exp_firr.copy(deep=True)
    exp_noirr_norm = exp_noirr.copy(deep=True)

    # loop over countries: compute ratio & apply normalization:
    for country, iso_nr in enumerate(country_list):
        idx = np.where((np.asarray(fao_code) == fao_country[country])
                       & (np.asarray(fao_crops) == (CROP_NAME[exp_firr.crop])['fao'])
                       & (fao_year >= yearrange[0]) & (fao_year <= yearrange[1]))
        if len(idx) >= 1:
            fao_crop_production[country] = np.mean(fao_values[idx])

        # if a country has no values in the exposure (e.g. Cyprus) the exposure value
        # is set to the FAO average value
        # in this case the ratio is left being 1 (as initiated)
        if exp_tot_production[country] == 0:
            exp_tot_production[country] = fao_crop_production[country]
        elif fao_crop_production[country] != np.nan and fao_crop_production[country] != 0:
            ratio[country] = fao_crop_production[country] / exp_tot_production[country]

        exp_firr_norm.gdf.value[exp_firr.gdf.region_id == iso_nr] = ratio[country] * \
        exp_firr.gdf.value[exp_firr.gdf.region_id == iso_nr]
        exp_noirr_norm.gdf.value[exp_firr.gdf.region_id == iso_nr] = ratio[country] * \
        exp_noirr.gdf.value[exp_noirr.gdf.region_id == iso_nr]

        if unit == 'USD/y' or exp_noirr.value_unit == 'USD/y':
            exp_noirr.set_value_to_usd(input_dir=input_dir)
        elif 'kcal' in unit or 'kcal' in exp_noirr.value_unit:
            exp_noirr.set_value_to_kcal(biomass=True)
            # FAO production is provided in biomass, not dry matter
        if unit == 'USD/y' or exp_firr.value_unit == 'USD/y':
            exp_firr.set_value_to_usd(input_dir=input_dir)
        elif 'kcal' in unit or 'kcal' in exp_firr.value_unit:
            exp_firr.set_value_to_kcal(biomass=True)

    exp_firr_norm.tag.description = exp_firr_norm.tag.description+' normalized'
    exp_noirr_norm.tag.description = exp_noirr_norm.tag.description+' normalized'

    if return_data:
        return country_list, ratio, exp_firr_norm, exp_noirr_norm, \
            fao_crop_production, exp_tot_production
    return country_list, ratio, exp_firr_norm, exp_noirr_norm

def normalize_several_exp(input_dir=None, output_dir=None,
                          yearrange=None, unit=None, return_data=True):
    """
    Multiple exposure sets saved as HDF5 files in input directory are normalized
    (i.e. bias corrected) against FAO statistics of crop production.
        Optional Parameters:
            input_dir (Path or str): directory containing exposure input data
            output_dir (Path or str): directory containing exposure datasets (output of
                                                                              exposure creation)
            yearrange (array): the mean crop production in this year range is used to normalize
                the exposure data (default 2008-2018)
            unit (str): unit in which to return exposure (t/y or USD/y)
            return_data (boolean): returned output
                True: lists containing data for each exposure file. Lists: crops, country list,
                    ratio = FAO/ISIMIP, normalized exposures, crop production
                    per country as documented by the FAO and calculated by the ISIMIP dataset
                False: lists containing data for each exposure file. Lists: crops, country list,
                    ratio = FAO/ISIMIP, normalized exposures

        Returns:
            crop_list (list): List of crops
            country_list (list): List of country codes (numerical ISO3)
            ratio (list): List of ratio of FAO crop production and aggregated exposure
                for each country
            exp_firr_norm (list): List of normalized CropProduction Exposures (full irrigation)
            exp_noirr_norm (list): List of normalize CropProduction Exposures (no irrigation)

        Returns (optional):
            fao_crop_production (list): FAO crop production value per country
            exp_tot_production(list): Exposure crop production value per country
                (before normalization)
    """
    if input_dir is None:
        input_dir = INPUT_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    if not unit:
        unit = 't/y'
    if yearrange is None:
        yearrange = YEARS_FAO

    filenames_firr = [f.parts[-1] for f in (output_dir / 'Exposure').iterdir() if
                      f.is_file() if not f.parts[-1].startswith('.') if
                      'firr' in f.parts[-1]]

    crop_list = list()
    countries_list = list()
    ratio_list = list()
    exp_firr_norm = list()
    exp_noirr_norm = list()
    fao_cp_list = list()
    exp_tot_cp_list = list()

    for file_firr in filenames_firr:
        _, _, crop_irr, years = file_firr.split('_')
        crop, _ = crop_irr.split('-')
        exp_firr = CropProduction()
        exp_firr.read_hdf5(str(Path(output_dir, 'Exposure', file_firr)))

        filename_noirr = 'crop_production_' + crop + '-' + 'noirr' + '_' + years
        exp_noirr = CropProduction()
        exp_noirr.read_hdf5(str(Path(output_dir, 'Exposure', filename_noirr)))

        if return_data:
            countries, ratio, exp_firr2, exp_noirr2, fao_cp, \
            exp_tot_cp = normalize_with_fao_cp(exp_firr, exp_noirr, input_dir=input_dir,
                                               yearrange=yearrange, unit=unit)
            fao_cp_list.append(fao_cp)
            exp_tot_cp_list.append(exp_tot_cp)
        else:
            countries, ratio, exp_firr2, exp_noirr2 = normalize_with_fao_cp(
                exp_firr, exp_noirr, input_dir=input_dir,
                yearrange=yearrange, unit=unit, return_data=False)

        crop_list.append(crop)
        countries_list.append(countries)
        ratio_list.append(ratio)
        exp_firr_norm.append(exp_firr2)
        exp_noirr_norm.append(exp_noirr2)

    if return_data:
        return crop_list, countries_list, ratio_list, exp_firr_norm, exp_noirr_norm, \
                fao_cp_list, exp_tot_cp_list
    return crop_list, countries_list, ratio_list, exp_firr_norm, exp_noirr_norm

def semilogplot_ratio(crop, countries, ratio, output_dir=None, save=True):
    """Plot ratio = FAO/ISIMIP against country codes.

        Parameters:
            crop (str): crop to plot
            countries (list): country codes of countries to plot
            ratio (array): ratio = FAO/ISIMIP crop production data of countries to plot
        Optional Parameters:
            save (boolean): True saves figure, else figure is not saved.
            output_dir (str): directory to save figure
        Returns:
            fig (plt figure handle)
            axes (plot axes handle)

    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    fig = plt.figure()
    axes = plt.gca()
    axes.scatter(countries[ratio != 1], ratio[ratio != 1])
    axes.set_yscale('log')
    axes.set_ylabel('Ratio= FAO / ISIMIP')
    axes.set_xlabel('ISO3 country code')
    axes.set_ylim(np.nanmin(ratio), np.nanmax(ratio))
    plt.title(crop)

    if save:
        target_dir = output_dir / 'Exposure_norm_plots'
        target_dir.mkdir(exist_ok=True)
        plt.savefig(target_dir / 'fig_ratio_norm_' + crop)
    return fig, axes
