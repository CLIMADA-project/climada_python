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

Define GDPAsset class.
"""

__all__ = ['GDP2Asset']

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import math
import geopandas as gpd
from climada.entity.exposures.base import Exposures, INDICATOR_IF
from climada.util.constants import GLB_CENTROIDS_NC
from climada.util.constants import NAT_REG_ID, NAT_REG_ID
from climada.util.interpolation import interpol_index

DEF_HAZ_TYPE = 'RF'

PATH = '/home/insauer/data/Tobias/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'
CONVERTER = '/home/insauer/data/Tobias/GDP2Asset_converter_2.5arcmin.nc'

REGION_MAP = NAT_REG_ID


class GDP2Asset(Exposures):

    @property
    def _constructor(self):
        return GDP2Asset

    def set_countries(self, countries=[], reg=[], ref_year=2016):
        """ Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list): list of country names ISO3
            ref_year (int, optional): reference year. Default: 2016
            res_km (float, optional): approx resolution in km. Default:
                nightlights resolution.
        """
        """TODO region selection"""
        gdp2a_list = []
        for cntr_ind in range(len(countries)):
            gdp2a_list.append(self._set_one_country(countries[cntr_ind], ref_year))
        Exposures.__init__(self, gpd.GeoDataFrame(pd.concat(gdp2a_list,
                                                            ignore_index=True)))
        self.ref_year = ref_year
        #self.tag = tag
        #self.tag.file_name = fn_nl
        self.value_unit = 'USD'
        self.crs = {'init': 'epsg:4326'}


    @staticmethod
    def _set_one_country(countryISO, ref_year):
        """ Extract coordinates of selected countries or region
        from NatID grid. If countries are given countries are cut,
        if only reg is given, the whole region is cut.
        Parameters:
            countryISO
        Raises:
            AttributeError
        Returns:
            np.array
        """
        exp_gdpasset = GDP2Asset()
        natID_info = pd.read_csv(REGION_MAP)
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
        natID = natID_info['ID'][np.isin(natID_info['ISO'], countryISO)]
        reg_id, if_rf = _fast_if_mapping(natID, natID_info)
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        coord = np.zeros((len(lon_coordinates), 2))
        coord[:, 1] = lon_coordinates
        coord[:, 0] = lat_coordinates
        assets = _read_GDP(coord, ref_year)
        reg_id_info = np.zeros((len(assets)))
        reg_id_info[:] = reg_id
        if_rf_info = np.zeros((len(assets)))
        if_rf_info[:] = if_rf
        exp_gdpasset['value'] = assets
        exp_gdpasset['latitude'] = coord[:, 0]
        exp_gdpasset['longitude'] = coord[:, 1]
        exp_gdpasset[INDICATOR_IF + DEF_HAZ_TYPE] = if_rf_info
        exp_gdpasset['region_id'] = reg_id_info
        return exp_gdpasset

def _read_GDP(shp_exposures, ref_year):
    gdp_file = xr.open_dataset(PATH)
    asset_converter = xr.open_dataset(CONVERTER)
    gdp_lon = gdp_file.lon.data
    gdp_lat = gdp_file.lat.data
    time = gdp_file.time.dt.year
    year_index = np.where(time == ref_year)[0][0]
    conv_lon = asset_converter.lon.data
    conv_lat = asset_converter.lat.data
    gridX, gridY = np.meshgrid(conv_lon, conv_lat)
    coordinates = np.zeros((gridX.size, 2))
    coordinates[:, 0] = gridY.flatten()
    coordinates[:, 1] = gridX.flatten()
    gdp = gdp_file.gdp_grid[year_index, :, :].data
    conv_factors = asset_converter.conversion_factor.data
    asset = gdp * conv_factors
    asset = sp.interpolate.interpn((gdp_lat, gdp_lon),
                                   np.nan_to_num(asset),
                                   (shp_exposures[:, 0],
                                   shp_exposures[:, 1]),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=None)
    return asset


def _fast_if_mapping(countryID, natID_info):
    nat = natID_info['ID']
    if_RF = natID_info['if_RF']
    reg_ID = natID_info['Reg_ID']
    fancy_if = np.zeros((max(nat) + 1))
    fancy_if[:] = np.nan
    fancy_if[nat] = if_RF
    fancy_reg = np.zeros((max(nat) + 1))
    fancy_reg[:] = np.nan
    fancy_reg[nat] = reg_ID
    reg_id = fancy_reg[countryID]
    if_rf = fancy_if[countryID]
    return reg_id, if_rf


def map_info():
    grid = xr.open_dataset(GLB_CENTROIDS_NC)
    info = pd.read_csv(REGION_MAP)
    nat = info['ID']
    if_RF = info['if_RF']
    reg_ID = info['Reg_ID']
    fancy_array = np.zeros((max(nat) + 1))
    fancy_array[:] = np.nan
    fancy_array[nat] = if_RF

    natId = np.nan_to_num(grid.NatIdGrid.data)
    natShape = natId.shape
    natId = natId.astype(int).flatten()
    transla = fancy_array[natId]
    #print(list(np.unique(transla)[0:8]))
    return transla

def assign_reg(Id):
    info = pd.read_csv(REGION_MAP)
    return info.loc[info["ID"] == Id, "Reg_ID"]


def assign_if(Id, transl):
    return transl[Id]
