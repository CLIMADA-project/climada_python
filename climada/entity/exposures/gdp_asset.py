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
import os
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import logging
import geopandas as gpd
from climada.entity.exposures.base import Exposures, INDICATOR_IF
from climada.util.coordinates import pts_to_raster_meta
from climada.util.constants import GLB_CENTROIDS_NC
from climada.util.constants import NAT_REG_ID, SYSTEM_DIR
from climada.util.constants import DEMO_GDP2ASSET
from climada.util.constants import DEF_CRS
from climada.entity.tag import Tag
LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'RF'

CONVERTER = os.path.join(SYSTEM_DIR, 'GDP2Asset_converter_2.5arcmin.nc')


class GDP2Asset(Exposures):

    @property
    def _constructor(self):
        return GDP2Asset

    def set_countries(self, countries=[], reg=[], ref_year=2000,
                      path=None):
        """ Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list): list of country names ISO3
            ref_year (int, optional): reference year. Default: 2016
            path (string): path to exposure dataset (ISIMIP)
        """
        gdp2a_list = []
        try:

            if not countries:
                if reg:
                    natID_info = pd.read_csv(NAT_REG_ID)
                    natISO = natID_info["ISO"][np.isin(natID_info["Reg_name"],
                                                       reg)]
                    countries = np.array(natISO)
                else:
                    LOGGER.error('set_countries requires countries or reg')
                    raise ValueError

            for cntr_ind in range(len(countries)):
                gdp2a_list.append(self._set_one_country(countries[cntr_ind],
                                                        ref_year, path))
            Exposures.__init__(self, gpd.GeoDataFrame(
                        pd.concat(gdp2a_list, ignore_index=True)))
        except KeyError:
            LOGGER.error('Exposure countries: ' + str(countries) + ' or reg ' +
                         str(reg) + ' could not be set, check ISO3 or' +
                         ' reference year ' + str(ref_year))
            raise KeyError
        self.ref_year = ref_year
        self.value_unit = 'USD'
        self.tag = Tag()
        self.tag.description = 'GDP2Asset ' + str(self.ref_year)
        self.crs = DEF_CRS
        # set meta
        res = 0.0416666
        rows, cols, ras_trans = pts_to_raster_meta((self.longitude.min(),
                                                    self.latitude.min(),
                                                    self.longitude.max(),
                                                    self.latitude.max()), res)
        self.meta = {'width': cols, 'height': rows, 'crs': self.crs,
                     'transform': ras_trans}

    @staticmethod
    def _set_one_country(countryISO, ref_year, path=None):
        """ Extract coordinates of selected countries or region
        from NatID grid.
        Parameters:
            countryISO(str): ISO3 of country
            ref_year(int): year under consideration
            path(str): path for gdp-files
        Raises:
            KeyError, OSError
        Returns:
            np.array
        """
        exp_gdpasset = GDP2Asset()
        natID_info = pd.read_csv(NAT_REG_ID)
        try:
            isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
            isimip_lon = isimip_grid.lon.data
            isimip_lat = isimip_grid.lat.data
            gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
            if not any(np.isin(natID_info['ISO'], countryISO)):
                LOGGER.error('Wrong country ISO ' + str(countryISO))
                raise KeyError
            natID = natID_info['ID'][np.isin(natID_info['ISO'], countryISO)]
            reg_id, if_rf = _fast_if_mapping(natID, natID_info)
            isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        except OSError:
            LOGGER.error('Problems while reading ,' + path +
                         ' check exposure_file specifications')
            raise OSError
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        coord = np.zeros((len(lon_coordinates), 2))
        coord[:, 1] = lon_coordinates
        coord[:, 0] = lat_coordinates
        assets = _read_GDP(coord, ref_year, path)
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


def _read_GDP(shp_exposures, ref_year, path=None):
    """ Read GDP-values for the selected area and convert it to asset.
        Parameters:
            shp_exposure(2d-array float): coordinates of area
            ref_year(int): year under consideration
            path(str): path for gdp-files
        Raises:
            KeyError, OSError
        Returns:
            np.array
        """
    try:
        gdp_file = xr.open_dataset(path)
        asset_converter = xr.open_dataset(CONVERTER)
        gdp_lon = gdp_file.lon.data
        gdp_lat = gdp_file.lat.data
        time = gdp_file.time.dt.year
    except OSError:
        LOGGER.error('Problems while reading ,' + path +
                     ' check exposure_file specifications')
        raise OSError
    try:
        year_index = np.where(time == ref_year)[0][0]
    except IndexError:
        LOGGER.error('No data available for year ' + str(ref_year))
        raise KeyError
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
    """ Assign region-ID and impact function id.
        Parameters:
            countryID (int)
            natID_info: dataframe of lookuptable
        Raises:
            KeyError
        Returns:
            float,float
        """
    nat = natID_info['ID']
    if_RF = natID_info['if_RF']
    reg_ID = natID_info['Reg_ID']
    fancy_if = np.zeros((max(nat) + 1))
    fancy_if[:] = np.nan
    fancy_if[nat] = if_RF
    fancy_reg = np.zeros((max(nat) + 1))
    fancy_reg[:] = np.nan
    fancy_reg[nat] = reg_ID
    try:
        reg_id = fancy_reg[countryID]
        if_rf = fancy_if[countryID]
    except KeyError:
        LOGGER.error('Country ISO unknown')
        raise KeyError
    return reg_id, if_rf
