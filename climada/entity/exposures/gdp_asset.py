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

__all__ = ['GDPAsset']

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from climada.entity.exposures.base import Exposures
from climada.util.constants import GLB_CENTROIDS_NC
from climada.util.constants import NAT_REG_ID
from climada.util.interpolation import interpol_index

PATH = '/home/insauer/data/Tobias/gdp_1850-2100_5min_yearly_final.nc'
CONVERTER = '/home/insauer/data/Tobias/GDP2Asset_converter_5arcmin_final.nc'


class GDPAsset(Exposures):

    @property
    def _constructor(self):
        return GDPAsset

    def set_countries(self, countries, reg=[], ref_year=2016, res_km=None):
        """ Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list or dict): list of country names (admin0) or dict
                with key = admin0 name and value = [admin1 names]
            ref_year (int, optional): reference year. Default: 2016
            res_km (float, optional): approx resolution in km. Default:
                nightlights resolution.
            from_hr (bool, optional): force to use higher resolution image,
                independently of its year of acquisition.
            kwargs (optional): 'gdp' and 'inc_grp' dictionaries with keys the
                country ISO_alpha3 code. 'poly_val' polynomial transformation
                [1,x,x^2,...] to apply to nightlight (DEF_POLY_VAL used if not
                provided). If provided, these are used.
        """
        shp_exposures = self.select_area(countries, reg)
        latitude, longitude, assets = self.read_GDP(shp_exposures,ref_year)
        exp_bkmrb= GDPAsset()
        exp_bkmrb['value'] = assets
        exp_bkmrb['latitude'] = latitude
        exp_bkmrb['longitude'] = longitude
        Exposures.__init__(self, gpd.GeoDataFrame(exp_bkmrb))

    
    def select_area(self, countries=[], reg=[]):
        """ Extract coordinates of selected countries or region
        from NatID grid. If countries are given countries are cut,
        if only reg is given, the whole region is cut.
        Parameters:
            countries: List of countries
            reg: List of regions
        Raises:
            AttributeError
        Returns:
            np.array
        """
        natID_info = pd.read_excel(NAT_REG_ID)
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)

        if countries:
            natID = natID_info["NatID"][np.isin(natID_info["ISO3"], countries)]
        elif reg:
            natID = natID_info["NatID"][np.isin(natID_info["TCRegName"], reg)]
        else:
            coord = np.zeros((gridX.size, 2))
            coord[:, 1] = gridX.flatten()
            coord[:, 0] = gridY.flatten()
            return coord
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        coord = np.zeros((len(lon_coordinates), 2))
        coord[:, 1] = lon_coordinates
        coord[:, 0] = lat_coordinates
        return coord

    def read_GDP(self, shp_exposures, ref_year):
        gdp_file = xr.open_dataset(PATH)
        asset_converter = xr.open_dataset(CONVERTER)
        gdp_lon = gdp_file.lon.data
        gdp_lat = gdp_file.lat.data
        conv_lon = asset_converter.lon.data
        conv_lat = asset_converter.lat.data
        gridX, gridY = np.meshgrid(conv_lon, conv_lat)
        coordinates = np.zeros((gridX.size, 2))
        coordinates[:, 0] = gridY.flatten()
        coordinates[:, 1] = gridX.flatten()
        gdp = gdp_file.gdp[2, :, :].data
        conv_factors = asset_converter.conversion_factor.data
        asset = gdp * conv_factors
        asset_index = interpol_index(coordinates, shp_exposures)
        #nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
        asset = asset.flatten()
        asset = asset[asset_index]
        latitude = coordinates[asset_index, 0]
        longitude = coordinates[asset_index, 1]
        return latitude, longitude, asset


