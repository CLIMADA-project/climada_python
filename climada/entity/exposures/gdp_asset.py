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

Define GDPAsset class.
"""

__all__ = ['GDP2Asset']
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
from climada.entity.tag import Tag
import climada.util.coordinates as u_coord
from climada.util.constants import RIVER_FLOOD_REGIONS_CSV, SYSTEM_DIR
from .base import Exposures, INDICATOR_IMPF

LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'RF'

CONVERTER = SYSTEM_DIR.joinpath('GDP2Asset_converter_2.5arcmin.nc')


class GDP2Asset(Exposures):

    def set_countries(self, countries=[], reg=[], ref_year=2000,
                      path=None):
        """Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list): list of country names ISO3
            ref_year (int, optional): reference year. Default: 2016
            path (string): path to exposure dataset (ISIMIP)
        """
        gdp2a_list = []
        tag = Tag()

        if path is None:
            raise NameError('No path for exposure data set')

        if not Path(path).is_file():
            raise NameError('Invalid path %s' % path)
        try:

            if not countries:
                if reg:
                    natISO = u_coord.region2isos(reg)
                    countries = np.array(natISO)
                else:
                    raise ValueError('set_countries requires countries or reg')

            for cntr_ind in range(len(countries)):
                gdp2a_list.append(self._set_one_country(countries[cntr_ind],
                                                        ref_year, path))
                tag.description += ("{} GDP2Asset \n").\
                    format(countries[cntr_ind])
        except KeyError as err:
            raise KeyError(f'Exposure countries: {countries} or reg {reg} could not be set, '
                           f'check ISO3 or reference year {ref_year}') from err

        tag.description += 'GDP2Asset ' + str(self.ref_year)
        Exposures.__init__(
            self,
            data=Exposures.concat(gdp2a_list).gdf,
            ref_year=ref_year,
            tag=tag,
            value_unit='USD'
        )

        # set meta
        res = 0.0416666

        rows, cols, ras_trans = u_coord.pts_to_raster_meta(
            (self.gdf.longitude.min(), self.gdf.latitude.min(),
             self.gdf.longitude.max(), self.gdf.latitude.max()), res)
        self.meta = {'width': cols, 'height': rows, 'crs': self.crs,
                     'transform': ras_trans}



    @staticmethod
    def _set_one_country(countryISO, ref_year, path=None):
        """Extract coordinates of selected countries or region
        from NatID grid.
        Parameters:
            countryISO(str): ISO3 of country
            ref_year(int): year under consideration
            path(str): path for gdp-files
        Raises:
            KeyError, OSError
        Returns:
            GDP2Asset
        """
        natID = u_coord.country_iso2natid(countryISO)
        natID_info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
        reg_id, impf_rf = _fast_impf_mapping(natID, natID_info)
        lat, lon = u_coord.get_region_gridpoints(countries=[natID], iso=False, basemap="isimip")
        coord = np.stack([lat, lon], axis=1)
        assets = _read_GDP(coord, ref_year, path)
        reg_id_info = np.full((len(assets),), reg_id)
        impf_rf_info = np.full((len(assets),), impf_rf)

        exp_gdpasset = GDP2Asset()
        exp_gdpasset.gdf['value'] = assets
        exp_gdpasset.gdf['latitude'] = coord[:, 0]
        exp_gdpasset.gdf['longitude'] = coord[:, 1]
        exp_gdpasset.gdf[INDICATOR_IMPF + DEF_HAZ_TYPE] = impf_rf_info
        exp_gdpasset.gdf['region_id'] = reg_id_info
        return exp_gdpasset


def _read_GDP(shp_exposures, ref_year, path=None):
    """Read GDP-values for the selected area and convert it to asset.
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
    except OSError as err:
        raise OSError(f'Problems while reading {path} check exposure_file specifications') from err
    try:
        year_index = np.where(time == ref_year)[0][0]
    except IndexError as err:
        raise KeyError(f'No data available for year {ref_year}') from err
    conv_lon = asset_converter.lon.data
    conv_lat = asset_converter.lat.data
    gridX, gridY = np.meshgrid(conv_lon, conv_lat)
    coordinates = np.zeros((gridX.size, 2))
    coordinates[:, 0] = gridY.flatten()
    coordinates[:, 1] = gridX.flatten()
    gdp = gdp_file.gdp_grid[year_index, :, :].data
    _test_gdp_centr_match(gdp_lat, gdp_lon, shp_exposures)
    conv_factors = asset_converter.conversion_factor.data
    if gdp.shape == conv_factors.shape:
        asset = gdp * conv_factors
        asset = sp.interpolate.interpn((gdp_lat, gdp_lon),
                                       np.nan_to_num(asset),
                                       (shp_exposures[:, 0],
                                        shp_exposures[:, 1]),
                                       method='nearest',
                                       bounds_error=False,
                                       fill_value=None)
    else:
        conv_factors = sp.interpolate.interpn((conv_lat, conv_lon),
                                              np.nan_to_num(conv_factors),
                                              (shp_exposures[:, 0],
                                               shp_exposures[:, 1]),
                                              method='nearest',
                                              bounds_error=False,
                                              fill_value=None)

        gdp = sp.interpolate.interpn((gdp_lat, gdp_lon),
                                     np.nan_to_num(gdp),
                                     (shp_exposures[:, 0],
                                      shp_exposures[:, 1]),
                                     method='nearest',
                                     bounds_error=False,
                                     fill_value=None)
        asset = gdp * conv_factors

    return asset


def _test_gdp_centr_match(gdp_lat, gdp_lon, shp_exposures):

    if (max(gdp_lat) + 0.5 < max(shp_exposures[:, 0])) or\
           (max(gdp_lon) + 0.5 < max(shp_exposures[:, 1])) or\
           (min(gdp_lat) - 0.5 > min(shp_exposures[:, 0])) or\
           (min(gdp_lon) - 0.5 > min(shp_exposures[:, 1])):

        raise IOError('Asset Data does not match selected country')


def _fast_impf_mapping(countryID, natID_info):
    """Assign region-ID and impact function id.
        Parameters:
            countryID (int)
            natID_info: dataframe of lookuptable
        Raises:
            KeyError
        Returns:
            float,float
        """
    nat = natID_info['ID']
    impf_RF = natID_info['impf_RF']
    reg_ID = natID_info['Reg_ID']
    fancy_impf = np.zeros((max(nat) + 1))
    fancy_impf[:] = np.nan
    fancy_impf[nat] = impf_RF
    fancy_reg = np.zeros((max(nat) + 1))
    fancy_reg[:] = np.nan
    fancy_reg[nat] = reg_ID
    try:
        reg_id = fancy_reg[countryID]
        impf_rf = fancy_impf[countryID]
    except KeyError as err:
        raise KeyError(f'Country ISO unknown: {countryID}') from err
    return reg_id, impf_rf
