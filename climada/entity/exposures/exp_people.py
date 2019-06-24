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

Define ExpPop class.
"""

__all__ = ['ExpPop']
import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp
import logging
import geopandas as gpd
from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, INDICATOR_IF
from climada.util.constants import GLB_CENTROIDS_NC
from climada.util.constants import NAT_REG_ID
from climada.util.constants import DEF_CRS
LOGGER = logging.getLogger(__name__)

DEF_HAZ_TYPE = 'RF'

PEOPLE_DATASET = ('/home/insauer/data/Tobias/' +
                  'hyde_ssp2_1980-2015_0150as_yearly_zip.nc4')


class ExpPop(Exposures):

    @property
    def _constructor(self):
        return ExpPop

    def set_countries(self, countries=[], reg=[], ref_year=2000,
                      path=PEOPLE_DATASET):
        """ Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list): list of country names ISO3
            ref_year (int, optional): reference year. Default: 2016
            path (string): path to exposure dataset
        """
        gdp2a_list = []
        tag = Tag()
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
                tag.description += ("{} Exposed Population\n").\
                    format(countries[cntr_ind])
            Exposures.__init__(self, gpd.GeoDataFrame(
                        pd.concat(gdp2a_list, ignore_index=True)))
        except KeyError:
            LOGGER.error('Exposure countries: ' + str(countries) + ' or reg ' +
                         str(reg) + ' could not be set, check ISO3 or' +
                         ' reference year ' + str(ref_year))
            raise KeyError
        self.ref_year = ref_year
        self.value_unit = 'people'
        self.tag = tag
        self.crs = DEF_CRS

    @staticmethod
    def _set_one_country(countryISO, ref_year, path=PEOPLE_DATASET):
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
        exp_people = ExpPop()
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
            isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        except OSError:
            LOGGER.error('Problems while reading ,' + path +
                         ' check exposure_file specifications')
            raise OSError

        reg_id = natID_info.loc[natID_info['ISO'] == countryISO, 'Reg_ID']
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        coord = np.zeros((len(lon_coordinates), 2))
        coord[:, 1] = lon_coordinates
        coord[:, 0] = lat_coordinates
        assets = _read_people(coord, ref_year, path)
        reg_id_info = np.zeros((len(assets)))
        nat_id_info = np.zeros((len(assets)))
        reg_id_info[:] = reg_id
        nat_id_info[:] = natID
        if_rf_info = np.zeros((len(assets)))
        if_rf_info[:] = 7
        exp_people['value'] = assets
        exp_people['latitude'] = coord[:, 0]
        exp_people['longitude'] = coord[:, 1]
        exp_people[INDICATOR_IF + DEF_HAZ_TYPE] = if_rf_info
        exp_people['region_id'] = reg_id_info
        exp_people['country_id'] = nat_id_info
        return exp_people


def _read_people(shp_exposures, ref_year, path=PEOPLE_DATASET):
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
    ref_year = ref_year-1860
    try:
        pop_file = xr.open_dataset(filename_or_obj=path, decode_times=False)
        pop_lon = pop_file.lon.data
        pop_lat = pop_file.lat.data
        time = pop_file.time
        print(time)
    except OSError:
        LOGGER.error('Problems while reading ,' + path +
                     ' check exposure_file specifications')
        raise OSError
    try:
        year_index = np.where(time == ref_year)[0][0]
    except IndexError:
        LOGGER.error('No data available for year ' + str(1860 + ref_year))
        raise KeyError

    pop = pop_file.var1[year_index, :, :].data
    population = sp.interpolate.interpn((pop_lat, pop_lon),
                                        np.nan_to_num(pop),
                                        (shp_exposures[:, 0],
                                         shp_exposures[:, 1]),
                                        method='nearest',
                                        bounds_error=False,
                                        fill_value=None)
    return population
