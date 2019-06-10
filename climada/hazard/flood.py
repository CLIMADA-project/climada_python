"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define RiverFlood class.
"""

__all__ = ['RiverFlood']

import logging
import os
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import math
import datetime as dt
from datetime import date
import geopandas as gpd
from climada.util.constants import NAT_REG_ID, DATA_DIR, GLB_CENTROIDS_NC
from climada.util.interpolation import interpol_index
from scipy import sparse
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from shapely.geometry import Point

from climada.util.alpha_shape import alpha_shape

HAZ_DEMO_FLDDPH = os.path.join(DATA_DIR, 'demo',
                               'flddph_WaterGAP2_miroc5_historical_flopros_gev_picontrol_2000_0.1.nc')
HAZ_DEMO_FLDFRC = os.path.join(DATA_DIR, 'demo',
                               'fldfrc_WaterGAP2_miroc5_historical_flopros_gev_picontrol_2000_0.1.nc')


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RF'
""" Hazard type acronym RiverFlood"""

RF_MODEL = ['ORCHIDEE',
            'H08',
            'LPJmL',
            'MPI-HM',
            'PCR-GLOBWB',
            'WaterGAP2',
            'CLM',
            'JULES-TUC'
            'JULES-UoE',
            'VIC',
            'VEGAS'
            ]
CL_MODEL = ['gfdl-esm2m',
            'hadgem2-es',
            'ipsl-cm5a-lr',
            'miroc5',
            'wfdei',
            'gswp3',
            'princeton',
            'watch'
            ]
SCENARIO = ['',
            'historical',
            'rcp26',
            'rcp60'
            ]

CENTR_HANDLING = ['align', 'full_hazard']
PROT_STD = 'flopros'


class RiverFlood(Hazard):
    """Contains flood events
    Flood intensities are calculated by means of the
    CaMa-Flood global hydrodynamic model

    Attributes:
        fld_area_per_centroid (2d array(n_events x n_centroids))
        tot_fld_area_event    (1d array(n_events))
        annual_fld_area_per_centroid (2d array(n_years x n_centroids))
        annual_fld_area        (1d array (n_years))
        average_annual_fld_area (float)
        average_fld_area_event  (float)
    """
    def __init__(self):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE)

    def set_from_nc(self, flood_dir=None, centroids=None, reg=None,
                    countries=[], years=[2000], file_name=None,
                    rf_model=RF_MODEL[0], cl_model=CL_MODEL[5],
                    scenario=SCENARIO[1], centr_handling=CENTR_HANDLING[0],
                    prot_std=PROT_STD, dph_path=None, frc_path=None):
        """Wrapper to fill hazard from nc_flood file
        Parameters:
            flood_dir (string): location of flood data
            centroids (Centroids): centroids
            flood_dir (): string folder location of flood data
            rf_model: run-off model
            cl_model: climate model
            scenario: climate change scenario
            prot_std: protection standard
            final (string): standard file ending
        raises:
            NameError
        """
        if dph_path is None or frc_path is None:
            if flood_dir is not None:
                if os.path.exists(flood_dir):
                    dph_path, frc_path = self._select_model_run(flood_dir,
                                                                rf_model,
                                                                cl_model,
                                                                scenario,
                                                                prot_std)
                else:
                    LOGGER.warning('Flood directory does not exist,\
                                   setting Demo directory')
                    dph_path = os.path.join(DATA_DIR, HAZ_DEMO_FLDDPH)
                    frc_path = os.path.join(DATA_DIR, HAZ_DEMO_FLDFRC)
            else:
                LOGGER.warning('Flood directory does not exist,\
                                   setting Demo directory')
                """TODO: Put demo file in directory"""
                dph_path = os.path.join(DATA_DIR, HAZ_DEMO_FLDDPH)
                frc_path = os.path.join(DATA_DIR, HAZ_DEMO_FLDFRC)
        else:
            if not os.path.exists(dph_path):
                LOGGER.error('Invalid flood-file path')
                raise NameError
            if not os.path.exists(frc_path):
                LOGGER.error('Invalid flood-file path')
                raise NameError
        if centroids is not None:
            self.centroids = centroids
            centr_handling = 'align'
        elif countries or reg:
            self.centroids = RiverFlood.select_exact_area(countries, reg)
        else:
            centr_handling = 'full_hazard'
        intensity, fraction = self._read_nc(years, centr_handling,
                                            dph_path, frc_path)
        if scenario == 'historical':
            self.orig = np.full((self._n_events), True, dtype=bool)
        else:
            self.orig = np.full((self._n_events), False, dtype=bool)
        self.intensity = sparse.csr_matrix(intensity)
        self.fraction = sparse.csr_matrix(fraction)
        self.event_id = np.arange(1, self._n_events + 1)
        self.frequency = np.ones(self._n_events) / self._n_events
        return self

    def _read_nc(self, years, centr_handling, dph_path, frc_path):
        """ extract and flood intesity and fraction from flood
            data
        Returns:
            np.arrays
        """
        try:
            flood_dph = xr.open_dataset(dph_path)
            flood_frc = xr.open_dataset(frc_path)
            lon = flood_dph.lon.data
            lat = flood_dph.lat.data
            time = flood_dph.time.data
            event_index = self._select_event(time, years)
            self._n_events = len(event_index)
            self.date = np.array([dt.datetime(flood_dph.time[i].dt.year,
                                 flood_dph.time[i].dt.month,
                                 flood_dph.time[i].dt.day).toordinal()
                                 for i in event_index])
        except KeyError:
            LOGGER.error('Invalid dimensions or variables in file')
            raise KeyError
        except OSError:
            LOGGER.error('Problems while file reading,\
                         check flood_file specifications')
            raise NameError
        if centr_handling == 'full_hazard':
            if len(event_index) > 1:
                LOGGER.warning('Calculates global hazard,\
                               advanced memory requirements')
            LOGGER.warning('Calculates global hazard,\
                           select area to reduce runtime')
            self._set_centroids_from_file(lon, lat)
            try:
                intensity = np.nan_to_num(np.array([flood_dph.flddph[i].data.flatten()
                                          for i in event_index]))
                fraction = np.nan_to_num(np.array([flood_frc.fldfrc[i].data.flatten()
                                                  for i in event_index]))
            except MemoryError:
                LOGGER.error('Too many events for grid size')
                raise MemoryError
        else:
            n_centroids = self.centroids.coord.shape[0]
            win = self._cut_window(lon, lat)
            lon_coord = lon[win[0, 0]:win[1, 0] + 1]
            lat_coord = lat[win[0, 1]:win[1, 1] + 1]
            dph_window = flood_dph.flddph[event_index, win[0, 1]:win[1, 1] + 1,
                                          win[0, 0]:win[1, 0] + 1].data
            frc_window = flood_frc.fldfrc[event_index, win[0, 1]:win[1, 1] + 1,
                                          win[0, 0]:win[1, 0] + 1].data
            self. window = win
            try:
                intensity, fraction = _interpolate(lat_coord, lon_coord,
                                                   dph_window, frc_window,
                                                   self.centroids.lon,
                                                   self.centroids.lat,
                                                   n_centroids, self._n_events)
            except MemoryError:
                LOGGER.error('Too many events for grid size')
                raise MemoryError

        return intensity, fraction

    def _select_model_run(self, flood_dir, rf_model, cl_model, scenario,
                          prot_std, proj=False):
        """Provides paths for selected models to incorporate flood depth
        and fraction
        Parameters:
            flood_dir(string): string folder location of flood data
            rf_model (string): run-off model
            cl_model (string): climate model
            scenario (string): climate change scenario
            prot_std (string): protection standard
        """
        if proj is False:
            final = 'gev_0.1.nc'
            dph_file = 'flddph_{}_{}_{}_{}'\
                       .format(rf_model, cl_model, prot_std, final)
            frc_file = 'fldfrc_{}_{}_{}_{}'\
                       .format(rf_model, cl_model, prot_std, final)
        else:
            final = 'gev_picontrol_2000_0.1.nc'
            dph_file = 'flddph_{}_{}_{}_{}_{}'\
                       .format(rf_model, cl_model, scenario, prot_std, final)
            frc_file = 'fldfrc_{}_{}_{}_{}_{}'\
                       .format(rf_model, cl_model, scenario, prot_std, final)
        dph_path = os.path.join(flood_dir, dph_file)
        frc_path = os.path.join(flood_dir, frc_file)
        return dph_path, frc_path

    def _set_centroids_from_file(self, lon, lat):
        self.centroids = Centroids()
        gridX, gridY = np.meshgrid(lon, lat)
        self.centroids.lon = np.zeros((gridX.size))
        self.centroids.lat = np.zeros((gridY.size))
        self.centroids.lon = gridX.flatten()
        self.centroids.lat = gridY.flatten()
        self.centroids.id = np.arange(self.centroids.coord.shape[0])

    def _select_event(self, time, years):
        event_names = pd.to_datetime(time).year
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('Years not in file')  # check this
            raise AttributeError
        self.event_name = list(map(str, pd.to_datetime(time[event_index])))
        return event_index

    def _cut_window(self, lon, lat):
        """ Determine size of window to extract flood data.
        Parameters:
            lon: flood-file longitude coordinates
            lat: flood-file latitude coordinates
        Returns:
            np.array
        """
        lon_min = math.floor(min(self.centroids.coord[:, 1]))
        lon_max = math.ceil(max(self.centroids.coord[:, 1]))
        lat_min = math.floor(min(self.centroids.coord[:, 0]))
        lat_max = math.ceil(max(self.centroids.coord[:, 0]))
        diff_lon = np.diff(lon)[0]
        diff_lat = np.diff(lat)[0]
        win = np.zeros((2, 2), dtype=int)
        win[0, 0] = min(np.where((lon >= lon_min - diff_lon) &
                                 (lon <= lon_max + diff_lon))[0])
        win[1, 0] = max(np.where((lon >= lon_min - diff_lon) &
                                 (lon <= lon_max + diff_lon))[0])
        win[0, 1] = min(np.where((lat >= lat_min - diff_lat) &
                                 (lat <= lat_max + diff_lat))[0])
        win[1, 1] = max(np.where((lat >= lat_min - diff_lat) &
                                 (lat <= lat_max + diff_lat))[0])
        return win

    def set_flooded_area(self):
        """ Calculates flooded area for hazard. sets yearly flooded area and
            flooded area per event
        Raises:
            MemoryError
        """
        self.centroids.set_area_pixel()
        area_centr = self.centroids.area_pixel
        event_years = np.array([date.fromordinal(self.date[i]).year
                                for i in range(len(self.date))])
        years = np.unique(event_years)
        year_event_mask = self._annual_event_mask(event_years, years)

        try:
            self.fld_area_per_centroid = np.zeros((self._n_events,
                                                   len(self.centroids.lon)))
            self.annual_fld_area_per_centroid = np.zeros((len(years), len(self.centroids.lon)))
            self.fld_area_per_centroid = np.multiply(self.fraction.todense(),area_centr)
            self.tot_fld_area_event = np.sum(self.fld_area_per_centroid, axis = 1)
            for year_ind in range(len(years)):
                self.annual_fld_area_per_centroid[year_ind, :] = np.sum(self.fld_area_per_centroid[year_event_mask[year_ind, :],:], axis = 0)
            self.annual_fld_area = np.sum(self.annual_fld_area_per_centroid, axis=1) 
            self.average_annual_fld_area = np.mean(self.annual_fld_area)
            self.average_fld_area_event = np.mean(self.tot_fld_area_event)
        except MemoryError:
            self.fld_area_per_centroid = None
            self.tot_fld_area = None
            self.annual_fld_area_per_centroid = None
            self.annual_fld_area = None
            self.average_annual_fld_area = None
            self.average_fld_area_event = None
            LOGGER.warning('Number of events and considered area exceed memory capacities,\
                           area has not been calculated,\
                           attributes set to None')

    def set_flooded_area_cut(self, coordinates, centr_indices=None):
        """ Calculates flooded area for any window given with coordinates or 
            from indices of hazard centroids. sets yearly flooded area and
            per event
        Parameters:
            coordinates(2d array): coordinates of window
            centr_indices(1d array): indices of hazard centroid
        Raises:
            MemoryError
        """
        if centr_indices is None:
            centr_indices = interpol_index(self.centroids.coord, coordinates)
        self.centroids.set_area_pixel()
        area_centr = self.centroids.area_pixel[centr_indices]
        event_years = np.array([date.fromordinal(self.date[i]).year
                                for i in range(len(self.date))])
        years = np.unique(event_years)
        year_event_mask = self._annual_event_mask(event_years, years)
        try:
            self.fld_area_per_centroid = np.zeros((self._n_events,len(centr_indices)))
            self.annual_fld_area_per_centroid = np.zeros((len(years),len(centr_indices)))
            self.fld_area_per_centroid = np.multiply(self.fraction[:, centr_indices].todense(),area_centr)
            self.tot_fld_area_event = np.sum(self.fld_area_per_centroid, axis = 1)
            for year_ind in range(len(years)):
                self.annual_fld_area_per_centroid[year_ind,:] = np.sum(self.fld_area_per_centroid[year_event_mask[year_ind, :],:], axis = 0)
            self.annual_fld_area = np.sum(self.annual_fld_area_per_centroid, axis=1)
            self.average_annual_fld_area = np.mean(self.annual_fld_area)
            self.average_fld_area_event = np.mean(self.tot_fld_area_event)
            
        except MemoryError:
            self.fld_area_per_centroid = None
            self.tot_fld_area_event = None
            self.annual_fld_area_per_centroid = None
            self.annual_fld_area = None
            self.average_annual_fld_area = None
            self.average_fld_area_event = None
            LOGGER.warning('Number of events and considered area exceed memory capacities,\
                           area has not been calculated,\
                           attributes set to None')

    def _annual_event_mask(self, event_years, years):
        event_mask = np.full((len(years), len(event_years)), False, dtype=bool)
        for year_ind in range(len(years)):
            events = np.where(event_years == years[year_ind])[0]
            event_mask[year_ind, events] = True
        return event_mask

    def select_window_area(countries=[], reg=[]):
        """ Extract coordinates of selected countries or region
        from NatID in a rectangular box. If countries are given countries are cut,
        if only reg is given, the whole region is cut.
        Parameters:
            countries: List of countries
            reg: List of regions
        Raises:
            AttributeError
        Returns:
            np.array
        """
        centroids = Centroids()
        natID_info = pd.read_csv(NAT_REG_ID)
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
        if countries:
            if not any(np.isin(natID_info['ISO'], countries)):
                LOGGER.error('Wrong country ISO')
                raise KeyError
            natID = natID_info["ID"][np.isin(natID_info["ISO"], countries)]
        elif reg:
            natID = natID_info["ID"][np.isin(natID_info["Reg_name"], reg)]
            if not any(np.isin(natID_info["Reg_name"], reg)):
                LOGGER.error('Wrong region shortcut')
                raise KeyError
        else:
            centroids.lat = np.zeros((gridX.size))
            centroids.lon = np.zeros((gridX.size))
            centroids.lon = gridX.flatten()
            centroids.lat = gridY.flatten()
            centroids.id = np.arange(centroids.lon.shape[0])
            centroids.id = np.arange(centroids.lon.shape[0])
            return centroids
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        lon_min = math.floor(min(lon_coordinates))
        if lon_min <= -179:
            lon_inmin = 0
        else:
            lon_inmin = min(np.where((isimip_lon >= lon_min))[0]) - 1
        lon_max = math.ceil(max(lon_coordinates))
        if lon_max >= 179:
            lon_inmax = len(isimip_lon) - 1
        else:
            lon_inmax = max(np.where((isimip_lon <= lon_max))[0]) + 1
        lat_min = math.floor(min(lat_coordinates))
        if lat_min <= -89:
            lat_inmin = 0
        else:
            lat_inmin = min(np.where((isimip_lat >= lat_min))[0]) - 1
        lat_max = math.ceil(max(lat_coordinates))
        if lat_max >= 89:
            lat_max = len(isimip_lat) - 1
        else:
            lat_inmax = max(np.where((isimip_lat <= lat_max))[0]) + 1
        lon = isimip_lon[lon_inmin: lon_inmax]
        lat = isimip_lat[lat_inmin: lat_inmax]

        gridX, gridY = np.meshgrid(lon, lat)
        lat = np.zeros((gridX.size))
        lon = np.zeros((gridX.size))
        lon = gridX.flatten()
        lat = gridY.flatten()
        centroids.set_lat_lon(lat, lon)
        centroids.id = np.arange(centroids.coord.shape[0])
        centroids.set_region_id()

        return centroids

    def select_exact_area(countries=[], reg=[]):
        """ Extract coordinates of selected countries or region
        from NatID grid. If countries are given countries are cut,
        if only reg is given, the whole region is cut.
        Parameters:
            countries: List of countries
            reg: List of regions
        Raises:
            KeyError
        Returns:
            centroids
        """
        centroids = Centroids()
        natID_info = pd.read_csv(NAT_REG_ID)
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
        try:
            if countries:
                if not any(np.isin(natID_info['ISO'], countries)):
                    LOGGER.error('Wrong region shortcut')
                    raise KeyError
                natID = natID_info["ID"][np.isin(natID_info["ISO"], countries)]
            elif reg:
                if not any(np.isin(natID_info["Reg_name"], reg)):
                    LOGGER.error('Wrong region shortcut')
                    raise KeyError
                natID = natID_info["ID"][np.isin(natID_info["Reg_name"], reg)]
            else:
                centroids.lon = np.zeros((gridX.size))
                centroids.lat = np.zeros((gridX.size))
                centroids.lon = gridX.flatten()
                centroids.lat = gridY.flatten()
                centroids.id = np.arange(centroids.lon.shape[0])
                return centroids
        except KeyError:
            LOGGER.error('Selected country or region does\
                          not match reference file')
            raise KeyError
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        centroids.set_lat_lon(lat_coordinates, lon_coordinates)
        centroids.id = np.arange(centroids.lon.shape[0])
        centroids.set_region_id()
        return centroids

    def select_exact_area_polygon(countries=[], reg=[]):
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
        centroids = Centroids()
        natID_info = pd.read_csv(NAT_REG_ID)
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
        if countries:
            natID = natID_info["ID"][np.isin(natID_info["ISO"], countries)]
        elif reg:
            natID = natID_info["ID"][np.isin(natID_info["Reg_name"], reg)]
        else:
            centroids.coord = np.zeros((gridX.size, 2))
            centroids.coord[:, 1] = gridX.flatten()
            centroids.coord[:, 0] = gridY.flatten()
            centroids.id = np.arange(centroids.coord.shape[0])
            return centroids
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        centroids.coord = np.zeros((len(lon_coordinates), 2))
        centroids.coord[:, 1] = lon_coordinates
        centroids.coord[:, 0] = lat_coordinates
        centroids.id = np.arange(centroids.coord.shape[0])
        orig_proj = 'epsg:4326'
        country = gpd.GeoDataFrame()
        country['geometry'] = list(zip(centroids.coord[:, 1],
                                       centroids.coord[:, 0]))
        country['geometry'] = country['geometry'].apply(Point)
        country.crs = {'init': orig_proj}
        points = country.geometry.values
        concave_hull, _ = alpha_shape(points, alpha=1)

        return concave_hull


def _interpolate(lat, lon, dph_window, frc_window, centr_lon, centr_lat,
                 n_centr, n_ev, method='nearest'):
    """ Prepares data for interpolation and applies interpolation function,
    to assign flood parameters to chosen centroids.
        Parameters:
            lat (1d array): first axis for grid
            lon (1d array): second axis for grid
            dph_window (3d array): depth values
            frc_window (3d array): fraction
            centr_lon (1d array): centroids lon
            centr_lat (1d array): centroids lat
            n_centr (int): number centroids
            n_ev (int): number of events
        Returns:
            np.arrays
        """
    if lat[0] - lat[1] > 0:
        lat = np.flipud(lat)
        dph_window = np.flip(dph_window, axis=1)
        frc_window = np.flip(frc_window, axis=1)
    if lon[0] - lon[1] > 0:
        lon = np.flipud(lon)
        dph_window = np.flip(dph_window, axis=2)
        frc_window = np.flip(frc_window, axis=2)

    intensity = np.zeros((dph_window.shape[0], n_centr))
    fraction = np.zeros((dph_window.shape[0], n_centr))
    for i in range(n_ev):
        intensity[i, :] = sp.interpolate.interpn((lat, lon),
                                                 np.nan_to_num(dph_window[i, :, :]),
                                                 (centr_lat, centr_lon),
                                                 method='nearest',
                                                 bounds_error=False,
                                                 fill_value=None)
        fraction[i, :] = sp.interpolate.interpn((lat, lon),
                                                np.nan_to_num(frc_window[i, :, :]),
                                                (centr_lat, centr_lon),
                                                method='nearest',
                                                bounds_error=False,
                                                fill_value=None)
    return intensity, fraction

