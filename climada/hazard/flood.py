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

import matplotlib.pyplot as plt
import geopandas as gpd
from climada.util.constants import NAT_REG_ID, DATA_DIR, HAZ_DEMO_FLDPH, HAZ_DEMO_FLFRC, GLB_CENTROIDS_NC
from pyproj import Proj, transform
from climada.util.constants import ONE_LAT_KM
from scipy.interpolate import NearestNDInterpolator
#from shapely.ops import transform
from functools import partial
from scipy import sparse
from numba import jit
from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.util.interpolation import index_nn_aprox
from shapely.geometry import Polygon
from climada.util.coordinates import get_country_geometries
LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RF'
""" Hazard type acronym RiverFlood"""


class RiverFlood(Hazard):
    """Contains flood events
    Flood intensities are calculated by means of the
    CaMa-Flood global hydrodynamic model

    Attributes:
    TODO: Maybe define model characteristics as attributes
    """
    def __init__(self):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE)
        self.fld_area = 0

    def set_from_nc(self, flood_dir=None, centroids=None, reg= None,
                    countries=[], years=[2000], rf_model='ORCHIDEE',
                    cl_model='gswp3', scenario='', centr_handling='align',
                    prot_std='flopros', final=None):
        """Wrapper to fill hazard from nc_flood file
        Parameters:
            flood_dir (string): location of flood data
            centroids (Centroids): centroids
            flood_dir (): string folder location of flood data
            rf_model: run-off model
            cl_model: clima model
            scenario: climate change scenario
            prot_std: protection standard
            final (string): standard file ending
        raises:
            AttributeError
        """
        if flood_dir is not None:
            if os.path.exists(flood_dir):
                self.__select_ModelRun(flood_dir, rf_model, cl_model, scenario,
                                       prot_std)
            else:
                LOGGER.warning('Flood directory does not exist,\
                               setting Demo directory')
                """TODO: Put demo file in directory"""
                self._dph_path = os.path.join(DATA_DIR, HAZ_DEMO_FLDPH)
                self._frc_path = os.path.join(DATA_DIR, HAZ_DEMO_FLFRC)
        else:
            LOGGER.warning('Flood directory does not exist,\
                               setting Demo directory')
            """TODO: Put demo file in directory"""
            self._dph_path = os.path.join(DATA_DIR, HAZ_DEMO_FLDPH)
            self._frc_path = os.path.join(DATA_DIR, HAZ_DEMO_FLFRC)
        if centroids is not None:
            self.centroids = centroids
            centr_handling = 'align'
        elif countries or reg:
            self.centroids = RiverFlood.select_area(countries, reg)
        else:
            centr_handling = 'full_hazard'
        #TODO: exceptions could be catched here
        intensity, fraction = self._read_nc(years, centr_handling)       
        if scenario == 'historical':
            self.orig = np.full((self._n_events), True, dtype=bool)
        else:
            self.orig = np.full((self._n_events), False, dtype=bool)
        self.intensity = sparse.csr_matrix(intensity)
        self.fraction = sparse.csr_matrix(fraction)
        fig, ax = self.centroids.plot()
        plt.show(block = True)

        self.event_id = np.arange(1, self._n_events + 1)
        self.frequency = np.ones(self._n_events) / self._n_events
        return self

    def _read_nc(self, years, centr_handling):
        """ extract and flood intesity and fraction from flood
            data
        Returns:
            np.arrays
        """
        try:
            flood_dph = xr.open_dataset(self._dph_path)
            flood_frc = xr.open_dataset(self._frc_path)
            lon = flood_dph.lon.data
            lat = flood_dph.lat.data
            time = flood_dph.time.data
            event_index = self.__select_event(time, years)
            self._n_events = len(event_index)
            self.date = np.array([dt.datetime(
                                 flood_dph.time.dt.year[i],
                                 flood_dph.time.dt.month[i],
                                 flood_dph.time.dt.day[i]).toordinal()
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
            self.__set_centroids_from_file(lon, lat)
            try:
                intensity = np.array([flood_dph.flddph[i].data.flatten()
                                     for i in event_index])
                fraction = np.array([flood_frc.fldfrc[i].data.flatten()
                                     for i in event_index])
                print(intensity)
            except MemoryError:
                LOGGER.error('Too many events for grid size')
                raise MemoryError
        else:
            n_centroids = self.centroids.coord.shape[0]
            win = self.__cut_window(lon, lat)
            lon_coord = lon[win[0, 0]:win[1, 0] + 1]
            lat_coord = lat[win[0, 1]:win[1, 1] + 1]
            dph_window = flood_dph.flddph[event_index, win[0, 1]:win[1, 1] + 1,
                                          win[0, 0]:win[1, 0] + 1].data
            frc_window = flood_frc.fldfrc[event_index, win[0, 1]:win[1, 1] + 1,
                                          win[0, 0]:win[1, 0] + 1].data
            try:
                intensity, fraction = _interpolate(lat_coord, lon_coord,
                                                   dph_window, frc_window,
                                                   self.centroids.coord[:, 1],
                                                   self.centroids.coord[:, 0],
                                                   n_centroids, self._n_events)
            except MemoryError:
                LOGGER.error('Too many events for grid size')
                raise MemoryError
        #self.centr_area(lon, lat)
        return intensity, fraction

    def __select_ModelRun(self, flood_dir, rf_model, cl_model, scenario,
                          prot_std, final=None):
        """Provides paths for selected models to incorporate flood depth
        and fraction
        Parameters:
            flood_dir(string): string folder location of flood data
            rf_model (string): run-off model
            cl_model (string): clima model
            scenario (string): climate change scenario
            prot_std (string): protection standard
        """
        if final is None:
            final = 'gev_0.1.nc'
        dph_file = 'flddph_{}_{}_{}{}_{}'\
                         .format(rf_model, cl_model, scenario, prot_std, final)
        frc_file = 'fldfrc_{}_{}_{}{}_{}'\
                         .format(rf_model, cl_model, scenario, prot_std, final)
        self._dph_path = os.path.join(flood_dir, dph_file)
        self._frc_path = os.path.join(flood_dir, frc_file)

    def __set_centroids_from_file(self, lon, lat):
        self.centroids = Centroids()
        gridX, gridY = np.meshgrid(lon, lat)
        self.centroids.coord = np.zeros((gridX.size, 2))
        self.centroids.coord[:, 1] = gridX.flatten()
        self.centroids.coord[:, 0] = gridY.flatten()
        dlon = np.diff(lon)[0]
        dlat = np.diff(lat)[0]
        self.centroids.resolution = (dlon, dlat)
        self.centroids.set_area_per_centroid()
        self.centroids.id = np.arange(self.centroids.coord.shape[0])

    def __select_event(self, time, years):
        event_names = pd.to_datetime(time).year
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('Years not in file')  # check this
            raise AttributeError
        self.event_name = list(map(str, pd.to_datetime(time[event_index]).
                               year))
        return event_index

    def __cut_window(self, lon, lat):
        """ Determine size of window to extract flood data.
        Parameters:
            countries: List of countries
            reg: List of regions
        Raises:
            AttributeError
        Returns:
            np.array
        """
        lon_min = math.floor(min(self.centroids.coord[:, 1]))
        lon_max = math.ceil(max(self.centroids.coord[:, 1]))
        lat_min = math.floor(min(self.centroids.coord[:, 0]))
        lat_max = math.ceil(max(self.centroids.coord[:, 0]))
        window = np.zeros((2, 2), dtype=int)
        window[0, 0] = min(np.where((lon >= lon_min) & (lon <= lon_max))[0])
        window[1, 0] = max(np.where((lon >= lon_min) & (lon <= lon_max))[0])
        window[0, 1] = min(np.where((lat >= lat_min) & (lat <= lat_max))[0])
        window[1, 1] = max(np.where((lat >= lat_min) & (lat <= lat_max))[0])
        return window

    def set_flooded_area(self, centr_indices):
        area_centr = self.centroids.area_per_centroid[centr_indices]
        self.fld_area = area_centr * self.fraction[0, centr_indices].transpose()

    def centr_area(self, lon, lat):
        """TODO write method to calculate improved centroids area using 
           exact coordinate transformation"""
        dlon = 360 / 8640
        dlat = 180 / 4320
        print(sum(self.centroids.area_per_centroid))
        centr_area = np.zeros((self.centroids.coord.shape[0]))
        centr_index = 0
        inProj = Proj("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
        outProj = Proj(init='epsg:3857')

        for i in range(self.centroids.coord.shape[0]):
            lon = [self.centroids.coord[i, 1] - dlon / 2., self.centroids.coord[i, 1] + dlon / 2.]
            lat = [self.centroids.coord[i, 0] - dlat / 2., self.centroids.coord[i, 0] + dlat]
            xx, yy = transform(inProj, outProj, lon, lat)
            box1 = box(xx[0], yy[0], xx[1], yy[1])
            projected_area = box1.area
            centr_area[centr_index] = projected_area
            centr_index += 1



    @property
    def dph_file(self):
        return self._dph_file
    @property
    def frc_file(self):
        return self._frc_file

    @staticmethod
    def select_area(countries=[], reg=[]):
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
        dlon = np.diff(isimip_lon)[0]
        dlat = np.diff(isimip_lat)[0]

        if countries:
            natID = natID_info["ID"][np.isin(natID_info["ISO"], countries)]
        elif reg:
            natID = natID_info["ID"][np.isin(natID_info["Reg_name"], reg)]
        else:
            centroids.coord = np.zeros((gridX.size, 2))
            centroids.coord[:, 1] = gridX.flatten()
            centroids.coord[:, 0] = gridY.flatten()
            centroids.id = np.arange(centroids.coord.shape[0])
            centroids.resolution = (dlon, dlat)
            centroids.set_area_per_centroid()
            centroids.id = np.arange(centroids.coord.shape[0])
            return centroids
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]

        lon_min = math.floor(min(lon_coordinates))
        lon_inmin = min(np.where((isimip_lon >= lon_min))[0]) - 1
        lon_max = math.ceil(max(lon_coordinates))
        lon_inmax = max(np.where((isimip_lon <= lon_max))[0]) + 1
        lat_min = math.floor(min(lat_coordinates))
        lat_inmin = min(np.where((isimip_lat >= lat_min))[0]) - 1
        lat_max = math.ceil(max(lat_coordinates))
        lat_inmax = max(np.where((isimip_lat <= lat_max))[0]) + 1

        lon = isimip_lon[lon_inmin: lon_inmax]
        lat = isimip_lat[lat_inmin: lat_inmax]
        gridX, gridY = np.meshgrid(lon, lat)
        centroids.coord = np.zeros((gridX.size, 2))
        centroids.coord[:, 1] = gridX.flatten()
        centroids.coord[:, 0] = gridY.flatten()
        centroids.resolution = (dlon, dlat)
        centroids.set_area_per_centroid()
        centroids.id = np.arange(centroids.coord.shape[0])
        centroids.set_region_id()
        return centroids

    @staticmethod
    def select_exact_area(countries=[], reg=[]):
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
        centroids.set_region_id()
        return centroids


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


def _interpolate2(lat, lon, dph_window, frc_window, centr_lon, centr_lat,
                  n_centr, n_ev, method='nearest'):

    gridX, gridY = np.meshgrid(lon, lat)
    centroids = np.zeros((gridX.size, 2))
    centroids[:, 0] = gridY.flatten()
    centroids[:, 1] = gridX.flatten()
    coordinates = np.zeros((n_centr, 2))
    coordinates[:, 0] = centr_lat
    coordinates[:, 1] = centr_lon
    test_index = interpol_index(centroids=centroids, coordinates=coordinates)
    dph_window = np.reshape(dph_window, (n_ev, gridX.size))
    frc_window = np.reshape(frc_window, (n_ev, gridX.size))
    intensity = dph_window[:, test_index]
    fraction = frc_window[:, test_index]
    return intensity, fraction
