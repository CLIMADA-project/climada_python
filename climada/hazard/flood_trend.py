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

__all__ = ['FloodTrend']

import logging
import os
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import math
import datetime as dt
import numpy.ma as ma
from datetime import date
import geopandas as gpd
from climada.util.constants import NAT_REG_ID, GLB_CENTROIDS_NC
from climada.util.constants import HAZ_DEMO_FLDDPH, HAZ_DEMO_FLDFRC
from climada.util.interpolation import interpol_index
from scipy import sparse
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from shapely.geometry import Point
import copy
from climada.util.alpha_shape import alpha_shape

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'FT'
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

PROT_STD = 'flopros'


class FloodTrend(Hazard):
    """Contains flood events
    Flood intensities are calculated by means of the
    CaMa-Flood global hydrodynamic model

    Attributes:
        fla_ev_centr    (2d array(n_events x n_centroids)) flooded area in
                        every centroid for every event
        fla_event       (1d array(n_events)) total flooded area for every event
        fla_ann_centr   (2d array(n_years x n_centroids)) flooded area in
                        every centroid for every event
        fla_annual      (1d array (n_years)) total flooded area for every year
        fla_ann_av      (float) average flooded area per year
        fla_ev_av       (float) average flooded area per event
    """
    def __init__(self):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE)

    def set_from_nc(self, flood_dir=None, dph_path=None,
                    centroids=None, countries=[], reg=None, years=[2000]):
        """Wrapper to fill hazard from nc_flood file
        Parameters:
            flood_dir (string): location of flood data
                (can be used when different model-runs are considered,
                dph_path and frc_path must be None)
            dph_path (string): Flood file to read (depth)
            frc_path (string): Flood file to read (fraction)
            centroids (Centroids): centroids
                (area that is considered, reg and country must be None)
            countries (list of countries ISO3) selection of countries
                (reg must be None!)
            reg (list of regions): can be set with region code if whole areas
                are considered (if not None, countries and centroids
                are ignored)
            years (int list): years that are considered
            rf_model: run-off model (only when flood_dir is selected)
            cl_model: climate model (only when flood_dir is selected)
            scenario: climate change scenario (only when flood_dir is selected)
            prot_std: protection standard (only when flood_dir is selected)
        raises:
            NameError
        """
        if not os.path.exists(dph_path):
            LOGGER.error('Invalid flood-file path ' + dph_path)
            raise NameError

        if centroids is not None:
            self.centroids = centroids
            centr_handling = 'align'
        elif countries or reg:
            self.centroids = FloodTrend.select_exact_area(countries, reg)
            centr_handling = 'align'
        else:
            centr_handling = 'full_hazard'

        intensity, fraction = self._read_nc(years, centr_handling, dph_path)

        self.orig = np.full((self._n_events), True, dtype=bool)

        self.intensity = sparse.csr_matrix(intensity)
        self.fraction = sparse.csr_matrix(fraction)
        self.event_id = np.arange(1, self._n_events + 1)
        self.units = 'm'
        date_test = np.array([date.fromordinal(self.date[i]).year
                              for i in range(len(self.date))])
        
        time_frame = np.max(date_test) + 1 - np.min(date_test)
        
        time_frame = len(self.event_id)

        self.frequency = np.ones(self._n_events) / time_frame
        return self

    def _read_nc(self, years, centr_handling, dph_path):
        """ extract and flood intesity and fraction from flood
            data
        Returns:
            np.arrays
        """
        try:
            flood_dph = xr.open_dataset(dph_path)
            lon = flood_dph.lon.data
            lat = flood_dph.lat.data
            time = flood_dph.time.data
            time = [dt.datetime(year=2005, month = 1, day = 1)]
            event_index = [0]
            self._n_events = len(event_index)
            try:
                self.date = np.array([time[i].toordinal() for i in event_index])
            except TypeError:
                LOGGER.error('Invalid date format, could not set date')
        except KeyError:
            LOGGER.error('Invalid dimensions or variables in file ' +
                         dph_path)
            raise KeyError
       
        n_centroids = self.centroids.size
        win = self._cut_window(lon, lat)
        lon_coord = lon[win[0, 0]:win[1, 0] + 1]
        lat_coord = lat[win[0, 1]:win[1, 1] + 1]
        dph_window = flood_dph.Discharge[event_index, win[0, 1]:win[1, 1] + 1,
                                      win[0, 0]:win[1, 0] + 1].data

        self.window = win
        try:
            intensity, fraction = _interpolate(lat_coord, lon_coord,
                                               dph_window,
                                               self.centroids.lon,
                                               self.centroids.lat,
                                               n_centroids, self._n_events)
        except MemoryError:
            LOGGER.error('Too many events for grid size')
            raise MemoryError

        return intensity, fraction
    
    def set_intesity_threshold(self, threshold):
        
        new_intensity = self.intensity.todense()
        
        mask = np.less(new_intensity, threshold)
        new_intensity[mask] = 0
        
        self.intensity = sparse.csr_matrix(new_intensity)
        

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
        self.centroids.set_lat_lon(gridY.flatten(), gridX.flatten())

    def _select_event(self, time, years):
        event_names = time.astype('datetime64[Y]').astype(int) + 1970
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('No events found for selected ' + str(years))
            raise AttributeError
        self.event_name = list(map(str, event_names[event_index]))
        return event_index
    
    def exclude_insig_trends(self, path):

        trend_sig = xr.open_dataset(path)
        lon = trend_sig.lon.data
        lat = trend_sig.lat.data
        sign =trend_sig.variables['p-value'][0,:,:]

        if lat[0] - lat[1] > 0:
            lat = np.flipud(lat)
            sign = np.flip(sign, axis=1)
        if lon[0] - lon[1] > 0:
            lon = np.flipud(lon)
            sign = np.flip(sign, axis=2)

        significance = \
            sp.interpolate.interpn((lat, lon),
                                   np.nan_to_num(sign),
                                   (self.centroids.lat, self.centroids.lon),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=None)
        
        sig_mask = np.array(significance< 0.1)*1
        
        new_trends = np.multiply(self.intensity.todense(), sig_mask)
        self.intensity = sparse.csr_matrix(new_trends)
        
    def get_dis_mask(self, dis = 'pos'):
        
        new_intensity = self.intensity.todense()

        if dis == 'pos':
            dis_map = np.greater(new_intensity, 0)
        else:
            dis_map = np.less(new_intensity, 0)
        
        new_trends = dis_map.astype(int)
        self.intensity = sparse.csr_matrix(new_trends)
        self.fraction = sparse.csr_matrix(new_trends)

    
    

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
        

    def _annual_event_mask(self, event_years, years):
        event_mask = np.full((len(years), len(event_years)), False, dtype=bool)
        for year_ind in range(len(years)):
            events = np.where(event_years == years[year_ind])[0]
            event_mask[year_ind, events] = True
        return event_mask

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
                    LOGGER.error('Country ISO3s ' + str(countries) +
                                 ' unknown')
                    raise KeyError
                natID = natID_info["ID"][np.isin(natID_info["ISO"], countries)]
            elif reg:
                if not any(np.isin(natID_info["Reg_name"], reg)):
                    LOGGER.error('Shortcuts ' + str(reg) + ' unknown')
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
            LOGGER.error('Selected country or region do ' +
                         'not match reference file')
            raise KeyError
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
        centroids.set_lat_lon(lat_coordinates, lon_coordinates)
        centroids.id = np.arange(centroids.lon.shape[0])
        #centroids.set_region_id()
        return centroids



def _interpolate(lat, lon, dph_window, centr_lon, centr_lat,
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
    if lon[0] - lon[1] > 0:
        lon = np.flipud(lon)
        dph_window = np.flip(dph_window, axis=2)

    intensity = np.zeros((dph_window.shape[0], n_centr))
    fraction = np.ones((dph_window.shape[0], n_centr))
    
    for i in range(n_ev):
        intensity[i, :] = \
            sp.interpolate.interpn((lat, lon),
                                   np.nan_to_num(dph_window[i, :, :]),
                                   (centr_lat, centr_lon),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=None)

    return intensity, fraction


