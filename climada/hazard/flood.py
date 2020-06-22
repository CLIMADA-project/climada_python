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
import datetime as dt
from datetime import date
from climada.util.constants import HAZ_DEMO_FLDDPH, HAZ_DEMO_FLDFRC
from climada.util.interpolation import interpol_index
from scipy import sparse
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids

from climada.util.coordinates import get_region_gridpoints

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

PROT_STD = 'flopros'


class RiverFlood(Hazard):
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

    def set_from_nc(self, flood_dir=None, dph_path=None, frc_path=None,
                    centroids=None, countries=[], reg=None, years=[2000],
                    rf_model=RF_MODEL[0], cl_model=CL_MODEL[5],
                    scenario=SCENARIO[1], prot_std=PROT_STD):
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
        if dph_path is None or frc_path is None:
            if flood_dir is not None:
                if os.path.exists(flood_dir):
                    dph_path, frc_path = self._select_model_run(flood_dir,
                                                                rf_model,
                                                                cl_model,
                                                                scenario,
                                                                prot_std)
                else:
                    dph_path = HAZ_DEMO_FLDDPH
                    frc_path = HAZ_DEMO_FLDFRC
                    LOGGER.warning('Flood directory ' + flood_dir +
                                   ' does not exist, setting Demo files ' +
                                   str(dph_path) + ' and ' + str(frc_path))
            else:
                dph_path = HAZ_DEMO_FLDDPH
                frc_path = HAZ_DEMO_FLDFRC
                LOGGER.warning('Flood directory not set ' +
                               ', setting Demo files ' +
                               str(dph_path) + ' and ' + str(frc_path))
        else:
            if not os.path.exists(dph_path):
                LOGGER.error('Invalid flood-file path ' + dph_path)
                raise NameError
            if not os.path.exists(frc_path):
                LOGGER.error('Invalid flood-file path ' + frc_path)
                raise NameError
        if centroids is not None:
            self.centroids = centroids
            centr_handling = 'align'
        elif countries or reg:
            self.centroids = RiverFlood.select_exact_area(countries, reg)
            centr_handling = 'align'
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
        self.units = 'm'
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
            LOGGER.error('Invalid dimensions or variables in file ' +
                         dph_path + ' or ' + frc_path)
            raise KeyError
        except OSError:
            LOGGER.error('Problems while reading file ' + dph_path +
                         ' or ' + frc_path +
                         ' check flood_file specifications')
            raise NameError
        if centr_handling == 'full_hazard':
            if len(event_index) > 1:
                LOGGER.warning('Calculates global hazard' +
                               ' advanced memory requirements')
            LOGGER.warning('Calculates global hazard, select area  with ' +
                           'countries, reg or centroids in set_from_nc ' +
                           'to reduce runtime')
            self._set_centroids_from_file(lon, lat)
            try:
                intensity = np.nan_to_num(np.array(
                        [flood_dph.flddph[i].data.flatten()
                            for i in event_index]))
                fraction = np.nan_to_num(np.array(
                        [flood_frc.fldfrc[i].data.flatten()
                            for i in event_index]))
            except MemoryError:
                LOGGER.error('Too many events for grid size')
                raise MemoryError
        else:
            n_centroids = self.centroids.size
            win = self._cut_window(lon, lat)
            lon_coord = lon[win[0, 0]:win[1, 0] + 1]
            lat_coord = lat[win[0, 1]:win[1, 1] + 1]
            dph_window = flood_dph.flddph[event_index, win[0, 1]:win[1, 1] + 1,
                                          win[0, 0]:win[1, 0] + 1].data
            frc_window = flood_frc.fldfrc[event_index, win[0, 1]:win[1, 1] + 1,
                                          win[0, 0]:win[1, 0] + 1].data
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
        self.centroids.set_lat_lon(gridY.flatten(), gridX.flatten())

    def _select_event(self, time, years):
        event_names = pd.to_datetime(time).year
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('No events found for selected ' + str(years))
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
        steps = np.array([lon[1] - lon[0], lat[1] - lat[0]])
        bounds = np.asarray(self.centroids.total_bounds)
        bounds[[0,1]] = np.floor(bounds[[0,1]]) - steps
        bounds[[2,3]] = np.ceil(bounds[[2,3]]) + steps
        return np.array([
            ((lon >= bounds[0]) & (lon <= bounds[2])).nonzero()[0][[0,-1]],
            ((lat >= bounds[1]) & (lat <= bounds[3])).nonzero()[0][[0,-1]],
        ], dtype=np.int).T

    def set_flooded_area(self):
        """ Calculates flooded area for hazard. sets yearly flooded area and
            flooded area per event
        Raises:
            MemoryError
        """
        self.centroids.set_area_pixel()
        area_centr = self.centroids.area_pixel
        event_years = np.array([date.fromordinal(d).year for d in self.date])
        years = np.unique(event_years)
        year_ev_mk = np.vstack([(event_years == y) for y in years])

        try:
            self.fla_ev_centr = self.fraction.toarray() * area_centr
            self.fla_event = np.sum(self.fla_ev_centr, axis=1)
            self.fla_ann_centr = [self.fla_ev_centr[mask,:].sum(axis=0) \
                                  for mask in year_ev_mk]
            self.fla_ann_centr = np.vstack(self.fla_ann_centr)
            self.fla_annual = self.fla_ann_centr.sum(axis=1)
            self.fla_ann_av = self.fla_annual.mean()
            self.fla_ev_av = self.fla_event.mean()
        except MemoryError:
            self.fla_ev_centr = None
            self.tot_fld_area = None
            self.fla_ann_centr = None
            self.fla_annual = None
            self.fla_ann_av = None
            self.fla_ev_av = None
            LOGGER.warning('Number of events and slected area exceed ' +
                           'memory capacities, area has not been calculated,' +
                           ' attributes set to None')

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
        year_ev_mk = np.vstack([(event_years == y) for y in years])
        try:
            self.fla_ev_centr = np.zeros((self._n_events, len(centr_indices)))
            self.fla_ann_centr = np.zeros((len(years), len(centr_indices)))
            self.fla_ev_centr = np.array(np.multiply(
                    self.fraction[:, centr_indices].todense(), area_centr))
            self.fla_event = np.sum(self.fla_ev_centr, axis=1)
            for year_ind in range(len(years)):
                self.fla_ann_centr[year_ind, :] = \
                    np.sum(self.fla_ev_centr[year_ev_mk[year_ind, :], :],
                           axis=0)
            self.fla_annual = np.sum(self.fla_ann_centr, axis=1)
            self.fla_ann_av = np.mean(self.fla_annual)
            self.fla_ev_av = np.mean(self.fla_event)

        except MemoryError:
            self.fla_ev_centr = None
            self.fla_event = None
            self.fla_ann_centr = None
            self.fla_annual = None
            self.fla_ann_av = None
            self.fla_ev_av = None
            LOGGER.warning('Number of events and slected area exceed ' +
                           'memory capacities, area has not been calculated,' +
                           ' attributes set to None')

    def select_window_area(countries=[], reg=[]):
        """ Extract coordinates of selected countries or region
        from NatID in a rectangular box. If countries are given countries
        are cut, if only reg is given, the whole region is cut.
        Parameters:
            countries: List of countries
            reg: List of regions
        Raises:
            AttributeError
        Returns:
            np.array
        """
        lat, lon = get_region_gridpoints(countries, regions=reg, rect=True)
        centroids = Centroids()
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
        lat, lon = get_region_gridpoints(countries=countries, regions=reg)
        centroids = Centroids()
        centroids.set_lat_lon(lat, lon)
        centroids.id = np.arange(centroids.lon.shape[0])
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
        intensity[i, :] = \
            sp.interpolate.interpn((lat, lon),
                                   np.nan_to_num(dph_window[i, :, :]),
                                   (centr_lat, centr_lon),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=None)
        fraction[i, :] = \
            sp.interpolate.interpn((lat, lon),
                                   np.nan_to_num(frc_window[i, :, :]),
                                   (centr_lat, centr_lon),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=None)
    return intensity, fraction
