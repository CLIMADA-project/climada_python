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
from rasterio.warp import Resampling
import shapely
import geopandas as gpd
from sklearn.neighbors import BallTree

from climada.util.constants import NAT_REG_ID, GLB_CENTROIDS_NC
from climada.util.interpolation import interpol_index
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.util.coordinates import get_land_geometry

from shapely.geometry import Point
from shapely.geometry.multipolygon import MultiPolygon
from climada.util.alpha_shape import alpha_shape,plot_polygon


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RF'
""" Hazard type acronym RiverFlood"""


#RF_MODEL = ['ORCHIDEE',
#            'H08',
#            'LPJmL',
#            'MPI-HM',
#            'PCR-GLOBWB',
#            'WaterGAP2',
#            'CLM',
#            'JULES-TUC'
#            'JULES-UoE',
#            'VIC',
#            'VEGAS'
#            ]
#CL_MODEL = ['gfdl-esm2m',
#            'hadgem2-es',
#            'ipsl-cm5a-lr',
#            'miroc5',
#            'wfdei',
#            'gswp3',
#            'princeton',
#            'watch'
#            ]
#SCENARIO = ['',
#            'historical',
#            'rcp26',
#            'rcp60'
#            ]
#
#PROT_STD = 'flopros'


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

    def set_from_nc(self, dph_path=None, frc_path=None, origin=False,
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

        raises:
            NameError
        """
        if dph_path is None:
            LOGGER.error('No flood-depth-path set')
            raise NameError
        if frc_path is None:
            LOGGER.error('No flood-fraction-path set')
            raise NameError
        if not os.path.exists(dph_path):
            LOGGER.error('Invalid flood-file path ' + dph_path)
            raise NameError
        if not os.path.exists(frc_path):
            LOGGER.error('Invalid flood-file path ' + frc_path)
            raise NameError

        flood_dph = xr.open_dataset(dph_path)
        time = flood_dph.time.data
        event_index = self._select_event(time, years) 
        bands = event_index + 1

        if countries or reg:
            # centroids as points
            dest_centroids, iso_codes, natID = RiverFlood._select_exact_area(
                countries, reg)

            # envelope containing counties
            cntry_geom = get_land_geometry(iso_codes)
            self.set_raster(files_intensity=[dph_path],
                            files_fraction=[frc_path], band=bands.tolist(),
                            geometry=cntry_geom)
            min_lon, min_lat, max_lon, max_lat = self.centroids.total_bounds
            self.reproject_raster(transform=dest_centroids.meta['transform'],
                                  width=dest_centroids.meta['width'],
                                  height=dest_centroids.meta['height'],
                                  resampling=Resampling.nearest)
            self.centroids.set_meta_to_lat_lon()
            min_lon, min_lat, max_lon, max_lat
            in_country = self._intersect_area(natID, min_lon, min_lat, max_lon, max_lat)
            in_country = np.flip(in_country, axis = 0).flatten()
            self.centroids.set_lat_lon(self.centroids.lat[in_country],
                                       self.centroids.lon[in_country])
            self.intensity = self.intensity[:, in_country]
            self.fraction = self.fraction[:, in_country]
            
        elif not centroids:
            # centroids as raster
            self.set_raster(files_intensity=[dph_path],
                            files_fraction=[frc_path],
                            band=event_index+1)
        else: # use given centroids
            if centroids.meta or grid_is_regular(centroids)[0]:
                if not centroids.meta:
                    centroids.set_lat_lon_to_meta()
                self.set_raster(files_intensity=[dph_path],
                                files_fraction=[frc_path], band=event_index+1,
                                transform=centroids.meta['transform'],
                                width=centroids.meta['width'],
                                height=centroids.meta['height'],
                                resampling=Resampling.nearest)
            else:
                centroids.set_lat_lon_to_meta()
                self.set_raster(files_intensity=[dph_path],
                                files_fraction=[frc_path], band=event_index+1,
                                transform=centroids.meta['transform'],
                                width=centroids.meta['width'],
                                height=centroids.meta['height'],
                                resampling=Resampling.nearest)
                self.centroids.set_meta_to_lat_lon()
                tree = BallTree(np.radians(self.centroids.coord), 
                                metric='haversine')
                assigned = tree.query(np.radians(centroids.coord), k=1,
                                      dualtree=True, breadth_first=False)
                self.centroids = centroids
                self.intensity = self.intensity[:, assigned]
                self.fraction = self.fraction[:, assigned]

        self.units = 'm'
        self.tag.file_name = dph_path + ';' + frc_path
        self.event_id = np.arange(self.intensity.shape[0])
        self.event_name = list(map(str, years))

        if origin:
            self.orig = np.ones(self.size, bool)
        else:
            self.orig = np.zeros(self.size, bool)

        self.frequency = np.ones(self.size) / self.size
        self.date = np.array([dt.datetime(flood_dph.time[i].dt.year,
                              flood_dph.time[i].dt.month,
                              flood_dph.time[i].dt.day).toordinal()
                              for i in event_index])

    def _intersect_area(self, natID, min_lon, min_lat, max_lon, max_lat):
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        #min_lon, min_lat, max_lon, max_lat = self.centroids.total_bounds
        lon_o = np.argmin(np.abs(min_lon - isimip_lon))
        lon_f = np.argmin(np.abs(max_lon - isimip_lon))
        lat_o = np.argmin(np.abs(min_lat - isimip_lat))
        lat_f = np.argmin(np.abs(max_lat - isimip_lat))
        isimip_NatIdGrid = isimip_grid.NatIdGrid[lat_o:lat_f+1, lon_o:lon_f+1].data
        # für alle lander
        in_country = np.isin(isimip_NatIdGrid, natID)
        
        return in_country




    def _read_nc(self, years, dph_path, frc_path):
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

        win = self._cut_window(lon, lat)
        lon_coord = lon[win[0, 0]:win[1, 0] + 1]
        lat_coord = lat[win[0, 1]:win[1, 1] + 1]
        dph_window = flood_dph.flddph[event_index, win[0, 1]:win[1, 1] + 1,
                                      win[0, 0]:win[1, 0] + 1].data
        frc_window = flood_frc.fldfrc[event_index, win[0, 1]:win[1, 1] + 1,
                                      win[0, 0]:win[1, 0] + 1].data
        self. window = win

        intensity, fraction = _interpolate(lat_coord, lon_coord,
                                           dph_window, frc_window,
                                           self.centroids.lon,
                                           self.centroids.lat,
                                           self._n_events)

        return intensity, fraction

    def _select_event(self, time, years):
        event_names = pd.to_datetime(time).year
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('No events found for selected ' + str(years))
            raise AttributeError
        self.event_name = list(map(str, pd.to_datetime(time[event_index])))
        return event_index
    
    def exclude_returnlevel(self, path):

        flood_frc = xr.open_dataset(path)
        lon = flood_frc.lon.data
        lat = flood_frc.lat.data
        win = self._cut_window(lon, lat)
        frc_window = flood_frc.fldfrc[:, win[0, 1]:win[1, 1] + 1,
                                      win[0, 0]:win[1, 0] + 1].data
        lon = lon[win[0, 0]:win[1, 0] + 1]
        lat = lat[win[0, 1]:win[1, 1] + 1]
        if lat[0] - lat[1] > 0:
            lat = np.flipud(lat)
            frc_window = np.flip(frc_window, axis=1)
        if lon[0] - lon[1] > 0:
            lon = np.flipud(lon)
            frc_window = np.flip(frc_window, axis=2)

        fraction = \
            sp.interpolate.interpn((lat, lon),
                                   np.nan_to_num(frc_window[0, :, :]),
                                   (self.centroids.lat, self.centroids.lon),
                                   method='nearest',
                                   bounds_error=False,
                                   fill_value=None)
        new_fraction = np.subtract(self.fraction.todense(), fraction)
        new_fraction = new_fraction.clip(0)
        self.fraction = sparse.csr_matrix(new_fraction)

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
        year_ev_mk = self._annual_event_mask(event_years, years)

        self.fla_ann_centr = np.zeros((len(years),
                                       len(self.centroids.lon)))
        self.fla_ev_centr = np.array(np.multiply(self.fraction.todense(),
                                                 area_centr))
        self.fla_event = np.sum(self.fla_ev_centr, axis=1)
        for year_ind in range(len(years)):
            self.fla_ann_centr[year_ind, :] =\
                np.sum(self.fla_ev_centr[year_ev_mk[year_ind, :], :],
                       axis=0)
        self.fla_annual = np.sum(self.fla_ann_centr, axis=1)
        self.fla_ann_av = np.mean(self.fla_annual)
        self.fla_ev_av = np.mean(self.fla_event)
        

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
        year_ev_mk = self._annual_event_mask(event_years, years)
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

    def _annual_event_mask(self, event_years, years):
        event_mask = np.full((len(years), len(event_years)), False, dtype=bool)
        for year_ind in range(len(years)):
            events = np.where(event_years == years[year_ind])[0]
            event_mask[year_ind, events] = True
        return event_mask
#
#    def select_window_area(countries=[], reg=[]):
#        """ Extract coordinates of selected countries or region
#        from NatID in a rectangular box. If countries are given countries
#        are cut, if only reg is given, the whole region is cut.
#        Parameters:
#            countries: List of countries
#            reg: List of regions
#        Raises:
#            AttributeError
#        Returns:
#            np.array
#        """
#        centroids = Centroids()
#        natID_info = pd.read_csv(NAT_REG_ID)
#        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
#        isimip_lon = isimip_grid.lon.data
#        isimip_lat = isimip_grid.lat.data
#        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
#        if countries:
#            if not any(np.isin(natID_info['ISO'], countries)):
#                LOGGER.error('Country ISO3s ' + str(countries) + ' unknown')
#                raise KeyError
#            natID = natID_info["ID"][np.isin(natID_info["ISO"], countries)]
#        elif reg:
#            natID = natID_info["ID"][np.isin(natID_info["Reg_name"], reg)]
#            if not any(np.isin(natID_info["Reg_name"], reg)):
#                LOGGER.error('Shortcuts ' + str(reg) + ' unknown')
#                raise KeyError
#        else:
#            centroids.lat = np.zeros((gridX.size))
#            centroids.lon = np.zeros((gridX.size))
#            centroids.lon = gridX.flatten()
#            centroids.lat = gridY.flatten()
#            centroids.id = np.arange(centroids.lon.shape[0])
#            centroids.id = np.arange(centroids.lon.shape[0])
#            return centroids
#        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
#        natID_pos = np.isin(isimip_NatIdGrid, natID)
#        lon_coordinates = gridX[natID_pos]
#        lat_coordinates = gridY[natID_pos]
#        lon_min = math.floor(min(lon_coordinates))
#        if lon_min <= -179:
#            lon_inmin = 0
#        else:
#            lon_inmin = min(np.where((isimip_lon >= lon_min))[0]) - 1
#        lon_max = math.ceil(max(lon_coordinates))
#        if lon_max >= 179:
#            lon_inmax = len(isimip_lon) - 1
#        else:
#            lon_inmax = max(np.where((isimip_lon <= lon_max))[0]) + 1
#        lat_min = math.floor(min(lat_coordinates))
#        if lat_min <= -89:
#            lat_inmin = 0
#        else:
#            lat_inmin = min(np.where((isimip_lat >= lat_min))[0]) - 1
#        lat_max = math.ceil(max(lat_coordinates))
#        if lat_max >= 89:
#            lat_max = len(isimip_lat) - 1
#        else:
#            lat_inmax = max(np.where((isimip_lat <= lat_max))[0]) + 1
#        lon = isimip_lon[lon_inmin: lon_inmax]
#        lat = isimip_lat[lat_inmin: lat_inmax]
#
#        gridX, gridY = np.meshgrid(lon, lat)
#        lat = np.zeros((gridX.size))
#        lon = np.zeros((gridX.size))
#        lon = gridX.flatten()
#        lat = gridY.flatten()
#        centroids.set_lat_lon(lat, lon)
#        centroids.id = np.arange(centroids.coord.shape[0])
#        centroids.set_region_id()
#
#        return centroids

    def _select_exact_area(countries=[], reg=[]):
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
                iso_codes = countries
            elif reg:
                if not any(np.isin(natID_info["Reg_name"], reg)):
                    LOGGER.error('Shortcuts ' + str(reg) + ' unknown')
                    raise KeyError
                natID = natID_info["ID"][np.isin(natID_info["Reg_name"], reg)]
                iso_codes = natID_info["ISO"][np.isin(natID_info["Reg_name"],
                                              reg)].tolist()
            else:
                centroids.set_lat_lon(gridY.flatten(), gridX.flatten())
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
        centroids.set_lat_lon_to_meta()
        return centroids, iso_codes, natID

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
            LOGGER.error('Not valid inputs')
            raise ValueError
        isimip_NatIdGrid = isimip_grid.NatIdGrid.data
        natID_pos = np.isin(isimip_NatIdGrid, natID)
        lon_coordinates = gridX[natID_pos]
        lat_coordinates = gridY[natID_pos]
#        orig_proj = 'epsg:4326'
#        country = gpd.GeoDataFrame()
#        country['geometry'] = list(zip(lon_coordinates,
#                                       lat_coordinates))
#        country['geometry'] = country['geometry'].apply(Point)
#        country.crs = {'init': orig_proj}
#        points = country.geometry.values
        points = list(map(Point, np.array([lon_coordinates, lat_coordinates]).transpose()))

#        points = list(zip(lon_coordinates,
#                                       lat_coordinates))
        concave_hull, _ = alpha_shape(points, alpha=20)
        #plot_polygon(concave_hull)

        for lat, lon in zip(lat_coordinates, lon_coordinates):
            if not concave_hull.contains(Point(lon, lat)):
                print('error')


        return concave_hull

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    s = '{}'.format(f)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(f, n)
    i, p, d = s.partition('.')
    return '.'.join([i, (d+'0'*n)[:n]])

def _interpolate(lat, lon, dph_window, frc_window, centr_lon, centr_lat,
                 n_ev, method='nearest'):
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

    intensity = np.zeros((dph_window.shape[0], centr_lon.size))
    fraction = np.zeros((dph_window.shape[0], centr_lon.size))
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
