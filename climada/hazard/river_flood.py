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
import geopandas as gpd
import datetime as dt
from datetime import date
from rasterio.warp import Resampling
import copy
from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
from climada.util.coordinates import get_region_gridpoints,\
                                     region2isos, country_iso2natid
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.util.coordinates import get_land_geometry, read_raster

NATID_INFO = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RF'
"""Hazard type acronym RiverFlood"""


class RiverFlood(Hazard):
    """Contains flood events
    Flood intensities are calculated by means of the
    CaMa-Flood global hydrodynamic model

    Attributes:

        fla_event       (1d array(n_events)) total flooded area for every event
        fla_annual      (1d array (n_years)) total flooded area for every year
        fla_ann_av      (float) average flooded area per year
        fla_ev_av       (float) average flooded area per event
        fla_ann_centr   (2d array(n_years x n_centroids)) flooded area in
                        every centroid for every event
        fla_ev_centr    (2d array(n_events x n_centroids)) flooded area in
                        every centroid for every event

    """

    def __init__(self):
        """Empty constructor"""

        Hazard.__init__(self, HAZ_TYPE)

    def set_from_nc(self, dph_path=None, frc_path=None, origin=False,
                    centroids=None, countries=None, reg=None, shape=None, ISINatIDGrid=False,
                    years=[2000]):
        """Wrapper to fill hazard from nc_flood file
        Parameters:
            dph_path (string): Flood file to read (depth)
            frc_path (string): Flood file to read (fraction)
            origin (bool): Historical or probabilistic event
            centroids (Centroids): centroids to extract
            countries (list of countries ISO3) selection of countries
                (reg must be None!)
            reg (list of regions): can be set with region code if whole areas
                are considered (if not None, countries and centroids
                are ignored)
            ISINatIDGrid (Bool): Indicates whether ISIMIP_NatIDGrid is used
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
            LOGGER.error('Invalid flood-file path %s', dph_path)
            raise NameError
        if not os.path.exists(frc_path):
            LOGGER.error('Invalid flood-file path %s', frc_path)
            raise NameError

        with xr.open_dataset(dph_path) as flood_dph:
            time = flood_dph.time.data

        event_index = self._select_event(time, years)
        bands = event_index + 1

        if countries or reg:
            # centroids as points
            if ISINatIDGrid:

                dest_centroids = RiverFlood._select_exact_area(countries, reg)[0]
                meta_centroids = copy.copy(dest_centroids)
                meta_centroids.set_lat_lon_to_meta()

                self.set_raster(files_intensity=[dph_path],
                                files_fraction=[frc_path], band=bands.tolist(),
                                transform=meta_centroids.meta['transform'],
                                width=meta_centroids.meta['width'],
                                height=meta_centroids.meta['height'],
                                resampling=Resampling.nearest)
                x_i = ((dest_centroids.lon - self.centroids.meta['transform'][2]) /
                       self.centroids.meta['transform'][0]).astype(int)
                y_i = ((dest_centroids.lat - self.centroids.meta['transform'][5]) /
                       self.centroids.meta['transform'][4]).astype(int)

                fraction = self.fraction[:, y_i * self.centroids.meta['width'] + x_i]
                intensity = self.intensity[:, y_i * self.centroids.meta['width'] + x_i]

                self.centroids = dest_centroids
                self.intensity = sp.sparse.csr_matrix(intensity)
                self.fraction = sp.sparse.csr_matrix(fraction)
            else:
                if reg:
                    iso_codes = region2isos(reg)
                    # envelope containing counties
                    cntry_geom = get_land_geometry(iso_codes)
                    self.set_raster(files_intensity=[dph_path],
                                    files_fraction=[frc_path],
                                    band=bands.tolist(),
                                    geometry=cntry_geom)
                    # self.centroids.set_meta_to_lat_lon()
                else:
                    cntry_geom = get_land_geometry(countries)
                    self.set_raster(files_intensity=[dph_path],
                                    files_fraction=[frc_path],
                                    band=bands.tolist(),
                                    geometry=cntry_geom)
                    # self.centroids.set_meta_to_lat_lon()

        elif shape:
            shapes = gpd.read_file(shape)

            rand_geom = shapes.geometry[0]

            self.set_raster(files_intensity=[dph_path],
                            files_fraction=[frc_path],
                            band=bands.tolist(),
                            geometry=rand_geom)
            return

        elif not centroids:
            # centroids as raster
            self.set_raster(files_intensity=[dph_path],
                            files_fraction=[frc_path],
                            band=bands.tolist())
            # self.centroids.set_meta_to_lat_lon()

        else:  # use given centroids
            # if centroids.meta or grid_is_regular(centroids)[0]:
            """TODO: implement case when meta or regulargrid is defined
                     centroids.meta or grid_is_regular(centroidsxarray)[0]:
                     centroids>flood --> error
                     reprojection, resampling.average (centroids< flood)
                     (transform)
                     reprojection change resampling"""
            # else:
            if centroids.meta:
                centroids.set_meta_to_lat_lon()
            metafrc, fraction = read_raster(frc_path, band=bands.tolist())
            metaint, intensity = read_raster(dph_path, band=bands.tolist())
            x_i = ((centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)
            fraction = fraction[:, y_i * metafrc['width'] + x_i]
            intensity = intensity[:, y_i * metaint['width'] + x_i]
            self.centroids = centroids
            self.intensity = sp.sparse.csr_matrix(intensity)
            self.fraction = sp.sparse.csr_matrix(fraction)

        self.units = 'm'
        self.tag.file_name = dph_path + ';' + frc_path
        self.event_id = np.arange(self.intensity.shape[0])
        self.event_name = list(map(str, years))

        if origin:
            self.orig = np.ones(self.size, bool)
        else:
            self.orig = np.zeros(self.size, bool)

        self.frequency = np.ones(self.size) / self.size

        with xr.open_dataset(dph_path) as flood_dph:
            self.date = np.array([dt.datetime(flood_dph.time[i].dt.year,
                                              flood_dph.time[i].dt.month,
                                              flood_dph.time[i].dt.day).toordinal()
                                  for i in event_index])

    def _select_event(self, time, years):
        """
        Selects events only in specific years and returns corresponding event
        indices
        Parameters:
            time: event time stemps (array datetime64)
            years: years to be selcted (int array)
        Raises:
            KeyError
        Returns:
            event indices (int array)
        """
        event_names = pd.to_datetime(time).year
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('No events found for selected %s', years)
            raise AttributeError
        self.event_name = list(map(str, pd.to_datetime(time[event_index])))
        return event_index

    def exclude_trends(self, fld_trend_path, dis):
        """
        Function allows to exclude flood impacts that are caused in areas
        exposed discharge trends other than the selected one. (This function
        is only needed for very specific applications)
        Raises:
            NameError
        """
        if not os.path.exists(fld_trend_path):
            LOGGER.error('Invalid ReturnLevel-file path %s', fld_trend_path)
            raise NameError
        else:
            metafrc, trend_data = read_raster(fld_trend_path, band=[1])
            x_i = ((self.centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((self.centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)

        trend = trend_data[:, y_i * metafrc['width'] + x_i]

        if dis == 'pos':
            dis_map = np.greater(trend, 0)
        else:
            dis_map = np.less(trend, 0)

        new_trends = dis_map.astype(int)

        new_intensity = np.multiply(self.intensity.todense(), new_trends)
        new_fraction = np.multiply(self.fraction.todense(), new_trends)

        self.intensity = sp.sparse.csr_matrix(new_intensity)
        self.fraction = sp.sparse.csr_matrix(new_fraction)

    def exclude_returnlevel(self, frc_path):
        """
        Function allows to exclude flood impacts below a certain return level
        by manipulating flood fractions in a way that the array flooded more
        frequently than the treshold value is excluded. (This function
        is only needed for very specific applications)
        Raises:
            NameErroris function
        """

        if not os.path.exists(frc_path):
            LOGGER.error('Invalid ReturnLevel-file path %s', frc_path)
            raise NameError
        else:
            metafrc, fraction = read_raster(frc_path, band=[1])
            x_i = ((self.centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((self.centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)
            fraction = fraction[:, y_i * metafrc['width'] + x_i]
            new_fraction = np.array(np.subtract(self.fraction.todense(),
                                                fraction))
            new_fraction = new_fraction.clip(0)
            self.fraction = sp.sparse.csr_matrix(new_fraction)

    def set_flooded_area(self, save_centr=False):
        """
        Calculates flooded area for hazard. sets yearly flooded area and
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

        fla_ann_centr = np.zeros((len(years), len(self.centroids.lon)))
        fla_ev_centr = np.array(np.multiply(self.fraction.todense(),
                                            area_centr))
        self.fla_event = np.sum(fla_ev_centr, axis=1)
        for year_ind in range(len(years)):
            fla_ann_centr[year_ind, :] =\
                np.sum(fla_ev_centr[year_ev_mk[year_ind, :], :],
                       axis=0)
        self.fla_annual = np.sum(fla_ann_centr, axis=1)
        self.fla_ann_av = np.mean(self.fla_annual)
        self.fla_ev_av = np.mean(self.fla_event)
        if save_centr:
            self.fla_ann_centr = sp.sparse.csr_matrix(fla_ann_centr)
            self.fla_ev_centr = sp.sparse.csr_matrix(fla_ev_centr)

    def _annual_event_mask(self, event_years, years):
        """Assignes events to each year
        Returns:
            bool array (columns contain events, rows contain years)
        """
        event_mask = np.full((len(years), len(event_years)), False, dtype=bool)
        for year_ind in range(len(years)):
            events = np.where(event_years == years[year_ind])[0]
            event_mask[year_ind, events] = True
        return event_mask

    def set_flood_volume(self, save_centr=False):
        """Calculates flooded area for hazard. sets yearly flooded area and
            flooded area per event
        Raises:
            MemoryError
        """

        fv_ann_centr = np.multiply(self.fla_ann_centr.todense(), self.intensity.todense())

        if save_centr:
            self.fv_ann_centr = sp.sparse.csr_matrix(self.fla_ann_centr)
        self.fv_annual = np.sum(fv_ann_centr, axis=1)

    @staticmethod
    def _select_exact_area(countries=[], reg=[]):
        """Extract coordinates of selected countries or region
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
        lat, lon = get_region_gridpoints(countries=countries, regions=reg,
                                         basemap="isimip", resolution=150)

        if reg:
            country_isos = region2isos(reg)
        else:
            country_isos = countries

        natIDs = country_iso2natid(country_isos)

        centroids = Centroids()
        centroids.set_lat_lon(lat, lon)
        centroids.id = np.arange(centroids.lon.shape[0])
        # centroids.set_region_id()
        return centroids, country_isos, natIDs
