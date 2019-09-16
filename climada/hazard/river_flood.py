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
from rasterio.warp import Resampling
import copy
from climada.util.constants import NAT_REG_ID, GLB_CENTROIDS_NC
from climada.hazard.base import Hazard
from climada.hazard.centroids import Centroids
from climada.util.coordinates import get_land_geometry, read_raster

NATID_INFO = pd.read_csv(NAT_REG_ID)


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RF'
""" Hazard type acronym RiverFlood"""


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
                    centroids=None, countries=[], reg=None, ISINatIDGrid=False,
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
            if ISINatIDGrid:

                dest_centroids, isos, natID = RiverFlood._select_exact_area(
                    countries, reg)
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

                fraction = self.fraction[:, y_i*self.centroids.meta['width'] + x_i]
                intensity = self.intensity[:, y_i*self.centroids.meta['width'] + x_i]

                self.centroids = dest_centroids
                self.intensity = sp.sparse.csr_matrix(intensity)
                self.fraction = sp.sparse.csr_matrix(fraction)
            else:
                if reg:
                    iso_codes = NATID_INFO['ISO']\
                                [np.isin(NATID_INFO["Reg_name"], reg)].tolist()
                    # envelope containing counties
                    cntry_geom = get_land_geometry(iso_codes)
                    self.set_raster(files_intensity=[dph_path],
                                    files_fraction=[frc_path],
                                    band=bands.tolist(),
                                    geometry=cntry_geom)
                    self.centroids.set_meta_to_lat_lon()
                else:
                    cntry_geom = get_land_geometry(countries)
                    self.set_raster(files_intensity=[dph_path],
                                    files_fraction=[frc_path],
                                    band=bands.tolist(),
                                    geometry=cntry_geom)
                    self.centroids.set_meta_to_lat_lon()

        elif not centroids:
            # centroids as raster
            self.set_raster(files_intensity=[dph_path],
                            files_fraction=[frc_path],
                            band=bands.tolist())
            self.centroids.set_meta_to_lat_lon()
        else:  # use given centroids
            # if centroids.meta or grid_is_regular(centroids)[0]:
            """TODO: implement case when meta or regulargrid is defined
                     centroids.meta or grid_is_regular(centroids)[0]:
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
            fraction = fraction[:, y_i*metafrc['width'] + x_i]
            intensity = intensity[:, y_i*metaint['width'] + x_i]
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
        self.date = np.array([dt.datetime(flood_dph.time[i].dt.year,
                              flood_dph.time[i].dt.month,
                              flood_dph.time[i].dt.day).toordinal()
                              for i in event_index])

    def _select_event(self, time, years):
        event_names = pd.to_datetime(time).year
        event_index = np.where(np.isin(event_names, years))[0]
        if len(event_index) == 0:
            LOGGER.error('No events found for selected ' + str(years))
            raise AttributeError
        self.event_name = list(map(str, pd.to_datetime(time[event_index])))
        return event_index

    def exclude_returnlevel(self, frc_path):
        if not os.path.exists(frc_path):
            LOGGER.error('Invalid ReturnLevel-file path ' + frc_path)
            raise NameError
        else:
            metafrc, fraction = read_raster(frc_path, band=[1])
            x_i = ((self.centroids.lon - metafrc['transform'][2]) /
                   metafrc['transform'][0]).astype(int)
            y_i = ((self.centroids.lat - metafrc['transform'][5]) /
                   metafrc['transform'][4]).astype(int)
            fraction = fraction[:, y_i*metafrc['width'] + x_i]
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
        event_mask = np.full((len(years), len(event_years)), False, dtype=bool)
        for year_ind in range(len(years)):
            events = np.where(event_years == years[year_ind])[0]
            event_mask[year_ind, events] = True
        return event_mask

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
        isimip_grid = xr.open_dataset(GLB_CENTROIDS_NC)
        isimip_lon = isimip_grid.lon.data
        isimip_lat = isimip_grid.lat.data
        gridX, gridY = np.meshgrid(isimip_lon, isimip_lat)
        try:
            if countries:
                if not any(np.isin(NATID_INFO['ISO'], countries)):
                    LOGGER.error('Country ISO3s ' + str(countries) +
                                 ' unknown')
                    raise KeyError
                natID = NATID_INFO["ID"][np.isin(NATID_INFO["ISO"], countries)]
                iso_codes = countries
            elif reg:
                if not any(np.isin(NATID_INFO["Reg_name"], reg)):
                    LOGGER.error('Shortcuts ' + str(reg) + ' unknown')
                    raise KeyError
                natID = NATID_INFO["ID"][np.isin(NATID_INFO["Reg_name"], reg)]
                iso_codes = NATID_INFO["ISO"][np.isin(NATID_INFO["Reg_name"],
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
        # centroids.set_lat_lon_to_meta()
        return centroids, iso_codes, natID
