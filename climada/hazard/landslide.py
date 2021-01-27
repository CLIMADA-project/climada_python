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
Define Landslide class.
"""

__all__ = ['Landslide']

import logging
from pathlib import Path
import glob

import shlex
import subprocess
from scipy import sparse
from scipy.stats import binom
import geopandas
import pyproj
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import requests
import geopandas as gpd
import numpy as np
import shapely
from haversine import haversine

from climada import CONFIG
from climada.hazard.base import Hazard
from climada.util.constants import SYSTEM_DIR as LS_FILE_DIR

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'LS'


class Landslide(Hazard):
    """Landslide Hazard set generation.
    Attributes:
    """

    def __init__(self):
        """Empty constructor."""
        Hazard.__init__(self, HAZ_TYPE)
        self.tag.haz_type = 'LS'


    def _get_window_from_coords(self, path_sourcefile, bbox=[]):
        ###### would fit better into base calss for sub-function of hazard.set_raster()########
        """get row, column, width and height required for rasterio window function
        from coordinate values of bounding box
        Parameters:
            bbox (array): [north, east, south, west]
            large_file (str): path of file from which window should be read in
        Returns:
            window_array (array): corner, width & height for Window() function of rasterio
        """
        with rasterio.open(path_sourcefile) as src:
            utm = pyproj.Proj(init='epsg:4326')  # Pass CRS of image from rasterio

        lonlat = pyproj.Proj(init='epsg:4326')
        lon, lat = (bbox[3], bbox[0])
        west, north = pyproj.transform(lonlat, utm, lon, lat)

        # What is the corresponding row and column in our image?
        row, col = src.index(west, north)  # spatial --> image coordinates

        lon, lat = (bbox[1], bbox[2])
        east, south = pyproj.transform(lonlat, utm, lon, lat)
        row2, col2 = src.index(east, south)
        width = abs(col2 - col)
        height = abs(row2 - row)

        window_array = [col, row, width, height]

        return window_array

    def _get_raster_meta(self, path_sourcefile, window_array):
        """get geo-meta data from raster files to set centroids adequately"""
        raster = rasterio.open(path_sourcefile, 'r',
                               window=Window(window_array[0], window_array[1],
                                             window_array[2], window_array[3]))
        pixel_width = raster.meta['transform'][0]
        pixel_height = raster.meta['transform'][4]

        return pixel_height, pixel_width

    def _intensity_cat_to_prob(self, max_prob):
        """convert NASA nowcasting categories into occurrence probabilities:
            highest value category value receives a prob of max_prob, lowest category value
            receives a prob value of 0"""
        self.intensity_cat = self.intensity.copy()  # save prob values
        self.intensity = self.intensity.astype(float)
        self.intensity.data = self.intensity.data.astype(float)
        max_value = float(max(self.intensity_cat.data))
        min_value = float(min(self.intensity_cat.data))

        for i, j in zip(*self.intensity.nonzero()):
            self.intensity[i, j] = float((self.intensity[i, j] - min_value) /
                                         (max_value - min_value) * max_prob)


    def _intensity_prob_to_binom(self, n_years):
        """convert occurrence probabilities in NGI/UNEP landslide hazard map into binary
        occurrences (yes/no) within a given time frame.

        Parameters
        ----------
        n_years : int
            the timespan of the probabilistic simulation in years

        Returns
        -------
        intensity_prob : csr matrix
            initial probabilities of ls occurrence per year per pixel
        intensity : csr matrix
            binary (0/1) occurrence within pixel
        """

        self.intensity_prob = self.intensity.copy()  # save prob values

        for i, j in zip(*self.intensity.nonzero()):
            if binom.rvs(n=n_years, p=self.intensity[i, j]) >= 1:
                self.intensity[i, j] = 1
            else:
                self.intensity[i, j] = 0

    def _intensity_binom_to_range(self, max_dist):
        """Affected neighbourhood' of pixels within certain threshold from ls occurrence
        can be included (takes long to compute, though).
        Parameters:
            max_dist (int): distance in metres (up to max ~1100) until which
                neighbouring pixels count as affected.
        Returns:
            intensity (csr matrix): range (0-1) where 0 = no occurrence, 1 = direct
                occurrence, ]0-1[ = relative distance to pixel with direct occurrence
        """
        self.intensity = self.intensity.tolil()
        # find all other pixels within certain distance from corresponding centroid,
        for i, j in zip(*self.intensity.nonzero()):
            subset_neighbours = self.centroids.geometry.cx[
                (self.centroids.coord[j][1] - 0.01):(self.centroids.coord[j][1] + 0.01),
                (self.centroids.coord[j][0] - 0.01):(self.centroids.coord[j][0] + 0.01)
            ]  # 0.01° = 1.11 km approximately
            for centroid in subset_neighbours:
                ix = subset_neighbours[subset_neighbours == centroid].index[0]
                # calculate dist, assign intensity [0-1] linearly until max_dist
                if haversine(self.centroids.coord[ix], self.centroids.coord[j], unit='m')\
                <= max_dist:
                    actual_dist = haversine(
                        self.centroids.coord[ix],
                        self.centroids.coord[j], unit='m')
                    # this step changes sparsity of matrix -->
                    # converted to lil_matrix, as more efficient
                    self.intensity[i, ix] = (max_dist - actual_dist) / max_dist
        self.intensity = self.intensity.tocsr()

    def plot_raw(self, ev_id=1, **kwargs):
        """Plot raw LHM data using imshow and without cartopy

        Parameters:
            ev_id (int, optional): event id. Default: 1.
            intensity (bool, optional): plot intensity if True, fraction otherwise
            kwargs (optional): arguments for imshow matplotlib function

        Returns:
            matplotlib.image.AxesImage
        """
        if not self.centroids.meta:
            LOGGER.error('No raster data set')
            raise ValueError
        try:
            event_pos = np.where(self.event_id == ev_id)[0][0]
        except IndexError:
            LOGGER.error('Wrong event id: %s.', ev_id)
            raise ValueError from IndexError

        return plt.imshow(self.intensity_prob[event_pos, :].toarray().
                          reshape(self.centroids.shape), **kwargs)

    def _incl_affected_surroundings(self, max_dist):
        """
        Change centroids' geometry from POINT to circular POLYGON within 
        given radius.
        
        Parameters
        ----------
            max_dist (int): distance in metres until which
                surroundings of a shapely.Point count as affected
        Returns
        -------
            
        """
        self.centroids.set_geometry_points()
        self.centroids.geometry = self.centroids.geometry.buffer(
            deg_from_dist(max_dist))
        

    def _gdf_from_bbox(self, bbox, path_sourcefile):
        """
        load geo-dataframe with from shp-file for certain bounding box
        
        Parameters
        ----------
            bbox (list): [N, E , S, W] geographic extent of interest
            path_sourcefile (str): path to shapefile with point data
        Returns
        --------
            (gdf): geopandas geodataframe with points inside bbox
        """
        return gpd.read_file(path_sourcefile).cx[bbox[3]:bbox[1], 
                                                 bbox[2]:bbox[0]]


    def set_ls_hist(self, bbox, path_sourcefile, incl_surrounding=False, 
                          max_dist=100, check_plots=1):
        """
        Set historic landslide (ls) hazard from historical point records, 
        for example as can be retrieved from the NASA COOLR initiative,
        which is the largest global ls repository, for a specific geographic
        extent.
        Also, include affected surrounding by defining a circular extent around
        ls points (intensity is binary - 0 marking no ls, 1 marking ls 
        occurrence.
        See tutorial for details; the global ls catalog from NASA COOLR can be
        downloaded from https://maps.nccs.nasa.gov/arcgis/apps/webappviewer/index.html?id=824ea5864ec8423fb985b33ee6bc05b7
        
        Parameters:
        ----------
            bbox (list): [N, E , S, W] geographic extent of interest
            path_sourcefile (str): path to shapefile (.shp) with ls point data
            incl_surrounding (bool): default is False. Whether to include 
                circular surroundings from point hazard as affected area.
            max_dist (int): distance in metres around which point hazard should
                be extended to if incl_surrounding = True. Default is 100m.
        Returns:
        --------
            self (Landslide() inst.): instance filled with historic LS hazard 
                set for either point hazards or polygons with specified 
                surrounding extent.
        """
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()

        if not path_sourcefile:
            LOGGER.error('No sourcefile (.shp) with historic LS points given')
            raise ValueError()

        ls_gdf_bbox = self._gdf_from_bbox(bbox, path_sourcefile)

        self.centroids.set_lat_lon(ls_gdf_bbox.geometry.y, 
                                   ls_gdf_bbox.geometry.x)
        
        n_cen = n_ev = len(ls_gdf_bbox)

        self.intensity = sparse.csr_matrix(
            np.diag(np.diag(np.ones((n_ev, n_cen)))))
        self.units = 'm/m'
        self.event_id = np.arange(n_ev, dtype=int)
        self.orig = np.ones(n_ev, bool)
        if hasattr(ls_gdf_bbox, 'ev_date'):
            self.date = ls_gdf_bbox.ev_date
        else:
            LOGGER.info('No event dates set from source')
        self.frequency = np.ones(n_ev)/n_ev
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1)
        
        if incl_surrounding:
            LOGGER.info(f'Include surroundings up to {max_dist} metres')
            self._incl_affected_surroundings(max_dist)
        
        self.check()
        self.centroids.check()

        if check_plots == 1:
            self.centroids.plot()
        
        return self


    def set_ls_prob(self, bbox, path_sourcefile, check_plots=1):
        """
        Set probabilistic annual landslide hazard (fraction, intensity and 
        frequency) for a defined bounding box.
        The hazard data for which this function is explicitly written
        is readily provided by UNEP & the Norwegian Geotechnical Institute 
        (NGI), and can be downloaded and unzipped from 
        https://preview.grid.unep.ch/index.php?preview=data&events=landslides&evcat=2&lang=eng 
        for precipitation-triggered landslide and from
        https://preview.grid.unep.ch/index.php?preview=data&events=landslides&evcat=1&lang=eng 
        for earthquake-triggered landslides.
        Original data is given in expected annual probability and percentage 
        of pixel of occurrence of a potentially destructive landslide event
        x 1000000.
        More details can be found in the landslide tutorial and under above-
        mentioned links.
        
        The data is structured such that intensity takes a binary value (0 - no
        ls occurrence probability; 1 - ls occurrence probabilty >0) and 
        fraction stores the actual probability * fraction of affected pixel. 
        Frequency is 1 everywhere, since the data represents annual occurrence 
        probability.
        
        Impact functions should hence be in the form of a step function, 
        defining impact for intensity 0 and (near to) 1.
        
        Parameters:
        ----------
            bbox (array): [N, E , S, W] geographic extent of interest
            path_sourcefile (str):  path to UNEP/NGI ls hazard file (.tif) 
                     
        Returns:
        --------
            self (Landslide() instance): probabilistic LS hazard 
        """
        
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()
        
        if not path_sourcefile:
            LOGGER.error('Empty path to landslide hazard set, please specify.')
            raise ValueError()
        
        # read in hazard set raster (by default stored in self.intensity)
        self.set_raster([path_sourcefile], 
                        geometry=[shapely.geometry.box(*bbox[::-1], ccw=True)])
        
        # specify annual frequency:
        self.frequency = np.array([1])
        # reassign intensity to self.fraction, correct by factor 1mio, 
        self.fraction = self.intensity.copy()/10e6
        # set intensity to 1 wherever there's non-zero fraction
        dense_frac = self.fraction.copy().todense()
        dense_frac[dense_frac!=0] = 1
        self.intensity = sparse.csr_matrix(dense_frac)
        # meaningless, such that check() method passes:
        self.date= np.array([])
        self.event_name = []
        
        self.centroids.set_raster_file(path_sourcefile, 
                                       geometry=[shapely.geometry.box(*bbox[::-1], ccw=True)])
        if "unnamed" in self.centroids.crs.wkt:
            self.centroids.meta['crs'] = {'init': 'epsg:4326', 'no_defs': True}
        
        self.check()
        self.centroids.check()

        if check_plots == 1:
            self.plot_intensity(0)
            self.plot_fraction(0)

        return self
