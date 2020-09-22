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
from scipy import sparse
import geopandas as gpd
import pyproj
import rasterio
from rasterio.windows import Window
import numpy as np
from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'LS'


class Landslide(Hazard):
    """Landslide Hazard set generation.
    Attributes:
    """

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.tag.haz_type = 'LS'


    def _get_window_from_coords(self, path_sourcefile, bbox=[]):
        #TODO: decide whether to move this into hazard base as sub-function of hazard.set_raster()
        """
        get row, column, width and height required for rasterio window function
        from coordinate values of bounding box
        
        Parameters:
            bbox (list): [north, east, south, west]
            path_sourcefile (str): path of file from which window should be read in
        Returns:
            window_array (array): corner, width & height for Window() function of rasterio
        """
        with rasterio.open(path_sourcefile) as src:
            utm = pyproj.Proj(init='epsg:4326') # Pass CRS of image from rasterio
            lonlat = pyproj.Proj(init='epsg:4326')
            lon, lat = (bbox[3], bbox[0])
            west, north = pyproj.transform(lonlat, utm, lon, lat)

            row, col = src.index(west, north) # spatial --> image coordinates

            lon, lat = (bbox[1], bbox[2])
            east, south = pyproj.transform(lonlat, utm, lon, lat)
            row2, col2 = src.index(east, south)
        width = abs(col2-col)
        height = abs(row2-row)

        window_array = [col, row, width, height]

        return window_array


    def _get_raster_meta(self, path_sourcefile, window_array):
        #TODO: decide whether to move this into hazard base as sub-function of hazard.set_raster()
        """
        get geo-meta data from raster files to set centroids adequately
        """
        raster = rasterio.open(path_sourcefile, 'r', \
                               window=Window(window_array[0], window_array[1],\
                                               window_array[2], window_array[3]))
        pixel_width = raster.meta['transform'][0]
        pixel_height = raster.meta['transform'][4]

        return pixel_height, pixel_width


    def _incl_affected_surroundings(self, max_dist):
        """
        Affected neighbourhood' of points within certain threshold from ls 
        POINT occurrence
        
        Parameters:
            max_dist (int): distance in metres until which
                surroundings of a shapely.Point count as affected
        Returns:
            
        """
        # TODO: redo this function to incorporate neighbourhood of points!
        self.centroids.set_meta_to_lat_lon()
        self.centroids.set_geometry_points()
        buffer_deg = 'tranfo function from metres to degrees given projection'
        self.centroids.geometry = self.centroids.geometry.buffer(buffer_deg)

    def _get_hist_events(self, bbox, path_sourcefile):
        """
        load geo-df with landslide event poits from a global landslide (ls)
        catalog for certain bounding box
        
        Parameters:
            bbox (list): [N, E , S, W] geographic extent of interest
            path_sourcefile (str): path to shapefile with ls point data
        Returns:
            (gdf): geopandas dataframe with ls points inside bbox
        """
        return gpd.read_file(path_sourcefile).cx[bbox[3]:bbox[1], bbox[2]:bbox[0]]


    def set_ls_hist(self, bbox, path_sourcefile, incl_surrounding=False, 
                          max_dist=1000, check_plots=1):
        """
        set landslide (ls) hazard from historical point records, e.g. as 
        documented by in the NASA COOLR initiative (see tutorial for source)
        
        Parameters:
            bbox (list): [N, E , S, W] geographic extent of interest
            path_sourcefile (str): path to shapefile with ls point data
            max_dist (int): distance until which neighbouring pixels should count as affected
                if incl_neighbour = True. Default is 1000m.
        Returns:
            Landslide() module: historic LS hazard set
        """
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()

        if not path_sourcefile:
            LOGGER.error('No sourcefile (.shp) with historic LS points given')
            raise ValueError()

        ls_gdf_bbox = self._get_hist_events(bbox, path_sourcefile)

        self.centroids.set_lat_lon(ls_gdf_bbox.geometry.y, 
                                   ls_gdf_bbox.geometry.x)
        
        n_cen = n_ev = len(ls_gdf_bbox)

        self.intensity = sparse.csr_matrix(np.diag(np.diag(np.ones((n_ev, n_cen)))))
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
        self.check()
        
        # TODO: incorporate conversion of points to raster within specified distance
        if incl_surrounding:
            LOGGER.info('Finding neighbouring pixels...')
            self._incl_affected_surroundings(max_dist)
            self.check()      

        if check_plots == 1:
            self.centroids.plot()
            self.plot_intensity(0)
        return self

    def set_ls_prob(self, bbox, path_sourcefile, check_plots=1):
        """
        Parameters:
            bbox (array): [N, E , S, W] geographic extent of interest
            path_sourcefile (str):  path to UNEP landslide hazard file, with 
                annual occurrence probabilities (see tutorial for retrieval)
            
        Returns:
            Landslide() instance: probabilistic LS hazard 
        """
        
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()
        
        if not path_sourcefile:
            LOGGER.error('Empty path to landslide hazard set, please specify.')
            raise ValueError()

        window_array = self._get_window_from_coords(path_sourcefile,bbox)
        pixel_height, pixel_width = self._get_raster_meta(path_sourcefile,
                                                          window_array)
        
        # read in hazard set raster (by default stored in self.intensity)
        self.set_raster([path_sourcefile], 
                        window=Window(window_array[0], window_array[1],
                                      window_array[2], window_array[3]))
        
        # UNEP / NGI raster file refers to annual frequency * fraction of affected pixel *1 mio
        # --> correct, such that intensity = 1 wherever nonzero, and fraction accurate
        # TODO: verify that splitting of intensity & fraction logically consistent & used in impact calc. 
        self.frequency = np.array([1])
        self.fraction = self.intensity.copy()/10e6
        # set intensity to 1 wherever there's non-zero occurrence frequency
        # TODO: check why intensities < 1 still exist in plot.
        dense_frac = self.fraction.copy().todense()
        dense_frac[dense_frac!=0] = 1
        self.intensity = sparse.csr_matrix(dense_frac)

        # meaningless, such that check() method passes:
        self.date= np.array([])
        self.event_name = []
        
        self.centroids.set_raster_from_pix_bounds(bbox[0], bbox[3], 
                                                  pixel_height, 
                                                  pixel_width,
                                                  window_array[3], 
                                                  window_array[2])
        self.check()

        if check_plots == 1:
            self.plot_intensity(0)
            self.plot_fraction(0)

        return self
