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
import os
from scipy import sparse
import geopandas
import pyproj
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
import numpy as np
from haversine import haversine
from climada.hazard.base import Hazard
from climada.util.constants import DATA_DIR

LOGGER = logging.getLogger(__name__)

LS_FILE_DIR = os.path.join(DATA_DIR, 'system')

HAZ_TYPE = 'LS'


""" for future: implement a function that downloads COOLR data by command, not manually"""
# def get_coolr_shp(save_path=os.getcwd()):
#    """for LS_MODEL[0]: download most up-to-date version of historic LS records from
#    global landslide catalog (COOLR of NASA) in shape-file format (zip)"""

#   trials didn't work.
#    url = 'https://maps.nccs.nasa.gov/arcgis/home/item.html?id=ff4478ca84d24766bd79ac186bb60d9c#data'
#    resp_glc = requests.get(url=url)

#    url = 'https://data.nasa.gov/api/geospatial/h9d8-neg4?method=export&format=Shapefile'
#    # Timeout error, website is currently not working
#    LOGGER.info('requesting data from %s', url)
#    response = requests.get(url=url)
#    LOGGER.info('downloading content...')
#    open((save_path+'/global_LS_catalogue'+'.zip'), 'wb').write(response.content)
#


class Landslide(Hazard):
    """Landslide Hazard set generation.
    Attributes:
    """

    def __init__(self):
        """Empty constructor. """
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
        """get geo-meta data from raster files to set centroids adequately"""
        raster = rasterio.open(path_sourcefile, 'r', \
                               window=Window(window_array[0], window_array[1],\
                                               window_array[2], window_array[3]))
        pixel_width = raster.meta['transform'][0]
        pixel_height = raster.meta['transform'][4]

        return pixel_height, pixel_width


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
                (self.centroids.coord[j][1]-0.01):(self.centroids.coord[j][1]+0.01),
                (self.centroids.coord[j][0]-0.01):(self.centroids.coord[j][0]+0.01)
                ]# 0.01° = 1.11 km approximately
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
                    self.intensity[i, ix] = (max_dist-actual_dist)/max_dist
        self.intensity = self.intensity.tocsr()

    def plot_raw(self, ev_id=1, **kwargs):
        """ Plot raw LHM data using imshow and without cartopy

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

        return plt.imshow(self.intensity[event_pos, :].todense(). \
                              reshape(self.centroids.shape), **kwargs)

    def plot_events(self, ev_id=1, **kwargs):
        """ Plot LHM event data using imshow and without cartopy

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

        return plt.imshow(self.intensity[event_pos, :].todense(). \
                              reshape(self.centroids.shape), **kwargs)

    def _get_hist_events(self, bbox, coolr_path):
        """for LS_MODEL[0]: load gdf with landslide event POINTS from
        global landslide catalog (COOLR of NASA) for bbox of interest"""

        try:
            ls_gdf = geopandas.read_file(coolr_path)
        except:
            raise ValueError("The nasa_global_landslide_catalog_point files could not be found."\
                             + "The script looked for the files on the following path: {}".format(coolr_path))
        ls_gdf_bbox = ls_gdf.cx[bbox[3]:bbox[1], bbox[2]:bbox[0]]
        return ls_gdf_bbox

    def set_ls_model_hist(self, bbox, path_sourcefile, check_plots=1):
        """
        set LS from historical records documented in the NASA COOLR initiative
        
        Parameters:
            bbox (array): [N, E , S, W] for which LS hazard should be calculated.
            path_sourcefile (str): path to shapefile with COOLR data, retrieved previously as
                described in tutorial
        Returns:
            Landslide() module: LS hazard set, historic
        """
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()

        if not path_sourcefile:
            LOGGER.error('No sourcefile, please specify one containing historic LS points')
            raise ValueError()

        ls_gdf_bbox = self._get_hist_events(bbox, path_sourcefile)

        self.centroids.set_lat_lon(ls_gdf_bbox.latitude, ls_gdf_bbox.longitude)
        n_cen = ls_gdf_bbox.latitude.size # number of centroids
        n_ev = n_cen

        self.intensity = sparse.csr_matrix(np.diag(np.diag(np.ones((n_ev, n_cen)))))
        self.units = 'm/m'
        self.event_id = np.arange(n_ev, dtype=int)
        self.orig = np.ones(n_ev, bool)
        self.date = ls_gdf_bbox.ev_date
        self.frequency = np.ones(n_ev)/n_ev
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1)
        self.check()

        if check_plots == 1:
            self.centroids.plot()
        return self

    def set_ls_model_prob(self, bbox, path_sourcefile = [], 
                          incl_neighbour=False, max_dist=1000, check_plots=1):
        """
        Parameters:
            bbox (array): [N, E , S, W] for which LS hazard should be calculated
            incl_neighbour (bool): whether to include affected neighbouring pixels
                with dist <= max_dist. Default is false
            max_dist (int): distance until which neighbouring pixels should count as affected
                if incl_neighbour = True. Default is 1000m.
            path_sourcefile (str):  path to NGI/UNEP file,
                retrieved previously as descriped in tutorial and stored in climada/data.
        Returns:
            Landslide() module: probabilistic LS hazard set
        """

        path_sourcefile = os.path.join(LS_FILE_DIR, 'ls_pr_NGI_UNEP/ls_pr.tif')
        
        if not bbox:
            LOGGER.error('Empty bounding box, please set bounds.')
            raise ValueError()

        window_array = self._get_window_from_coords(path_sourcefile,\
                                                    bbox)
        pixel_height, pixel_width = self._get_raster_meta(path_sourcefile, window_array)
        self.set_raster([path_sourcefile], window=Window(window_array[0], window_array[1],\
                                        window_array[2], window_array[3]))
        n_ev = self.intensity.nnz
        n_cen = self.intensity.shape[1]
        self.intensity_temp = []
        for i, prob_val in enumerate(np.squeeze(np.asarray(self.intensity.todense()))):
            if prob_val:
                event_intensity = np.zeros(n_cen)
                event_intensity[i] = 1
                if self.intensity_temp==[]:
                    self.intensity_temp = event_intensity
                else:
                    self.intensity_temp = sparse.vstack([
                            sparse.csr_matrix(self.intensity_temp), 
                            sparse.csr_matrix(event_intensity)])
        
        self.frequency = self.intensity.data/10e6
        self.intensity = sparse.csr_matrix(self.intensity_temp.copy())
        self.fraction = self.intensity.copy()
        self.event_id = np.arange(n_ev, dtype=int)
        self.event_name = self.event_id.copy()
        self.orig = np.ones(n_ev, bool)
        self.date= np.array([])
        self.event_name = []
        
        self.centroids.set_raster_from_pix_bounds(bbox[0], bbox[3], pixel_height, pixel_width,\
                                                  window_array[3], window_array[2])
                    
        LOGGER.info('Generating landslides...')
        self.check()

        if incl_neighbour:
            LOGGER.info('Finding neighbouring pixels...')
            self.centroids.set_meta_to_lat_lon()
            self.centroids.set_geometry_points()
            self._intensity_binom_to_range(max_dist)
            self.check()

        if check_plots == 1:
            self.plot_intensity(0)

        return self


