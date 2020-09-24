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
import numpy as np
from climada.hazard.base import Hazard
import shapely

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
                          max_dist=100, check_plots=1):
        """
        Set historic landslide (ls) hazard from historical point records, 
        for example as can be retrieved from the NASA COOLR initiative,
        which is the largest global ls repository, for a specific geographic
        extent.
        Also, include affected surrounding by defining a circular extent around
        ls points (intensity is binary - 0 marking no ls, 1 marking ls occurrence.
        See tutorial for details; the global ls catalog from NASA COOLR can be
        downloaded from 
        
        Parameters:
            bbox (list): [N, E , S, W] geographic extent of interest
            path_sourcefile (str): path to shapefile (.shp) with ls point data
            incl_surrounding (bool): default is False. Whether to include circular
                surroundings from point hazard as affected area.
            max_dist (int): distance in metres until which 
                if incl_neighbour = True. Default is 100m.
        Returns:
            self (Landslide inst.): instance filled with historic LS hazard set
                for either point hazards or polygons with specified surrounding extent.
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
        ls occurrence probability; 1 - ls occurrence probabilty >0) and fraction
        stores the actual probability * fraction of affected pixel. Frequency 
        is 1 everywhere, since the data represents annual occurrence probability.
        
        Impact functions should hence be in the form of a step function, defining
        impact for intensity 0 and (near to) 1.
        
        Parameters:
            bbox (array): [N, E , S, W] geographic extent of interest
            path_sourcefile (str):  path to UNEP/NGI landslide hazard (.tif) file     
        Returns:
            Landslide() instance: probabilistic LS hazard 
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
