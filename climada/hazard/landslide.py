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

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binom
import shapely

from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'LS'


def mapping_point2grid(geometry, ymax, xmin, res):
    """Given the coordinates of a point, find the index of a grid cell from 
    a raster into which it falls.
    
    Parameters
    ---------
    geometry : shapely.geometry.Point object
        Point which should be evaluated
    ymax: float
        coords of top left corner of raster file - y
    xmin: float
        coords top left corner of raster file - x
    res: float or tuple
        resolution of raster file. Float if res_x=res_y else (res_x, res_y).
    
    Returns
    ------- 
    col, row : tuple
        column index and row index in grid matrix where point falls into
    """
    if isinstance(res, tuple):
        res_x, res_y = res
    else:
        res_x = res_y = res
    col = int((geometry.x - xmin) / res_x)
    row = int((ymax - geometry.y) / res_y)
    return col, row
    
def mapping_grid2flattened(col, row, matrix_shape):
    """ given a col and row index and the initial 2D matrix shape,
    return the 1-dimensional index of the same point in the flattened matrix
    - assumes concatenation in x-direction 
    
    Parameters
    ----------
    col : int
        Column Index of an entry in the original matrix
    row : int
        Row index of an entry in the original matrix
    Returns
    -------
    index (1D) of the point in the flattened array (int)
    """
    return row * matrix_shape[1] + col

def sample_events_from_probs(prob_matrix, n_years):
    """sample an event set for a specified representative time span from
    a hazard layer with annual occurrence probabilities 
    Draws events from a binomial distribution with p ("success", aka LS
    occurrence, probability = fraction of the cell) and n (number of trials
    aka years in simulation). Fraction is then converted to the integer
    number of "successes" per cell.

    Parameters
    ----------
    prob_matrix : scipy.sparse.csr matrix
        matrix where each entry has an annual probability of occurrence of 
        an event [0,1]
    n_years : int
        the timespan of the probabilistic simulation in years

    Returns
    -------
    prob_matrix : scipy.sparse.csr matrix
        csr matrix with number of success from sampling process per grid cell

    See also
    --------
    set_ls_prob(), scipy.stats.binom.rvs()

    """
    LOGGER.info('Sampling landslide events for a %i year period' % n_years)

    for i, j in zip(*prob_matrix.nonzero()):
        prob_matrix[i, j] = binom.rvs(n=n_years,
                                        p=prob_matrix[i, j])
    return sparse.csr_matrix(prob_matrix)


class Landslide(Hazard):
    """Landslide Hazard set generation.
    Attributes:
    """

    def __init__(self):
        """Empty constructor."""
        Hazard.__init__(self, HAZ_TYPE)
        self.tag.haz_type = 'LS'


    def set_ls_hist(self, bbox, path_sourcefile, res=0.0083333, check_plots=False):
        """
        Set historic landslide (ls) raster hazard from historical point records,
        for example as can be retrieved from the NASA COOLR initiative,
        which is the largest global ls repository, for a specific geographic
        extent.
        Points are assigned to the gridcell they fall into, and the whole grid-
        cell hence counts as equally affected.
        Event frequencies are roughly estimated by 1/(total time passed in event set),
        but such estimates are not meaningful and shouldn't be used for 
        probabilistic calculations! Use the probabilistic method for this!
        
        See tutorial for details; the global ls catalog from NASA COOLR can be
        downloaded from https://maps.nccs.nasa.gov/arcgis/apps/webappviewer/index.html?id=824ea5864ec8423fb985b33ee6bc05b7

        Parameters:
        ----------
        bbox : tuple
            (minx, miny, maxx, maxy) geographic extent of interest
         path_sourcefile : str
             path to shapefile (.shp) with ls point data
        res : float
            resolution in degrees of the final grid cells which are created.
            (1° ~ 111 km at the equator). 
            Affects the hazard extent! Default is 0.00833
        check_plots : bool
            Whether to plot centroids, intensity & fraction of the hazard set.
            
        Returns:
        --------
            self (Landslide() inst.): instance filled with historic LS hazard
                set for either point hazards or polygons with specified
                surrounding extent.
        """
        ls_gdf_bbox = gpd.read_file(path_sourcefile, bbox=bbox)
        
        self.centroids.set_raster_from_pnt_bounds(bbox,res)

        n_ev = len(ls_gdf_bbox)
        
        # assign lat-lon points of LS events to corresponding grid & flattened
        # grid-index
        ls_gdf_bbox = ls_gdf_bbox.reindex(columns = ls_gdf_bbox.columns.tolist() + 
                                          ['col','row'])
        ls_gdf_bbox[['col', 'row']] = ls_gdf_bbox.apply(
            lambda row: mapping_point2grid(row.geometry, bbox[-1],bbox[0], 
                                           res),
            axis = 1).tolist()
        
        ls_gdf_bbox['flat_ix'] = ls_gdf_bbox.apply(
            lambda row: mapping_grid2flattened(row.col, row.row, 
                                               self.centroids.shape), 
            axis = 1)
        
        self.intensity = sparse.csr_matrix(
            (np.ones(n_ev),(np.arange(n_ev),ls_gdf_bbox.flat_ix)), 
            shape=(n_ev, self.centroids.size))
        self.units = 'm/m'
        self.event_id = np.arange(n_ev, dtype=int)
        self.orig = np.ones(n_ev, bool)
        if hasattr(ls_gdf_bbox, 'ev_date'):
            self.date = pd.to_datetime(ls_gdf_bbox.ev_date, yearfirst=True)
        else:
            LOGGER.info('No event dates set from source')
        if not self.date.empty:
            self.frequency = np.ones(n_ev)/(
                (self.date.max()-self.date.min()).value/3.154e+16)
        else: 
            LOGGER.info('No frequency can be derived, no event dates')
        self.fraction = self.intensity.copy()

        self.check()

        if check_plots:
            self.centroids.plot()
            self.plot_intensity(0)

        return self


    def set_ls_prob(self, bbox, path_sourcefile, corr_fact=10e6, n_years=500, 
                    check_plots=False):
        """
        Set probabilistic landslide hazard (fraction, intensity and
        frequency) for a defined bounding box and time period from a raster.
        The hazard data for which this function is explicitly written
        is readily provided by UNEP & the Norwegian Geotechnical Institute
        (NGI), and can be downloaded and unzipped from
        https://preview.grid.unep.ch/index.php?preview=data&events=landslides&evcat=2&lang=eng
        for precipitation-triggered landslide and from
        https://preview.grid.unep.ch/index.php?preview=data&events=landslides&evcat=1&lang=eng
        for earthquake-triggered landslides.
        It works of course with any similar raster file.
        Original data is given in expected annual probability and percentage
        of pixel of occurrence of a potentially destructive landslide event
        x 1000000 (so be sure to adjust this by setting the correction factor).
        More details can be found in the landslide tutorial and under above-
        mentioned links.

        The annual occurrence probabilites are sampled from a binomial 
        distribution; intensity takes a binary value (0 - no
        ls occurrence; 1 - ls occurrence) and
        fraction stores the actual the occurrence count (0 to n) per grid cell.
        Frequency is occurrence count / n_years.

        Impact functions, since they act on the intensity, should hence be in
        the form of a step function,
        defining impact for intensity 0 and (close to) 1.

        Parameters
        ----------
        bbox : tuple
            (minx, miny, maxx, maxy) geographic extent of interest
        path_sourcefile : str
            path to UNEP/NGI ls hazard file (.tif)
        corr_fact : float or int
            factor by which to divide the values in the original probability
            file, in case not scaled to [0,1]. Default is 1'000'000
        n_years : int
            sampling period
        check_plots : bool
            Whether to plot centroids, intensity & fraction of the hazard set.

        Returns
        -------
        self : climada.hazard.Landslide instance
            probabilistic LS hazard
            
        See also
        --------
        sample_events_from_probs()
        """

        # raster with occurrence probs  by default stored in self.intensity:
        self.set_raster([path_sourcefile],
                        geometry=[shapely.geometry.box(*bbox, ccw=True)])

        # sample events from probabilities
        self.fraction = sample_events_from_probs(self.intensity/corr_fact,
                                                  n_years)
            
        # set frequ. to no. of occurrences per cell / timespan:
        self.frequency = self.fraction/n_years
        
        # set intensity to 1 wherever an occurrence:
        self.intensity = self.fraction.copy()
        self.intensity[self.intensity.nonzero()]=1       
    
        # meaningless, such that check() method passes:
        self.date = np.array([])
        self.event_name = []

        self.centroids.set_raster_file(path_sourcefile,
                                       geometry=[shapely.geometry.box(
                                           *bbox, ccw=True)])        
        if "unnamed" in self.centroids.crs.wkt:
            self.centroids.meta['crs'] = {'init': 'epsg:4326', 'no_defs': True}
     
        self.centroids.set_geometry_points()
        self.check()

        if check_plots:
            self.plot_intensity(0)
            self.plot_fraction(0)

        return self
