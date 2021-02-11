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
from scipy import sparse
from scipy.stats import binom
import shapely

from climada.hazard.base import Hazard

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


    def set_ls_hist(self, bbox, path_sourcefile, check_plots=1):
        """
        Set historic landslide (ls) hazard from historical point records,
        for example as can be retrieved from the NASA COOLR initiative,
        which is the largest global ls repository, for a specific geographic
        extent.
        See tutorial for details; the global ls catalog from NASA COOLR can be
        downloaded from https://maps.nccs.nasa.gov/arcgis/apps/webappviewer/index.html?id=824ea5864ec8423fb985b33ee6bc05b7

        Parameters:
        ----------
            bbox (tuple): (minx, miny, maxx, maxy) geographic extent of interest
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

        ls_gdf_bbox = gpd.read_file(path_sourcefile, bbox=bbox)

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

        # TODO: Implement a kwarg option to include affected surroundings
        # around point; see issue on github
        # if incl_surrounding:
        #     LOGGER.info(f'Include surroundings up to {max_dist} metres')
        #     self._incl_affected_surroundings(max_dist)

        self.check()
        self.centroids.check()

        if check_plots == 1:
            self.centroids.plot()

        return self


    def set_ls_prob(self, bbox, path_sourcefile, check_plots=1):
        """
        Set probabilistic annual landslide hazard (fraction, intensity and
        frequency) for a defined bounding box from a raster.
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
        x 1000000 (so be sure to adjust this by dividing through 1m afterwards).
        More details can be found in the landslide tutorial and under above-
        mentioned links.

        The hazard is set such that intensity takes a binary value (0 - no
        ls occurrence probability; 1 - ls occurrence probabilty >0) and
        fraction stores the actual probability * fraction of affected pixel.
        Frequency is 1 everywhere, since the data represents annual occurrence
        probability.

        Impact functions, since they act on the intensity, should hence be in
        the form of a step function,
        defining impact for intensity 0 and (close to) 1.

        Parameters:
        ----------
            bbox : tuple
                (minx, miny, maxx, maxy) geographic extent of interest
            path_sourcefile : str
                path to UNEP/NGI ls hazard file (.tif)

        Returns:
        --------
            self : climada.hazard.Landslide instance
                    probabilistic LS hazard
        """

        # raster by default stored in self.intensity
        self.set_raster([path_sourcefile],
                        geometry=[shapely.geometry.box(*bbox, ccw=True)])

        # set frequ. to 1 everywhere since annual:
        self.frequency = np.array([1])
        # reassign intensity to self.fraction,
        self.fraction = self.intensity.copy()
        # set intensity to 1 wherever there's non-zero fraction
        dense_frac = self.fraction.copy().todense()
        dense_frac[dense_frac != 0] = 1
        self.intensity = sparse.csr_matrix(dense_frac)
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

        if check_plots == 1:
            self.plot_intensity(0)
            self.plot_fraction(0)

        return self

    def sample_events_from_probs(self, n_years):
        """sample an event set for a specified representative time span from
        a hazard layer with annual occurrence probabilities (such as the
        landslide hazard file from UNEP & NGI).
        Draws events from a binomial distribution with p ("success", aka LS
        occurrence, probability = fraction of the cell) and n (number of trials
        aka years in simulation). Fraction is then converted to the integer
        number of "successes" per cell.

        Parameters
        ----------
        n_years : int
            the timespan of the probabilistic simulation in years

        Returns
        -------
        self : climada.hazard.Landslide instance
            with fraction_prob : csr matrix, storing initial probabilities of
            ls occurrence per year per pixel and
            fraction : csr matrix with an integer count of LS occurrences
                within the pixel over the amount of sampled time

        See also
        --------
        set_ls_prob(), scipy.stats.binom.rvs()

        """
        LOGGER.info('Sampling landslide events for a %i year period' % n_years)
        self.fraction_prob = self.fraction.copy()  # save prob values

        for i, j in zip(*self.fraction.nonzero()):
            self.fraction[i, j] = binom.rvs(n=n_years,
                                            p=self.fraction[i, j])
