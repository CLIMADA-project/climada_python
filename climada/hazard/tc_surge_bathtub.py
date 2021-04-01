"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define TCSurgeBathtub class.
"""

__all__ = ['TCSurgeBathtub']

import copy
import logging

import numpy as np
import rasterio.warp
import scipy.sparse as sp

from climada.hazard.base import Hazard
import climada.util.coordinates as u_coord

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TCSurgeBathtub'
"""Hazard type acronym for this module"""

MAX_DIST_COAST = 50
"""Maximum inland distance of the centroids in km."""

MAX_ELEVATION = 10
"""Maximum elevation of the centroids in m."""

MAX_LATITUDE = 61
"""Maximum latitude of potentially affected centroids."""


class TCSurgeBathtub(Hazard):
    """TC surge heights in m, a bathtub model with wind-surge relationship and inland decay."""

    def __init__(self):
        Hazard.__init__(self, HAZ_TYPE)


    @staticmethod
    def from_tc_winds(wind_haz, topo_path, inland_decay_rate=0.2, add_sea_level_rise=0.0):
        """Compute tropical cyclone surge from input winds.

        Parameters
        ----------
        wind_haz : TropCyclone
            Tropical cyclone wind hazard object.
        topo_path : str
            Path to a raster file containing gridded elevation data.
        inland_decay_rate : float, optional
            Decay rate of surge when moving inland in meters per km. Set to 0 to deactivate
            this effect. The default value of 0.2 is taken from Section 5.2.1 of the monograph
            Pielke and Pielke (1997): Hurricanes: their nature and impacts on society.
            https://rogerpielkejr.com/2016/10/10/hurricanes-their-nature-and-impacts-on-society/
        add_sea_level_rise : float, optional
            Sea level rise effect in meters to be added to surge height.
        """
        centroids = copy.deepcopy(wind_haz.centroids)

        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        # Select wind-affected centroids which are inside MAX_DIST_COAST and |lat| < 61
        if not centroids.dist_coast.size or np.all(centroids.dist_coast >= 0):
            centroids.set_dist_coast(signed=True, precomputed=True)
        coastal_msk = (wind_haz.intensity > 0).sum(axis=0).A1 > 0
        coastal_msk &= (centroids.dist_coast < 0)
        coastal_msk &= (centroids.dist_coast >= -MAX_DIST_COAST * 1000)
        coastal_msk &= (np.abs(centroids.lat) <= MAX_LATITUDE)

        # Load elevation at coastal centroids
        coastal_centroids_h = u_coord.read_raster_sample(
            topo_path, centroids.lat[coastal_msk], centroids.lon[coastal_msk])

        # Update selected coastal centroids to exclude high-lying locations
        # We only update the previously selected centroids (for which elevation info was obtained)
        elevation_msk = (coastal_centroids_h >= 0)
        elevation_msk &= (coastal_centroids_h <= MAX_ELEVATION + add_sea_level_rise)
        coastal_msk[coastal_msk] = elevation_msk

        # Elevation data and coastal/non-coastal indices are used later in the code
        coastal_centroids_h = coastal_centroids_h[elevation_msk]
        coastal_idx = coastal_msk.nonzero()[0]
        noncoastal_idx = (~coastal_msk).nonzero()[0]

        # Initialize intensity array at coastal centroids
        inten_surge = wind_haz.intensity.copy()
        inten_surge[:,noncoastal_idx] *= 0
        inten_surge.eliminate_zeros()

        # Conversion of wind to surge using the linear wind-surge relationship from
        # figure 2 of the following paper:
        #
        #   Xu, Liming (2010): A Simple Coastline Storm Surge Model Based on Pre-run SLOSH Outputs.
        #   In: 29th Conference on Hurricanes and Tropical Meteorology, 10–14 May. Tucson, Arizona.
        #   https://ams.confex.com/ams/pdfpapers/168806.pdf
        inten_surge.data = 0.1023 * np.fmax(inten_surge.data - 26.8224, 0) + 1.8288

        if inland_decay_rate != 0:
            # Add decay according to distance from coast
            dist_coast_km = np.abs(centroids.dist_coast[coastal_idx]) / 1000
            coastal_centroids_h += inland_decay_rate * dist_coast_km
        coastal_centroids_h -= add_sea_level_rise

        # Efficient way to subtract from selected columns of sparse csr matrix
        nz_coastal_cents = inten_surge[:,coastal_idx].nonzero()
        nz_coastal_cents_inten = (nz_coastal_cents[0], coastal_idx[nz_coastal_cents[1]])
        inten_surge[nz_coastal_cents_inten] -= coastal_centroids_h[nz_coastal_cents[1]]

        # Discard negative (invalid/unphysical) surge height values
        inten_surge.data = np.fmax(inten_surge.data, 0)
        inten_surge.eliminate_zeros()

        # Get fraction of (large) centroid cells on land according to the given (high-res) DEM
        fract_surge = sp.csr_matrix(_fraction_on_land(centroids, topo_path))
        fract_surge = sp.csr_matrix(np.ones((inten_surge.shape[0], 1))) * fract_surge

        # Set other attributes
        haz = TCSurgeBathtub()
        haz.centroids = centroids
        haz.units = 'm'
        haz.event_id = wind_haz.event_id
        haz.event_name = wind_haz.event_name
        haz.date = wind_haz.date
        haz.orig = wind_haz.orig
        haz.frequency = wind_haz.frequency
        haz.intensity = inten_surge
        haz.fraction = fract_surge
        return haz


def _fraction_on_land(centroids, topo_path):
    """Determine fraction of each centroid cell that is on land.

    Typically, the resolution of the provided DEM data set is much higher than the resolution
    of the centroids so that the centroid cells might be partly above and partly below sea level.
    This function computes for each centroid cell the fraction of its area that is on land.

    Parameters
    ----------
    centroids : Centroids
        Centroids to consider
    topo_path : str
        Path to a raster file containing gridded elevation data.

    Returns
    -------
    fractions : ndarray of shape (ncentroids,)
        For each centroid, the fraction of it's cell area that is on land according to the DEM.
    """
    bounds = np.array(centroids.total_bounds)
    shape = [0, 0]
    if centroids.meta:
        shape = centroids.shape
        cen_trans = centroids.meta['transform']
    else:
        shape[0], shape[1], cen_trans = u_coord.pts_to_raster_meta(
            bounds, min(u_coord.get_resolution(centroids.lat, centroids.lon)))

    read_raster_buffer = 0.5 * max(np.abs(cen_trans[0]), np.abs(cen_trans[4]))
    bounds += read_raster_buffer * np.array([-1., -1., 1., 1.])
    on_land, dem_trans = u_coord.read_raster_bounds(topo_path, bounds)
    on_land = (on_land > 0).astype(np.float64)

    with rasterio.open(topo_path, 'r') as src:
        dem_crs = src.crs
        dem_nodata = src.nodata

    fractions = np.zeros(shape, dtype=np.float64)
    rasterio.warp.reproject(source=on_land, destination=fractions,
                            src_transform=dem_trans, src_crs=dem_crs,
                            dst_transform=cen_trans, dst_crs=centroids.crs,
                            resampling=rasterio.warp.Resampling.average,
                            src_nodata=dem_nodata, dst_nodata=0.0)

    if not centroids.meta:
        x_i = ((centroids.lon - cen_trans[2]) / cen_trans[0]).astype(int)
        y_i = ((centroids.lat - cen_trans[5]) / cen_trans[4]).astype(int)
        fractions = fractions[y_i, x_i]

    return fractions.reshape(-1)
