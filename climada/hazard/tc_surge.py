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

Define TCSurge class.
"""

__all__ = ['TCSurge']

import logging
import numpy as np
from scipy.sparse import csr_matrix
import rasterio
from rasterio.warp import Resampling, reproject

from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import TMP_ELEVATION_FILE, DEM_NODATA, \
MAX_DEM_TILES_DOWN
from climada.util.coordinates import get_resolution, pts_to_raster_meta

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TS'
""" Hazard type acronym for Tropical Cyclone """

DECAY_RATE_M_KM = 0.2
""" Decay rate of surge in meters per each km """

DECAY_INLAND_DIST_KM = 50
""" Maximum inland distance of decay in km """

DECAY_MAX_ELEVATION = 10
""" Maximum elevation where decay is implemented """

DEM_PRODUCT = 'SRTM1'
""" DEM to use: SRTM1 (30m) or SRTM3 (90m) """

class TCSurge(Hazard):
    """Contains tropical cyclone surges per event. """

    intensity_thres = -1000
    """ Maximum surge height after decay implementation """

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)

    def set_from_winds(self, tc_wind, dist_coast_decay=True, dem_product='SRTM3',
                       min_centr_resol=1.0e-8, as_pixel=None, scheduler=None,
                       add_sea_level_rise=0):
        """ Compute tropical cyclone surge from input winds.

        Parameters:
            tc_wind (TropCyclone): tropical cyclone winds
            dist_coast_decay (bool): implement decay according to distance coast
            dem_product (str): DEM to use: 'SRTM1' (30m) or 'SRTM3' (90m)
            min_centr_resol (float, optional): minimum points centroids resolution to
                use when interpolating DEM data. Used in get_resolution method.
                If get_resolution(centroids.lat, centroids.lon) gives a too low
                number, set this parameter to the right centroids resolution.
                Default: 1.0e-8.
            as_pixel (bool, optional): use average (pixel) or nearest neighbor (points)
                to interpolate DEM to centroids. Default: True if centroids.meta
                is present, False otherwise.
            scheduler (str): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
            add_sea_level_rise (float, optional): sea level rise effect in meters
                to be added to surge height. Currently only 1 value per computation
                implemented. TODO: add option to add spatially or temporary
                explicit sea level rise values instead of one value for all events.

        Raises:
            ValueError
        """
        if tc_wind.centroids.size != tc_wind.intensity.shape[1]:
            LOGGER.error('Wrong input variables. Unequal sizes: %s != %s',
                         str(tc_wind.centroids.size), str(tc_wind.intensity.shape[1]))
            raise ValueError

        if as_pixel is None and tc_wind.centroids.meta:
            as_pixel = True

        # set needed attributes to centroids: on_land, elevation
        _set_centroids_att(tc_wind.centroids, dist_coast_decay, dem_product,
                           as_pixel, min_centr_resol, scheduler)

        # conversion wind to surge
        inten_surge = _wind_to_surge(tc_wind.intensity)

        # decay surge
        inten_surge, fract_surge = _surge_decay(inten_surge, tc_wind.centroids,
                                                dem_product, as_pixel, min_centr_resol,
                                                add_sea_level_rise)

        # set other attributes
        self.units = 'm'
        self.centroids = tc_wind.centroids
        self.event_id = tc_wind.event_id
        self.event_name = tc_wind.event_name
        self.date = tc_wind.date
        self.orig = tc_wind.orig
        self.frequency = tc_wind.frequency
        self.intensity = inten_surge
        self.fraction = fract_surge

def _wind_to_surge(inten_wind):
    """ Compute surge heights (m) from wind gusts (m/s). Approximation to
    SLOSH model, see also http://www.nhc.noaa.gov/surge/slosh.php.

    Parameter:
        inten_wind (sparse.csr_matrix): intensity matrix with wind gust values of TC

    Returns:
        sparse.csr_matrix
    """
    inten_surge = inten_wind.copy()
    # m/s converted to m surge height
    inten_surge.data = 0.1023*np.maximum(inten_wind.data-26.8224, 0)+1.8288
    return inten_surge

def _surge_decay(inten_surge, centroids, dem_product, set_fraction, min_resol, \
                 add_sea_level_rise):
    """ Substract DEM height and decay factor from initial surge height and
    computes corresponding fraction matrix.

    Parameter:
        inten_surge (sparse.csr_matrix): initial surge height in m
        centroids (Centroids): centroids, either raster or points
        dem_product (str): DEM to use: 'SRTM1' (30m) or 'SRTM3' (90m)
        set_fraction (bool, optional): set fraction matrix different to ones.
            Default: True
        min_resol (float, optional): minimum points centroids resolution to
            use when interpolating DEM data. Used in get_resolution method.
            If get_resolution(centroids.lat, centroids.lon) gives a too low
            number, set this parameter to the right centroids resolution.
            Default: 1.0e-8.
        add_sea_level_rise (float): sea level rise in meters to be added to surge

    Returns:
        inten_surge (sparse.csr_matrix), fract_surge (sparse.csr_matrix)
    """
    import elevation
    
    inland_decay = _calc_inland_decay(centroids)

    # substract event by event to avoid to densificate all the matrix
    inten_surge = _substract_sparse_surge(inten_surge, centroids.elevation, \
                                          inland_decay, add_sea_level_rise)

    # if set_fraction: fraction is fraction of centroids of DEM on land in
    # given centroids cell. Else fraction is ones where intensity.
    if set_fraction:
        bounds = np.array(centroids.total_bounds)
        if centroids.meta:
            shape = centroids.shape
            ras_trans = centroids.meta['transform']
        else:
            shape = [0, 0]
            shape[0], shape[1], ras_trans = pts_to_raster_meta(bounds, \
                min(get_resolution(centroids.lat, centroids.lon, min_resol)))
        bounds += np.array([-.05, -.05, .05, .05])
        elevation.clip(bounds, output=TMP_ELEVATION_FILE, product=dem_product,
                       max_download_tiles=MAX_DEM_TILES_DOWN)
        fract_surge = np.zeros(shape)
        with rasterio.open(TMP_ELEVATION_FILE, 'r') as src:
            on_land = src.read(1)
            on_land[on_land > 0] = 1 # 1 land
            on_land[on_land <= 0] = 0 # 0 water
            on_land[on_land == src.nodata] = 0 # 0 nodata
            reproject(source=on_land, destination=fract_surge,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ras_trans, dst_crs=centroids.crs,
                      resampling=Resampling.average,
                      src_nodata=src.nodata, dst_nodata=src.nodata)
        if centroids.meta:
            fract_surge = csr_matrix(fract_surge.flatten())
            fract_surge = csr_matrix(np.ones([inten_surge.shape[0], 1])) * fract_surge
        else:
            x_i = ((centroids.lon - ras_trans[2]) / ras_trans[0]).astype(int)
            y_i = ((centroids.lat - ras_trans[5]) / ras_trans[4]).astype(int)
            fract_surge = csr_matrix(fract_surge[y_i, x_i])
    else:
        fract_surge = inten_surge.copy()
        fract_surge.data.fill(1)

    return inten_surge, fract_surge

def _calc_inland_decay(centroids):
    """ Compute decay coefficient in positive elevations

    Parameter:
        centroids (Centroids): centroids

    Returns:
        np.array
    """
    if centroids.dist_coast.size:
        LOGGER.info('Restricting to centroids in elevation range ]0..%s] m ' \
            'and closer than %s km to coast with a decay of %s m/km inland.', \
            str(DECAY_MAX_ELEVATION), str(DECAY_INLAND_DIST_KM), str(DECAY_RATE_M_KM))
        inland_decay = np.maximum(centroids.dist_coast/1000*DECAY_RATE_M_KM, 0)
        elev_pos = np.logical_or(centroids.elevation > DECAY_MAX_ELEVATION, \
            np.logical_and(centroids.elevation > 0, \
            centroids.dist_coast > DECAY_INLAND_DIST_KM*1000))
        inland_decay[elev_pos] = 1000 # strong decay
    else:
        LOGGER.info('Restricting to centroids in elevation range ]0..%s] m.', \
            str(DECAY_MAX_ELEVATION))
        inland_decay = np.zeros(centroids.size)
        inland_decay[centroids.elevation > DECAY_MAX_ELEVATION] = 1000 # strong decay

    # no decay where there's no elevation
    inland_decay[centroids.elevation == DEM_NODATA] = 0
    inland_decay[centroids.elevation <= 0] = 0
    return inland_decay

def _substract_sparse_surge(inten_surge, centr_elevation, inland_decay, add_sea_level_rise):
    """ Substract elevation on land and decay coefficient to surge

    Parameter:
        inten_surge (sparse.csr_matrix): surge matrix
        centr_elevation (np.array): elevation of each centroid
        inland_decay (np.array): decay coefficient for each centroid
        add_sea_level_rise (float): sea level rise in meters to be added to surge

    Returns:
        sparse.csr_matrix
    """
    # only apply decay in positive elevations
    remove_elev = centr_elevation.copy()
    remove_elev[centr_elevation == DEM_NODATA] = 0
    remove_elev[centr_elevation <= 0] = 0

    inten_out = inten_surge.tolil()
    for i_row in range(inten_out.shape[0]):
        row_pos = inten_out.rows[i_row] # only apply decay if there is surge
        inten_out[i_row, row_pos] += -remove_elev[row_pos] - inland_decay[row_pos] +\
                                     add_sea_level_rise


    return inten_out.maximum(0)

def _set_centroids_att(centroids, dist_coast_decay, dem_product, as_pixel=None,
                       min_resol=1.0e-8, scheduler=None):
    """
    Set necessary attributes to centroids.

    Parameter:
        centroids (Centroids): centroids, either raster or points
        dist_coast_decay (bool): implement decay according to distance coast
        dem_product (str): DEM to use: 'SRTM1' (30m) or 'SRTM3' (90m)
        as_pixel (bool, optional): use average (pixel) or nearest neighbor (points)
            to interpolate DEM to centroids
        min_resol (float, optional): minimum centroids resolution to use.
            Default: 1.0e-8
        scheduler (str): used for dask map_partitions. “threads”,
            “synchronous” or “processes”
    """
    # if points, take elevation nearest neighbor. if grid, take average.
    if not centroids.elevation.size:
        if as_pixel:
            resampling = Resampling.average
        else:
            resampling = Resampling.nearest
        centroids.set_elevation(product=dem_product, resampling=resampling,
                                nodata=DEM_NODATA, min_resol=min_resol)
    if dist_coast_decay and not centroids.dist_coast.size:
        centroids.set_dist_coast(scheduler)
