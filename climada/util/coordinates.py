"""
Define GridPoints class
"""

__all__ = ['GridPoints']

import logging
import numpy as np

from climada.util.interpolation import METHOD, DIST_DEF, interpol_index

LOGGER = logging.getLogger(__name__)

class GridPoints(np.ndarray):
    """Define grid using 2d numpy array. Each row is a point. The first column
    is for latitudes and the second for longitudes (in degrees)."""
    def __new__(cls, input_array=None):
        if input_array is not None:
            obj = np.asarray(input_array).view(cls)
            obj.check()
        else:
            obj = np.empty((0, 2)).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def check(self):
        """Check shape. Repeated points are allowed"""
        if self.shape[1] != 2:
            LOGGER.error("GridPoints with wrong shape: %s != %s",
                         self.shape[1], 2)
            raise ValueError

    def resample(self, coord, method=METHOD[0], distance=DIST_DEF[1]):
        """ Input GridPoints are resampled to current grid by interpolating
        their values.

        Parameters:
            coord (2d array): First column contains latitude, second
                column contains longitude. Each row is a geographic point.
            method (str, optional): interpolation method. Default: nearest
                neighbor.
            distance (str, optional): metric to use. Default: haversine

        Returns:
            np.array
        """
        return interpol_index(self, coord, method, distance)

    def resample_agg_to_lower_res(self, coord):
        """ Input GridPoints are resampled to current grid of lower resolution
        by aggregating the values of the higher resolution grid.

        Parameters:
            coord (2d array): First column contains latitude, second
                column contains longitude. Each row is a geographic point.
        """
        raise NotImplementedError

    def is_regular(self):
        """Return True if grid is regular."""
        regular = False
        _, count_lat = np.unique(self[:, 0], return_counts=True)
        _, count_lon = np.unique(self[:, 1], return_counts=True)
        uni_lat_size = np.unique(count_lat).size
        uni_lon_size = np.unique(count_lon).size
        if uni_lat_size == uni_lon_size and uni_lat_size == 1 \
        and count_lat[0] > 1 and count_lon[0] > 1:
            regular = True
        return regular

    @property
    def lat(self):
        """Get latitude."""
        return self[:, 0]

    @property
    def lon(self):
        """Get longitude."""
        return self[:, 1]
