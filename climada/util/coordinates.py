"""
Define coordinates class
"""

__all__ = ['Coordinates',
           'IrregularGrid',
           'RegularGrid'
          ]

import numpy as np

import climada.util.interpolation as interp

class Coordinates(object):
    """Interface for Coordinates definition as regular or irregular grid."""
    @property
    def shape(self):
        """Set emtpy shape for 2D array."""
        return (0, 2)

    def resample(self, coord, method=interp.METHOD[0], \
                    distance=interp.DIST_DEF[1]):
        """ Input coordinates are resampled to current grid.

        Parameters:
            coord (2d array): First column contains latitude, second
                column contains longitude. Each row is a geographic point.
            method (str, optional): interpolation method. Default: nearest
                neighbor.
            distance (str, optional): metric to use. Default: haversine

        Returns:
            np.array
        """
        raise NotImplementedError

class IrregularGrid(np.ndarray, Coordinates):
    """Define irregular grid using 2d numpy array. Each row contains the
    coordinates for one exposure. The first column is for latitudes and the
    second for longitudes (in degrees)."""
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def resample(self, coord, method=interp.METHOD[0], \
                    distance=interp.DIST_DEF[0]):
        """ Input coordinates are resampled to current grid.

        Parameters:
            coord (2d array): First column contains latitude, second
                column contains longitude. Each row is a geographic point.
            method (str, optional): interpolation method. Default: nearest
                neighbor.
            distance (str, optional): metric to use. Default: haversine

        Returns:
            np.array
        """
        return interp.interpol_index(self, coord, method, distance)

class RegularGrid(Coordinates):
    """Define regular grid."""
    # TODO
    def __init__(self, d_lat, d_lon, ini_lat, ini_lon):
        self.d_lat = d_lat
        self.d_lon = d_lon
        self.ini_lat = ini_lat
        self.ini_lon = ini_lon
