"""
Define Centroids class.
"""

__all__ = ['Centroids',
           'FILE_EXT']

import os
import copy
import logging
import numpy as np

from climada.hazard.centroids.tag import Tag
from climada.hazard.centroids.source import READ_SET
import climada.util.checker as check
from climada.util.coordinates import GridPoints, dist_to_coast, coord_on_land
import climada.util.plot as plot
from climada.util.files_handler import to_list, get_file_names
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS',
            '.csv': 'CSV',
           }

""" Supported files format to read from """

class Centroids(object):
    """Definition of the hazard GridPoints.

    Attributes:
        tag (Tag): information about the source
        coord (np.array or GridPoints): 2d array with lat in first column and
            lon in second, or GridPoints instance. "lat" and "lon" are
            descriptors of the latitude and longitude respectively.
        id (np.array): an id for each centroid
        region_id (np.array, optional): region id for each centroid
            (when defined)
        dist_coast (np.array, optional): distance to coast in km
        admin0_name (str, optional): admin0 country name
        admin0_iso3 (str, optional): admin0 ISO3 country name
        resolution (float tuple, optional): If coord is a regular grid, then
            a tuple of the form (res_lat, res_lon) can be set. Uses the same
            units as the coord attribute.
    """

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters:
            file_name (str, optional): name of the source file
            description (str, optional): description of the source data

        Raises:
            ValueError

        Examples:
            Fill centroids attributes by hand:

            >>> centr = Centroids()
            >>> centr.coord = np.array([[0,-1], [0, -2]])
            >>> ...

            Read centroids from file:

            >>> centr = Centroids(HAZ_DEMO_MAT, 'Centroids demo')
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self.coord = GridPoints()
        self.id = np.array([], int)
        self.region_id = np.array([], int)
        self.dist_coast = np.array([], float)
        self.admin0_name = ''
        self.admin0_iso3 = ''

    def check(self):
        """Check instance attributes. Ids are unique.

        Raises:
            ValueError
        """
        num_exp = len(self.id)
        if np.unique(self.id).size != num_exp:
            LOGGER.error("There are centroids with the same identifier.")
            raise ValueError
        check.shape(num_exp, 2, self.coord, 'Centroids.coord')
        if num_exp > 0 and np.unique(self.coord, axis=0).size \
        != 2*self.coord.shape[0]:
            LOGGER.error("There centroids with the same GridPoints.")
            raise ValueError
        check.array_optional(num_exp, self.region_id,
                             'Centroids.region_id')
        check.array_optional(num_exp, self.dist_coast,
                             'Centroids.dist_coast')

    def read(self, files, descriptions='', var_names=None):
        """ Read and check centroids.

        Parameters:
            files (str or list(str)): absolute file name(s) or folder name
                containing the files to read
            descriptions (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in
                the file (default: check def_source_vars() function)

        Raises:
            TypeError, ValueError
        """
        all_files = get_file_names(files)
        desc_list = to_list(len(all_files), descriptions, 'descriptions')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        for file, desc, var in zip(all_files, desc_list, var_list):
            self.append(Centroids._read_one(file, desc, var))

    def append(self, centroids):
        """Append centroids values with NEW GridPoints. Id is perserved if
        not present in current centroids. Otherwise, a new id is provided.

        Parameters:
            centroids (Centroids): Centroids instance to append
        """
        self.tag.append(centroids.tag)

        if self.id.size == 0:
            centroids.check()
            self.__dict__ = centroids.__dict__.copy()
            return
        elif centroids.id.size == 0:
            return
        elif np.array_equal(centroids.coord, self.coord):
            return

        # GridPoints of centroids that are not in self
        dtype = {'names':['f{}'.format(i) for i in range(2)],
                 'formats':2 * [centroids.coord.dtype]}
        new_pos = np.in1d(centroids.coord.copy().view(dtype),
                          self.coord.copy().view(dtype), invert=True)
        new_pos = np.argwhere(new_pos).squeeze(axis=1)
        if not new_pos.size:
            return

        centroids.check()
        self.coord = np.append(self.coord, centroids.coord[new_pos, :], axis=0)
        self.id = np.append(self.id, centroids.id[new_pos], axis=0)

        if (self.region_id.size == 0) | (centroids.region_id.size == 0):
            self.region_id = np.array([], int)
        else:
            self.region_id = np.append(self.region_id,
                                       centroids.region_id[new_pos])

        if (self.dist_coast.size == 0) | (centroids.dist_coast.size == 0):
            self.dist_coast = np.array([], float)
        else:
            self.dist_coast = np.append(self.dist_coast,
                                        centroids.dist_coast[new_pos])

        # Check id
        _, unique_idx = np.unique(self.id, return_index=True)
        rep_id = [pos for pos in range(self.id.size) if pos not in unique_idx]
        sup_id = np.max(self.id) + 1
        self.id[rep_id] = np.arange(sup_id, sup_id+len(rep_id))

    def calc_dist_to_coast(self):
        """Compute distance to coast for each centroids (dist_coast variable).
        No distinction between sea and land centroids."""
        self.dist_coast = dist_to_coast(self.coord)

    def plot(self, **kwargs):
        """ Plot centroids points over earth.

        Parameters:
            kwargs (optional): arguments for scatter matplotlib function

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if 's' not in kwargs:
            kwargs['s'] = 1
        fig, axis = plot.make_map()
        axis = axis[0][0]
        plot.add_shapes(axis)
        axis.set_title(self.tag.join_file_names())
        axis.scatter(self.lon, self.lat, **kwargs)

        return fig, axis

    def get_nearest_id(self, lat, lon):
        """Get id of nearest centroid coordinate.

        Parameters:
            lat (float): latitude
            lon (float): longitude

        Returns:
            int
        """
        idx = np.linalg.norm(np.abs(self.coord-[lat, lon]), axis=1).argmin()
        return self.id[idx]

    def set_area_per_centroid(self):
        """ If the centroids are on a regular grid and have their resolution
            set, set the area_per_centroid attribute, assuming degrees for the
            GridPoints and km2 for the area.
        """
        try:
            self.resolution
        except ValueError:
            LOGGER.warning('Resolution attribute was not set')

        lat_res_km = self.resolution[0] * ONE_LAT_KM
        lat_unique = np.array(np.unique(self.lat))
        lon_res_km = self.resolution[1] * ONE_LAT_KM * \
            np.cos(lat_unique/180*np.pi)
        lon_unique_n = len(np.unique(self.lon))
        area_per_lat = lat_res_km * lon_res_km
        self.area_per_centroid = np.tile(area_per_lat, lon_unique_n)

    def set_on_land(self):
        """ Add the _on_land attribute, i.e. if a centroid is on land
        """
        self._on_land = coord_on_land(self.lat, self.lon)

    @property 
    def on_land(self):
        """ Retuns a logical array of centroids on land
        """
        if self._on_land is None:
            self.set_on_land()
        return self._on_land

    @property
    def resolution(self):
        """ Returns a tuple of the resolution in the same unit as the coords.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        """ Set the resolution asset after making sure that the coordinates
            are in a regular grid. Coerces floats to a tuple.
        """
        assert self.coord.is_regular, 'The coords are not on a regular grid'
        if type(res) is float:
            res = (res, res)
        assert type(res) is tuple, 'Use a tuple like (lat, lon).'
        self._resolution = res

    @property
    def lat(self):
        """ Get latitude from coord array """
        return self.coord[:, 0]

    @property
    def lon(self):
        """ Get longitude from coord array """
        return self.coord[:, 1]

    @property
    def size(self):
        """ Get count of centroids """
        return self.id.size

    @staticmethod
    def get_sup_file_format():
        """ Get supported file extensions that can be read.

        Returns:
            list(str)
        """
        return list(FILE_EXT.keys())

    @staticmethod
    def get_def_file_var_names(src_format):
        """Get default variable names for given file format.

        Parameters:
            src_format (str): extension of the file, e.g. '.xls', '.mat'.

        Returns:
            dict: dictionary with variable names
        """
        try:
            if '.' not in src_format:
                src_format = '.' + src_format
            return copy.deepcopy(READ_SET[FILE_EXT[src_format]][0])
        except KeyError:
            LOGGER.error('File extension not supported: %s.', src_format)
            raise ValueError

    @property
    def coord(self):
        """ Return coord"""
        return self._coord

    @coord.setter
    def coord(self, value):
        """ If it is not a GridPoints instance, put it."""
        if not isinstance(value, GridPoints):
            self._coord = GridPoints(value)
        else:
            self._coord = value

    @staticmethod
    def _read_one(file_name, description='', var_names=None):
        """Read input file.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data
            var_names (dict, optional): name of the variables in the file

        Raises:
            ValueError

        Returns:
            ImpactFuncSet
        """
        LOGGER.info('Reading file: %s', file_name)
        new_cent = Centroids()
        new_cent.tag = Tag(file_name, description)

        extension = os.path.splitext(file_name)[1]
        try:
            reader = READ_SET[FILE_EXT[extension]][1]
        except KeyError:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        reader(new_cent, file_name, var_names)

        return new_cent

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
