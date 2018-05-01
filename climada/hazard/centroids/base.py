"""
Define Centroids class.
"""

__all__ = ['Centroids',
           'FILE_EXT']

import os
import copy
import logging
from array import array
import numpy as np

from climada.hazard.centroids.tag import Tag
from climada.hazard.centroids.source import READ_SET
import climada.util.checker as check
from climada.util.coordinates import Coordinates, IrregularGrid
import climada.util.plot as plot
from climada.util.files_handler import to_list, get_file_names

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS'
           }
""" Supported files format to read from """

class Centroids(object):
    """Definition of the hazard coordinates.

    Attributes:
        tag (Tag): information about the source
        coord (np.array or Coordinates): 2d array with lat in first column and
            lon in second, or Coordinates instance. "lat" and "lon" are
            descriptors of the latitude and longitude respectively.
        id (np.array): an id for each centroid
        region_id (np.array, optional): region id for each centroid
            (when defined)
        dist_coast (np.array, optional): distance to coast in km
        admin0_name (str, optional): admin0 country name
        admin0_iso3 (str, optional): admin0 ISO3 country name
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

            >>> centr = Centroids(HAZ_TEST_XLS, 'Centroids demo')
        """
        self.clear()
        if file_name != '':
            self.read(file_name, description)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = Tag()
        self.coord = Coordinates()
        self.id = np.array([], int)
        self.region_id = np.array([], int)
        self.dist_coast = np.array([], float)
        self.admin0_name = ''
        self.admin0_iso3 = ''

    def check(self):
        """Check instance attributes.

        Raises:
            ValueError
        """
        num_exp = len(self.id)
        if np.unique(self.id).size != num_exp:
            LOGGER.error("There are centroids with the same identifier.")
            raise ValueError
        check.shape(num_exp, 2, self.coord, 'Centroids.coord')
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
        """Append input centroids coordinates to current. Id is perserved if
        not present in current centroids. Otherwise, a new id is provided.
        Returns the array position of each appended centroid.

        Parameters:
            centroids (Centroids): Centroids instance to append

        Returns:
            array
        """
        self.tag.append(centroids.tag)

        if self.id.size == 0:
            centroids.check()
            self.__dict__ = centroids.__dict__.copy()
            return np.arange(centroids.id.size)
        elif centroids.id.size == 0:
            return np.array([])

        # Check if region id need to be considered
        regions = True
        if (self.region_id.size == 0) | (centroids.region_id.size == 0):
            regions = False
            self.region_id = np.array([], int)

        # Check if dist to coast need to be considered
        dist = True
        if (self.dist_coast.size == 0) | (centroids.dist_coast.size == 0):
            dist = False
            self.dist_coast = np.array([], float)

        new_pos, new_id, new_reg, new_dist, new_lat, new_lon = \
            self._append_one(centroids, regions, dist)

        self.coord = np.append(self.coord, np.transpose( \
                               np.array([new_lat, new_lon])), axis=0)
        self.id = np.append(self.id, new_id).astype(int)
        if regions:
            self.region_id = np.append(self.region_id, new_reg)
        if dist:
            self.dist_coast = np.append(self.dist_coast, new_dist)

        return new_pos

    def calc_dist_to_coast(self):
        """ Compute dist_coast value."""
        # TODO: climada//code/helper_functions/climada_distance2coast_km.m
        raise NotImplementedError

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
        min_lat, max_lat = self.lat.min(), self.lat.max()
        min_lon, max_lon = self.lon.min(), self.lon.max()
        axis.set_extent(([int(min_lon), int(max_lon),
                          int(min_lat), int(max_lat)]))
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

    @property
    def lat(self):
        """ Get latitude from coord array """
        return self.coord[:, 0]

    @property
    def lon(self):
        """ Get longitude from coord array """
        return self.coord[:, 1]

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
        """ If it is not a Coordinates instance, put it as IrregularGrid."""
        if not isinstance(value, Coordinates):
            # Set coordinates as irregular grid
            self._coord = IrregularGrid(value)
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

    def _append_one(self, centroids, regions, dist):
        """Append one by one centroid."""
        new_pos = array('l')
        new_id = array('L')
        new_reg = array('l')
        new_lat = array('d')
        new_lon = array('d')
        new_dist = array('d')
        max_id = int(np.max(self.id))
        # Check if new coordinates are all contained in self
        if set(centroids.lat).issubset(set(self.lat)) and \
                set(centroids.lon).issubset(set(self.lon)):
            new_pos = np.arange(self.id.size)
            return new_pos, new_id, new_reg, new_dist, new_lat, new_lon
        centroids.check()
        # TODO speedup select only new centroids
        for cnt, (centr_id, centr) \
        in enumerate(zip(centroids.id, centroids.coord)):
            found = np.where((centr == self.coord).all(axis=1))[0]
            if found.size > 0:
                new_pos.append(found[0])
                if (centr_id in self.id) and \
                (centr_id != self.id[found[0]]):
                    max_id += 1
                    self.id[found[0]] = max_id
                else:
                    self.id[found[0]] = centr_id
                    max_id = max(max_id, centr_id)
                if regions:
                    self.region_id[found[0]] = centroids.region_id[cnt]
                if dist:
                    self.dist_coast[found[0]] = centroids.dist_coast[cnt]
            else:
                new_pos.append(self.coord.shape[0] + len(new_lat))
                new_lat.append(centr[0])
                new_lon.append(centr[1])
                if centr_id in self.id:
                    max_id += 1
                    new_id.append(max_id)
                else:
                    new_id.append(centr_id)
                    max_id = max(max_id, centr_id)
                if regions:
                    new_reg.append(centroids.region_id[cnt])
                if dist:
                    new_dist.append(centroids.dist_coast[cnt])
        return new_pos, new_id, new_reg, new_dist, new_lat, new_lon

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
