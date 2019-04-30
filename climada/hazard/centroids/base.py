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

Define Centroids class.
"""

__all__ = ['Centroids',
           'FILE_EXT']

import os
import copy
import logging
import numpy as np
import shapely.vectorized

from climada.hazard.centroids.tag import Tag
from climada.hazard.centroids.source import READ_SET
import climada.util.checker as check
from climada.util.coordinates import (
    coord_on_land,
    dist_to_coast,
    get_country_geometries,
    grid_is_regular,
)
import climada.util.plot as u_plot
from climada.util.files_handler import to_list, get_file_names
from climada.util.constants import ONE_LAT_KM

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS',
            '.csv': 'CSV',
           }

""" Supported files format to read from """

class Centroids():
    """Definition of the hazard coordinates.

    Attributes:
        tag (Tag): information about the source
        coord (np.array): 2d array with lat in first column and
            lon in second. "lat" and "lon" are descriptors of the latitude and
            longitude respectively.
        id (np.array): an id for each centroid
        region_id (np.array, optional): region id for each centroid
            (when defined)
        name (str, optional): name of centroids (e.g. country, region)
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
        self.coord = np.array([], float)
        self.id = np.array([], int)
        self.region_id = np.array([], int)
        self.name = ''
        self._resolution = None

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
            LOGGER.error("There centroids with the same points.")
            raise ValueError
        # check all 1-dim variable set
        for var_name, var_val in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                check.array_optional(num_exp, var_val, 'Centroids.'+var_name)

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

    def append(self, centroids, set_uni_id=True):
        """Append centroids values with NEW points. Id is perserved if
        not present in current centroids. Otherwise, a new id is provided.

        Parameters:
            centroids (Centroids): Centroids instance to append
            set_uni_id (bool, optional): set centroids.id to unique values

        Returns:
            np.array(bool)): array of size centroids with True in the new
                elements (which have been put at the end of self)
        """
        self.tag.append(centroids.tag)

        if self.id.size == 0:
            centroids.check()
            self.__dict__ = copy.deepcopy(centroids.__dict__)
            return np.array([])
        if centroids.id.size == 0:
            return np.array([])
        if np.array_equal(centroids.coord, self.coord):
            return np.array([])

        # points of centroids that are not in self
        dtype = {'names':['f{}'.format(i) for i in range(2)],
                 'formats':2 * [centroids.coord.dtype]}
        new_pos = np.in1d(centroids.coord.copy().view(dtype),
                          self.coord.copy().view(dtype), invert=True)
        if not np.argwhere(new_pos).squeeze(axis=1).size:
            return new_pos

        centroids.check()
        self.coord = np.append(self.coord, centroids.coord[new_pos, :], axis=0)
        # append all 1-dim arrays (not the optionals)
        for (var_name, var_val), cen_val in zip(self.__dict__.items(),
                                                centroids.__dict__.values()):
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 and \
            var_val.size and cen_val.size:
                setattr(self, var_name, np.append(var_val, cen_val[new_pos]). \
                        astype(var_val.dtype, copy=False))
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.array([]))

        # Check id
        if set_uni_id:
            _, unique_idx = np.unique(self.id, return_index=True)
            rep_id = [pos for pos in range(self.id.size) if pos not in unique_idx]
            sup_id = np.max(self.id) + 1
            self.id[rep_id] = np.arange(sup_id, sup_id+len(rep_id))

        return new_pos

    def select(self, reg_id=None, sel_cen=None):
        """ Get copy new instance with all the attributs in given region

        Parameters:
            reg_id (int or list): regions to select
            sel_cen (np.array, bool): logical vector of centroids to select

        Returns:
            Centroids
        """
        cen = self.__class__()

        if reg_id is None and sel_cen is None:
            LOGGER.error('Supply either reg_id or sel_cen')
            return
        elif sel_cen is not None:
            pass
        elif reg_id is not None:
            if not isinstance(reg_id, list):
                reg_id = [reg_id]
            sel_cen = np.isin(self.region_id, reg_id)
            if not np.any(sel_cen) and reg_id is not None:
                LOGGER.info('No exposure with region id %s.', reg_id)
                return None

        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 and \
            var_val.size:
                setattr(cen, var_name, var_val[sel_cen])
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 2 and \
            var_val.size:
                setattr(cen, var_name, var_val[sel_cen, :])
            elif isinstance(var_val, list) and var_val:
                setattr(cen, var_name, [var_val[idx] for idx in sel_cen])
            else:
                setattr(cen, var_name, var_val)
        return cen

    def plot(self, **kwargs):
        """ Plot centroids points over earth.

        Parameters:
            kwargs (optional): arguments for scatter matplotlib function

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if 's' not in kwargs:
            kwargs['s'] = 1
        fig, axis = u_plot.make_map()
        axis = axis[0][0]
        u_plot.add_shapes(axis)
        axis.set_title(self.tag.join_file_names())
        axis.scatter(self.lon, self.lat, **kwargs)

        return fig, axis

    def set_dist_coast(self):
        """Add dist_coast attribute: distance to coast in km for each centroids.
        No distinction between sea and land centroids."""
        self.dist_coast = dist_to_coast(self.coord)

    def set_area_per_centroid(self):
        """If the centroids are on a regular grid, we may infer the area per
        pixel from their spacing, i.e. resolution.

        Sets the area_per_centroid attribute, assuming degrees for the points
        and km2 for the area.
        """
        lat_res_km = self.resolution[0] * ONE_LAT_KM
        lat_unique = np.array(np.unique(self.lat))
        lon_res_km = self.resolution[1] * ONE_LAT_KM * \
            np.cos(lat_unique/180*np.pi)
        lon_unique_n = len(np.unique(self.lon))
        area_per_lat = lat_res_km * lon_res_km
        self.area_per_centroid = np.repeat(area_per_lat, lon_unique_n)

    def set_on_land(self):
        """ Add the on_land attribute, i.e. if a centroid is on land.
        """
        self.on_land = coord_on_land(self.lat, self.lon)

    def get_nearest_id(self, lat, lon):
        """Get id of nearest centroid coordinate. Not thought to be called
        recursively!

        Parameters:
            lat (float): latitude
            lon (float): longitude

        Returns:
            int
        """
        idx = np.linalg.norm(np.abs(self.coord-[lat, lon]), axis=1).argmin()
        return self.id[idx]

    def set_region_id(self):
        """ Set the region_id to the adm0 ISO_N3 (country) code as indicated by
        natural earth data. Currently simply iterates over all countries in
        extent given by Centroids instance. Not terribly efficient; could
        implement a raster burn method if centroids lie on regular grid. Could
        also employ parallelization.

        Take heed: the natural earth dataset has errors and misclassifies,
        among others, Norway, Somaliland, and Kosovo, using -99 instead of
        their assigned codes. Have a look at the natural earth shapefiles and
        attribute tables. 
        """
        countries = get_country_geometries(extent=self.extent)
        self.region_id = np.zeros(self.size, dtype=int)
        for geom in zip(countries.geometry, countries.ISO_N3):
            select = shapely.vectorized.contains(geom[0], self.lon, self.lat)
            self.region_id[select] = geom[1]

    def remove_duplicate_coord(self):
        """ Checks whether there are duplicate coordinates and removes the du-
        plicates. The first appearance of the coordinates is left untouched
        while all duplicates later in the array are removed."""
        if np.unique(self.coord, axis=0).size != 2*self.coord.shape[0]:
            LOGGER.info('Removing duplicate centroids:')
            coords, inv, c_dupl = np.unique(self.coord, axis=0, \
                                          return_inverse=True, return_counts=True)
            i_delete = []
            for i_ in np.where(c_dupl > 1)[0]:
                LOGGER.info(str(coords[i_]))
                i_delete.extend(np.where(inv == i_)[0][1:])
            i_delete = np.sort(i_delete)
            original_len = len(getattr(self, 'coord'))
            for attribute in self.__dict__.keys():
                if type(getattr(self, attribute)) is np.ndarray \
                and len(getattr(self, attribute)) == original_len:
                    setattr(self, attribute, \
                            np.delete(getattr(self, attribute), i_delete, axis=0))
        else:
            LOGGER.info('No centroids with duplicate coordinates found.')

    @property
    def resolution(self):
        """ Returns a tuple of the resolution in the same unit as the coords.
        """
        if self._resolution is None:
            assert grid_is_regular(self.coord), 'Centroids not a regular grid'
            lats = np.unique(self.lat)
            lons = np.unique(self.lon)
            res_lat = lats[1] - lats[0]
            res_lon = lons[1] - lons[0]
            self._resolution = (res_lat, res_lon)

        return self._resolution

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

    @property
    def extent(self):
        """ Gets geographical extent as tuple

        Returns:
            extent (tuple, optional): (min_lon, max_lon, min_lat, max_lat)
        """
        return (
            float(np.min(self.lon)),
            float(np.max(self.lon)),
            float(np.min(self.lat)),
            float(np.max(self.lat)),
        )

    @property
    def shape_grid(self):
        """If the centroids lie on a regular grid, return its shape as a tuple
        of the form (n_lat, n_lon), that is, (height, width) """
        assert grid_is_regular(self.coord), 'Coords are not on a regular grid'
        return (
            np.unique(self.lat).size,
            np.unique(self.lon).size
        )

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
