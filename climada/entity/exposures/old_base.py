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

Define Exposures class.
"""

__all__ = ['Exposures',
           'FILE_EXT']

import os
import copy
import logging
import numpy as np

from climada.entity.exposures.source import READ_SET, DEF_REF_YEAR
from climada.util.files_handler import to_list, get_file_names
import climada.util.checker as check
from climada.entity.tag import Tag
from climada.util.coordinates import GridPoints
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS'
           }
""" Supported files format to read from """

class Exposures():
    """Defines exposures attributes and basic methods. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (Tag): information about the source data
        ref_year (int): reference year
        value_unit (str): unit of the exposures values
        id (np.array): an id for each exposure
        coord (np.array or GridPoints): 2d array with lat in first column and
            lon in second, or GridPoints instance. "lat" and "lon" are
            descriptors of the latitude and longitude respectively.
        value (np.array): a value for each exposure
        impact_id (dict): for each hazard type (key in dict), np.array of
            impact function id corresponding to each exposure
        deductible (np.array, optional): deductible value for each exposure
        cover (np.array, optional): cover value for each exposure
        category_id (np.array, optional): category id for each exposure
            (when defined)
        region_id (np.array, optional): region id for each exposure
            (when defined)
        assigned (dict, optional): for a given hazard, position of the
            centroid(s) affecting each exposure. Filled in 'assign_centroids'
            method.
    """

    vars_oblig = {'tag',
                  'ref_year',
                  'value_unit',
                  '_coord',
                  'value',
                  'impact_id',
                  'id'
                 }
    """Name of the variables needed to compute the impact. Types: scalar, str,
    list, 1dim np.array of size num_exposures, GridPoints and Tag."""

    vars_def = {'assigned'
               }
    """Name of the variables used in impact calculation whose value is
    descriptive or can be recomputed. Types: dict.
    """

    vars_opt = {'deductible',
                'cover',
                'category_id',
                'region_id'
               }
    """Name of the variables that aren't need to compute the impact. Types:
    scalar, string, list, 1dim np.array of size num_exposures."""

    def __init__(self, file_name='', description=''):
        """Fill values from file, if provided.

        Parameters:
            file_name (str or list(str), optional): absolute file name(s) or
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises:
            ValueError

        Examples:
            Fill exposures with values and check consistency data:

            >>> exp_city = Exposures()
            >>> exp_city.coord = np.array([[40.1, 8], [40.2, 8], [40.3, 8]])
            >>> exp_city.value = np.array([5604, 123, 9005001])
            >>> exp_city.impact_id = {'FL': np.array([1, 1, 1])}
            >>> exp_city.id = np.array([11, 12, 13])
            >>> exp_city.check()

            Read exposures from ENT_TEMPLATE_XLS and checks consistency data.

            >>> exp_city = Exposures(ENT_TEMPLATE_XLS)
        """
        self.tag = Tag()
        self.ref_year = DEF_REF_YEAR
        self.value_unit = ''
        # Following values defined for each exposure
        # Obligatory variables
        self.coord = GridPoints()
        self.value = np.array([], float)
        self.impact_id = dict() # {np.array([], int or str)}
        self.id = np.array([], int)
        # Optional variables.
        self.deductible = np.array([], float)
        self.cover = np.array([], float)
        self.category_id = np.array([], int)
        self.region_id = np.array([], int)
        self.assigned = dict()

        if file_name != '':
            self.read(file_name, description)

    def clear(self):
        """Reinitialize attributes."""
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.array([]))
            else:
                setattr(self, var_name, var_val.__class__())
        self.ref_year = DEF_REF_YEAR

    def assign(self, hazard):
        """Compute the hazard centroids affecting to each exposure. Returns the
        position of the centroids, not their ids.

        Parameters:
            hazard (subclass Hazard): one hazard

        Raises:
            ValueError
        """
        self.assigned[hazard.tag.haz_type] = \
            hazard.centroids.coord.resample_nn(self.coord)

    def check(self):
        """Check instance attributes.

        Raises:
            ValueError
        """
        num_exp = len(self.id)
        if np.unique(self.id).size != num_exp:
            LOGGER.error("There are exposures with the same identifier.")
            raise ValueError

        check.check_oligatories(self.__dict__, self.vars_oblig, 'Exposures.',
                                num_exp, num_exp, 2)
        if not self.impact_id:
            LOGGER.error('No impact function id set.')
        for if_haz, if_id in self.impact_id.items():
            if not if_haz:
                LOGGER.warning('Exposures.impact_id: impact_id hazard type ' \
                               'not set.')
            check.size(num_exp, if_id, 'Exposures.impact_id')

        check.check_optionals(self.__dict__, self.vars_opt, 'Exposures.', num_exp)
        check.empty_optional(self.assigned, "Exposures.assigned")
        for ass_haz, ass in self.assigned.items():
            if not ass_haz:
                LOGGER.warning('Exposures.assigned: assigned hazard type ' \
                               'not set.')
            check.array_optional(num_exp, ass, 'Exposures.assigned')

    def plot(self, mask=None, ignore_zero=False, pop_name=True, buffer_deg=0.0,
             extend='neither', **kwargs):
        """Plot exposures values sum binned over Earth's map. An other function
        for the bins can be set through the key reduce_C_function.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer_deg (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            kwargs (optional): arguments for hexbin matplotlib function, e.g.
                reduce_C_function=np.average. Default: reduce_C_function=np.sum

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        title = self.tag.description
        cbar_label = 'Value (%s)' % self.value_unit
        if 'reduce_C_function' not in kwargs:
            kwargs['reduce_C_function'] = np.sum
        if mask is None:
            mask = np.ones((self.value.size,), dtype=bool)
        if ignore_zero:
            pos_vals = self.value[mask] > 0
        else:
            pos_vals = np.ones((self.value[mask].size,), dtype=bool)
        return u_plot.geo_bin_from_array(self.value[mask][pos_vals], \
            self.coord[mask][pos_vals], cbar_label, title, pop_name, \
            buffer_deg, extend, **kwargs)

    def read(self, files, descriptions='', var_names=None):
        """Read and check exposures.

        Parameters:
            files (str or list(str)): absolute file name(s) or folder name
                containing the files to read
            descriptions (str or list(str), optional): one description of the
                data or a description of each data file
            var_names (dict or list(dict), default): name of the variables in
                the file (default: check def_source_vars() function)

        Raises:
            ValueError
        """
        # Construct absolute path file names
        all_files = get_file_names(files)
        if not all_files:
            LOGGER.warning('No valid file provided: %s', files)
        desc_list = to_list(len(all_files), descriptions, 'descriptions')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        for file, desc, var in zip(all_files, desc_list, var_list):
            self.append(Exposures._read_one(file, desc, var))

    def remove(self, exp_id):
        """Remove one exposure with given id.

        Parameters:
            exp_id (list(int) or int): exposure ids.
        """
        if not isinstance(exp_id, list):
            exp_id = [exp_id]
        try:
            pos_del = []
            for one_id in exp_id:
                pos_del.append(np.argwhere(self.id == one_id)[0][0])
        except IndexError:
            LOGGER.info('No exposure with id %s.', exp_id)
            return

        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 \
            and var_val.size:
                setattr(self, var_name, np.delete(var_val, pos_del))
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 2:
                setattr(self, var_name, np.delete(var_val, pos_del, axis=0))

        old_assigned = self.assigned.copy()
        for key, val in old_assigned.items():
            self.assigned[key] = np.delete(val, pos_del)

        old_impact_id = self.impact_id.copy()
        for key, val in old_impact_id.items():
            self.impact_id[key] = np.delete(val, pos_del)

    def append(self, exposures):
        """Check and append variables of input Exposures to current Exposures.

        Parameters:
            exposures (Exposures): Exposures instance to append to current

        Raises:
            ValueError
        """
        exposures.check()
        if self.id.size == 0:
            self.__dict__ = copy.deepcopy(exposures.__dict__)
            return

        if self.ref_year != exposures.ref_year:
            LOGGER.error("Append not possible. Different reference years.")
            raise ValueError
        if not self.value_unit and exposures.value_unit:
            self.value_unit = exposures.value_unit
            LOGGER.info("Exposures units set to %s.", self.value_unit)
        elif not exposures.value_unit:
            LOGGER.info("Exposures units set to %s.", self.value_unit)
        elif self.value_unit != exposures.value_unit:
            LOGGER.error("Append not possible. Different units: %s != %s.", \
                             self.value_unit, exposures.value_unit)
            raise ValueError
        self.tag.append(exposures.tag)

        # append all 1-dim variables and 2-dim coordinate variable
        for (var_name, var_val), haz_val in zip(self.__dict__.items(),
                                                exposures.__dict__.values()):
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 \
            and var_val.size:
                setattr(self, var_name, np.append(var_val, haz_val). \
                        astype(var_val.dtype, copy=False))
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 2:
                setattr(self, var_name, np.append(var_val, haz_val, axis=0). \
                        astype(var_val.dtype, copy=False))
            elif isinstance(var_val, list) and var_val:
                setattr(self, var_name, var_val + haz_val)

        self.coord = self._coord
        for key, val in exposures.assigned.items():
            if key in self.assigned:
                self.assigned[key] = np.append(
                    self.assigned[key], val).astype(val.dtype, copy=False)
        for key, val in exposures.impact_id.items():
            if key in self.impact_id:
                self.impact_id[key] = np.append(
                    self.impact_id[key], val).astype(val.dtype, copy=False)

        # provide new ids to repeated ones
        _, indices = np.unique(self.id, return_index=True)
        new_id = np.max(self.id) + 1
        for dup_id in np.delete(np.arange(self.id.size), indices):
            self.id[dup_id] = new_id
            new_id += 1

    def select(self, reg_id):
        """Return reference exposure of given region.

        Parameters:
            reg_id (int): region id to select

        Returns:
            Exposures
        """
        if not isinstance(reg_id, list):
            reg_id = [reg_id]
        if isinstance(reg_id, list):
            sel_idx = np.zeros(self.size, bool)
            for reg in reg_id:
                sel_idx = np.logical_or(sel_idx, self.region_id == reg)
        if not np.any(sel_idx):
            LOGGER.info('No exposure with region id %s.', reg_id)
            return None

        sel_exp = self.__class__()
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 and \
            var_val.size:
                setattr(sel_exp, var_name, var_val[sel_idx])
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 2:
                setattr(sel_exp, var_name, var_val[sel_idx, :])
            elif isinstance(var_val, list) and var_val:
                setattr(sel_exp, var_name, [var_val[idx] for idx in sel_idx])
            else:
                setattr(sel_exp, var_name, var_val)

        sel_exp.assigned = dict()
        for key, value in self.assigned.items():
            sel_exp.assigned[key] = value[sel_idx]

        sel_exp.impact_id = dict()
        for key, value in self.impact_id.items():
            sel_exp.impact_id[key] = value[sel_idx]

        sel_exp.tag = copy.copy(self.tag)
        sel_exp.tag.description = 'Region: ' + str(reg_id)

        return sel_exp

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
            src_format (str): extension of the file, e.g. '.xls', '.mat'

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
    def lat(self):
        """ Get latitude from coord array """
        return self._coord[:, 0]

    @property
    def lon(self):
        """ Get longitude from coord array """
        return self._coord[:, 1]

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

    @property
    def size(self):
        """ Get longitude from coord array """
        return self.value.size

    @classmethod
    def _read_one(cls, file_name, description='', var_names=None):
        """Read one file and fill attributes.

        Parameters:
            file_name (str): name of the source file
            description (str, optional): description of the source data
            var_names (dict, optional): name of the variables in the file

        Raises:
            ValueError

        Returns:
            Exposures
        """
        LOGGER.info('Reading file: %s', file_name)
        new_exp = cls()
        new_exp.tag = Tag(file_name, description)

        extension = os.path.splitext(file_name)[1]
        try:
            reader = READ_SET[FILE_EXT[extension]][1]
        except KeyError:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        reader(new_exp, file_name, var_names)

        return new_exp
