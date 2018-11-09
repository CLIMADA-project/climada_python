"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Hazard.
"""

__all__ = ['Hazard',
           'FILE_EXT']

import os
import copy
import logging
import datetime as dt
import warnings
import numpy as np
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.source import READ_SET
from climada.util.files_handler import to_list, get_file_names
import climada.util.plot as u_plot
import climada.util.checker as check
import climada.util.dates_times as u_dt
from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS'
           }
""" Supported files format to read from """

class Hazard():
    """Contains events of some hazard type defined at centroids. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (TagHazard): information about the source
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id (>0) of each event
        event_name (list(str)): name of each event (default: event_id)
        date (np.array): integer date corresponding to the proleptic
            Gregorian ordinal, where January 1 of year 1 has ordinal 1
            (ordinal format of datetime library)
        orig (np.array): flags indicating historical events (True)
            or probabilistic (False)
        frequency (np.array): frequency of each event in years
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """
    intensity_thres = 10
    """ Intensity threshold per hazard used to filter lower intensities. To be
    set for every hazard type """

    vars_oblig = {'tag',
                  'units',
                  'centroids',
                  'event_id',
                  'frequency',
                  'intensity',
                  'fraction'
                 }
    """Name of the variables needed to compute the impact. Types: scalar, str,
    list, 1dim np.array of size num_events, scipy.sparse matrix of shape
    num_events x num_centroids, Centroids and Tag."""

    vars_def = {'date',
                'orig',
                'event_name'
               }
    """Name of the variables used in impact calculation whose value is
    descriptive and can therefore be set with default values. Types: scalar,
    string, list, 1dim np.array of size num_events.
    """

    vars_opt = set()
    """Name of the variables that aren't need to compute the impact. Types:
    scalar, string, list, 1dim np.array of size num_events."""

    def __init__(self, haz_type='', file_name='', description='', centroids=None):
        """Initialize values from given file, if given.

        Parameters:
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC').
            file_name (str or list(str), optional): absolute file name(s) or \
                folder name containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids

        Raises:
            ValueError

        Examples:
            Fill hazard values by hand:

            >>> haz = Hazard()
            >>> haz.intensity = sparse.csr_matrix(np.zeros((2, 2)))
            >>> ...

            Take hazard values from file:

            >>> haz = Hazard('TC', HAZ_DEMO_MAT)

            Take centriods from a different source:

            >>> centr = Centroids(HAZ_DEMO_MAT, 'Centroids demo')
            >>> haz = Hazard('TC', HAZ_DEMO_MAT, 'Demo hazard.', centr)
        """
        self.tag = TagHazard()
        self.units = ''
        self.centroids = Centroids()
        # following values are defined for each event
        self.event_id = np.array([], int)
        self.frequency = np.array([], float)
        self.event_name = list()
        self.date = np.array([], int)
        self.orig = np.array([], bool)
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix(np.empty((0, 0))) # events x centroids
        self.fraction = sparse.csr_matrix(np.empty((0, 0)))  # events x centroids

        if '.' in haz_type and file_name == '':
            LOGGER.error("Provide hazard type.")
            raise ValueError
        self.tag.haz_type = haz_type
        if file_name != '':
            if haz_type == '':
                LOGGER.warning("Hazard type acronym not provided.")
            self.read(file_name, description, centroids)

    def clear(self):
        """Reinitialize attributes."""
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.array([], dtype=var_val.dtype))
            elif isinstance(var_val, sparse.csr_matrix):
                setattr(self, var_name, sparse.csr_matrix(np.empty((0, 0))))
            else:
                setattr(self, var_name, var_val.__class__())

    def check(self):
        """Check if the attributes contain consistent data.

        Raises:
            ValueError
        """
        self.centroids.check()
        self._check_events()

    def read(self, files, description='', centroids=None, var_names=None):
        """Set and check hazard, and centroids if not provided, from file.

        Parameters:
            files (str or list(str)): absolute file name(s) or folder name
                containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids
            var_names (dict or list(dict), default): name of the variables in
                the file (default: DEF_VAR_NAME defined in the source modules)

        Raises:
            ValueError
        """
        haz_type = self.tag.haz_type
        # Construct absolute path file names
        all_files = get_file_names(files)
        if not all_files:
            LOGGER.warning('No valid file provided: %s', files)
        desc_list = to_list(len(all_files), description, 'description')
        centr_list = to_list(len(all_files), centroids, 'centroids')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        for file, desc, centr, var in zip(all_files, desc_list, centr_list,
                                          var_list):
            self.append(self._read_one(file, haz_type, desc, centr, var))

    def select(self, date=None, orig=None):
        """Select events within provided date and/or historical or synthetical.
        Frequency of the events may need to be recomputed!

        Parameters:
            date (tuple(str or int), optional): (initial date, final date) in
                string ISO format or datetime ordinal integer
            orig (bool, optional): select only historical (True) or only
                synthetic (False)

        Returns:
            Hazard or children
        """
        haz = self.__class__()
        sel_idx = np.ones(self.event_id.size, bool)

        if isinstance(date, tuple):
            date_ini, date_end = date[0], date[1]
            if isinstance(date_ini, str):
                date_ini = u_dt.str_to_date(date[0])
                date_end = u_dt.str_to_date(date[1])

            sel_idx = np.logical_and(date_ini <= self.date,
                                     self.date <= date_end)
            if not np.any(sel_idx):
                LOGGER.info('No hazard in date range %s.', date)
                return None

        if isinstance(orig, bool):
            sel_idx = np.logical_and(sel_idx, self.orig.astype(bool) == orig)
            if not np.any(sel_idx):
                LOGGER.info('No hazard with %s tracks.', str(orig))
                return None

        sel_idx = np.argwhere(sel_idx).squeeze()
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 and \
            var_val.size:
                setattr(haz, var_name, var_val[sel_idx])
            elif isinstance(var_val, sparse.csr_matrix):
                setattr(haz, var_name, var_val[sel_idx, :])
            elif isinstance(var_val, list) and var_val:
                setattr(haz, var_name, [var_val[idx] for idx in sel_idx])
            else:
                setattr(haz, var_name, var_val)

        return haz

    def local_exceedance_inten(self, return_periods=(25, 50, 100, 250)):
        """ Compute exceedance intensity map for given return periods.

        Parameters:
            return_periods (np.array): return periods to consider

        Returns:
            np.array
        """
        LOGGER.info('Computing exceedance intenstiy map for return periods: %s',
                    return_periods)
        num_cen = self.intensity.shape[1]
        inten_stats = np.zeros((len(return_periods), num_cen))
        cen_step = int(CONFIG['global']['max_matrix_size']/self.intensity.shape[0])
        if not cen_step:
            LOGGER.error('Increase max_matrix_size configuration parameter to'\
                         ' > %s', str(self.intensity.shape[0]))
            raise ValueError
        # separte in chunks
        chk = -1
        for chk in range(int(num_cen/cen_step)):
            self._loc_return_inten(np.array(return_periods), \
                self.intensity[:, chk*cen_step:(chk+1)*cen_step].todense(), \
                inten_stats[:, chk*cen_step:(chk+1)*cen_step])
        self._loc_return_inten(np.array(return_periods), \
            self.intensity[:, (chk+1)*cen_step:].todense(), \
            inten_stats[:, (chk+1)*cen_step:])

        return inten_stats

    def plot_rp_intensity(self, return_periods=(25, 50, 100, 250), **kwargs):
        """Compute and plot hazard exceedance intensity maps for different
        return periods. Calls local_exceedance_inten.

        Parameters:
            return_periods (tuple(int), optional): return periods to consider
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot,
            np.ndarray (return_periods.size x num_centroids)
        """
        inten_stats = self.local_exceedance_inten(np.array(return_periods))
        colbar_name = 'Wind intensity (' + self.units + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period: ' + str(ret) + ' years')
        fig, axis = u_plot.geo_im_from_array(inten_stats, self.centroids.coord,
                                             colbar_name, title, **kwargs)
        return fig, axis, inten_stats

    def plot_intensity(self, event=None, centr=None, **kwargs):
        """Plot intensity values for a selected event or centroid.

        Parameters:
            event (int or str, optional): If event > 0, plot intensities of
                event with id = event. If event = 0, plot maximum intensity in
                each centroid. If event < 0, plot abs(event)-largest event. If
                event is string, plot events with that name.
            centr (int or tuple, optional): If centr > 0, plot intensity
                of all events at centroid with id = centr. If centr = 0,
                plot maximum intensity of each event. If centr < 0,
                plot abs(centr)-largest centroid where higher intensities
                are reached. If tuple with (lat, lon) plot intensity of nearest
                centroid.
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        Raises:
            ValueError
        """
        col_label = 'Intensity %s' % self.units
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(event, self.intensity, col_label, **kwargs)
        if centr is not None:
            if isinstance(centr, tuple):
                centr = self.centroids.get_nearest_id(centr[0], centr[1])
            return self._centr_plot(centr, self.intensity, col_label)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def plot_fraction(self, event=None, centr=None, **kwargs):
        """Plot fraction values for a selected event or centroid.

        Parameters:
            event (int or str, optional): If event > 0, plot fraction of event
                with id = event. If event = 0, plot maximum fraction in each
                centroid. If event < 0, plot abs(event)-largest event. If event
                is string, plot events with that name.
            centr (int or tuple, optional): If centr > 0, plot fraction
                of all events at centroid with id = centr. If centr = 0,
                plot maximum fraction of each event. If centr < 0,
                plot abs(centr)-largest centroid where highest fractions
                are reached. If tuple with (lat, lon) plot fraction of nearest
                centroid.
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        Raises:
            ValueError
        """
        col_label = 'Fraction'
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(event, self.fraction, col_label, **kwargs)
        if centr is not None:
            if isinstance(centr, tuple):
                centr = self.centroids.get_nearest_id(centr[0], centr[1])
            return self._centr_plot(centr, self.fraction, col_label)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def get_event_id(self, event_name):
        """"Get an event id from its name. Several events might have the same
        name.

        Parameters:
            event_name (str): Event name

        Returns:
            np.array(int)
        """
        list_id = self.event_id[[i_name for i_name, val_name \
            in enumerate(self.event_name) if val_name == event_name]]
        if list_id.size == 0:
            LOGGER.error("No event with name: %s", event_name)
            raise ValueError
        return list_id

    def get_event_name(self, event_id):
        """"Get the name of an event id.

        Parameters:
            event_id (int): id of the event

        Returns:
            str

        Raises:
            ValueError
        """
        try:
            return self.event_name[np.argwhere(
                self.event_id == event_id)[0][0]]
        except IndexError:
            LOGGER.error("No event with id: %s", event_id)
            raise ValueError

    def get_event_date(self, event=None):
        """ Return list of date strings for given event or for all events,
        if no event provided.

        Parameters:
            event (str or int, optional): event name or id.

        Returns:
            list(str)
        """
        if event is None:
            l_dates = [u_dt.date_to_str(date) for date in self.date]
        elif isinstance(event, str):
            ev_ids = self.get_event_id(event)
            l_dates = [u_dt.date_to_str(self.date[ \
                       np.argwhere(self.event_id == ev_id)[0][0]]) \
                       for ev_id in ev_ids]
        else:
            ev_idx = np.argwhere(self.event_id == event)[0][0]
            l_dates = [u_dt.date_to_str(self.date[ev_idx])]
        return l_dates

    def calc_year_set(self):
        """ From the dates of the original events, get number yearly events.

        Returns:
            dict: key are years, values array with event_ids of that year

        """
        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date[self.orig]])
        orig_yearset = {}
        for year in np.unique(orig_year):
            orig_yearset[year] = self.event_id[self.orig][orig_year == year]
        return orig_yearset

    def append(self, hazard, set_uni_id=True):
        """Append events in hazard. Id is perserved if not present in current
        hazard. Otherwise, a new id is provided. Calls Centroids.append.
        All arrays and lists of the instances are appended.

        Parameters:
            hazard (Hazard): Hazard instance to append to current
            set_uni_id (bool, optional): set event_id and centroids.id to
                unique values

        Raises:
            ValueError
        """
        hazard._check_events()
        if self.event_id.size == 0:
            for key in hazard.__dict__:
                try:
                    self.__dict__[key] = copy.deepcopy(hazard.__dict__[key])
                except TypeError:
                    self.__dict__[key] = copy.copy(hazard.__dict__[key])
            return

        if (self.units == '') and (hazard.units != ''):
            LOGGER.info("Initial hazard does not have units.")
            self.units = hazard.units
        elif hazard.units == '':
            LOGGER.info("Appended hazard does not have units.")
        elif self.units != hazard.units:
            LOGGER.error("Hazards with different units can't be appended: "
                         "%s != %s.", self.units, hazard.units)
            raise ValueError

        self.tag.append(hazard.tag)
        n_ini_ev = self.event_id.size
        # append all 1-dim variables
        for (var_name, var_val), haz_val in zip(self.__dict__.items(),
                                                hazard.__dict__.values()):
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 and \
            var_val.size:
                setattr(self, var_name, np.append(var_val, haz_val). \
                        astype(var_val.dtype, copy=False))
            elif isinstance(var_val, list) and var_val:
                setattr(self, var_name, var_val + haz_val)

        # append intensity and fraction:
        # if same centroids, just append events
        # else, check centroids correct column
        if np.array_equal(self.centroids.coord, hazard.centroids.coord):
            self.intensity = sparse.vstack([self.intensity, \
                hazard.intensity], format='csr')
            self.fraction = sparse.vstack([self.fraction, \
                hazard.fraction], format='csr')
        else:
            cen_self, cen_haz = self._append_haz_cent(hazard.centroids, set_uni_id)
            self.intensity = sparse.vstack([self.intensity, \
                sparse.lil_matrix((hazard.intensity.shape[0], \
                self.intensity.shape[1]))], format='lil')
            self.fraction = sparse.vstack([self.fraction, \
                sparse.lil_matrix((hazard.intensity.shape[0], \
                self.intensity.shape[1]))], format='lil')

            self.intensity[n_ini_ev:, cen_self] = hazard.intensity[:, cen_haz]
            self.fraction[n_ini_ev:, cen_self] = hazard.fraction[:, cen_haz]
            self.intensity = self.intensity.tocsr()
            self.fraction = self.fraction.tocsr()

        # Make event id unique
        if set_uni_id:
            _, unique_idx = np.unique(self.event_id, return_index=True)
            rep_id = [pos for pos in range(self.event_id.size)
                      if pos not in unique_idx]
            sup_id = np.max(self.event_id) + 1
            self.event_id[rep_id] = np.arange(sup_id, sup_id+len(rep_id))

    def remove_duplicates(self):
        """Remove duplicate events (events with same name and date)."""
        dup_pos = list()
        set_ev = set()
        for ev_pos, (ev_name, ev_date) in enumerate(zip(self.event_name,
                                                        self.date)):
            if (ev_name, ev_date) in set_ev:
                dup_pos.append(ev_pos)
            set_ev.add((ev_name, ev_date))
        if len(set_ev) == self.event_id.size:
            return

        for var_name, var_val in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.delete(var_val, dup_pos))
            elif isinstance(var_val, list):
                setattr(self, var_name, np.delete(var_val, dup_pos).tolist())

        mask = np.ones(self.intensity.shape, dtype=bool)
        mask[dup_pos, :] = False
        self.intensity = sparse.csr_matrix(self.intensity[mask].\
        reshape(self.event_id.size, self.intensity.shape[1]))
        self.fraction = sparse.csr_matrix(self.fraction[mask].\
        reshape(self.event_id.size, self.intensity.shape[1]))

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
    def size(self):
        """ Returns number of events """
        return self.event_id.size

    def _append_all(self, list_haz_ev):
        """Append event by event with same centroids. Takes centroids and units
        of first event.

        Parameters:
            list_haz_ev (list): Hazard instances with one event and same
                centroids
        """
        self.clear()
        num_ev = len(list_haz_ev)
        num_cen = list_haz_ev[0].centroids.size
        for var_name, var_val in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.zeros((num_ev,), dtype=var_val.dtype))
            elif isinstance(var_val, sparse.csr.csr_matrix):
                setattr(self, var_name, sparse.lil_matrix((num_ev, num_cen)))

        for i_ev, haz_ev in enumerate(list_haz_ev):
            for (var_name, var_val), ev_val in zip(self.__dict__.items(),
                                                   haz_ev.__dict__.values()):
                if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                    var_val[i_ev] = ev_val[0]
                elif isinstance(var_val, list):
                    var_val.extend(ev_val)
                elif isinstance(var_val, sparse.lil_matrix):
                    var_val[i_ev, :] = ev_val[0, :]
                elif isinstance(var_val, TagHazard):
                    var_val.append(ev_val)

        self.centroids = copy.deepcopy(list_haz_ev[0].centroids)
        self.units = list_haz_ev[0].units
        self.intensity = self.intensity.tocsr()
        self.fraction = self.fraction.tocsr()
        self.event_id = np.arange(1, num_ev+1)

    def _append_haz_cent(self, centroids, set_uni_id=True):
        """Append centroids. Get positions of new centroids.

        Parameters:
            centroids (Centroids): centroids to append
            set_uni_id (bool, optional): set centroids.id to unique values
        Returns:
            cen_self (np.array): positions in self of new centroids
            cen_haz (np.array): corresponding positions in centroids
        """
        # append different centroids
        n_ini_cen = self.centroids.id.size
        new_pos = self.centroids.append(centroids, set_uni_id)

        self.intensity = sparse.hstack([self.intensity, \
            sparse.lil_matrix((self.intensity.shape[0], \
            self.centroids.id.size - n_ini_cen))], format='lil')
        self.fraction = sparse.hstack([self.fraction, \
            sparse.lil_matrix((self.fraction.shape[0], \
            self.centroids.id.size - n_ini_cen))], format='lil')

        # compute positions of repeated and different centroids
        new_vals = np.argwhere(new_pos).squeeze(axis=1)
        rep_vals = np.argwhere(np.logical_not(new_pos)).squeeze(axis=1)

        if rep_vals.size:
            view_x = self.centroids.coord[:self.centroids.size-new_vals.size].\
                    astype(float).view(complex).reshape(-1,)
            view_y = centroids.coord[rep_vals].\
                    astype(float).view(complex).reshape(-1,)
            index = np.argsort(view_x)
            sorted_index = np.searchsorted(view_x[index], view_y)
            yindex = np.take(index, sorted_index, mode="clip")
        else:
            yindex = np.array([])

        cen_self = np.zeros(centroids.size, dtype=int)
        cen_haz = np.zeros(centroids.size, dtype=int)
        cen_self[:rep_vals.size] = yindex
        cen_haz[:rep_vals.size] = rep_vals
        cen_self[rep_vals.size:] = np.arange(new_vals.size) + n_ini_cen
        cen_haz[rep_vals.size:] = new_vals

        return cen_self, cen_haz

    @classmethod
    def _read_one(cls, file_name, haz_type, description='', centroids=None, \
                  var_names=None):
        """ Read hazard, and centroids if not provided, from input file.

        Parameters:
            file_name (str): name of the source file
            haz_type (str): acronym of the hazard type (e.g. 'TC')
            description (str, optional): description of the source data
            centroids (Centroids, optional): Centroids instance
            var_names (dict, optional): name of the variables in the file

        Raises:
            ValueError, KeyError

        Returns:
            Hazard or children
        """
        LOGGER.info('Reading file: %s', file_name)
        new_haz = cls()
        new_haz.tag = TagHazard(haz_type, file_name, description)

        extension = os.path.splitext(file_name)[1]
        try:
            reader = READ_SET[FILE_EXT[extension]][1]
        except KeyError:
            LOGGER.error('Input file extension not supported: %s.', extension)
            raise ValueError
        reader(new_haz, file_name, centroids, var_names)

        return new_haz

    def _events_set(self):
        """Generate set of tuples with (event_name, event_date) """
        ev_set = set()
        for ev_name, ev_date in zip(self.event_name, self.date):
            ev_set.add((ev_name, ev_date))
        return ev_set

    def _event_plot(self, event_id, mat_var, col_name, **kwargs):
        """"Plot an event of the input matrix.

        Parameters:
            event_id (int or np.array(int)): If event_id > 0, plot mat_var of
                event with id = event_id. If event_id = 0, plot maximum
                mat_var in each centroid. If event_id < 0, plot
                abs(event_id)-largest event.
            mat_var (sparse matrix): Sparse matrix where each row is an event
            col_name (sparse matrix): Colorbar label
            kwargs (optional): arguments for pcolormesh matplotlib function

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if not isinstance(event_id, np.ndarray):
            event_id = np.array([event_id])
        array_val = list()
        l_title = list()
        for ev_id in event_id:
            if ev_id > 0:
                try:
                    event_pos = np.where(self.event_id == ev_id)[0][0]
                except IndexError:
                    LOGGER.error('Wrong event id: %s.', ev_id)
                    raise ValueError from IndexError
                im_val = mat_var[event_pos, :].todense().transpose()
                title = 'Event ID %s: %s' % (str(self.event_id[event_pos]), \
                                          self.event_name[event_pos])
            elif ev_id < 0:
                max_inten = np.squeeze(np.asarray(np.sum(mat_var, axis=1)))
                event_pos = np.argpartition(max_inten, ev_id)[ev_id:]
                event_pos = event_pos[np.argsort(max_inten[event_pos])][0]
                im_val = mat_var[event_pos, :].todense().transpose()
                title = '%s-largest Event. ID %s: %s' % (np.abs(ev_id), \
                    str(self.event_id[event_pos]), self.event_name[event_pos])
            else:
                im_val = np.max(mat_var, axis=0).todense().transpose()
                title = '%s max intensity at each point' % self.tag.haz_type

            array_val.append(im_val)
            l_title.append(title)

        return u_plot.geo_im_from_array(array_val, self.centroids.coord,
                                        col_name, l_title, **kwargs)

    def _centr_plot(self, centr_id, mat_var, col_name):
        """"Plot a centroid of the input matrix.

        Parameters:
            centr_id (int): If centr_id > 0, plot mat_var
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum mat_var of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest mat_var
                are reached.
            mat_var (sparse matrix): Sparse matrix where each column represents
                a centroid
            col_name (sparse matrix): Colorbar label

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if centr_id > 0:
            try:
                centr_pos = np.where(self.centroids.id == centr_id)[0][0]
            except IndexError:
                LOGGER.error('Wrong centroid id: %s.', centr_id)
                raise ValueError from IndexError
            array_val = mat_var[:, centr_pos].todense()
            title = 'Centroid ID %s: (%s, %s)' % (str(centr_id), \
                    self.centroids.coord[centr_pos, 0], \
                    self.centroids.coord[centr_pos, 1])
        elif centr_id < 0:
            max_inten = np.squeeze(np.asarray(np.sum(mat_var, axis=0)))
            centr_pos = np.argpartition(max_inten, centr_id)[centr_id:]
            centr_pos = centr_pos[np.argsort(max_inten[centr_pos])][0]
            array_val = mat_var[:, centr_pos].todense()
            title = '%s-largest Centroid. ID %s: (%s, %s)' % \
                (np.abs(centr_id), str(self.centroids.id[centr_pos]), \
                 self.centroids.coord[centr_pos, 0], \
                 self.centroids.coord[centr_pos, 1])
        else:
            array_val = np.max(mat_var, axis=1).todense()
            title = '%s max intensity at each event' % self.tag.haz_type

        graph = u_plot.Graph2D(title)
        graph.add_subplot('Event number', col_name)
        graph.add_curve(range(len(array_val)), array_val, 'b')
        graph.set_x_lim(range(len(array_val)))
        return graph.get_elems()

    def _loc_return_inten(self, return_periods, inten, exc_inten):
        """ Compute local exceedence intensity for given return period.

        Parameters:
            return_periods (np.array): return periods to consider
            cen_pos (int): centroid position

        Returns:
            np.array
        """
        # sorted intensity
        sort_pos = np.argsort(inten, axis=0)[::-1, :]
        columns = np.ones(inten.shape, int)
        columns *= np.arange(columns.shape[1])
        inten_sort = inten[sort_pos, columns]
        # cummulative frequency at sorted intensity
        freq_sort = self.frequency[sort_pos]
        np.cumsum(freq_sort, axis=0, out=freq_sort)

        for cen_idx in range(inten.shape[1]):
            exc_inten[:, cen_idx] = self._cen_return_inten(
                inten_sort[:, cen_idx], freq_sort[:, cen_idx],
                self.intensity_thres, return_periods)

    def _check_events(self):
        """ Check that all attributes but centroids contain consistent data.
        Put default date, event_name and orig if not provided. Check not
        repeated events (i.e. with same date and name)

        Raises:
            ValueError
        """
        num_ev = len(self.event_id)
        num_cen = len(self.centroids.id)
        if np.unique(self.event_id).size != num_ev:
            LOGGER.error("There are events with the same identifier.")
            raise ValueError

        check.check_oligatories(self.__dict__, self.vars_oblig, 'Hazard.',
                                num_ev, num_ev, num_cen)
        check.check_optionals(self.__dict__, self.vars_opt, 'Hazard.', num_ev)
        self.event_name = check.array_default(num_ev, self.event_name, \
            'Hazard.event_name', list(self.event_id))
        self.date = check.array_default(num_ev, self.date, 'Hazard.date', \
                            np.ones(self.event_id.shape, dtype=int))
        self.orig = check.array_default(num_ev, self.orig, 'Hazard.orig', \
                            np.zeros(self.event_id.shape, dtype=bool))
        if len(self._events_set()) != num_ev:
            LOGGER.error("There are events with same date and name.")
            raise ValueError

    @staticmethod
    def _cen_return_inten(inten, freq, inten_th, return_periods):
        """From ordered intensity and cummulative frequency at centroid, get
        exceedance intensity at input return periods.

        Parameters:
            inten (np.array): sorted intensity at centroid
            freq (np.array): cummulative frequency at centroid
            inten_th (float): intensity threshold
            return_periods (np.array): return periods

        Returns:
            np.array
        """
        inten_th = np.asarray(inten > inten_th).squeeze()
        inten_cen = inten[inten_th]
        freq_cen = freq[inten_th]
        if not inten_cen.size:
            return np.zeros((return_periods.size,))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pol_coef = np.polyfit(np.log(freq_cen), inten_cen, deg=1)
        except ValueError:
            pol_coef = np.polyfit(np.log(freq_cen), inten_cen, deg=0)
        inten_fit = np.polyval(pol_coef, np.log(1/return_periods))
        wrong_inten = np.logical_and(return_periods > np.max(1/freq_cen), \
                np.isnan(inten_fit))
        inten_fit[wrong_inten] = 0.

        return inten_fit

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
