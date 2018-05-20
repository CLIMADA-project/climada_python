"""
Define Hazard.
"""

__all__ = ['Hazard',
           'FILE_EXT']

import os
import copy
import logging
from operator import itemgetter
import datetime as dt
import numpy as np
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.source import READ_SET
from climada.util.files_handler import to_list, get_file_names
import climada.util.plot as plot
import climada.util.checker as check

LOGGER = logging.getLogger(__name__)

FILE_EXT = {'.mat':  'MAT',
            '.xls':  'XLS',
            '.xlsx': 'XLS'
           }
""" Supported files format to read from """

RETURN_PER = (25, 50, 100, 250)
""" Default return periods in statistics"""

INTENSITY_THRES = {'TC': 10,
                   'WS': 10,
                   'EQ': 10,
                   'FL': 10,
                   'VQ': 10,
                   'TS': 10,
                   'TR': 10,
                   'LS': 10,
                   'HS': 10,
                   'BF': 10
                  }
""" Intensity threshold per hazard used to filter lower intensities in
statistics """

class Hazard(object):
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
        frequency (np.array): frequency of each event in seconds
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """

    def __init__(self, haz_type='NA', file_name='', description='', \
                 centroids=None):
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

            >>> haz = Hazard('TC', HAZ_TEST_XLS)

            Take centriods from a different source:

            >>> centr = Centroids(HAZ_DEMO_MAT, 'Centroids demo')
            >>> haz = Hazard('TC', HAZ_DEMO_MAT, 'Demo hazard.', centr)
        """
        self.clear()
        if '.' in haz_type and file_name == '':
            LOGGER.error("Provide hazard type.")
            raise ValueError
        self.tag.haz_type = haz_type
        if file_name != '':
            if haz_type == 'NA':
                LOGGER.warning("Hazard type acronym not provided.")
            self.read(file_name, description, centroids)

    def clear(self):
        """Reinitialize attributes."""
        self.tag = TagHazard()
        self.units = 'NA'
        # following values are defined for each event
        self.centroids = Centroids()
        self.event_id = np.array([], int)
        self.frequency = np.array([])
        self.event_name = list()
        self.date = np.array([], int)
        self.orig = np.array([], bool)
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix([]) # events x centroids
        self.fraction = sparse.csr_matrix([])  # events x centroids

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
            file_name (str or list(str), optional): absolute file name(s) or
                folder name containing the files to read
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
        desc_list = to_list(len(all_files), description, 'description')
        centr_list = to_list(len(all_files), centroids, 'centroids')
        var_list = to_list(len(all_files), var_names, 'var_names')
        self.clear()
        for file, desc, centr, var in zip(all_files, desc_list, centr_list,
                                          var_list):
            self.append(self._read_one(file, haz_type, desc, centr, var))

    def plot_stats(self, return_periods=RETURN_PER, orig=False):
        """Compute and plot hazard intensity maps for different return periods.

        Parameters:
            return_periods (tuple(int), optional): return periods to consider
            orig (bool, optional): if true, only historical events considered

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot,
            np.ndarray (return_periods.size x num_centroids)
        """
        inten_stats = self._compute_stats(np.array(return_periods), orig)
        colbar_name = 'Wind intensity (' + self.units + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period ' + str(ret))
        fig, axis = plot.geo_im_from_array(inten_stats, self.centroids.coord, \
                                           colbar_name, title)
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
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.intensity, col_label, **kwargs)
        elif centr is not None:
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
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.fraction, col_label, **kwargs)
        elif centr is not None:
            if isinstance(centr, tuple):
                centr = self.centroids.get_nearest_id(centr[0], centr[1])
            return self._centr_plot(centr, self.fraction, col_label)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def event_name_to_id(self, event_name):
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

    def event_id_to_name(self, event_id):
        """"Get the name of an event id.

        Parameters:
            event_id (id): id of the event

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

    def append(self, hazard):
        """Append variables of the NEW events (i.e event name and date) in
        hazard. Id is perserved if not present in current hazard.
        Otherwise, a new id is provided.

        Parameters:
            hazard (Hazard): Hazard instance to append to current

        Raises:
            ValueError
        """
        self.tag.append(hazard.tag)
        if self.event_id.size == 0:
            hazard._check_events()
            for key in hazard.__dict__:
                self.__dict__[key] = copy.copy(hazard.__dict__[key])
            return

        if (self.units == 'NA') and (hazard.units != 'NA'):
            LOGGER.warning("Initial hazard does not have units.")
            self.units = hazard.units
        elif hazard.units == 'NA':
            LOGGER.warning("Appended hazard does not have units.")
        elif self.units != hazard.units:
            LOGGER.error("Hazards with different units can't be appended"\
                             + ": %s != %s.", self.units, hazard.units)
            raise ValueError

        # Add not repeated events
        hazard._check_events()
        n_ini_ev = self.event_id.size
        new_pos = self._append_events(hazard)
        n_add_ev = len(new_pos)
        if n_add_ev:
            part_name = itemgetter(*new_pos)(hazard.event_name)
            if isinstance(part_name, tuple):
                part_name = list(part_name)
            elif isinstance(part_name, str):
                part_name = [part_name]
            self.event_name = self.event_name + part_name
            self.event_id = np.append(self.event_id, \
                hazard.event_id[new_pos]).astype(int, copy=False)
            self.date = np.append(self.date, hazard.date[new_pos]).\
                astype(int, copy=False)
            self.orig = np.append(self.orig, hazard.orig[new_pos]).\
                astype(bool, copy=False)
            self.frequency = np.append(self.frequency, \
                hazard.frequency[new_pos]).astype(float, copy=False)
            self.intensity = sparse.vstack([self.intensity, \
                sparse.lil_matrix((n_add_ev, self.fraction.shape[1]))], \
                format='lil')
            self.fraction = sparse.vstack([self.fraction, \
                sparse.lil_matrix((n_add_ev, self.fraction.shape[1]))], \
                format='lil')

            # Add centroids
            cen_self, cen_haz = self._append_haz_cent(hazard.centroids)

            for i_ev in range(n_add_ev):
                self.intensity[n_ini_ev + i_ev, cen_self] = \
                    hazard.intensity[new_pos[i_ev], cen_haz]
                self.fraction[n_ini_ev + i_ev, cen_self] = \
                    hazard.fraction[new_pos[i_ev], cen_haz]

            self.intensity = self.intensity.tocsr()
            self.fraction = self.fraction.tocsr()

            # Check event id
            _, unique_idx = np.unique(self.event_id, return_index=True)
            rep_id = [pos for pos in range(self.event_id.size)
                      if pos not in unique_idx]
            sup_id = np.max(self.event_id) + 1
            self.event_id[rep_id] = np.arange(sup_id, sup_id+len(rep_id))

    def calc_year_set(self):
        """ From dates and original event flags, compute yearly events

        Returns:
            dict: key are years, values array with event_ids of that year

        """
        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date[self.orig]])
        orig_yearset = {}
        for year in np.unique(orig_year):
            orig_yearset[year] = self.event_id[self.orig][orig_year == year]
        return orig_yearset

    def get_date_strings(self, event=None):
        """ Return list of date strings for given event or for all events,
        if no event provided.

        Parameters:
            event (str or int, optional): event name or id.

        Returns:
            list(str)
        """
        if event is None:
            l_dates = [self._date_to_str(date) for date in self.date]
        elif isinstance(event, str):
            ev_ids = self.event_name_to_id(event)
            l_dates = [self._date_to_str(self.date[ \
                       np.argwhere(self.event_id == ev_id)[0][0]]) \
                       for ev_id in ev_ids]
        else:
            ev_idx = np.argwhere(self.event_id == event)[0][0]
            l_dates = [self._date_to_str(self.date[ev_idx])]
        return l_dates

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

    def _append_events(self, hazard):
        """ Get events of hazard not present in self.

        Parameters:
            hazard: hazard to append

        Returns:
            list
        """
        set_self = self._events_set()
        set_haz = hazard._events_set()
        new_ev = set_haz.difference(set_self)

        new_pos = list()
        if np.unique(hazard.date).size == 1:
            for ev_tup in new_ev:
                new_pos.append(hazard.event_name.index(ev_tup[0]))
            return new_pos

        for ev_tup in new_ev:
            date_pos = np.argwhere(ev_tup[1] == hazard.date).squeeze(axis=1)
            for pos_cand in date_pos:
                if hazard.event_name[pos_cand] == ev_tup[0]:
                    new_pos.append(pos_cand)
                    break
        return new_pos

    def _append_haz_cent(self, centroids):
        """Append centroids. Get positions of new centroids.

        Parameters:
            centroids (Centroids): centroids to append
        Returns:
            cen_self (np.array): positions in self of new centroids
            cen_haz (np.array): corresponding positions in centroids
        """
        # append different centroids
        n_ini_cen = self.centroids.id.size
        self.centroids.append(centroids)

        self.intensity = sparse.hstack([self.intensity, \
            sparse.lil_matrix((self.intensity.shape[0], \
            self.centroids.id.size - n_ini_cen))], format='lil')
        self.fraction = sparse.hstack([self.fraction, \
            sparse.lil_matrix((self.fraction.shape[0], \
            self.centroids.id.size - n_ini_cen))], format='lil')

        # compute positions of repeated and different centroids
        if np.array_equal(self.centroids.coord, centroids.coord):
            cen_self = np.arange(self.centroids.id.size)
            cen_haz = cen_self
        else:
            cen_self_sort = np.argsort(self.centroids.coord, axis=0)[:, 0]
            cen_haz = np.argsort(centroids.coord, axis=0)[:, 0]
            dtype = {'names':['f{}'.format(i) for i in range(2)],
                     'formats':2 * [centroids.coord.dtype]}
            cen_self = np.in1d(self.centroids.coord[cen_self_sort].view(dtype),
                               centroids.coord[cen_haz].view(dtype))
            cen_self = cen_self_sort[cen_self]
        return cen_self, cen_haz

    @staticmethod
    def _date_to_str(date):
        """ Compute date string from input datetime ordinal int. """
        return dt.date.fromordinal(date).isoformat()

    @staticmethod
    def _read_one(file_name, haz_type, description='', centroids=None, \
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
            Hazard
        """
        LOGGER.info('Reading file: %s', file_name)
        new_haz = Hazard()
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

        check.size(num_ev, self.frequency, 'Hazard.frequency')
        if num_ev == 0 and num_cen == 0:
            check.shape(1, num_cen, self.intensity, 'Hazard.intensity')
            check.shape(1, num_cen, self.fraction, 'Hazard.fraction')
        else:
            check.shape(num_ev, num_cen, self.intensity, 'Hazard.intensity')
            check.shape(num_ev, num_cen, self.fraction, 'Hazard.fraction')
        self.event_name = check.array_default(num_ev, self.event_name, \
            'Hazard.event_name', list(self.event_id))
        self.date = check.array_default(num_ev, self.date, 'Hazard.date', \
                            np.ones(self.event_id.shape, dtype=int))
        self.orig = check.array_default(num_ev, self.orig, 'Hazard.orig', \
                            np.zeros(self.event_id.shape, dtype=bool))
        if len(self._events_set()) != num_ev:
            LOGGER.error("There are events with same date and name.")
            raise ValueError

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

        return plot.geo_im_from_array(array_val, self.centroids.coord,
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

        graph = plot.Graph2D(title)
        graph.add_subplot('Event number', col_name)
        graph.add_curve(range(len(array_val)), array_val, 'b')
        graph.set_x_lim(range(len(array_val)))
        return graph.get_elems()

    def _compute_stats(self, return_periods, orig=False):
        """ Compute intensity map for given return periods.

        Parameters:
            return_periods (np.array): return periods to consider
            orig (bool, optional): if true, only historical events considered

        Returns:
            np.array
        """
        inten_stats = np.zeros((len(return_periods), self.intensity.shape[1]))
        inten = self.intensity
        freq = self.frequency
        if orig:
            inten = inten[self.orig, :]
            freq = freq[self.orig] * self.event_id.size / \
                self.orig.nonzero()[0].size
        for cen_pos in range(self.intensity.shape[1]):
            inten_loc = self._loc_return_inten(return_periods, cen_pos,
                                               inten, freq)
            inten_stats[:, cen_pos] = inten_loc
        return inten_stats

    def _loc_return_inten(self, return_periods, cen_pos, inten, freq):
        """ Compute local intensity for given return period.

        Parameters:
            return_periods (np.array): return periods to consider
            cen_pos (int): centroid position
            inten (sparse.csr_matrix): intensity of the events at centroids
            freq (np.array): events frequncy
        Returns:
            np.array
        """
        inten_pos = np.argwhere(inten[:, cen_pos] >
                                INTENSITY_THRES[self.tag.haz_type])[:, 0]
        if inten_pos.size == 0:
            return np.zeros((return_periods.size, ))
        inten_nz = np.asarray(inten[inten_pos, cen_pos].todense()).squeeze()
        sort_pos = inten_nz.argsort()[::-1]
        try:
            inten_sort = inten_nz[sort_pos]
        except IndexError as err:
            if inten_nz.shape == () and inten_nz.size == 1:
                inten_sort = np.array([inten_nz])
            else:
                raise err
        freq_sort = freq[inten_pos[sort_pos]]
        np.cumsum(freq_sort, out=freq_sort)
        try:
            pol_coef = np.polyfit(np.log(freq_sort), inten_sort, deg=1)
        except ValueError:
            pol_coef = np.polyfit(np.log(freq_sort), inten_sort, deg=0)
        inten_fit = np.polyval(pol_coef, np.log(1/return_periods))
        wrong_inten = np.logical_and(return_periods > np.max(1/freq_sort), \
                    np.isnan(inten_fit))
        return inten_fit[np.logical_not(wrong_inten)]

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
