"""
Define Hazard.
"""

__all__ = ['Hazard']

import os
import copy
from array import array
import logging
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

INTENSITY_THRES = {'TC': 10
                  }
""" Intensity threshold used to filter lower intensities in statistics """

# TODO: add original events array.

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

            >>> centr = Centroids(HAZ_TEST_XLS, 'Centroids demo')
            >>> haz = Hazard('TC', HAZ_TEST_XLS, 'Demo hazard.', centr)
        """
        self.clear()
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
            LOGGER.info('Reading file: %s', file)
            self.append(self._read_one(file, haz_type, desc, centr, var))

    def plot_stats(self, return_periods=RETURN_PER):
        """Compute and plot hazard intensity maps for different return periods.

        Parameters:
            return_periods (tuple(int), optional): return periods to consider

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot,
            np.ndarray (return_periods.size x num_centroids)
        """
        inten_stats = self._compute_stats(np.array(return_periods))
        colbar_name = 'Wind intensity (' + self.units + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period ' + str(ret))
        fig, axis = plot.geo_im_from_array(inten_stats, self.centroids.coord, \
                                           colbar_name, title)
        return fig, axis, inten_stats

    def plot_intensity(self, event=None, centr_id=None):
        """Plot intensity values for a selected event or centroid.

        Parameters:
            event (int or str, optional): If event > 0, plot intensities of
                event with id = event. If event = 0, plot maximum intensity in
                each centroid. If event < 0, plot abs(event)-largest event. If
                event is string, plot event with name in event.
            centr_id (int, optional): If centr_id > 0, plot intensities
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum intensity of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where higher intensities
                are reached.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        Raises:
            ValueError
        """
        col_label = 'Intensity %s' % self.units

        if event is not None:
            if isinstance(event, str):
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.intensity, col_label)
        elif centr_id is not None:
            return self._centr_plot(centr_id, self.intensity, col_label)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def plot_fraction(self, event=None, centr_id=None):
        """Plot fraction values for a selected event or centroid.

        Parameters:
            event (int or str, optional): If event > 0, plot fraction of event
                with id = event. If event = 0, plot maximum fraction in each
                centroid. If event < 0, plot abs(event)-largest event. If event
                is string, plot event with name in event.
            centr_id (int, optional): If centr_id > 0, plot fraction
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum fraction of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest fractions
                are reached.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

        Raises:
            ValueError
        """
        col_label = 'Fraction'
        if event is not None:
            if isinstance(event, str):
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.fraction, col_label)
        elif centr_id is not None:
            return self._centr_plot(centr_id, self.fraction, col_label)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def event_name_to_id(self, event_name):
        """"Get an event id from its name.

        Parameters:
            event_name (str): Event name

        Returns:
            int

        Raises:
            ValueError
        """
        try:
            return self.event_id[self.event_name.index(event_name)]
        except (ValueError, IndexError):
            LOGGER.error("No event with name: %s", event_name)
            raise ValueError

    def append(self, hazard):
        """Check and append variables of input Hazard to current Hazard.
        Repeated events and centroids will be overwritten.

        Parameters:
            hazard (Hazard): Hazard instance to append to current

        Raises:
            ValueError
        """
        hazard._check_events()
        if self.event_id.size == 0:
            for key in hazard.__dict__:
                self.__dict__[key] = copy.copy(hazard.__dict__[key])
            return

        n_ini_cen = self.centroids.id.size
        n_ini_ev = self.event_id.size
        self.tag.append(hazard.tag)
        if (self.units == 'NA') and (hazard.units != 'NA'):
            LOGGER.warning("Initial hazard does not have units.")
            self.units = hazard.units
        elif hazard.units == 'NA':
            LOGGER.warning("Appended hazard does not have units.")
        elif self.units != hazard.units:
            LOGGER.error("Hazards with different units can't be appended"\
                             + ": %s != %s.", self.units, hazard.units)
            raise ValueError

        # Add not repeated centroids
        # for each input centroid, position in the final centroid vector
        new_cen_pos = self.centroids.append(hazard.centroids)
        n_add_cen = np.where(np.array(new_cen_pos) >= n_ini_cen)[0].size
        if n_add_cen:
            self.intensity = sparse.hstack([self.intensity, \
                sparse.lil_matrix((self.intensity.shape[0], n_add_cen))], \
                format='lil')
            self.fraction = sparse.hstack([self.fraction, \
                sparse.lil_matrix((self.fraction.shape[0], n_add_cen))], \
                format='lil')

        # Add not repeated events
        # for each input event, position in the final event vector
        new_ev_pos = array('l')
        new_name = list()
        new_dt = array('L')
        new_id = array('L')
        self._append_events(hazard, new_ev_pos, new_name, new_id, new_dt)
        n_add_ev = np.where(np.array(new_ev_pos) >= n_ini_ev)[0].size
        sparse_add = n_add_ev - 1
        if n_ini_cen != 0 or n_ini_ev != 0:
            sparse_add = n_add_ev
        if n_add_ev:
            self.event_name = self.event_name + new_name
            self.event_id = np.append(self.event_id, np.array(new_id)).\
                astype(int, copy=False)
            self.date = np.append(self.date, np.array(new_dt)).\
                astype(int, copy=False)
            self.intensity = sparse.vstack([self.intensity, \
                sparse.lil_matrix((sparse_add, self.fraction.shape[1]))], \
                format='lil')
            self.fraction = sparse.vstack([self.fraction, \
                sparse.lil_matrix((sparse_add, self.fraction.shape[1]))], \
                format='lil')
            self.frequency = np.append(self.frequency, np.zeros(n_add_ev))

        # fill intensity, fraction and frequency
        for i_ev in range(hazard.event_id.size):
            self.intensity[new_ev_pos[i_ev], new_cen_pos] = \
            hazard.intensity[i_ev, :]
            self.fraction[new_ev_pos[i_ev], new_cen_pos] = \
            hazard.fraction[i_ev, :]
            self.frequency[new_ev_pos[i_ev]] = hazard.frequency[i_ev]

        self.intensity = self.intensity.tocsr()
        self.fraction = self.fraction.tocsr()

    def calc_probabilistic(self):
        """Compute and append probabilistic events from current historical."""
        LOGGER.error('Probabilistic set not implemented yet in %s.', self)
        raise NotImplementedError

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

    def _check_events(self):
        """ Check that all attributes but centroids contain consistent data.

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
        check.array_default(num_ev, self.event_name, 'Hazard.event_name', \
                            list(self.event_id))
        self.date = check.array_default(num_ev, self.date, 'Hazard.date', \
                            np.ones(self.event_id.shape, dtype=int))

    def _append_events(self, hazard, new_ev_pos, new_name, new_id, new_dt):
        """Iterate over hazard events and collect their new position"""
        try:
            max_id = int(np.max(self.event_id))
        except ValueError:
            max_id = 0
        for ev_id, ev_name, ev_dt in zip(hazard.event_id, hazard.event_name,
                                         hazard.date):
            try:
                found = self.event_name.index(ev_name)
                new_ev_pos.append(found)
                if ((ev_id in self.event_id) or (ev_id in new_id)) and \
                (ev_id != self.event_id[found]):
                    max_id += 1
                    self.event_id[found] = max_id
                else:
                    self.event_id[found] = ev_id
                    max_id = max(max_id, ev_id)
            except ValueError:
                new_ev_pos.append(len(self.event_id) + len(new_name))
                new_name.append(ev_name)
                new_dt.append(ev_dt)
                if (ev_id in self.event_id) or (ev_id in new_id):
                    max_id += 1
                    new_id.append(max_id)
                else:
                    new_id.append(ev_id)
                    max_id = max(max_id, ev_id)

    def _event_plot(self, event_id, mat_var, col_name):
        """"Plot an event of the input matrix.

        Parameters:
            event_id (int): If event_id > 0, plot mat_var of
                event with id = event_id. If event_id = 0, plot maximum
                mat_var in each centroid. If event_id < 0, plot
                abs(event_id)-largest event.
            mat_var (sparse matrix): Sparse matrix where each row is an event
            col_name (sparse matrix): Colorbar label

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if event_id > 0:
            try:
                event_pos = np.where(self.event_id == event_id)[0][0]
            except IndexError:
                LOGGER.error('Wrong event id: %s.', event_id)
                raise ValueError from IndexError
            array_val = mat_var[event_pos, :].todense().transpose()
            title = 'Event ID %s: %s' % (str(self.event_id[event_pos]), \
                                      self.event_name[event_pos])
        elif event_id < 0:
            max_inten = np.squeeze(np.asarray(np.sum(mat_var, axis=1)))
            event_pos = np.argpartition(max_inten, event_id)[event_id:]
            event_pos = event_pos[np.argsort(max_inten[event_pos])][0]
            array_val = mat_var[event_pos, :].todense().transpose()
            title = '%s-largest Event. ID %s: %s' % (np.abs(event_id), \
                str(self.event_id[event_pos]), self.event_name[event_pos])
        else:
            array_val = np.max(mat_var, axis=0).todense().transpose()
            title = '%s max intensity at each point' % self.tag.haz_type

        return plot.geo_im_from_array(array_val, self.centroids.coord, \
                                      col_name, title)

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

    def _compute_stats(self, return_periods):
        """ Compute intensity map for given return periods.

        Parameters:
            return_periods (np.array): return periods to consider

        Returns:
            np.array
        """
        inten_stats = np.zeros((len(return_periods), self.intensity.shape[1]))
        for cen_pos in range(self.intensity.shape[1]):
            inten_loc = self._loc_return_inten(return_periods, cen_pos)
            inten_stats[:, cen_pos] = inten_loc
        return inten_stats

    def _loc_return_inten(self, return_periods, cen_pos):
        """ Compute local intensity for given return period.

        Parameters:
            return_periods (np.array): return periods to consider
            cen_pos (int): centroid position

        Returns:
            np.array
        """
        inten_pos = np.argwhere(self.intensity[:, cen_pos] > \
                                INTENSITY_THRES[self.tag.haz_type])[:, 0]
        if inten_pos.size == 0:
            LOGGER.warning('No intensities over threshold %s for centroid '\
                           '%s.', INTENSITY_THRES[self.tag.haz_type], cen_pos)
            return np.zeros((return_periods.size, ))
        inten_nz = np.asarray(self.intensity[inten_pos, cen_pos]. \
                              todense()).squeeze()
        sort_pos = inten_nz.argsort()[::-1]
        inten_sort = inten_nz[sort_pos]
        freq_sort = self.frequency[inten_pos[sort_pos]]
        np.cumsum(freq_sort, out=freq_sort)
        pol_coef = np.polyfit(np.log(freq_sort), inten_sort, deg=1)
        inten_fit = np.polyval(pol_coef, np.log(1/return_periods))
        wrong_inten = np.logical_and(return_periods > np.max(1/freq_sort), \
                    np.isnan(inten_fit))
        return inten_fit[np.logical_not(wrong_inten)]

    def __str__(self):
        return self.tag.__str__()

    __repr__ = __str__
