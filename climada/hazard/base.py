"""
Define Hazard.
"""

__all__ = ['Hazard']

#from datetime import date
import os
import pickle
import numpy as np
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.source_excel import read as read_excel
from climada.hazard.source_mat import read as read_mat
import climada.util.plot as plot
import climada.util.checker as check

class Hazard(object):
    """Contains events of same hazard type defined at centroids. Interface.

    Attributes
    ----------
        tag (TagHazard): information about the source
        id (int): hazard id
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id (>0) of each event
        event_name (list): name of each event (set as event_id if no provided)
        frequency (np.array): frequency of each event in seconds
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """

    def __init__(self, file_name=None, haztype='NA', description=None):
        """Initialize values from given file, if given.

        Parameters
        ----------
            file_name (str, optional): file name to read
            description (str, optional): description of the data
            haztype (str, optional): acronym of the hazard type (e.g. 'TC')

        Raises
        ------
            ValueError
        """
        self.tag = TagHazard(file_name, description, haztype)
        self.id = 0
        self.units = 'NA'
        # following values are defined for each event
        self.centroids = Centroids()
        self.event_id = np.array([], np.int64)
        self.frequency = np.array([])
        self.event_name = list()
        #self.date = [date(1,1,1)]  # size: num_events
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix([]) # events x centroids
        self.fraction = sparse.csr_matrix([])  # events x centroids

        # Load values from file_name if provided
        if file_name is not None:
            if haztype == 'NA':
                raise ValueError('Provide hazard type acronym.')
            else:
                self.load(file_name, haztype, description)

    def load(self, file_name, haztype, description=None, centroids=None,\
             out_file_name=None):
        """Read, check hazard (and its contained centroids) and save to pkl.

        Parameters
        ----------
            file_name (str): name of the source file
            haztype (str): acronym of the hazard type (e.g. 'TC')
            description (str, optional): description of the source data
            centroids (Centroids, optional): Centroids instance
            out_file_name (str, optional): output file name to save as pkl

        Raises
        ------
            ValueError
        """
        self.read(file_name, description, haztype, centroids)
        self.check()
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)

    def check(self):
        """ Checks if the attributes contain consistent data.

        Raises
        ------
            ValueError
        """
        self.centroids.check()
        num_ev = len(self.event_id)
        num_cen = len(self.centroids.id)
        check.size(num_ev, self.frequency, 'Hazard.frequency')
        check.shape(num_ev, num_cen, self.intensity, 'Hazard.intensity')
        check.shape(num_ev, num_cen, self.fraction, 'Hazard.fraction')
        check.array_default(num_ev, self.event_name, 'Hazard.event_name', \
                            list(self.event_id))

    def read(self, file_name, haztype, description=None, centroids=None):
        """ Read input file. If centroids are not provided, they are read
        from file_name.

        Parameters
        ----------
            file_name (str): name of the source file
            haztype (str): acronym of the hazard type (e.g. 'TC')
            description (str, optional): description of the source data
            centroids (Centroids, optional): Centroids instance

        Raises
        ------
            ValueError, KeyError
        """
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            read_mat(self, file_name, haztype, description, centroids)
        elif (extension == '.xlsx') or (extension == '.xls'):
            read_excel(self, file_name, haztype, description, centroids)
        else:
            raise TypeError('Input file extension not supported: %s.' % \
                            extension)

    def plot_stats(self):
        """Plots describing hazard."""
        # TODO

    def plot_intensity(self, event=None, centr_id=None):
        """Plot intensity values for a selected event or centroid.

        Parameters
        ----------
            event (int or str, optional): If event > 0, plot intensities of
                event with id = event. If event = 0, plot maximum intensity in
                each centroid. If event < 0, plot abs(event)-largest event. If
                event is string, plot event with name in event.
            centr_id (int, optional): If centr_id > 0, plot intensities
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum intensity of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where higher intensities
                are reached.

        Returns
        -------
            matplotlib.figure.Figure (optional)

        Raises
        ------
            ValueError
        """
        col_label = 'Intensity %s' % self.units

        if event is not None:
            if isinstance(event, str):
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.intensity, col_label)
        elif centr_id is not None:
            return self._centr_plot(centr_id, self.intensity, col_label)
        else:
            raise ValueError('Provide one event id or one centroid id.')

    def plot_fraction(self, event=None, centr_id=None):
        """Plot fraction values for a selected event or centroid.

        Parameters
        ----------
            event (int or str, optional): If event > 0, plot fraction of event
                with id = event. If event = 0, plot maximum fraction in each
                centroid. If event < 0, plot abs(event)-largest event. If event
                is string, plot event with name in event.
            centr_id (int, optional): If centr_id > 0, plot fraction
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum fraction of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest fractions
                are reached.

        Returns
        -------
            matplotlib.figure.Figure (optional)

        Raises
        ------
            ValueError
        """
        col_label = 'Fraction'
        if event is not None:
            if isinstance(event, str):
                event = self.event_name_to_id(event)
            return self._event_plot(event, self.fraction, col_label)
        elif centr_id is not None:
            return self._centr_plot(centr_id, self.fraction, col_label)
        else:
            raise ValueError('Provide one event id or one centroid id.')

    def event_name_to_id(self, event_name):
        """"Get an event id from its name.

        Parameters
        ----------
            event_name (str): Event name

        Returns
        -------
            int

        Raises
        ------
            ValueError
        """
        try:
            event_id = self.event_id[self.event_name.index(event_name)]
        except:
            raise ValueError('No event with name: ' + event_name)
        return event_id

    def calc_future(self, conf):
        """ Compute the future hazard following the configuration """
        # TODO

    def _event_plot(self, event_id, mat_var, col_name):
        """"Plot an event of the input matrix.

        Parameters
        ----------
            event_id (int): If event_id > 0, plot mat_var of
                event with id = event_id. If event_id = 0, plot maximum
                mat_var in each centroid. If event_id < 0, plot
                abs(event_id)-largest event.
            mat_var (sparse matrix): Sparse matrix where each row is an event
            col_name (sparse matrix): Colorbar label

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if event_id > 0:
            try:
                event_pos = np.where(self.event_id == event_id)[0][0]
            except IndexError:
                raise IndexError('Wrong event id: %s.' % event_id)
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
            title = '%s max intensity at each point' % self.tag.type

        return plot.geo_im_from_array(self.centroids.coord, array_val, \
                                   col_name, title)

    def _centr_plot(self, centr_id, mat_var, col_name):
        """"Plot a centroid of the input matrix.

        Parameters
        ----------
            centr_id (int): If centr_id > 0, plot mat_var
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum mat_var of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest mat_var
                are reached.
            mat_var (sparse matrix): Sparse matrix where each column represents
                a centroid
            col_name (sparse matrix): Colorbar label

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if centr_id > 0:
            try:
                centr_pos = np.where(self.centroids.id == centr_id)[0][0]
            except IndexError:
                raise IndexError('Wrong centroid id: %s.' % centr_id)
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
            title = '%s max intensity at each event' % self.tag.type

        graph = plot.Graph2D(title)
        graph.add_subplot('Event number', col_name)
        graph.add_curve(range(len(array_val)), array_val, 'b')
        graph.set_x_lim(range(len(array_val)))
        plot.show()
        return graph.get_elems()
