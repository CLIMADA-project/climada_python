"""
Define Hazard ABC.
"""

#from datetime import date
import numpy as np
from scipy import sparse

from climada.hazard.loader import Loader as LoaderHaz
import climada.util.auxiliar as aux
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.source_mat import CentroidsMat

class Hazard(LoaderHaz):
    """Contains events of same hazard type defined at centroids. Interface.

    Attributes
    ----------
        tag (TagHazard): information about the source
        id (int): hazard id
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id of each event
        frequency (np.array): frequency of each event
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """

    def __init__(self, file_name=None, description=None, haztype=None):
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
        self.centroids = CentroidsMat()
        self.event_id = np.array([], np.int64)
        self.frequency = np.array([])
        #self.name = [""]
        #self.date = [date(1,1,1)]  # size: num_events
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix([]) # events x centroids
        self.fraction = sparse.csr_matrix([])  # events x centroids

        # Load values from file_name if provided
        if file_name is not None:
            self.load(file_name, description, haztype)

    def calc_future(self, conf):
        """ Compute the future hazard following the configuration """
        # TODO

    def check(self):
        """ Checks if the attributes contain consistent data.

        Raises
        ------
            ValueError
        """
        self.centroids.check()
        num_ev = len(self.event_id)
        num_cen = len(self.centroids.id)
        aux.check_size(num_ev, self.frequency, 'Hazard.frequency')
        aux.check_shape(num_ev, num_cen, self.intensity, 'Hazard.intensity')
        aux.check_shape(num_ev, num_cen, self.fraction, 'Hazard.fraction')
