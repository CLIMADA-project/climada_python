"""
Define Hazard ABC.
"""

#from datetime import date
import abc
import pickle
import numpy as np
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.source_mat import CentroidsMat

class Hazard(metaclass=abc.ABCMeta):
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

    def is_hazard(self):
        """ Checks if the attributes contain consistent data.

        Raises
        ------
            ValueError
        """
        # TODO: raise Error if instance is not well filled

    def load(self, file_name, description=None, haztype=None, centroids=None,
             out_file_name=None):
        """Read, check hazard (and its contained centroids) and save to pkl.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
            haztype (str, optional): acronym of the hazard type (e.g. 'TC')
            centroids (Centroids, optional): Centroids instance
            out_file_name (str, optional): output file name to save as pkl

        Raises
        ------
            ValueError
        """
        self._read(file_name, description, haztype, centroids)
        self.is_hazard()
        self.centroids.is_centroids()
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)

    @abc.abstractmethod
    def _read(self, file_name, description=None, haztype=None,
              centroids=None):
        """ Read input file. Abstract method. To be implemented by subclass.
        If centroids are not provided, they are read from file_name.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
            haztype (str, optional): acronym of the hazard type (e.g. 'TC')
            centroids (Centroids, optional): Centroids instance
        """
