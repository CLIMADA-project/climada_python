"""
Define Tropical Cyclone.
"""

__all__ = ['Drought']

import logging

from climada.hazard.base import Hazard

LOGGER = logging.getLogger(__name__)

class Drought(Hazard):
    """Contains events of Tropical Cyclones.

    Attributes
    ----------
        tag (TagHazard): information about the source
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id (>0) of each event
        event_name (list): name of each event (set as event_id if no provided)
        frequency (np.array): frequency of each event in seconds
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """

    def __init__(self, file_name='', description='', centroids=None):
        """Initialize values from given file, if given.

        Parameters
        ----------
            file_name (str or list(str), optional): file name(s) or folder name 
                containing the files to read
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
            description (str or list(str), optional): one description of the
                data or a description of each data file
            centroids (Centroids or list(Centroids), optional): Centroids

        Raises
        ------
            ValueError
        """
        self.haz_type = 'TC'
        Hazard.__init__(self, file_name, self.haz_type, description, centroids)
    
    @staticmethod
    def _read_one(file_name, haz_type, description, centroids, var_names=None):
        """ Read input file. If centroids are not provided, they are read
        from file_name.

        Parameters
        ----------
            file_name (str): name of the source file
            haz_type (str): acronym of the hazard type (e.g. 'TC')
            description (str): description of the source data
            centroids (Centroids, optional): Centroids instance
            var_names (dict): name of the variables in the file (e.g. 
                      DEF_VAR_NAME defined in the source modules)

        Raises
        ------
            ValueError, KeyError
        """
#        new_haz = Drought()
        raise NotImplementedError
#        return new_haz
