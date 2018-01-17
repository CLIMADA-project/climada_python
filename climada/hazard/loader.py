"""
Define Loader class.
"""

import pickle

class Loader(object):
    """Define functions to load data."""

    def read(self, file_name, description=None, haztype=None,
             centroids=None):
        """ Read input file. To be implemented by subclass.
        If centroids are not provided, they are read from file_name.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
            haztype (str, optional): acronym of the hazard type (e.g. 'TC')
            centroids (Centroids, optional): Centroids instance

        Raises
        ------
            ValueError, KeyError
        """
        raise NotImplementedError

    def check(self):
        """Check instance attributes. To be implemented by subclass.

        Raises
        ------
            ValueError
        """
        raise NotImplementedError

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
        self.read(file_name, description, haztype, centroids)
        self.check()
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)
