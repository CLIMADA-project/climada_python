"""
Define Loader class.
"""

import pickle

class Loader(object):
    """Define functions to load data."""

    def read(self, file_name, description=None):
        """Read input file. To be implemented by subclass.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
        """
        raise NotImplementedError

    def check(self):
        """Check instance attributes. To be implemented by subclass.

        Raises
        ------
            ValueError
        """
        raise NotImplementedError

    def load(self, file_name, description=None, out_file_name=None):
        """Read, check and save as pkl, if output file name.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data
            out_file_name (str, optional): output file name to save as pkl

        Raises
        ------
            ValueError
        """
        self.read(file_name, description)
        self.check()
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)
