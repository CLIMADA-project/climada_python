"""
Define Tag class.
"""

__all__ = ['Tag']

class Tag(object):
    """Definition of one Exposures, DiscRates, ImpactFuncs or Measures tag.

    Attributes
    ----------
        file_name (str): name of the source file
        description (str): description of the data
    """

    def __init__(self, file_name=None, description=None):
        """Initialize values.

        Parameters
        ----------
            file_name (str, optional): file name to read
            description (str, optional): description of the data
        """
        if file_name is None:
            self.file_name = ''
        else:
            self.file_name = file_name
        if description is None:
            self.description = ''
        else:
            self.description = description
        #self._next = 'NA'
        #self._prev = 'NA'
