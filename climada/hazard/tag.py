"""
Define Tag class.
"""

__all__ = ['Tag']

class Tag(object):
    """Definition of one hazard tag.

    Attributes
    ----------
        file_name (str): name of the source file
        description (str): description of the data
        type (str): acronym defining of the hazard type (e.g. 'TC')
    """

    def __init__(self, file_name=None, description=None, haz_type=None):
        """Initialize values.

        Parameters
        ----------
            file_name (str, optional): file name to read
            description (str, optional): description of the data
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
        """
        self.file_name = file_name
        self.description = description
        self.type = haz_type
        #self._next = 'NA'
        #self._prev = 'NA'
