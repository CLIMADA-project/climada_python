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
        if file_name is None:
            self.file_name = ''
        else:
            self.file_name = file_name
        if description is None:
            self.description = ''
        else:
            self.description = description
        if haz_type is None:
            self.type = 'NA'
        else:
            self.type = haz_type
        #self._next = 'NA'
        #self._prev = 'NA'
