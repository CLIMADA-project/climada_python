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

    def __init__(self, file_name='', description=''):
        """Initialize values.

        Parameters
        ----------
            file_name (str, optional): file name to read
            description (str, optional): description of the data
        """
        self.file_name = file_name
        self.description = description        

    def append(self, tag):
        """Append input Tag instance information to current Tag."""
        # add file name if not present in tag
        if self.file_name == '':
            self.file_name = tag.file_name  
            self.description = tag.description
        elif tag.file_name not in self.file_name:
            if not isinstance(self.file_name, list):
                self.file_name = [self.file_name]
            self.file_name.append(tag.file_name)
    
            if not isinstance(self.description, list):
                self.description = [self.description]
            self.description.append(tag.description)
