"""
Define Tag class.
"""

__all__ = ['Tag']

class Tag(object):
    """Definition of one hazard tag.

    Attributes
    ----------
        file_name (str or list(str)): name of the source file(s)
        haz_type (str): acronym defining the hazard type (e.g. 'TC')
        description (str or list(str)): description(s) of the data
    """

    def __init__(self, file_name='', haz_type='NA', description=''):
        """Initialize values.

        Parameters
        ----------
            file_name (str or list(str), optional): file name(s) to read
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC')
            description (str or list(str), optional): description of the data
            
        """
        self.file_name = file_name
        self.haz_type = haz_type
        self.description = description

    def append(self, tag):
        """Append input Tag instance information to current Tag."""
        if self.haz_type == 'NA':
            self.haz_type = tag.haz_type
        if tag.haz_type != self.haz_type:
            raise ValueError("Hazards of different type can't be appended: "\
                             + "%s != %s." % (self.haz_type, tag.haz_type))

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
        