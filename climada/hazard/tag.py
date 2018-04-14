"""
Define Tag class.
"""

import logging

__all__ = ['Tag']

LOGGER = logging.getLogger(__name__)

class Tag(object):
    """Contain information used to tag a Hazard.

    Attributes:
        file_name (str or list(str)): name of the source file(s)
        haz_type (str): acronym defining the hazard type (e.g. 'TC')
        description (str or list(str)): description(s) of the data
    """

    def __init__(self, haz_type='NA', file_name='', description=''):
        """Initialize values.

        Parameters:
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC').
            file_name (str or list(str), optional): file name(s) to read
            description (str or list(str), optional): description of the data

        """
        self.haz_type = haz_type
        self.file_name = file_name
        self.description = description

    def append(self, tag):
        """Append input Tag instance information to current Tag."""
        if self.haz_type == 'NA':
            self.haz_type = tag.haz_type
        if tag.haz_type != self.haz_type:
            LOGGER.error("Hazards of different type can't be appended:"\
                 + " %s != %s.", self.haz_type, tag.haz_type)
            raise ValueError

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

    def __str__(self):
        return ' Type: ' + self.haz_type + '\n File: ' + self.file_name +\
            '\n Description: ' + self.description

    __repr__ = __str__
