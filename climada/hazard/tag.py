"""
Define Tag class.
"""

__all__ = ['Tag']

import logging
import os

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

    def join_file_names(self):
        """ Get a string with the joined file names. """
        if not isinstance(self.file_name, list):
            join_file = os.path.splitext(os.path.basename(self.file_name))[0]
        else:
            join_file = ' + '.join([os.path.splitext(
                os.path.basename(file))[0] for file in self.file_name])
        return join_file

    def join_descriptions(self):
        """ Get a string with the joined descriptions. """
        if not isinstance(self.file_name, list):
            join_desc = os.path.splitext(os.path.basename(self.description))[0]
        else:
            join_desc = ' + '.join([os.path.splitext(
                os.path.basename(file))[0] for file in self.description])
        return join_desc

    def __str__(self):
        return ' Type: ' + self.haz_type + '\n File: ' + \
            self.join_file_names() + '\n Description: ' + \
            self.join_descriptions()

    __repr__ = __str__
