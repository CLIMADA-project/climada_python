"""
Define Tag class.
"""

__all__ = ['Tag']

import os

class Tag(object):
    """Source data tag for Exposures, DiscRates, ImpactFuncSet, MeasureSet.

    Attributes:
        file_name (str): name of the source file
        description (str): description of the data
    """

    def __init__(self, file_name='', description=''):
        """Initialize values.

        Parameters:
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
        elif tag.file_name == '':
            return
        else:
            if not isinstance(self.file_name, list):
                self.file_name = [self.file_name]
            if not isinstance(tag.file_name, list):
                to_add = [tag.file_name]
            else:
                to_add = tag.file_name
            self.file_name.extend(to_add)

            if not isinstance(self.description, list):
                self.description = [self.description]
            if not isinstance(tag.description, list):
                tag.description = [tag.description]
            self.description.extend(tag.description)

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
        if not isinstance(self.description, list):
            join_desc = self.description
        else:
            join_desc = ' + '.join([desc for desc in self.description])
        return join_desc

    def __str__(self):
        return ' File: ' + self.join_file_names() + '\n Description: ' + \
            self.join_descriptions()

    __repr__ = __str__
