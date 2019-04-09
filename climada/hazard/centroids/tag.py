"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Tag class.
"""

__all__ = ['Tag']

import os

class Tag(object):
    """Source data tag for Centroids.

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
        return ' File: ' + self.join_file_names() + '\n Description: ' + \
            self.join_descriptions()

    __repr__ = __str__
