"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Tag class.
"""

__all__ = ['Tag']

import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

class Tag(object):
    """Contain information used to tag a Hazard.

    Attributes
    ----------
    file_name : str or list(str)
        name of the source file(s)
    haz_type : str
        acronym defining the hazard type (e.g. 'TC')
    description : str or list(str)
        description(s) of the data
    """

    def __init__(self,
                 haz_type: str = '',
                 file_name: str = '',
                 description: str = ''):
        """Initialize values.

        Parameters
        ----------
        haz_type : str
            acronym of the hazard type (e.g. 'TC').
        file_name : str or list(str), optional
            file name(s) to read
        description : str or list(str), optional
            description of the data
        """
        self.haz_type = haz_type
        self.file_name = file_name
        self.description = description

    def append(self, tag):
        """Append input Tag instance information to current Tag."""
        if self.haz_type == '':
            self.haz_type = tag.haz_type
        if tag.haz_type != self.haz_type:
            raise ValueError("Hazards of different type can't be appended: %s != %s."
                             % (self.haz_type, tag.haz_type))

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
                to_add = [tag.description]
            else:
                to_add = tag.description
            self.description.extend(to_add)

    def join_file_names(self):
        """Get a string with the joined file names."""
        if not isinstance(self.file_name, list):
            join_file = Path(self.file_name).stem
        else:
            join_file = ' + '.join([
                Path(single_name).stem
                for single_name in self.file_name
            ])
        return join_file

    def join_descriptions(self):
        """Get a string with the joined descriptions."""
        if not isinstance(self.file_name, list):
            join_desc = self.description
        else:
            join_desc = ' + '.join([file for file in self.description])
        return join_desc

    def __str__(self):
        return ' Type: ' + self.haz_type + '\n File: ' + \
            self.join_file_names() + '\n Description: ' + \
            self.join_descriptions()

    __repr__ = __str__
