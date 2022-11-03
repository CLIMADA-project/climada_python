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

class Tag():
    """Source data tag for Exposures, DiscRates, ImpactFuncSet, MeasureSet.

    Attributes
    ----------
    file_name : str
        name of the source file
    description : str
        description of the data
    """

    def __init__(self,
                 file_name: str = '',
                 description: str = ''):
        """Initialize values.

        Parameters
        ----------
        file_name : str, optional
            file name to read
        description : str, optional
            description of the data
        """
        self.file_name = str(file_name)
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
            if tag.file_name not in self.file_name:
                self.file_name += ' + ' + tag.file_name
            if tag.description not in self.description:
                self.description += ' + ' + tag.description

    def __str__(self):
        return ' File: ' + self.file_name + '\n Description: ' + \
            self.description

    __repr__ = __str__
