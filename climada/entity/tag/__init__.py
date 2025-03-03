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

Define (deprecated) Tag class.
"""

from deprecation import deprecated

from .tag import Tag as _Tag


# deprecating the whole class instead of just the constructor (s.b.) would be preferable
# but deprecated classes seem to cause an issue when unpickling with the pandas.HDFStore
# which is used in Exposures.from_hdf5()
#
# @deprecated(details="This class is not supported anymore.")
class Tag(_Tag):
    """kept for backwards compatibility with climada <= 3.3"""

    @deprecated(
        details="This class is not supported anymore and will be removed in the next"
        " version of climada."
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
