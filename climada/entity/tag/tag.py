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

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import h5py

STR_DT = h5py.special_dtype(vlen=str)


def _distinct_list_of_str(list_of_str: list, arg: Union[list, str, object]):
    if arg:
        if isinstance(arg, list):
            for itm in arg:
                if str(itm) not in list_of_str:
                    list_of_str.append(str(itm))
        else:
            list_of_str.append(str(arg))
    return list_of_str


class Tag:
    """Deprecated since climada 4.*. This class is only used for unpickling, e.g., when reading
    Exposures hdf5 data files that have been created with climada <=3.*.

    Attributes
    ----------
    file_name : List[str]
        name of the source file
    description : List[str]
        description of the data
    """

    def __init__(
        self,
        file_name: Union[List[str], str] = None,
        description: Union[List[str], str] = None,
    ):
        """Initialize values.

        Parameters
        ----------
        file_name : Any, optional
            file name to read
        description : Any, optional
            description of the data
        """
        self.file_name = _distinct_list_of_str([], file_name)
        self.description = _distinct_list_of_str([], description)

    def __getattribute__(self, name):
        # Need to override this because of pickle.load, which is used in Exposures.from_hdf5:
        # the attribute assignment there is not done neither via __init__ nor via __setattr__.
        # The outcome is e.g., a description of type str
        val = super().__getattribute__(name)
        if name in ["file_name", "description"] and not isinstance(val, list):
            if not val:
                return []
            return [str(val)]
        return val

    def append(self, tag: Tag):
        """Append input Tag instance information to current Tag."""
        self.file_name = _distinct_list_of_str(self.file_name, tag.file_name)
        self.description = _distinct_list_of_str(self.description, tag.description)

    def join_file_names(self):
        """Get a string with the joined file names."""
        return " + ".join([Path(single_name).stem for single_name in self.file_name])

    def join_descriptions(self):
        """Get a string with the joined descriptions."""
        return " + ".join(self.description)

    def __str__(self):
        return (
            " File: "
            + self.join_file_names()
            + "\n Description: "
            + self.join_descriptions()
        )

    __repr__ = __str__

    def to_hdf5(self, hf_data):
        """Create a dataset in the given hdf5 file and fill it with content

        Parameters
        ----------
        hf_data : h5py.File
            will be updated during the call
        """
        hf_str = hf_data.create_dataset(
            "file_name", (len(self.file_name),), dtype=STR_DT
        )
        for i, name in enumerate(self.file_name):
            hf_str[i] = name
        hf_str = hf_data.create_dataset(
            "description", (len(self.description),), dtype=STR_DT
        )
        for i, desc in enumerate(self.description):
            hf_str[i] = desc

    @classmethod
    def from_hdf5(cls, hf_data):
        """Create a Tag from content of the given hdf5 file

        Parameters
        ----------
        hf_data : h5py.File

        Returns
        -------
        Tag
        """
        return cls(
            file_name=[x.decode() for x in hf_data.get("file_name")],
            description=[x.decode() for x in hf_data.get("description")],
        )
