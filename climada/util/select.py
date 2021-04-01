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

module containing functions to support various select methods.
"""



import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

def get_attributes_with_matching_dimension(obj, dims):
    """
    Get the attributes of an object that have len(dims) number
    of dimensions or more, and all dims are individual parts of the
    attribute's shape.


    Parameters
    ----------
    obj : object of any class
        The object from which matching attributes are returned
    dims : list[int]
        List of dimensions size to match

    Returns
    -------
    list_of_attrs : list[str]
        List of names of the attributes with matching dimensions
    """

    list_of_attrs = []
    for attr, value in obj.__dict__.items():

        try:

            if isinstance(value, list):
                value = np.array(value)

            if all([dims.count(i) <= value.shape.count(i) for i in set(dims)]):
                list_of_attrs.append(attr)

        except AttributeError:
            pass

    return list_of_attrs
