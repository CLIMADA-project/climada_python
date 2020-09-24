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

Functionalities to handle HDF5 files. Used for MATLAB files as well.
"""

__all__ = ['read',
           'get_string',
           'get_str_from_ref',
           'get_list_str_from_ref',
           'get_sparse_csr_mat'
          ]

import logging
from scipy import sparse
import numpy as np
import h5py

def read(file_name, with_refs=False):
    """Load a hdf5 data structure from a file.

        Parameters:
            file_name: file to load
            with_refs: enable loading of the references. Default is unset,
                since it increments the execution time considerably.

        Returns:
            contents: dictionary structure containing all the variables.

        Examples:
            Contents contains the Matlab data in a dictionary.

            >>> contents = read("/pathto/dummy.mat")

            Contents contains the Matlab data and its reference in a
            dictionary.

            >>> contents = read("/pathto/dummy.mat", True)

        Raises:
            Exception while reading
        """
    def get_group(group):
        """Recursive function to get variables from a group."""
        contents = {}
        for name, obj in list(group.items()):
            if isinstance(obj, h5py.Dataset):
                contents[name] = np.array(obj)
            elif isinstance(obj, h5py.Group):
                # it is a group, so call self recursively
                if with_refs or name != "#refs#":
                    contents[name] = get_group(obj)
            # other objects such as links are ignored
        return contents

    try:
        file = h5py.File(file_name, 'r')
        contents = get_group(file)
        file.close()
        return contents
    except OSError:
        logger = logging.getLogger(__name__)
        logger.error('Invalid file: %s', file_name)

def get_string(array):
    """Form string from input array of unisgned integers.

        Parameters:
            array: array of integers

        Returns:
            string
    """
    return u''.join(chr(int(c)) for c in array)

def get_str_from_ref(file_name, var):
    """Form string from a reference HDF5 variable of the given file.

        Parameters:
            file_name: matlab file name
            var: HDF5 reference variable

        Returns:
            string
    """
    file = h5py.File(file_name, 'r')
    obj = file[var]
    return get_string(obj)


def get_list_str_from_ref(file_name, var):
    """Form list of strings from a reference HDF5 variable of the given file.

        Parameters:
            file_name: matlab file name
            var: array of HDF5 reference variable

        Returns:
            string
    """
    name_list = []
    file = h5py.File(file_name, 'r')
    for name in var:
        name_list.append(get_string(file[name[0]][:]).strip())
    return name_list

def get_sparse_csr_mat(mat_dict, shape):
    """Form sparse matrix from input hdf5 sparse matrix data type.

        Parameters:
            mat_dict: dictionary containing the sparse matrix information.
            shape: tuple describing output matrix shape.

        Returns:
            sparse csr matrix
    """
    # Check if input has all the necessary data of a sparse matrix
    if ('data' not in mat_dict) or ('ir' not in mat_dict) or \
    ('jc' not in mat_dict):
        raise ValueError('Input data is not a sparse matrix.')

    return sparse.csc_matrix((mat_dict['data'], mat_dict['ir'],
                              mat_dict['jc']), shape).tocsr()
