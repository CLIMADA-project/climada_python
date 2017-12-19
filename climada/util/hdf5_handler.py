"""
=====================
hdf5_handler module
=====================

Functionalities to handle HDF5 files. Used for MATLAB files as well.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Mon Dec 18 11:45:21 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

from scipy import sparse
import numpy as np
import h5py

def read(file_name, with_refs=False):
    '''Load a hdf5 data structure from a file.
    INPUTS:
        - file_name: file to load
        - with_refs: enable loading of the references. Default is unset, \
        since it increments the execution time considerably.
    OUTPUTS:
        - contents: dictionary structure containing all the variables.
    '''
    def get_group(group):
        '''Recursive function to get variables from a group.'''
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
    except Exception:
        print('Error reading ' + file_name)
        raise

def get_string(array):
    '''Form string from input array of unisgned integers'''
    return u''.join(chr(c) for c in array)

def get_sparse_mat(mat_dict, shape):
    '''Form sparse matrix from input hdf5 sparse matrix data type'''
    # Check if input has all the necessary data of a sparse matrix
    if ('data' not in mat_dict) or ('ir' not in mat_dict) or \
    ('jc' not in mat_dict):
        raise ValueError('Input data is not a sparse matrix.')

    return sparse.csc_matrix((mat_dict['data'], mat_dict['ir'], \
                              mat_dict['jc']), shape)
