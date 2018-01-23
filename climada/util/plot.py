"""
Define auxiliary functions for plots.
"""

import numpy as np
from scipy.interpolate import griddata
import h5py
from matplotlib import cm
import matplotlib.pyplot as plt

import climada.util.hdf5_handler as hdf5
from climada.util.constants import SHAPES_MAT

# Number of pixels in one direction in rendered image
RESOLUTION = 200
# Percentage of pixel to add as border
BUFFER_PERCEN = 10

def add_shapes(subplot):
    """Overlay Earth's contries shapes to given matplotlib.pyplot subplot."""
    file = h5py.File(SHAPES_MAT, 'r')
    h5f = hdf5.read(SHAPES_MAT)
    for shape_i in range(h5f['shapes']['X'].size):
        # Special case since we had to restrict to domestic. See admin0.txt
        if file[h5f['shapes']['X_ALL'][0][shape_i]][:].nonzero()[0].size > 0:
            h5f['shapes']['X'][0][shape_i] = h5f['shapes']['X_ALL'][0][shape_i]
            h5f['shapes']['Y'][0][shape_i] = h5f['shapes']['Y_ALL'][0][shape_i]
        subplot.plot(file[h5f['shapes']['X'][0][shape_i]][:], \
                     file[h5f['shapes']['Y'][0][shape_i]][:], color='0.3', \
                     linewidth=0.5)

def geo_im_from_array(geo_coord, array_im, var_name, title):
    """2D plot of image defined in array over input coordinates.

    Parameters
    ----------
        geo_coord (2d np.array): (lat, lon) for each point
        array_im (np.array): values at each point in geo_coord that are
            ploted
        var_name (str): label to be shown in the colorbar
        title (str): figure title

    Returns
    -------
        matplotlib.figure.Figure (optional)

    Raises
    ------
        ValueError
    """
    if geo_coord.shape[0] != array_im.size:
        raise ValueError("Size mismatch in input array: %s != %s." % \
                         (geo_coord.shape[0], array_im.size))
    # Create regular grid where to interpolate the array
    min_lat = np.min(geo_coord[:, 0])
    max_lat = np.max(geo_coord[:, 0])
    min_lon = np.min(geo_coord[:, 1])
    max_lon = np.max(geo_coord[:, 1])

    grid_x, grid_y = np.mgrid[min_lon : max_lon : complex(0, RESOLUTION), \
                              min_lat : max_lat : complex(0, RESOLUTION)]
    grid_im = griddata((geo_coord[:, 1], geo_coord[:, 0]), array_im, \
                             (grid_x, grid_y))
    grid_im = np.squeeze(grid_im)
    grid_im = np.flip(grid_im.transpose(), axis=0)

    # Define axes
    # add buffer around the lon,lat extent
    buffer = np.max([np.abs(max_lon - min_lon)/BUFFER_PERCEN/2, \
                     np.abs(max_lat - min_lat)/BUFFER_PERCEN/2])
    axes = ([min_lon - buffer, max_lon + buffer, min_lat - buffer, \
             max_lat + buffer])

    # Plot image
    fig, subplot = plt.subplots()
    subplot.axis(axes)
    subimg = subplot.imshow(grid_im, extent=axes, cmap=cm.jet, \
                            interpolation='bilinear')
    plt.colorbar(subimg, ax=subplot, label=var_name)
    subplot.set_xlabel('lon')
    subplot.set_ylabel('lat')
    fig.suptitle(title)

    # Add Earth's countries shapes
    add_shapes(subplot)

    plt.show()
    return fig

def graph_2d(var, xlabel, ylabel, title):
    """Simple 2D graph."""
    fig = plt.figure()
    plt.plot(var)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig.suptitle(title)

    plt.show()
    return fig
