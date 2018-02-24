"""
Define auxiliary functions for plots.
"""

__all__ = ['Graph2D',
           'geo_bin_from_array',
           'geo_im_from_array',
           'show'
          ]

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
# Degrees to add in the border
BUFFER_DEG = 1
# Enable/Disable show
SHOW = False

# TODO: change to use basemap
def add_shapes(axis):
    """Overlay Earth's contries shapes to given matplotlib.pyplot axis."""
    file = h5py.File(SHAPES_MAT, 'r')
    h5f = hdf5.read(SHAPES_MAT)
    for shape_i in range(h5f['shapes']['X'].size):
        # Special case since we had to restrict to domestic. See admin0.txt
        if file[h5f['shapes']['X_ALL'][0][shape_i]][:].nonzero()[0].size > 0:
            h5f['shapes']['X'][0][shape_i] = h5f['shapes']['X_ALL'][0][shape_i]
            h5f['shapes']['Y'][0][shape_i] = h5f['shapes']['Y_ALL'][0][shape_i]
        axis.plot(file[h5f['shapes']['X'][0][shape_i]][:], \
                     file[h5f['shapes']['Y'][0][shape_i]][:], color='0.3', \
                     linewidth=0.5)

def geo_bin_from_array(geo_coord, array_val, var_name, title):
    """Plot array values binned over input coordinates.

    Parameters
    ----------
        geo_coord (2d np.array): (lat, lon) for each point
        array_val (np.array): values at each point in geo_coord that are
            binned
        var_name (str): label to be shown in the colorbar
        title (str): figure title

    Returns
    -------
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

    Raises
    ------
        ValueError
    """
    if geo_coord.shape[0] != array_val.size:
        raise ValueError("Size mismatch in input array: %s != %s." % \
                         (geo_coord.shape[0], array_val.size))

    # Define axes
    # add buffer around the lon,lat extent
    min_lat, max_lat, min_lon, max_lon = get_borders(geo_coord)
    axes = ([min_lon - BUFFER_DEG, max_lon + BUFFER_DEG, min_lat - BUFFER_DEG,\
             max_lat + BUFFER_DEG])

    # Plot bins
    fig, axs = plt.subplots()
    plt.axis(axes)
    # By default, this makes the mean value of all values in each bin. It can
    # be changed by setting reduce_C_function
    plt.hexbin(geo_coord[:, 1], geo_coord[:, 0], C=array_val, cmap='Wistia', \
               gridsize=int(array_val.size/2))
    col_bar = plt.colorbar()
    col_bar.set_label(var_name)
    axs.set_xlabel('lon')
    axs.set_ylabel('lat')
    fig.suptitle(title)

    # Add Earth's countries shapes
    add_shapes(axs)

    # show figure if activate and return figure and axis
    if SHOW:
        plt.show()
    return fig, axs

def geo_im_from_array(geo_coord, array_im, var_name, title):
    """Image plot defined in array over input coordinates.

    Parameters
    ----------
        geo_coord (2d np.array): (lat, lon) for each point
        array_im (np.array): values at each point in geo_coord that are
            ploted
        var_name (str): label to be shown in the colorbar
        title (str): figure title

    Returns
    -------
        matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot

    Raises
    ------
        ValueError
    """
    if geo_coord.shape[0] != array_im.size:
        raise ValueError("Size mismatch in input array: %s != %s." % \
                         (geo_coord.shape[0], array_im.size))
    # Create regular grid where to interpolate the array
    min_lat, max_lat, min_lon, max_lon = get_borders(geo_coord)
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
    fig, axs = plt.subplots()
    axs.axis(axes)
    subimg = axs.imshow(grid_im, extent=axes, cmap=cm.jet, \
                            interpolation='bilinear')
    plt.colorbar(subimg, ax=axs, label=var_name)
    axs.set_xlabel('lon')
    axs.set_ylabel('lat')
    fig.suptitle(title)

    # Add Earth's countries shapes
    add_shapes(axs)

    # show figure if activate and return figure and axis
    if SHOW:
        plt.show()
    return fig, axs

def show():
    """Show plot just if SHOW constant activated."""
    if SHOW:
        plt.tight_layout()
        plt.show()

class Graph2D(object):
    """2D graph object. Handles various subplots and curves."""
    def __init__(self, title='', num_subplots=1, num_row=None, num_col=None):

        if (num_row is None) or (num_col is None):
            self._compute_size(num_subplots)
        else:
            self.num_row = num_row
            self.num_col = num_col
        self.fig = plt.figure(figsize=plt.figaspect(self.num_row/self.num_col))
        self.fig.suptitle(title)
        self.curr_ax = -1
        self.axs = []
        if self.num_col > 1:
            self.fig.subplots_adjust(wspace=0.3)
        if self.num_row > 1:
            self.fig.subplots_adjust(hspace=0.7)

    def add_subplot(self, xlabel, ylabel, title='', on_grid=True):
        """Add subplot to figure."""
        self.curr_ax += 1
        if self.curr_ax >= self.num_col * self.num_row:
            raise ValueError("Number of subplots in figure exceeded. Figure " \
                             "contains only %s subplots." % str(self.num_col +\
                             self.num_row))
        new_ax = self.fig.add_subplot(self.num_row, self.num_col, \
                                      self.curr_ax + 1)
        new_ax.set_xlabel(xlabel)
        new_ax.set_ylabel(ylabel)
        new_ax.set_title(title)
        new_ax.grid(on_grid)
        self.axs.append(new_ax)

    def add_curve(self, var_x, var_y, var_style, var_label=None):
        """Add (x, y) curve to current subplot."""
        if self.curr_ax == -1:
            raise ValueError('Add a subplot first!')
        axis = self.axs[self.curr_ax]
        axis.plot(var_x, var_y, var_style, label=var_label)
        if var_label is not None:
            axis.legend(loc='best')
        axis.grid(True)

    def set_x_lim(self, var_x):
        """Set x axis limits from minimum and maximum provided values."""
        axis = self.axs[self.curr_ax]
        axis.set_xlim([np.min(var_x), np.max(var_x)])

    def set_y_lim(self, var_y):
        """Set y axis limits from minimum and maximum provided values."""
        axis = self.axs[self.curr_ax]
        axis.set_ylim([np.min(var_y), np.max(var_y)])

    def get_elems(self):
        """Return figure and list of all axes (of each subplot).

        Returns
        -------
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        return self.fig, self.axs

    def _compute_size(self, num_sub):
        """Compute number of rows and columns of subplots in figure."""
        if num_sub <= 3:
            self.num_col = num_sub
            self.num_row = 1
        else:
            if num_sub % 3 == 0:
                self.num_col = 3
                self.num_row = int(num_sub/3)
            else:
                self.num_col = 2
                self.num_row = int(num_sub/2) + num_sub % 2

def get_borders(geo_coord):
    """Get min and max latitude and min and max longitude (in this order)."""
    return np.min(geo_coord[:, 0]), np.max(geo_coord[:, 0]), \
        np.min(geo_coord[:, 1]), np.max(geo_coord[:, 1])
