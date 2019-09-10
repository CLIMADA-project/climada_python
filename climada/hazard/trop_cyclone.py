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

Define TropCyclone class.
"""

__all__ = ['TropCyclone']

import itertools
import logging
import datetime as dt
import numpy as np
from scipy import sparse
import matplotlib.animation as animation
from numba import jit
from tqdm import tqdm
import time

from climada.hazard.base import Hazard
from climada.hazard.tag import Tag as TagHazard
from climada.hazard.tc_tracks import TCTracks
from climada.hazard.tc_wind import windfield
from climada.hazard.centroids.centr import Centroids
from climada.util.constants import GLB_CENTROIDS_MAT
import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'TC'
""" Hazard type acronym for Tropical Cyclone """

INLAND_MAX_DIST_KM = 1000
""" Maximum inland distance of the centroids in km """

MODEL_VANG = {'H08': 0
             }
""" Enumerate different symmetric wind field calculation."""

class TropCyclone(Hazard):
    """Contains tropical cyclone events.
    Attributes:
        category (np.array(int)): for every event, the TC category using the
            Saffir-Simpson scale:
                -1 tropical depression
                 0 tropical storm
                 1 Hurrican category 1
                 2 Hurrican category 2
                 3 Hurrican category 3
                 4 Hurrican category 4
                 5 Hurrican category 5
    """
    intensity_thres = 17.5
    """ intensity threshold for storage in m/s """

    vars_opt = Hazard.vars_opt.union({'category'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.category = np.array([], int)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def set_from_tracks(self, tracks, centroids=None, description='',
                        model='H08'):
        """Clear and model tropical cyclone from input IBTrACS tracks.
        Parallel process.
        Parameters:
            tracks (TCTracks): tracks of events
            centroids (Centroids, optional): Centroids where to model TC.
                Default: global centroids.
            description (str, optional): description of the events
            model (str, optional): model to compute gust. Default Holland2008.
        Raises:
            ValueError
        """
        num_tracks = tracks.size
        if centroids is None:
            centroids = Centroids()
            centroids.read_mat(GLB_CENTROIDS_MAT)
        # Select centroids which are inside INLAND_MAX_DIST_KM and lat < 61
        coastal_idx = coastal_centr_idx(centroids)
        if not centroids.coord.size:
            centroids.set_meta_to_lat_lon()

        LOGGER.info('Mapping %s tracks to %s centroids.', str(tracks.size),
                    str(centroids.size))
        if self.pool:
            chunksize = min(num_tracks//self.pool.ncpus, 1000)
            tc_haz = self.pool.map(self._tc_from_track, tracks.data,
                                   itertools.repeat(centroids, num_tracks),
                                   itertools.repeat(coastal_idx, num_tracks),
                                   itertools.repeat(model, num_tracks),
                                   chunksize=chunksize)
        else:
            tc_haz = list()
            for track in tracks.data:
                tc_haz.append(self._tc_from_track(track, centroids, coastal_idx,
                                                  model))
        LOGGER.debug('Append events.')
        self._append_all(tc_haz)
        LOGGER.debug('Compute frequency.')
        self._set_frequency(tracks.data)
        self.tag.description = description

    @staticmethod
    def video_intensity(track_name, tracks, centroids, file_name=None,
                        writer=animation.PillowWriter(bitrate=500),
                        **kwargs):
        """ Generate video of TC wind fields node by node and returns its
        corresponding TropCyclone instances and track pieces.

        Parameters:
            track_name (str): name of the track contained in tracks to record
            tracks (TCTracks): tracks
            centroids (Centroids): centroids where wind fields are mapped
            file_name (str, optional): file name to save video, if provided
            writer = (matplotlib.animation.*, optional): video writer. Default:
                pillow with bitrate=500
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            list(TropCyclone), list(np.array)

        Raises:
            ValueError
        """
        # initialization
        track = tracks.get_track(track_name)
        if not track:
            LOGGER.error('%s not found in track data.', track_name)
            raise ValueError
        idx_plt = np.argwhere(np.logical_and(np.logical_and(np.logical_and( \
            track.lon.values < centroids.total_bounds[2] + 1, \
            centroids.total_bounds[0] - 1 < track.lon.values), \
            track.lat.values < centroids.total_bounds[3] + 1), \
            centroids.total_bounds[1] - 1 < track.lat.values)).reshape(-1)

        tc_list = []
        tr_coord = {'lat':[], 'lon':[]}
        for node in range(idx_plt.size-2):
            tr_piece = track.sel(time=slice(track.time.values[idx_plt[node]], \
                track.time.values[idx_plt[node+2]]))
            tr_piece.attrs['n_nodes'] = 2 # plot only one node
            tr_sel = TCTracks()
            tr_sel.append(tr_piece)
            tr_coord['lat'].append(tr_sel.data[0].lat.values[:-1])
            tr_coord['lon'].append(tr_sel.data[0].lon.values[:-1])

            tc_tmp = TropCyclone()
            tc_tmp.set_from_tracks(tr_sel, centroids)
            tc_tmp.event_name = [track.name + ' ' + time.strftime("%d %h %Y %H:%M", \
                time.gmtime(tr_sel.data[0].time[1].values.astype(int)/1000000000))]
            tc_list.append(tc_tmp)

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'Greys'
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.array([tc_.intensity.min() for tc_ in tc_list]).min()
        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.array([tc_.intensity.max() for tc_ in tc_list]).max()

        def run(node):
            tc_list[node].plot_intensity(1, axis=axis, **kwargs)
            axis.plot(tr_coord['lon'][node], tr_coord['lat'][node], 'k')
            axis.set_title(tc_list[node].event_name[0])
            pbar.update()

        if file_name:
            LOGGER.info('Generating video %s', file_name)
            fig, axis = u_plot.make_map()
            pbar = tqdm(total=idx_plt.size-2)
            ani = animation.FuncAnimation(fig, run, frames=idx_plt.size-2,
                                          interval=500, blit=False)
            ani.save(file_name, writer=writer)
            pbar.close()
        return tc_list, tr_coord

    def _set_frequency(self, tracks):
        """Set hazard frequency from tracks data.
        Parameters:
            tracks (list(xr.Dataset))
        """
        if not tracks:
            return
        delta_time = np.max([np.max(track.time.dt.year.values) \
            for track in tracks]) - np.min([np.min(track.time.dt.year.values) \
            for track in tracks]) + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    @staticmethod
    @jit
    def _tc_from_track(track, centroids, coastal_centr, model='H08'):
        """ Set hazard from input file. If centroids are not provided, they are
        read from the same file.
        Parameters:
            track (xr.Dataset): tropical cyclone track.
            centroids (Centroids): Centroids instance. Use global
                centroids if not provided.
            coastal_centr (np.array): indeces of centroids close to coast.
            model (str, optional): model to compute gust. Default Holland2008.
        Raises:
            ValueError, KeyError
        Returns:
            TropCyclone
        """
        new_haz = TropCyclone()
        new_haz.tag = TagHazard(HAZ_TYPE, 'IBTrACS: ' + track.name)
        new_haz.intensity = gust_from_track(track, centroids, coastal_centr,
                                            model)
        new_haz.units = 'm/s'
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        # frequency set when all tracks available
        new_haz.frequency = np.array([1])
        new_haz.event_name = [track.sid]
        new_haz.fraction = new_haz.intensity.copy()
        new_haz.fraction.data.fill(1)
        # store date of start
        new_haz.date = np.array([dt.datetime(
            track.time.dt.year[0], track.time.dt.month[0],
            track.time.dt.day[0]).toordinal()])
        new_haz.orig = np.array([track.orig_event_flag])
        new_haz.category = np.array([track.category])
        return new_haz

def coastal_centr_idx(centroids, lat_max=61):
    """ Compute centroids indices which are inside INLAND_MAX_DIST_KM and
    with lat < lat_max.

    Parameters:
        lat_max (float, optional): Maximum latitude to consider. Default: 61.

    Returns:
        np.array
    """
    if not centroids.dist_coast.size:
        centroids.set_dist_coast()
    return np.logical_and(centroids.dist_coast < INLAND_MAX_DIST_KM*1000,
                          centroids.lat < lat_max).nonzero()[0]

def gust_from_track(track, centroids, coastal_idx=None, model='H08'):
    """ Compute wind gusts at centroids from track. Track is interpolated to
    configured time step.

    Parameters:
        track (xr.Dataset): track infomation
        centroids (Centroids): centroids where gusts are computed
        coastal_idx (np.array): indices of centroids which are close to coast
        model (str, optional): model to compute gust. Default Holland2008

    Returns:
        sparse.csr_matrix
    """
    if coastal_idx is None:
        coastal_idx = coastal_centr_idx(centroids)
    try:
        mod_id = MODEL_VANG[model]
    except KeyError:
        LOGGER.error('Not implemented model %s.', model)
        raise ValueError
    # Compute wind gusts
    intensity = windfield(track, centroids.coord, coastal_idx, mod_id,
                          TropCyclone.intensity_thres)
    return sparse.csr_matrix(intensity)

def surge_from_wind(intensity):
    """ """
    raise NotImplementedError
