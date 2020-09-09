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

Define Hazard.
"""

__all__ = ['Hazard']

import copy
import itertools
import logging
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import sparse
import matplotlib.pyplot as plt
import h5py
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, calculate_default_transform

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.centr import Centroids
import climada.util.plot as u_plot
import climada.util.checker as check
import climada.util.dates_times as u_dt
from climada.util.config import CONFIG
import climada.util.hdf5_handler as hdf5
import climada.util.coordinates as co

LOGGER = logging.getLogger(__name__)

DEF_VAR_EXCEL = {'sheet_name': {'inten': 'hazard_intensity',
                                'freq': 'hazard_frequency'
                               },
                 'col_name': {'cen_id': 'centroid_id/event_id',
                              'even_id': 'event_id',
                              'even_dt': 'event_date',
                              'even_name': 'event_name',
                              'freq': 'frequency',
                              'orig': 'orig_event_flag'
                             },
                 'col_centroids': {'sheet_name': 'centroids',
                                   'col_name': {'cen_id': 'centroid_id',
                                                'lat': 'latitude',
                                                'lon': 'longitude'
                                               }
                                  }
                }
"""Excel variable names"""

DEF_VAR_MAT = {'field_name': 'hazard',
               'var_name': {'per_id': 'peril_ID',
                            'even_id': 'event_ID',
                            'ev_name': 'name',
                            'freq': 'frequency',
                            'inten': 'intensity',
                            'unit': 'units',
                            'frac': 'fraction',
                            'comment': 'comment',
                            'datenum': 'datenum',
                            'orig': 'orig_event_flag'
                           },
               'var_cent': {'field_names': ['centroids', 'hazard'],
                            'var_name': {'cen_id': 'centroid_ID',
                                         'lat': 'lat',
                                         'lon': 'lon'
                                        }
                           }
              }
"""MATLAB variable names"""

class Hazard():
    """Contains events of some hazard type defined at centroids. Loads from
    files with format defined in FILE_EXT.

    Attributes:
        tag (TagHazard): information about the source
        units (str): units of the intensity
        centroids (Centroids): centroids of the events
        event_id (np.array): id (>0) of each event
        event_name (list(str)): name of each event (default: event_id)
        date (np.array): integer date corresponding to the proleptic
            Gregorian ordinal, where January 1 of year 1 has ordinal 1
            (ordinal format of datetime library)
        orig (np.array): flags indicating historical events (True)
            or probabilistic (False)
        frequency (np.array): frequency of each event in years
        intensity (sparse.csr_matrix): intensity of the events at centroids
        fraction (sparse.csr_matrix): fraction of affected exposures for each
            event at each centroid
    """
    intensity_thres = 10
    """Intensity threshold per hazard used to filter lower intensities. To be
    set for every hazard type"""

    vars_oblig = {'tag',
                  'units',
                  'centroids',
                  'event_id',
                  'frequency',
                  'intensity',
                  'fraction'
                 }
    """Name of the variables needed to compute the impact. Types: scalar, str,
    list, 1dim np.array of size num_events, scipy.sparse matrix of shape
    num_events x num_centroids, Centroids and Tag."""

    vars_def = {'date',
                'orig',
                'event_name'
               }
    """Name of the variables used in impact calculation whose value is
    descriptive and can therefore be set with default values. Types: scalar,
    string, list, 1dim np.array of size num_events.
    """

    vars_opt = set()
    """Name of the variables that aren't need to compute the impact. Types:
    scalar, string, list, 1dim np.array of size num_events."""

    def __init__(self, haz_type='', pool=None):
        """Initialize values.

        Parameters:
            haz_type (str, optional): acronym of the hazard type (e.g. 'TC').

        Examples:
            Fill hazard values by hand:

            >>> haz = Hazard('TC')
            >>> haz.intensity = sparse.csr_matrix(np.zeros((2, 2)))
            >>> ...

            Take hazard values from file:

            >>> haz = Hazard('TC', HAZ_DEMO_MAT)
            >>> haz.read_mat(HAZ_DEMO_MAT, 'demo')

        """
        self.tag = TagHazard()
        self.tag.haz_type = haz_type
        self.units = ''
        self.centroids = Centroids()
        # following values are defined for each event
        self.event_id = np.array([], int)
        self.frequency = np.array([], float)
        self.event_name = list()
        self.date = np.array([], int)
        self.orig = np.array([], bool)
        # following values are defined for each event and centroid
        self.intensity = sparse.csr_matrix(np.empty((0, 0)))  # events x centroids
        self.fraction = sparse.csr_matrix(np.empty((0, 0)))  # events x centroids
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

    def clear(self):
        """Reinitialize attributes."""
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.array([], dtype=var_val.dtype))
            elif isinstance(var_val, sparse.csr_matrix):
                setattr(self, var_name, sparse.csr_matrix(np.empty((0, 0))))
            else:
                setattr(self, var_name, var_val.__class__())

    def check(self):
        """Check dimension of attributes.

        Raises:
            ValueError
        """
        self.centroids.check()
        self._check_events()

    def set_raster(self, files_intensity, files_fraction=None, attrs=None,
                   band=None, src_crs=None, window=False, geometry=False,
                   dst_crs=False, transform=None, width=None, height=None,
                   resampling=Resampling.nearest):
        """Append intensity and fraction from raster file. 0s put to the masked
        values. File can be partially read using window OR geometry.
        Alternatively, CRS and/or transformation can be set using dst_crs and/or
        (transform, width and height).

        Parameters:
            files_intensity (list(str)): file names containing intensity
            files_fraction (list(str)): file names containing fraction
            attrs (dict, optional): name of Hazard attributes and their values
            band (list(int), optional): bands to read (starting at 1), default [1]
            src_crs (crs, optional): source CRS. Provide it if error without it.
            window (rasterio.windows.Windows, optional): window where data is
                extracted
            geometry (shapely.geometry, optional): consider pixels only in shape
            dst_crs (crs, optional): reproject to given crs
            transform (rasterio.Affine): affine transformation to apply
            wdith (float): number of lons for transform
            height (float): number of lats for transform
            resampling (rasterio.warp,.Resampling optional): resampling
                function used for reprojection to dst_crs
        """
        if not attrs:
            attrs = {}
        if not band:
            band = [1]
        if files_fraction is not None and len(files_intensity) != len(files_fraction):
            LOGGER.error('Number of intensity files differs from fraction files: %s != %s',
                         len(files_intensity), len(files_fraction))
            raise ValueError
        self.tag.file_name = str(files_intensity) + ' ; ' + str(files_fraction)

        self.centroids = Centroids()
        if self.pool:
            chunksize = min(len(files_intensity) // self.pool.ncpus, 1000)
            # set first centroids
            inten_list = [sparse.csr.csr_matrix(self.centroids.set_raster_file(
                files_intensity[0], band, src_crs, window, geometry, dst_crs,
                transform, width, height, resampling))]
            inten_list += self.pool.map(
                self.centroids.set_raster_file,
                files_intensity[1:], itertools.repeat(band), itertools.repeat(src_crs),
                itertools.repeat(window), itertools.repeat(geometry),
                itertools.repeat(dst_crs), itertools.repeat(transform),
                itertools.repeat(width), itertools.repeat(height),
                itertools.repeat(resampling), chunksize=chunksize)
            self.intensity = sparse.vstack(inten_list, format='csr')
            if files_fraction is not None:
                fract_list = self.pool.map(
                    self.centroids.set_raster_file,
                    files_fraction, itertools.repeat(band), itertools.repeat(src_crs),
                    itertools.repeat(window), itertools.repeat(geometry),
                    itertools.repeat(dst_crs), itertools.repeat(transform),
                    itertools.repeat(width), itertools.repeat(height),
                    itertools.repeat(resampling), chunksize=chunksize)
                self.fraction = sparse.vstack(fract_list, format='csr')
        else:
            inten_list = []
            for file in files_intensity:
                inten_list.append(self.centroids.set_raster_file(
                    file, band, src_crs, window, geometry, dst_crs, transform,
                    width, height, resampling))
            self.intensity = sparse.vstack(inten_list, format='csr')
            if files_fraction is not None:
                fract_list = []
                for file in files_fraction:
                    fract_list.append(self.centroids.set_raster_file(
                        file, band, src_crs, window, geometry, dst_crs, transform,
                        width, height, resampling))
                self.fraction = sparse.vstack(fract_list, format='csr')

        if files_fraction is None:
            self.fraction = self.intensity.copy()
            self.fraction.data.fill(1)

        if 'event_id' in attrs:
            self.event_id = attrs['event_id']
        else:
            self.event_id = np.arange(1, self.intensity.shape[0] + 1)
        if 'frequency' in attrs:
            self.frequency = attrs['frequency']
        else:
            self.frequency = np.ones(self.event_id.size)
        if 'event_name' in attrs:
            self.event_name = attrs['event_name']
        else:
            self.event_name = list(map(str, self.event_id))
        if 'date' in attrs:
            self.date = np.array([attrs['date']])
        else:
            self.date = np.ones(self.event_id.size)
        if 'orig' in attrs:
            self.orig = np.array([attrs['orig']])
        else:
            self.orig = np.ones(self.event_id.size, bool)
        if 'unit' in attrs:
            self.unit = attrs['unit']

    def set_vector(self, files_intensity, files_fraction=None, attrs=None,
                   inten_name=None, frac_name=None, dst_crs=None):
        """Read vector files format supported by fiona. Each intensity name is
        considered an event.

        Parameters:
            files_intensity (list(str)): file names containing intensity,
                default: ['intensity']
            files_fraction (list(str)): file names containing fraction,
                default: ['fraction']
            attrs (dict, optional): name of Hazard attributes and their values
            inten_name (list(str), optional): name of variables containing
                the intensities of each event
            frac_name (list(str), optional): name of variables containing
                the fractions of each event
            dst_crs (crs, optional): reproject to given crs
        """
        if not attrs:
            attrs = {}
        if not inten_name:
            inten_name = ['intensity']
        if not frac_name:
            inten_name = ['fraction']
        if files_fraction is not None and len(files_intensity) != len(files_fraction):
            LOGGER.error('Number of intensity files differs from fraction files: %s != %s',
                         len(files_intensity), len(files_fraction))
            raise ValueError
        self.tag.file_name = str(files_intensity) + ' ; ' + str(files_fraction)

        self.centroids = Centroids()
        for file in files_intensity:
            inten = self.centroids.set_vector_file(file, inten_name, dst_crs)
            self.intensity = sparse.vstack([self.intensity, inten], format='csr')
        if files_fraction is None:
            self.fraction = self.intensity.copy()
            self.fraction.data.fill(1)
        else:
            for file in files_fraction:
                fract = self.centroids.set_vector_file(file, frac_name, dst_crs)
                self.fraction = sparse.vstack([self.fraction, fract], format='csr')

        if 'event_id' in attrs:
            self.event_id = attrs['event_id']
        else:
            self.event_id = np.arange(1, self.intensity.shape[0] + 1)
        if 'frequency' in attrs:
            self.frequency = attrs['frequency']
        else:
            self.frequency = np.ones(self.event_id.size)
        if 'event_name' in attrs:
            self.event_name = attrs['event_name']
        else:
            self.event_name = list(map(str, self.event_id))
        if 'date' in attrs:
            self.date = np.array([attrs['date']])
        else:
            self.date = np.ones(self.event_id.size)
        if 'orig' in attrs:
            self.orig = np.array([attrs['orig']])
        else:
            self.orig = np.ones(self.event_id.size, bool)
        if 'unit' in attrs:
            self.unit = attrs['unit']

    def reproject_raster(self, dst_crs=False, transform=None, width=None, height=None,
                         resampl_inten=Resampling.nearest, resampl_fract=Resampling.nearest):
        """Change current raster data to other CRS and/or transformation

        Parameters:
            dst_crs (crs, optional): reproject to given crs
            transform (rasterio.Affine): affine transformation to apply
            wdith (float): number of lons for transform
            height (float): number of lats for transform
            resampl_inten (rasterio.warp,.Resampling optional): resampling
                function used for reprojection to dst_crs for intensity
            resampl_fract (rasterio.warp,.Resampling optional): resampling
                function used for reprojection to dst_crs for fraction
        """
        if not self.centroids.meta:
            LOGGER.error('Raster not set')
            raise ValueError
        if not dst_crs:
            dst_crs = self.centroids.meta['crs']
        if transform and not width or transform and not height:
            LOGGER.error('Provide width and height to given transformation.')
            raise ValueError
        if not transform:
            transform, width, height = calculate_default_transform(
                self.centroids.meta['crs'], dst_crs, self.centroids.meta['width'],
                self.centroids.meta['height'], self.centroids.meta['transform'][2],
                (self.centroids.meta['transform'][5]
                 + self.centroids.meta['height'] * self.centroids.meta['transform'][4]),
                (self.centroids.meta['transform'][2]
                 + self.centroids.meta['width'] * self.centroids.meta['transform'][0]),
                self.centroids.meta['transform'][5])
        dst_meta = self.centroids.meta.copy()
        dst_meta.update({'crs': dst_crs, 'transform': transform,
                         'width': width, 'height': height
                        })
        intensity = np.zeros((self.size, dst_meta['height'], dst_meta['width']))
        fraction = np.zeros((self.size, dst_meta['height'], dst_meta['width']))
        kwargs = {'src_transform': self.centroids.meta['transform'],
                  'src_crs': self.centroids.meta['crs'],
                  'dst_transform': transform, 'dst_crs': dst_crs,
                  'resampling': resampl_inten}
        for idx_ev, inten in enumerate(self.intensity.toarray()):
            reproject(
                source=np.asarray(inten.reshape((self.centroids.meta['height'],
                                                 self.centroids.meta['width']))),
                destination=intensity[idx_ev, :, :],
                **kwargs)
        kwargs.update(resampling=resampl_fract)
        for idx_ev, fract in enumerate(self.fraction.toarray()):
            reproject(
                source=np.asarray(
                    fract.reshape((self.centroids.meta['height'],
                                   self.centroids.meta['width']))),
                destination=fraction[idx_ev, :, :],
                **kwargs)
        self.centroids.meta = dst_meta
        self.intensity = sparse.csr_matrix(
            intensity.reshape(self.size, dst_meta['height'] * dst_meta['width']))
        self.fraction = sparse.csr_matrix(
            fraction.reshape(self.size, dst_meta['height'] * dst_meta['width']))
        self.check()

    def reproject_vector(self, dst_crs, scheduler=None):
        """Change current point data to a a given projection

        Parameters:
            dst_crs (crs): reproject to given crs
            scheduler (str, optional): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        self.centroids.set_geometry_points(scheduler)
        self.centroids.geometry = self.centroids.geometry.to_crs(dst_crs)
        self.centroids.lat = self.centroids.geometry[:].y
        self.centroids.lon = self.centroids.geometry[:].x
        self.check()

    def raster_to_vector(self):
        """Change current raster to points (center of the pixels)"""
        self.centroids.set_meta_to_lat_lon()
        self.centroids.meta = dict()
        self.check()

    def vector_to_raster(self, scheduler=None):
        """Change current point data to a raster with same resolution

        Parameters:
            scheduler (str, optional): used for dask map_partitions. “threads”,
                “synchronous” or “processes”
        """
        points_df = gpd.GeoDataFrame(crs=self.centroids.geometry.crs)
        points_df['latitude'] = self.centroids.lat
        points_df['longitude'] = self.centroids.lon
        val_names = ['val' + str(i_ev) for i_ev in range(2 * self.size)]
        for i_ev, inten_name in enumerate(val_names):
            if i_ev < self.size:
                points_df[inten_name] = np.asarray(self.intensity[i_ev, :].toarray()).reshape(-1)
            else:
                points_df[inten_name] = np.asarray(self.fraction[i_ev - self.size, :].toarray()).\
                reshape(-1)
        raster, meta = co.points_to_raster(points_df, val_names, scheduler=scheduler)
        self.intensity = sparse.csr_matrix(raster[:self.size, :, :].reshape(self.size, -1))
        self.fraction = sparse.csr_matrix(raster[self.size:, :, :].reshape(self.size, -1))
        self.centroids = Centroids()
        self.centroids.meta = meta
        self.check()

    def read_mat(self, file_name, description='', var_names=None):
        """Read climada hazard generate with the MATLAB code.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): description of the data
            var_names (dict, default): name of the variables in the file,
                default: DEF_VAR_MAT constant

        Raises:
            KeyError
        """
        if not var_names:
            var_names = DEF_VAR_MAT
        LOGGER.info('Reading %s', file_name)
        self.clear()
        self.tag.file_name = file_name
        self.tag.description = description
        try:
            data = hdf5.read(file_name)
            try:
                data = data[var_names['field_name']]
            except KeyError:
                pass

            haz_type = hdf5.get_string(data[var_names['var_name']['per_id']])
            self.tag.haz_type = haz_type
            self.centroids.read_mat(file_name, var_names=var_names['var_cent'])
            self._read_att_mat(data, file_name, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable: %s", str(var_err))
            raise var_err

    def read_excel(self, file_name, description='', var_names=None):
        """Read climada hazard generate with the MATLAB code.

        Parameters:
            file_name (str): absolute file name
            description (str, optional): description of the data
            centroids (Centroids, optional): provide centroids if not contained
                in the file
            var_names (dict, default): name of the variables in the file,
                default: DEF_VAR_EXCEL constant

        Raises:
            KeyError
        """
        if not var_names:
            var_names = DEF_VAR_EXCEL
        LOGGER.info('Reading %s', file_name)
        haz_type = self.tag.haz_type
        self.clear()
        self.tag.file_name = file_name
        self.tag.haz_type = haz_type
        self.tag.description = description
        try:
            self.centroids.read_excel(file_name, var_names=var_names['col_centroids'])
            self._read_att_excel(file_name, var_names)
        except KeyError as var_err:
            LOGGER.error("Not existing variable: %s", str(var_err))
            raise var_err

    def select(self, event_names=None, date=None, orig=None, reg_id=None, reset_frequency=False):
        """Select events within provided date and/or (historical or synthetical)
        and/or region. Frequency of the events may need to be recomputed!

        Parameters:
            event_names (list(str), optional): names of event
            date (tuple(str or int), optional): (initial date, final date) in
                string ISO format ('2011-01-02') or datetime ordinal integer
            orig (bool, optional): select only historical (True) or only
                synthetic (False)
            reg_id (int, optional): region identifier of the centroids's
                region_id attibute
            reset_frequency (boolean): change frequency of events proportional to
                difference between first and last year (old and new)
                default = False

        Returns:
            Hazard or children
        """
        if type(self) is Hazard:
            haz = Hazard(self.tag.haz_type)
        else:
            haz = self.__class__()
        sel_ev = np.ones(self.event_id.size, dtype=bool)
        sel_cen = np.ones(self.centroids.size, dtype=bool)

        # filter events by date
        if isinstance(date, tuple):
            date_ini, date_end = date[0], date[1]
            if isinstance(date_ini, str):
                date_ini = u_dt.str_to_date(date[0])
                date_end = u_dt.str_to_date(date[1])
            sel_ev &= (date_ini <= self.date) & (self.date <= date_end)
            if not np.any(sel_ev):
                LOGGER.info('No hazard in date range %s.', date)
                return None

        # filter events hist/synthetic
        if isinstance(orig, bool):
            sel_ev &= (self.orig.astype(bool) == orig)
            if not np.any(sel_ev):
                LOGGER.info('No hazard with %s tracks.', str(orig))
                return None

        # filter centroids
        if reg_id is not None:
            sel_cen &= (self.centroids.region_id == reg_id)
            if not np.any(sel_cen):
                LOGGER.info('No hazard centroids with region %s.', str(reg_id))
                return None

        # filter events based on name
        sel_ev = np.argwhere(sel_ev).reshape(-1)
        if isinstance(event_names, list):
            filtered_events = [self.event_name[i] for i in sel_ev]
            try:
                new_sel = [filtered_events.index(n) for n in event_names]
            except ValueError as err:
                name = str(err).replace(" is not in list", "")
                LOGGER.info('No hazard with name %s', name)
                return None
            sel_ev = sel_ev[new_sel]

        sel_cen = sel_cen.nonzero()[0]
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1 \
                                               and var_val.size > 0:
                setattr(haz, var_name, var_val[sel_ev])
            elif isinstance(var_val, sparse.csr_matrix):
                setattr(haz, var_name, var_val[sel_ev, :][:, sel_cen])
            elif isinstance(var_val, list) and var_val:
                setattr(haz, var_name, [var_val[idx] for idx in sel_ev])
            elif var_name == 'centroids':
                if reg_id is not None:
                    setattr(haz, var_name, var_val.select(reg_id))
                else:
                    setattr(haz, var_name, var_val)
            else:
                setattr(haz, var_name, var_val)

        # reset frequency if date span has changed (optional):
        if reset_frequency:
            year_span_old = np.abs(dt.datetime.fromordinal(self.date.max()).year -
                                   dt.datetime.fromordinal(self.date.min()).year) + 1
            year_span_new = np.abs(dt.datetime.fromordinal(haz.date.max()).year -
                                   dt.datetime.fromordinal(haz.date.min()).year) + 1
            haz.frequency = haz.frequency * year_span_old / year_span_new

        haz.sanitize_event_ids()
        return haz

    def local_exceedance_inten(self, return_periods=(25, 50, 100, 250)):
        """Compute exceedance intensity map for given return periods.

        Parameters:
            return_periods (np.array): return periods to consider

        Returns:
            np.array
        """
        # warn if return period is above return period of rarest event:
        for period in return_periods:
            if period > 1 / self.frequency.min():
                LOGGER.warning('Return period %1.1f exceeds max. event return period.', period)
        LOGGER.info('Computing exceedance intenstiy map for return periods: %s',
                    return_periods)
        num_cen = self.intensity.shape[1]
        inten_stats = np.zeros((len(return_periods), num_cen))
        cen_step = int(CONFIG['global']['max_matrix_size'] / self.intensity.shape[0])
        if not cen_step:
            LOGGER.error('Increase max_matrix_size configuration parameter to'
                         ' > %s', str(self.intensity.shape[0]))
            raise ValueError
        # separte in chunks
        chk = -1
        for chk in range(int(num_cen / cen_step)):
            self._loc_return_inten(
                np.array(return_periods),
                self.intensity[:, chk * cen_step:(chk + 1) * cen_step].toarray(),
                inten_stats[:, chk * cen_step:(chk + 1) * cen_step])
        self._loc_return_inten(
            np.array(return_periods),
            self.intensity[:, (chk + 1) * cen_step:].toarray(),
            inten_stats[:, (chk + 1) * cen_step:])
        # set values below 0 to zero if minimum of hazard.intensity >= 0:
        if self.intensity.min() >= 0 and np.min(inten_stats) < 0:
            LOGGER.warning('Exceedance intenstiy values below 0 are set to 0. \
                   Reason: no negative intensity values were found in hazard.')
            inten_stats[inten_stats < 0] = 0
        return inten_stats

    def plot_rp_intensity(self, return_periods=(25, 50, 100, 250),
                          smooth=True, axis=None, **kwargs):
        """Compute and plot hazard exceedance intensity maps for different
        return periods. Calls local_exceedance_inten.

        Parameters:
            return_periods (tuple(int), optional): return periods to consider
            smooth (bool, optional): smooth plot to plot.RESOLUTIONxplot.RESOLUTION
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots

        Returns:
            matplotlib.axes._subplots.AxesSubplot,
            np.ndarray (return_periods.size x num_centroids)
        """
        self._set_coords_centroids()
        inten_stats = self.local_exceedance_inten(np.array(return_periods))
        colbar_name = 'Intensity (' + self.units + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period: ' + str(ret) + ' years')
        _, axis = u_plot.geo_im_from_array(inten_stats, self.centroids.coord,
                                           colbar_name, title, smooth=smooth,
                                           axes=axis, **kwargs)
        return axis, inten_stats

    def plot_intensity(self, event=None, centr=None, smooth=True, axis=None,
                       **kwargs):
        """Plot intensity values for a selected event or centroid.

        Parameters:
            event (int or str, optional): If event > 0, plot intensities of
                event with id = event. If event = 0, plot maximum intensity in
                each centroid. If event < 0, plot abs(event)-largest event. If
                event is string, plot events with that name.
            centr (int or tuple, optional): If centr > 0, plot intensity
                of all events at centroid with id = centr. If centr = 0,
                plot maximum intensity of each event. If centr < 0,
                plot abs(centr)-largest centroid where higher intensities
                are reached. If tuple with (lat, lon) plot intensity of nearest
                centroid.
            smooth (bool, optional): Rescale data to RESOLUTIONxRESOLUTION pixels (see constant
                in module `climada.util.plot`)
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots or for plot function used in centroids plots

        Returns:
            matplotlib.axes._subplots.AxesSubplot

        Raises:
            ValueError
        """
        self._set_coords_centroids()
        col_label = 'Intensity (%s)' % self.units
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(event, self.intensity, col_label,
                                    smooth, axis, **kwargs)
        if centr is not None:
            if isinstance(centr, tuple):
                _, _, centr = self.centroids.get_closest_point(centr[0], centr[1])
            return self._centr_plot(centr, self.intensity, col_label, axis, **kwargs)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def plot_fraction(self, event=None, centr=None, smooth=True, axis=None,
                      **kwargs):
        """Plot fraction values for a selected event or centroid.

        Parameters:
            event (int or str, optional): If event > 0, plot fraction of event
                with id = event. If event = 0, plot maximum fraction in each
                centroid. If event < 0, plot abs(event)-largest event. If event
                is string, plot events with that name.
            centr (int or tuple, optional): If centr > 0, plot fraction
                of all events at centroid with id = centr. If centr = 0,
                plot maximum fraction of each event. If centr < 0,
                plot abs(centr)-largest centroid where highest fractions
                are reached. If tuple with (lat, lon) plot fraction of nearest
                centroid.
            smooth (bool, optional): Rescale data to RESOLUTIONxRESOLUTION pixels (see constant
                in module `climada.util.plot`)
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for pcolormesh matplotlib function
                used in event plots or for plot function used in centroids plots

        Returns:
            matplotlib.axes._subplots.AxesSubplot

        Raises:
            ValueError
        """
        self._set_coords_centroids()
        col_label = 'Fraction'
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(event, self.fraction, col_label, smooth, axis,
                                    **kwargs)
        if centr is not None:
            if isinstance(centr, tuple):
                _, _, centr = self.centroids.get_closest_point(centr[0], centr[1])
            return self._centr_plot(centr, self.fraction, col_label, axis, **kwargs)

        LOGGER.error("Provide one event id or one centroid id.")
        raise ValueError

    def sanitize_event_ids(self):
        """Make sure that event ids are unique"""
        if np.unique(self.event_id).size != self.event_id.size:
            LOGGER.debug('Resetting event_id.')
            self.event_id = np.arange(1, self.event_id.size + 1)

    def get_event_id(self, event_name):
        """Get an event id from its name. Several events might have the same
        name.

        Parameters:
            event_name (str): Event name

        Returns:
            np.array(int)
        """
        list_id = self.event_id[[i_name for i_name, val_name in enumerate(self.event_name)
                                 if val_name == event_name]]
        if list_id.size == 0:
            LOGGER.error("No event with name: %s", event_name)
            raise ValueError
        return list_id

    def get_event_name(self, event_id):
        """Get the name of an event id.

        Parameters:
            event_id (int): id of the event

        Returns:
            str

        Raises:
            ValueError
        """
        try:
            return self.event_name[np.argwhere(
                self.event_id == event_id)[0][0]]
        except IndexError:
            LOGGER.error("No event with id: %s", event_id)
            raise ValueError

    def get_event_date(self, event=None):
        """Return list of date strings for given event or for all events,
        if no event provided.

        Parameters:
            event (str or int, optional): event name or id.

        Returns:
            list(str)
        """
        if event is None:
            l_dates = [u_dt.date_to_str(date) for date in self.date]
        elif isinstance(event, str):
            ev_ids = self.get_event_id(event)
            l_dates = [
                u_dt.date_to_str(self.date[np.argwhere(self.event_id == ev_id)[0][0]])
                for ev_id in ev_ids]
        else:
            ev_idx = np.argwhere(self.event_id == event)[0][0]
            l_dates = [u_dt.date_to_str(self.date[ev_idx])]
        return l_dates

    def calc_year_set(self):
        """From the dates of the original events, get number yearly events.

        Returns:
            dict: key are years, values array with event_ids of that year

        """
        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date[self.orig]])
        orig_yearset = {}
        for year in np.unique(orig_year):
            orig_yearset[year] = self.event_id[self.orig][orig_year == year]
        return orig_yearset

    def append(self, hazard):
        """Append events and centroids in hazard.

        Parameters:
            hazard (Hazard): Hazard instance to append to current

        Raises:
            ValueError
        """
        hazard._check_events()
        if self.event_id.size == 0:
            for key in hazard.__dict__:
                try:
                    self.__dict__[key] = copy.deepcopy(hazard.__dict__[key])
                except TypeError:
                    self.__dict__[key] = copy.copy(hazard.__dict__[key])
            return

        if (self.units == '') and (hazard.units != ''):
            LOGGER.info("Initial hazard does not have units.")
            self.units = hazard.units
        elif hazard.units == '':
            LOGGER.info("Appended hazard does not have units.")
        elif self.units != hazard.units:
            LOGGER.error("Hazards with different units can't be appended: "
                         "%s != %s.", self.units, hazard.units)
            raise ValueError

        centroids_equal = self.centroids.equal(hazard.centroids)
        if not centroids_equal:
            self.centroids.append(hazard.centroids)

        # n_ini_ev = self.event_id.size
        for var_name in vars(self).keys():
            var_old = getattr(self, var_name)
            var_new = getattr(hazard, var_name)
            var_combined = [var_old, var_new]
            if isinstance(var_new, sparse.csr.csr_matrix):
                if centroids_equal:
                    var_combined = sparse.vstack(var_combined, format='csr')
                else:
                    var_combined = sparse.block_diag(var_combined, format='csr')
                setattr(self, var_name, var_combined)
            elif isinstance(var_new, np.ndarray) and var_new.ndim == 1:
                setattr(self, var_name, np.hstack(var_combined))
            elif isinstance(var_new, list):
                setattr(self, var_name, sum(var_combined, []))
            elif isinstance(var_new, TagHazard):
                var_old.append(var_new)

        self.sanitize_event_ids()

    def remove_duplicates(self):
        """Remove duplicate events (events with same name and date)."""
        events = list(zip(self.event_name, self.date))
        set_ev = set(events)
        if len(set_ev) == self.event_id.size:
            return
        unique_pos = sorted([events.index(event) for event in set_ev])
        for var_name, var_val in vars(self).items():
            if isinstance(var_val, sparse.csr.csr_matrix):
                setattr(self, var_name, var_val[unique_pos, :])
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, var_val[unique_pos])
            elif isinstance(var_val, list):
                setattr(self, var_name, [var_val[p] for p in unique_pos])

    def set_frequency(self, yearrange=None):
        """Set hazard frequency from yearrange or intensity matrix.

        Optional parameters:
            yearrange (tuple or list): year range to be used to compute frequency
                per event. If yearrange is not given (None), the year range is
                derived from self.date
        """
        if not yearrange:
            delta_time = dt.datetime.fromordinal(int(np.max(self.date))).year - \
                     dt.datetime.fromordinal(int(np.min(self.date))).year + 1
        else:
            delta_time = max(yearrange)-min(yearrange)+1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    @property
    def size(self):
        """Returns number of events"""
        return self.event_id.size

    def write_raster(self, file_name, intensity=True):
        """Write intensity or fraction as GeoTIFF file. Each band is an event

        Parameters:
            file_name (str): file name to write in tif format
            intensity (bool): if True, write intensity, otherwise write fraction
        """
        variable = self.intensity
        if not intensity:
            variable = self.fraction
        if self.centroids.meta:
            co.write_raster(file_name, variable.toarray(), self.centroids.meta)
        else:
            pixel_geom = self.centroids.calc_pixels_polygons()
            profile = self.centroids.meta
            profile.update(driver='GTiff', dtype=rasterio.float32, count=self.size)
            with rasterio.open(file_name, 'w', **profile) as dst:
                LOGGER.info('Writing %s', file_name)
                for i_ev in range(variable.shape[0]):
                    raster = rasterize(
                        [(x, val) for (x, val) in
                         zip(pixel_geom, np.array(variable[i_ev, :].toarray()).reshape(-1))],
                        out_shape=(profile['height'], profile['width']),
                        transform=profile['transform'], fill=0,
                        all_touched=True, dtype=profile['dtype'],)
                    dst.write(raster.astype(profile['dtype']), i_ev + 1)

    def write_hdf5(self, file_name, todense=False):
        """Write hazard in hdf5 format.

        Parameters:
            file_name (str): file name to write, with h5 format
        """
        LOGGER.info('Writing %s', file_name)
        hf_data = h5py.File(file_name, 'w')
        str_dt = h5py.special_dtype(vlen=str)
        for (var_name, var_val) in self.__dict__.items():
            if var_name == 'centroids':
                self.centroids.write_hdf5(hf_data.create_group(var_name))
            elif var_name == 'tag':
                hf_str = hf_data.create_dataset('haz_type', (1,), dtype=str_dt)
                hf_str[0] = var_val.haz_type
                hf_str = hf_data.create_dataset('file_name', (1,), dtype=str_dt)
                hf_str[0] = str(var_val.file_name)
                hf_str = hf_data.create_dataset('description', (1,), dtype=str_dt)
                hf_str[0] = str(var_val.description)
            elif isinstance(var_val, sparse.csr_matrix):
                if todense:
                    hf_data.create_dataset(var_name, data=var_val.toarray())
                else:
                    hf_csr = hf_data.create_group(var_name)
                    hf_csr.create_dataset('data', data=var_val.data)
                    hf_csr.create_dataset('indices', data=var_val.indices)
                    hf_csr.create_dataset('indptr', data=var_val.indptr)
                    hf_csr.attrs['shape'] = var_val.shape
            elif isinstance(var_val, str):
                hf_str = hf_data.create_dataset(var_name, (1,), dtype=str_dt)
                hf_str[0] = var_val
            elif isinstance(var_val, list) and isinstance(var_val[0], str):
                hf_str = hf_data.create_dataset(var_name, (len(var_val),), dtype=str_dt)
                for i_ev, var_ev in enumerate(var_val):
                    hf_str[i_ev] = var_ev
            elif var_val is not None and var_name != 'pool':
                hf_data.create_dataset(var_name, data=var_val)
        hf_data.close()

    def read_hdf5(self, file_name):
        """Read hazard in hdf5 format.

        Parameters:
            file_name (str): file name to read, with h5 format
        """
        LOGGER.info('Reading %s', file_name)
        self.clear()
        hf_data = h5py.File(file_name, 'r')
        for (var_name, var_val) in self.__dict__.items():
            if var_name == 'centroids':
                self.centroids.read_hdf5(hf_data.get(var_name))
            elif var_name == 'tag':
                self.tag.haz_type = hf_data.get('haz_type')[0]
                self.tag.file_name = hf_data.get('file_name')[0]
                self.tag.description = hf_data.get('description')[0]
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.array(hf_data.get(var_name)))
            elif isinstance(var_val, sparse.csr_matrix):
                hf_csr = hf_data.get(var_name)
                if isinstance(hf_csr, h5py.Dataset):
                    setattr(self, var_name, sparse.csr_matrix(hf_csr))
                else:
                    setattr(self, var_name, sparse.csr_matrix((hf_csr['data'][:],
                                                               hf_csr['indices'][:],
                                                               hf_csr['indptr'][:]),
                                                              hf_csr.attrs['shape']))
            elif isinstance(var_val, str):
                setattr(self, var_name, hf_data.get(var_name)[0])
            elif isinstance(var_val, list):
                setattr(self, var_name, np.array(hf_data.get(var_name)).tolist())
            else:
                setattr(self, var_name, hf_data.get(var_name))
        hf_data.close()

    def concatenate(self, haz_src, append=False):
        """Concatenate events of several hazards

        Parameters:
            haz_src (list): Hazard instances with same centroids and units
            append (bool): If True, append the concatenated hazards to this
                instance, otherwise replace all data in this instance by the
                concatenated data. Default: False.
        """
        if append:
            haz_src = [self] + haz_src
        else:
            self.clear()
            self.centroids = copy.deepcopy(haz_src[-1].centroids)
            self.units = haz_src[-1].units

        # check for new variables
        for key_new in vars(haz_src[-1]).keys():
            if not hasattr(self, key_new):
                setattr(self, key_new, getattr(haz_src[-1], key_new))

        for var_name in vars(self).keys():
            var_src = [getattr(haz, var_name) for haz in haz_src]
            if isinstance(var_src[-1], sparse.csr.csr_matrix):
                setattr(self, var_name, sparse.vstack(var_src, format='csr'))
            elif isinstance(var_src[-1], np.ndarray) and var_src[-1].ndim == 1:
                setattr(self, var_name, np.hstack(var_src))
            elif isinstance(var_src[-1], list):
                setattr(self, var_name, sum(var_src, []))
            elif isinstance(var_src[-1], TagHazard):
                tag_dst = getattr(self, var_name)
                [tag_dst.append(tag) for tag in var_src if tag is not tag_dst]

        self.sanitize_event_ids()

    def _set_coords_centroids(self):
        """If centroids are raster, set lat and lon coordinates"""
        if self.centroids.meta and not self.centroids.coord.size:
            self.centroids.set_meta_to_lat_lon()

    def _events_set(self):
        """Generate set of tuples with (event_name, event_date)"""
        ev_set = set()
        for ev_name, ev_date in zip(self.event_name, self.date):
            ev_set.add((ev_name, ev_date))
        return ev_set

    def _event_plot(self, event_id, mat_var, col_name, smooth, axis=None, **kwargs):
        """Plot an event of the input matrix.

        Parameters:
            event_id (int or np.array(int)): If event_id > 0, plot mat_var of
                event with id = event_id. If event_id = 0, plot maximum
                mat_var in each centroid. If event_id < 0, plot
                abs(event_id)-largest event.
            mat_var (sparse matrix): Sparse matrix where each row is an event
            col_name (sparse matrix): Colorbar label
            smooth (bool, optional): smooth plot to plot.RESOLUTIONxplot.RESOLUTION
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for pcolormesh matplotlib function

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if not isinstance(event_id, np.ndarray):
            event_id = np.array([event_id])
        array_val = list()
        l_title = list()
        for ev_id in event_id:
            if ev_id > 0:
                try:
                    event_pos = np.where(self.event_id == ev_id)[0][0]
                except IndexError:
                    LOGGER.error('Wrong event id: %s.', ev_id)
                    raise ValueError from IndexError
                im_val = mat_var[event_pos, :].toarray().transpose()
                title = 'Event ID %s: %s' % (str(self.event_id[event_pos]),
                                             self.event_name[event_pos])
            elif ev_id < 0:
                max_inten = np.asarray(np.sum(mat_var, axis=1)).reshape(-1)
                event_pos = np.argpartition(max_inten, ev_id)[ev_id:]
                event_pos = event_pos[np.argsort(max_inten[event_pos])][0]
                im_val = mat_var[event_pos, :].toarray().transpose()
                title = '%s-largest Event. ID %s: %s' % (np.abs(ev_id),
                                                         str(self.event_id[event_pos]),
                                                         self.event_name[event_pos])
            else:
                im_val = np.max(mat_var, axis=0).toarray().transpose()
                title = '%s max intensity at each point' % self.tag.haz_type

            array_val.append(im_val)
            l_title.append(title)

        return u_plot.geo_im_from_array(array_val, self.centroids.coord, col_name,
                                        l_title, smooth=smooth, axes=axis, **kwargs)

    def _centr_plot(self, centr_idx, mat_var, col_name, axis=None, **kwargs):
        """Plot a centroid of the input matrix.

        Parameters:
            centr_id (int): If centr_id > 0, plot mat_var
                of all events at centroid with id = centr_id. If centr_id = 0,
                plot maximum mat_var of each event. If centr_id < 0,
                plot abs(centr_id)-largest centroid where highest mat_var
                are reached.
            mat_var (sparse matrix): Sparse matrix where each column represents
                a centroid
            col_name (sparse matrix): Colorbar label
            axis (matplotlib.axes._subplots.AxesSubplot, optional): axis to use
            kwargs (optional): arguments for plot matplotlib function

        Returns:
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        coord = self.centroids.coord
        if centr_idx > 0:
            try:
                centr_pos = centr_idx
            except IndexError:
                LOGGER.error('Wrong centroid id: %s.', centr_idx)
                raise ValueError from IndexError
            array_val = mat_var[:, centr_pos].toarray()
            title = 'Centroid %s: (%s, %s)' % (str(centr_idx),
                                               coord[centr_pos, 0],
                                               coord[centr_pos, 1])
        elif centr_idx < 0:
            max_inten = np.asarray(np.sum(mat_var, axis=0)).reshape(-1)
            centr_pos = np.argpartition(max_inten, centr_idx)[centr_idx:]
            centr_pos = centr_pos[np.argsort(max_inten[centr_pos])][0]
            array_val = mat_var[:, centr_pos].toarray()

            title = '%s-largest Centroid. %s: (%s, %s)' % \
                (np.abs(centr_idx), str(centr_pos), coord[centr_pos, 0],
                 coord[centr_pos, 1])
        else:
            array_val = np.max(mat_var, axis=1).toarray()
            title = '%s max intensity at each event' % self.tag.haz_type

        if not axis:
            _, axis = plt.subplots(1)
        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        axis.set_title(title)
        axis.set_xlabel('Event number')
        axis.set_ylabel(str(col_name))
        axis.plot(range(len(array_val)), array_val, **kwargs)
        axis.set_xlim([0, len(array_val)])
        return axis

    def _loc_return_inten(self, return_periods, inten, exc_inten):
        """Compute local exceedence intensity for given return period.

        Parameters:
            return_periods (np.array): return periods to consider
            cen_pos (int): centroid position

        Returns:
            np.array
        """
        # sorted intensity
        sort_pos = np.argsort(inten, axis=0)[::-1, :]
        columns = np.ones(inten.shape, int)
        # pylint: disable=unsubscriptable-object  # pylint/issues/3139
        columns *= np.arange(columns.shape[1])
        inten_sort = inten[sort_pos, columns]
        # cummulative frequency at sorted intensity
        freq_sort = self.frequency[sort_pos]
        np.cumsum(freq_sort, axis=0, out=freq_sort)

        for cen_idx in range(inten.shape[1]):
            exc_inten[:, cen_idx] = self._cen_return_inten(
                inten_sort[:, cen_idx], freq_sort[:, cen_idx],
                self.intensity_thres, return_periods)

    def _check_events(self):
        """Check that all attributes but centroids contain consistent data.
        Put default date, event_name and orig if not provided. Check not
        repeated events (i.e. with same date and name)

        Raises:
            ValueError
        """
        num_ev = len(self.event_id)
        num_cen = self.centroids.size
        if np.unique(self.event_id).size != num_ev:
            LOGGER.error("There are events with the same identifier.")
            raise ValueError

        check.check_oligatories(self.__dict__, self.vars_oblig, 'Hazard.',
                                num_ev, num_ev, num_cen)
        check.check_optionals(self.__dict__, self.vars_opt, 'Hazard.', num_ev)
        self.event_name = check.array_default(num_ev, self.event_name,
                                              'Hazard.event_name',
                                              list(self.event_id))
        self.date = check.array_default(num_ev, self.date, 'Hazard.date',
                                        np.ones(self.event_id.shape, dtype=int))
        self.orig = check.array_default(num_ev, self.orig, 'Hazard.orig',
                                        np.zeros(self.event_id.shape, dtype=bool))
        if len(self._events_set()) != num_ev:
            LOGGER.error("There are events with same date and name.")
            raise ValueError

    @staticmethod
    def _cen_return_inten(inten, freq, inten_th, return_periods):
        """From ordered intensity and cummulative frequency at centroid, get
        exceedance intensity at input return periods.

        Parameters:
            inten (np.array): sorted intensity at centroid
            freq (np.array): cummulative frequency at centroid
            inten_th (float): intensity threshold
            return_periods (np.array): return periods

        Returns:
            np.array
        """
        inten_th = np.asarray(inten > inten_th).squeeze()
        inten_cen = inten[inten_th]
        freq_cen = freq[inten_th]
        if not inten_cen.size:
            return np.zeros((return_periods.size,))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pol_coef = np.polyfit(np.log(freq_cen), inten_cen, deg=1)
        except ValueError:
            pol_coef = np.polyfit(np.log(freq_cen), inten_cen, deg=0)
        inten_fit = np.polyval(pol_coef, np.log(1 / return_periods))
        wrong_inten = (return_periods > np.max(1 / freq_cen)) & np.isnan(inten_fit)
        inten_fit[wrong_inten] = 0.

        return inten_fit

    def _read_att_mat(self, data, file_name, var_names):
        """Read MATLAB hazard's attributes."""
        self.frequency = np.squeeze(data[var_names['var_name']['freq']])
        self.orig = np.squeeze(data[var_names['var_name']['orig']]).astype(bool)
        self.event_id = np.squeeze(
            data[var_names['var_name']['even_id']].astype(np.int, copy=False))
        try:
            self.units = hdf5.get_string(data[var_names['var_name']['unit']])
        except KeyError:
            pass

        n_cen = self.centroids.size
        n_event = len(self.event_id)
        try:
            self.intensity = hdf5.get_sparse_csr_mat(
                data[var_names['var_name']['inten']], (n_event, n_cen))
        except ValueError as err:
            LOGGER.error('Size missmatch in intensity matrix.')
            raise err
        try:
            self.fraction = hdf5.get_sparse_csr_mat(
                data[var_names['var_name']['frac']], (n_event, n_cen))
        except ValueError as err:
            LOGGER.error('Size missmatch in fraction matrix.')
            raise err
        except KeyError:
            self.fraction = sparse.csr_matrix(np.ones(self.intensity.shape,
                                                      dtype=np.float))
        # Event names: set as event_id if no provided
        try:
            self.event_name = hdf5.get_list_str_from_ref(
                file_name, data[var_names['var_name']['ev_name']])
        except KeyError:
            self.event_name = list(self.event_id)
        try:
            comment = hdf5.get_string(data[var_names['var_name']['comment']])
            self.tag.description += ' ' + comment
        except KeyError:
            pass

        try:
            datenum = data[var_names['var_name']['datenum']].squeeze()
            self.date = np.array([
                (dt.datetime.fromordinal(int(date))
                 + dt.timedelta(days=date % 1)
                 - dt.timedelta(days=366)).toordinal()
                for date in datenum])
        except KeyError:
            pass

    def _read_att_excel(self, file_name, var_names):
        """Read Excel hazard's attributes."""
        dfr = pd.read_excel(file_name, var_names['sheet_name']['freq'])

        num_events = dfr.shape[0]
        self.frequency = dfr[var_names['col_name']['freq']].values
        self.orig = dfr[var_names['col_name']['orig']].values.astype(bool)
        self.event_id = dfr[var_names['col_name']['even_id']].values. \
            astype(int, copy=False)
        self.date = dfr[var_names['col_name']['even_dt']].values. \
            astype(int, copy=False)
        self.event_name = dfr[var_names['col_name']['even_name']].values.tolist()

        dfr = pd.read_excel(file_name, var_names['sheet_name']['inten'])
        # number of events (ignore centroid_ID column)
        # check the number of events is the same as the one in the frequency
        if dfr.shape[1] - 1 is not num_events:
            LOGGER.error('Hazard intensity is given for a number of events '
                         'different from the number of defined in its frequency: '
                         '%s != %s', dfr.shape[1] - 1, num_events)
            raise ValueError
        # check number of centroids is the same as retrieved before
        if dfr.shape[0] is not self.centroids.size:
            LOGGER.error('Hazard intensity is given for a number of centroids '
                         'different from the number of centroids defined: %s != %s',
                         dfr.shape[0], self.centroids.size)
            raise ValueError

        self.intensity = sparse.csr_matrix(dfr.values[:, 1:num_events + 1].transpose())
        self.fraction = sparse.csr_matrix(np.ones(self.intensity.shape,
                                                  dtype=np.float))
