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

Define Hazard.
"""

__all__ = ['Hazard']

import copy
import datetime as dt
import itertools
import logging
import pathlib
import warnings

import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathos.pools import ProcessPool as Pool
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy import sparse

from climada.hazard.tag import Tag as TagHazard
from climada.hazard.centroids.centr import Centroids
import climada.util.plot as u_plot
import climada.util.checker as u_check
import climada.util.dates_times as u_dt
from climada import CONFIG
import climada.util.hdf5_handler as u_hdf5
import climada.util.coordinates as u_coord
from climada.util.constants import ONE_LAT_KM
from climada.util.coordinates import NEAREST_NEIGHBOR_THRESHOLD

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
    """
    Contains events of some hazard type defined at centroids. Loads from
    files with format defined in FILE_EXT.

    Attributes
    ----------
    tag : TagHazard
        information about the source
    units : str
        units of the intensity
    centroids : Centroids
        centroids of the events
    event_id : np.array
        id (>0) of each event
    event_name : list(str)
        name of each event (default: event_id)
    date : np.array
        integer date corresponding to the proleptic
        Gregorian ordinal, where January 1 of year 1 has ordinal 1
        (ordinal format of datetime library)
    orig : np.array
        flags indicating historical events (True)
        or probabilistic (False)
    frequency : np.array
        frequency of each event in years
    intensity : sparse.csr_matrix
        intensity of the events at centroids
    fraction : sparse.csr_matrix
        fraction of affected exposures for each
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
        """
        Initialize values.

        Parameters
        ----------
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
        pool : pathos.pool, optional
            Pool that will be used for parallel computation when applicable. Default: None

        Examples
        --------
        Fill hazard values by hand:

        >>> haz = Hazard('TC')
        >>> haz.intensity = sparse.csr_matrix(np.zeros((2, 2)))
        >>> ...

        Take hazard values from file:

        >>> haz = Hazard.from_mat(HAZ_DEMO_MAT, 'demo')

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
        """Reinitialize attributes (except the process Pool)."""
        for (var_name, var_val) in self.__dict__.items():
            if isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(self, var_name, np.array([], dtype=var_val.dtype))
            elif isinstance(var_val, sparse.csr_matrix):
                setattr(self, var_name, sparse.csr_matrix(np.empty((0, 0))))
            elif not isinstance(var_val, Pool):
                setattr(self, var_name, var_val.__class__())

    def check(self):
        """Check dimension of attributes.

        Raises
        ------
        ValueError
        """
        self.centroids.check()
        self._check_events()

    @classmethod
    def from_raster(cls, files_intensity, files_fraction=None, attrs=None,
                    band=None, haz_type=None, pool=None, src_crs=None, window=False,
                    geometry=False, dst_crs=False, transform=None, width=None,
                    height=None, resampling=Resampling.nearest):
        """Create Hazard with intensity and fraction values from raster files

        If raster files are masked, the masked values are set to 0.

        Files can be partially read using either window or geometry. Additionally, the data is
        reprojected when custom dst_crs and/or transform, width and height are specified.

        Parameters
        ----------
        files_intensity : list(str)
            file names containing intensity
        files_fraction : list(str)
            file names containing fraction
        attrs : dict, optional
            name of Hazard attributes and their values
        band : list(int), optional
            bands to read (starting at 1), default [1]
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
            Default: None, which will use the class default ('' for vanilla
            `Hazard` objects, and hard coded in some subclasses)
        pool : pathos.pool, optional
            Pool that will be used for parallel computation when applicable.
            Default: None
        src_crs : crs, optional
            source CRS. Provide it if error without it.
        window : rasterio.windows.Windows, optional
            window where data is
            extracted
        geometry : shapely.geometry, optional
            consider pixels only in shape
        dst_crs : crs, optional
            reproject to given crs
        transform : rasterio.Affine
            affine transformation to apply
        wdith : float, optional
            number of lons for transform
        height : float, optional
            number of lats for transform
        resampling : rasterio.warp.Resampling, optional
            resampling function used for reprojection to dst_crs

        Return
        ------
        Hazard
        """
        if isinstance(files_intensity, (str, pathlib.Path)):
            files_intensity = [files_intensity]
        if isinstance(files_fraction, (str, pathlib.Path)):
            files_fraction = [files_fraction]
        if not attrs:
            attrs = {}
        if not band:
            band = [1]
        if files_fraction is not None and len(files_intensity) != len(files_fraction):
            raise ValueError('Number of intensity files differs from fraction files: %s != %s'
                             % (len(files_intensity), len(files_fraction)))

        if haz_type is not None:
            try:
                haz = cls(haz_type=haz_type, pool=pool)
            except TypeError:
                haz = cls(pool=pool)
                LOGGER.warn("haz_type argument ('%s') is ignored, haz_type is determined by class",
                            haz_type)
        else:
            haz = cls(pool=pool)

        haz.tag.file_name = str(files_intensity) + ' ; ' + str(files_fraction)

        haz.centroids = Centroids.from_raster_file(
            files_intensity[0], src_crs=src_crs, window=window, geometry=geometry, dst_crs=dst_crs,
            transform=transform, width=width, height=height, resampling=resampling)
        if haz.pool:
            chunksize = min(len(files_intensity) // haz.pool.ncpus, 1000)
            inten_list = haz.pool.map(
                haz.centroids.values_from_raster_files,
                [[f] for f in files_intensity],
                itertools.repeat(band), itertools.repeat(src_crs),
                itertools.repeat(window), itertools.repeat(geometry),
                itertools.repeat(dst_crs), itertools.repeat(transform),
                itertools.repeat(width), itertools.repeat(height),
                itertools.repeat(resampling), chunksize=chunksize)
            haz.intensity = sparse.vstack(inten_list, format='csr')
            if files_fraction is not None:
                fract_list = haz.pool.map(
                    haz.centroids.values_from_raster_files,
                    [[f] for f in files_fraction],
                    itertools.repeat(band), itertools.repeat(src_crs),
                    itertools.repeat(window), itertools.repeat(geometry),
                    itertools.repeat(dst_crs), itertools.repeat(transform),
                    itertools.repeat(width), itertools.repeat(height),
                    itertools.repeat(resampling), chunksize=chunksize)
                haz.fraction = sparse.vstack(fract_list, format='csr')
        else:
            haz.intensity = haz.centroids.values_from_raster_files(
                files_intensity, band=band, src_crs=src_crs, window=window, geometry=geometry,
                dst_crs=dst_crs, transform=transform, width=width, height=height,
                resampling=resampling)
            if files_fraction is not None:
                haz.fraction = haz.centroids.values_from_raster_files(
                    files_fraction, band=band, src_crs=src_crs, window=window, geometry=geometry,
                    dst_crs=dst_crs, transform=transform, width=width, height=height,
                    resampling=resampling)

        if files_fraction is None:
            haz.fraction = haz.intensity.copy()
            haz.fraction.data.fill(1)

        if 'event_id' in attrs:
            haz.event_id = attrs['event_id']
        else:
            haz.event_id = np.arange(1, haz.intensity.shape[0] + 1)
        if 'frequency' in attrs:
            haz.frequency = attrs['frequency']
        else:
            haz.frequency = np.ones(haz.event_id.size)
        if 'event_name' in attrs:
            haz.event_name = attrs['event_name']
        else:
            haz.event_name = list(map(str, haz.event_id))
        if 'date' in attrs:
            haz.date = np.array([attrs['date']])
        else:
            haz.date = np.ones(haz.event_id.size)
        if 'orig' in attrs:
            haz.orig = np.array([attrs['orig']])
        else:
            haz.orig = np.ones(haz.event_id.size, bool)
        if 'unit' in attrs:
            haz.unit = attrs['unit']

        return haz

    def set_raster(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_raster."""
        LOGGER.warning("The use of Hazard.set_raster is deprecated."
                       "Use Hazard.from_raster instead.")
        self.__dict__ = Hazard.from_raster(*args, **kwargs).__dict__

    def set_vector(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_vector."""
        LOGGER.warning("The use of Hazard.set_vector is deprecated."
                       "Use Hazard.from_vector instead.")
        self.__dict__ = Hazard.from_vector(*args, **kwargs).__dict__

    @classmethod
    def from_vector(cls, files_intensity, files_fraction=None, attrs=None,
                    inten_name=None, frac_name=None, dst_crs=None, haz_type=None):
        """Read vector files format supported by fiona. Each intensity name is
        considered an event.

        Parameters
        ----------
        files_intensity : list(str)
            file names containing intensity, default: ['intensity']
        files_fraction : (list(str))
            file names containing fraction,
            default: ['fraction']
        attrs : dict, optional
            name of Hazard attributes and their values
        inten_name : list(str), optional
            name of variables containing the intensities of each event
        frac_name : list(str), optional
            name of variables containing
            the fractions of each event
        dst_crs : crs, optional
            reproject to given crs
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
            default: None, which will use the class default ('' for vanilla
            `Hazard` objects, hard coded in some subclasses)

        Returns
        -------
        haz : climada.hazard.Hazard
            Hazard from vector file
        """
        if not attrs:
            attrs = {}
        if not inten_name:
            inten_name = ['intensity']
        if not frac_name:
            inten_name = ['fraction']
        if files_fraction is not None and len(files_intensity) != len(files_fraction):
            raise ValueError('Number of intensity files differs from fraction files: %s != %s'
                             % (len(files_intensity), len(files_fraction)))

        haz = cls() if haz_type is None else cls(haz_type)
        haz.tag.file_name = str(files_intensity) + ' ; ' + str(files_fraction)

        if len(files_intensity) > 0:
            haz.centroids = Centroids.from_vector_file(files_intensity[0], dst_crs=dst_crs)
        elif files_fraction is not None and len(files_fraction) > 0:
            haz.centroids = Centroids.from_vector_file(files_fraction[0], dst_crs=dst_crs)
        else:
            haz.centroids = Centroids()

        haz.intensity = haz.centroids.values_from_vector_files(
            files_intensity, val_names=inten_name, dst_crs=dst_crs)
        if files_fraction is None:
            haz.fraction = haz.intensity.copy()
            haz.fraction.data.fill(1)
        else:
            haz.fraction = haz.centroids.values_from_vector_files(
                files_fraction, val_names=frac_name, dst_crs=dst_crs)

        if 'event_id' in attrs:
            haz.event_id = attrs['event_id']
        else:
            haz.event_id = np.arange(1, haz.intensity.shape[0] + 1)
        if 'frequency' in attrs:
            haz.frequency = attrs['frequency']
        else:
            haz.frequency = np.ones(haz.event_id.size)
        if 'event_name' in attrs:
            haz.event_name = attrs['event_name']
        else:
            haz.event_name = list(map(str, haz.event_id))
        if 'date' in attrs:
            haz.date = np.array([attrs['date']])
        else:
            haz.date = np.ones(haz.event_id.size)
        if 'orig' in attrs:
            haz.orig = np.array([attrs['orig']])
        else:
            haz.orig = np.ones(haz.event_id.size, bool)
        if 'unit' in attrs:
            haz.unit = attrs['unit']

        return haz

    def reproject_raster(self, dst_crs=False, transform=None, width=None, height=None,
                         resampl_inten=Resampling.nearest, resampl_fract=Resampling.nearest):
        """Change current raster data to other CRS and/or transformation

        Parameters
        ----------
        dst_crs: crs, optional
            reproject to given crs
        transform: rasterio.Affine
            affine transformation to apply
        wdith: float
            number of lons for transform
        height: float
            number of lats for transform
        resampl_inten: rasterio.warp,.Resampling optional
            resampling function used for reprojection to dst_crs for intensity
        resampl_fract: rasterio.warp,.Resampling optional
            resampling function used for reprojection to dst_crs for fraction
        """
        if not self.centroids.meta:
            raise ValueError('Raster not set')
        if not dst_crs:
            dst_crs = self.centroids.meta['crs']
        if transform and not width or transform and not height:
            raise ValueError('Provide width and height to given transformation.')
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

        Parameters
        ----------
        dst_crs : crs
            reproject to given crs
        scheduler : str, optional
            used for dask map_partitions. “threads”,
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

        Parameters
        ----------
        scheduler : str, optional
            used for dask map_partitions. “threads”,
            “synchronous” or “processes”
        """
        points_df = gpd.GeoDataFrame()
        points_df['latitude'] = self.centroids.lat
        points_df['longitude'] = self.centroids.lon
        val_names = ['val' + str(i_ev) for i_ev in range(2 * self.size)]
        for i_ev, inten_name in enumerate(val_names):
            if i_ev < self.size:
                points_df[inten_name] = np.asarray(self.intensity[i_ev, :].toarray()).reshape(-1)
            else:
                points_df[inten_name] = np.asarray(self.fraction[i_ev - self.size, :].toarray()). \
                    reshape(-1)
        raster, meta = u_coord.points_to_raster(points_df, val_names,
                                                crs=self.centroids.geometry.crs,
                                                scheduler=scheduler)
        self.intensity = sparse.csr_matrix(raster[:self.size, :, :].reshape(self.size, -1))
        self.fraction = sparse.csr_matrix(raster[self.size:, :, :].reshape(self.size, -1))
        self.centroids = Centroids()
        self.centroids.meta = meta
        self.check()

    def read_mat(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_mat."""
        LOGGER.warning("The use of Hazard.read_mat is deprecated."
                       "Use Hazard.from_mat instead.")
        self.__dict__ = Hazard.from_mat(*args, **kwargs).__dict__

    @classmethod
    def from_mat(cls, file_name, description='', var_names=None):
        """Read climada hazard generate with the MATLAB code in .mat format.

        Parameters
        ----------
        file_name : str
            absolute file name
        description : str, optional
            description of the data
        var_names : dict, optional
            name of the variables in the file,
            default: DEF_VAR_MAT constant

        Returns
        -------
        haz : climada.hazard.Hazard
            Hazard object from the provided MATLAB file

        Raises
        ------
        KeyError
        """
        if not var_names:
            var_names = DEF_VAR_MAT
        LOGGER.info('Reading %s', file_name)
        haz = cls()
        haz.tag.file_name = str(file_name)
        haz.tag.description = description
        try:
            data = u_hdf5.read(file_name)
            try:
                data = data[var_names['field_name']]
            except KeyError:
                pass

            haz_type = u_hdf5.get_string(data[var_names['var_name']['per_id']])
            haz.tag.haz_type = haz_type
            haz.centroids = Centroids.from_mat(file_name, var_names=var_names['var_cent'])
            haz._read_att_mat(data, file_name, var_names)
        except KeyError as var_err:
            raise KeyError("Variable not in MAT file: " + str(var_err)) from var_err
        return haz

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_excel."""
        LOGGER.warning("The use of Hazard.read_excel is deprecated."
                       "Use Hazard.from_excel instead.")
        self.__dict__ = Hazard.from_excel(*args, **kwargs).__dict__

    @classmethod
    def from_excel(cls, file_name, description='', var_names=None, haz_type=None):
        """Read climada hazard generated with the MATLAB code in Excel format.

        Parameters
        ----------
        file_name : str
            absolute file name
        description : str, optional
            description of the data
        var_names (dict, default): name of the variables in the file,
            default: DEF_VAR_EXCEL constant
        haz_type : str, optional
            acronym of the hazard type (e.g. 'TC').
            Default: None, which will use the class default ('' for vanilla `Hazard` objects, and hard coded in some
            subclasses)

        Returns
        -------
        haz : climada.hazard.Hazard
            Hazard object from the provided Excel file

        Raises
        ------
        KeyError
        """
        if not var_names:
            var_names = DEF_VAR_EXCEL
        LOGGER.info('Reading %s', file_name)
        haz = cls() if haz_type is None else cls(haz_type)
        haz.tag.file_name = file_name
        haz.tag.description = description
        try:
            haz.centroids = Centroids.from_excel(file_name, var_names=var_names['col_centroids'])
            haz._read_att_excel(file_name, var_names)
        except KeyError as var_err:
            raise KeyError("Variable not in Excel file: " + str(var_err)) from var_err
        return haz

    def select(self, event_names=None, event_id=None, date=None, orig=None,
               reg_id=None, extent=None, reset_frequency=False):
        """Select events matching provided criteria

        The frequency of events may need to be recomputed (see `reset_frequency`)!

        Parameters
        ----------
        event_names : list of str, optional
            Names of events.
        event_id : list of int, optional
            Id of events. Default is None.
        date : array-like of length 2 containing str or int, optional
            (initial date, final date) in string ISO format ('2011-01-02') or datetime
            ordinal integer.
        orig : bool, optional
            Select only historical (True) or only synthetic (False) events.
        reg_id : int, optional
            Region identifier of the centroids' region_id attibute.
        extent: tuple(float, float, float, float), optional
            Extent of centroids as (min_lon, max_lon, min_lat, max_lat).
            The default is None.
        reset_frequency : bool, optional
            Change frequency of events proportional to difference between first and last
            year (old and new). Default: False.

        Returns
        -------
        haz : Hazard or None
            If no event matching the specified criteria is found, None is returned.
        """
        if type(self) is Hazard:
            haz = Hazard(self.tag.haz_type)
        else:
            haz = self.__class__()

        #filter events
        sel_ev = np.ones(self.event_id.size, dtype=bool)

        # filter events by date
        if date is not None:
            date_ini, date_end = date
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
                LOGGER.info('No hazard with %s original events.', str(orig))
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

        # filter events based on id
        if isinstance(event_id, list):
            # preserves order of event_id
            sel_ev = np.array([
                np.argwhere(self.event_id == n)[0,0]
                for n in event_id
                if n in self.event_id[sel_ev]
                ])

        # filter centroids
        sel_cen = self.centroids.select_mask(reg_id=reg_id, extent=extent)
        if not np.any(sel_cen):
            LOGGER.info('No hazard centroids within extent and region')
            return None

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
                if reg_id is None and extent is None:
                    new_cent = var_val
                else:
                    new_cent = var_val.select(sel_cen=sel_cen)
                setattr(haz, var_name, new_cent)
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

    def select_tight(self, buffer=NEAREST_NEIGHBOR_THRESHOLD/ONE_LAT_KM,
                     val='intensity'):
        """
        Reduce hazard to those centroids spanning a minimal box which
        contains all non-zero intensity or fraction points.

        Parameters
        ----------
        buffer : float, optional
            Buffer of box in the units of the centroids.
            The default is approximately equal to the default threshold
            from the assign_centroids method (works if centroids in
            lat/lon)
        val: string, optional
            Select tight by non-zero 'intensity' or 'fraction'. The
            default is 'intensity'.

        Returns
        -------
        Hazard
            Copy of the Hazard with centroids reduced to minimal box. All other
            hazard properties are carried over without changes.

        See also
        --------
        self.select: Method to select centroids by lat/lon extent
        util.coordinates.assign_coordinates: algorithm to match centroids.

        """

        if val == 'intensity':
            cent_nz = (self.intensity != 0).sum(axis=0).nonzero()[1]
        if val == 'fraction':
            cent_nz = (self.fraction != 0).sum(axis=0).nonzero()[1]
        lon_nz = self.centroids.lon[cent_nz]
        lat_nz = self.centroids.lat[cent_nz]
        ext = u_coord.latlon_bounds(lat=lat_nz, lon=lon_nz, buffer=buffer)
        return self.select(extent=(ext[0], ext[2], ext[1], ext[3]))

    def local_exceedance_inten(self, return_periods=(25, 50, 100, 250)):
        """Compute exceedance intensity map for given return periods.

        Parameters
        ----------
        return_periods : np.array
            return periods to consider

        Returns
        -------
        inten_stats: np.array
        """
        # warn if return period is above return period of rarest event:
        for period in return_periods:
            if period > 1 / self.frequency.min():
                LOGGER.warning('Return period %1.1f exceeds max. event return period.', period)
        LOGGER.info('Computing exceedance intenstiy map for return periods: %s',
                    return_periods)
        num_cen = self.intensity.shape[1]
        inten_stats = np.zeros((len(return_periods), num_cen))
        cen_step = CONFIG.max_matrix_size.int() // self.intensity.shape[0]
        if not cen_step:
            raise ValueError('Increase max_matrix_size configuration parameter to > %s'
                             % str(self.intensity.shape[0]))
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
                          smooth=True, axis=None, figsize=(9, 13), adapt_fontsize=True,
                          **kwargs):
        """Compute and plot hazard exceedance intensity maps for different
        return periods. Calls local_exceedance_inten.

        Parameters
        ----------
        return_periods: tuple(int), optional
            return periods to consider
        smooth: bool, optional
            smooth plot to plot.RESOLUTIONxplot.RESOLUTION
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: tuple, optional
            figure size for plt.subplots
        kwargs: optional
            arguments for pcolormesh matplotlib function used in event plots

        Returns
        -------
        axis, inten_stats:  matplotlib.axes._subplots.AxesSubplot, np.ndarray
            intenstats is return_periods.size x num_centroids
        """
        self._set_coords_centroids()
        inten_stats = self.local_exceedance_inten(np.array(return_periods))
        colbar_name = 'Intensity (' + self.units + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period: ' + str(ret) + ' years')
        axis = u_plot.geo_im_from_array(inten_stats, self.centroids.coord,
                                        colbar_name, title, smooth=smooth,
                                        axes=axis, figsize=figsize, adapt_fontsize=adapt_fontsize, **kwargs)
        return axis, inten_stats

    def plot_intensity(self, event=None, centr=None, smooth=True, axis=None, adapt_fontsize=True,
                       **kwargs):
        """Plot intensity values for a selected event or centroid.

        Parameters
        ----------
        event: int or str, optional
            If event > 0, plot intensities of
            event with id = event. If event = 0, plot maximum intensity in
            each centroid. If event < 0, plot abs(event)-largest event. If
            event is string, plot events with that name.
        centr: int or tuple, optional
            If centr > 0, plot intensity
            of all events at centroid with id = centr. If centr = 0,
            plot maximum intensity of each event. If centr < 0,
            plot abs(centr)-largest centroid where higher intensities
            are reached. If tuple with (lat, lon) plot intensity of nearest
            centroid.
        smooth: bool, optional
            Rescale data to RESOLUTIONxRESOLUTION pixels (see constant
            in module `climada.util.plot`)
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs: optional
            arguments for pcolormesh matplotlib function
            used in event plots or for plot function used in centroids plots

        Returns
        -------
            matplotlib.axes._subplots.AxesSubplot

        Raises
        ------
            ValueError
        """
        self._set_coords_centroids()
        col_label = 'Intensity (%s)' % self.units
        crs_epsg, _ = u_plot.get_transformation(self.centroids.geometry.crs)
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(event, self.intensity, col_label,
                                    smooth, crs_epsg, axis, adapt_fontsize=adapt_fontsize, **kwargs)
        if centr is not None:
            if isinstance(centr, tuple):
                _, _, centr = self.centroids.get_closest_point(centr[0], centr[1])
            return self._centr_plot(centr, self.intensity, col_label, axis, **kwargs)

        raise ValueError("Provide one event id or one centroid id.")

    def plot_fraction(self, event=None, centr=None, smooth=True, axis=None,
                      **kwargs):
        """Plot fraction values for a selected event or centroid.

        Parameters
        ----------
        event: int or str, optional
            If event > 0, plot fraction of event
            with id = event. If event = 0, plot maximum fraction in each
            centroid. If event < 0, plot abs(event)-largest event. If event
            is string, plot events with that name.
        centr: int or tuple, optional
            If centr > 0, plot fraction
            of all events at centroid with id = centr. If centr = 0,
            plot maximum fraction of each event. If centr < 0,
            plot abs(centr)-largest centroid where highest fractions
            are reached. If tuple with (lat, lon) plot fraction of nearest
            centroid.
        smooth: bool, optional
            Rescale data to RESOLUTIONxRESOLUTION pixels (see constant
            in module `climada.util.plot`)
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs: optional
            arguments for pcolormesh matplotlib function
            used in event plots or for plot function used in centroids plots

        Returns
        -------
            matplotlib.axes._subplots.AxesSubplot

        Raises
        ------
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

        raise ValueError("Provide one event id or one centroid id.")

    def sanitize_event_ids(self):
        """Make sure that event ids are unique"""
        if np.unique(self.event_id).size != self.event_id.size:
            LOGGER.debug('Resetting event_id.')
            self.event_id = np.arange(1, self.event_id.size + 1)

    def get_event_id(self, event_name):
        """Get an event id from its name. Several events might have the same
        name.

        Parameters
        ----------
        event_name: str
            Event name

        Returns
        -------
        list_id: np.array(int)
        """
        list_id = self.event_id[[i_name for i_name, val_name in enumerate(self.event_name)
                                 if val_name == event_name]]
        if list_id.size == 0:
            raise ValueError("No event with name: %s" % event_name)
        return list_id

    def get_event_name(self, event_id):
        """Get the name of an event id.

        Parameters
        ----------
        event_id: int
            id of the event

        Returns
        -------
            str

        Raises
        ------
            ValueError
        """
        try:
            return self.event_name[np.argwhere(
                self.event_id == event_id)[0][0]]
        except IndexError as err:
            raise ValueError(f"No event with id: {event_id}") from err

    def get_event_date(self, event=None):
        """Return list of date strings for given event or for all events,
        if no event provided.

        Parameters
        ----------
        event: str or int, optional
            event name or id.

        Returns
        -------
        l_dates: list(str)
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

        Returns
        -------
        orig_yearset: dict
            key are years, values array with event_ids of that year

        """
        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date[self.orig]])
        orig_yearset = {}
        for year in np.unique(orig_year):
            orig_yearset[year] = self.event_id[self.orig][orig_year == year]
        return orig_yearset

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
            elif isinstance(var_val, list) and len(var_val) > 0:
                setattr(self, var_name, [var_val[p] for p in unique_pos])

    def set_frequency(self, yearrange=None):
        """Set hazard frequency from yearrange or intensity matrix.

        Parameters
        ----------
        yearrange: tuple or list, optional
            year range to be used to compute frequency
            per event. If yearrange is not given (None), the year range is
            derived from self.date
        """
        if not yearrange:
            delta_time = dt.datetime.fromordinal(int(np.max(self.date))).year - \
                         dt.datetime.fromordinal(int(np.min(self.date))).year + 1
        else:
            delta_time = max(yearrange) - min(yearrange) + 1
        num_orig = self.orig.nonzero()[0].size
        if num_orig > 0:
            ens_size = self.event_id.size / num_orig
        else:
            ens_size = 1
        self.frequency = np.ones(self.event_id.size) / delta_time / ens_size

    @property
    def size(self):
        """Return number of events."""
        return self.event_id.size

    def write_raster(self, file_name, intensity=True):
        """Write intensity or fraction as GeoTIFF file. Each band is an event

        Parameters
        ----------
        file_name: str
            file name to write in tif format
        intensity: bool
            if True, write intensity, otherwise write fraction
        """
        variable = self.intensity
        if not intensity:
            variable = self.fraction
        if self.centroids.meta:
            u_coord.write_raster(file_name, variable.toarray(), self.centroids.meta)
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
                        all_touched=True, dtype=profile['dtype'], )
                    dst.write(raster.astype(profile['dtype']), i_ev + 1)

    def write_hdf5(self, file_name, todense=False):
        """Write hazard in hdf5 format.

        Parameters
        ----------
        file_name: str
            file name to write, with h5 format
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
            elif isinstance(var_val, list) and var_val and isinstance(var_val[0], str):
                hf_str = hf_data.create_dataset(var_name, (len(var_val),), dtype=str_dt)
                for i_ev, var_ev in enumerate(var_val):
                    hf_str[i_ev] = var_ev
            elif var_val is not None and var_name != 'pool':
                try:
                    hf_data.create_dataset(var_name, data=var_val)
                except TypeError:
                    LOGGER.warning(
                        f"write_hdf5: the class member {var_name} is skipped, due to its "
                        f"type, {var_val.__class__.__name__}, for which writing to hdf5 "
                        "is not implemented. Reading this H5 file will probably lead to "
                        f"{var_name} being set to its default value."
                    )
        hf_data.close()

    def read_hdf5(self, *args, **kwargs):
        """This function is deprecated, use Hazard.from_hdf5."""
        LOGGER.warning("The use of Hazard.read_hdf5 is deprecated."
                       "Use Hazard.from_hdf5 instead.")
        self.__dict__ = Hazard.from_hdf5(*args, **kwargs).__dict__

    @classmethod
    def from_hdf5(cls, file_name):
        """Read hazard in hdf5 format.

        Parameters
        ----------
        file_name: str
            file name to read, with h5 format

        Returns
        -------
        haz : climada.hazard.Hazard
            Hazard object from the provided MATLAB file

        """
        LOGGER.info('Reading %s', file_name)
        haz = cls()
        hf_data = h5py.File(file_name, 'r')
        for (var_name, var_val) in haz.__dict__.items():
            if var_name != 'tag' and var_name not in hf_data.keys():
                continue
            if var_name == 'centroids':
                haz.centroids = Centroids.from_hdf5(hf_data.get(var_name))
            elif var_name == 'tag':
                haz.tag.haz_type = u_hdf5.to_string(hf_data.get('haz_type')[0])
                haz.tag.file_name = u_hdf5.to_string(hf_data.get('file_name')[0])
                haz.tag.description = u_hdf5.to_string(hf_data.get('description')[0])
            elif isinstance(var_val, np.ndarray) and var_val.ndim == 1:
                setattr(haz, var_name, np.array(hf_data.get(var_name)))
            elif isinstance(var_val, sparse.csr_matrix):
                hf_csr = hf_data.get(var_name)
                if isinstance(hf_csr, h5py.Dataset):
                    setattr(haz, var_name, sparse.csr_matrix(hf_csr))
                else:
                    setattr(haz, var_name, sparse.csr_matrix((hf_csr['data'][:],
                                                               hf_csr['indices'][:],
                                                               hf_csr['indptr'][:]),
                                                              hf_csr.attrs['shape']))
            elif isinstance(var_val, str):
                setattr(haz, var_name, u_hdf5.to_string(hf_data.get(var_name)[0]))
            elif isinstance(var_val, list):
                var_value = [x for x in map(u_hdf5.to_string, np.array(hf_data.get(var_name)).tolist())]
                setattr(haz, var_name, var_value)
            else:
                setattr(haz, var_name, hf_data.get(var_name))

        hf_data.close()
        return haz

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

    def _event_plot(self, event_id, mat_var, col_name, smooth, crs_espg, axis=None,
                    figsize=(9, 13), adapt_fontsize=True, **kwargs):
        """Plot an event of the input matrix.

        Parameters
        ----------
        event_id: int or np.array(int)
            If event_id > 0, plot mat_var of
            event with id = event_id. If event_id = 0, plot maximum
            mat_var in each centroid. If event_id < 0, plot
            abs(event_id)-largest event.
        mat_var: sparse matrix
            Sparse matrix where each row is an event
        col_name: sparse matrix
            Colorbar label
        smooth: bool, optional
            smooth plot to plot.RESOLUTIONxplot.RESOLUTION
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: tuple, optional
            figure size for plt.subplots
        kwargs: optional
            arguments for pcolormesh matplotlib function

        Returns
        -------
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
                except IndexError as err:
                    raise ValueError(f'Wrong event id: {ev_id}.') from err
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
                                        l_title, smooth=smooth, axes=axis,
                                        figsize=figsize, proj=crs_espg, adapt_fontsize=adapt_fontsize, **kwargs)

    def _centr_plot(self, centr_idx, mat_var, col_name, axis=None, **kwargs):
        """Plot a centroid of the input matrix.

        Parameters
        ----------
        centr_id: int
            If centr_id > 0, plot mat_var
            of all events at centroid with id = centr_id. If centr_id = 0,
            plot maximum mat_var of each event. If centr_id < 0,
            plot abs(centr_id)-largest centroid where highest mat_var
            are reached.
        mat_var: sparse matrix
            Sparse matrix where each column represents
            a centroid
        col_name: sparse matrix
            Colorbar label
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs: optional
            arguments for plot matplotlib function

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        coord = self.centroids.coord
        if centr_idx > 0:
            try:
                centr_pos = centr_idx
            except IndexError as err:
                raise ValueError(f'Wrong centroid id: {centr_idx}.') from err
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

        Parameters
        ----------
        return_periods: np.array
            return periods to consider
        cen_pos: int
            centroid position

        Returns
        -------
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

        Raises
        ------
            ValueError
        """
        num_ev = len(self.event_id)
        num_cen = self.centroids.size
        if np.unique(self.event_id).size != num_ev:
            raise ValueError("There are events with the same identifier.")

        u_check.check_oligatories(self.__dict__, self.vars_oblig, 'Hazard.',
                                  num_ev, num_ev, num_cen)
        u_check.check_optionals(self.__dict__, self.vars_opt, 'Hazard.', num_ev)
        self.event_name = u_check.array_default(num_ev, self.event_name,
                                                'Hazard.event_name', list(self.event_id))
        self.date = u_check.array_default(num_ev, self.date, 'Hazard.date',
                                          np.ones(self.event_id.shape, dtype=int))
        self.orig = u_check.array_default(num_ev, self.orig, 'Hazard.orig',
                                          np.zeros(self.event_id.shape, dtype=bool))
        if len(self._events_set()) != num_ev:
            raise ValueError("There are events with same date and name.")

    @staticmethod
    def _cen_return_inten(inten, freq, inten_th, return_periods):
        """From ordered intensity and cummulative frequency at centroid, get
        exceedance intensity at input return periods.

        Parameters
        ----------
        inten: np.array
            sorted intensity at centroid
        freq: np.array
            cummulative frequency at centroid
        inten_th: float
            intensity threshold
        return_periods: np.array
            return periods

        Returns
        -------
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
            data[var_names['var_name']['even_id']].astype(int, copy=False))
        try:
            self.units = u_hdf5.get_string(data[var_names['var_name']['unit']])
        except KeyError:
            pass

        n_cen = self.centroids.size
        n_event = len(self.event_id)
        try:
            self.intensity = u_hdf5.get_sparse_csr_mat(
                data[var_names['var_name']['inten']], (n_event, n_cen))
        except ValueError as err:
            raise ValueError('Size missmatch in intensity matrix.') from err
        try:
            self.fraction = u_hdf5.get_sparse_csr_mat(
                data[var_names['var_name']['frac']], (n_event, n_cen))
        except ValueError as err:
            raise ValueError('Size missmatch in fraction matrix.') from err
        except KeyError:
            self.fraction = sparse.csr_matrix(np.ones(self.intensity.shape, dtype=float))
        # Event names: set as event_id if no provided
        try:
            self.event_name = u_hdf5.get_list_str_from_ref(
                file_name, data[var_names['var_name']['ev_name']])
        except KeyError:
            self.event_name = list(self.event_id)
        try:
            comment = u_hdf5.get_string(data[var_names['var_name']['comment']])
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
            raise ValueError('Hazard intensity is given for a number of events '
                             'different from the number of defined in its frequency: '
                             f'{dfr.shape[1] - 1} != {num_events}')
        # check number of centroids is the same as retrieved before
        if dfr.shape[0] is not self.centroids.size:
            raise ValueError('Hazard intensity is given for a number of centroids '
                             'different from the number of centroids defined: '
                             f'{dfr.shape[0]} != {self.centroids.size}')

        self.intensity = sparse.csr_matrix(dfr.values[:, 1:num_events + 1].transpose())
        self.fraction = sparse.csr_matrix(np.ones(self.intensity.shape, dtype=float))

    def append(self, *others):
        """Append the events and centroids to this hazard object.

        All of the given hazards must be of the same type and use the same units as self. The
        centroids of all hazards must have the same CRS.

        The following kinds of object attributes are processed:

        - All centroids are combined together using `Centroids.union`.

        - Lists, 1-dimensional arrays (NumPy) and sparse CSR matrices (SciPy) are concatenated.
        Sparse matrices are concatenated along the first (vertical) axis.

        - All `tag` attributes are appended to `self.tag`.

        For any other type of attribute: A ValueError is raised if an attribute of that name is
        not defined in all of the non-empty hazards at least. However, there is no check that the
        attribute value is identical among the given hazard objects. The initial attribute value of
        `self` will not be modified.

        Note: Each of the hazard's `centroids` attributes might be modified in place in the sense
        that missing properties are added, but existing ones are not overwritten. In case of raster
        centroids, conversion to point centroids is applied so that raster information (meta) is
        lost. For more information, see `Centroids.union`.

        Parameters
        ----------
        others : one or more climada.hazard.Hazard objects
            Hazard instances to append to self

        Raises
        ------
        TypeError, ValueError

        See Also
        --------
        Hazard.concat : concatenate 2 or more hazards
        Centroids.union : combine centroids
        """
        if len(others) == 0:
            return
        haz_list = [self] + list(others)
        haz_list_nonempty = [haz for haz in haz_list if haz.size > 0]

        for haz in haz_list:
            haz._check_events()

        # check type, unit, and attribute consistency among hazards
        haz_types = {haz.tag.haz_type for haz in haz_list if haz.tag.haz_type != ''}
        if len(haz_types) > 1:
            raise ValueError(f"The given hazards are of different types: {haz_types}. "
                             "The hazards are incompatible and cannot be concatenated.")
        self.tag.haz_type = haz_types.pop()

        haz_classes = {type(haz) for haz in haz_list}
        if len(haz_classes) > 1:
            raise TypeError(f"The given hazards are of different classes: {haz_classes}. "
                            "The hazards are incompatible and cannot be concatenated.")

        units = {haz.units for haz in haz_list if haz.units != ''}
        if len(units) > 1:
            raise ValueError(f"The given hazards use different units: {units}. "
                             "The hazards are incompatible and cannot be concatenated.")
        elif len(units) == 0:
            units = {''}
        self.units = units.pop()

        attributes = sorted(set.union(*[set(vars(haz).keys()) for haz in haz_list]))
        for attr_name in attributes:
            if not all(hasattr(haz, attr_name) for haz in haz_list_nonempty):
                raise ValueError(f"Attribute {attr_name} is not shared by all hazards. "
                                 "The hazards are incompatible and cannot be concatenated.")

        # append all tags (to keep track of input files and descriptions)
        for haz in haz_list:
            if haz.tag is not self.tag:
                self.tag.append(haz.tag)

        # map individual centroids objects to union
        centroids = Centroids.union(*[haz.centroids for haz in haz_list])
        hazcent_in_cent_idx_list = [
            u_coord.assign_coordinates(haz.centroids.coord, centroids.coord, threshold=0)
            for haz in haz_list_nonempty
        ]

        # concatenate array and list attributes of non-empty hazards
        for attr_name in attributes:
            attr_val_list = [getattr(haz, attr_name) for haz in haz_list_nonempty]
            if isinstance(attr_val_list[0], sparse.csr.csr_matrix):
                # map sparse matrix onto centroids
                setattr(self, attr_name, sparse.vstack([
                    sparse.csr_matrix(
                        (matrix.data, cent_idx[matrix.indices], matrix.indptr),
                        shape=(matrix.shape[0], centroids.size)
                    )
                    for matrix, cent_idx in zip(attr_val_list, hazcent_in_cent_idx_list)
                ], format='csr'))
            elif isinstance(attr_val_list[0], np.ndarray) and attr_val_list[0].ndim == 1:
                setattr(self, attr_name, np.hstack(attr_val_list))
            elif isinstance(attr_val_list[0], list):
                setattr(self, attr_name, sum(attr_val_list, []))

        self.centroids = centroids
        self.sanitize_event_ids()

    @classmethod
    def concat(cls, haz_list):
        """
        Concatenate events of several hazards of same type.

        This function creates a new hazard of the same class as the first hazard in the given list
        and then applies the `append` method. Please refer to the docs of `Hazard.append` for
        caveats and limitations of the concatenation procedure.

        For centroids, tags, lists, arrays and sparse matrices, the remarks in `Hazard.append`
        apply. All other attributes are copied from the first object in `haz_list`.

        Note that `Hazard.concat` can be used to concatenate hazards of a subclass. The result's
        type will be the subclass. However, calling `concat([])` (with an empty list) is equivalent
        to instantiation without init parameters. So, `Hazard.concat([])` is equivalent to
        `Hazard()`. If `HazardB` is a subclass of `Hazard`, then `HazardB.concat([])` is equivalent
        to `HazardB()` (unless `HazardB` overrides the `concat` method).

        Parameters
        ----------
        haz_list : list of climada.hazard.Hazard objects
            Hazard instances of the same hazard type (subclass).

        Returns
        -------
        haz_concat : instance of climada.hazard.Hazard
            This will be of the same type (subclass) as all the hazards in `haz_list`.

        See Also
        --------
        Hazard.append : append hazards to a hazard in place
        Centroids.union : combine centroids
        """
        if len(haz_list) == 0:
            return cls()
        haz_concat = haz_list[0].__class__()
        haz_concat.tag.haz_type = haz_list[0].tag.haz_type
        for attr_name, attr_val in vars(haz_list[0]).items():
            # to save memory, only copy simple attributes like
            # "units" that are not explicitly handled by Hazard.append
            if not (isinstance(attr_val, (list, np.ndarray, sparse.csr.csr_matrix))
                    or attr_name in ["tag", "centroids"]):
                setattr(haz_concat, attr_name, copy.deepcopy(attr_val))
        haz_concat.append(*haz_list)
        return haz_concat

    def change_centroids(self, centroids, threshold=NEAREST_NEIGHBOR_THRESHOLD):
        """
        Assign (new) centroids to hazard.

        Centoids of the hazard not in centroids are mapped onto the nearest
        point. Fails if a point is further than threshold from the closest
        centroid.

        The centroids must have the same CRS as self.centroids.

        Parameters
        ----------
        haz: Hazard
            Hazard instance
        centroids: Centroids
            Centroids instance on which to map the hazard.
        threshold: int or float
            Threshold (in km) for mapping haz.centroids not in centroids.
            Argument is passed to climada.util.coordinates.assign_coordinates.
            Default: 100 (km)

        Returns
        -------
        haz_new_cent: Hazard
            Hazard projected onto centroids

        Raises
        ------
        ValueError


        See Also
        --------
        util.coordinates.assign_coordinates: algorithm to match centroids.

        """
        # define empty hazard
        haz_new_cent = copy.deepcopy(self)
        haz_new_cent.centroids = centroids

        # indices for mapping matrices onto common centroids
        if centroids.meta:
            new_cent_idx = u_coord.assign_grid_points(
                self.centroids.lon, self.centroids.lat,
                centroids.meta['width'], centroids.meta['height'],
                centroids.meta['transform'])
            if -1 in new_cent_idx:
                raise ValueError("At least one hazard centroid is out of"
                                 "the raster defined by centroids.meta."
                                 " Please choose a larger raster.")
        else:
            new_cent_idx = u_coord.assign_coordinates(
                self.centroids.coord, centroids.coord, threshold=threshold
            )
            if -1 in new_cent_idx:
                raise ValueError("At least one hazard centroid is at a larger "
                                 f"distance than the given threshold {threshold} "
                                 "from the given centroids. Please choose a "
                                 "larger threshold or enlarge the centroids")

        if np.unique(new_cent_idx).size < new_cent_idx.size:
            raise ValueError("At least two hazard centroids are mapped to the same "
                             "centroids. Please make sure that the given centroids "
                             "cover the same area like the original centroids and "
                             "are not of lower resolution.")

        # re-assign attributes intensity and fraction
        for attr_name in ["intensity", "fraction"]:
            matrix = getattr(self, attr_name)
            setattr(haz_new_cent, attr_name,
                    sparse.csr_matrix(
                        (matrix.data, new_cent_idx[matrix.indices], matrix.indptr),
                        shape=(matrix.shape[0], centroids.size)
                    ))

        return haz_new_cent

    @property
    def cent_exp_col(self):
        from climada.entity.exposures import INDICATOR_CENTR
        return INDICATOR_CENTR + self.tag.haz_type

    @property
    def haz_type(self):
        return self.tag.haz_type

    def get_mdr(self, cent_idx, impf):
        """
        Return Mean Damage Ratio (mdr) for chosen centroids (cent_idx)
        for given impact function.

        Parameters
        ----------
        cent_idx : array-like
            array of indices of chosen centroids from hazard
        impf : ImpactFunc
            impact function to compute mdr

        Returns
        -------
        sparse.csr_matrix
            sparse matrix (n_events x len(cent_idx)) with mdr values

        """
        uniq_cent_idx, indices = np.unique(cent_idx, return_inverse=True)      #costs about 30ms for small datasets
        mdr = self.intensity[:, uniq_cent_idx]
        if impf.calc_mdr(0) == 0:
            mdr.data = impf.calc_mdr(mdr.data)
        else:
            LOGGER.warning("Impact function id=%d has mdr(0) != 0."
            "The mean damage ratio must thus be computed for all values of"
            "hazard intensity including 0 which can be very time consuming.",
            impf.id)
            raise NotImplementedError("Not yet implemented.")
        return mdr[:, indices]

    def get_paa(self, cent_idx, impf):
        uniq_cent_idx, indices = np.unique(cent_idx, return_inverse=True)      #costs about 30ms for small datasets
        paa = self.intensity[:, uniq_cent_idx]
        paa.data = np.interp(paa.data, impf.intensity, impf.paa)
        return paa[:, indices]

    def get_fraction(self, cent_idx):
        """
        Return fraction for chosen centroids (cent_idx).

        Parameters
        ----------
        cent_idx : array-like
            array of indices of chosen centroids from hazard

        Returns
        -------
        sparse.csr_matrix
            sparse matrix (n_events x len(cent_idx)) with fraction values

        """
        return self.fraction[:, cent_idx]
