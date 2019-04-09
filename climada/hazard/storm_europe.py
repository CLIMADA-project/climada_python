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

Define StormEurope class.
"""

__all__ = ['StormEurope']

import logging
import numpy as np
import xarray as xr
from scipy import sparse

from climada.hazard.base import Hazard
from climada.hazard.centroids.base import Centroids
from climada.hazard.tag import Tag as TagHazard
from climada.util.files_handler import get_file_names
from climada.util.dates_times import datetime64_to_ordinal, last_year, \
    first_year

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WS'
""" Hazard type acronym for Winter Storm """


class StormEurope(Hazard):
    """Contains european winter storm events. Historic storm events can be
    downloaded at http://wisc.climate.copernicus.eu/

    Attributes:
        ssi_wisc (np.array, float): Storm Severity Index as recorded in the
            footprint files; this is _not_ the same as that computed by the
            MATLAB climada version. Apparently not reproducible from the
            max_wind_gust values only.
        ssi_dawkins (np.array, float): Storm Severity Index as defined in
            Dawkins, 2016, doi:10.5194/nhess-16-1999-2016
            Can be set using self.set_ssi_dawkins()
        ssi_wisc_gust (np.array): SSI according to the WISC definition,
            calculated using only gust values. See self.set_ssi_wisc_gust()
    """
    intensity_thres = 14.7
    """ intensity threshold for storage in m/s; same as in WISC """

    vars_opt = Hazard.vars_opt.union({'ssi_wisc', 'ssi_dawkins'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.ssi_wisc = np.array([], float)

    def read_footprints(self, path, description=None,
                        ref_raster=None, centroids=None,
                        files_omit='fp_era20c_1990012515_701_0.nc'):
        """Clear instance and read WISC footprints into it. Read Assumes that
        all footprints have the same coordinates as the first file listed/first
        file in dir.

        Parameters:
            path (str, list(str)): A location in the filesystem. Either a
                path to a single netCDF WISC footprint, or a folder
                containing only footprints, or a globbing pattern to one or
                more footprints.
            description (str, optional): description of the events, defaults
                to 'WISC historical hazard set'
            ref_raster (str, optional): Reference netCDF file from which to
                construct a new barebones Centroids instance. Defaults to
                the first file in path.
            centroids (Centroids, optional): A Centroids struct, overriding
                ref_raster
            files_omit (str, list(str), optional): List of files to omit;
                defaults to one duplicate storm present in the WISC set as
                of 2018-09-10.
        """

        self.clear()

        file_names = get_file_names(path)

        if ref_raster is not None and centroids is not None:
            LOGGER.warning('Overriding ref_raster with centroids')

        if centroids is not None:
            pass
        elif ref_raster is not None:
            centroids = self._centroids_from_nc(ref_raster)
        elif ref_raster is None:
            centroids = self._centroids_from_nc(file_names[0])

        if isinstance(files_omit, str):
            files_omit = [files_omit]

        LOGGER.info('Commencing to iterate over netCDF files.')

        for file_name in file_names:
            if any(fo in file_name for fo in files_omit):
                LOGGER.info("Omitting file %s", file_name)
                continue
            new_haz = self._read_one_nc(file_name, centroids)
            if new_haz is not None:
                self.append(new_haz)

        self.event_id = np.arange(1, len(self.event_id)+1)
        self.frequency = np.divide(
            np.ones_like(self.date),
            (last_year(self.date) - first_year(self.date))
        )

        self.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved, too large to pickle',
            description='WISC historical hazard set.'
        )
        if description is not None:
            self.tag.description = description

    def _read_one_nc(self, file_name, centroids):
        """ Read a single WISC footprint. Assumes a time dimension of length 1.
        Omits a footprint if another file with the same timestamp has already
        been read.

        Parameters:
            file_name (str): Absolute or relative path to *.nc
            centroids (Centroids): Centr. instance that matches the
                coordinates used in the *.nc, only validated by size.

        Returns:
            new_haz (StormEurope): Hazard instance for one single storm.
       """
        ncdf = xr.open_dataset(file_name)

        if centroids.size != (ncdf.sizes['latitude'] * ncdf.sizes['longitude']):
            ncdf.close()
            LOGGER.warning(('Centroids size doesn\'t match NCDF dimensions. '
                            'Omitting file %s.'), file_name)
            return None

        # xarray does not penalise repeated assignments, see
        # http://xarray.pydata.org/en/stable/data-structures.html
        stacked = ncdf.max_wind_gust.stack(
            intensity=('latitude', 'longitude', 'time')
        )
        stacked = stacked.where(stacked > self.intensity_thres)
        stacked = stacked.fillna(0)

        # fill in values from netCDF
        new_haz = StormEurope()
        new_haz.event_name = [ncdf.storm_name]
        new_haz.date = np.array([datetime64_to_ordinal(ncdf.time.data[0])])
        new_haz.intensity = sparse.csr_matrix(stacked)
        new_haz.ssi_wisc = np.array([float(ncdf.ssi)])
        new_haz.time_bounds = np.array(ncdf.time_bounds)

        # fill in default values
        new_haz.centroids = centroids
        new_haz.units = 'm/s'
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        new_haz.orig = np.array([True])

        ncdf.close()
        return new_haz

    @staticmethod
    def _centroids_from_nc(file_name):
        """Construct Centroids from the grid described by 'latitude' and
        'longitude' variables in a netCDF file.
        """
        LOGGER.info('Constructing centroids from %s', file_name)
        ncdf = xr.open_dataset(file_name)
        if hasattr(ncdf, 'latitude'):
            lats = ncdf.latitude.data
            lons = ncdf.longitude.data
            new_coord = np.array([
                    np.repeat(lats, len(lons)),
                    np.tile(lons, len(lats)),
                    ]).T
        elif hasattr(ncdf,'lat'):
            lats = ncdf.lat.data
            lons = ncdf.lon.data
            new_coord = np.array([
                    np.repeat(lats, len(lons)),
                    np.tile(lons, len(lats)),
                    ]).T
        elif hasattr(ncdf,'lat_1'):
            lats = ncdf.lat_1.data
            lons = ncdf.lon_1.data
            new_coord = np.array([
                    lats.reshape(-1),
                    lons.reshape(-1),
                    ]).T
        else:
            raise AttributeError('netcdf file has no field named latitude or other know abrivation for coordinates.') 
            
        cent = Centroids()
        cent.coord = new_coord
        cent.id = np.arange(0, len(cent.coord))
        if hasattr(ncdf,'geospatial_lat_resolution'):
            cent.resolution = (float(ncdf.geospatial_lat_resolution),
                               float(ncdf.geospatial_lon_resolution))
            cent.set_area_per_centroid()
        cent.tag.description = 'Centroids constructed from: ' + file_name
        ncdf.close()

        cent.set_on_land()

        return cent

    def plot_ssi(self):
        """Ought to plot the SSI versus the xs_freq, which presumably is the
        excess frequency.
        """
        raise NotImplementedError

    def set_ssi_dawkins(self, on_land=True, threshold=None):
        """ Calculate the SSI according to Dawkins, the definition used matches
        the MATLAB version. Threshold value must be determined _before_ call to
        self.read_footprints()
        ssi = sum_i(area_cell_i * intensity_cell_i^3)

        Parameters:
            on_land (bool): Only calculate the SSI for areas on land,
                ignoring the intensities at sea. Defaults to true, whereas
                the MATLAB version did not.
            threshold (float, optional): Threshold used in index definition

        Attributes:
            self.ssi_dawkins (np.array): SSI per event
        """
        if on_land is True:
            area_c = self.centroids.area_per_centroid \
                * self.centroids.on_land
        else:
            area_c = self.centroids.area_per_centroid

        if threshold is None:
            intensity = self.intensity
        else:
            assert threshold >= self.intensity_thres, \
                'threshold cannot be below threshold upon read_footprint'
            intensity = self.intensity > threshold

        self.ssi_dawkins = np.zeros(self.intensity.shape[0])

        for i, inten_i in enumerate(intensity):
            ssi = area_c * inten_i.power(3).todense().T
            # crossproduct due to transposition
            self.ssi_dawkins[i] = ssi.item(0)

    def set_ssi_wisc_gust(self, threshold=None):
        """Calculate the SSI according to the WISC definition found at
        wisc.climate.copernicus.eu/wisc/#/help/products#tier1_section
        ssi = sum(area_on_land) * mean(intensity > threshold)^3
        We _assume_ that only the area where the threshold is exceeded
        Note that this does not reproduce self.ssi_wisc, presumably because the
        footprint only contains the maximum wind gusts instead of the sustained
        wind speeds over the 72 hour window.

        Parameters:
            threshold (float, optional): Threshold used in index definition

        Attributes:
            self.ssi_wisc_gust (np.array): SSI per event
        """
        cent = self.centroids

        if threshold is None:
            intensity = self.intensity
        else:
            assert threshold >= self.intensity_thres, \
                'threshold cannot be below threshold upon read_footprint'
            intensity = self.intensity > threshold

        self.ssi_wisc_gust = np.zeros(intensity.shape[0])

        for i, inten_i in enumerate(intensity[:, cent.on_land]):
            area = np.sum(cent.area_per_centroid[inten_i.indices])
            inten_mean = np.mean(inten_i)
            self.ssi_wisc_gust[i] = area * np.power(inten_mean, 3)

    def generate_prob_storms(self, countries=528):
        """ Generate 6 probabilistic storms per historic storm. As per doi:
            10.1007/s10584-009-9712-1

        Caveats of the development version
            - Does not save generated storms to disk; memory may be limiting
            - Can only use country code for country selection
            - Drops event names and ssi as provided by WISC

        The event ids ending on 00 are historic, the next 29 are synthetic, and
        so forth for every historic event

        Parameters:
            country (int): iso_n3 code of the country we want the generated
            hazard set to be returned for.

        Returns:
            new_haz (StormEurope): A new hazard set for the given country.
        """
        if self.centroids.region_id.size == 0:
            self.centroids.set_region_id()
        # bool vector selecting the targeted centroids
        if not isinstance(countries, list):
            countries = [countries]
        select_centroids = np.isin(self.centroids.region_id, countries)

        shape_grid = self.centroids.shape_grid

        # init probabilistic array
        n_events = 6*5
        n_out = n_events * self.size
        intensity_prob = np.ndarray((n_out, np.count_nonzero(select_centroids)))

        LOGGER.info('Commencing probabilistic calculations')
        for index, intensity1d in enumerate(self.intensity):
            # indices for return matrix
            index_start = index * n_events
            index_end = (index + 1) * n_events

            # returned slice is of (n_events, sum(select_centroids))
            intensity_prob[index_start:index_end, :] =\
                self._hist2prob(
                    intensity1d,
                    shape_grid,
                    select_centroids,
                    n_events
                )

        LOGGER.info('Generating new StormEurope instance')
        new_haz = StormEurope()
        new_haz.intensity = sparse.csr_matrix(intensity_prob)

        # don't use synthetic dates; just repeat the historic dates
        new_haz.date = np.repeat(self.date, n_events)

        # subsetting centroids
        new_haz.centroids = self.centroids.select(reg_id=countries)
        new_haz.units = 'm/s'

        # construct new event ids
        base = np.repeat((self.event_id * 100), n_events)
        synth_id = np.tile(np.arange(n_events), self.size)
        new_haz.event_id = base + synth_id

        # frequency still based on the historic number of years
        new_haz.frequency = np.divide(
            np.ones_like(new_haz.event_id),
            (last_year(self.date) - first_year(self.date))
        )

        self.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved automatically',
            description='WISC probabilistic hazard set using Schwierz et al.'
        )

        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        new_haz.orig = (new_haz.event_id % 100 == 0)
        new_haz.check()

        return new_haz

    @staticmethod
    def _hist2prob(intensity1d, shape_grid, select_centroids, n_events,
                   spatial_shift=4, power=1.1):
        """
        Internal function, intended to be called from generate_prob_storms.
        Generates six permutations based on one historical storm event, which
        it then moves around by spatial_shift gridpoints to the east, west, and
        north.

        Parameters:
            intensity1d (scipy.sparse.csr_matrix, 1 by n): One historic event
            shape_grid (tuple): Shape of the original footprint grid
            select_centroids (np.ndarray(dty=bool))
        """
        shape_ndarray = shape_grid + tuple([n_events])

        shape_ndarray = tuple([n_events]) + shape_grid

        # reshape to the raster that the data represents
        intensity2d = intensity1d.reshape(shape_grid)

        # scipy.sparse.csr.csr_matrix elementwise methods (to avoid this:
        # https://github.com/ContinuumIO/anaconda-issues/issues/9129 )
        intensity2d_sqrt = intensity2d.sqrt().todense()
        intensity2d_pwr = intensity2d.power(power).todense()
        intensity2d = intensity2d.todense()

        # intermediary 3d array: (lat, lon, events)
        intensity3d_prob = np.ndarray(shape_ndarray)

        # the six variants of intensity transformation
        # 1. translation only
        intensity3d_prob[0] = intensity2d

        intensity3d_prob[1] = intensity2d - (0.1 * intensity2d_sqrt)
        intensity3d_prob[2] = intensity2d + (0.1 * intensity2d_sqrt)

        intensity3d_prob[3] = intensity2d - (0.1 * intensity2d_pwr)
        intensity3d_prob[4] = intensity2d + (0.1 * intensity2d_pwr)

        intensity3d_prob[5] = intensity2d \
                              - (0.05 * intensity2d_pwr) \
                              - (0.05 * intensity2d_sqrt)

        # spatial shifts
        # northward
        intensity3d_prob[6:12, :-spatial_shift, :] = \
            intensity3d_prob[0:6, spatial_shift:, :]
        # southward
        intensity3d_prob[12:18, spatial_shift:, :] = \
            intensity3d_prob[0:6, :-spatial_shift, :]
        # eastward
        intensity3d_prob[18:24, :, spatial_shift:] = \
            intensity3d_prob[0:6, :, :-spatial_shift]
        # westward
        intensity3d_prob[24:30, :, :-spatial_shift] = \
            intensity3d_prob[0:6, :, spatial_shift:]

        intensity_out = intensity3d_prob.reshape(n_events, np.prod(shape_grid))
        return intensity_out[:, select_centroids]
