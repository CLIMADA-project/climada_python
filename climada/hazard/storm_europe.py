"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

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

        for fn in file_names:
            if any(fo in fn for fo in files_omit):
                LOGGER.info("Omitting file %s", fn)
                continue
            new_haz = self._read_one_nc(fn, centroids)
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
        lats = ncdf.latitude.data
        lons = ncdf.longitude.data
        cent = Centroids()
        cent.coord = np.array([
            np.repeat(lats, len(lons)),
            np.tile(lons, len(lats)),
        ]).T
        cent.id = np.arange(0, len(cent.coord))
        cent.resolution = (float(ncdf.geospatial_lat_resolution),
                           float(ncdf.geospatial_lon_resolution))
        cent.tag.description = 'Centroids constructed from: ' + file_name
        ncdf.close()

        cent.set_area_per_centroid()
        cent.set_on_land()

        return cent

    def plot_ssi(self):
        """Ought to plot the SSI versus the xs_freq, which presumably is the
        excess frequency.
        """
        pass

    def set_ssi_dawkins(self, on_land=True):
        """ Calculate the SSI according to Dawkins, the definition used matches
        the MATLAB version. Threshold value must be determined _before_ call to
        self.read_footprints()
        ssi = sum_i(area_cell_i * intensity_cell_i^3)

        Parameters:
            on_land (bool): Only calculate the SSI for areas on land,
                ignoring the intensities at sea. Defaults to true, whereas
                the MATLAB version did not.

        Attributes:
            self.ssi_dawkins (np.array): SSI per event
        """
        if on_land is True:
            area_c = self.centroids.area_per_centroid \
                * self.centroids.on_land
        else:
            area_c = self.centroids.area_per_centroid

        self.ssi_dawkins = np.zeros(self.intensity.shape[0])

        for i, inten_i in enumerate(self.intensity):
            ssi = area_c * inten_i.power(3).todense().T
            # crossproduct due to transposition
            self.ssi_dawkins[i] = ssi.item(0)

    def set_ssi_wisc_gust(self):
        """Calculate the SSI according to the WISC definition found at
        wisc.climate.copernicus.eu/wisc/#/help/products#tier1_section
        ssi = sum(area_on_land) * mean(intensity > threshold)^3
        Note that this does not reproduce self.ssi_wisc, presumably because the
        footprint only contains the maximum wind gusts instead of the sustained
        wind speeds over the 72 hour window.

        Attributes:
            self.ssi_wisc_gust (np.array): SSI per event
        """
        cent = self.centroids

        self.ssi_wisc_gust = np.zeros(self.intensity.shape[0])

        area = sum(cent.area_per_centroid * cent.on_land)

        for i, inten_i in enumerate(self.intensity[:, cent.on_land]):
            inten_mean = np.mean(inten_i)
            self.ssi_wisc_gust[i] = area * \
                np.power(inten_mean, 3)
