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
from climada.util.files_handler import get_file_names, to_list
from climada.util.dates_times import _datetime64_to_ordinal

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WS'
""" Hazard type acronym for Winter Storm """


class StormEurope(Hazard):
    """Contains european winter storm events.

    Attributes:
        ssi (float): Storm Severity Index, as recorded in the footprint
            files; this is _not_ the same as that computed by the Matlab
            climada version.
            cf. Lamb and Frydendahl (1991)
            "Historic Storms of the North Sea, British Isles and
            Northwest Europe", ISBN: 978-0-521-37522-1
            SSI = v [m/s] ^ 3 * duration [h] * area [km^2 or m^2]
    """
    intensity_thres = 15
    """ intensity threshold for storage in m/s """

    vars_opt = Hazard.vars_opt.union({'ssi'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        self.ssi = np.array([], int)

    def read_footprints(self, path, description=None,
                        ref_raster=None, centroids=None,
                        files_omit='fp_era20c_1990012515_701_0.nc'):
        """Clear instance and read WISC footprints. Read Assumes that all
        footprints have the same coordinates as the first file listed/first
        file in dir.

        Parameters:
            path (str, list(str)): A location in the filesystem. Either a
                path to a single netCDF WISC footprint, or a folder containing
                only footprints, or a globbing pattern to one or more
                footprints.
            description (str, optional): description of the events, defaults to
                'WISC historical hazard set'
            ref_raster (str, optional): Reference netCDF file from which to
                construct a new barebones Centroids instance. Defaults to the
                first file in path.
            centroids (Centroids, optional): A Centroids struct, overriding
                ref_raster
            files_omit (str, list(str), optional): List of files to omit;
                defaults to one duplicate storm present in the WISC set as of
                2018-09-10.
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

        for fn in file_names:
            if any(fo in fn for fo in files_omit):
                LOGGER.info("Omitting file %s", fn)
                continue
            self.append(self._read_one_nc(fn, centroids))

        self.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved, too large to pickle',
            description='WISC historical hazard set.'
        )
        if description is not None:
            self.tag.description = description

    @classmethod
    def _read_one_nc(cls, file_name, centroids):
        """ Read a single WISC footprint. Assumes a time dimension of length
            1. Omits a footprint if another file with the same timestamp has
            already been read.

            Parameters:
                nc (xarray.Dataset): File connection to netcdf
                file_name (str): Absolute or relative path to *.nc
                centroids (Centroids): Centr. instance that matches the
                    coordinates used in the *.nc, only validated by size.
        """
        nc = xr.open_dataset(file_name)

        if centroids.size != (nc.sizes['latitude'] * nc.sizes['longitude']):
            raise ValueError('Number of centroids and grid size don\'t match.')

        # xarray does not penalise repeated assignments, see
        # http://xarray.pydata.org/en/stable/data-structures.html
        stacked = nc.max_wind_gust.stack(
            intensity=('latitude', 'longitude', 'time')
        )
        stacked = stacked.where(stacked > cls.intensity_thres)
        stacked = stacked.fillna(0)

        # fill in values from netCDF
        new_haz = StormEurope()
        new_haz.event_name = [nc.storm_name]
        new_haz.date = np.array([
            _datetime64_to_ordinal(nc.time.data[0])
        ])
        new_haz.intensity = sparse.csr_matrix(stacked)
        new_haz.ssi = np.array([float(nc.ssi)])
        new_haz.time_bounds = np.array(nc.time_bounds)

        # fill in default values
        new_haz.centroids = centroids
        new_haz.units = 'm/s'
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        new_haz.orig = np.array([True])

        nc.close()
        return new_haz

    @staticmethod
    def _centroids_from_nc(file_name):
        """ Construct Centroids from the grid described by 'latitude'
            and 'longitude' variables in a netCDF file.
        """
        nc = xr.open_dataset(file_name)
        lats = nc.latitude.data
        lons = nc.longitude.data
        ct = Centroids()
        ct.coord = np.array([
            np.repeat(lats, len(lons)),
            np.tile(lons, len(lats)),
        ]).T
        ct.id = np.arange(0, len(ct.coord))
        ct.tag.description = 'Centroids constructed from: ' + file_name

        return ct

    def plot_ssi(self):
        """ Ought to plot the SSI versus the xs_freq, which presumably is the
            excess frequency. """
        pass
