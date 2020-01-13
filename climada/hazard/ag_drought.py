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

Define AgriculturalDrought (AD) class.
WORK IN PROGRESS
"""

__all__ = ['AgriculturalDrought']

import logging
import re
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import cartopy
import shapely.geometry


from climada.hazard.base import Hazard
#from climada.hazard.isimip_data import _read_one_nc # example

DFL_CROP = ''

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'AD'
""" Hazard type acronym for Agricultural Drought """


class AgriculturalDrought(Hazard):
    """Contains agricultural drought events.

    Attributes:
        ...
    """

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        #initilize the attributes that are unique to our hazard
        self.crop = DFL_CROP

#    def set_hist_events(self, centroids=None):
#        """
#
#        Parameters:
#            ...: ...
#        """
#        LOGGER.info('Setting up historical events.')
#        self.clear()


    def set_from_single_run(self, file, lonmin=-85, latmin=-180, lonmax=85, \
                            latmax=180, years=np.zeros(2)):
        """ Reads netcdf file and initializes a hazard

        Parameters:
            file (string): path to netcdf file

        Returns:
            hazard
        """

        #determine time period that is covered by the input data
        string = re.search('annual_(.+?)_', file)
        if string:
            years_start = int(string.group(1))

        string = re.search(str(years_start)+'_(.+?).nc', file)
        if string:
            years_end = int(string.group(1))

        nr_years = years_end - years_start

        if years.any() == 0:
            id_bands = np.arange(1, nr_years+2).tolist()
            event_list = [str(n) for n in range(years_start, years_end+1)]
        else:
            id_bands = np.arange(years[0]-years_start-1, nr_years-(years_end-years[1])).tolist()
            event_list = [str(n) for n in range(years[0], years[1]+1)]

        #extract additional information of original file
        data = xr.open_dataset(file, decode_times=False)

        self.set_raster([file], band=id_bands, \
                        geometry=list([shapely.geometry.box(lonmin, latmin, lonmax, latmax)]))
        self.check()
        self.crop = data.crop
        self.event_name = event_list
        self.frequency = np.ones(len(self.event_name))*(1/len(self.event_name))
        self.fraction = np.ones(len(self.event_name))

        return self

    def calc_his_mean(self):
        """ Calculates mean of the given hazard

            Returns:
                historic mean
        """
        #hist_mean = sparse.csr_matrix([np.size(self.intensity,1),1])
        hist_mean = np.mean(self.intensity, 0)

        return hist_mean


    def set_int_to_rel_yield(self, hist_mean):
        """ Sets relativ yield to intensity

            Parameter:
                hazard
                historic mean

            Returns:
                hazard
        """
        hazard_matrix = self.intensity

        for event in range(len(self.event_id)):
            hazard_matrix[event, :] = self.intensity[event]/hist_mean[0, :]

        self.intensity = hazard_matrix

        return self

    def plot_time_series(self, years=np.zeros(2)):
        """ Plots a time series of intensities

            Returns:
        """

        if years.any() == 0:
            event_list = self.event_name
        else:
            event_list = [str(n) for n in range(years[0], years[1]+1)]

        self.centroids.set_meta_to_lat_lon()

        len_lat = abs(self.centroids.lat[0]-self.centroids.lat[-1])*(2.5/13.5)
        len_lon = abs(self.centroids.lon[0]-self.centroids.lon[-1])*(5/26)

        nr_subplots = len(event_list)

        if len_lon >= len_lat:
            #colums = int(np.floor(nr_subplots/((len_lon/len_lat)+1)))
            colums = int(np.floor(np.sqrt(nr_subplots/(len_lon/len_lat))))
            rows = int(np.ceil(nr_subplots/colums))
        else:
            rows = int(np.floor(np.sqrt(nr_subplots/(len_lat/len_lon))))
            colums = int(np.ceil(nr_subplots/colums))

        fig, axes = plt.subplots(rows, colums, sharex=True, sharey=True, \
                                 figsize=(colums*len_lon, rows*len_lat), \
                                 subplot_kw=dict(projection=cartopy.crs.PlateCarree()))

        colum = 0
        row = 0

        for year in range(nr_subplots):
            axes.flat[year-1].set_extent([np.min(self.centroids.lon), np.max(self.centroids.lon), \
                      np.min(self.centroids.lat), np.max(self.centroids.lat)])

            if rows == 1:
                self.plot_intensity(event=event_list[year], axis=axes[colum], \
                                    cmap='RdBu', vmin=0, vmax=2)
            elif colums == 1:
                self.plot_intensity(event=event_list[year], axis=axes[row], \
                                    cmap='RdBu', vmin=0, vmax=2)
            else:
                self.plot_intensity(event=event_list[year], axis=axes[row, colum], \
                                    cmap='RdBu', vmin=0, vmax=2)

            if colum <= colums-2:
                colum = colum + 1
            else:
                colum = 0
                row = row + 1

        return fig
