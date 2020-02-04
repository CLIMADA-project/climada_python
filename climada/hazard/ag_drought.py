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
from scipy import sparse
import scipy.stats


from climada.hazard.base import Hazard
from climada.util import dates_times as dt

DFL_CROP = ''
INT_DEF = 'Yearly Yield'

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'AD'
""" Hazard type acronym for Agricultural Drought """


class AgriculturalDrought(Hazard):
    """Contains agricultural drought events.

    Attributes:
        crop (str): crop type (e.g. wheat)
        intensity_def (str): intensity defined as the Yearly Yield / Relative Yield / Percentile
    """

    def __init__(self, pool=None):
        """Empty constructor. """
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        self.crop = DFL_CROP
        self.intensity_def = INT_DEF

#    def set_hist_events(self, centroids=None):
#        """
#
#        Parameters:
#            ...: ...
#        """
#        LOGGER.info('Setting up historical events.')
#        self.clear()


    def set_from_single_run(self, file_path=None, lonmin=-85, latmin=-180, lonmax=85, \
                            latmax=180, years_user=None):
        """ Reads netcdf file and initializes a hazard

        Parameters:
            file_path (str): path to netcdf file
            lonmin, latin, lonmax, latmax (int, optional) : bounding box to extract
            years_user (array, optional) : start and end year specified by the user

        Returns:
            hazard
        """

        if file_path is None:
            LOGGER.error('No drough-file-path set')
            raise NameError

        #determine time period that is covered by the input data
        years_file = np.zeros(2)
        string = re.search('annual_(.+?)_', file_path)
        if string:
            years_file[0] = int(string.group(1))

        string = re.search(str(int(years_file[0]))+'_(.+?).nc', file_path)
        if string:
            years_file[1] = int(string.group(1))

        if years_user is None:
            id_bands = np.arange(1, years_file[1] - years_file[0]+2).tolist()
            event_list = [str(n) for n in range(int(years_file[0]), int(years_file[1]+1))]
        else:
            id_bands = np.arange(years_user[0]-years_file[0]-1, \
                                 years_user[1] - years_file[0]).tolist()
            event_list = [str(n) for n in range(int(years_user[0]), int(years_user[1]+1))]

        date = [event_list[n]+'-01-01' for n in range(len(event_list))]

        #extract additional information of original file
        data = xr.open_dataset(file_path, decode_times=False)

        self.set_raster([file_path], band=id_bands, \
                        geometry=list([shapely.geometry.box(lonmin, latmin, lonmax, latmax)]))
        self.check()
        self.crop = data.crop
        self.event_name = event_list
        self.frequency = np.ones(len(self.event_name))*(1/len(self.event_name))
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)
        self.units = 't / y'
        self.date = np.array(dt.str_to_date(date))

        return self

    def calc_mean(self):
        """ Calculates mean of the given hazard

            Returns:
                mean(array): contains mean value over the given time period for every centroid
        """
        hist_mean = np.mean(self.intensity, 0)
        #hist_mean[hist_mean == 0] = np.nan

        return hist_mean


    def set_rel_yield_to_int(self, hist_mean):
        """ Sets relative yield to intensity (yearly yield / historic mean) per centroid

            Parameter:
                historic mean (array): historic mean per centroid

            Returns:
                hazard with modified intensity
        """

        hazard_matrix = np.empty(self.intensity.shape)
        hazard_matrix[:, :] = np.nan
        idx = np.where(hist_mean != 0)[1]

        for event in range(len(self.event_id)):
            hazard_matrix[event, idx] = self.intensity[event, idx]/hist_mean[0, idx]

        self.intensity = sparse.csr_matrix(hazard_matrix)
        self.intensity_def = 'Relative Yield'
        self.units = ''

        return self

    def set_percentile_to_int(self, reference_intensity=None):
        """ Sets percentile to intensity

            Parameter:
                reference_intensity (AD): intensity to be used as reference (e.g. the historic
                                    intensity can be used in order to be able to directly compare
                                    historic and future projection data)

            Returns:
                hazard with modified intensity
        """
        hazard_matrix = np.zeros(self.intensity.shape)
        if reference_intensity is None:
            reference_intensity = self.intensity

        for centroid in range(self.intensity.shape[1]):
            array = (reference_intensity[:, centroid].toarray()).reshape(\
                    reference_intensity.shape[0])
            for event in range(self.intensity.shape[0]):
                value = self.intensity[event, centroid]
                hazard_matrix[event, centroid] = (scipy.stats.percentileofscore(array, value))/100

        self.intensity = sparse.csr_matrix(hazard_matrix)
        self.intensity_def = 'Percentile'
        self.units = ''

        return self

    def plot_intensity_agd(self, event, dif=0, axis=None, **kwargs):
        """ Plots intensity with predefined settings depending on the intensity definition

        Parameters:
            event (int or str): event_id or event_name
            dif (int): variable signilizing whether absolute values or the difference between
                future and historic are plotted (dif=0: his/fut values; dif=1: difference = fut-his)
            axis (geoaxes): axes to plot on
        """
        if dif == 0:
            if self.intensity_def == 'Yearly Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='YlGn', vmin=0, vmax=10, \
                                           **kwargs)
            elif self.intensity_def == 'Relative Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=0, vmax=2, \
                                           **kwargs)
            elif self.intensity_def == 'Percentile':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=0, vmax=1, \
                                           **kwargs)
        elif dif == 1:
            if self.intensity_def == 'Yearly Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=-2, vmax=2, \
                                           **kwargs)
            elif self.intensity_def == 'Relative Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=-0.5, \
                                           vmax=0.5, **kwargs)

        return axes

    def plot_time_series(self, years=None):
        """ Plots a time series of intensities (a series of sub plots)

        Returns:
            figure
        """

        if years is None:
            event_list = self.event_name
        else:
            event_list = [str(n) for n in range(years[0], years[1]+1)]

        self.centroids.set_meta_to_lat_lon()

        len_lat = abs(self.centroids.lat[0]-self.centroids.lat[-1])*(2.5/13.5)
        len_lon = abs(self.centroids.lon[0]-self.centroids.lon[-1])*(5/26)

        nr_subplots = len(event_list)

        if len_lon >= len_lat:
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
            axes.flat[year].set_extent([np.min(self.centroids.lon), np.max(self.centroids.lon), \
                      np.min(self.centroids.lat), np.max(self.centroids.lat)])

            if rows == 1:
                self.plot_intensity_agd(event=event_list[year], axis=axes[colum])
            elif colums == 1:
                self.plot_intensity_agd(event=event_list[year], axis=axes[row])
            else:
                self.plot_intensity_agd(event=event_list[year], axis=axes[row, colum])

            if colum <= colums-2:
                colum = colum + 1
            else:
                colum = 0
                row = row + 1

        return fig

    def plot_comparing_maps(self, his, fut, axes, nr_cli_models=1, model=1):
        """ Plots comparison maps of historic and future data and their difference fut-his

        Parameters:
            his (sparse matrix): historic mean annual yield or mean relative yield
            fut (sparse matrix): future mean annual yield or mean relative yield
            axes (Geoaxes): subplot axes that can be generated with ag_drought_util.setup_subplots
            nr_cli_models (int): number of climate models and respectively nr of rows within
                                    the subplot
            model (int): current model/row to plot

        Returns:
            geoaxes
        """
        dif = fut - his
        self.event_id = 0

        for subplot in range(3):

            if self.intensity_def == 'Yearly Yield':
                self.units = 't / y'
            elif self.intensity_def == 'Relative Yield':
                self.units = ''

            if subplot == 0:
                self.intensity = sparse.csr_matrix(his)
                dif_def = 0
            elif subplot == 1:
                self.intensity = sparse.csr_matrix(fut)
                dif_def = 0
            elif subplot == 2:
                self.intensity = sparse.csr_matrix(dif)
                dif_def = 1


            if nr_cli_models == 1:
                ax1 = self.plot_intensity_agd(event=0, dif=dif_def, axis=axes[subplot])
            else:
                ax1 = self.plot_intensity_agd(event=0, dif=dif_def, axis=axes[model, subplot])

            ax1.set_title('')

        if nr_cli_models == 1:
            cols = ['Historical', 'Future', 'Difference = Future - Historical']
            for ax0, col in zip(axes, cols):
                ax0.set_title(col, size='large')

        return axes
