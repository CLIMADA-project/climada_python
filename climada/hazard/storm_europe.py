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

import os
import shutil
import bz2
import logging
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from datetime import datetime


from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
from climada.hazard.tag import Tag as TagHazard
from climada.util.files_handler import get_file_names, download_file
from climada.util.dates_times import (
    datetime64_to_ordinal,
    last_year,
    first_year,
    date_to_str
)
from climada.util import CONFIG

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WS'
"""Hazard type acronym for Winter Storm"""

N_PROB_EVENTS = 5 * 6
"""Number of events per historic event in probabilistic dataset"""


class StormEurope(Hazard):
    """A hazard set containing european winter storm events. Historic storm
    events can be downloaded at http://wisc.climate.copernicus.eu/

    Attributes:
        ssi_wisc (np.array, float): Storm Severity Index (SSI) as recorded in
            the footprint files; apparently not reproducible from the footprint
            values only.
        ssi (np.array, float): SSI as set by set_ssi; uses the Dawkins
            definition by default.
    """

    intensity_thres = 14.7
    """Intensity threshold for storage in m/s; same as used by WISC SSI
        calculations."""

    vars_opt = Hazard.vars_opt.union({'ssi_wisc', 'ssi', 'ssi_full_area'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self):
        """Calls the Hazard init dunder. Sets unit to 'm/s'."""
        Hazard.__init__(self, HAZ_TYPE)
        self.units = 'm/s'
        self.ssi = np.array([], float)
        self.ssi_wisc = np.array([], float)
        self.ssi_full_area = np.array([], float)

    def read_footprints(self, path, description=None,
                        ref_raster=None, centroids=None,
                        files_omit='fp_era20c_1990012515_701_0.nc',
                        combine_threshold=None):
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
            combine_threshold (int, optional): threshold for combining events
                in number of days. if the difference of the dates (self.date)
                of two events is smaller or equal to this threshold, the two
                events are combined into one.
                Default is None, Advised for WISC is 2
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

        self.event_id = np.arange(1, len(self.event_id) + 1)
        self.frequency = np.divide(
            np.ones_like(self.date),
            (last_year(self.date) - first_year(self.date))
        )

        self.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved by default',
            description='WISC historical hazard set.'
        )
        if description is not None:
            self.tag.description = description

        if combine_threshold is not None:
            LOGGER.info('Combining events with small difference in date.')
            difference_date = np.diff(self.date)
            for event_id_i in self.event_id[
                    np.append(difference_date <= combine_threshold, False)]:
                event_ids = [event_id_i, event_id_i + 1]
                self._combine_events(event_ids)

    def _read_one_nc(self, file_name, centroids):
        """Read a single WISC footprint. Assumes a time dimension of length 1.
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

        # fill in default values
        new_haz.centroids = centroids
        new_haz.event_id = np.array([1])
        new_haz.frequency = np.array([1])
        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        new_haz.orig = np.array([True])

        ncdf.close()
        return new_haz


    def read_icon_grib(self, run_date, event_date=None,
                       model_name='icon-eu-eps', description=None, 
                       delete_raw_data=True):
        """Clear instance and download and read dwd icon weather forecast 
        footprints into it. One event is one full day in UCT. Current setup 
        works for runs starting at 00H and 12H. Otherwise the aggregation is 
        inaccurate, because of the given file structure with 1-hour, 3-hour 
        and 6-hour maxima provided.

        Parameters:
            run_date (datetime): The starting timepoint of the forecast run 
                of the icon model
            event_date (datetime, optional): one day within the forecast
                period, only this day (00H-24H) will be included in the hazard
            model_name (str,optional): select the name of the icon model to
                be downloaded. Must match the url string 
                (see _download_icon_grib for further info)
            description (str, optional): description of the events, defaults
                to 'icon weather forecast'
            delete_raw_data (bool,optional): select if downloaded raw data in 
                .grib.bz2 file format should be stored on the computer or 
                removed
        """
        self.clear()
        # download files, if they don't already exist
        file_names = _download_icon_grib(run_date,model_name=model_name)
       
        # create centroids
        nc_centroids_file = self._get_icon_centroids(model_name)
        self.centroids = self._centroids_from_nc(nc_centroids_file)
        
        # read intensity from files
        for ind_i, file_i in enumerate(file_names):
            gripfile_path_i = os.path.join(file_i[:-4])
            with open(file_i, 'rb') as source, open(gripfile_path_i, 'wb') as dest:
                dest.write(bz2.decompress(source.read()))
            ds_i = xr.open_dataset(gripfile_path_i, engine='cfgrib')
            # os.remove(gripfile_path_i)
            if ind_i == 0:
                stacked = ds_i
            else:
                stacked = xr.concat([stacked,ds_i], 'valid_time')
                
        # create intensity matrix with max for each full day
        if not (run_date.hour == 0 or run_date.hour == 12):
            LOGGER.warn('The event definition is inaccuratly implemented for '
                        'starting times, which are not 00H or 12H.')
        if event_date:
            stacked2 = stacked.sel(valid_time=event_date.strftime('%Y-%m-%d')).groupby('valid_time.day').max()
            considered_dates = np.datetime64(event_date)
        else:
            time_covered_step = stacked['valid_time'].diff('valid_time')
            time_covered_day = time_covered_step.groupby('valid_time.day').sum()
            days_to_consider = time_covered_day > np.timedelta64(18,'h') # forecast run covered at least 18 hours of a day
            stacked2 = stacked.groupby('valid_time.day').max().sel(day=days_to_consider)
            unique_dates = stacked['valid_time'].groupby('valid_time.day').min()
            considered_dates = unique_dates.sel(day=days_to_consider).values
        stacked3 = stacked2.stack(intensity=('day', 'number'))
        stacked3 = stacked3.where(stacked3 > self.intensity_thres)
        stacked3 = stacked3.fillna(0)
        
    
        # fill in values from netCDF
        self.intensity = sparse.csr_matrix(stacked3.gust.T)
        self.event_id = np.arange(stacked3.day.size)+1
    
        # fill in default values
        self.units = 'm/s'
        self.fraction = self.intensity.copy().tocsr()
        self.fraction.data.fill(1)
        self.orig = np.ones_like(self.event_id)*False
        self.orig[(stacked3.number == 1).values] = True
        
        self.date = np.repeat(
            np.array(datetime64_to_ordinal(considered_dates)),
            stacked.number.size
            )
        self.event_name = [date_i + '_ens' + str(ens_i) 
                           for date_i, ens_i in zip(
                                   date_to_str(self.date),
                                   stacked3.number.values
                                   )
                           ]
        self.frequency = np.divide(
                np.ones_like(self.event_id),
                stacked.number.size)
        if not description:
            description = ('icon weather forecast windfield ' +
                           'for run startet at ' +
                           run_date.strftime('%Y%m%d%H'))
            
        self.tag = TagHazard(
                HAZ_TYPE, 'Hazard set not saved, too large to pickle',
                description=description
            )
        self.check()
    

        
        # delete generated .grib2 and .4cc40.idx files
        for ind_i, file_i in enumerate(file_names):
            gripfile_path_i = os.path.join(file_i[:-4])
            os.remove(gripfile_path_i)
            idxfile_path_i = gripfile_path_i + '.4cc40.idx'
            os.remove(idxfile_path_i)

        if delete_raw_data:
            #delete downloaded .bz2 files
            _delete_icon_grib(run_date, model_name=model_name)


    @staticmethod
    def _get_icon_centroids(model_name='icon-eu-eps'):
        """ create centroids based on netcdf files provided by dwd, links 
        found here: 
        https://www.dwd.de/DE/leistungen/opendata/neuigkeiten/opendata_dez2018_02.html
        """
        
        # define url and filename
        url = 'https://opendata.dwd.de/weather/lib/cdo/'
        if model_name=='icon-eu-eps':
            file_name = 'icon_grid_0028_R02B07_N02.nc.bz2'
        elif model_name=='icon-eu':
            file_name = 'icon_grid_0024_R02B06_G.nc.bz2'
        else:
            LOGGER.error(('Creation of centroids for the icon model ' +
                          model_name + 'is not implemented. Please define ' +
                          'the default values in the code first.'))
            raise ValueError
        bz2_pathfile = os.path.join(CONFIG['local_data']['save_dir'],
                                     file_name)
        nc_pathfile = bz2_pathfile[:-4]
        
        # download and unzip file
        if not os.path.exists(nc_pathfile):
            if not os.path.exists(bz2_pathfile):
                try:
                    download_file(url + file_name)
                    shutil.move(file_name, CONFIG['local_data']['save_dir'])
                except ValueError as err:
                    LOGGER.error('Error while downloading %s.', url + file_name)
                    raise err
            with open(bz2_pathfile, 'rb') as source, open(nc_pathfile, 'wb') as dest:
                dest.write(bz2.decompress(source.read()))
            os.remove(bz2_pathfile)
        
        return nc_pathfile


    @staticmethod
    def _centroids_from_nc(file_name):
        """Construct Centroids from the grid described by 'latitude' and
        'longitude' variables in a netCDF file.
        """
        LOGGER.info('Constructing centroids from %s', file_name)
        cent = Centroids()
        ncdf = xr.open_dataset(file_name)
        create_meshgrid = True
        if hasattr(ncdf, 'latitude'):
            lats = ncdf.latitude.data
            lons = ncdf.longitude.data
        elif hasattr(ncdf, 'lat'):
            lats = ncdf.lat.data
            lons = ncdf.lon.data
        elif hasattr(ncdf, 'lat_1'):
            lats = ncdf.lat_1.data
            lons = ncdf.lon_1.data
        elif hasattr(ncdf, 'clat'):
            lats = ncdf.clat.data
            lons = ncdf.clon.data
            if ncdf.clat.attrs['units']=='radian':
                lats = np.rad2deg(lats)
                lons = np.rad2deg(lons)
            create_meshgrid = False
        else:
            raise AttributeError('netcdf file has no field named latitude or '
                                 'other know abrivation for coordinates.')
        ncdf.close()
        
        if create_meshgrid:
            lats, lons = np.array([np.repeat(lats, len(lons)),
                                   np.tile(lons, len(lats))])
        cent = Centroids()
        cent.set_lat_lon(lats, lons)
        cent.set_area_pixel()
        cent.set_on_land()

        return cent

    def _combine_events(self, event_ids):
        """combine the intensities of two events using max and adjust event_id, event_name,
        date etc of the hazard

        the event_ids must be consecutive for the event_name field to behave correctly

        Parameters:
            event_ids (array): two consecutive event ids
        """
        select_event_ids = np.isin(self.event_id, event_ids)
        select_other_events = np.invert(select_event_ids)
        intensity_tmp = self.intensity[select_event_ids, :].max(axis=0)
        self.intensity = self.intensity[select_other_events, :]
        self.intensity = sparse.vstack([self.intensity, sparse.csr_matrix(intensity_tmp)])
        self.event_id = np.append(self.event_id[select_other_events], self.event_id.max() + 1)
        self.date = np.append(self.date[select_other_events],
                              np.round(self.date[select_event_ids].mean()))
        name_2 = self.event_name.pop(np.where(select_event_ids)[0][1])
        name_1 = self.event_name.pop(np.where(select_event_ids)[0][0])
        self.event_name.append(name_1 + '_' + name_2)
        fraction_tmp = self.fraction[select_event_ids, :].max(axis=0)
        self.fraction = self.fraction[select_other_events, :]
        self.fraction = sparse.vstack([self.fraction, sparse.csr_matrix(fraction_tmp)])

        self.frequency = np.append(self.frequency[select_other_events],
                                   self.frequency[select_event_ids].mean())
        self.orig = np.append(self.orig[select_other_events],
                              self.orig[select_event_ids].max())
        if self.ssi_wisc.size > 0:
            self.ssi_wisc = np.append(self.ssi_wisc[select_other_events],
                                      np.nan)
        if self.ssi.size > 0:
            self.ssi = np.append(self.ssi[select_other_events],
                                 np.nan)
        if self.ssi_full_area.size > 0:
            self.ssi_full_area = np.append(self.ssi_full_area[select_other_events],
                                           np.nan)
        self.check()

    def calc_ssi(self, method='dawkins', intensity=None, on_land=True,
                 threshold=None, sel_cen=None):
        """Calculate the SSI, method must either be 'dawkins' or 'wisc_gust'.

        'dawkins', after Dawkins et al. (2016),
        doi:10.5194/nhess-16-1999-2016, matches the MATLAB version.
        ssi = sum_i(area_cell_i * intensity_cell_i^3)

        'wisc_gust', according to the WISC Tier 1 definition found at
        https://wisc.climate.copernicus.eu/wisc/#/help/products#tier1_section
        ssi = sum(area_on_land) * mean(intensity)^3

        In both definitions, only raster cells that are above the threshold are
        used in the computation.
        Note that this method does not reproduce self.ssi_wisc, presumably
        because the footprint only contains the maximum wind gusts instead of
        the sustained wind speeds over the 72 hour window. The deviation may
        also be due to differing definitions of what lies on land (i.e. Syria,
        Russia, Northern Africa and Greenland are exempt).

        Parameters:
            method (str): Either 'dawkins' or 'wisc_gust'
            intensity (scipy.sparse.csr): Intensity matrix; defaults to
                self.intensity
            on_land (bool): Only calculate the SSI for areas on land,
                ignoring the intensities at sea. Defaults to true, whereas
                the MATLAB version did not.
            threshold (float, optional): Intensity threshold used in index
                definition. Cannot be lower than the read-in value.
            sel_cen (np.array, bool): A boolean vector selecting centroids.
                Takes precendence over on_land.

        Attributes:
            self.ssi_dawkins (np.array): SSI per event
        """
        if intensity is not None:
            if not isinstance(intensity, sparse.csr_matrix):
                intensity = sparse.csr_matrix(intensity)
            else:
                pass
        else:
            intensity = self.intensity

        if threshold is not None:
            assert threshold >= self.intensity_thres, \
                'threshold cannot be below threshold upon read_footprint'
            intensity = intensity.multiply(intensity > threshold)
        else:
            intensity = intensity.multiply(intensity > self.intensity_thres)

        cent = self.centroids

        if sel_cen is not None:
            pass
        elif on_land is True:
            sel_cen = cent.on_land
        else:  # select all centroids
            sel_cen = np.ones_like(cent.area_pixel, dtype=bool)

        ssi = np.zeros(intensity.shape[0])

        if method == 'dawkins':
            area_c = cent.area_pixel / 1000 / 1000 * sel_cen
            for i, inten_i in enumerate(intensity):
                ssi_i = area_c * inten_i.power(3).todense().T
                # matrix crossproduct (row x column vector)
                ssi[i] = ssi_i.item(0)

        elif method == 'wisc_gust':
            for i, inten_i in enumerate(intensity[:, sel_cen]):
                area = np.sum(cent.area_pixel[inten_i.indices]) / 1000 / 1000
                inten_mean = np.mean(inten_i)
                ssi[i] = area * np.power(inten_mean, 3)

        return ssi

    def set_ssi(self, **kwargs):
        """Wrapper around calc_ssi for setting the self.ssi attribute.

        Parameters:
            **kwargs: passed on to calc_ssi

        Attributes:
            ssi (np.array): SSI per event
        """
        self.ssi = self.calc_ssi(**kwargs)

    def plot_ssi(self, full_area=False):
        """Plot the distribution of SSIs versus their cumulative exceedance
            frequencies, highlighting historical storms in red.

        Returns:
            fig (matplotlib.figure.Figure)
            ax (matplotlib.axes._subplots.AxesSubplot)
        """
        if full_area:
            ssi = self.ssi_full_area
        else:
            ssi = self.ssi

        # data wrangling
        ssi_freq = pd.DataFrame({
            'ssi': ssi,
            'freq': self.frequency,
            'orig': self.orig,
        })
        ssi_freq = ssi_freq.sort_values('ssi', ascending=False)
        ssi_freq['freq_cum'] = np.cumsum(ssi_freq.freq)

        ssi_hist = ssi_freq.loc[ssi_freq.orig].copy()
        ssi_hist.freq = ssi_hist.freq * self.orig.size / self.orig.sum()
        ssi_hist['freq_cum'] = np.cumsum(ssi_hist.freq)

        # plotting
        fig, axs = plt.subplots()
        axs.plot(ssi_freq.freq_cum, ssi_freq.ssi, label='All Events')
        axs.scatter(ssi_hist.freq_cum, ssi_hist.ssi,
                    color='red', label='Historic Events')
        axs.legend()
        axs.set_xlabel('Exceedance Frequency [1/a]')
        axs.set_ylabel('Storm Severity Index')
        axs.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        plt.show()

        return fig, axs

    def generate_prob_storms(self, reg_id=528, spatial_shift=4, ssi_args={},
                             **kwargs):
        """Generates a new hazard set with one original and 29 probabilistic
        storms per historic storm. This represents a partial implementation of
        the Monte-Carlo method described in section 2.2 of Schwierz et al.
        (2010), doi:10.1007/s10584-009-9712-1.
        It omits the rotation of the storm footprints, as well as the pseudo-
        random alterations to the intensity.

        In a first step, the original intensity and five additional intensities
        are saved to an array. In a second step, those 6 possible intensity
        levels are shifted by n raster pixels into each direction (N/S/E/W).

        Caveats:
            - Memory safety is an issue; trial with the entire dataset resulted
              in 60GB of swap memory being used...
            - Can only use numeric region_id for country selection
            - Drops event names as provided by WISC

        Parameters:
            region_id (int, list of ints, or None): iso_n3 code of the
                countries we want the generated hazard set to be returned for.
            spatial_shift (int): amount of raster pixels to shift by
            ssi_args (dict): A dictionary of arguments passed to calc_ssi
            **kwargs: keyword arguments passed on to self._hist2prob()

        Returns:
            new_haz (StormEurope): A new hazard set for the given country.
                Centroid attributes are preserved. self.orig attribute is set
                to True for original storms (event_id ending in 00). Also
                contains a ssi_prob attribute,
        """
        # bool vector selecting the targeted centroids
        if reg_id is not None:
            if self.centroids.region_id.size == 0:
                self.centroids.set_region_id()
            if not isinstance(reg_id, list):
                reg_id = [reg_id]
            sel_cen = np.isin(self.centroids.region_id, reg_id)

        else:  # shifting truncates valid centroids
            sel_cen = np.zeros(self.centroids.shape, bool)
            sel_cen[
                spatial_shift:-spatial_shift,
                spatial_shift:-spatial_shift
            ] = True
            sel_cen = sel_cen.reshape(self.centroids.size)

        # init probabilistic array
        n_out = N_PROB_EVENTS * self.size
        intensity_prob = sparse.lil_matrix((n_out, np.count_nonzero(sel_cen)))
        ssi = np.zeros(n_out)

        LOGGER.info('Commencing probabilistic calculations')
        for index, intensity1d in enumerate(self.intensity):
            # indices for return matrix
            start = index * N_PROB_EVENTS
            end = (index + 1) * N_PROB_EVENTS

            intensity_prob[start:end, :], ssi[start:end] =\
                self._hist2prob(
                    intensity1d,
                    sel_cen,
                    spatial_shift,
                    ssi_args,
                    **kwargs)

        LOGGER.info('Generating new StormEurope instance')
        new_haz = StormEurope()
        new_haz.intensity = sparse.csr_matrix(intensity_prob)
        new_haz.ssi_full_area = ssi

        # don't use synthetic dates; just repeat the historic dates
        new_haz.date = np.repeat(self.date, N_PROB_EVENTS)

        # subsetting centroids
        new_haz.centroids = self.centroids.select(sel_cen=sel_cen)

        # construct new event ids
        base = np.repeat((self.event_id * 100), N_PROB_EVENTS)
        synth_id = np.tile(np.arange(N_PROB_EVENTS), self.size)
        new_haz.event_id = base + synth_id

        # frequency still based on the historic number of years
        new_haz.frequency = np.divide(np.repeat(self.frequency, N_PROB_EVENTS),
                                      N_PROB_EVENTS)

        new_haz.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved by default',
            description='WISC probabilistic hazard set according to Schwierz et al.'
        )

        new_haz.fraction = new_haz.intensity.copy().tocsr()
        new_haz.fraction.data.fill(1)
        new_haz.orig = (new_haz.event_id % 100 == 0)

        new_haz.check()

        return new_haz

    def _hist2prob(self, intensity1d, sel_cen, spatial_shift, ssi_args={},
                   power=1.15, scale=0.0225):
        """Internal function, intended to be called from generate_prob_storms.
        Generates six permutations based on one historical storm event, which
        it then moves around by spatial_shift gridpoints to the east, west, and
        north.

        Parameters
        ----------
        intensity1d : scipy.sparse.csr_matrix, 1 by n
            One historic event
        sel_cen : np.ndarray(dtype=bool)
            which centroids to return
        spatial_shift : int
            amount of raster cells to shift by
        power : float
            power to be applied elementwise
        scale : float
            weight of probabilistic component
        ssi_args : dict
            named arguments passed on to calc_ssi

        Returns
        -------
        intensity : np.array
            Synthetic intensities of shape (N_PROB_EVENTS, length(sel_cen))
        ssi : np.array
            SSI per synthetic event according to provided method.
        """
        shape_ndarray = tuple([N_PROB_EVENTS]) + self.centroids.shape

        # reshape to the raster that the data represents
        intensity2d = intensity1d.reshape(self.centroids.shape)

        # scipy.sparse.csr.csr_matrix elementwise methods (to avoid this:
        # https://github.com/ContinuumIO/anaconda-issues/issues/9129 )
        intensity2d_sqrt = intensity2d.power(1.0 / power).todense()
        intensity2d_pwr = intensity2d.power(power).todense()
        intensity2d = intensity2d.todense()

        # intermediary 3d array: (lat, lon, events)
        intensity3d_prob = np.ndarray(shape_ndarray)

        # the six variants of intensity transformation
        # 1. translation only
        intensity3d_prob[0] = intensity2d

        # 2. and 3. plusminus scaled sqrt
        intensity3d_prob[1] = intensity2d - (scale * intensity2d_sqrt)
        intensity3d_prob[2] = intensity2d + (scale * intensity2d_sqrt)

        # 4. and 5. plusminus scaled power
        intensity3d_prob[3] = intensity2d - (scale * intensity2d_pwr)
        intensity3d_prob[4] = intensity2d + (scale * intensity2d_pwr)

        # 6. minus scaled sqrt and pwr
        intensity3d_prob[5] = (intensity2d
                               - (0.5 * scale * intensity2d_pwr)
                               - (0.5 * scale * intensity2d_sqrt))

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

        intensity_out = intensity3d_prob.reshape(
            N_PROB_EVENTS,
            np.prod(self.centroids.shape)
        )

        ssi = self.calc_ssi(intensity=intensity_out, **ssi_args)

        return intensity_out[:, sel_cen], ssi




def _download_icon_grib(run_date,model_name='icon-eu-eps',
                        parameter_name='vmax_10m',
                        max_lead_time=None):
    """download the gribfiles of a weather forecast run for a certain 
    weather parameter from opendata.dwd.de/weather/nwp/. 
    
    Parameters:
        run_date (datetime): The starting timepoint of the forecast run 
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        parameter_name (str): the name of the meteorological parameter 
            written as it appears in the folder structure in 
            opendata.dwd.de/weather/nwp/
        max_lead_time (int): number of hours for which files should be 
            downloaded, will default to maximum available data
    
    Returns:
        file_names (list): a list of filenames that link to all just 
            downloaded or available files from the forecast run, defined by
            the input parameters
    """
    
    LOGGER.info(('Downloading icon grib files of model ' +
                 model_name + ' for parameter ' + parameter_name +
                 ' with starting date ' + run_date.strftime('%Y%m%d%H') +
                 '.'))
    
    url, file_name, lead_times = _create_icon_grib_name(run_date,
                                                        model_name,
                                                        parameter_name,
                                                        max_lead_time)
        
    #download all files
    file_names = []
    for lead_i in lead_times:
        file_name_i = file_name.format(lead_i=lead_i)
        full_path_name_i = os.path.join(CONFIG['local_data']['save_dir'],
                                        file_name_i)
        file_names.append(full_path_name_i)
        if os.path.exists(full_path_name_i):
            continue
        try:
            download_file(url + file_name_i)
            shutil.move(file_name_i, CONFIG['local_data']['save_dir'])
        except ValueError as err:
            LOGGER.error('Error while downloading %s.', file_name_i)
            raise err
        
    return file_names
        
    
 
def _delete_icon_grib(run_date,model_name='icon-eu-eps',
                        parameter_name='vmax_10m',
                        max_lead_time=None):
    """delete the downloaded gribfiles of a weather forecast run for a 
    certain weather parameter from opendata.dwd.de/weather/nwp/. 
    
    Parameters:
        run_date (datetime): The starting timepoint of the forecast run 
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        parameter_name (str): the name of the meteorological parameter 
            written as it appears in the folder structure in 
            opendata.dwd.de/weather/nwp/
        max_lead_time (int): number of hours for which files should be 
            deleted, will default to maximum available data
    """    
    
    url, file_name, lead_times = _create_icon_grib_name(run_date,
                                                        model_name,
                                                        parameter_name,
                                                        max_lead_time)
    #delete all files
    for lead_i in lead_times:
        file_name_i = file_name.format(lead_i=lead_i)
        full_path_name_i = os.path.join(CONFIG['local_data']['save_dir'],
                                        file_name_i)
        if os.path.exists(full_path_name_i):
          os.remove(full_path_name_i)
        else:
          LOGGER.warning('File %s does not exist and could not be deleted.',
                         full_path_name_i)


def _create_icon_grib_name(run_date,model_name='icon-eu-eps',
                        parameter_name='vmax_10m',
                        max_lead_time=None):
    """create all parameters to download or delete gribfiles of a weather 
    forecast run for a certain weather parameter from 
    opendata.dwd.de/weather/nwp/. 
    
    Parameters:
        run_date (datetime): The starting timepoint of the forecast run 
        model_name (str): the name of the forecast model written as it appears
            in the folder structure in opendata.dwd.de/weather/nwp/
        parameter_name (str): the name of the meteorological parameter 
            written as it appears in the folder structure in 
            opendata.dwd.de/weather/nwp/
        max_lead_time (int): number of hours for which files should be 
            selected, will default to maximum available data
    """    
    # define defaults of the url for each model and parameter combination
    if (model_name=='icon-eu-eps') & (parameter_name=='vmax_10m'):
        file_extension = '_europe_icosahedral_single-level_'
        max_lead_time_default = 120 # maximum available data
        lead_times = np.concatenate((np.arange(1,49),
                                     np.arange(51,73,3),
                                     np.arange(78,121,6)
                                     ))
    else:
        LOGGER.error(('Download for model ' + model_name + 
                      ' and parameter ' + parameter_name + 
                      ' is not yet implemented. Please define ' +
                      'the default values in the code first.'))
        raise ValueError
        
    # create the url for download
    url = ('https://opendata.dwd.de/weather/nwp/' +
           model_name +
           '/grib/' +
           run_date.strftime('%H') + 
           '/' + 
           parameter_name +
           '/')
    file_name = (model_name +
                 file_extension +
                 run_date.strftime('%Y%m%d%H') +
                 '_' +
                 '{lead_i:03}' +
                 '_' +
                 parameter_name +
                 '.grib2.bz2')
    
    
    # define the leadtimes
    if  not max_lead_time:
        max_lead_time = max_lead_time_default
    elif max_lead_time > max_lead_time_default:
        LOGGER.warning(('Parameter max_lead_time ' +
                        str(max_lead_time) + ' is bigger than maximum ' +
                        'available files. max_lead_time is adjusted to ' +
                        str(max_lead_time_default)))
        max_lead_time = max_lead_time_default
    lead_times = lead_times[lead_times<=max_lead_time]
    
    return url, file_name, lead_times