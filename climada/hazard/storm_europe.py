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

Define StormEurope class.
"""

__all__ = ['StormEurope']

import bz2
import datetime as dt
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from climada.util.config import CONFIG
from climada.hazard.base import Hazard
from climada.hazard.centroids.centr import Centroids
from climada.hazard.tag import Tag as TagHazard
from climada.util.files_handler import get_file_names
from climada.util.dates_times import (datetime64_to_ordinal,
                                      last_year,
                                      first_year,
                                      date_to_str
                                      )
from climada.util.dwd_icon_loader import (download_icon_centroids_file,
                                          download_icon_grib,
                                          delete_icon_grib,
                                          )

LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'WS'
"""Hazard type acronym for Winter Storm"""

N_PROB_EVENTS = 5 * 6
"""Number of events per historic event in probabilistic dataset"""


class StormEurope(Hazard):
    """A hazard set containing european winter storm events. Historic storm
    events can be downloaded at http://wisc.climate.copernicus.eu/ and read
    with `from_footprints`. Weather forecasts can be automatically downloaded from
    https://opendata.dwd.de/ and read with from_icon_grib(). Weather forecast
    from the COSMO-Consortium http://www.cosmo-model.org/ can be read with
    from_cosmoe_file().

    Attributes
    ----------
    ssi_wisc : np.array, float
        Storm Severity Index (SSI) as recorded in
        the footprint files; apparently not reproducible from the footprint
        values only.
    ssi : np.array, float
        SSI as set by set_ssi; uses the Dawkins definition by default.
    """

    intensity_thres = 14.7
    """Intensity threshold for storage in m/s; same as used by WISC SSI calculations."""

    vars_opt = Hazard.vars_opt.union({'ssi_wisc', 'ssi', 'ssi_full_area'})
    """Name of the variables that aren't need to compute the impact."""

    def __init__(self,
                 units: str = 'm/s',
                 ssi: Optional[np.ndarray] = None,
                 ssi_wisc: Optional[np.ndarray] = None,
                 ssi_full_area: Optional[np.ndarray] = None,
                 **kwargs):
        """Initialize a StormEurope object

        Parameters
        ----------
        units : str
            The units of the Hazard intensity. Defaults to 'm/s'
        ssi : numpy.ndarray
            The Storm Severity Index (SSI). 1d vector, same length as number of
            storms. Defaults to an empty array.
        ssi_wisc : numpy.ndarray
            The Storm Severity Index (SSI) as reconded in the footprint files.
            1d vector, same length as number of storms. Defaults to an empty
            array.
        ssi_full_area : numpy.ndarray
            1d vector, same length as number of storms.
            Used to conserve the the index that is derived from the area and windspeed
            of the full geographic extent of a storm system. Needed for cropping the
            `StormEurope` object into a smaller region`.
            Defaults to an empty array.
        """
        kwargs.setdefault('haz_type', HAZ_TYPE)
        Hazard.__init__(self, units=units, **kwargs)
        self.ssi = ssi if ssi is not None else np.array([], float)
        self.ssi_wisc = ssi_wisc if ssi_wisc is not None else np.array([], float)
        self.ssi_full_area = ssi_full_area if ssi_full_area is not None else np.array([], float)

    def read_footprints(self, *args, **kwargs):
        """This function is deprecated, use StormEurope.from_footprints instead."""
        LOGGER.warning("The use of StormEurope.read_footprints is deprecated."
                       "Use StormEurope.from_footprints instead.")
        self.__dict__ = StormEurope.from_footprints(*args, **kwargs).__dict__

    @classmethod
    def from_footprints(cls, path, description=None, ref_raster=None, centroids=None,
                        files_omit='fp_era20c_1990012515_701_0.nc', combine_threshold=None,
                        intensity_thres=None):
        """Create new StormEurope object from WISC footprints.

        Assumes that all footprints have the same coordinates as the first file listed/first
        file in dir.

        Parameters
        ----------
        path : str, list(str)
            A location in the filesystem. Either a
            path to a single netCDF WISC footprint, or a folder
            containing only footprints, or a globbing pattern to one or
            more footprints.
        description : str, optional
            description of the events, defaults
            to 'WISC historical hazard set'
        ref_raster : str, optional
            Reference netCDF file from which to
            construct a new barebones Centroids instance. Defaults to
            the first file in path.
        centroids : Centroids, optional
            A Centroids struct, overriding
            ref_raster
        files_omit : str, list(str), optional
            List of files to omit;
            defaults to one duplicate storm present in the WISC set as
            of 2018-09-10.
        combine_threshold : int, optional
            threshold for combining events
            in number of days. if the difference of the dates (self.date)
            of two events is smaller or equal to this threshold, the two
            events are combined into one.
            Default is None, Advised for WISC is 2
        intensity_thres : float, optional
            Intensity threshold for storage in m/s. Default: class attribute
            StormEurope.intensity_thres (same as used by WISC SSI calculations)

        Returns
        -------
        haz : StormEurope
            StormEurope object with data from WISC footprints.
        """
        # pylint: disable=protected-access
        intensity_thres = cls.intensity_thres if intensity_thres is None else intensity_thres
        file_names = get_file_names(path)

        if ref_raster is not None and centroids is not None:
            LOGGER.warning('Overriding ref_raster with centroids')

        if centroids is not None:
            pass
        elif ref_raster is not None:
            centroids = cls._centroids_from_nc(ref_raster)
        elif ref_raster is None:
            centroids = cls._centroids_from_nc(file_names[0])

        if isinstance(files_omit, str):
            files_omit = [files_omit]

        LOGGER.info('Commencing to iterate over netCDF files.')

        file_names = set(file_names)
        files_to_read = sorted(file_names.difference(files_omit))
        files_to_skip = sorted(file_names.intersection(files_omit))
        if files_to_skip:
            LOGGER.info("Omitting files %s", files_to_skip)
        hazard_list = [
            cls._read_one_nc(
                file_name,
                centroids,
                intensity_thres) for file_name in files_to_read]
        haz = cls.concat([haz for haz in hazard_list if haz is not None])

        # Fix values after concatenation
        haz.event_id = np.arange(1, len(haz.event_id) + 1)
        haz.frequency = np.divide(
            np.ones_like(haz.date),
            np.max([(last_year(haz.date) - first_year(haz.date)), 1])
        )

        if description is None:
            description = "WISC historical hazard set."
        haz.tag = TagHazard(
            HAZ_TYPE, 'Hazard set not saved by default',
            description=description,
        )

        if combine_threshold is not None:
            LOGGER.info('Combining events with small difference in date.')
            difference_date = np.diff(haz.date)
            for event_id_i in haz.event_id[
                    np.append(difference_date <= combine_threshold, False)]:
                event_ids = [event_id_i, event_id_i + 1]
                haz._combine_events(event_ids)
        return haz

    @classmethod
    def _read_one_nc(cls, file_name, centroids, intensity_thres):
        """Read a single WISC footprint. Assumes a time dimension of length 1.
        Omits a footprint if another file with the same timestamp has already
        been read.

        Parameters
        ----------
        file_name : str
            Absolute or relative path to *.nc
        centroids : Centroids
            Centr. instance that matches the
            coordinates used in the *.nc, only validated by size.
        intensity_thres : float
            Intensity threshold for storage in m/s.

        Returns
        -------
        new_haz : StormEurope
            Hazard instance for one single storm.
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
        stacked = stacked.where(stacked > intensity_thres)
        stacked = stacked.fillna(0)

        # fill in values from netCDF
        ssi_wisc = np.array([float(ncdf.ssi)])
        intensity = sparse.csr_matrix(stacked)
        new_haz = cls(ssi_wisc=ssi_wisc,
                      intensity=intensity,
                      event_name=[ncdf.storm_name],
                      date=np.array([datetime64_to_ordinal(ncdf.time.data[0])]),
                      # fill in default values
                      centroids=centroids,
                      event_id=np.array([1]),
                      frequency=np.array([1]),
                      orig=np.array([True]),)

        ncdf.close()
        return new_haz

    def read_cosmoe_file(self, *args, **kwargs):
        """This function is deprecated, use StormEurope.from_cosmoe_file instead."""
        LOGGER.warning("The use of StormEurope.read_cosmoe_file is deprecated."
                       "Use StormEurope.from_cosmoe_file instead.")
        self.__dict__ = StormEurope.from_cosmoe_file(*args, **kwargs).__dict__

    @classmethod
    def from_cosmoe_file(cls, fp_file, run_datetime, event_date=None,
                         model_name='COSMO-2E', description=None, intensity_thres=None):
        """Create a new StormEurope object with gust footprint from weather forecast.

        The funciton is designed for the COSMO ensemble model used by
        the COSMO Consortium http://www.cosmo-model.org/ and postprocessed to
        an netcdf file using fieldextra. One event is one full day in UTC.
        Works for MeteoSwiss model output of
        COSMO-1E (11 members, resolution 1.1 km, forecast period 33-45 hours)
        COSMO-2E (21 members, resolution 2.2 km, forecast period 5 days)

        The frequency of each event is informed by their probability in the ensemble
        forecast and is equal to 1/11 or 1/21 for COSMO-1E or COSMO-2E, respectively.

        Parameters
        ----------
        fp_file : str
            string directing to one netcdf file
        run_datetime : datetime
            The starting timepoint of the forecast run
            of the cosmo model
        event_date : datetime, optional
            one day within the forecast
            period, only this day (00H-24H) will be included in the hazard
        model_name : str,optional
            provide the name of the COSMO model,
            for the description (e.g., 'COSMO-1E', 'COSMO-2E')
        description : str, optional
            description of the events, defaults
            to a combination of model_name and run_datetime
        intensity_thres : float, optional
            Intensity threshold for storage in m/s. Default: class attribute
            StormEurope.intensity_thres (same as used by WISC SSI calculations)

        Returns
        -------
        haz : StormEurope
            StormEurope object with data from COSMO ensemble file.
        """
        intensity_thres = cls.intensity_thres if intensity_thres is None else intensity_thres

        # read intensity from file
        ncdf = xr.open_dataset(fp_file)
        ncdf = ncdf.assign_coords(date=('time',ncdf["time"].dt.floor("D").values))

        if event_date:
            try:
                stacked = ncdf.sel(
                    time=event_date.strftime('%Y-%m-%d')
                    ).groupby('date').max().stack(intensity=('y_1', 'x_1'))
            except KeyError as ker:
                raise ValueError('Extraction of date and coordinates failed. This is most likely '
                                 'because the selected event_date '
                                 f'{event_date.strftime("%Y-%m-%d")} is not contained in the '
                                 'weather forecast selected by fp_file {fp_file}. Please adjust '
                                 f'event_date or fp_file.') from ker
            considered_dates = np.datetime64(event_date)
        else:
            time_covered_step = ncdf['time'].diff('time')
            time_covered_day = time_covered_step.groupby('date').sum()
            # forecast run should cover at least 18 hours of a day
            considered_dates_bool = time_covered_day >= np.timedelta64(18,'h')
            stacked = ncdf.groupby('date').max()\
                          .sel(date=considered_dates_bool)\
                          .stack(intensity=('y_1', 'x_1'))
            considered_dates = stacked['date'].values
        stacked = stacked.stack(date_ensemble=('date', 'epsd_1'))
        stacked = stacked.where(stacked.VMAX_10M > intensity_thres)
        stacked = stacked.fillna(0)

        # fill in values from netCDF
        intensity = sparse.csr_matrix(stacked.VMAX_10M.T)
        event_id = np.arange(stacked.date_ensemble.size) + 1
        date = np.repeat(
            np.array(datetime64_to_ordinal(considered_dates)),
            np.unique(ncdf.epsd_1).size
        )
        orig = np.full_like(event_id, False)
        orig[(stacked.epsd_1 == 0).values] = True
        if description is None:
            description = (model_name +
                           ' weather forecast windfield ' +
                           'for run startet at ' +
                           run_datetime.strftime('%Y%m%d%H'))

        # Create Hazard
        haz = cls(
            intensity=intensity,
            event_id=event_id,
            centroids = cls._centroids_from_nc(fp_file),
            # fill in default values
            orig=orig,
            date=date,
            event_name=[date_i + '_ens' + str(ens_i)
                        for date_i, ens_i
                        in zip(date_to_str(date), stacked.epsd_1.values + 1)],
            frequency=np.divide(
                np.ones_like(event_id),
                np.unique(ncdf.epsd_1).size),
            description=description,
            file_name="Hazard set not saved, too large to pickle",
        )

        # close netcdf file
        ncdf.close()
        haz.check()
        return haz

    def read_icon_grib(self, *args, **kwargs):
        """This function is deprecated, use StormEurope.from_icon_grib instead."""
        LOGGER.warning("The use of StormEurope.read_icon_grib is deprecated."
                       "Use StormEurope.from_icon_grib instead.")
        self.__dict__ = StormEurope.from_icon_grib(*args, **kwargs).__dict__

    @classmethod
    def from_icon_grib(cls, run_datetime, event_date=None, model_name='icon-eu-eps',
                       description=None, grib_dir=None, delete_raw_data=True,
                       intensity_thres=None):
        """Create new StormEurope object from DWD icon weather forecast footprints.

        New files are available for 24 hours on
        https://opendata.dwd.de, old files can be processed if they are
        already stored in grib_dir.
        One event is one full day in UTC. Current setup works for runs
        starting at 00H and 12H. Otherwise the aggregation is inaccurate,
        because of the given file structure with 1-hour, 3-hour and
        6-hour maxima provided.

        The frequency for one event is 1/(number of ensemble members)

        Parameters
        ----------
        run_datetime : datetime
            The starting timepoint of the forecast run
            of the icon model
        event_date : datetime, optional
            one day within the forecast
            period, only this day (00H-24H) will be included in the hazard
        model_name : str,optional
            select the name of the icon model to
            be downloaded. Must match the url on https://opendata.dwd.de
            (see download_icon_grib for further info)
        description : str, optional
            description of the events, defaults
            to a combination of model_name and run_datetime
        grib_dir : str, optional
            path to folder, where grib files are
            or should be stored
        delete_raw_data : bool,optional
            select if downloaded raw data in
            .grib.bz2 file format should be stored on the computer or
            removed
        intensity_thres : float, optional
            Intensity threshold for storage in m/s. Default: class attribute
            StormEurope.intensity_thres (same as used by WISC SSI calculations)

        Returns
        -------
        haz : StormEurope
            StormEurope object with data from DWD icon weather forecast footprints.
        """
        # pylint: disable=protected-access
        intensity_thres = cls.intensity_thres if intensity_thres is None else intensity_thres

        if not (run_datetime.hour == 0 or run_datetime.hour == 12):
            LOGGER.warning('The event definition is inaccuratly implemented '
                           'for starting times, which are not 00H or 12H.')
        # download files, if they don't already exist
        file_names = download_icon_grib(
            run_datetime, model_name=model_name, download_dir=grib_dir)

        # create centroids
        nc_centroids_file = download_icon_centroids_file(model_name, grib_dir)

        # read intensity from files
        for ind_i, file_i in enumerate(file_names):
            gripfile_path_i = Path(file_i[:-4])
            with open(file_i, 'rb') as source, open(gripfile_path_i, 'wb') as dest:
                dest.write(bz2.decompress(source.read()))
            ds_i = xr.open_dataset(gripfile_path_i, engine='cfgrib')
            if ind_i == 0:
                stacked = ds_i
            else:
                stacked = xr.concat([stacked,ds_i], 'valid_time')

        # create intensity matrix with max for each full day
        stacked = stacked.assign_coords(
            date=('valid_time', stacked["valid_time"].dt.floor("D").values))
        if event_date:
            try:
                stacked = stacked.sel(
                    valid_time=event_date.strftime('%Y-%m-%d')).groupby('date').max()
            except KeyError as ker:
                raise ValueError('Extraction of date and coordinates failed. This is most likely '
                                 'because the selected event_date '
                                 f'{event_date.strftime("%Y-%m-%d")} is not contained in the '
                                 'weather forecast selected by run_datetime'
                                 f'{run_datetime.strftime("%Y-%m-%d %H:%M")}. Please adjust '
                                 'event_date or run_datetime.') from ker

            considered_dates = np.datetime64(event_date)
        else:
            time_covered_step = stacked['valid_time'].diff('valid_time')
            time_covered_day = time_covered_step.groupby('date').sum()
            # forecast run should cover at least 18 hours of a day
            considered_dates_bool = time_covered_day >= np.timedelta64(18,'h')
            stacked = stacked.groupby('date').max().sel(date=considered_dates_bool)
            considered_dates = stacked['date'].values
        stacked = stacked.stack(date_ensemble=('date', 'number'))
        stacked = stacked.where(stacked > intensity_thres)
        stacked = stacked.fillna(0)

        event_id = np.arange(stacked.date_ensemble.size) + 1
        date = np.repeat(
            np.array(datetime64_to_ordinal(considered_dates)),
            np.unique(stacked.number).size
        )
        orig = np.full_like(event_id, False)
        orig[(stacked.number == 1).values] = True
        if description is None:
            description = ('icon weather forecast windfield for run started at ' +
                           run_datetime.strftime('%Y%m%d%H'))

        # Create Hazard
        haz = cls(
            intensity=sparse.csr_matrix(stacked.gust.T),
            centroids=cls._centroids_from_nc(nc_centroids_file),
            event_id=event_id,
            date=date,
            orig=orig,
            event_name=[date_i + '_ens' + str(ens_i)
                        for date_i, ens_i
                        in zip(date_to_str(date), stacked.number.values)],
            frequency=np.divide(
                np.ones_like(event_id),
                np.unique(stacked.number).size),
            description=description,
            file_name="Hazard set not saved, too large to pickle")
        haz.check()

        # delete generated .grib2 and .4cc40.idx files
        for ind_i, file_i in enumerate(file_names):
            gripfile_path_i = Path(file_i[:-4])
            idxfile_path_i = next(gripfile_path_i.parent.glob(
                str(gripfile_path_i.name) + '.*.idx'))
            gripfile_path_i.unlink()
            idxfile_path_i.unlink()

        if delete_raw_data:
            #delete downloaded .bz2 files
            delete_icon_grib(run_datetime, model_name=model_name, download_dir=grib_dir)

        return haz

    @staticmethod
    def _centroids_from_nc(file_name):
        """Construct Centroids from the grid described by 'latitude' and
        'longitude' variables in a netCDF file.
        """
        LOGGER.info('Constructing centroids from %s', file_name)
        ncdf = xr.open_dataset(file_name)
        create_meshgrid = True
        if hasattr(ncdf, 'latitude'):
            lats = ncdf.latitude.data
            lons = ncdf.longitude.data
        elif hasattr(ncdf, 'lat'):
            lats = ncdf.lat.data
            lons = ncdf.lon.data
        elif hasattr(ncdf, 'lat_1'):
            if len(ncdf.lon_1.shape)>1 & \
                (ncdf.lon_1.shape == ncdf.lat_1.shape) \
                :
                lats = ncdf.lat_1.data.flatten()
                lons = ncdf.lon_1.data.flatten()
                create_meshgrid = False
            else:
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
        cent = Centroids.from_lat_lon(lats, lons)
        cent.set_area_pixel()
        cent.set_on_land()

        return cent

    def _combine_events(self, event_ids):
        """combine the intensities of two events using max and adjust event_id, event_name,
        date etc of the hazard

        the event_ids must be consecutive for the event_name field to behave correctly

        Parameters
        ----------
        event_ids : array
            two consecutive event ids
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

        Parameters
        ----------
        method : str
            Either 'dawkins' or 'wisc_gust'
        intensity : scipy.sparse.csr
            Intensity matrix; defaults to
            self.intensity
        on_land : bool
            Only calculate the SSI for areas on land,
            ignoring the intensities at sea. Defaults to true, whereas
            the MATLAB version did not.
        threshold : float, optional
            Intensity threshold used in index
            definition. Cannot be lower than the read-in value.
        sel_cen : np.array, bool
            A boolean vector selecting centroids.
            Takes precendence over on_land.
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

        Parameters
        ----------
        kwargs :
            passed on to calc_ssi
        """
        self.ssi = self.calc_ssi(**kwargs)

    def plot_ssi(self, full_area=False):
        """Plot the distribution of SSIs versus their cumulative exceedance
            frequencies, highlighting historical storms in red.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes._subplots.AxesSubplot
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

    def generate_prob_storms(self, reg_id=528, spatial_shift=4, ssi_args=None,
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

        Parameters
        ----------
        region_id : int, list of ints, or None
            iso_n3 code of the
            countries we want the generated hazard set to be returned for.
        spatial_shift : int
            amount of raster pixels to shift by
        ssi_args : dict
            A dictionary of arguments passed to calc_ssi
        kwargs :
            keyword arguments passed on to self._hist2prob()

        Returns
        -------
        new_haz : StormEurope
            A new hazard set for the given country.
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
        base = np.repeat((self.event_id * 100), N_PROB_EVENTS)
        synth_id = np.tile(np.arange(N_PROB_EVENTS), self.size)
        event_id = base + synth_id
        new_haz = StormEurope(
            ssi_full_area=ssi,
            intensity=sparse.csr_matrix(intensity_prob),
            # don't use synthetic dates; just repeat the
            # historic dates
            date=np.repeat(self.date, N_PROB_EVENTS),
            # subsetting centroids
            centroids=self.centroids.select(sel_cen=sel_cen),
            # construct new event ids

            event_id=event_id,
            # frequency still based on the historic number of
            # years
            frequency=np.divide(np.repeat(self.frequency, N_PROB_EVENTS),
                                N_PROB_EVENTS),
            file_name='Hazard set not saved by default',
            description='WISC probabilistic hazard set according to Schwierz et al.',
            orig=(event_id % 100 == 0),
        )
        new_haz.check()

        return new_haz

    def _hist2prob(self, intensity1d, sel_cen, spatial_shift, ssi_args=None,
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
        if not ssi_args:
            ssi_args = {}
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


# pylint: disable=invalid-name
def generate_WS_forecast_hazard(run_datetime=None,
                                event_date=None,
                                haz_model='icon-eu-eps',
                                haz_raw_storage=None,
                                save_haz=True):
    """use the initialization time (run_datetime), the date of the event and
    specify the forecast model (haz_model) to generate a Hazard from forecast
    data either by download or through reading from existing file.

    Parameters
    ----------
    run_datetime: datetime.datetime, optional
        The starting timepoint of the forecast run
        of the icon model, defaults to today 00:00 hours
    event_date: datetime.datetime, optional
        one day within the forecast
        period, only this day (00H-24H) will be included in the hazard,
        defaults to the day after tomorrow
    haz_model: str, optional
        select the name of the model to
        be used: one of ['icon-eu-eps', 'icon-d2-eps', 'cosmo1e_file',
                         'cosmo2e_file'],
        default is 'icon-eu-eps'.
    haz_raw_storage: str, optional
        path to folder, where netcdf files
        are stored, mendatory when haz_model is 'cosmo1e_file' or
        'cosmo2e_file'. the string must contain one set of curly brakets,
        used by the function str.format to include
        run_datetime.strftime('%y%m%d%H').
        Default None resolves to "cosmoe_forecast_{}_vmax.nc" in
        CONFIG.hazard.storm_europe.forecast_dir
    save_haz: bool, optional
        flag if resulting hazard should be saved in
        CONFIG.hazard.storm_europe.forecast_dir, default is True.

    Returns
    -------
    Hazard
        hazard
    str
        haz_model
    datetime.datetime
        run_datetime
    datetime.datetime
        event_date
    """
    FORECAST_DIR = CONFIG.hazard.storm_europe.forecast_dir.dir()

    if run_datetime is None:
        run_datetime = dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    if event_date is None:
        event_date = dt.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0) \
                   + dt.timedelta(days=2)

    if haz_model in ['cosmo1e_file', 'cosmo2e_file']:
        if haz_model == 'cosmo1e_file':
            haz_model='C1E'
            full_model_name_temp = 'COSMO-1E'
        if haz_model == 'cosmo2e_file':
            haz_model='C2E'
            full_model_name_temp = 'COSMO-2E'
        haz_file_name = (f'{HAZ_TYPE}_{haz_model}_run{run_datetime.strftime("%Y%m%d%H")}'
                         f'_event{event_date.strftime("%Y%m%d")}.hdf5')
        haz_file = FORECAST_DIR / haz_file_name
        if haz_file.exists():
            LOGGER.info('Loading hazard from %s.', haz_file)
            hazard = StormEurope.from_hdf5(haz_file)
        else:
            LOGGER.info('Generating %s hazard.', haz_model)
            if not haz_raw_storage:
                haz_raw_storage = FORECAST_DIR / "cosmoe_forecast_{}_vmax.nc"
            fp_file = Path(str(haz_raw_storage).format(run_datetime.strftime('%y%m%d%H')))
            hazard = StormEurope.from_cosmoe_file(
                fp_file,
                event_date=event_date,
                run_datetime=run_datetime,
                model_name=full_model_name_temp
            )
            if save_haz:
                hazard.write_hdf5(haz_file)
    elif haz_model in ['icon-eu-eps', 'icon-d2-eps']:
        if haz_model == 'icon-eu-eps':
            full_model_name_temp = haz_model
            haz_model='IEE'
        if haz_model == 'icon-d2-eps':
            full_model_name_temp = haz_model
            haz_model='IDE'
        haz_file_name = (f'{HAZ_TYPE}_{haz_model}_run{run_datetime.strftime("%Y%m%d%H")}'
                         f'_event{event_date.strftime("%Y%m%d")}.hdf5')
        haz_file = FORECAST_DIR / haz_file_name
        if haz_file.exists():
            LOGGER.info('Loading hazard from %s.', haz_file)
            hazard = StormEurope.from_hdf5(haz_file)
        else:
            LOGGER.info('Generating %s hazard.', haz_model)
            hazard = StormEurope.from_icon_grib(
                run_datetime,
                event_date=event_date,
                delete_raw_data=False,
                model_name=full_model_name_temp
            )
            if save_haz:
                hazard.write_hdf5(haz_file)
    else:
        raise NotImplementedError("specific 'WS' hazard not implemented yet. " +
                                  "Please specify a valid value for haz_model.")

    # check if hazard is successfully generated for Forecast
    if not isinstance(hazard, Hazard):
        LOGGER.warning('Hazard generation unsuccessful.')
    return hazard, haz_model, run_datetime, event_date
