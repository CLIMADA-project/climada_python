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

Define AgriculturalDrought (AD) class.
WORK IN PROGRESS
"""

__all__ = ['RelativeCropyield']

import logging
from pathlib import Path
import copy

import numpy as np
from matplotlib import pyplot as plt
import cartopy
import shapely.geometry
from scipy import sparse
import scipy.stats
import h5py
import xarray as xr

from climada.hazard.base import Hazard
from climada.util import dates_times as dt
from climada.util import coordinates as coord
from climada import CONFIG


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RC'
"""Hazard type acronym for Relative Cropyield"""

INT_DEF = 'Yearly Yield'

BBOX = (-180, -85, 180, 85)  # [Lon min, lat min, lon max, lat max]
""""Default geographical bounding box of the total global agricultural land extent"""

# ! deposit the input files in: climada_python/data/ISIMIP_crop/Input/Hazard
DATA_DIR = CONFIG.hazard.relative_cropyield.local_data.dir()
INPUT_DIR = DATA_DIR.joinpath('Input', 'Hazard')
"""default paths for input and output data:"""
OUTPUT_DIR = DATA_DIR.joinpath('Output')


#ISIMIP input data specific global variables
"""start and end years per senario as in ISIMIP-filenames"""
YEARCHUNKS = {'ISIMIP2a': {'yearrange': (1980, 1999), 'startyear': 1980, 'endyear': 1999,
                           'yearrange_mean': (1980, 1999)},
              'historical': {'yearrange': (1976, 2005), 'startyear': 1861, 'endyear': 2005,
                             'yearrange_mean': (1976, 2005)},
              'historical_ISIMIP3b': {'yearrange': (1850, 2014),
                                      'startyear': 1850, 'endyear': 2014,
                                      'yearrange_mean': (1983, 2013)},
              'rcp26': {'yearrange': (2006, 2099), 'startyear': 2006, 'endyear': 2099},
              'rcp26-2': {'yearrange': (2100, 2299), 'startyear': 2100, 'endyear': 2299},
              'rcp60': {'yearrange': (2006, 2099), 'startyear': 2006, 'endyear': 2099},
              'rcp85': {'yearrange': (2006, 2099), 'startyear': 2006, 'endyear': 2099},
              'ssp585': {'yearrange': (2015, 2100), 'startyear': 2015, 'endyear': 2100},
              'ssp126': {'yearrange': (2015, 2100), 'startyear': 2015, 'endyear': 2100},
              }



FN_STR_VAR = 'global_annual'
"""filename of ISIMIP output constant part"""


class RelativeCropyield(Hazard):
    """Agricultural climate risk: Relative Cropyield (relative to historical mean);
    Each year corresponds to one hazard event;
    Based on modelled crop yield, from ISIMIP (www.isimip.org, required input data).
    Attributes as defined in Hazard and the here defined additional attributes.

    Attributes:
        crop_type (str): crop type ('whe' for wheat, 'mai' for maize, 'soy' for soybeans
                                    and 'ric' for rice)
        intensity_def (str): intensity defined as:
            'Yearly Yield' [t/(ha*y)], 'Relative Yield', or 'Percentile'
    """

    def __init__(self, pool=None):
        """Empty constructor."""
        Hazard.__init__(self, HAZ_TYPE)
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        self.crop = ''
        self.intensity_def = INT_DEF

    def set_from_isimip_netcdf(self, input_dir=None, filename=None, bbox=None,
                               yearrange=None, ag_model=None, cl_model=None, bias_corr=None,
                               scenario=None, soc=None, co2=None, crop=None,
                               irr=None, fn_str_var=None):

        """Wrapper to fill hazard from crop yield NetCDF file.
        Build and tested for output from ISIMIP2 and ISIMIP3, but might also work
        for other NetCDF containing gridded crop model output from other sources.
        Parameters:
            input_dir (Path or str): path to input data directory,
                default: {CONFIG.exposures.crop_production.local_data}/Input/Exposure
            filename (string): name of netcdf file in input_dir. If filename is given,
                the other parameters specifying the model run are not required!
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for hazard set, f.i. (1976, 2005)
            ag_model (str): abbrev. agricultural model (only when input_dir is selected)
                f.i. 'clm-crop', 'gepic','lpjml','pepic'
            cl_model (str): abbrev. climate model (only when input_dir is selected)
                f.i. ['gfdl-esm2m', 'hadgem2-es','ipsl-cm5a-lr','miroc5'
            bias_corr (str): bias correction of climate forcing,
                f.i. 'ewembi' (ISIMIP2b, default) or 'w5e5' (ISIMIP3b)
            scenario (str): climate change scenario (only when input_dir is selected)
                f.i. 'historical' or 'rcp60' or 'ISIMIP2a'
            soc (str): socio-economic trajectory (only when input_dir is selected)
                f.i. '2005soc' or 'histsoc'
            co2 (str): CO2 forcing scenario (only when input_dir is selected)
                f.i. 'co2' or '2005co2'
            crop (str): crop type (only when input_dir is selected)
                f.i. 'whe', 'mai', 'soy' or 'ric'
            irr (str): irrigation type (only when input_dir is selected)
                f.i 'noirr' or 'irr'
            fn_str_var (str): FileName STRing depending on VARiable and
                ISIMIP simuation round
        raises:
            NameError
        """
        if not fn_str_var:
            fn_str_var = FN_STR_VAR
        if scenario is None:
            scenario = 'historical'
        if bias_corr is None:
            bias_corr = 'ewembi'
        if bbox is None:
            bbox = BBOX
        if input_dir is None:
            input_dir = INPUT_DIR
        input_dir = Path(input_dir)
        if not Path(input_dir).is_dir():
            raise NameError('Input directory %s does not exist' % input_dir)

        # The filename is set or other variables (cl_model, scenario) are extracted of the
        # specified filename
        if filename is None:
            yearchunk = YEARCHUNKS[scenario]
            filename = '{}_{}_{}_{}_{}_{}_yield-{}-{}_{}_{}_{}.nc'.format(
                ag_model, cl_model, bias_corr, scenario, soc, co2,
                crop, irr, fn_str_var, yearchunk['startyear'], yearchunk['endyear'])
        elif scenario == 'ISIMIP2a':
            (_, _, _, _, _, _, _, crop, _, _, startyear, endyearnc) = filename.split('_')
            endyear, _ = endyearnc.split('.')
            yearchunk = dict()
            yearchunk = {'yearrange': (int(startyear), int(endyear)),
                         'startyear': int(startyear), 'endyear': int(endyear)}
        elif scenario == 'test_file':
            yearchunk = dict()
            yearchunk = {'yearrange': (1976, 2005), 'startyear': 1861,
                         'endyear': 2005, 'yearrange_mean': (1976, 2005)}
            ag_model, cl_model, _, _, soc, co2, crop_prop, *_ = filename.split('_')
            _, crop, irr = crop_prop.split('-')
        else: # get yearchunk from filename, e.g., for rcp2.6 extended and ISIMIP3
            (_, _, _, _, _, _, crop_irr, _, _, year1, year2) = filename.split('_')
            yearchunk = {'yearrange': (int(year1), int(year2.split('.')[0])),
                         'startyear': int(year1),
                         'endyear': int(year2.split('.')[0])}
            _, crop, irr = crop_irr.split('-')

        # if no yearrange is given, load full range from input file:
        if yearrange is None or len(yearrange) == 0:
            yearrange = yearchunk['yearrange']

        # define indexes of the netcdf-bands to be extracted, and the
        # corresponding event names and dates
        # corrected indexes due to the bands in input starting with the index=1
        id_bands = np.arange(yearrange[0] - yearchunk['startyear'] + 1,
                             yearrange[1] - yearchunk['startyear'] + 2).tolist()

        # hazard setup: set attributes
        [lonmin, latmin, lonmax, latmax] = bbox
        self.set_raster([str(Path(input_dir, filename))], band=id_bands,
                        geometry=list([shapely.geometry.box(lonmin, latmin, lonmax, latmax)]))

        self.intensity.data[np.isnan(self.intensity.data)] = 0.0
        self.intensity.todense()
        self.crop = crop
        self.event_name = [str(n) for n in range(int(yearrange[0]), int(yearrange[-1] + 1))]
        self.frequency = np.ones(len(self.event_name)) * (1 / len(self.event_name))
        self.fraction = self.intensity.copy()
        self.fraction.data.fill(1.0)
        self.units = 't / y / ha'
        self.date = np.array(dt.str_to_date(
            [event_ + '-01-01' for event_ in self.event_name]))
        self.centroids.set_meta_to_lat_lon()
        self.centroids.region_id = (
            coord.coord_on_land(self.centroids.lat, self.centroids.lon)).astype(dtype=int)
        self.check()
        return self

    def calc_mean(self, yearrange_mean=None,
                  save=False, output_dir=None):
        """Calculates mean of the hazard for a given reference time period

            Optional Parameters:
            yearrange_mean (array): time period used to calculate the mean intensity
                default: 1976-2005 (historical)
            save (boolean): save mean to file? default: False
            output_dir (str or Path): path of output directory,
                default: {CONFIG.exposures.crop_production.local_data}/Output

            Returns:
                hist_mean(array): contains mean value over the given reference
                    time period for each centroid
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR
        output_dir = Path(output_dir)
        if yearrange_mean is None:
            yearrange_mean = YEARCHUNKS['historical']['yearrange_mean']
        startyear, endyear = yearrange_mean
        event_list = [str(n) for n in range(int(startyear), int(endyear + 1))]
        mean = self.select(event_names=event_list).intensity.mean(axis=0)
        hist_mean = np.squeeze(np.asarray(mean))

        if save:
            # generate output directories if they do not exist yet
            mean_dir = output_dir / 'Hist_mean'
            mean_dir.mkdir(parents=True, exist_ok=True)
            # save mean_file
            mean_file = h5py.File(Path(
                mean_dir,
                f'hist_mean_{self.crop}_{startyear}-{endyear}.hdf5'
            ), 'w')
            mean_file.create_dataset('mean', data=hist_mean)
            mean_file.create_dataset('lat', data=self.centroids.lat)
            mean_file.create_dataset('lon', data=self.centroids.lon)
            mean_file.close()

        return hist_mean


    def set_rel_yield_to_int(self, hist_mean):
        """Sets relative yield (yearly yield / historic mean) as intensity

            Parameters:
                hist_mean (array): historic mean per centroid

            Returns:
                hazard with modified intensity [unitless]
        """
        # determine idx of the centroids with a mean yield !=0
        [idx] = np.where(hist_mean != 0)

        # initialize new hazard_matrix
        hazard_matrix = np.zeros(self.intensity.shape, dtype=np.float32)

        # compute relative yield for each event:
        for event in range(len(self.event_id)):
            hazard_matrix[event, idx] = (self.intensity[event, idx] / hist_mean[idx])-1

        self.intensity = sparse.csr_matrix(hazard_matrix)
        self.intensity_def = 'Relative Yield'
        self.units = ''

        return self

    def set_percentile_to_int(self, reference_intensity=None):
        """Sets percentile to intensity

            Parameters:
                reference_intensity (AD): intensity to be used as reference
                    (e.g. the historic intensity can be used in order to be able
                     to directly compare historic and future projection data)

            Returns:
                hazard with modified intensity
        """
        hazard_matrix = np.zeros(self.intensity.shape)
        if reference_intensity is None:
            reference_intensity = self.intensity

        for centroid in range(self.intensity.shape[1]):
            nevents = reference_intensity.shape[0]
            array = reference_intensity[:, centroid].toarray().reshape(nevents)
            hazard_matrix[:, centroid] = np.array([scipy.stats.percentileofscore(array, event)
                                                   for event in array])/100
        self.intensity = sparse.csr_matrix(hazard_matrix)
        self.intensity_def = 'Percentile'
        self.units = ''

        return self

    def plot_intensity_cp(self, event=None, dif=False, axis=None, **kwargs):
        """Plots intensity with predefined settings depending on the intensity definition

        Optional Parameters:
            event (int or str): event_id or event_name
            dif (boolean): variable signilizing whether absolute values or the difference between
                future and historic are plotted (False: his/fut values; True: difference = fut-his)
            axis (geoaxes): axes to plot on

        Returns:
            axes (geoaxes)
        """
        if not dif:
            if self.intensity_def == 'Yearly Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='YlGn', vmin=0, vmax=10,
                                           **kwargs)
            elif self.intensity_def == 'Relative Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=-1, vmax=1,
                                           **kwargs)
            elif self.intensity_def == 'Percentile':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=0, vmax=1,
                                           **kwargs)
        else:
            if self.intensity_def == 'Yearly Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=-2, vmax=2,
                                           **kwargs)
            elif self.intensity_def == 'Relative Yield':
                axes = self.plot_intensity(event=event, axis=axis, cmap='RdBu', vmin=-0.5,
                                           vmax=0.5, **kwargs)

        return axes

    def plot_time_series(self, event=None):
        """Plots a time series of intensities (a series of sub plots)

        Optional Parameters:
            event (int or str): event_id or event_name

        Returns:
            figure
        """

        # if no event range is given, all events contained in self are plotted
        # in the case that a specific range is given as input (event) only the events
        # within this time range are plotted
        if event is None:
            event = self.event_name
        else:
            event = [str(n) for n in range(event[0], event[1] + 1)]

        self.centroids.set_meta_to_lat_lon()

        # definition of plot extents
        len_lat = abs(self.centroids.lat[0] - self.centroids.lat[-1]) * (2.5 / 13.5)
        len_lon = abs(self.centroids.lon[0] - self.centroids.lon[-1]) * (5 / 26)

        nr_subplots = len(event)

        if len_lon >= len_lat:
            colums = int(np.floor(np.sqrt(nr_subplots / (len_lon / len_lat))))
            rows = int(np.ceil(nr_subplots / colums))
        else:
            rows = int(np.floor(np.sqrt(nr_subplots / (len_lat / len_lon))))
            colums = int(np.ceil(nr_subplots / colums))

        fig, axes = plt.subplots(rows, colums, sharex=True, sharey=True,
                                 figsize=(colums * len_lon, rows * len_lat),
                                 subplot_kw=dict(projection=cartopy.crs.PlateCarree()))
        colum = 0
        row = 0

        for year in range(nr_subplots):
            axes.flat[year].set_extent([np.min(self.centroids.lon), np.max(self.centroids.lon),
                                        np.min(self.centroids.lat), np.max(self.centroids.lat)])

            if rows == 1:
                self.plot_intensity_cp(event=event[year], axis=axes[colum])
            elif colums == 1:
                self.plot_intensity_cp(event=event[year], axis=axes[row])
            else:
                self.plot_intensity_cp(event=event[year], axis=axes[row, colum])

            if colum <= colums - 2:
                colum = colum + 1
            else:
                colum = 0
                row = row + 1

        return fig


def set_multiple_rc_from_isimip(input_dir=None, output_dir=None, bbox=None,
                                isimip_run=None, yearrange_his=None, yearrange_mean=None,
                                return_data=False, save=True, combine_subcrops=True):

    """Wrapper to generate full hazard set from all ISIMIP-NetCDF files with
    crop yield in a given input directory and save it to output directory.

        Optional Parameters:
            input_dir (pathlib.Path or str): path to input data directory,
                default: {CONFIG.exposures.crop_production.local_data}/Input/Exposure
            output_dir (pathlib.Path or str): path to output data directory,
                default: {CONFIG.exposures.crop_production.local_data}/Output
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            isimip_run (string): name of the ISIMIP run (f.i. ISIMIP2a or ISIMIP2b)
            yearrange_his (int tuple): year range for the historical hazard sets
            yearrange_mean (int tuple): year range for the historical mean
            return_data (boolean): returned output
                False: returns list of filenames only
                True: returns also list of data
            save (boolean): save output data to output_dir
            combine_subcrops (boolean): combine crops: ric=ri1+ri2, whe=swh+wwh
        Return:
            filename_list (list): list of filenames

        Optional Return:
            output_list (list): list of generated output data (hazards and historical mean)
    """
    if bbox is None:
        bbox = BBOX
    if isimip_run is None:
        isimip_run = 'ISIMIP2b'
    if input_dir is None:
        input_dir = INPUT_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if not (isinstance(input_dir, Path) and input_dir.is_dir()):
        raise NameError('input_dir needs to be valid directory given as str or Path instance')
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if not (isinstance(output_dir, Path) and output_dir.is_dir()):
        raise NameError('output_dir needs to be valid directory given as str or Path instance')

    filenames = [f.name for f in input_dir.iterdir()
                 if f.is_file() and not f.name.startswith('.') and not f.name.startswith('mask')]

    # generate output directories if they do not exist yet
    Path(output_dir, 'Hazard').mkdir(parents=True, exist_ok=True)
    Path(output_dir, 'Hist_mean').mkdir(parents=True, exist_ok=True)

    filename_list = list()
    output_list = list()

    (his_file_list, file_props, hist_mean_per_crop,
     scenario_list, _, combi_crop_list) = init_hazard_sets_isimip(filenames,
                                                                  input_dir=input_dir,
                                                                  bbox=bbox, isimip_run=isimip_run,
                                                                  yearrange_his=yearrange_his,
                                                                  combine_subcrops=combine_subcrops)

    if (yearrange_mean is None) and (isimip_run == 'ISIMIP2b'):
        yearrange_mean = YEARCHUNKS[file_props[his_file_list[0]]['scenario']]['yearrange_mean']
    elif (yearrange_mean is None) and (isimip_run == 'ISIMIP3b'):
        yearrange_mean = YEARCHUNKS['historical_ISIMIP3b']['yearrange_mean']
        # (1983, 2013) # c.f. Jaegermeyr et al. on ISIMIP3b
    elif (yearrange_mean is None) and (isimip_run == 'ISIMIP2a'):
        yearrange_mean = YEARCHUNKS['ISIMIP2a']['yearrange_mean']
        # (1980, 1999)

    for his_file in his_file_list:
        haz_his, filename, hist_mean = calc_his_haz_isimip(his_file, file_props,
                                                           input_dir=input_dir, bbox=bbox,
                                                           yearrange_mean=yearrange_mean)
        # save the historical mean depending on the crop-irrigation combination
        # the idx keeps track of the row in which the hist_mean values are written per crop-irr to
        # ensure that all files are assigned to the corresponding crop-irr combination
        hist_mean_per_crop[file_props[his_file]['combi_crop_irr']]['value'][
            hist_mean_per_crop[file_props[his_file]['combi_crop_irr']]['idx'], :] = hist_mean
        hist_mean_per_crop[file_props[his_file]['combi_crop_irr']]['idx'] += 1

        filename_list.append(filename)
        if return_data:
            output_list.append(haz_his)
        else: output_list.append(None)
        if save:
            haz_his.select(reg_id=1).write_hdf5(str(Path(output_dir, 'Hazard', filename)))

        if isimip_run in ('ISIMIP2b', 'ISIMIP3b'):
            # compute the relative yield for all future scenarios with the corresponding
            # historic mean
            for scenario in scenario_list: # loop over all scenarios except historical
                # check whether future file exists for given historical file and scenario:
                haz_fut = None
                filename = None
                fut_file = '{}_{}_{}_{}_{}_{}_yield-{}-{}_{}_{}_{}.nc'.format(
                    file_props[his_file]['ag_model'],
                    file_props[his_file]['cl_model'],
                    file_props[his_file]['bias_corr'],
                    scenario,
                    file_props[his_file]['soc'],
                    file_props[his_file]['co2'],
                    file_props[his_file]['crop'],
                    file_props[his_file]['irr'],
                    FN_STR_VAR,
                    YEARCHUNKS[scenario]['startyear'],
                    YEARCHUNKS[scenario]['endyear']
                )

                if Path(input_dir, fut_file).is_file():
                    # if true, calculate and save future hazard set:
                    haz_fut, filename = calc_fut_haz_isimip(his_file, scenario,
                                                            file_props, hist_mean,
                                                            input_dir=input_dir,
                                                            bbox=bbox,
                                                            fut_file=fut_file)
                    filename_list.append(filename)
                    if save:
                        haz_fut.select(reg_id=1)\
                            .write_hdf5(str(Path(output_dir, 'Hazard', filename)))
                    if return_data:
                        output_list.append(haz_fut)
                    else: output_list.append(None)

                if scenario == 'rcp26': # also test for extended
                    # check whether future file exists for given historical file and scenario:
                    fut_file = '{}_{}_{}_{}_{}_{}_yield-{}-{}_{}_{}_{}.nc'.format(
                        file_props[his_file]['ag_model'],
                        file_props[his_file]['cl_model'],
                        file_props[his_file]['bias_corr'],
                        scenario,
                        file_props[his_file]['soc'],
                        file_props[his_file]['co2'],
                        file_props[his_file]['crop'],
                        file_props[his_file]['irr'],
                        FN_STR_VAR,
                        YEARCHUNKS['rcp26-2']['startyear'],
                        YEARCHUNKS['rcp26-2']['endyear']
                    )

                    if Path(input_dir, fut_file).is_file():
                        # if true, calculate and save future hazard set:
                        haz_fut, filename = calc_fut_haz_isimip(his_file, scenario,
                                                                file_props, hist_mean,
                                                                input_dir=input_dir,
                                                                bbox=bbox, fut_file=fut_file)
                        filename_list.append(filename)
                        if save:
                            haz_fut.select(reg_id=1)\
                                .write_hdf5(str(Path(output_dir, 'Hazard', filename)))
                        if return_data:
                            output_list.append(haz_fut)
                        else: output_list.append(None)

    # calculate mean hist_mean for each crop-irrigation combination and save as hdf5
    # in output_dir (required for full exposure set preparation):
    for combi_crop_irr in combi_crop_list:
        mean = np.mean((hist_mean_per_crop[combi_crop_irr])['value'], 0)
        mean_filename = ('hist_mean_' + combi_crop_irr + '_' + str(yearrange_mean[0]) +'-' +
                         str(yearrange_mean[1]) + '.hdf5')
        filename_list.append(mean_filename)
        output_list.append(mean)

    if save: # save hist_mean files to hdf5 file:
        for idx, filename in enumerate(filename_list):
            if 'hist_mean_' in filename:
                mean_file = h5py.File(str(Path(output_dir, 'Hist_mean', filename)), 'w')
                mean_file.create_dataset('mean', data=output_list[idx])
                mean_file.create_dataset('lat', data=haz_his.centroids.lat)
                mean_file.create_dataset('lon', data=haz_his.centroids.lon)
                mean_file.close()

    return filename_list, output_list

def init_hazard_sets_isimip(filenames, input_dir=None, bbox=None, isimip_run=None,
                            yearrange_his=None, combine_subcrops=True):
    """Initialize full hazard set.

        Parameters:
            filenames (list): list of filenames

        Optional Parameters:
            input_dir (pathlib.Path): path to input data directory, default: INPUT_DIR
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max], default: BBOX
            isimip_run (string): name of the ISIMIP run ('ISIMIP2a', 'ISIMIP2b', or 'ISIMIP3b')
                Deafult: 'ISIMIP2b''
            yearrange_his (int tuple): year range for the historical hazard sets
            combine_subcrops (bool): ignore crops ri2 (2nd harvest rice) and wwh (winter wheat)
                at this step (will be added to ri1 and wwh to form ric and whe later on)

        Return:
            his_file_list (list): list of historical input hazard files
            file_props (dict): file properties of all historical input hazard files
            hist_mean_per_crop (dict): empty dictonary to save hist_mean values for each
                crop-irr combination
            scenario_list (list): list of all future scenarios
            crop_irr_list (list): list of all crop-irr combinations
            combi_crop_irr_list (list): list of all crop-irr combinations for combined crops
    """
    # set default values for parameters not defined:
    if input_dir is None:
        input_dir = Path(INPUT_DIR)
    if bbox is None:
        bbox = BBOX
    if isimip_run is None:
        isimip_run = 'ISIMIP2b'
    crop_irr_list = list()
    combi_crop_irr_list = list()
    file_props = dict()
    his_file_list = list()
    scenario_list = list()

    for file in filenames:
        if isimip_run in ('ISIMIP2b', 'ISIMIP3b'):
            ag_model, cl_model, bias_corr, scenario, soc, co2, crop_prop, *_ = file.split('_')
            _, crop, irr = crop_prop.split('-')
            combi_crop = crop
            if combine_subcrops: # applies to ISIMIP3b: sum yield for sub crop types
                if crop in ('ri2', 'wwh'):
                    # skip ri2 (2nd harvest rice) and wwh (winter wheat) at this step
                    continue
                if crop == 'ri1': # first harvest rice
                    combi_crop = 'ric'
                if crop == 'swh': # spring wheat
                    combi_crop = 'whe'
            if scenario == 'historical':
                his_file_list.append(file)
                if (yearrange_his is None) and (isimip_run == 'ISIMIP2b'):
                    yearrange_his = YEARCHUNKS[scenario]['yearrange']
                elif (yearrange_his is None) and (isimip_run == 'ISIMIP3b'):
                    yearrange_his = YEARCHUNKS['historical_ISIMIP3b']['yearrange']
                startyear, endyear = yearrange_his
                file_props[file] = {'ag_model': ag_model, 'cl_model': cl_model,
                                    'bias_corr': bias_corr, 'soc': soc,
                                    'scenario': scenario, 'co2':co2, 'crop': crop, 'irr': irr,
                                    'startyear': startyear, 'endyear': endyear,
                                    'crop_irr': f'{crop}-{irr}', 'combi_crop': combi_crop,
                                    'combi_crop_irr': f'{combi_crop}-{irr}'
                                    }
            elif scenario not in scenario_list:
                scenario_list.append(scenario)

        elif isimip_run == 'ISIMIP2a':
            (ag_model, cl_model, biasco, scenario, harm, irr, _, crop, _, _,
             startyear, endyearnc) = file.split('_')
            combi_crop = crop
            endyear, _ = endyearnc.split('.')
            if yearrange_his is not None:
                startyear, endyear = (YEARCHUNKS[scenario])['yearrange']

            file_props[file] = dict()
            file_props[file] = {'ag_model': ag_model, 'cl_model': cl_model, 'scenario': 'ISIMIP2a',
                                'bc':biasco, 'harm':harm, 'crop': crop, 'irr': irr,
                                'crop_irr': f'{crop}-{irr}', 'startyear': int(startyear),
                                'endyear': int(endyear), 'combi_crop': combi_crop,
                                'combi_crop_irr': f'{combi_crop}-{irr}'}
            his_file_list.append(file)
        elif isimip_run == 'test_file':
            ag_model, cl_model, _, _, soc, co2, crop_prop, *_ = file.split('_')
            _, crop, irr = crop_prop.split('-')
            combi_crop = crop
            his_file_list.append(file)
            startyear, endyear = yearrange_his
            file_props[file] = {'ag_model': ag_model, 'cl_model': cl_model, 'soc':soc,
                                'scenario': 'test_file', 'co2':co2, 'crop': crop, 'irr': irr,
                                'startyear': startyear, 'endyear': endyear,
                                'crop_irr': f'{crop}-{irr}', 'combi_crop': combi_crop,
                                'combi_crop_irr': f'{combi_crop}-{irr}'}
        else:
            raise ValueError(f'Invalid value for isimip_run: {isimip_run}')

        if f'{crop}-{irr}' not in crop_irr_list:
            crop_irr_list.append(f'{crop}-{irr}')
        if f'{combi_crop}-{irr}' not in combi_crop_irr_list:
            combi_crop_irr_list.append(f'{combi_crop}-{irr}')

    # generate hazard using the first file to determine the size of the historic mean
    # file structure: ag_model _ cl_model _ scenario _ soc _ co2 _
    #   yield-crop-irr _ fn_str_var _ startyear _ endyear . nc
    #e.g. gepic_gfdl-esm2m_ewembi_historical_2005soc_co2_yield-whe-noirr_
    #   global_annual_1861_2005.nc
    haz_dummy = RelativeCropyield()
    haz_dummy.set_from_isimip_netcdf(input_dir=input_dir, filename=his_file_list[0], bbox=bbox,
                                     scenario=file_props[his_file_list[0]]['scenario'],
                                     yearrange=(file_props[his_file_list[0]]['startyear'],
                                                file_props[his_file_list[0]]['endyear']))

    # initiate the historic mean for each combination of crop and irrigation type
    # the idx keeps track of the row in which the hist_mean values are written per crop-irr to
    # ensure that all files are assigned to the corresponding crop-irr combination
    hist_mean_per_crop = dict()
    for i, combi_crop_irr in enumerate(combi_crop_irr_list):
        crop, irr = crop_irr_list[i].split('-')
        amount_crop_irr = sum((crop in s) and (irr in s) for s in his_file_list)
        hist_mean_per_crop[combi_crop_irr] = dict()
        hist_mean_per_crop[combi_crop_irr] = {
            'value': np.zeros([amount_crop_irr, haz_dummy.intensity.shape[1]]),
            'idx': 0}

    return his_file_list, file_props, hist_mean_per_crop, scenario_list, \
        crop_irr_list, combi_crop_irr_list

def calc_his_haz_isimip(his_file, file_props, input_dir=None, bbox=None,
                        yearrange_mean=None):
    """Create historical hazard and calculate historical mean.

        Parameters:
            his_file (string): file name of historical input hazard file
            file_props (dict): file properties of all historical input hazard files
            input_dir (Path): path to input data directory, default: INPUT_DIR
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange_mean (int tuple): year range for the historical mean
                default: 1976 - 2005

        Return:
            haz_his (RelativeCropyield): historical hazard
            filename (string): name to save historical hazard
            hist_mean (array): historical mean of the historical hazard
    """
    if input_dir is None:
        input_dir = Path(INPUT_DIR)
    if bbox is None:
        bbox = BBOX

    haz_his = RelativeCropyield()
    haz_his.set_from_isimip_netcdf(input_dir=input_dir, filename=his_file, bbox=bbox,
                                   scenario=file_props[his_file]['scenario'],
                                   yearrange=np.array([file_props[his_file]['startyear'],
                                                       file_props[his_file]['endyear']]))
    crop = file_props[his_file]['crop']
    # combine subcrops if crop and combi_crop not the same:
    if crop != file_props[his_file]['combi_crop']:
        if crop in ('ri2', 'wwh'):
            raise ValueError('Invalid subcrop type before combination')
        elif crop == 'swh': # whe = swh+wwh
            his_file2 = his_file.replace('_yield-swh-', '_yield-wwh-')
        elif crop == 'ri1': # ric = ri1+ri2
            his_file2 = his_file.replace('_yield-ri1-', '_yield-ri2-')
        else:
            his_file2 = None
        if his_file2 and (input_dir / his_file2).is_file():
            haz_his2 = RelativeCropyield()
            haz_his2.set_from_isimip_netcdf(input_dir=input_dir, filename=his_file2, bbox=bbox,
                                            scenario=file_props[his_file]['scenario'],
                                            yearrange=np.array([file_props[his_file]['startyear'],
                                                                file_props[his_file]['endyear']]))
            # The masks in the NetCDF file divides all wheat growing areas globally in two distinct
            # categories, either growing winter wheat (wwh) or spring wheat (swh). Since the hazard
            # sets for wwh and swh are calculated for the whole globe ("all crops everywhere") we
            # need to decide for each grid cell which one to take when combining 'wwh' and 'swh'
            # into one combined crop wheat (whe):
            if crop == 'swh':
            # mask of winter wheat in spring wheat and vice versa:
                whe_mask = read_wheat_mask_isimip3(input_dir=input_dir, bbox=bbox)
                haz_his.intensity = sparse.csr_matrix(np.multiply(haz_his.intensity.todense(),
                                                                  whe_mask.swh_mask.values.flatten()
                                                                  )
                                                      )
                haz_his2.intensity = sparse.csr_matrix(
                    np.multiply(haz_his2.intensity.todense(), whe_mask.wwh_mask.values.flatten()
                                ))
            # replace NaN by 0.0:
            haz_his.intensity.data[np.isnan(haz_his.intensity.data)] = 0.0
            haz_his2.intensity.data[np.isnan(haz_his2.intensity.data)] = 0.0
            # sum intensities of subcrops while intensity is still abs. yield:
            haz_his.intensity = haz_his.intensity + haz_his2.intensity
            haz_his.crop = file_props[his_file]['combi_crop']
    # hazard intensity is transformed from absolute yield [t / (ha * yr)]
    # to yield relative to historical mean of same run (fractional yield):
    hist_mean = haz_his.calc_mean(yearrange_mean)
    haz_his.set_rel_yield_to_int(hist_mean)

    crop_irr = file_props[his_file]['combi_crop_irr']
    if file_props[his_file]['scenario'] == 'ISIMIP2a':
        filename = ('haz' + '_' + file_props[his_file]['ag_model'] + '_' +
                    file_props[his_file]['cl_model'] +'_' + file_props[his_file]['bc'] +
                    '_' + file_props[his_file]['harm'] + '_' + crop_irr + '_' +
                    str(file_props[his_file]['startyear']) + '-' +
                    str(file_props[his_file]['endyear']) + '.hdf5')
    else:
        filename = ('haz' + '_' + file_props[his_file]['ag_model'] + '_' +
                    file_props[his_file]['cl_model'] + '_' + file_props[his_file]['scenario'] +
                    '_' + file_props[his_file]['soc'] + '_' + file_props[his_file]['co2'] +
                    '_' + crop_irr + '_' + str(file_props[his_file]['startyear']) + '-' +
                    str(file_props[his_file]['endyear']) + '.hdf5')

    return haz_his, filename, hist_mean

def calc_fut_haz_isimip(his_file, scenario, file_props, hist_mean, input_dir=None,
                        bbox=None, fut_file=None, yearrange_fut=None):
    """Create future hazard.

        Parameters:
            his_file (string): file name of historical input hazard file
            scenario (string): future scenario, e.g. rcp60
            file_props (dict): file properties of all historical input hazard files
            hist_mean (array): historical mean of the historical hazard for the same model
                combination and crop-irr cobination

        Optional Parameters:
            input_dir (Path): path to input data directory, default: INPUT_DIR
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            fut_file (string): file name of future input hazard file. If given,
                prefered over scenario and file_props
            yearrange_fut (tuple): start and end year to be extracted
                For None (default) yearrange_fut is set from filename or YEARCHUNKS

        Return:
            haz_fut (RelativeCropyield): future hazard
            filename (string): name to save future hazard
    """
    if input_dir is None:
        input_dir = Path(INPUT_DIR)
    if bbox is None:
        bbox = BBOX
    if yearrange_fut is None:
        if isinstance(fut_file, str) and (len(fut_file.split('_')[-2]) == 4):
            yearrange_fut = (int(fut_file.split('_')[-2]),
                             int(fut_file.split('_')[-1].split('.')[0])
                             )
        else: # define yearrange from defaults if not specified by user or filename:
            yearrange_fut = (YEARCHUNKS[scenario]['startyear'],
                             YEARCHUNKS[scenario]['endyear'])

    startyear, endyear = yearrange_fut

    haz_fut = RelativeCropyield()
    haz_fut.set_from_isimip_netcdf(input_dir=input_dir, filename=fut_file,
                                   bbox=bbox, yearrange=yearrange_fut,
                                   ag_model=file_props[his_file]['ag_model'],
                                   cl_model=file_props[his_file]['cl_model'],
                                   bias_corr=file_props[his_file]['bias_corr'],
                                   scenario=scenario,
                                   soc=file_props[his_file]['soc'],
                                   co2=file_props[his_file]['co2'],
                                   crop=file_props[his_file]['crop'],
                                   irr=file_props[his_file]['irr'])

    crop = file_props[his_file]['crop']
    # combine subcrops if crop and combi_crop not the same:
    if crop != file_props[his_file]['combi_crop']:
        if crop in ('ri2', 'wwh'):
            raise ValueError('Invalid subcrop type before combination')
        elif crop == 'swh': # whe = swh+wwh
            fut_file2 = fut_file.replace('_yield-swh-', '_yield-wwh-')
        elif crop == 'ri1': # ric = ri1+ri2
            fut_file2 = fut_file.replace('_yield-ri1-', '_yield-ri2-')
        else:
            fut_file2 = None
        if fut_file2 and Path(input_dir, fut_file2).is_file():
            haz_fut2 = RelativeCropyield()
            haz_fut2.set_from_isimip_netcdf(input_dir=input_dir, filename=fut_file2,
                                            bbox=bbox, yearrange=yearrange_fut,
                                            ag_model=file_props[his_file]['ag_model'],
                                            cl_model=file_props[his_file]['cl_model'],
                                            bias_corr=file_props[his_file]['bias_corr'],
                                            scenario=scenario,
                                            soc=file_props[his_file]['soc'],
                                            co2=file_props[his_file]['co2'],
                                            crop=file_props[his_file]['crop'],
                                            irr=file_props[his_file]['irr'])
            if crop == 'swh':
            # mask of winter wheat in spring wheat and vice versa:
                whe_mask = read_wheat_mask_isimip3(input_dir=input_dir, bbox=bbox)
                haz_fut.intensity = sparse.csr_matrix(np.multiply(haz_fut.intensity.todense(),
                                                                  whe_mask.swh_mask.values.flatten()
                                                                  )
                                                      )
                haz_fut2.intensity = sparse.csr_matrix(
                    np.multiply(haz_fut2.intensity.todense(), whe_mask.wwh_mask.values.flatten()
                                ))
            # replace NaN by 0.0:
            haz_fut.intensity.data[np.isnan(haz_fut.intensity.data)] = 0.0
            haz_fut2.intensity.data[np.isnan(haz_fut2.intensity.data)] = 0.0
            # sum intensities of subcrops while intensity is still abs. yield:
            haz_fut.intensity = haz_fut.intensity + haz_fut2.intensity
            haz_fut.crop = file_props[his_file]['combi_crop']

    haz_fut.set_rel_yield_to_int(hist_mean) # set intensity to relative yield
    filename = ('haz' + '_' + file_props[his_file]['ag_model'] + '_' +
                file_props[his_file]['cl_model'] + '_' + scenario + '_' +
                file_props[his_file]['soc'] + '_' + file_props[his_file]['co2'] +
                '_' + file_props[his_file]['combi_crop'] + '-' + file_props[his_file]['irr']+ '_' +
                str(startyear) + '-' + str(endyear) + '.hdf5')
    return haz_fut, filename

def read_wheat_mask_isimip3(input_dir=None, filename=None, bbox=None):
    """for ISIMIP3, get masks for spring wheat (swh) and winter wheat (wwh).
    Required in set_multiple_rc_from_isimip() if isimip_version is ISIMIP3b and
    combine_crops is True.

    Optional Parameters:
        input_dir (Path or str): path to directory containing input file, default: INPUT_DIR
        filename (str): name of file
        bbox (tuple): geogr. bounding box, tuple or array with for elements.

    Returns:
        whe_mask (xarray)"""

    if input_dir is None:
        input_dir = Path(INPUT_DIR)
    if filename is None:
        filename = CONFIG.hazard.relative_cropyield.filename_wheat_mask.str()
    if bbox is None:
        bbox = BBOX

    whe_mask = xr.open_dataset(Path(input_dir, filename), decode_times=False)
    [lonmin, latmin, lonmax, latmax] = bbox
    return whe_mask.sel(lon=slice(lonmin, lonmax), lat=slice(latmax, latmin))

def plot_comparing_maps(haz_his, haz_fut, axes=None, nr_cli_models=1, model=1):
    """Plots comparison maps of historic and future data and their difference fut-his

    Parameters:
        haz_his (RelativeCropyield): historic hazard
        haz_fut (RelativeCropyield): future hazard
        axes (Geoaxes): subplot axes that are generated if not given
            (sets the figure size depending on the extent of the single plots and
             the amount of rows)
        nr_cli_models (int): number of climate models and respectively nr of rows within
            the subplot
        model (int): current row to plot - this method can be used in a loop to plot
            subplots in one figure consisting of several rows of subplots.
            One row displays the intensity for present and future climate and the difference of
            the two for one model-combination (ag_model and cl_model)

    Returns:
        figure, geoaxes
    """


    if axes is None:
        len_lat = (np.max(haz_his.centroids.lat)-np.min(haz_his.centroids.lat))*(2.5/13.5)
        len_lon = (np.max(haz_his.centroids.lon)-np.min(haz_his.centroids.lon))*(5/26)

        fig, axes = plt.subplots(nr_cli_models, 3, figsize=(3*len_lon, nr_cli_models*len_lat), \
                                 subplot_kw=dict(projection=cartopy.crs.PlateCarree()))

        for subplot in range(3*nr_cli_models):
            axes.flat[subplot].set_extent([np.min(haz_his.centroids.lon),
                                           np.max(haz_his.centroids.lon),
                                           np.min(haz_his.centroids.lat),
                                           np.max(haz_his.centroids.lat)])

    haz2plot = RelativeCropyield()
    haz2plot = copy.deepcopy(haz_his)
    haz2plot.event_id = 0

    his_mean = sparse.csr_matrix(haz_his.intensity.mean(axis=0))
    fut_mean = sparse.csr_matrix(haz_fut.intensity.mean(axis=0))

    for subplot in range(3):

        if subplot == 0:
            haz2plot.intensity = his_mean
        elif subplot == 1:
            haz2plot.intensity = fut_mean
        elif subplot == 2:
            haz2plot.intensity = fut_mean - his_mean

        if nr_cli_models == 1:
            ax1 = haz2plot.plot_intensity_cp(event=0, dif=0, axis=axes[subplot])
        else:
            ax1 = haz2plot.plot_intensity_cp(event=0, dif=1, axis=axes[model, subplot])

        ax1.set_title('')

    if nr_cli_models == 1:
        cols = ['Historical', 'Future', 'Difference = Future - Historical']
        for ax0, col in zip(axes, cols):
            ax0.set_title(col, size='large')

    return fig, axes
