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

__all__ = ['RelativeCropyield']

import logging
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
import cartopy
import shapely.geometry
from scipy import sparse
import scipy.stats
import h5py


from climada.hazard.base import Hazard
from climada.util import dates_times as dt
from climada.util import coordinates as coord
from climada.util.constants import DATA_DIR


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RC'
"""Hazard type acronym for Relative Cropyield"""

INT_DEF = 'Yearly Yield'

BBOX = np.array([-180, -85, 180, 85])  # [Lon min, lat min, lon max, lat max]
""""Default geographical bounding box of the total global agricultural land extent"""

# ! deposit the input files in: climada_python/data/ISIMIP_crop/Input/Hazard
INPUT_DIR = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Input', 'Hazard')
"""default paths for input and output data:"""
OUTPUT_DIR = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Output')


#ISIMIP input data specific global variables
YEARCHUNKS = dict()
"""start and end years per senario as in ISIMIP-filenames"""
YEARCHUNKS['ISIMIP2a'] = dict()
YEARCHUNKS['ISIMIP2a'] = {'yearrange': np.array([1980, 1999]), 'startyear': 1980,
                          'endyear': 1999, 'yearrange_mean': np.array([1980, 1999])}
YEARCHUNKS['historical'] = dict()
YEARCHUNKS['historical'] = {'yearrange': np.array([1976, 2005]), 'startyear': 1861,
                            'endyear': 2005, 'yearrange_mean': np.array([1976, 2005])}
YEARCHUNKS['rcp60'] = dict()
YEARCHUNKS['rcp60'] = {'yearrange': np.array([2006, 2099]), 'startyear': 2006,
                       'endyear': 2099}

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

    def set_from_single_run(self, input_dir=None, filename=None, bbox=BBOX,
                            yearrange=(YEARCHUNKS['historical'])['yearrange'],
                            ag_model=None, cl_model=None, scenario='historical',
                            soc=None, co2=None, crop=None, irr=None, fn_str_var=FN_STR_VAR):

        """Wrapper to fill hazard from nc_dis file from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for hazard set, f.i. (1976, 2005)
            ag_model (str): abbrev. agricultural model (only when input_dir is selected)
                f.i. 'clm-crop', 'gepic','lpjml','pepic'
            cl_model (str): abbrev. climate model (only when input_dir is selected)
                f.i. ['gfdl-esm2m', 'hadgem2-es','ipsl-cm5a-lr','miroc5'
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
        if input_dir is not None:
            if not os.path.exists(input_dir):
                LOGGER.error('Input directory %s does not exist', input_dir)
                raise NameError
        else:
            LOGGER.error('Input directory %s not set', input_dir)
            raise NameError


        # The filename is set or other variables (cl_model, scenario) are extracted of the
        # specified filename
        if filename is None:
            yearchunk = YEARCHUNKS[scenario]
            filename = os.path.join(input_dir, '%s_%s_ewembi_%s_%s_%s_yield-%s-%s_%s_%s_%s.nc' \
                                    %(ag_model, cl_model, scenario, soc, co2, crop,
                                      irr, fn_str_var, str(yearchunk['startyear']),
                                      str(yearchunk['endyear'])))

        elif scenario == 'ISIMIP2a':
            (_, _, _, _, _, _, _, crop, _, _, startyear, endyearnc) = filename.split('_')
            endyear, _ = endyearnc.split('.')
            yearchunk = dict()
            yearchunk = {'yearrange': np.array([int(startyear), int(endyear)]),
                         'startyear': int(startyear), 'endyear': int(endyear)}
            filename = os.path.join(input_dir, filename)
        elif scenario == 'test_file':
            yearchunk = dict()
            yearchunk = {'yearrange': np.array([1976, 2005]), 'startyear': 1861,
                         'endyear': 2005, 'yearrange_mean': np.array([1976, 2005])}
            ag_model, cl_model, _, _, soc, co2, crop_prop, *_ = filename.split('_')
            _, crop, irr = crop_prop.split('-')
            filename = os.path.join(input_dir, filename)
        else:
            yearchunk = YEARCHUNKS[scenario]
            (_, _, _, _, _, _, crop_irr, *_) = filename.split('_')
            _, crop, irr = crop_irr.split('-')
            filename = os.path.join(input_dir, filename)




        # define indexes of the netcdf-bands to be extracted, and the
        # corresponding event names and dates
        # corrected indexes due to the bands in input starting with the index=1
        id_bands = np.arange(yearrange[0] - yearchunk['startyear'] + 1,
                             yearrange[1] - yearchunk['startyear'] + 2).tolist()

        # hazard setup: set attributes
        [lonmin, latmin, lonmax, latmax] = bbox
        self.set_raster([filename], band=id_bands,
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

    def calc_mean(self, yearrange_mean=(YEARCHUNKS['historical'])['yearrange_mean'],
                  save=False, output_dir=OUTPUT_DIR):
        """Calculates mean of the hazard for a given reference time period

            Optional Parameters:
            yearrange_mean (array): time period used to calculate the mean intensity
                default: 1976-2005 (historical)
            save (boolean): save mean to file? default: False
            output_dir (str): path of output directory

            Returns:
                hist_mean(array): contains mean value over the given reference
                    time period for each centroid
        """
        startyear, endyear = yearrange_mean
        event_list = [str(n) for n in range(int(startyear), int(endyear + 1))]
        mean = self.select(event_names=event_list).intensity.mean(axis=0)
        hist_mean = np.squeeze(np.asarray(mean))

        if save:
            # generate output directories if they do not exist yet
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            mean_dir = os.path.join(output_dir, 'Hist_mean')
            if not os.path.exists(mean_dir):
                os.mkdir(mean_dir)
            # save mean_file
            mean_file = h5py.File(mean_dir + 'hist_mean_' + self.crop + '_' + str(startyear) +
                                  '-' + str(endyear) + '.hdf5', 'w')
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

    def plot_comparing_maps(self, his, fut, axes, nr_cli_models=1, model=1):
        """Plots comparison maps of historic and future data and their difference fut-his

        Parameters:
            his (sparse matrix): historic mean annual yield or mean relative yield
            fut (sparse matrix): future mean annual yield or mean relative yield
            axes (Geoaxes): subplot axes that can be generated with ag_drought_util.setup_subplots
            nr_cli_models (int): number of climate models and respectively nr of rows within
                                    the subplot
            model (int): current row to plot - this method can be used in a loop to plot
                subplots in one figure consisting of several rows of subplots.
                One row displays the intensity for present and future climate and the difference of
                the two for one model-combination (ag_model and cl_model)


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
                ax1 = self.plot_intensity_cp(event=0, dif=dif_def, axis=axes[subplot])
            else:
                ax1 = self.plot_intensity_cp(event=0, dif=dif_def, axis=axes[model, subplot])

            ax1.set_title('')

        if nr_cli_models == 1:
            cols = ['Historical', 'Future', 'Difference = Future - Historical']
            for ax0, col in zip(axes, cols):
                ax0.set_title(col, size='large')

        return axes


def generate_full_hazard_set(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, bbox=BBOX,
                             isimip_run='ISIMIP2b', yearrange_his=None, yearrange_mean=None,
                             return_data=False, save=True):

    """Wrapper to generate full hazard set and save it to output directory.

        Optional Parameters:
            input_dir (string): path to input data directory
            output_dir (string): path to output data directory
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            isimip_run (string): name of the ISIMIP run (ISIMIP2a or ISIMIP2b)
            yearrange_his (int tuple): year range for the historical hazard sets
            yearrange_mean (int tuple): year range for the historical mean
            return_data (boolean): returned output
                False: returns list of filenames only
                True: returns also list of data
            save (boolean): save output data to output_dir

        Return:
            filename_list (list): list of filenames

        Optional Return:
            output_list (list): list of generated output data (hazards and historical mean)

    """

    filenames = [f for f in listdir(input_dir) if (isfile(join(input_dir, f))) if not
                 f.startswith('.')]


    # generate output directories if they do not exist yet
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'Hazard')):
        os.mkdir(os.path.join(output_dir, 'Hazard'))
    if not os.path.exists(os.path.join(output_dir, 'Hist_mean')):
        os.mkdir(os.path.join(output_dir, 'Hist_mean'))

    filename_list = list()
    output_list = list()

    (his_file_list, file_props, hist_mean_per_crop,
     scenario_list, crop_list) = init_hazard_set(filenames, input_dir, bbox, isimip_run,
                                                 yearrange_his)

    if yearrange_mean is None:
        yearrange_mean = (YEARCHUNKS[(file_props[his_file_list[0]])['scenario']])['yearrange_mean']

    for his_file in his_file_list:
        haz_his, filename, hist_mean = calc_his_haz(his_file, file_props, input_dir, bbox,
                                                    yearrange_mean)
        # save the historical mean depending on the crop-irrigation combination
        # the idx keeps track of the row in which the hist_mean values are written per crop-irr to
        # ensure that all files are assigned to the corresponding crop-irr combination
        hist_mean_per_crop[(file_props[his_file])['crop_irr']]['value'][
            hist_mean_per_crop[(file_props[his_file])['crop_irr']]['idx'], :] = hist_mean
        hist_mean_per_crop[file_props[his_file]['crop_irr']]['idx'] += 1

        filename_list.append(filename)
        output_list.append(haz_his)


        if isimip_run == 'ISIMIP2b':
            # compute the relative yield for all future scenarios with the corresponding
            # historic mean
            for scenario in scenario_list:
                haz_fut, filename = calc_fut_haz(his_file, scenario, file_props, hist_mean,
                                                 input_dir, bbox)
                filename_list.append(filename)
                output_list.append(haz_fut)

    # calculate mean hist_mean for each crop-irrigation combination and save as hdf5 in output_dir
    for crop_irr in crop_list:
        mean = np.mean((hist_mean_per_crop[crop_irr])['value'], 0)
        mean_filename = ('hist_mean_' + crop_irr + '_' + str(yearrange_mean[0]) +'-' +
                         str(yearrange_mean[1]) + '.hdf5')
        filename_list.append(mean_filename)
        output_list.append(mean)

    if save:
        for idx, filename in enumerate(filename_list):
            if 'haz' in filename:
                output_list[idx].select(reg_id=1).write_hdf5(os.path.join(output_dir,
                                                                          'Hazard', filename))
            elif 'mean' in filename:
                mean_file = h5py.File(os.path.join(output_dir, 'Hist_mean', filename), 'w')
                mean_file.create_dataset('mean', data=output_list[idx])
                mean_file.create_dataset('lat', data=haz_his.centroids.lat)
                mean_file.create_dataset('lon', data=haz_his.centroids.lon)
                mean_file.close()
                # save historic mean as netcdf (saves mean, lat and lon as arrays)
                # mean_file = xr.Dataset({'mean': mean, 'lat': haz_his.centroids.lat, \
                #                         'lon': haz_his.centroids.lon})
                # mean_file.to_netcdf(mean_dir+'hist_mean_'+crop_irr+'.nc')

    if not return_data:
        return filename_list
    return filename_list, output_list

def init_hazard_set(filenames, input_dir=INPUT_DIR, bbox=BBOX, isimip_run='ISIMIP2b',
                    yearrange_his=None):

    """Initialize fulll hazard set.

        Parameters:
            filenames (list): list of filenames
            input_dir (string): path to input data directory
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            isimip_run (string): name of the ISIMIP run (ISIMIP2a or ISIMIP2b)
            yearrange_his (int tuple): year range for the historical hazard sets

        Return:
            his_file_list (list): list of historical input hazard files
            file_props (dict): file properties of all historical input hazard files
            hist_mean_per_crop (dict): empty dictonary to save hist_mean values for each
                crop-irr combination
            scenario_list (list): list of all future scenarios
            crop_list (list): list of all crop-irr combinations

    """

    crop_list = list()
    file_props = dict()
    his_file_list = list()
    scenario_list = list()



    for file in filenames:
        if isimip_run == 'ISIMIP2b':
            ag_model, cl_model, _, scenario, soc, co2, crop_prop, *_ = file.split('_')
            _, crop, irr = crop_prop.split('-')
            if 'historical' in file:
                his_file_list.append(file)
                if yearrange_his is None:
                    yearrange_his = (YEARCHUNKS[scenario])['yearrange']
                startyear, endyear = yearrange_his
                file_props[file] = {'ag_model': ag_model, 'cl_model': cl_model, 'soc':soc,
                                    'scenario': scenario, 'co2':co2, 'crop': crop, 'irr': irr,
                                    'startyear': startyear, 'endyear': endyear,
                                    'crop_irr': crop+'-'+irr}
            elif scenario not in scenario_list:
                scenario_list.append(scenario)

        elif isimip_run == 'ISIMIP2a':
            (ag_model, cl_model, biasco, scenario, harm, irr, _, crop, _, _,
             startyear, endyearnc) = file.split('_')
            endyear, _ = endyearnc.split('.')
            if yearrange_his is not None:
                startyear, endyear = (YEARCHUNKS[scenario])['yearrange']

            file_props[file] = dict()
            file_props[file] = {'ag_model': ag_model, 'cl_model': cl_model, 'scenario': 'ISIMIP2a',
                                'bc':biasco, 'harm':harm, 'crop': crop, 'irr': irr,
                                'crop_irr': crop+'-'+irr, 'startyear': int(startyear),
                                'endyear': int(endyear)}
            his_file_list.append(file)
        elif isimip_run == 'test_file':
            ag_model, cl_model, _, _, soc, co2, crop_prop, *_ = file.split('_')
            _, crop, irr = crop_prop.split('-')
            his_file_list.append(file)
            startyear, endyear = yearrange_his
            file_props[file] = {'ag_model': ag_model, 'cl_model': cl_model, 'soc':soc,
                                'scenario': 'test_file', 'co2':co2, 'crop': crop, 'irr': irr,
                                'startyear': startyear, 'endyear': endyear,
                                'crop_irr': crop+'-'+irr}

        crop_irr = crop + '-' + irr
        if crop_irr not in crop_list:
            crop_list.append(crop_irr)

    # generate hazard using the first file to determine the size of the historic mean
    # file structure: ag_model _ cl_model _ scenario _ soc _ co2 _
    #   yield-crop-irr _ fn_str_var _ startyear _ endyear . nc
    #e.g. gepic_gfdl-esm2m_ewembi_historical_2005soc_co2_yield-whe-noirr_
    #   global_annual_1861_2005.nc
    haz_dummy = RelativeCropyield()
    haz_dummy.set_from_single_run(input_dir=input_dir, filename=his_file_list[0], bbox=bbox,
                                  scenario=(file_props[his_file_list[0]])['scenario'],
                                  yearrange=np.array([(file_props[his_file_list[0]])['startyear'],
                                                      (file_props[his_file_list[0]])['endyear']]))

    # initiate the historic mean for each combination of crop and irrigation type
    # the idx keeps track of the row in which the hist_mean values are written per crop-irr to
    # ensure that all files are assigned to the corresponding crop-irr combination
    hist_mean_per_crop = dict()
    for crop_irr in crop_list:
        amount_crop_irr = sum(crop_irr in s for s in his_file_list)
        hist_mean_per_crop[crop_irr] = dict()
        hist_mean_per_crop[crop_irr] = {
            'value': np.zeros([amount_crop_irr, haz_dummy.intensity.shape[1]]),
            'idx': 0}

    return his_file_list, file_props, hist_mean_per_crop, scenario_list, crop_list
    # if isimip_run == 'ISIMIP2a':
    #     return crop_list, his_file_list, yearrange_list, hist_mean_per_crop, file_props
    # return crop_list, his_file_list, scenario_list, hist_mean_per_crop, file_props

def calc_his_haz(his_file, file_props, input_dir=INPUT_DIR, bbox=BBOX, yearrange_mean=None):

    """Create historical hazard and calculate historical mean.

        Parameters:
            his_file (string): file name of historical input hazard file
            file_props (dict): file properties of all historical input hazard files
            input_dir (string): path to input data directory
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange_mean (int tuple): year range for the historical mean
                default: 1976 - 2005


        Return:
            haz_his (RelativeCropyield): historical hazard
            filename (string): name to save historical hazard
            hist_mean (array): historical mean of the historical hazard

    """


    haz_his = RelativeCropyield()
    haz_his.set_from_single_run(input_dir=input_dir, filename=his_file, bbox=bbox,
                                scenario=(file_props[his_file])['scenario'],
                                yearrange=np.array([(file_props[his_file])['startyear'],
                                                    (file_props[his_file])['endyear']]))

    hist_mean = haz_his.calc_mean(yearrange_mean)
    haz_his.set_rel_yield_to_int(hist_mean)

    crop_irr = (file_props[his_file])['crop'] + '-' + (file_props[his_file])['irr']
    if (file_props[his_file])['scenario'] == 'ISIMIP2a':
        filename = ('haz' + '_' + (file_props[his_file])['ag_model'] + '_' +
                    (file_props[his_file])['cl_model'] +'_' + (file_props[his_file])['bc'] +
                    '_' + (file_props[his_file])['harm'] + '_' + crop_irr + '_' +
                    str((file_props[his_file])['startyear']) + '-' +
                    str((file_props[his_file])['endyear']) + '.hdf5')
    else:
        filename = ('haz' + '_' + (file_props[his_file])['ag_model'] + '_' +
                    (file_props[his_file])['cl_model'] + '_' + (file_props[his_file])['scenario'] +
                    '_' + (file_props[his_file])['soc'] + '_' + (file_props[his_file])['co2'] +
                    '_' + crop_irr + '_' + str((file_props[his_file])['startyear']) + '-' +
                    str((file_props[his_file])['endyear']) + '.hdf5')


    return haz_his, filename, hist_mean

def calc_fut_haz(his_file, scenario, file_props, hist_mean, input_dir=INPUT_DIR, bbox=BBOX):

    """Create future hazard.

        Parameters:
            his_file (string): file name of historical input hazard file
            scenario (string): future scenario, e.g. rcp60
            file_props (dict): file properties of all historical input hazard files
            hist_mean (array): historical mean of the historical hazard for the same model
                combination and crop-irr cobination
            input_dir (string): path to input data directory
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]

        Return:
            haz_fut (RelativeCropyield): future hazard
            filename (string): name to save historical hazard


    """

    yearrange_fut = np.array([(YEARCHUNKS[scenario])['startyear'],
                              (YEARCHUNKS[scenario])['endyear']])
    startyear, endyear = yearrange_fut
    haz_fut = RelativeCropyield()
    haz_fut.set_from_single_run(input_dir=input_dir, bbox=bbox, yearrange=yearrange_fut,
                                ag_model=(file_props[his_file])['ag_model'],
                                cl_model=(file_props[his_file])['cl_model'],
                                scenario=scenario,
                                soc=(file_props[his_file])['soc'],
                                co2=(file_props[his_file])['co2'],
                                crop=(file_props[his_file])['crop'],
                                irr=(file_props[his_file])['irr'])
    haz_fut.set_rel_yield_to_int(hist_mean)
    filename = ('haz' + '_' + (file_props[his_file])['ag_model'] + '_' +
                (file_props[his_file])['cl_model'] + '_' + scenario + '_' +
                (file_props[his_file])['soc'] + '_' + (file_props[his_file])['co2'] +
                '_' + (file_props[his_file])['crop'] + '-' + (file_props[his_file])['irr']+ '_' +
                str(startyear) + '-' + str(endyear) + '.hdf5')

    return haz_fut, filename
