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
from climada.util import coordinates
from climada.util.constants import DATA_DIR


LOGGER = logging.getLogger(__name__)

HAZ_TYPE = 'RC'
"""Hazard type acronym for Relative Cropyield"""

AG_MODEL = ['gepic',
            'lpjml',
            'pepic'
            ]
"""crop model names as in ISIMIP-filenames"""

CL_MODEL = ['gfdl-esm2m',
            'hadgem2-es',
            'ipsl-cm5a-lr',
            'miroc5'
            ]
"""climate model names as in ISIMIP-filenames"""

SCENARIO = ['historical',
            'rcp60'
            ]
"""climate scenario names as in ISIMIP-filenames"""

SOC = ['2005soc',
       'histsoc'
       ]
"""socio-economic forcing settings as in ISIMIP-filenames"""

CO2 = ['co2',
       '2005co2'
       ]
"""CO2 forcing settings as in ISIMIP-filenames"""

CROP = ['whe',
        'mai',
        'soy',
        'ric'
       ]
"""crop types as in ISIMIP-filenames"""

IRR = ['noirr',
       'irr']
"""non-irrigated/irrigated as in ISIMIP-filenames"""

FN_STR_VAR = 'global_annual'
"""filename of ISIMIP output constant part"""

YEARCHUNKS = dict()
"""start and end years per senario as in ISIMIP-filenames"""
YEARCHUNKS[SCENARIO[0]] = dict()
YEARCHUNKS[SCENARIO[0]] = {'yearrange': np.array([1976, 2005]), 'startyear': 1861, 'endyear': 2005}
YEARCHUNKS[SCENARIO[1]] = dict()
YEARCHUNKS[SCENARIO[1]] = {'yearrange': np.array([2006, 2099]), 'startyear': 2006, 'endyear': 2099}

BBOX = np.array([-180, -85, 180, 85])  # [Lon min, lat min, lon max, lat max]
"""geographical bounding box in decimal degrees (lon from -180 to 180)"""

INT_DEF = 'Yearly Yield'

# ! deposit the input files in: climada_python/data/ISIMIP_crop/Input/Hazard
INPUT_DIR = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Input', 'Hazard')
"""default paths for input and output data:"""
OUTPUT_DIR = os.path.join(DATA_DIR, 'ISIMIP_crop', 'Output')

class RelativeCropyield(Hazard):
    """Agricultural climate risk: Relative Cropyield (relative to historical mean);
    Each year corresponds to one hazard event;
    Based on modelled crop yield, from ISIMIP (www.isimip.org, required input data).

    Attributes:
        crop_type (str): crop type (e.g. whe for wheat)
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

        self.crop = CROP[0]
        self.intensity_def = INT_DEF

    def set_from_single_run(self, input_dir=None, bbox=BBOX,
                            yearrange=(YEARCHUNKS[SCENARIO[0]])['yearrange'],
                            ag_model=AG_MODEL[0], cl_model=CL_MODEL[0],
                            scenario=SCENARIO[0], soc=SOC[0], co2=CO2[0],
                            crop=CROP[0], irr=IRR[0], fn_str_var=FN_STR_VAR):

        """Wrapper to fill hazard from nc_dis file from ISIMIP
        Parameters:
            input_dir (string): path to input data directory
            bbox (list of four floats): bounding box:
                [lon min, lat min, lon max, lat max]
            yearrange (int tuple): year range for hazard set, f.i. (1976, 2005)
            ag_model (str): abbrev. agricultural model (only when input_dir is selected)
                f.i. 'gepic' etc.
            cl_model (str): abbrev. climate model (only when input_dir is selected)
                f.i. 'gfdl-esm2m' etc.
            scenario (str): climate change scenario (only when input_dir is selected)
                f.i. 'historical' or 'rcp60'
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
                LOGGER.warning('Input directory %s does not exist', input_dir)
                raise NameError
        else:
            LOGGER.warning('Input directory %s not set', input_dir)
            raise NameError

        yearchunk = YEARCHUNKS[scenario]
        filename = os.path.join(input_dir,
                                '%s_%s_ewembi_%s_%s_%s_yield-%s-%s_%s_%s_%s.nc' \
                                    %(ag_model, cl_model, scenario, soc, co2, crop,
                                      irr, fn_str_var, str(yearchunk['startyear']),
                                      str(yearchunk['endyear'])))

        # define indexes of the netcdf-bands to be extracted, and the
        # corresponding event names and dates
        # corrected indexes due to the bands in input starting with the index=1
        id_bands = np.arange(yearrange[0] - yearchunk['startyear'] + 1,
                             yearrange[1] - yearchunk['startyear'] + 2).tolist()

        # hazard setup: set attributes
        self.set_raster([filename], band=id_bands,
                        geometry=list([shapely.geometry.box(bbox[0], bbox[1], bbox[2], bbox[3])]))

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
            coordinates.coord_on_land(self.centroids.lat, self.centroids.lon)).astype(dtype=int)
        self.check()
        return self

    def calc_mean(self, yearrange=(YEARCHUNKS[SCENARIO[0]])['yearrange'], save=False,
                  output_dir=OUTPUT_DIR):
        """Calculates mean of the hazard for a given reference time period

            Optional Parameters:
                yearrange (array): time period used to calculate the mean intensity
                default: 1976-2005 (historical)
            save (boolean): save mean to file? default: False
            output_dir (str): path of output directory

            Returns:
                hist_mean(array): contains mean value over the given reference
                    time period for each centroid
        """
        event_list = [str(n) for n in range(int(yearrange[0]), int(yearrange[1] + 1))]
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
            mean_file = h5py.File(mean_dir + 'hist_mean_' + self.crop + '_' + str(yearrange[0]) +
                                  '-' + str(yearrange[1]) + '.hdf5', 'w')
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
        idx = np.where(hist_mean != 0)[0]

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
            for event in range(nevents):
                value = self.intensity[event, centroid]
                hazard_matrix[event, centroid] = (scipy.stats.percentileofscore(array, value)
                                                  / 100)

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

        if event is None:
            event = self.event_name
        else:
            event = [str(n) for n in range(event[0], event[1] + 1)]

        self.centroids.set_meta_to_lat_lon()

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
                ax1 = self.plot_intensity_cp(event=0, dif=dif_def, axis=axes[subplot])
            else:
                ax1 = self.plot_intensity_cp(event=0, dif=dif_def, axis=axes[model, subplot])

            ax1.set_title('')

        if nr_cli_models == 1:
            cols = ['Historical', 'Future', 'Difference = Future - Historical']
            for ax0, col in zip(axes, cols):
                ax0.set_title(col, size='large')

        return axes


def init_full_hazard_set(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, bbox=BBOX,
                         yearrange=(YEARCHUNKS[SCENARIO[0]])['yearrange'],
                         return_data=False):
    """Generates hazard set for all files contained in the input directory and saves them
    as hdf5 files to the output directory.

        Optional Parameters:
        input_dir (string): path to input data directory
        output_dir (string): path to output data directory
        bbox (list of four floats): bounding box:
            [lon min, lat min, lon max, lat max]
        yearrange (int tuple): year range for hazard set, f.i. (2001, 2005)
        return_data (str): returned output
            False: returns list of filenames only
            True: returns also list of data

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

    # generate lists of splitted historical filenames (in order to differentiate betweeen
    # the used ag_model, cl_model, crop, etc.); list of the future scenarios; list
    # of all crop-irr combinations
    file_props = list()
    crop_list = list()
    scenario_list = list()
    for file in (filenames):
        file_prop = file.split('_')
        if file_prop[3] == 'historical':
            file_props.append(file_prop)
        elif file_prop[3] not in scenario_list:
            scenario_list.append(file_prop[3])
        crop_irr = ((file_prop[6]).split('-'))[1] + '-' + ((file_prop[6]).split('-'))[2]
        if crop_irr not in crop_list:
            crop_list.append(crop_irr)

    # generate hazard using the first file to determine the size of the historic mean
    # file structure: ag_model _ cl_model _ scenario _ soc _ co2 _
    #   yield-crop-irr _ fn_str_var _ startyear _ endyear . nc
    #e.g. gepic_gfdl-esm2m_ewembi_historical_2005soc_co2_yield-whe-noirr_
    #   global_annual_1861_2005.nc
    cp_zero = RelativeCropyield()
    cp_zero.set_from_single_run(input_dir=input_dir, bbox=bbox, yearrange=yearrange,
                                ag_model=(file_props[0])[0], cl_model=(file_props[0])[1],
                                scenario=(file_props[0])[3], soc=(file_props[0])[4], co2=(file_props[0])[5],
                                crop=(((file_props[0])[6]).split('-'))[1],
                                irr=(((file_props[0])[6]).split('-'))[2])
    hist_mean = cp_zero.calc_mean()
    cp_zero.set_rel_yield_to_int(hist_mean)

    # initiate the historic mean for each combination of crop and irrigation type
    hist_mean_per_crop = dict()
    for idx, crop in enumerate(crop_list):
        hist_mean_per_crop[idx] = dict()
        hist_mean_per_crop[crop] = {
            'value': np.zeros([int(len(filenames) / len(crop_list)), len(hist_mean)]),
            'idx': 0,
        }

    # calculate hazard as relative yield for all historic files and related future scenarios
    # and save them as hdf5 file in the output directory
    filename_list = list()
    output_list = list()
    for file_prop in file_props:
        # historic file
        crop_irr = (((file_prop)[6]).split('-'))[1] + '-' + (((file_prop)[6]).split('-'))[2]

        cp_his = RelativeCropyield()
        cp_his.set_from_single_run(input_dir=input_dir, bbox=bbox, yearrange=yearrange,
                                   ag_model=(file_prop)[0], cl_model=(file_prop)[1],
                                   scenario=(file_prop)[3], soc=(file_prop)[4],
                                   co2=(file_prop)[5], crop=(((file_prop)[6]).split('-'))[1],
                                   irr=(((file_prop)[6]).split('-'))[2])
        hist_mean = cp_his.calc_mean()
        cp_his.set_rel_yield_to_int(hist_mean)
        hist_mean_per_crop[crop_irr]['value'][hist_mean_per_crop[crop_irr]['idx'], :] = hist_mean
        hist_mean_per_crop[crop_irr]['idx'] = hist_mean_per_crop[crop_irr]['idx'] + 1

        filename = ('haz' + '_' + (file_prop)[0] + '_' + (file_prop)[1] + '_'
                    + (file_prop)[3] + '_' + (file_prop)[4] + '_' + (file_prop)[5] + '_'
                    + crop_irr + '_' + str(yearrange[0]) + '-' + str(yearrange[1]) + '.hdf5')
        filename_list.append(filename)
        output_list.append(cp_his)
        cp_his.select(reg_id=1).write_hdf5(os.path.join(output_dir, 'Hazard', filename))

        # compute the relative yield for all future scenarios with the corresponding historic mean
        for scenario in scenario_list:
            yearrange_fut = np.array([(YEARCHUNKS[scenario])['startyear'],
                                      (YEARCHUNKS[scenario])['endyear']])
            cp_fut = RelativeCropyield()
            cp_fut.set_from_single_run(input_dir=input_dir, bbox=bbox, yearrange=yearrange_fut,
                                       ag_model=(file_prop)[0], cl_model=(file_prop)[1],
                                       scenario=scenario, soc=(file_prop)[4],
                                       co2=(file_prop)[5], crop=(((file_prop)[6]).split('-'))[1],
                                       irr=(((file_prop)[6]).split('-'))[2])
            cp_fut.set_rel_yield_to_int(hist_mean)
            filename = ('haz' + '_' + (file_prop)[0] + '_' + (file_prop)[1] + '_'
                        + scenario + '_' + (file_prop)[4] + '_' + (file_prop)[5]
                        + '_' + crop_irr + '_' + str(yearrange_fut[0]) + '-'
                        + str(yearrange_fut[1]) + '.hdf5')
            filename_list.append(filename)
            output_list.append(cp_fut)
            cp_fut.select(reg_id=1).write_hdf5(os.path.join(output_dir, 'Hazard', filename))

    # calculate mean hist_mean for each crop-irrigation combination and save as hdf5 in output_dir
    for crop in crop_list:
        mean = np.mean((hist_mean_per_crop[crop])['value'], 0)
        mean_filename = 'hist_mean_' + crop + '_' + str(yearrange[0]) + \
        '-' + str(yearrange[1]) + '.hdf5'
        filename_list.append(mean_filename)
        output_list.append(mean)
        mean_file = h5py.File(os.path.join(output_dir, 'Hist_mean', mean_filename), 'w')
        mean_file.create_dataset('mean', data=mean)
        mean_file.create_dataset('lat', data=cp_his.centroids.lat)
        mean_file.create_dataset('lon', data=cp_his.centroids.lon)
        mean_file.close()

        # save historic mean as netcdf (saves mean, lat and lon as arrays)
#        mean_file = xr.Dataset({'mean': mean, 'lat': cp_his.centroids.lat, \
#                                'lon': cp_his.centroids.lon})
#        mean_file.to_netcdf(mean_dir+'hist_mean_'+crop+'.nc')

    if not return_data:
        return filename_list
    return filename_list, output_list
