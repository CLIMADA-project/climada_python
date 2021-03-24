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

Define Forecast
"""

__all__ = ['Forecast']



import logging
import datetime as dt
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import pyproj
import shapely
from cartopy.io import shapereader
from mpl_toolkits.axes_grid1 import make_axes_locatable
from iso3166 import countries as iso_cntry

from climada.entity.exposures import Exposures, LitPop
from climada.hazard import StormEurope, Hazard
from climada.engine import Impact
from climada.entity.impact_funcs import ImpactFuncSet
from climada.entity.impact_funcs.storm_europe import IFStormEurope
from climada.entity.impact_funcs.trop_cyclone import IFTropCyclone
import climada.util.plot as u_plot
from climada.util.config import CONFIG
from climada.util.files_handler import to_list
from climada.util.coordinates import coord_on_land as u_coord_on_land

LOGGER = logging.getLogger(__name__)

DATA_DIR = str(CONFIG.local_data.save_dir.dir())

FORECAST_DIR = str(Path(DATA_DIR) / 'forecast')

FORECAST_PLOT_DIR = str(Path(FORECAST_DIR) / 'plots')

# defining colormaps
# The colors are in line the european meteoalarm colors http://www.meteoalarm.info/ 
# and with the colors used at MeteoSwiss https://www.natural-hazards.ch/home/dealing-with-natural-hazards/explanation-of-the-danger-levels.html
# colors for warning levels
COLORS_WARN = np.array([[204 / 255, 255 / 255, 102 / 255, 1],  # green
                        [255 / 255, 255 / 255, 0 / 255, 1],  # yellow
                        [255 / 255, 153 / 255, 0 / 255, 1],  # orange
                        [255 / 255, 0 / 255, 0 / 255, 1],  # red
                        [128 / 255, 0 / 255, 0 / 255, 1],  # dark red
                        ])
#colors for warning probabilities
warnprob_colors = np.array([[255 / 255, 255 / 255, 255 / 255, 1],  # white
                            [215 / 255, 227 / 255, 238 / 255, 1],  # lightest blue
                            [181 / 255, 202 / 255, 255 / 255, 1],  # ligthblue
                            [127 / 255, 151 / 255, 255 / 255, 1],  # blue
                            [171 / 255, 207 / 255, 99 / 255, 1],  # green
                            [232 / 255, 245 / 255, 158 / 255, 1],  # light yellow?
                            [255 / 255, 250 / 255, 20 / 255, 1],  # yellow
                            [255 / 255, 209 / 255, 33 / 255, 1],  # light orange
                            [255 / 255, 163 / 255, 10 / 255, 1],  # orange
                            [255 / 255, 76 / 255, 0 / 255, 1],  # red
                            ])
warnprob_colors_extended = np.repeat(warnprob_colors, 10, axis=0)
CMAP_WARNPROB = ListedColormap(warnprob_colors_extended)
# colors for impact forecast
color_map_pre = plt.get_cmap('plasma', 90)
impact_colors = color_map_pre(np.linspace(0, 1, 90))
white_extended = np.repeat([[255 / 255, 255 / 255, 255 / 255, 1]], 10, axis=0)
impact_colors_extended = np.append(white_extended, impact_colors, axis=0)
CMAP_IMPACT = ListedColormap(impact_colors_extended)


class Forecast():
    """Forecast definition. Compute an impact forecast with predefined hazard
    originating from a forecast (like numerical weather prediction models),
    exposure and impact. Use the calc() method to calculate a forecasted
    impact. Then use the plotting methods to illustrate the forecasted impacts.

    Attributes:
        run_datetime (datetime.datetime):
        event_date (datetime.datetime):
        hazard (Hazard):
        haz_model (str):
        exposure (Exposure):
        country (str):
        vulnerability (ImpactFuncSet):
    """

    def __init__(self,
                 hazard_dict,
                 exposure,
                 impact_funcs,
                 haz_model = None,
                 exposure_name = None):
        """ Initialization with hazard, exposure and vulnerability. 
        Parameters:
            hazard_dict (dict, {datetime.datetime: Hazard}): dictionary of the
                format {run_datetime: Hazard} with run_datetime being the
                initialization time of a weather forecast run and Hazard being
                a climada Hazard derived from that forecast for one event.
                A probabilistic representation of that one event is possible,
                as long as the attribute Hazard.date is the same for all
                events. Several run_datetime:Hazard combinations for the same
                event can be provided.
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncSet): impact functions
            haz_model (str, optional): short string specifying the model used 
                to create the hazard, if possible three big letters.
                default is 'NWP' for numerical weather prediction.
            exposure_name (str, optional): short string specifying the exposure,
                which is used in filenames.
        """
        self.run_datetime = [key for key in hazard_dict.keys()]
        self.hazard = [hazard_dict[key] for key in self.run_datetime]
        #check event_date
        hazard_dates = [date for hazard in self.hazard for date in hazard.date]
        if not len(np.unique(hazard_dates)) == 1:
            ValueError('Please provide hazards containing only one ' +
                       'event_date. The current hazards contain several ' +
                       'events with different event_dates and the Forecast ' +
                       'class cannot function proberly with such hazards.')
        self.event_date = dt.datetime.fromordinal(np.unique(hazard_dates)[0])
        if haz_model == None:
            self.haz_model = 'NWP'
        else:
            self.haz_model = haz_model
        self.exposure = exposure
        if exposure_name == None:
            try:
                self.exposure_name = iso_cntry.get(exposure.gdf.region_id.unique()[0]).name
            except:
                self.exposure_name = 'custom'
        else:
            self.exposure_name = exposure_name
        self.vulnerability = impact_funcs
        self._impact = [Impact() for i in range(len(self.run_datetime))]

    def ei_exp(self, run_datetime = None):
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self._impact[haz_ind].eai_exp

    def ai_agg(self, run_datetime = None):
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self._impact[haz_ind].aai_agg

    def haz_summary_str(self, run_datetime = None):
        """ provide a summary string for the hazard part of the forecast
        Parameters:
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
        """
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return (self.hazard[haz_ind].tag.haz_type +
                '_' +
                self.haz_model +
                '_run' +
                run_datetime.strftime('%Y%m%d%H') +
                '_event' +
                self.event_date.strftime('%Y%m%d')
                )

    def summary_str(self, run_datetime = None):
        """ provide a summary string for the impact forecast
        Parameters:
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
        """
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return (self.haz_summary_str(run_datetime) +
                '_' +
                self.exposure_name
                )

    def lead_time(self, run_datetime = None):
        """ provide the lead time for the impact forecast
        Parameters:
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
        """
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self.event_date - run_datetime

    def calc(self, force_reassign=False, check_plot = False):
        """ calculate the impact using 
        exposure, hazard, and vulnerabilty. """
        # generate folders
        if not Path(FORECAST_PLOT_DIR).exists():
            Path(FORECAST_PLOT_DIR).mkdir(parents=True)     

        # force reassign
        if force_reassign:
            self.exposure.assign_centroids(self.hazard)
        # calc impact
        for ind_i, haz_i in enumerate(self.hazard):
            self._impact[ind_i].calc(self.exposure,
                         self.vulnerability,
                         haz_i,
                         save_mat=True)
            if check_plot:
                self._impact[ind_i].plot_hexbin_ei_exposure()

    def plot_imp_map(self,
                     run_datetime = None,
                     save_fig=True,
                     close_fig=True,
                     shapes_file=None):
        """ plot a map of the impacts 
        Parameters:
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
            save_fig (bool): True default to save the figure
            close_fig (bool): True default to close the figure
            shapes_file (str): points to a .shp-file with polygons
                in crs=epsg(21781)
        """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        # plot impact map of all runs
        map_file_name = (self.summary_str(run_datetime) +
                         '_impact_map' +
                         '.jpeg')
        map_file_name_full = Path(FORECAST_PLOT_DIR) / map_file_name
        lead_time_str = '{:.0f}'.format(self.lead_time(run_datetime).days +
                                        self.lead_time(run_datetime).seconds/60/60/24) # cant show '2' and '2.5'
        title_dict = {'event_day': self.event_date.strftime('%a %d %b %Y 00-24UTC'),
                      'run_start': (run_datetime.strftime('%d.%m.%Y %HUTC +') +
                                    lead_time_str +
                                    'd'),
                      'explain_text': ('mean building damage caused by wind'),
                      'model_text': "CLIMADA IMPACT"}
        f, ax = self._plot_imp_map(
                             run_datetime,
                             mask=self._impact[haz_ind].eai_exp > 1000,
                             title=title_dict,
                             cbar_label=('forecasted damage per gridcell [' +
                                         self._impact[haz_ind].unit + 
                                         ']'),
                             polygon_file=shapes_file
                             )
        if save_fig:
            f.savefig(map_file_name_full)
        if close_fig:
            f.clf()
            plt.close(f)

    def _plot_imp_map(self, run_datetime, mask, title, cbar_label, polygon_file = None):
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        # tryout new plot with right projection
        extend = 'neither'
        crs_epsg = ccrs.Mercator()
        value = self._impact[haz_ind].eai_exp
        #    value[np.invert(mask)] = np.nan
        coord = self._impact[haz_ind].coord_exp

        # Generate array of values used in each subplot
        array_sub = value
        shapes = True
        if not polygon_file:
            shapes = False
        proj = crs_epsg
        var_name = cbar_label
        geo_coord = coord
        num_im, list_arr = u_plot._get_collection_arrays(array_sub)
        list_tit = to_list(num_im, title, 'title')
        list_name = to_list(num_im, var_name, 'var_name')
        list_coord = to_list(num_im, geo_coord, 'geo_coord')

        kwargs = dict()
        kwargs['cmap'] = CMAP_IMPACT
        kwargs['s'] = 5
        kwargs['marker'] = ','
        kwargs['norm'] = BoundaryNorm(np.append(np.append([0], [10 ** x for x in np.arange(0, 2.9, 2.9 / 9)]),
                                                [10 ** x for x in np.arange(3, 7, 4 / 90)]), CMAP_IMPACT.N, clip=True)

        # Generate each subplot
        fig, axis_sub = u_plot.make_map(num_im, proj=proj)
        if not isinstance(axis_sub, np.ndarray):
            axis_sub = np.array([[axis_sub]])
        fig.set_size_inches(9, 8)
        for array_im, axis, tit, name, coord in zip(list_arr, axis_sub.flatten(), list_tit, list_name, list_coord):
            if coord.shape[0] != array_im.size:
                raise ValueError("Size mismatch in input array: %s != %s." % \
                                 (coord.shape[0], array_im.size))
    
            # Binned image with coastlines
            extent = u_plot._get_borders(coord)
            axis.set_extent((extent), ccrs.PlateCarree())
            # if pop_name:
            #     u_plot.add_populated_places(axis, extent, proj)
    
            hex_bin = axis.scatter(coord[:, 1], coord[:, 0], c=array_im, \
                                   transform=ccrs.PlateCarree(), **kwargs)
            if shapes:
                # add warning regions
                shp = shapereader.Reader(polygon_file)
                project_crs = lambda x, y: pyproj.transform(pyproj.Proj(init='epsg:21781'), pyproj.Proj(init='epsg:4150'),
                                                            x, y)
                for geometry, record in zip(shp.geometries(), shp.records()):
                    geom2 = shapely.ops.transform(project_crs, geometry)
                    axis.add_geometries([geom2], crs=ccrs.PlateCarree(), facecolor='', \
                                        edgecolor='gray')
                
            else: # add country boundaries
                u_plot.add_shapes(axis)

            # Create colorbar in this axis
            cbax = make_axes_locatable(axis).append_axes('bottom', size="6.5%", \
                                                         pad=0.1, axes_class=plt.Axes)
            cbar = plt.colorbar(hex_bin, cax=cbax, orientation='horizontal',
                                extend=extend)
            cbar.set_label(name)
            cbar.formatter.set_scientific(False)
            cbar.set_ticks([0, 1000, 10000, 100000, 1000000])
            cbar.set_ticklabels(['0', "1'000", "10'000", "100'000", "1'000'000"])
            plt.figtext(0.125, 0.84, tit['model_text'], fontsize='x-large', color='k', ha='left')
            plt.figtext(0.125, 0.81, tit['explain_text'], fontsize='x-large', color='k', ha='left')
            plt.figtext(0.9, 0.84, tit['event_day'], fontsize='x-large', color='r', ha='right')
            plt.figtext(0.9, 0.81, tit['run_start'], fontsize='x-large', color='k', ha='right')
            plt.subplots_adjust(top=0.8)
            # axis.set_extent((5.70, 10.49, 45.7, 47.81), crs=ccrs.PlateCarree())
    
        return fig, axis

    def plot_hist(self,run_datetime = None, save_fig=True,close_fig=True):
        """ plot histogram of the forecasted impacts all ensemble members
        Parameters:
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
            save_fig (bool): True default to save the figure
            close_fig (bool): True default to close the figure
        """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        # plot histogram of all runs
        histbin_file_name = (self.summary_str(run_datetime) +
                             '_histbin' +
                             '.svg')
        histbin_file_name_full = Path(FORECAST_PLOT_DIR) / histbin_file_name
        lower_bound = np.max([np.floor(np.log10(np.max([np.min(self._impact[haz_ind].at_event), 0.1]))), 0])
        upper_bound = np.max([np.ceil(np.log10(np.max([np.max(self._impact[haz_ind].at_event), 0.1]))), lower_bound + 5]) + 0.1
        bins_log = np.arange(lower_bound, upper_bound, 0.5)
        bins = 10 ** bins_log
        f = plt.figure(figsize=(9, 8))
        ax = f.add_subplot(111)
        plt.xscale('log')
        plt.hist([np.max([x, 1]) for x in self._impact[haz_ind].at_event],
                 bins=bins,
                 weights=np.ones(len(self._impact[haz_ind].at_event)) / len(self._impact[haz_ind].at_event))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        #                        formatter.format = "%:'.0f"
        ax.xaxis.set_major_formatter(formatter)
    
        x_ticklabels = list()
        x_ticks = list()
        ticklabel_dict = {1: "1",
                          10: "10",
                          100: "100",
                          1000: "1'000",
                          10000: "10'000",
                          100000: "100'000",
                          1000000: "1 million",
                          10000000: "10 million",
                          100000000: "100 million",
                          1000000000: "1 billion",
                          10000000000: "10 billion",
                          100000000000: "100 billion",
                          1000000000000: "1 trillion",
                          10000000000000: "10 trillion",
                          100000000000000: "100 trillion",
                          }
        for i in np.arange(15):
            tick_i = 10.0 ** i
            x_ticks.append(tick_i)
            x_ticklabels.append(ticklabel_dict[tick_i])
        ax.xaxis.set_ticks(x_ticks)
        ax.xaxis.set_ticklabels(x_ticklabels)
        plt.xticks(rotation=15, horizontalalignment='right')
        plt.xlim([(10 ** -0.25) * bins[0], (10 ** 0.25) * bins[-1]])
        #                        plt.title('COSMO-E impact forecast\n'+
        #                                  'for ' + event_day.strftime('%A, %d.%m.%Y') +
        #                                  '\n' +
        #                                  'run started at ' +
        #                                  each_run.strftime('%d.%m.%Y %H:%M'))
        lead_time_str = '{:.0f}'.format(self.lead_time(run_datetime).days + self.lead_time(run_datetime).seconds/60/60/24) # cant show '2' and '2.5'
        title_dict = {'event_day': self.event_date.strftime('%a %d %b %Y 00-24UTC'),
                      'run_start': run_datetime.strftime('%d.%m.%Y %HUTC +') + lead_time_str + 'd',
                      'explain_text': ('total building damage caused by wind'),
                      'model_text': "CLIMADA IMPACT"}
        plt.figtext(0.125, 0.98, title_dict['model_text'], fontsize='x-large', color='k', ha='left')
        plt.figtext(0.125, 0.94, title_dict['explain_text'], fontsize='x-large', color='k', ha='left')
        plt.figtext(0.9, 0.98, title_dict['event_day'], fontsize='x-large', color='r', ha='right')
        plt.figtext(0.9, 0.94, title_dict['run_start'], fontsize='x-large', color='k', ha='right')
        plt.xlabel('forecasted total damage for ' + self.exposure_name +
                   ' [' + self._impact[haz_ind].unit + ']')
        plt.ylabel('probability')
        plt.text(0.75, 0.85,
                 # 'mean damage:\n' + locale.currency(np.round(imp.at_event.mean()), grouping=True),
                 'mean damage:\nCHF ' + '{0:,}'.format(np.int(np.round(self._impact[haz_ind].at_event.mean()))).replace(',', "'"),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax.transAxes)
        if save_fig:
            plt.savefig(histbin_file_name_full)
        if close_fig:
            plt.clf()
            plt.close(f)

    def plot_exceedence_prob(self, threshold, explain_str=None,
                             run_datetime = None, save_fig=True,close_fig=True):
        """ plot exceedence map
        Parameters:
            threshold (float): threshold of impact unit for which exceedence
                probability should be plotted
            explain_str (str, optional): short str which explains threshold,
                explain_str is included in the title of the figure
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
            save_fig (bool): True default to save the figure
            close_fig (bool): True default to close the figure
        """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        wind_map_file_name = (self.summary_str(run_datetime) +
                 '_exceed_' +
                 str(threshold) +
                 '_map.jpeg')
        wind_map_file_name_full = Path(FORECAST_PLOT_DIR) / wind_map_file_name
        lead_time_str = '{:.0f}'.format(self.lead_time(run_datetime).days + self.lead_time(run_datetime).seconds/60/60/24) # cant show '2' and '2.5'
        title_dict = {'event_day': self.event_date.strftime('%a %d %b %Y 00-24UTC'),
                      'run_start': run_datetime.strftime('%d.%m.%Y %HUTC +') + lead_time_str + 'd',
                      'explain_text': ('threshold: ' +
                                       str(threshold) +
                                       ' ' +
                                       self._impact[haz_ind].unit) if explain_str is None else explain_str,
                      'model_text': 'Exceedance probability map'}
        cbar_label = 'probabilty of reaching threshold'
        f, ax = self._plot_exc_prob(run_datetime, threshold, title_dict, cbar_label)
        if save_fig:
            plt.savefig(wind_map_file_name_full)
        if close_fig:
            plt.clf()
            plt.close(f)

    def _plot_exc_prob(self, run_datetime, threshold, title, cbar_label, polygon_file = None, mask = None):
        """  plot the probability of reaching a threshold """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        extend = 'neither'
        try:
            crs_epsg = ccrs.epsg(2056)
        except:
            crs_epsg = ccrs.Mercator()
    
        #    title = "title"
        #    cbar_label = 'Value ()'
        value = np.squeeze(np.asarray((self._impact[haz_ind].imp_mat > threshold).sum(axis=0) / self._impact[haz_ind].event_id.size))
        if mask is not None:
            value[np.invert(mask)] = np.nan
        #    value[value==0] = np.nan
        coord = self._impact[haz_ind].coord_exp
        # u_plot.geo_scatter_from_array(value, coord, cbar_label, title, \
        #    pop_name, buffer, extend, proj=crs_epsg)
    
        # Generate array of values used in each subplot
        array_sub = value
        shapes = True
        if not polygon_file:
            shapes = False

        proj = crs_epsg
        var_name = cbar_label
        geo_coord = coord
        num_im, list_arr = u_plot._get_collection_arrays(array_sub)
        list_tit = to_list(num_im, title, 'title')
        list_name = to_list(num_im, var_name, 'var_name')
        list_coord = to_list(num_im, geo_coord, 'geo_coord')
    
        kwargs = dict()
        kwargs['cmap'] = CMAP_WARNPROB
        kwargs['s'] = 5
        kwargs['marker'] = ','
        kwargs['norm'] = BoundaryNorm([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], CMAP_WARNPROB.N, clip=True)

        # Generate each subplot
        fig, axis_sub = u_plot.make_map(num_im, proj=proj)
        if not isinstance(axis_sub, np.ndarray):
            axis_sub = np.array([[axis_sub]])
        fig.set_size_inches(9, 8)
        for array_im, axis, tit, name, coord in zip(list_arr, axis_sub.flatten(), list_tit, list_name, list_coord):
            if coord.shape[0] != array_im.size:
                raise ValueError("Size mismatch in input array: %s != %s." % \
                                 (coord.shape[0], array_im.size))
    
            hex_bin = axis.scatter(coord[:, 1], coord[:, 0], c=array_im, \
                                   transform=ccrs.PlateCarree(), **kwargs)
            if shapes:
                # add warning regions
                shp = shapereader.Reader(polygon_file)
                project_crs = lambda x, y: pyproj.transform(pyproj.Proj(init='epsg:21781'), pyproj.Proj(init='epsg:4150'),
                                                            x, y)
                for geometry, record in zip(shp.geometries(), shp.records()):
                    geom2 = shapely.ops.transform(project_crs, geometry)
                    axis.add_geometries([geom2], crs=ccrs.PlateCarree(), facecolor='', \
                                        edgecolor='gray')
                # add country boundaries
            #            u_plot.add_shapes(axis)
    
            # Create colorbar in this axis
            cbax = make_axes_locatable(axis).append_axes('bottom', size="6.5%", \
                                                         pad=0.1, axes_class=plt.Axes)
            cbar = plt.colorbar(hex_bin, cax=cbax, orientation='horizontal',
                                extend=extend)
            cbar.set_label(name)
            plt.figtext(0.125, 0.84, tit['model_text'], fontsize='x-large', color='k', ha='left')
            plt.figtext(0.125, 0.81, tit['explain_text'], fontsize='x-large', color='k', ha='left')
            plt.figtext(0.9, 0.84, tit['event_day'], fontsize='x-large', color='r', ha='right')
            plt.figtext(0.9, 0.81, tit['run_start'], fontsize='x-large', color='k', ha='right')
            #        axis.set_title(tit)
            axis.set_extent((5.70, 10.49, 45.7, 47.81), crs=ccrs.PlateCarree())
    
        return fig, axis
    
    def plot_warn_map(self, polygon_file = None,
                      polygon_file_crs = 'epsg:4326',
                      thresholds = [2,3,4,5],
                      decision_level = 'exposure_point',
                      probability_aggregation = 0.5, area_aggregation = 0.5,
                      title = 'WARNINGS',
                      explain_text =  'warn level based on thresholds',
                      run_datetime = None,
                      save_fig=True,close_fig=True):
        """ plot map colored with 5 warning colors for all regions in provided
        shape file.
        Parameters:
            polygon_file (str): path to shp-file containing warning region polygons
            polygon_file_crs (str): string of pattern <provider>:<code> specifying
                the crs. has to be readable by pyproj.Proj
            thresholds (list of 4 floats): thresholds for coloring region
                ins second, third, forth and fifth warning color
            decision_level (str): either 'exposure_point'  or 'polygon'
            probability_aggregation (float or str): either a float between
                [0..1] spezifying a quantile or 'mean' or 'sum'
            area_aggregation (float or str): either a float between
                [0..1] specifying a quantile or 'mean' or 'sum'
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
            save_fig (bool): True default to save the figure
            close_fig (bool): True default to close the figure
        """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        warn_map_file_name = (self.summary_str(run_datetime) +
                              '_warn_map.jpeg')
        warn_map_file_name_full = Path(FORECAST_PLOT_DIR) / warn_map_file_name
        decision_dict = {'probability_aggregation': probability_aggregation,
                         'area_aggregation': area_aggregation}
        lead_time_str = '{:.0f}'.format(self.lead_time(run_datetime).days + self.lead_time(run_datetime).seconds/60/60/24) # cant show '2' and '2.5'
        title_dict = {'event_day': self.event_date.strftime('%a %d %b %Y 00-24UTC'),
                      'run_start': run_datetime.strftime('%d.%m.%Y %HUTC +') + lead_time_str + 'd',
                      'explain_text': explain_text,
                      'model_text': title}

        f, ax = self._plot_warn(run_datetime, thresholds, decision_level, decision_dict,
                                polygon_file, polygon_file_crs, title_dict)
        if save_fig:
            plt.savefig(warn_map_file_name_full)
        if close_fig:
            plt.clf()
            plt.close(f)

    def _plot_warn(self, run_datetime, thresholds,
                   decision_level, decision_dict,
                   polygon_file, polygon_file_crs,
                   title):
        """ plotting the warning level of each warning region based on thresholds """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        try:
            crs_epsg = ccrs.epsg(2056)
        except:
            crs_epsg = ccrs.Mercator()
    
        # u_plot.geo_scatter_from_array(value, coord, cbar_label, title, \
        #    pop_name, buffer, extend, proj=crs_epsg)
    
        # Generate array of values used in each subplot

        proj = crs_epsg



    
        kwargs = dict()
        kwargs['cmap'] = CMAP_WARNPROB
        kwargs['s'] = 5
        kwargs['marker'] = ','
        kwargs['norm'] = BoundaryNorm([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], CMAP_WARNPROB.N, clip=True)
    
        # Generate each subplot
        fig, axis = u_plot.make_map(1, proj=proj)
        if isinstance(axis, np.ndarray):
            axis = axis[0]
        tit = title
        fig.set_size_inches(9, 8)


    
        # add warning regions
        shp = shapereader.Reader(polygon_file)
        transformer = pyproj.Transformer.from_crs(polygon_file_crs,
                                                  # self._impact.crs['init'],
                                                  self._impact[haz_ind].crs,
                                                  always_xy=True)
        # project_crs = lambda x, y: pyproj.transform(pyproj.Proj(init='epsg:21781'),
        #                                             pyproj.Proj(init='epsg:4150'),
        #                                             x, y)
        for geometry, record in zip(shp.geometries(), shp.records()):
            geom2 = shapely.ops.transform(transformer.transform, geometry)
            # geom2 = shapely.ops.transform(project_crs, geometry)
            in_geom = u_coord_on_land(lat=self._impact[haz_ind].coord_exp[:, 0],
                                      lon=self._impact[haz_ind].coord_exp[:, 1],
                                      land_geom=geom2) 
            if not in_geom.any():
                continue
            # decide warninglevel
            warn_level = 0
            for ind_i, warn_thres_i in enumerate(thresholds):
                if decision_level == 'exposure_point':
                    if not (isinstance(decision_dict['probability_aggregation'], float)
                            &
                            isinstance(decision_dict['area_aggregation'], float)):
                        ValueError(" If decision_level is 'grid_point'," +
                                   "parameters probability_aggregation and " +
                                   "area_aggregation of " +
                                   "Forecast.plot_warn_map() must both be " +
                                   "floats between [0..1]. Which each " +
                                   "specify quantiles.")
                    # decision at each grid_point
                    probabilities = np.squeeze(np.asarray((self._impact[haz_ind].imp_mat >= warn_thres_i).sum(axis=0) / self._impact[haz_ind].event_id.size))
                    # quantiles over probability
                    area = (probabilities[in_geom] >= decision_dict['probability_aggregation']).sum()
                    # quantiles over area
                    if area >= (in_geom.sum() * decision_dict['area_aggregation']):
                        warn_level = ind_i+1
                elif decision_level == 'polygon':
                    #aggregation over area
                    if isinstance(decision_dict['area_aggregation'], float):
                        value_per_member = np.percentile(self._impact[haz_ind].imp_mat[:,in_geom].todense(),
                                                         decision_dict['area_aggregation'],
                                                         axis=1)
                    elif  decision_dict['area_aggregation']=='sum':
                        value_per_member = np.sum(self._impact[haz_ind].imp_mat[:,in_geom].todense(),
                                                         axis=1)
                    elif  decision_dict['area_aggregation']=='mean':
                        value_per_member = np.mean(self._impact[haz_ind].imp_mat[:,in_geom].todense(),
                                                         axis=1)
                    else:
                        ValueError("Parameter area_aggregation of " +
                                   "Forecast.plot_warn_map() must eiter be " +
                                   "a float between [0..1], which " +
                                   "specifys a quantile. or 'sum' or 'mean'.")
                    #aggregation over members/probability
                    if isinstance(decision_dict['probability_aggregation'], float):
                        value_per_region = np.percentile(value_per_member,
                                                         decision_dict['probability_aggregation'])
                    elif  decision_dict['probability_aggregation']=='sum':
                        value_per_region = np.sum(value_per_member)
                    elif  decision_dict['probability_aggregation']=='mean':
                        value_per_region = np.mean(value_per_member)
                    else:
                        ValueError("Parameter probability_aggregation of " +
                                   "Forecast.plot_warn_map() must eiter be " +
                                   "a float between [0..1], which " +
                                   "specifys a quantile. or 'sum' or 'mean'.")
                    #warn level decision
                    if value_per_region >= warn_thres_i:
                        warn_level = ind_i+1
                else:
                    ValueError("Parameter decision_level of " +
                               "Forecast.plot_warn_map() must eiter be " +
                               "'grid_point' or 'polygon'.")
            # plot warn_region with specific color (dependent on warning level)
            axis.add_geometries([geom2],
                                crs=ccrs.PlateCarree(),
                                facecolor=COLORS_WARN[warn_level, :],
                                edgecolor='gray')

        # Create legend in this axis
        legend_elements = [
            Patch(facecolor=COLORS_WARN[0, :],
                  edgecolor='gray',
                  label='1: Minimal or no hazard'),  # Warning level 1
            Patch(facecolor=COLORS_WARN[1, :],
                  edgecolor='gray',
                  label='2: Moderate hazard'),
            Patch(facecolor=COLORS_WARN[2, :],
                  edgecolor='gray',
                  label='3: Significant hazard'),
            Patch(facecolor=COLORS_WARN[3, :],
                  edgecolor='gray',
                  label='4: Severe hazard'),
            Patch(facecolor=COLORS_WARN[4, :],
                  edgecolor='gray',
                  label='5: Very severe hazard'),
        ]
        axis.legend(handles=legend_elements, loc='upper center', framealpha=0.5,
                    bbox_to_anchor=(0.5, -0.02), ncol=3)
        plt.figtext(0.125, 0.84, tit['model_text'], fontsize='x-large', color='k', ha='left')
        plt.figtext(0.125, 0.81, tit['explain_text'], fontsize='x-large', color='k', ha='left')
        plt.figtext(0.9, 0.84, tit['event_day'], fontsize='x-large', color='r', ha='right')
        plt.figtext(0.9, 0.81, tit['run_start'], fontsize='x-large', color='k', ha='right')

        axis.set_extent((5.70, 10.49, 45.7, 47.81), crs=ccrs.PlateCarree())
        return fig, axis

    def plot_hexbin_ei_exposure(self,run_datetime = None):
        """ plot the expected impact
        Parameters:
            run_datetime (datetime.datetime, optional): select the used hazard
                by the run_datetime
        """
        # select hazard with run_datetime
        if run_datetime == None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self._impact[haz_ind].plot_hexbin_eai_exposure()
