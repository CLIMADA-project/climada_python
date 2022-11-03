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

Define Forecast
"""

__all__ = ["Forecast"]


import logging
import datetime as dt
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import pyproj
import shapely
from cartopy.io import shapereader
from mpl_toolkits.axes_grid1 import make_axes_locatable

from climada.hazard import Hazard
from climada.entity import Exposures
from climada.entity.impact_funcs import ImpactFuncSet
from climada.engine import ImpactCalc
import climada.util.plot as u_plot
from climada.util.config import CONFIG
from climada.util.files_handler import to_list
import climada.util.coordinates as u_coord
from climada.util.value_representation import (
    value_to_monetary_unit as u_value_to_monetary_unit,
)

LOGGER = logging.getLogger(__name__)

DATA_DIR = CONFIG.local_data.save_dir.str()

FORECAST_DIR = CONFIG.engine.forecast.local_data.str()

FORECAST_PLOT_DIR = CONFIG.engine.forecast.plot_dir.dir()

# defining colormaps
# The colors are in line the european meteoalarm colors http://www.meteoalarm.info/
# and with the colors used at MeteoSwiss
# https://www.natural-hazards.ch/home/dealing-with-natural-hazards/explanation-of-the-danger-levels.html
# colors for warning levels
COLORS_WARN = np.array(
    [
        [204 / 255, 255 / 255, 102 / 255, 1],  # green
        [255 / 255, 255 / 255, 0 / 255, 1],  # yellow
        [255 / 255, 153 / 255, 0 / 255, 1],  # orange
        [255 / 255, 0 / 255, 0 / 255, 1],  # red
        [128 / 255, 0 / 255, 0 / 255, 1],  # dark red
    ]
)
# colors for warning probabilities
warnprob_colors = np.array(
    [
        [255 / 255, 255 / 255, 255 / 255, 1],  # white
        [215 / 255, 227 / 255, 238 / 255, 1],  # lightest blue
        [181 / 255, 202 / 255, 255 / 255, 1],  # ligthblue
        [127 / 255, 151 / 255, 255 / 255, 1],  # blue
        [171 / 255, 207 / 255, 99 / 255, 1],  # green
        [232 / 255, 245 / 255, 158 / 255, 1],  # light yellow?
        [255 / 255, 250 / 255, 20 / 255, 1],  # yellow
        [255 / 255, 209 / 255, 33 / 255, 1],  # light orange
        [255 / 255, 163 / 255, 10 / 255, 1],  # orange
        [255 / 255, 76 / 255, 0 / 255, 1],  # red
    ]
)
warnprob_colors_extended = np.repeat(warnprob_colors, 10, axis=0)
CMAP_WARNPROB = ListedColormap(warnprob_colors_extended)
# colors for impact forecast
color_map_pre = plt.get_cmap("plasma", 90)
impact_colors = color_map_pre(np.linspace(0, 1, 90))
white_extended = np.repeat([[255 / 255, 255 / 255, 255 / 255, 1]], 10, axis=0)
impact_colors_extended = np.append(white_extended, impact_colors, axis=0)
CMAP_IMPACT = ListedColormap(impact_colors_extended)


class Forecast:
    """Forecast definition. Compute an impact forecast with predefined hazard
    originating from a forecast (like numerical weather prediction models),
    exposure and impact. Use the calc() method to calculate a forecasted
    impact. Then use the plotting methods to illustrate the forecasted impacts.
    By default plots are saved under in a '/forecast/plots' folder in the
    configurable save_dir in local_data (see climada.util.config) under a name
    summarizing the Hazard type, haz model name, initialization time of the
    forecast run, event date, exposure name and the plot title.
    As the class is relatively new, there might be future changes to the attributes,
    the methods, and the parameters used to call the methods.
    It was discovered at some point, that there might be a memory leak in
    matplotlib even when figures are closed
    (https://github.com/matplotlib/matplotlib/issues/8519). Due to this reason
    the plotting functions in this module have the flag close_fig, to close
    figures within the function scope, which might mitigate that problem if a
    script runs this plotting functions many times.

    Attributes
    ----------
    run_datetime : list of datetime.datetime
        initialization time of the forecast model run used to create the Hazard
    event_date: datetime.datetime
        Date on which the Hazard event takes place
    hazard : list of CLIMADA Hazard
        List of the hazard forecast with different lead times.
    haz_model : str
        Short string specifying the model used to create the hazard,
        if possible three big letters.
    exposure : Exposure
        an CLIMADA Exposures containg values at risk
    exposure_name : str
        string specifying the exposure (e.g. 'EU'), which is used to
        name output files.
    vulnerability : ImpactFuncSet
        Set of impact functions used in the impact calculation.
    """

    def __init__(
        self,
        hazard_dict: Dict[str, Hazard],
        exposure: Exposures,
        impact_funcs: ImpactFuncSet,
        haz_model: str = "NWP",
        exposure_name: Optional[str] = None
    ):
        """Initialization with hazard, exposure and vulnerability.

        Parameters
        ----------
        hazard_dict : dict
            Dictionary of the format {run_datetime: Hazard} with run_datetime
            being the initialization time of a weather forecast run and Hazard
            being a CLIMADA Hazard derived from that forecast for one event.
            A probabilistic representation of that one event is possible,
            as long as the attribute Hazard.date is the same for all
            events. Several run_datetime:Hazard combinations for the same
            event can be provided.
        exposure : Exposures
        impact_funcs : ImpactFuncSet
        haz_model : str, optional
            Short string specifying the model used to create the hazard,
            if possible three big letters. Default is 'NWP' for numerical
            weather prediction.
        exposure_name : str, optional
            string specifying the exposure (e.g. 'EU'), which is used to
            name output files. If ``None``, the name will be inferred from the Exposures
            GeoDataframe ``region_id`` column, using the corresponding name of the region
            with the lowest ISO 3166-1 numeric code. If that fails, it defaults to
            ``"custom"``.
        """
        self.run_datetime = list(hazard_dict.keys())
        self.hazard = list(hazard_dict.values())
        # check event_date
        hazard_date = np.unique(
            [date for hazard in self.hazard for date in hazard.date]
        )
        if not len(hazard_date) == 1:
            raise ValueError(
                "Please provide hazards containing only one "
                + "event_date. The current hazards contain several "
                + "events with different event_dates and the Forecast "
                + "class cannot function proberly with such hazards."
            )
        self.event_date = dt.datetime.fromordinal(hazard_date[0])
        self.haz_model = haz_model
        self.exposure = exposure
        if exposure_name is None:
            try:
                self.exposure_name = u_coord.country_to_iso(
                    exposure.gdf.region_id.unique()[0], "name"
                )
            except (KeyError, AttributeError):
                self.exposure_name = "custom"
        else:
            self.exposure_name = exposure_name
        self.vulnerability = impact_funcs
        self._impact = [None for dt in self.run_datetime]

    def ei_exp(self, run_datetime=None):
        """
        Expected impact per exposure

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        Returns
        -------
        float
        """
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self._impact[haz_ind].eai_exp

    def ai_agg(self, run_datetime=None):
        """average impact aggregated over all exposures

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        Returns
        -------
        float
        """
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self._impact[haz_ind].aai_agg

    def haz_summary_str(self, run_datetime=None):
        """provide a summary string for the hazard part of the forecast

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        Returns
        -------
        str
            summarizing the most important information about the hazard
        """
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return (
            self.hazard[haz_ind].tag.haz_type
            + "_"
            + self.haz_model
            + "_run"
            + run_datetime.strftime("%Y%m%d%H")
            + "_event"
            + self.event_date.strftime("%Y%m%d")
        )

    def summary_str(self, run_datetime=None):
        """provide a summary string for the impact forecast

        Parameters
        ----------
        run_datetime: datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.

        Returns
        -------
        str
            summarizing the most important information about
            the impact forecast
        """
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        return self.haz_summary_str(run_datetime) + "_" + self.exposure_name

    def lead_time(self, run_datetime=None):
        """provide the lead time for the impact forecast

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.

        Returns
        -------
        datetime.timedelta
            the difference between the initialization time of the forecast
            model run and the date of the event, commenly named lead time
        """
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        return self.event_date - run_datetime

    def calc(self, force_reassign=False):
        """calculate the impacts for all lead times using
        exposure, all hazards of all run_datetime, and ImpactFunctionSet.

        Parameters
        ----------
        force_reassign : bool, optional
            Reassign hazard centroids to the exposure for all hazards,
            default is false.
        """
        # calc impact
        if self.hazard:
            self.exposure.assign_centroids(self.hazard[0], overwrite=force_reassign)
        for ind_i, haz_i in enumerate(self.hazard):
            self._impact[ind_i] = ImpactCalc(self.exposure, self.vulnerability, haz_i)\
                                  .impact(save_mat=True, assign_centroids=False)

    def plot_imp_map(
        self,
        run_datetime=None,
        save_fig=True,
        close_fig=False,
        polygon_file=None,
        polygon_file_crs="epsg:4326",
        proj=ccrs.PlateCarree(),
        figsize=(9, 13),
        adapt_fontsize=True,
    ):
        """plot a map of the impacts

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        save_fig : bool, optional
            Figure is saved if True, folder is within your configurable
            save_dir and filename is derived from the method summary_str()
            (for more details see class docstring).  Default is True.
        close_fig : bool, optional
            Figure not drawn if True. Default is False.
        polygon_file : str, optional
            Points to a .shp-file with polygons do be drawn as outlines on
            the plot, default is None to not draw the lines. please also
            specify the crs in the parameter polygon_file_crs.
        polygon_file_crs : str, optional
            String of pattern <provider>:<code> specifying
            the crs. has to be readable by pyproj.Proj. Default is
            'epsg:4326'.
        proj : ccrs
            coordinate reference system used in coordinates
            The default is ccrs.PlateCarree()
        figsize : tuple
            figure size for plt.subplots, width, height in inches
            The default is (9, 13)
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.

        Returns
        -------
        axes: cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        # select hazard with run_datetime
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        # plot impact map of all runs
        map_file_name = self.summary_str(run_datetime) + "_impact_map" + ".jpeg"
        map_file_name_full = FORECAST_PLOT_DIR / map_file_name
        lead_time_str = "{:.0f}".format(
            self.lead_time(run_datetime).days
            + self.lead_time(run_datetime).seconds / 60 / 60 / 24
        )
        title_dict = {
            "event_day": self.event_date.strftime("%a %d %b %Y 00-24UTC"),
            "run_start": (
                run_datetime.strftime("%d.%m.%Y %HUTC +") + lead_time_str + "d"
            ),
            "explain_text": ("mean building damage caused by wind"),
            "model_text": "CLIMADA IMPACT",
        }
        fig, axes = self._plot_imp_map(
            run_datetime,
            title=title_dict,
            cbar_label=(
                "forecasted damage per gridcell [" + self._impact[haz_ind].unit + "]"
            ),
            polygon_file=polygon_file,
            polygon_file_crs=polygon_file_crs,
            proj=proj,
            figsize=figsize,
            adapt_fontsize=adapt_fontsize,
        )
        if save_fig:
            fig.savefig(map_file_name_full)
        if close_fig:
            fig.clf()
            plt.close(fig)
        return axes

    def _plot_imp_map(
        self,
        run_datetime,
        title,
        cbar_label,
        polygon_file=None,
        polygon_file_crs="epsg:4326",
        proj=ccrs.PlateCarree(),
        figsize=(9, 13),
        adapt_fontsize=True,
    ):
        # select hazard with run_datetime
        # pylint: disable=protected-access
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        # tryout new plot with right projection
        extend = "neither"
        value = self._impact[haz_ind].eai_exp
        #    value[np.invert(mask)] = np.nan
        coord = self._impact[haz_ind].coord_exp

        # Generate array of values used in each subplot
        array_sub = value
        shapes = True
        if not polygon_file:
            shapes = False
        var_name = cbar_label
        geo_coord = coord
        num_im, list_arr = u_plot._get_collection_arrays(array_sub)
        list_tit = to_list(num_im, title, "title")
        list_name = to_list(num_im, var_name, "var_name")
        list_coord = to_list(num_im, geo_coord, "geo_coord")

        kwargs = dict()
        kwargs["cmap"] = CMAP_IMPACT
        kwargs["s"] = 5
        kwargs["marker"] = ","
        kwargs["norm"] = BoundaryNorm(
            np.append(
                np.append([0], [10**x for x in np.arange(0, 2.9, 2.9 / 9)]),
                [10**x for x in np.arange(3, 7, 4 / 90)],
            ),
            CMAP_IMPACT.N,
            clip=True,
        )

        # Generate each subplot
        fig, axis_sub, _fontsize = u_plot.make_map(
            num_im, proj=proj, figsize=figsize, adapt_fontsize=adapt_fontsize
        )
        if not isinstance(axis_sub, np.ndarray):
            axis_sub = np.array([[axis_sub]])
        fig.set_size_inches(9, 8)
        for array_im, axis, tit, name, coord in zip(
            list_arr, axis_sub.flatten(), list_tit, list_name, list_coord
        ):
            if coord.shape[0] != array_im.size:
                raise ValueError(
                    "Size mismatch in input array: %s != %s."
                    % (coord.shape[0], array_im.size)
                )
            # Binned image with coastlines
            extent = u_plot._get_borders(coord)
            axis.set_extent((extent), ccrs.PlateCarree())
            hex_bin = axis.scatter(
                coord[:, 1],
                coord[:, 0],
                c=array_im,
                transform=ccrs.PlateCarree(),
                **kwargs
            )
            if shapes:
                # add warning regions
                shp = shapereader.Reader(polygon_file)
                transformer = pyproj.Transformer.from_crs(
                    polygon_file_crs, self._impact[haz_ind].crs, always_xy=True
                )
                for geometry, _ in zip(shp.geometries(), shp.records()):
                    geom2 = shapely.ops.transform(transformer.transform, geometry)
                    axis.add_geometries(
                        [geom2],
                        crs=ccrs.PlateCarree(),
                        facecolor="none",
                        edgecolor="gray",
                    )
            else:  # add country boundaries
                u_plot.add_shapes(axis)
            # Create colorbar in this axis
            cbax = make_axes_locatable(axis).append_axes(
                "bottom", size="6.5%", pad=0.3, axes_class=plt.Axes
            )
            cbar = plt.colorbar(
                hex_bin, cax=cbax, orientation="horizontal", extend=extend
            )
            cbar.set_label(name)
            cbar.formatter.set_scientific(False)
            cbar.set_ticks([0, 1000, 10000, 100000, 1000000])
            cbar.set_ticklabels(["0", "1 000", "10 000", "100 000", "1 000 000"])
            title_position = {
                "model_text": [0.02, 0.85],
                "explain_text": [0.02, 0.81],
                "event_day": [0.98, 0.85],
                "run_start": [0.98, 0.81],
            }
            left_right = {
                "model_text": "left",
                "explain_text": "left",
                "event_day": "right",
                "run_start": "right",
            }
            color = {
                "model_text": "k",
                "explain_text": "k",
                "event_day": "r",
                "run_start": "k",
            }
            for t_i in tit:
                plt.figtext(
                    title_position[t_i][0],
                    title_position[t_i][1],
                    tit[t_i],
                    fontsize="xx-large",
                    color=color[t_i],
                    ha=left_right[t_i],
                )

        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        return fig, axis_sub

    def plot_hist(
        self, run_datetime=None, save_fig=True, close_fig=False, figsize=(9, 8)
    ):
        """plot histogram of the forecasted impacts all ensemble members

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        save_fig : bool, optional
            Figure is saved if True, folder is within your configurable
            save_dir and filename is derived from the method summary_str()
            (for more details see class docstring).  Default is True.
        close_fig : bool, optional
            Figure is not drawn if True. Default is False.
        figsize : tuple
            figure size for plt.subplots, width, height in inches
            The default is (9, 8)
        Returns
        -------
        axes : matplotlib.axes.Axes
        """
        # select hazard with run_datetime
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        # plot histogram of all runs
        histbin_file_name = self.summary_str(run_datetime) + "_histbin" + ".svg"
        histbin_file_name_full = FORECAST_PLOT_DIR / histbin_file_name
        lower_bound = np.max(
            [
                np.floor(
                    np.log10(np.max([np.min(self._impact[haz_ind].at_event), 0.1]))
                ),
                0,
            ]
        )
        upper_bound = (
            np.max(
                [
                    np.ceil(
                        np.log10(np.max([np.max(self._impact[haz_ind].at_event), 0.1]))
                    ),
                    lower_bound + 5,
                ]
            )
            + 0.1
        )
        bins_log = np.arange(lower_bound, upper_bound, 0.5)
        bins = 10**bins_log
        fig = plt.figure(figsize=figsize)
        axes = fig.add_subplot(111)
        plt.xscale("log")
        plt.hist(
            [np.max([x, 1]) for x in self._impact[haz_ind].at_event],
            bins=bins,
            weights=np.ones(len(self._impact[haz_ind].at_event))
            / len(self._impact[haz_ind].at_event),
        )
        axes.yaxis.set_major_formatter(PercentFormatter(1))
        formatter = ScalarFormatter()
        formatter.set_scientific(False)  # formatter.format = "%:'.0f"
        axes.xaxis.set_major_formatter(formatter)

        x_ticklabels = list()
        x_ticks = list()
        ticklabel_dict = {
            pow(10, i): self._number_to_str(pow(10, i)) for i in range(0, 15)
        }  # turn int to str

        for i in np.arange(15):
            tick_i = 10.0**i
            x_ticks.append(tick_i)
            x_ticklabels.append(ticklabel_dict[tick_i])
        axes.xaxis.set_ticks(x_ticks)
        axes.xaxis.set_ticklabels(x_ticklabels)
        plt.xticks(rotation=15, horizontalalignment="right")
        plt.xlim([(10**-0.25) * bins[0], (10**0.25) * bins[-1]])

        lead_time_str = "{:.0f}".format(
            self.lead_time(run_datetime).days
            + self.lead_time(run_datetime).seconds / 60 / 60 / 24
        )
        title_dict = {
            "event_day": self.event_date.strftime("%a %d %b %Y 00-24UTC"),
            "run_start": (
                run_datetime.strftime("%d.%m.%Y %HUTC +") + lead_time_str + "d"
            ),
            "explain_text": ("total building damage"),
            "model_text": "CLIMADA IMPACT",
        }
        title_position = {
            "model_text": [0.13, 0.94],
            "explain_text": [0.13, 0.9],
            "event_day": [0.9, 0.94],
            "run_start": [0.9, 0.9],
        }
        left_right = {
            "model_text": "left",
            "explain_text": "left",
            "event_day": "right",
            "run_start": "right",
        }
        color = {
            "model_text": "k",
            "explain_text": "k",
            "event_day": "r",
            "run_start": "k",
        }
        for t_i in title_dict:
            plt.figtext(
                title_position[t_i][0],
                title_position[t_i][1],
                title_dict[t_i],
                fontsize="xx-large",
                color=color[t_i],
                ha=left_right[t_i],
            )

        plt.xlabel(
            "forecasted total damage for "
            + self.exposure_name
            + " ["
            + self._impact[haz_ind].unit
            + "]"
        )
        plt.ylabel("probability")
        plt.text(
            0.75,
            0.85,
            "mean damage:\nCHF "
            + self._number_to_str(self._impact[haz_ind].at_event.mean()),
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes.transAxes,
        )
        if save_fig:
            plt.savefig(histbin_file_name_full)
        if close_fig:
            plt.clf()
            plt.close(fig)
        return axes

    @staticmethod
    def _number_to_str(number):
        """Generate numbers as str (with thousand, million, trillion).
            Example: 1000 becomes 1 thousand

        Parameters
        ----------
        number : int

        Returns
        -------
        str
        """
        number_names = {
            1: "",
            1000: "thousand",
            1000000: "million",
            1000000000: "billion",
            1000000000000: "trillion",
        }
        [value], name = u_value_to_monetary_unit(number, abbreviations=number_names)
        return "{:.0f} {}".format(value, name)

    def plot_exceedence_prob(
        self,
        threshold,
        explain_str=None,
        run_datetime=None,
        save_fig=True,
        close_fig=False,
        polygon_file=None,
        polygon_file_crs="epsg:4326",
        proj=ccrs.PlateCarree(),
        figsize=(9, 13),
        adapt_fontsize=True,
    ):
        """plot exceedence map

        Parameters
        ----------
        threshold : float
            Threshold of impact unit for which exceedence probability
            should be plotted.
        explain_str : str, optional
            Short str which explains threshold, explain_str is included
            in the title of the figure.
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        save_fig : bool, optional
            Figure is saved if True, folder is within your configurable
            save_dir and filename is derived from the method summary_str()
            (for more details see class docstring).  Default is True.
        close_fig : bool, optional
            Figure not drawn if True. Default is False.
        polygon_file : str, optional
            Points to a .shp-file with polygons do be drawn as outlines on
            the plot, default is None to not draw the lines. please also
            specify the crs in the parameter polygon_file_crs.
        polygon_file_crs : str, optional
            String of pattern <provider>:<code> specifying
            the crs. has to be readable by pyproj.Proj. Default is
            'epsg:4326'.
        proj : ccrs
            coordinate reference system used in coordinates
            The default is ccrs.PlateCarree()
        figsize : tuple
            figure size for plt.subplots, width, height in inches
            The default is (9, 13)
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.

        Returns
        -------
        axes: cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        # select hazard with run_datetime
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        wind_map_file_name = (
            self.summary_str(run_datetime) + "_exceed_" + str(threshold) + "_map.jpeg"
        )
        wind_map_file_name_full = FORECAST_PLOT_DIR / wind_map_file_name
        lead_time_str = "{:.0f}".format(
            self.lead_time(run_datetime).days
            + self.lead_time(run_datetime).seconds / 60 / 60 / 24
        )
        title_dict = {
            "event_day": self.event_date.strftime("%a %d %b %Y 00-24UTC"),
            "run_start": (
                run_datetime.strftime("%d.%m.%Y %HUTC +") + lead_time_str + "d"
            ),
            "explain_text": (
                "threshold: " + str(threshold) + " " + self._impact[haz_ind].unit
            )
            if explain_str is None
            else explain_str,
            "model_text": "Exceedance probability map",
        }
        cbar_label = "probabilty of reaching threshold"
        fig, axes = self._plot_exc_prob(
            run_datetime,
            threshold,
            title_dict,
            cbar_label,
            proj,
            polygon_file=polygon_file,
            polygon_file_crs=polygon_file_crs,
            figsize=figsize,
            adapt_fontsize=adapt_fontsize,
        )
        if save_fig:
            plt.savefig(wind_map_file_name_full)
        if close_fig:
            plt.clf()
            plt.close(fig)
        return axes

    def _plot_exc_prob(
        self,
        run_datetime,
        threshold,
        title,
        cbar_label,
        proj=ccrs.PlateCarree(),
        polygon_file=None,
        polygon_file_crs="epsg:4326",
        mask=None,
        figsize=(9, 13),
        adapt_fontsize=True,
    ):
        """plot the probability of reaching a threshold"""
        # select hazard with run_datetime
        # pylint: disable=protected-access
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        extend = "neither"
        value = np.squeeze(
            np.asarray(
                (self._impact[haz_ind].imp_mat > threshold).sum(axis=0)
                / self._impact[haz_ind].event_id.size
            )
        )
        if mask is not None:
            value[np.invert(mask)] = np.nan
        coord = self._impact[haz_ind].coord_exp
        # Generate array of values used in each subplot
        array_sub = value
        shapes = True
        if not polygon_file:
            shapes = False
        var_name = cbar_label
        geo_coord = coord
        num_im, list_arr = u_plot._get_collection_arrays(array_sub)
        list_tit = to_list(num_im, title, "title")
        list_name = to_list(num_im, var_name, "var_name")
        list_coord = to_list(num_im, geo_coord, "geo_coord")

        kwargs = dict()
        kwargs["cmap"] = CMAP_WARNPROB
        kwargs["s"] = 5
        kwargs["marker"] = ","
        kwargs["norm"] = BoundaryNorm(np.linspace(0, 1, 11), CMAP_WARNPROB.N, clip=True)

        # Generate each subplot
        fig, axis_sub, _fontsize = u_plot.make_map(
            num_im, proj=proj, figsize=figsize, adapt_fontsize=adapt_fontsize
        )
        if not isinstance(axis_sub, np.ndarray):
            axis_sub = np.array([[axis_sub]])
        fig.set_size_inches(9, 8)
        for array_im, axis, tit, name, coord in zip(
            list_arr, axis_sub.flatten(), list_tit, list_name, list_coord
        ):
            if coord.shape[0] != array_im.size:
                raise ValueError(
                    "Size mismatch in input array: %s != %s."
                    % (coord.shape[0], array_im.size)
                )

            hex_bin = axis.scatter(
                coord[:, 1],
                coord[:, 0],
                c=array_im,
                transform=ccrs.PlateCarree(),
                **kwargs
            )
            if shapes:
                # add warning regions
                shp = shapereader.Reader(polygon_file)
                transformer = pyproj.Transformer.from_crs(
                    polygon_file_crs, self._impact[haz_ind].crs, always_xy=True
                )
                for geometry, _ in zip(shp.geometries(), shp.records()):
                    geom2 = shapely.ops.transform(transformer.transform, geometry)
                    axis.add_geometries(
                        [geom2],
                        crs=ccrs.PlateCarree(),
                        facecolor="none",
                        edgecolor="gray",
                    )

            # Create colorbar in this axis
            cbax = make_axes_locatable(axis).append_axes(
                "bottom", size="6.5%", pad=0.3, axes_class=plt.Axes
            )
            cbar = plt.colorbar(
                hex_bin, cax=cbax, orientation="horizontal", extend=extend
            )
            cbar.set_label(name)
            title_position = {
                "model_text": [0.02, 0.94],
                "explain_text": [0.02, 0.9],
                "event_day": [0.98, 0.94],
                "run_start": [0.98, 0.9],
            }
            left_right = {
                "model_text": "left",
                "explain_text": "left",
                "event_day": "right",
                "run_start": "right",
            }
            color = {
                "model_text": "k",
                "explain_text": "k",
                "event_day": "r",
                "run_start": "k",
            }
            for t_i in tit:
                plt.figtext(
                    title_position[t_i][0],
                    title_position[t_i][1],
                    tit[t_i],
                    fontsize="xx-large",
                    color=color[t_i],
                    ha=left_right[t_i],
                )
            extent = u_plot._get_borders(coord)
            axis.set_extent((extent), ccrs.PlateCarree())
        fig.tight_layout()
        return fig, axis_sub

    def plot_warn_map(
        self,
        polygon_file=None,
        polygon_file_crs="epsg:4326",
        thresholds="default",
        decision_level="exposure_point",
        probability_aggregation=0.5,
        area_aggregation=0.5,
        title="WARNINGS",
        explain_text="warn level based on thresholds",
        run_datetime=None,
        proj=ccrs.PlateCarree(),
        figsize=(9, 13),
        save_fig=True,
        close_fig=False,
        adapt_fontsize=True,
    ):
        """plot map colored with 5 warning colors for all regions in provided
        shape file.

        Parameters
        ----------
        polygon_file : str, optional
            path to shp-file containing warning region polygons
        polygon_file_crs : str, optional
            String of pattern <provider>:<code> specifying
            the crs. has to be readable by pyproj.Proj. Default is
            'epsg:4326'.
        thresholds : list of 4 floats, optional
            Thresholds for coloring region in second, third, forth
            and fifth warning color.
        decision_level : str, optional
            Either 'exposure_point'  or 'polygon'. Default value is
            'exposure_point'.
        probability_aggregation : float or str, optional
            Either a float between [0..1] spezifying a quantile
            or 'mean' or 'sum'. Default value is 0.5.
        area_aggregation : float or str.
            Either a float between [0..1] specifying a quantile
            or 'mean' or 'sum'. Default value is 0.5.
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        title : str, optional
            Default is 'WARNINGS'.
        explain_text : str, optional
            Defaut is 'warn level based on thresholds'.
        proj : ccrs
            coordinate reference system used in coordinates
        figsize : tuple
            figure size for plt.subplots, width, height in inches
            The default is (9, 13)
        save_fig : bool, optional
            Figure is saved if True, folder is within your configurable
            save_dir and filename is derived from the method summary_str()
            (for more details see class docstring).  Default is True.
        close_fig : bool, optional
            Figure is not drawn if True. The default is False.
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.

        Returns
        -------
        axes : cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        # select hazard with run_datetime
        if thresholds == "default":
            thresholds = [2, 3, 4, 5]
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        warn_map_file_name = self.summary_str(run_datetime) + "_warn_map.jpeg"
        warn_map_file_name_full = FORECAST_PLOT_DIR / warn_map_file_name
        decision_dict = {
            "probability_aggregation": probability_aggregation,
            "area_aggregation": area_aggregation,
        }
        lead_time_str = "{:.0f}".format(
            self.lead_time(run_datetime).days
            + self.lead_time(run_datetime).seconds / 60 / 60 / 24
        )
        title_dict = {
            "event_day": self.event_date.strftime("%a %d %b %Y 00-24UTC"),
            "run_start": (
                run_datetime.strftime("%d.%m.%Y %HUTC +") + lead_time_str + "d"
            ),
            "explain_text": explain_text,
            "model_text": title,
        }

        fig, axes = self._plot_warn(
            run_datetime,
            thresholds,
            decision_level,
            decision_dict,
            polygon_file,
            polygon_file_crs,
            title_dict,
            proj,
            figsize=figsize,
            adapt_fontsize=adapt_fontsize,
        )
        if save_fig:
            plt.savefig(warn_map_file_name_full)
        if close_fig:
            plt.clf()
            plt.close(fig)
        return axes

    def _plot_warn(
        self,
        run_datetime,
        thresholds,
        decision_level,
        decision_dict,
        polygon_file,
        polygon_file_crs,
        title,
        proj=ccrs.PlateCarree(),
        figsize=(9, 13),
        adapt_fontsize=True,
    ):
        """plotting the warning level of each warning region based on thresholds"""
        # select hazard with run_datetime
        # pylint: disable=protected-access
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]

        kwargs = dict()
        kwargs["cmap"] = CMAP_WARNPROB
        kwargs["s"] = 5
        kwargs["marker"] = ","
        kwargs["norm"] = BoundaryNorm(np.linspace(0, 1, 11), CMAP_WARNPROB.N, clip=True)

        # Generate each subplot
        fig, axis, _fontsize = u_plot.make_map(
            1, proj=proj, figsize=figsize, adapt_fontsize=adapt_fontsize
        )
        if isinstance(axis, np.ndarray):
            axis = axis[0]
        tit = title
        fig.set_size_inches(9, 8)

        # add warning regions
        shp = shapereader.Reader(polygon_file)
        transformer = pyproj.Transformer.from_crs(
            polygon_file_crs, self._impact[haz_ind].crs, always_xy=True
        )
        # checking the decision dict and define the corresponding functions
        if not (
            isinstance(decision_dict["probability_aggregation"], float)
            & isinstance(decision_dict["area_aggregation"], float)
        ):
            ValueError(
                " If decision_level is 'exposure_point',"
                + "parameters probability_aggregation and "
                + "area_aggregation of "
                + "Forecast.plot_warn_map() must both be "
                + "floats between [0..1]. Which each "
                + "specify quantiles."
            )
        decision_dict_functions = decision_dict.copy()
        for aggregation in decision_dict:
            if isinstance(decision_dict[aggregation], float):
                decision_dict_functions[aggregation] = np.percentile
            elif decision_dict[aggregation] == "sum":
                decision_dict_functions[aggregation] = np.sum
            elif decision_dict[aggregation] == "mean":
                decision_dict_functions[aggregation] = np.mean
            else:
                raise ValueError(
                    "Parameter area_aggregation of "
                    + "Forecast.plot_warn_map() must eiter be "
                    + "a float between [0..1], which "
                    + "specifys a quantile. or 'sum' or 'mean'."
                )

        for geometry, _ in zip(shp.geometries(), shp.records()):
            geom2 = shapely.ops.transform(transformer.transform, geometry)
            in_geom = u_coord.coord_on_land(
                lat=self._impact[haz_ind].coord_exp[:, 0],
                lon=self._impact[haz_ind].coord_exp[:, 1],
                land_geom=geom2,
            )
            if not in_geom.any():
                continue
            # decide warning level
            warn_level = 0
            for ind_i, warn_thres_i in enumerate(thresholds):
                if decision_level == "exposure_point":
                    # decision at each grid_point
                    probabilities = np.squeeze(
                        np.asarray(
                            (self._impact[haz_ind].imp_mat >= warn_thres_i).sum(axis=0)
                            / self._impact[haz_ind].event_id.size
                        )
                    )
                    # quantiles over probability
                    area = (
                        probabilities[in_geom]
                        >= decision_dict["probability_aggregation"]
                    ).sum()
                    # quantiles over area
                    if area >= (in_geom.sum() * decision_dict["area_aggregation"]):
                        warn_level = ind_i + 1
                elif decision_level == "polygon":
                    # aggregation over area
                    if isinstance(decision_dict["area_aggregation"], float):
                        value_per_member = decision_dict_functions["area_aggregation"](
                            self._impact[haz_ind].imp_mat[:, in_geom].todense(),
                            decision_dict["area_aggregation"],
                            axis=1,
                        )
                    else:
                        value_per_member = decision_dict_functions["area_aggregation"](
                            self._impact[haz_ind].imp_mat[:, in_geom].todense(), axis=1
                        )
                    # aggregation over members/probability
                    if isinstance(decision_dict["probability_aggregation"], float):
                        value_per_region = decision_dict_functions[
                            "probability_aggregation"
                        ](value_per_member, decision_dict["probability_aggregation"])
                    else:
                        value_per_region = decision_dict_functions[
                            "probability_aggregation"
                        ](value_per_member)
                    # warn level decision
                    if value_per_region >= warn_thres_i:
                        warn_level = ind_i + 1
                else:
                    raise ValueError(
                        "Parameter decision_level of "
                        + "Forecast.plot_warn_map() must eiter be "
                        + "'exposure_point' or 'polygon'."
                    )
            # plot warn_region with specific color (dependent on warning level)
            axis.add_geometries(
                [geom2],
                crs=ccrs.PlateCarree(),
                facecolor=COLORS_WARN[warn_level, :],
                edgecolor="gray",
            )

        # Create legend in this axis
        hazard_levels = [
            "1: Minimal or no hazard",
            "2: Moderate hazard",
            "3: Significant hazard",
            "4: Severe hazard",
            "5: Very severe hazard",
        ]
        legend_elements = [
            Patch(facecolor=COLORS_WARN[n, :], edgecolor="gray", label=hazard_level)
            for n, hazard_level in enumerate(hazard_levels)
        ]

        axis.legend(
            handles=legend_elements,
            loc="upper center",
            framealpha=0.5,
            bbox_to_anchor=(0.5, -0.02),
            ncol=3,
        )
        title_position = {
            "model_text": [0.02, 0.91],
            "explain_text": [0.02, 0.87],
            "event_day": [0.98, 0.91],
            "run_start": [0.98, 0.87],
        }
        left_right = {
            "model_text": "left",
            "explain_text": "left",
            "event_day": "right",
            "run_start": "right",
        }
        color = {
            "model_text": "k",
            "explain_text": "k",
            "event_day": "r",
            "run_start": "k",
        }
        for t_i in tit:
            plt.figtext(
                title_position[t_i][0],
                title_position[t_i][1],
                tit[t_i],
                fontsize="xx-large",
                color=color[t_i],
                ha=left_right[t_i],
            )

        extent = u_plot._get_borders(self._impact[haz_ind].coord_exp)
        axis.set_extent((extent), ccrs.PlateCarree())
        fig.tight_layout()
        return fig, axis

    def plot_hexbin_ei_exposure(self, run_datetime=None, figsize=(9, 13)):
        """plot the expected impact

        Parameters
        ----------
        run_datetime : datetime.datetime, optional
            Select the used hazard by the run_datetime,
            default is first element of attribute run_datetime.
        figsize : tuple
            figure size for plt.subplots, width, height in inches
            The default is (9, 13)
        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        # select hazard with run_datetime
        if run_datetime is None:
            run_datetime = self.run_datetime[0]
        haz_ind = np.argwhere(np.isin(self.run_datetime, run_datetime))[0][0]
        return self._impact[haz_ind].plot_hexbin_eai_exposure(figsize=figsize)
