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

Define Hazard Plotting Methods.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from deprecation import deprecated

import climada.util.plot as u_plot

LOGGER = logging.getLogger(__name__)

# pylint: disable=no-member


class HazardPlot:
    """
    Contains all plotting methods of the Hazard class
    """

    @deprecated(
        details="The use of Hazard.plot_rp_intensity is deprecated."
        "Use Hazard.local_exceedance_intensity and util.plot.plot_from_gdf instead."
    )
    def plot_rp_intensity(
        self,
        return_periods=(25, 50, 100, 250),
        smooth=True,
        axis=None,
        figsize=(9, 13),
        adapt_fontsize=True,
        **kwargs,
    ):
        """
        This function is deprecated,
        use Impact.local_exceedance_impact and util.plot.plot_from_gdf instead.

        Compute and plot hazard exceedance intensity maps for different
        return periods. Calls local_exceedance_inten.

        Parameters
        ----------
        return_periods: tuple(int), optional
            return periods to consider
        smooth: bool, optional
            smooth plot to plot.RESOLUTIONxplot.RESOLUTION
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: tuple, optional
            figure size for plt.subplots
        kwargs: optional
            arguments for pcolormesh matplotlib function used in event plots

        Returns
        -------
        axis, inten_stats:  matplotlib.axes._subplots.AxesSubplot, np.ndarray
            intenstats is return_periods.size x num_centroids
        """
        inten_stats = self.local_exceedance_intensity(return_periods)[0].values[:, 1:].T
        inten_stats = inten_stats.astype(float)
        colbar_name = "Intensity (" + self.units + ")"
        title = list()
        for ret in return_periods:
            title.append("Return period: " + str(ret) + " years")
        axis = u_plot.geo_im_from_array(
            inten_stats,
            self.centroids.coord,
            colbar_name,
            title,
            smooth=smooth,
            axes=axis,
            figsize=figsize,
            adapt_fontsize=adapt_fontsize,
            **kwargs,
        )
        return axis, inten_stats

    def plot_intensity(
        self,
        event=None,
        centr=None,
        smooth=True,
        axis=None,
        adapt_fontsize=True,
        **kwargs,
    ):
        """Plot intensity values for a selected event or centroid.

        Parameters
        ----------
        event: int or str, optional
            If event > 0, plot intensities of
            event with id = event. If event = 0, plot maximum intensity in
            each centroid. If event < 0, plot abs(event)-largest event. If
            event is string, plot events with that name.
        centr: int or tuple, optional
            If centr > 0, plot intensity
            of all events at centroid with id = centr. If centr = 0,
            plot maximum intensity of each event. If centr < 0,
            plot abs(centr)-largest centroid where higher intensities
            are reached. If tuple with (lat, lon) plot intensity of nearest
            centroid.
        smooth: bool, optional
            Rescale data to RESOLUTIONxRESOLUTION pixels (see constant
            in module `climada.util.plot`)
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs: optional
            arguments for pcolormesh matplotlib function
            used in event plots or for plot function used in centroids plots

        Returns
        -------
            matplotlib.axes._subplots.AxesSubplot

        Raises
        ------
            ValueError
        """
        col_label = f"Intensity ({self.units})"
        crs_epsg, _ = u_plot.get_transformation(self.centroids.geometry.crs)
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(
                event,
                self.intensity,
                col_label,
                smooth,
                crs_epsg,
                axis,
                adapt_fontsize=adapt_fontsize,
                **kwargs,
            )
        if centr is not None:
            if isinstance(centr, tuple):
                _, _, centr = self.centroids.get_closest_point(centr[0], centr[1])
            return self._centr_plot(centr, self.intensity, col_label, axis, **kwargs)

        raise ValueError("Provide one event id or one centroid id.")

    def plot_fraction(self, event=None, centr=None, smooth=True, axis=None, **kwargs):
        """Plot fraction values for a selected event or centroid.

        Parameters
        ----------
        event: int or str, optional
            If event > 0, plot fraction of event
            with id = event. If event = 0, plot maximum fraction in each
            centroid. If event < 0, plot abs(event)-largest event. If event
            is string, plot events with that name.
        centr: int or tuple, optional
            If centr > 0, plot fraction
            of all events at centroid with id = centr. If centr = 0,
            plot maximum fraction of each event. If centr < 0,
            plot abs(centr)-largest centroid where highest fractions
            are reached. If tuple with (lat, lon) plot fraction of nearest
            centroid.
        smooth: bool, optional
            Rescale data to RESOLUTIONxRESOLUTION pixels (see constant
            in module `climada.util.plot`)
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs: optional
            arguments for pcolormesh matplotlib function
            used in event plots or for plot function used in centroids plots

        Returns
        -------
            matplotlib.axes._subplots.AxesSubplot

        Raises
        ------
            ValueError
        """
        col_label = "Fraction"
        if event is not None:
            if isinstance(event, str):
                event = self.get_event_id(event)
            return self._event_plot(
                event, self.fraction, col_label, smooth, axis, **kwargs
            )
        if centr is not None:
            if isinstance(centr, tuple):
                _, _, centr = self.centroids.get_closest_point(centr[0], centr[1])
            return self._centr_plot(centr, self.fraction, col_label, axis, **kwargs)

        raise ValueError("Provide one event id or one centroid id.")

    def _event_plot(
        self,
        event_id,
        mat_var,
        col_name,
        smooth,
        crs_espg,
        axis=None,
        figsize=(9, 13),
        adapt_fontsize=True,
        **kwargs,
    ):
        """Plot an event of the input matrix.

        Parameters
        ----------
        event_id: int or np.array(int)
            If event_id > 0, plot mat_var of
            event with id = event_id. If event_id = 0, plot maximum
            mat_var in each centroid. If event_id < 0, plot
            abs(event_id)-largest event.
        mat_var: sparse matrix
            Sparse matrix where each row is an event
        col_name: sparse matrix
            Colorbar label
        smooth: bool, optional
            smooth plot to plot.RESOLUTIONxplot.RESOLUTION
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        figsize: tuple, optional
            figure size for plt.subplots
        kwargs: optional
            arguments for pcolormesh matplotlib function

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        if not isinstance(event_id, np.ndarray):
            event_id = np.array([event_id])
        array_val = list()
        l_title = list()
        for ev_id in event_id:
            if ev_id > 0:
                try:
                    event_pos = np.where(self.event_id == ev_id)[0][0]
                except IndexError as err:
                    raise ValueError(f"Wrong event id: {ev_id}.") from err
                im_val = mat_var[event_pos, :].toarray().transpose()
                title = (
                    f"Event ID {self.event_id[event_pos]}: {self.event_name[event_pos]}"
                )
            elif ev_id < 0:
                max_inten = np.asarray(np.sum(mat_var, axis=1)).reshape(-1)
                event_pos = np.argpartition(max_inten, ev_id)[ev_id:]
                event_pos = event_pos[np.argsort(max_inten[event_pos])][0]
                im_val = mat_var[event_pos, :].toarray().transpose()
                title = (
                    f"{np.abs(ev_id)}-largest Event. ID {self.event_id[event_pos]}:"
                    f" {self.event_name[event_pos]}"
                )
            else:
                im_val = np.max(mat_var, axis=0).toarray().transpose()
                title = f"{self.haz_type} max intensity at each point"

            array_val.append(im_val)
            l_title.append(title)

        return u_plot.geo_im_from_array(
            array_val,
            self.centroids.coord,
            col_name,
            l_title,
            smooth=smooth,
            axes=axis,
            figsize=figsize,
            proj=crs_espg,
            adapt_fontsize=adapt_fontsize,
            **kwargs,
        )

    def _centr_plot(self, centr_idx, mat_var, col_name, axis=None, **kwargs):
        """Plot a centroid of the input matrix.

        Parameters
        ----------
        centr_id: int
            If centr_id > 0, plot mat_var
            of all events at centroid with id = centr_id. If centr_id = 0,
            plot maximum mat_var of each event. If centr_id < 0,
            plot abs(centr_id)-largest centroid where highest mat_var
            are reached.
        mat_var: sparse matrix
            Sparse matrix where each column represents
            a centroid
        col_name: sparse matrix
            Colorbar label
        axis: matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs: optional
            arguments for plot matplotlib function

        Returns
        -------
            matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
        """
        coord = self.centroids.coord
        if centr_idx > 0:
            try:
                centr_pos = centr_idx
            except IndexError as err:
                raise ValueError(f"Wrong centroid id: {centr_idx}.") from err
            array_val = mat_var[:, centr_pos].toarray()
            title = (
                f"Centroid {centr_idx}:"
                f" ({np.around(coord[centr_pos, 0], 3)}, {np.around(coord[centr_pos, 1],3)})"
            )
        elif centr_idx < 0:
            max_inten = np.asarray(np.sum(mat_var, axis=0)).reshape(-1)
            centr_pos = np.argpartition(max_inten, centr_idx)[centr_idx:]
            centr_pos = centr_pos[np.argsort(max_inten[centr_pos])][0]
            array_val = mat_var[:, centr_pos].toarray()

            title = (
                f"{np.abs(centr_idx)}-largest Centroid. {centr_pos}:"
                f" ({np.around(coord[centr_pos, 0], 3)}, {np.around(coord[centr_pos, 1], 3)})"
            )
        else:
            array_val = np.max(mat_var, axis=1).toarray()
            title = f"{self.haz_type} max intensity at each event"

        if not axis:
            _, axis = plt.subplots(1)
        if "color" not in kwargs:
            kwargs["color"] = "b"
        axis.set_title(title)
        axis.set_xlabel("Event number")
        axis.set_ylabel(str(col_name))
        axis.plot(range(len(array_val)), array_val, **kwargs)
        axis.set_xlim([0, len(array_val)])
        return axis
