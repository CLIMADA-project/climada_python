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

Define Impact and ImpactFreqCurve classes.
"""

__all__ = ['ImpactFreqCurve', 'Impact']

from dataclasses import dataclass, field
import logging
import copy
import csv
import warnings
import datetime as dt
from itertools import zip_longest
from typing import Any, Iterable, Union
from collections.abc import Collection
from pathlib import Path

import contextily as ctx
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import xlsxwriter
from tqdm import tqdm
import h5py

from climada.entity import Exposures, Tag
from climada.hazard import Tag as TagHaz
import climada.util.plot as u_plot
from climada import CONFIG
from climada.util.constants import DEF_CRS, CMAP_IMPACT, DEF_FREQ_UNIT
import climada.util.coordinates as u_coord
import climada.util.dates_times as u_dt
from climada.util.select import get_attributes_with_matching_dimension

LOGGER = logging.getLogger(__name__)

class Impact():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes
    ----------
    tag : dict
        dictionary of tags of exposures, impact functions set and
        hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHaz()}
    event_id : np.array
        id (>0) of each hazard event
    event_name : list
        list name of each hazard event
    date : np.array
        date if events as integer date corresponding to the
        proleptic Gregorian ordinal, where January 1 of year 1 has
        ordinal 1 (ordinal format of datetime library)
    coord_exp : np.array
        exposures coordinates [lat, lon] (in degrees)
    eai_exp : np.array
        expected impact for each exposure within a period of 1/frequency_unit
    at_event : np.array
        impact for each hazard event
    frequency : np.array
        frequency of event
    frequency_unit : str
        frequency unit used (given by hazard), default is '1/year'
    tot_value : float
        total exposure value affected
    aai_agg : float
        average impact within a period of 1/frequency_unit (aggregated)
    unit : str
        value unit used (given by exposures unit)
    imp_mat : sparse.csr_matrix
        matrix num_events x num_exp with impacts.
        only filled if save_mat is True in calc()
    """

    def __init__(self,
                 event_id=None,
                 event_name=None,
                 date=None,
                 frequency=None,
                 frequency_unit=DEF_FREQ_UNIT,
                 coord_exp=None,
                 crs=DEF_CRS,
                 eai_exp=None,
                 at_event=None,
                 tot_value=0,
                 aai_agg=0,
                 unit='',
                 imp_mat=None,
                 tag=None):
        """
        Init Impact object

        Parameters
        ----------
        event_id : np.array, optional
            id (>0) of each hazard event
        event_name : list, optional
            list name of each hazard event
        date : np.array, optional
            date if events as integer date corresponding to the
            proleptic Gregorian ordinal, where January 1 of year 1 has
            ordinal 1 (ordinal format of datetime library)
        frequency : np.array, optional
            frequency of event impact for each hazard event
        frequency_unit : np.array, optional
            frequency unit, default: '1/year'
        coord_exp : np.array, optional
            exposures coordinates [lat, lon] (in degrees)
        crs : Any, optional
            coordinate reference system
        eai_exp : np.array, optional
            expected impact for each exposure within a period of 1/frequency_unit
        at_event : np.array, optional
            impact for each hazard event
        tot_value : float, optional
            total exposure value affected
        aai_agg : float, optional
            average impact within a period of 1/frequency_unit (aggregated)
        unit : str, optional
            value unit used (given by exposures unit)
        imp_mat : sparse.csr_matrix, optional
            matrix num_events x num_exp with impacts.
        tag : dict, optional
            dictionary of tags of exposures, impact functions set and
            hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHaz()}
        """

        self.tag = tag or {}
        self.event_id = np.array([], int) if event_id is None else event_id
        self.event_name = [] if event_name is None else event_name
        self.date = np.array([], int) if date is None else date
        self.coord_exp = np.array([], float) if coord_exp is None else coord_exp
        self.crs = crs
        self.eai_exp = np.array([], float) if eai_exp is None else eai_exp
        self.at_event = np.array([], float) if at_event is None else at_event
        self.frequency = np.array([],float) if frequency is None else frequency
        self.frequency_unit = frequency_unit
        self.tot_value = tot_value
        self.aai_agg = aai_agg
        self.unit = unit

        if len(self.event_id) != len(self.event_name):
            raise AttributeError(
                f'Hazard event ids {len(self.event_id)} and event names'
                f' {len(self.event_name)} are not of the same length')
        if len(self.event_id) != len(self.date):
            raise AttributeError(
                f'Hazard event ids {len(self.event_id)} and event dates'
                f' {len(self.date)} are not of the same length')
        if len(self.event_id) != len(self.frequency):
            raise AttributeError(
                f'Hazard event ids {len(self.event_id)} and event frequency'
                f' {len(self.frequency)} are not of the same length')
        if len(self.event_id) != len(self.at_event):
            raise AttributeError(
                f'Number of hazard event ids {len(self.event_id)} is different '
                f'from number of at_event values {len(self.at_event)}')
        if len(self.coord_exp) != len(self.eai_exp):
            raise AttributeError('Number of exposures points is different from'
                                 'number of eai_exp values')
        if imp_mat is not None:
            self.imp_mat = imp_mat
            if self.imp_mat.size > 0:
                if len(self.event_id) != self.imp_mat.shape[0]:
                    raise AttributeError(
                        f'The number of rows {imp_mat.shape[0]} of the impact ' +
                        f'matrix is inconsistent with the number {len(event_id)} '
                        'of hazard events.')
                if len(self.coord_exp) != self.imp_mat.shape[1]:
                    raise AttributeError(
                        f'The number of columns {imp_mat.shape[1]} of the impact' +
                        f' matrix is inconsistent with the number {len(coord_exp)}'
                        ' exposures points.')
        else:
            self.imp_mat = sparse.csr_matrix(np.empty((0, 0)))



    def calc(self, exposures, impact_funcs, hazard, save_mat=False, assign_centroids=True):
        """This function is deprecated, use ``ImpactCalc.impact`` instead.
        """
        LOGGER.warning("The use of Impact().calc() is deprecated."
                       " Use ImpactCalc().impact() instead.")
        from climada.engine.impact_calc import ImpactCalc
        impcalc = ImpactCalc(exposures, impact_funcs, hazard)
        self.__dict__ = impcalc.impact(
            save_mat=save_mat,
            assign_centroids=assign_centroids
        ).__dict__

#TODO: new name
    @classmethod
    def from_eih(cls, exposures, impfset, hazard,
                 at_event, eai_exp, aai_agg, imp_mat=None):
        """
        Set Impact attributes from precalculated impact metrics.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposure used to compute imp_mat
        impfset: climada.entity.ImpactFuncSet
            impact functions set used to compute imp_mat
        hazard : climada.Hazard
            hazard used to compute imp_mat
        at_event : np.array
            impact for each hazard event
        eai_exp : np.array
            expected impact for each exposure within a period of 1/frequency_unit
        aai_agg : float
            average impact within a period of 1/frequency_unit (aggregated)
        imp_mat : sparse.csr_matrix, optional
            matrix num_events x num_exp with impacts.
            Default is None (empty sparse csr matrix).

        Returns
        -------
        climada.engine.impact.Impact
            impact with all risk metrics set based on the given impact matrix
        """
        return cls(
            event_id = hazard.event_id,
            event_name = hazard.event_name,
            date = hazard.date,
            frequency = hazard.frequency,
            frequency_unit = hazard.frequency_unit,
            coord_exp = np.stack([exposures.gdf.latitude.values,
                                  exposures.gdf.longitude.values],
                                 axis=1),
            crs = exposures.crs,
            unit = exposures.value_unit,
            tot_value = exposures.affected_total_value(hazard),
            eai_exp = eai_exp,
            at_event = at_event,
            aai_agg = aai_agg,
            imp_mat = imp_mat if imp_mat is not None else sparse.csr_matrix((0, 0)),
            tag = {'exp': exposures.tag,
                   'impf_set': impfset.tag,
                   'haz': hazard.tag
                   }
            )

    def transfer_risk(self, attachment, cover):
        """Compute the risk transfer for the full portfolio. This is the risk
        of the full portfolio summed over all events. For each
        event, the transfered risk amounts to the impact minus the attachment
        (but maximally equal to the cover) multiplied with the probability
        of the event.

        Parameters
        ----------
        attachment : float
            attachment per event for entire portfolio.
        cover : float
            cover per event for entire portfolio.

        Returns
        -------
        transfer_at_event : np.array
            risk transfered per event
        transfer_aai_agg : float
            average risk within a period of 1/frequency_unit, transfered
        """
        transfer_at_event = np.minimum(np.maximum(self.at_event - attachment, 0), cover)
        transfer_aai_agg = np.sum(transfer_at_event * self.frequency)
        return transfer_at_event, transfer_aai_agg

    def residual_risk(self, attachment, cover):
        """Compute the residual risk after application of insurance
        attachment and cover to entire portfolio. This is the residual risk
        of the full portfolio summed over all events. For each
        event, the residual risk is obtained by subtracting the transfered risk
        from the trom the total risk per event.
        of the event.

        Parameters
        ----------
        attachment : float
            attachment per event for entire portfolio.
        cover : float
            cover per event for entire portfolio.

        Returns
        -------
        residual_at_event : np.array
            residual risk per event
        residual_aai_agg : float
            average residual risk within a period of 1/frequency_unit

        See also
        --------
        transfer_risk: compute the transfer risk per portfolio.

        """
        transfer_at_event, _ = self.transfer_risk(attachment, cover)
        residual_at_event = np.maximum(self.at_event - transfer_at_event, 0)
        residual_aai_agg = np.sum(residual_at_event * self.frequency)
        return residual_at_event, residual_aai_agg

#TODO: rewrite and deprecate method
    def calc_risk_transfer(self, attachment, cover):
        """Compute traaditional risk transfer over impact. Returns new impact
        with risk transfer applied and the insurance layer resulting
        Impact metrics.

        Parameters
        ----------
        attachment : float
            (deductible)
        cover : float

        Returns
        -------
        climada.engine.impact.Impact
        """
        new_imp = copy.deepcopy(self)
        if attachment or cover:
            imp_layer = np.minimum(np.maximum(new_imp.at_event - attachment, 0), cover)
            new_imp.at_event = np.maximum(new_imp.at_event - imp_layer, 0)
            new_imp.aai_agg = np.sum(new_imp.at_event * new_imp.frequency)
            # next values are no longer valid
            new_imp.eai_exp = np.array([])
            new_imp.coord_exp = np.array([])
            new_imp.imp_mat = sparse.csr_matrix((0, 0))
            # insurance layer metrics
            risk_transfer = copy.deepcopy(new_imp)
            risk_transfer.at_event = imp_layer
            risk_transfer.aai_agg = np.sum(imp_layer * new_imp.frequency)
            return new_imp, risk_transfer

        return new_imp, Impact()

    def impact_per_year(self, all_years=True, year_range=None):
        """Calculate yearly impact from impact data.

        Note: the impact in a given year is summed over all events.
        Thus, the impact in a given year can be larger than the
        total affected exposure value.

        Parameters
        ----------
        all_years : boolean, optional
            return values for all years between first and
            last year with event, including years without any events.
            Default: True
        year_range : tuple or list with integers, optional
            start and end year
        Returns
        -------
        year_set : dict
            Key=year, value=Summed impact per year.
        """
        if year_range is None:
            year_range = []

        orig_year = np.array([dt.datetime.fromordinal(date).year
                              for date in self.date])
        if orig_year.size == 0 and len(year_range) == 0:
            return dict()
        if orig_year.size == 0 or (len(year_range) > 0 and all_years):
            years = np.arange(min(year_range), max(year_range) + 1)
        elif all_years:
            years = np.arange(min(orig_year), max(orig_year) + 1)
        else:
            years = np.array(sorted(np.unique(orig_year)))
        if not len(year_range) == 0:
            years = years[years >= min(year_range)]
            years = years[years <= max(year_range)]

        year_set = dict()

        for year in years:
            year_set[year] = sum(self.at_event[orig_year == year])
        return year_set

    def calc_impact_year_set(self,all_years=True, year_range=None):
        """This function is deprecated, use Impact.impact_per_year instead."""
        LOGGER.warning("The use of Impact.calc_impact_year_set is deprecated."
                       "Use Impact.impact_per_year instead.")
        return self.impact_per_year(all_years=all_years, year_range=year_range)

#TODO: rewrite and deprecate method
    def local_exceedance_imp(self, return_periods=(25, 50, 100, 250)):
        """Compute exceedance impact map for given return periods.
        Requires attribute imp_mat.

        Parameters
        ----------
        return_periods : Any, optional
            return periods to consider
            Dafault is (25, 50, 100, 250)

        Returns
        -------
        np.array
        """
        LOGGER.info('Computing exceedance impact map for return periods: %s',
                    return_periods)
        if self.imp_mat.size == 0:
            raise ValueError('Attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')
        num_cen = self.imp_mat.shape[1]
        imp_stats = np.zeros((len(return_periods), num_cen))
        cen_step = CONFIG.max_matrix_size.int() // self.imp_mat.shape[0]
        if not cen_step:
            raise ValueError('Increase max_matrix_size configuration parameter to > '
                             f'{self.imp_mat.shape[0]}')
        # separte in chunks
        chk = -1
        for chk in range(int(num_cen / cen_step)):
            self._loc_return_imp(np.array(return_periods),
                                 self.imp_mat[:, chk * cen_step:(chk + 1) * cen_step].toarray(),
                                 imp_stats[:, chk * cen_step:(chk + 1) * cen_step])
        self._loc_return_imp(np.array(return_periods),
                             self.imp_mat[:, (chk + 1) * cen_step:].toarray(),
                             imp_stats[:, (chk + 1) * cen_step:])

        return imp_stats

    def calc_freq_curve(self, return_per=None):
        """Compute impact exceedance frequency curve.

        Parameters
        ----------
        return_per : np.array, optional
            return periods where to compute
            the exceedance impact. Use impact's frequencies if not provided

        Returns
        -------
        ImpactFreqCurve
        """
        # Sort descendingly the impacts per events
        sort_idxs = np.argsort(self.at_event)[::-1]
        # Calculate exceedence frequency
        exceed_freq = np.cumsum(self.frequency[sort_idxs])
        # Set return period and impact exceeding frequency
        ifc_return_per = 1 / exceed_freq[::-1]
        ifc_impact = self.at_event[sort_idxs][::-1]

        if return_per is not None:
            interp_imp = np.interp(return_per, ifc_return_per, ifc_impact)
            ifc_return_per = return_per
            ifc_impact = interp_imp

        return ImpactFreqCurve(
            tag=self.tag,
            return_per=ifc_return_per,
            impact=ifc_impact,
            unit=self.unit,
            frequency_unit=self.frequency_unit,
            label='Exceedance frequency curve'
        )

    def _eai_title(self):
        if self.frequency_unit in ['1/year', 'annual', '1/y', '1/a']:
            return 'Expected annual impact'
        if self.frequency_unit in ['1/day', 'daily', '1/d']:
            return 'Expected daily impact'
        if self.frequency_unit in ['1/month', 'monthly', '1/m']:
            return 'Expected monthly impact'
        return f'Expected impact ({self.frequency_unit})'

    def plot_scatter_eai_exposure(self, mask=None, ignore_zero=False,
                                  pop_name=True, buffer=0.0, extend='neither',
                                  axis=None, adapt_fontsize=True, **kwargs):
        """Plot scatter expected impact within a period of 1/frequency_unit of each exposure.

        Parameters
        ----------
        mask : np.array, optional
            mask to apply to eai_exp plotted.
        ignore_zero : bool, optional
            flag to indicate if zero and negative
            values are ignored in plot. Default: False
        pop_name : bool, optional
            add names of the populated places
        buffer : float, optional
            border to add to coordinates.
            Default: 1.0.
        extend : str
            optional extend border colorbar with arrows.
            [ 'neither' | 'both' | 'min' | 'max' ]
        axis : matplotlib.axes.Axes, optional
            axis to use
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.
        kwargs : dict, optional
            arguments for hexbin matplotlib function

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_IMPACT

        eai_exp = self._build_exp()
        axis = eai_exp.plot_scatter(mask, ignore_zero, pop_name, buffer,
                                    extend, axis=axis, adapt_fontsize=adapt_fontsize, **kwargs)
        axis.set_title(self._eai_title())
        return axis

    def plot_hexbin_eai_exposure(self, mask=None, ignore_zero=False,
                                 pop_name=True, buffer=0.0, extend='neither',
                                 axis=None, adapt_fontsize=True, **kwargs):
        """Plot hexbin expected impact within a period of 1/frequency_unit of each exposure.

        Parameters
        ----------
            mask : np.array, optional
                mask to apply to eai_exp plotted.
            ignore_zero : bool, optional
                flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name : bool, optional
                add names of the populated places
            buffer : float, optional
                border to add to coordinates.
                Default: 1.0.
            extend : str, optional
                extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            axis : matplotlib.axes.Axes, optional
                axis to use
            adapt_fontsize : bool, optional
                If set to true, the size of the fonts will be adapted to the size of the figure.
                Otherwise the default matplotlib font size is used. Default: True
            kwargs : dict, optional
                arguments for hexbin matplotlib function

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_IMPACT

        eai_exp = self._build_exp()
        axis = eai_exp.plot_hexbin(mask, ignore_zero, pop_name, buffer,
                                   extend, axis=axis, adapt_fontsize=adapt_fontsize, **kwargs)
        axis.set_title(self._eai_title())
        return axis

    def plot_raster_eai_exposure(self, res=None, raster_res=None, save_tiff=None,
                                 raster_f=lambda x: np.log10((np.fmax(x + 1, 1))),
                                 label='value (log10)', axis=None, adapt_fontsize=True,
                                 **kwargs):
        """Plot raster expected impact within a period of 1/frequency_unit of each exposure.

        Parameters
        ----------
        res : float, optional
            resolution of current data in units of latitude
            and longitude, approximated if not provided.
        raster_res : float, optional
            desired resolution of the raster
        save_tiff :  str, optional
            file name to save the raster in tiff
            format, if provided
        raster_f : lambda function
            transformation to use to data. Default: log10 adding 1.
        label : str
            colorbar label
        axis : matplotlib.axes.Axes, optional
            axis to use
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.
        kwargs : dict, optional
            arguments for imshow matplotlib function

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        # we need to set geometry points because the `plot_raster` method accesses the
        # exposures' `gdf.crs` property, which raises an error when geometry is not set
        eai_exp.set_geometry_points()
        axis = eai_exp.plot_raster(res, raster_res, save_tiff, raster_f,
                                   label, axis=axis, adapt_fontsize=adapt_fontsize, **kwargs)
        axis.set_title(self._eai_title())
        return axis

    def plot_basemap_eai_exposure(self, mask=None, ignore_zero=False, pop_name=True,
                                  buffer=0.0, extend='neither', zoom=10,
                                  url=ctx.providers.Stamen.Terrain,
                                  axis=None, **kwargs):
        """Plot basemap expected impact of each exposure within a period of 1/frequency_unit.

        Parameters
        ----------
        mask : np.array, optional
            mask to apply to eai_exp plotted.
        ignore_zero : bool, optional
            flag to indicate if zero and negative
            values are ignored in plot. Default: False
        pop_name : bool, optional
            add names of the populated places
        buffer : float, optional
            border to add to coordinates. Default: 0.0.
        extend : str, optional
            extend border colorbar with arrows.
            [ 'neither' | 'both' | 'min' | 'max' ]
        zoom : int, optional
            zoom coefficient used in the satellite image
        url : str, optional
            image source, e.g. ctx.providers.OpenStreetMap.Mapnik
        axis : matplotlib.axes.Axes, optional
            axis to use
        kwargs : dict, optional
            arguments for scatter matplotlib function, e.g.
            cmap='Greys'. Default: 'Wistia'

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_IMPACT
        eai_exp = self._build_exp()
        axis = eai_exp.plot_basemap(mask, ignore_zero, pop_name, buffer,
                                    extend, zoom, url, axis=axis, **kwargs)
        axis.set_title(self._eai_title())
        return axis

    def plot_hexbin_impact_exposure(self, event_id=1, mask=None, ignore_zero=False,
                                    pop_name=True, buffer=0.0, extend='neither',
                                    axis=None, adapt_fontsize=True, **kwargs):
        """Plot hexbin impact of an event at each exposure.
        Requires attribute imp_mat.

        Parameters
        ----------
        event_id : int, optional
            id of the event for which to plot the impact.
            Default: 1.
        mask : np.array, optional
            mask to apply to impact plotted.
        ignore_zero : bool, optional
            flag to indicate if zero and negative
            values are ignored in plot. Default: False
        pop_name : bool, optional
            add names of the populated places
        buffer : float, optional
            border to add to coordinates.
            Default: 1.0.
        extend : str, optional
            extend border colorbar with arrows.
            [ 'neither' | 'both' | 'min' | 'max' ]
        axis : matplotlib.axes.Axes
            optional axis to use
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.
        kwargs : dict, optional
            arguments for hexbin matplotlib function

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if self.imp_mat.size == 0:
            raise ValueError('Attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_IMPACT
        impact_at_events_exp = self._build_exp_event(event_id)
        axis = impact_at_events_exp.plot_hexbin(mask, ignore_zero, pop_name,
                                                buffer, extend, axis=axis,
                                                adapt_fontsize=adapt_fontsize,
                                                **kwargs)

        return axis

    def plot_basemap_impact_exposure(self, event_id=1, mask=None, ignore_zero=False,
                                     pop_name=True, buffer=0.0, extend='neither', zoom=10,
                                     url=ctx.providers.Stamen.Terrain,
                                     axis=None, **kwargs):
        """Plot basemap impact of an event at each exposure.
        Requires attribute imp_mat.

        Parameters
        ----------
        event_id : int, optional
            id of the event for which to plot the impact.
            Default: 1.
        mask : np.array, optional
            mask to apply to impact plotted.
        ignore_zero : bool, optional
            flag to indicate if zero and negative
            values are ignored in plot. Default: False
        pop_name : bool, optional
            add names of the populated places
        buffer : float, optional
            border to add to coordinates. Default: 0.0.
        extend : str, optional
            extend border colorbar with arrows.
            [ 'neither' | 'both' | 'min' | 'max' ]
        zoom : int, optional
            zoom coefficient used in the satellite image
        url : str, optional
            image source, e.g. ctx.providers.OpenStreetMap.Mapnik
        axis : matplotlib.axes.Axes, optional
            axis to use
        kwargs : dict, optional
            arguments for scatter matplotlib function, e.g. cmap='Greys'.
            Default: 'Wistia'

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if self.imp_mat.size == 0:
            raise ValueError('Attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')

        if event_id not in self.event_id:
            raise ValueError(f'Event ID {event_id} not found')
        if 'cmap' not in kwargs:
            kwargs['cmap'] = CMAP_IMPACT
        impact_at_events_exp = self._build_exp_event(event_id)
        axis = impact_at_events_exp.plot_basemap(mask, ignore_zero, pop_name,
                                                 buffer, extend, zoom, url, axis=axis, **kwargs)

        return axis

    def plot_rp_imp(self, return_periods=(25, 50, 100, 250),
                    log10_scale=True, smooth=True, axis=None, **kwargs):
        """Compute and plot exceedance impact maps for different return periods.
        Calls local_exceedance_imp.

        Parameters
        ----------
        return_periods : tuple of int, optional
            return periods to consider. Default: (25, 50, 100, 250)
        log10_scale : boolean, optional
            plot impact as log10(impact). Default: True
        smooth : bool, optional
            smooth plot to plot.RESOLUTIONxplot.RESOLUTION. Default: True
        kwargs : dict, optional
            arguments for pcolormesh matplotlib function
            used in event plots

        Returns
        -------
        axis : matplotlib.axes.Axes
        imp_stats : np.array
            return_periods.size x num_centroids
        """
        imp_stats = self.local_exceedance_imp(np.array(return_periods))
        if imp_stats.size == 0:
            raise ValueError('Error: Attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True')
        if log10_scale:
            if np.min(imp_stats) < 0:
                imp_stats_log = np.log10(abs(imp_stats) + 1)
                colbar_name = 'Log10(abs(Impact)+1) (' + self.unit + ')'
            elif np.min(imp_stats) < 1:
                imp_stats_log = np.log10(imp_stats + 1)
                colbar_name = 'Log10(Impact+1) (' + self.unit + ')'
            else:
                imp_stats_log = np.log10(imp_stats)
                colbar_name = 'Log10(Impact) (' + self.unit + ')'
        else:
            imp_stats_log = imp_stats
            colbar_name = 'Impact (' + self.unit + ')'
        title = list()
        for ret in return_periods:
            title.append('Return period: ' + str(ret) + ' years')
        axis = u_plot.geo_im_from_array(imp_stats_log, self.coord_exp,
                                        colbar_name, title, smooth=smooth, axes=axis, **kwargs)

        return axis, imp_stats

    def write_csv(self, file_name):
        """Write data into csv file. imp_mat is not saved.

        Parameters
        ----------
        file_name : str
            absolute path of the file
        """
        LOGGER.info('Writing %s', file_name)
        with open(file_name, "w", encoding='utf-8') as imp_file:
            imp_wr = csv.writer(imp_file)
            imp_wr.writerow(["tag_hazard", "tag_exposure", "tag_impact_func",
                             "unit", "tot_value", "aai_agg", "event_id",
                             "event_name", "event_date", "event_frequency", "frequency_unit",
                             "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"])
            csv_data = [[[self.tag['haz'].haz_type], [self.tag['haz'].file_name],
                         [self.tag['haz'].description]],
                        [[self.tag['exp'].file_name], [self.tag['exp'].description]],
                        [[self.tag['impf_set'].file_name], [self.tag['impf_set'].description]],
                        [self.unit], [self.tot_value], [self.aai_agg],
                        self.event_id, self.event_name, self.date,
                        self.frequency, [self.frequency_unit], self.at_event,
                        self.eai_exp, self.coord_exp[:, 0], self.coord_exp[:, 1],
                        [str(self.crs)]]
            for values in zip_longest(*csv_data):
                imp_wr.writerow(values)

    def write_excel(self, file_name):
        """Write data into Excel file. imp_mat is not saved.

        Parameters
        ----------
        file_name : str
            absolute path of the file
        """
        LOGGER.info('Writing %s', file_name)
        def write_col(i_col, imp_ws, xls_data):
            """Write one measure"""
            row_ini = 1
            for dat_row in xls_data:
                imp_ws.write(row_ini, i_col, dat_row)
                row_ini += 1

        imp_wb = xlsxwriter.Workbook(file_name)
        imp_ws = imp_wb.add_worksheet()

        header = ["tag_hazard", "tag_exposure", "tag_impact_func",
                  "unit", "tot_value", "aai_agg", "event_id",
                  "event_name", "event_date", "event_frequency", "frequency_unit",
                  "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"]
        for icol, head_dat in enumerate(header):
            imp_ws.write(0, icol, head_dat)
        data = [self.tag['haz'].haz_type, str(self.tag['haz'].file_name),
                str(self.tag['haz'].description)]
        write_col(0, imp_ws, data)
        data = [str(self.tag['exp'].file_name), str(self.tag['exp'].description)]
        write_col(1, imp_ws, data)
        data = [str(self.tag['impf_set'].file_name), str(self.tag['impf_set'].description)]
        write_col(2, imp_ws, data)
        write_col(3, imp_ws, [self.unit])
        write_col(4, imp_ws, [self.tot_value])
        write_col(5, imp_ws, [self.aai_agg])
        write_col(6, imp_ws, self.event_id)
        write_col(7, imp_ws, self.event_name)
        write_col(8, imp_ws, self.date)
        write_col(9, imp_ws, self.frequency)
        write_col(10, imp_ws, [self.frequency_unit])
        write_col(11, imp_ws, self.at_event)
        write_col(12, imp_ws, self.eai_exp)
        write_col(13, imp_ws, self.coord_exp[:, 0])
        write_col(14, imp_ws, self.coord_exp[:, 1])
        write_col(15, imp_ws, [str(self.crs)])

        imp_wb.close()

    def write_hdf5(self, file_path: Union[str, Path], dense_imp_mat: bool=False):
        """Write the data stored in this object into an H5 file.

        Try to write all attributes of this class into H5 datasets or attributes.
        By default, any iterable will be stored in a dataset and any string or scalar
        will be stored in an attribute. Dictionaries will be stored as groups, with
        the previous rules being applied recursively to their values.

        The impact matrix can be stored in a sparse or dense format.

        Notes
        -----
        This writer does not support attributes with variable types. Please make sure
        that ``event_name`` is a list of equally-typed values, e.g., all ``str``.

        Parameters
        ----------
        file_path : str or Path
            File path to write data into. The enclosing directory must exist.
        dense_imp_mat : bool
            If ``True``, write the impact matrix as dense matrix that can be more easily
            interpreted by common H5 file readers but takes up (vastly) more space.
            Defaults to ``False``.
        """
        # Define writers for all types (will be filled later)
        type_writers = dict()

        def write(group: h5py.Group, name: str, value: Any):
            """Write the given name-value pair with a type-specific writer.

            This selects a writer by calling ``isinstance(value, key)``, where ``key``
            iterates through the keys of ``type_writers``. If a type matches multiple
            entries in ``type_writers``, the *first* match is chosen.

            Parameters
            ----------
            group : h5py.Group
                The group in the H5 file to write into
            name : str
                The identifier of the value
            value : scalar or array
                The value/data to write

            Raises
            ------
            TypeError
                If no suitable writer could be found for a given type
            """
            for key, writer in type_writers.items():
                if isinstance(value, key):
                    return writer(group, name, value)

            raise TypeError(f"Could not find a writer for dataset: {name}")

        def _str_type_helper(values: Collection):
            """Return string datatype if we assume 'values' contains strings"""
            if isinstance(next(iter(values)), str):
                return h5py.string_dtype()
            return None

        def write_attribute(group, name, value):
            """Write any attribute. This should work for almost any data"""
            group.attrs[name] = value

        def write_dataset(group, name, value):
            """Write a dataset"""
            group.create_dataset(name, data=value, dtype=_str_type_helper(value))

        def write_dict(group, name, value):
            """Write a dictionary with unknown level recursively into a group"""
            group = group.create_group(name)
            for key, val in value.items():
                write(group, key, val)

        def write_tag(group, name, value):
            """Write a tag object using the dict writer"""
            write_dict(group, name, value.__dict__)

        def _write_csr_dense(group, name, value):
            """Write a CSR Matrix in dense format"""
            group.create_dataset(name, data=value.toarray())

        def _write_csr_sparse(group, name, value):
            """Write a CSR Matrix in sparse format"""
            group = group.create_group(name)
            group.create_dataset("data", data=value.data)
            group.create_dataset("indices", data=value.indices)
            group.create_dataset("indptr", data=value.indptr)
            group.attrs["shape"] = value.shape

        def write_csr(group, name, value):
            """Write a CSR matrix depending on user input"""
            if dense_imp_mat:
                _write_csr_dense(group, name, value)
            else:
                _write_csr_sparse(group, name, value)

        # Set up writers based on types
        # NOTE: 1) Many things are 'Collection', so make sure that precendence fits!
        #       2) Anything is 'object', so this serves as fallback/default.
        type_writers = {
            str: write_attribute,
            Tag: write_tag,
            TagHaz: write_tag,
            dict: write_dict,
            sparse.csr_matrix: write_csr,
            Collection: write_dataset,
            object: write_attribute,
        }

        # Open file in write mode
        with h5py.File(file_path, "w") as file:

            # Now write all attributes
            for name, value in self.__dict__.items():
                write(file, name, value)

    def write_sparse_csr(self, file_name):
        """Write imp_mat matrix in numpy's npz format."""
        LOGGER.info('Writing %s', file_name)
        np.savez(file_name, data=self.imp_mat.data, indices=self.imp_mat.indices,
                 indptr=self.imp_mat.indptr, shape=self.imp_mat.shape)

    @staticmethod
    def read_sparse_csr(file_name):
        """Read imp_mat matrix from numpy's npz format.

        Parameters
        ----------
        file_name : str

        Returns
        -------
        sparse.csr_matrix
        """
        LOGGER.info('Reading %s', file_name)
        loader = np.load(file_name)
        return sparse.csr_matrix(
            (loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    @classmethod
    def from_csv(cls, file_name):
        """Read csv file containing impact data generated by write_csv.

        Parameters
        ----------
        file_name : str
            absolute path of the file

        Returns
        -------
        imp : climada.engine.impact.Impact
            Impact from csv file
        """
        # pylint: disable=no-member
        LOGGER.info('Reading %s', file_name)
        imp_df = pd.read_csv(file_name)
        imp = cls()
        imp.unit = imp_df.unit[0]
        imp.tot_value = imp_df.tot_value[0]
        imp.aai_agg = imp_df.aai_agg[0]
        imp.event_id = imp_df.event_id[~np.isnan(imp_df.event_id)].values
        num_ev = imp.event_id.size
        imp.event_name = imp_df.event_name[:num_ev].values.tolist()
        imp.date = imp_df.event_date[:num_ev].values
        imp.at_event = imp_df.at_event[:num_ev].values
        imp.frequency = imp_df.event_frequency[:num_ev].values
        imp.frequency_unit = imp_df.frequency_unit[0] if 'frequency_unit' in imp_df \
                             else DEF_FREQ_UNIT
        imp.eai_exp = imp_df.eai_exp[~np.isnan(imp_df.eai_exp)].values
        num_exp = imp.eai_exp.size
        imp.coord_exp = np.zeros((num_exp, 2))
        imp.coord_exp[:, 0] = imp_df.exp_lat[:num_exp]
        imp.coord_exp[:, 1] = imp_df.exp_lon[:num_exp]
        try:
            imp.crs = u_coord.to_crs_user_input(imp_df.exp_crs.values[0])
        except AttributeError:
            imp.crs = DEF_CRS
        imp.tag['haz'] = TagHaz(str(imp_df.tag_hazard[0]),
                                 str(imp_df.tag_hazard[1]),
                                 str(imp_df.tag_hazard[2]))
        imp.tag['exp'] = Tag(str(imp_df.tag_exposure[0]),
                              str(imp_df.tag_exposure[1]))
        imp.tag['impf_set'] = Tag(str(imp_df.tag_impact_func[0]),
                                 str(imp_df.tag_impact_func[1]))
        return imp

    def read_csv(self, *args, **kwargs):
        """This function is deprecated, use Impact.from_csv instead."""
        LOGGER.warning("The use of Impact.read_csv is deprecated."
                       "Use Impact.from_csv instead.")
        self.__dict__ = Impact.from_csv(*args, **kwargs).__dict__

    @classmethod
    def from_excel(cls, file_name):
        """Read excel file containing impact data generated by write_excel.

        Parameters
        ----------
        file_name : str
            absolute path of the file

        Returns
        -------
        imp : climada.engine.impact.Impact
            Impact from excel file
        """
        LOGGER.info('Reading %s', file_name)
        dfr = pd.read_excel(file_name)
        imp =cls()
        imp.tag['haz'] = TagHaz(
            haz_type = dfr['tag_hazard'][0],
            file_name = dfr['tag_hazard'][1],
            description = dfr['tag_hazard'][2])
        imp.tag['exp'] = Tag()
        imp.tag['exp'].file_name = dfr['tag_exposure'][0]
        imp.tag['exp'].description = dfr['tag_exposure'][1]
        imp.tag['impf_set'] = Tag()
        imp.tag['impf_set'].file_name = dfr['tag_impact_func'][0]
        imp.tag['impf_set'].description = dfr['tag_impact_func'][1]

        imp.unit = dfr.unit[0]
        imp.tot_value = dfr.tot_value[0]
        imp.aai_agg = dfr.aai_agg[0]

        imp.event_id = dfr.event_id[~np.isnan(dfr.event_id.values)].values
        imp.event_name = dfr.event_name[:imp.event_id.size].values
        imp.date = dfr.event_date[:imp.event_id.size].values
        imp.frequency = dfr.event_frequency[:imp.event_id.size].values
        imp.frequency_unit = dfr.frequency_unit[0] if 'frequency_unit' in dfr else DEF_FREQ_UNIT
        imp.at_event = dfr.at_event[:imp.event_id.size].values

        imp.eai_exp = dfr.eai_exp[~np.isnan(dfr.eai_exp.values)].values
        imp.coord_exp = np.zeros((imp.eai_exp.size, 2))
        imp.coord_exp[:, 0] = dfr.exp_lat.values[:imp.eai_exp.size]
        imp.coord_exp[:, 1] = dfr.exp_lon.values[:imp.eai_exp.size]
        try:
            imp.crs = u_coord.to_csr_user_input(dfr.exp_crs.values[0])
        except AttributeError:
            imp.crs = DEF_CRS

        return imp

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use Impact.from_excel instead."""
        LOGGER.warning("The use of Impact.read_excel is deprecated."
                       "Use Impact.from_excel instead.")
        self.__dict__ = Impact.from_excel(*args, **kwargs).__dict__

    @classmethod
    def from_hdf5(cls, file_path: Union[str, Path]):
        """Create an impact object from an H5 file.

        This assumes a specific layout of the file. If values are not found in the
        expected places, they will be set to the default values for an ``Impact`` object.

        The following H5 file structure is assumed (H5 groups are terminated with ``/``,
        attributes are denoted by ``.attrs/``)::

            file.h5
            ├─ at_event
            ├─ coord_exp
            ├─ eai_exp
            ├─ event_id
            ├─ event_name
            ├─ frequency
            ├─ imp_mat
            ├─ tag/
            │  ├─ exp/
            │  │  ├─ .attrs/
            │  │  │  ├─ file_name
            │  │  │  ├─ description
            │  ├─ haz/
            │  │  ├─ .attrs/
            │  │  │  ├─ haz_type
            │  │  │  ├─ file_name
            │  │  │  ├─ description
            │  ├─ impf_set/
            │  │  ├─ .attrs/
            │  │  │  ├─ file_name
            │  │  │  ├─ description
            ├─ .attrs/
            │  ├─ aai_agg
            │  ├─ crs
            │  ├─ frequency_unit
            │  ├─ tot_value
            │  ├─ unit

        As per the :py:func:`climada.engine.impact.Impact.__init__`, any of these entries
        is optional. If it is not found, the default value will be used when constructing
        the Impact.

        The impact matrix ``imp_mat`` can either be an H5 dataset, in which case it is
        interpreted as dense representation of the matrix, or an H5 group, in which case
        the group is expected to contain the following data for instantiating a
        `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_::

            imp_mat/
            ├─ data
            ├─ indices
            ├─ indptr
            ├─ .attrs/
            │  ├─ shape

        Parameters
        ----------
        file_path : str or Path
            The file path of the file to read.

        Returns
        -------
        imp : Impact
            Impact with data from the given file
        """
        kwargs = dict()
        with h5py.File(file_path, "r") as file:

            # Impact matrix
            if "imp_mat" in file:
                impact_matrix = file["imp_mat"]
                if isinstance(impact_matrix, h5py.Dataset):  # Dense
                    impact_matrix = sparse.csr_matrix(impact_matrix)
                else:  # Sparse
                    impact_matrix = sparse.csr_matrix(
                        (
                            impact_matrix["data"],
                            impact_matrix["indices"],
                            impact_matrix["indptr"],
                        ),
                        shape=impact_matrix.attrs["shape"],
                    )
                kwargs["imp_mat"] = impact_matrix

            # Scalar attributes
            scalar_attrs = set(
                ("crs", "tot_value", "unit", "aai_agg", "frequency_unit")
            ).intersection(file.attrs.keys())
            kwargs.update({attr: file.attrs[attr] for attr in scalar_attrs})

            # Array attributes
            # NOTE: Need [:] to copy array data. Otherwise, it would be a view that is
            #       invalidated once we close the file.
            array_attrs = set(
                ("event_id", "date", "coord_exp", "eai_exp", "at_event", "frequency")
            ).intersection(file.keys())
            kwargs.update({attr: file[attr][:] for attr in array_attrs})

            # Special handling for 'event_name' because it's a list of strings
            if "event_name" in file:
                # pylint: disable=no-member
                kwargs["event_name"] = list(file["event_name"].asstr()[:])

            # Tags
            if "tag" in file:
                tag_kwargs = dict()
                tag_group = file["tag"]
                subtags = set(("exp", "impf_set")).intersection(tag_group.keys())
                tag_kwargs.update({st: Tag(**tag_group[st].attrs) for st in subtags})

                # Special handling for hazard because it has another tag type
                if "haz" in tag_group:
                    tag_kwargs["haz"] = TagHaz(**tag_group["haz"].attrs)
                kwargs["tag"] = tag_kwargs

        # Create the impact object
        return cls(**kwargs)

    @staticmethod
    def video_direct_impact(exp, impf_set, haz_list, file_name='',
                            writer=animation.PillowWriter(bitrate=500),
                            imp_thresh=0, args_exp=None, args_imp=None,
                            ignore_zero=False, pop_name=False):
        """
        Computes and generates video of accumulated impact per input events
        over exposure.

        Parameters
        ----------
        exp : climada.entity.Exposures
            exposures instance, constant during all video
        impf_set : climada.entity.ImpactFuncSet
            impact functions
        haz_list : (list(Hazard))
            every Hazard contains an event; all hazards
            use the same centroids
        file_name : str, optional
            file name to save video, if provided
        writer : matplotlib.animation.*, optional
            video writer. Default: pillow with bitrate=500
        imp_thresh : float, optional
            represent damages greater than threshold. Default: 0
        args_exp : dict, optional
            arguments for scatter (points) or hexbin (raster)
            matplotlib function used in exposures
        args_imp : dict, optional
            arguments for scatter (points) or hexbin (raster)
            matplotlib function used in impact
        ignore_zero : bool, optional
            flag to indicate if zero and negative
            values are ignored in plot. Default: False
        pop_name : bool, optional
            add names of the populated places
            The default is False.

        Returns
        -------
        list of Impact
        """
        if args_exp is None:
            args_exp = dict()
        if args_imp is None:
            args_imp = dict()
        imp_list = []
        exp_list = []
        imp_arr = np.zeros(len(exp.gdf))
        # assign centroids once for all
        exp.assign_centroids(haz_list[0])
        for i_time, _ in enumerate(haz_list):
            imp_tmp = Impact()
            imp_tmp.calc(exp, impf_set, haz_list[i_time], assign_centroids=False)
            imp_arr = np.maximum(imp_arr, imp_tmp.eai_exp)
            # remove not impacted exposures
            save_exp = imp_arr > imp_thresh
            imp_tmp.coord_exp = imp_tmp.coord_exp[save_exp, :]
            imp_tmp.eai_exp = imp_arr[save_exp]
            imp_list.append(imp_tmp)
            exp_list.append(~save_exp)

        v_lim = [np.array([haz.intensity.min() for haz in haz_list]).min(),
                 np.array([haz.intensity.max() for haz in haz_list]).max()]

        if 'vmin' not in args_exp:
            args_exp['vmin'] = exp.gdf.value.values.min()

        if 'vmin' not in args_imp:
            args_imp['vmin'] = np.array([imp.eai_exp.min() for imp in imp_list
                                         if imp.eai_exp.size]).min()

        if 'vmax' not in args_exp:
            args_exp['vmax'] = exp.gdf.value.values.max()

        if 'vmax' not in args_imp:
            args_imp['vmax'] = np.array([imp.eai_exp.max() for imp in imp_list
                                         if imp.eai_exp.size]).max()

        if 'cmap' not in args_exp:
            args_exp['cmap'] = 'winter_r'

        if 'cmap' not in args_imp:
            args_imp['cmap'] = 'autumn_r'


        plot_raster = False
        if exp.meta:
            plot_raster = True

        def run(i_time):
            haz_list[i_time].plot_intensity(1, axis=axis, cmap='Greys', vmin=v_lim[0],
                                            vmax=v_lim[1], alpha=0.8)
            if plot_raster:
                exp.plot_hexbin(axis=axis, mask=exp_list[i_time], ignore_zero=ignore_zero,
                                pop_name=pop_name, **args_exp)
                if imp_list[i_time].coord_exp.size:
                    imp_list[i_time].plot_hexbin_eai_exposure(axis=axis, pop_name=pop_name,
                                                              **args_imp)
                    fig.delaxes(fig.axes[1])
            else:
                exp.plot_scatter(axis=axis, mask=exp_list[i_time], ignore_zero=ignore_zero,
                                 pop_name=pop_name, **args_exp)
                if imp_list[i_time].coord_exp.size:
                    imp_list[i_time].plot_scatter_eai_exposure(axis=axis, pop_name=pop_name,
                                                               **args_imp)
                    fig.delaxes(fig.axes[1])
            fig.delaxes(fig.axes[1])
            fig.delaxes(fig.axes[1])
            axis.set_xlim(haz_list[-1].centroids.lon.min(), haz_list[-1].centroids.lon.max())
            axis.set_ylim(haz_list[-1].centroids.lat.min(), haz_list[-1].centroids.lat.max())
            axis.set_title(haz_list[i_time].event_name[0])
            pbar.update()

        if file_name:
            LOGGER.info('Generating video %s', file_name)
            fig, axis, _fontsize = u_plot.make_map()
            ani = animation.FuncAnimation(fig, run, frames=len(haz_list),
                                          interval=500, blit=False)
            pbar = tqdm(total=len(haz_list))
            fig.tight_layout()
            ani.save(file_name, writer=writer)
            pbar.close()

        return imp_list

#TODO: rewrite and deprecate method
    def _loc_return_imp(self, return_periods, imp, exc_imp):
        """Compute local exceedence impact for given return period.

        Parameters
        ----------
        return_periods : np.array
            return periods to consider
        cen_pos :int
            centroid position

        Returns
        -------
        np.array
        """
        # sorted impacts
        sort_pos = np.argsort(imp, axis=0)[::-1, :]
        columns = np.ones(imp.shape, int)
        # pylint: disable=unsubscriptable-object  # pylint/issues/3139
        columns *= np.arange(columns.shape[1])
        imp_sort = imp[sort_pos, columns]
        # cummulative frequency at sorted intensity
        freq_sort = self.frequency[sort_pos]
        np.cumsum(freq_sort, axis=0, out=freq_sort)

        for cen_idx in range(imp.shape[1]):
            exc_imp[:, cen_idx] = self._cen_return_imp(
                imp_sort[:, cen_idx], freq_sort[:, cen_idx],
                0, return_periods)

    def _build_exp(self):
        return Exposures(
            data={
                'value': self.eai_exp,
                'latitude': self.coord_exp[:, 0],
                'longitude': self.coord_exp[:, 1],
            },
            crs=self.crs,
            value_unit=self.unit,
            ref_year=0,
            tag=Tag(),
            meta=None
        )

    def _build_exp_event(self, event_id):
        """Write impact of an event as Exposures

        Parameters
        ----------
        event_id : int
            id of the event
        """
        [[idx]] = (self.event_id == event_id).nonzero()
        return Exposures(
            data={
                'value': self.imp_mat[idx].toarray().ravel(),
                'latitude': self.coord_exp[:, 0],
                'longitude': self.coord_exp[:, 1],
            },
            crs=self.crs,
            value_unit=self.unit,
            ref_year=0,
            tag=Tag(),
            meta=None
        )

    @staticmethod
    def _cen_return_imp(imp, freq, imp_th, return_periods):
        """From ordered impact and cummulative frequency at centroid, get
        exceedance impact at input return periods.

        Parameters
        ----------
        imp : np.array
            sorted impact at centroid
        freq : np.array
            cummulative frequency at centroid
        imp_th : float
            impact threshold
        return_periods : np.array
            return periods

        Returns
        -------
        np.array
        """
        imp_th = np.asarray(imp > imp_th).squeeze()
        imp_cen = imp[imp_th]
        freq_cen = freq[imp_th]
        if not imp_cen.size:
            return np.zeros((return_periods.size,))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pol_coef = np.polyfit(np.log(freq_cen), imp_cen, deg=1)
        except ValueError:
            pol_coef = np.polyfit(np.log(freq_cen), imp_cen, deg=0)
        imp_fit = np.polyval(pol_coef, np.log(1 / return_periods))
        wrong_inten = (return_periods > np.max(1 / freq_cen)) & np.isnan(imp_fit)
        imp_fit[wrong_inten] = 0.

        return imp_fit

    def select(self,
               event_ids=None, event_names=None, dates=None,
               coord_exp=None):
        """
        Select a subset of events and/or exposure points from the impact.
        If multiple input variables are not None, it returns all the impacts
        matching at least one of the conditions.

        Notes
        -----
            the frequencies are NOT adjusted. Method to adjust frequencies
        and obtain correct eai_exp:
            1- Select subset of impact according to your choice
            imp = impact.select(...)
            2- Adjust manually the frequency of the subset of impact
            imp.frequency = [...]
            3- Use select without arguments to select all events and recompute
            the eai_exp with the updated frequencies.
            imp = imp.select()

        Parameters
        ----------
        event_ids : list of int, optional
            Selection of events by their id. The default is None.
        event_names : list of str, optional
            Selection of events by their name. The default is None.
        dates : tuple, optional
            (start-date, end-date), events are selected if they are >=
            than start-date and <= than end-date. Dates in same format
            as impact.date (ordinal format of datetime library)
            The default is None.
        coord_exp : np.array, optional
            Selection of exposures coordinates [lat, lon] (in degrees)
            The default is None.

        Raises
        ------
        ValueError
            If the impact matrix is missing, the eai_exp and aai_agg cannot
            be updated for a selection of events and/or exposures.

        Returns
        -------
        imp : climada.engine.impact.Impact
            A new impact object with a selection of events and/or exposures
        """

        nb_events = self.event_id.size
        nb_exp = len(self.coord_exp)

        if self.imp_mat.shape != (nb_events, nb_exp):
            raise ValueError("The impact matrix is missing or incomplete. "
                             "The eai_exp and aai_agg cannot be computed. "
                             "Please recompute impact.calc() with save_mat=True "
                             "before using impact.select()")

        if nb_events == nb_exp:
            LOGGER.warning("The number of events is equal to the number of "
                           "exposure points. It is not possible to "
                           "differentiate events and exposures attributes. "
                           "Please add/remove one event/exposure point. "
                           "This is a purely technical limitation of this "
                           "method.")
            return None

        imp = copy.deepcopy(self)

        # apply event selection to impact attributes
        sel_ev = self._selected_events_idx(event_ids, event_names, dates, nb_events)
        if sel_ev is not None:
            # set all attributes that are 'per event', i.e. have a dimension
            # of length equal to the number of events (=nb_events)
            for attr in get_attributes_with_matching_dimension(imp, [nb_events]):
                value = imp.__getattribute__(attr)
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        setattr(imp, attr, value[sel_ev])
                    else:
                        LOGGER.warning("Found a multidimensional numpy array "
                                       "with one dimension matching the number of events. "
                                       "But multidimensional numpy arrays are not handled "
                                       "in impact.select")
                elif isinstance(value, sparse.csr_matrix):
                    setattr(imp, attr, value[sel_ev, :])
                elif isinstance(value, list) and value:
                    setattr(imp, attr, [value[idx] for idx in sel_ev])
                else:
                    pass

            LOGGER.info("The eai_exp and aai_agg are computed for the "
                        "selected subset of events WITHOUT modification of "
                        "the frequencies.")

        # apply exposure selection to impact attributes
        if coord_exp is not None:
            sel_exp = self._selected_exposures_idx(coord_exp)
            imp.coord_exp = imp.coord_exp[sel_exp]
            imp.imp_mat = imp.imp_mat[:, sel_exp]

            # .A1 reduce 1d matrix to 1d array
            imp.at_event = imp.imp_mat.sum(axis=1).A1
            imp.tot_value = None
            LOGGER.info("The total value cannot be re-computed for a "
                        "subset of exposures and is set to None.")

        # cast frequency vector into 2d array for sparse matrix multiplication
        freq_mat = imp.frequency.reshape(len(imp.frequency), 1)
        # .A1 reduce 1d matrix to 1d array
        imp.eai_exp = imp.imp_mat.multiply(freq_mat).sum(axis=0).A1
        imp.aai_agg = imp.eai_exp.sum()

        return imp

    def _selected_events_idx(self, event_ids, event_names, dates, nb_events):
        if all(var is None for var in [dates, event_ids, event_names]):
            return None

        # filter events by date
        if dates is None:
            mask_dt = np.zeros(nb_events, dtype=bool)
        else:
            mask_dt = np.ones(nb_events, dtype=bool)
            date_ini, date_end = dates
            if isinstance(date_ini, str):
                date_ini = u_dt.str_to_date(date_ini)
                date_end = u_dt.str_to_date(date_end)
            mask_dt &= (date_ini <= self.date)
            mask_dt &= (self.date <= date_end)
            if not np.any(mask_dt):
                LOGGER.info('No impact event in given date range %s.', dates)

        sel_dt = mask_dt.nonzero()[0]  # Convert bool to indices

        # filter events by id
        if event_ids is None:
            sel_id = np.array([], dtype=int)
        else:
            (sel_id,) = np.isin(self.event_id, event_ids).nonzero()
            # pylint: disable=no-member
            if sel_id.size == 0:
                LOGGER.info('No impact event with given ids %s found.', event_ids)

        # filter events by name
        if event_names is None:
            sel_na = np.array([], dtype=int)
        else:
            (sel_na,) = np.isin(self.event_name, event_names).nonzero()
            # pylint: disable=no-member
            if sel_na.size == 0:
                LOGGER.info('No impact event with given names %s found.', event_names)

        # select events with machting id, name or date field.
        sel_ev = np.unique(np.concatenate([sel_dt, sel_id, sel_na]))

        # if no event found matching ids, names or dates, warn the user
        if sel_ev.size == 0:
            LOGGER.warning("No event matches the selection. ")

        return sel_ev

    def _selected_exposures_idx(self, coord_exp):
        assigned_idx = u_coord.assign_coordinates(self.coord_exp, coord_exp, threshold=0)
        sel_exp = (assigned_idx >= 0).nonzero()[0]
        if sel_exp.size == 0:
            LOGGER.warning("No exposure coordinates match the selection.")
        return sel_exp

    @classmethod
    def concat(cls, imp_list: Iterable, reset_event_ids: bool = False):
        """Concatenate impact objects with the same exposure

        This function is useful if, e.g. different impact functions
        have to be applied for different seasons (e.g. for agricultural impacts).

        It checks if the exposures of the passed impact objects are identical and then

        - concatenates the attributes ``event_id``, ``event_name``, ``date``,
          ``frequency``, ``imp_mat``, ``at_event``,
        - sums up the values of attributes ``eai_exp``, ``aai_exp``
        - and takes the following attributes from the first impact object in the passed
          impact list: ``coord_exp``, ``crs``, ``unit``, ``tot_value``, ``tag``,
          ``frequency_unit``

        If event ids are not unique among the passed impact objects an error is raised.
        In this case, the user can set ``reset_event_ids=True`` to create unique event ids
        for the concatenated impact.

        If all impact matrices of the impacts in ``imp_list`` are empty,
        the impact matrix of the concatenated impact is also empty.

        Parameters
        ----------
        imp_list : Iterable of climada.engine.impact.Impact
            Iterable of Impact objects to concatenate
        reset_event_ids: boolean, optional
            Reset event ids of the concatenated impact object

        Returns
        --------
        impact: climada.engine.impact.Impact
            New impact object which is a concatenation of all impacts

        Notes
        -----
        - Concatenation of impacts with different exposure (e.g. different countries)
          could also be implemented here in the future.
        """
        def check_unique_attr(attr_name: str):
            """Check if an attribute is unique among all impacts"""
            if len({getattr(imp, attr_name) for imp in imp_list}) > 1:
                raise ValueError(
                    f"Attribute '{attr_name}' must be unique among impacts"
                )

        # Check if single-value attribute are unique
        for attr in ("crs", "tot_value", "unit", "frequency_unit"):
            check_unique_attr(attr)

        # Check exposure coordinates
        imp_iter = iter(imp_list)
        first_imp = next(imp_iter)
        for imp in imp_iter:
            if not np.array_equal(first_imp.coord_exp, imp.coord_exp):
                raise ValueError("The impacts have different exposure coordinates")

        # Stack attributes
        def stack_attribute(attr_name: str) -> np.ndarray:
            """Stack an attribute of all impacts passed to this method"""
            return np.concatenate([getattr(imp, attr_name) for imp in imp_list])

        # Concatenate event IDs
        event_ids = stack_attribute("event_id")
        if reset_event_ids:
            # NOTE: event_ids must not be zero!
            event_ids = np.array(range(len(event_ids))) + 1
        else:
            unique_ids, count = np.unique(event_ids, return_counts=True)
            if np.any(count > 1):
                raise ValueError(
                    f"Duplicate event IDs: {unique_ids[count > 1]}\n"
                    "Consider setting 'reset_event_ids=True'"
                )

        # Concatenate impact matrices
        imp_mats = [imp.imp_mat for imp in imp_list]
        if len({mat.shape[1] for mat in imp_mats}) > 1:
            raise ValueError(
                "Impact matrices do not have the same number of exposure points"
            )
        imp_mat = sparse.vstack(imp_mats)

        # Concatenate other attributes
        kwargs = {
            attr: stack_attribute(attr) for attr in ("date", "frequency", "at_event")
        }

        # Get remaining attributes from first impact object in list
        return cls(
            event_id=event_ids,
            event_name=list(stack_attribute("event_name").flat),
            coord_exp=first_imp.coord_exp,
            crs=first_imp.crs,
            unit=first_imp.unit,
            tot_value=first_imp.tot_value,
            eai_exp=np.nansum([imp.eai_exp for imp in imp_list], axis=0),
            aai_agg=np.nansum([imp.aai_agg for imp in imp_list]),
            imp_mat=imp_mat,
            tag=first_imp.tag,
            frequency_unit=first_imp.frequency_unit,
            **kwargs,
        )


@dataclass
class ImpactFreqCurve():
    """Impact exceedence frequency curve.
    """

    tag : dict = field(default_factory=dict)
    """dictionary of tags of exposures, impact functions set and
        hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHaz()}"""

    return_per : np.array = np.array([])
    """return period"""

    impact : np.array = np.array([])
    """impact exceeding frequency"""

    unit : str = ''
    """value unit used (given by exposures unit)"""

    frequency_unit : str = DEF_FREQ_UNIT
    """value unit used (given by exposures unit)"""

    label : str = ''
    """string describing source data"""

    def plot(self, axis=None, log_frequency=False, **kwargs):
        """Plot impact frequency curve.

        Parameters
        ----------
        axis : matplotlib.axes.Axes, optional
            axis to use
        log_frequency : boolean, optional
            plot logarithmioc exceedance frequency on x-axis
        kwargs : dict, optional
            arguments for plot matplotlib function, e.g. color='b'

        Returns
        -------
        matplotlib.axes.Axes
        """
        if not axis:
            _, axis = plt.subplots(1, 1)
        axis.set_title(self.label)
        axis.set_ylabel('Impact (' + self.unit + ')')
        if log_frequency:
            axis.set_xlabel(f'Exceedance frequency ({self.frequency_unit})')
            axis.set_xscale('log')
            axis.plot(self.return_per**-1, self.impact, **kwargs)
        else:
            axis.set_xlabel('Return period (year)')
            axis.plot(self.return_per, self.impact, **kwargs)
        return axis
