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

import logging
import copy
import csv
import warnings
import datetime as dt
from itertools import zip_longest
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import xlsxwriter
from tqdm import tqdm


from climada.entity import Exposures, Tag
from climada.hazard import Tag as TagHaz
import climada.util.plot as u_plot
from climada import CONFIG
from climada.util.constants import DEF_CRS, CMAP_IMPACT
import climada.util.coordinates as u_coord
import climada.util.dates_times as u_dt
from climada.util.select import get_attributes_with_matching_dimension

LOGGER = logging.getLogger(__name__)

class ImpactCalc():
    """
    Class to compute impacts from exposures, impact function set and hazard
    """

    def __init__(self,
                 exposures,
                 impfset,
                 hazard,
                 imp_mat=None):
        """
        Initialize an ImpactCalc object.

        The dimension of the imp_mat variable must be compatible with the
        exposures and hazard objects.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposure used to compute imp_mat
        impf_set: climada.entity.ImpactFuncSet
            impact functions set used to compute imp_mat
        hazard : climada.Hazard
            hazard used to compute imp_mat
        imp_mat : sparse.csr_matrix, optional
            matrix num_events x num_exp with impacts.
            Default is an empty matrix.

        Returns
        -------
        None.

        """

        imp_mat = sparse.csr_matrix(np.empty((0, 0))) if imp_mat is None else imp_mat
        self.exposures = exposures
        self.impfset = impfset
        self.hazard = hazard
        self.imp_mat = imp_mat
        self.n_exp_pnt = self.exposures.gdf.shape[0]
        self.n_events = self.hazard.size

    @property
    def deductible(self):
        """
        Deductible from the exposures

        Returns
        -------
        np.array
            The deductible per exposure point

        """
        return self.exposures.gdf['deductible']

    @property
    def cover(self):
        """
        Cover from the exposures

        Returns
        -------
        np.array
            The cover per exposure point

        """
        return self.exposures.gdf['cover']

    def impact(self, save_mat=True):
        """Compute the impact of a hazard on exposures.

        Parameters
        ----------
        save_mat : bool
            if true, save the total impact matrix (events x exposures)

        Examples
        --------
            >>> haz = Hazard.from_mat(HAZ_DEMO_MAT)  # Set hazard
            >>> impfset = ImpactFuncSet.from_excel(ENT_TEMPLATE_XLS) # Set impact functions
            >>> exp = Exposures(pd.read_excel(ENT_TEMPLATE_XLS)) # Set exposures
            >>> impcalc = ImpactCal(exp, impfset, haz)
            >>> imp = impcalc.insured_impact()
            >>> imp.aai_agg

        Note
        ----
        Deductible and/or cover values in the exposures are ignored.
        """
        impf_col = self.exposures.get_impf_column(self.hazard.haz_type)
        exp_gdf = self.minimal_exp_gdf(impf_col)
        LOGGER.info('Calculating impact for %s assets (>0) and %s events.',
                    self.n_events, self.n_events)
        imp_mat_gen = self.imp_mat_gen(exp_gdf, impf_col)
        if save_mat:
            self.imp_mat = self.stitch_impact_matrix(imp_mat_gen)
            at_event, eai_exp, aai_agg = self.risk_metrics()
        else:
            at_event, eai_exp, aai_agg = self.stitch_risk_metrics(imp_mat_gen)
        return Impact.from_eih(
            self.exposures, self.impfset, self.hazard,
            at_event, eai_exp, aai_agg, self.imp_mat
            )

    def insured_impact(self, save_mat=False):
        """Compute the impact of a hazard on exposures with a deductible and/or
        cover.

        For each exposure point, the impact per event is obtained by
        substracting the deductible (and is maximally equal to the cover).

        Parameters
        ----------
        save_mat : bool
            if true, save the total impact matrix (events x exposures)

        Examples
        --------
            >>> haz = Hazard.from_mat(HAZ_DEMO_MAT)  # Set hazard
            >>> impfset = ImpactFuncSet.from_excel(ENT_TEMPLATE_XLS) # Set impact functions
            >>> exp = Exposures(pd.read_excel(ENT_TEMPLATE_XLS)) # Set exposures
            >>> impcalc = ImpactCal(exp, impfset, haz)
            >>> imp = impcalc.insured_impact()
            >>> imp.aai_agg

        See also
        --------
        apply_deductible_to_mat:
            apply deductible to impact matrix
        apply_cover_to_mat:
            apply cover to impact matrix
        """
        impf_col = self.exposures.get_impf_column(self.hazard.haz_type)
        exp_gdf = self.minimal_exp_gdf(impf_col)
        LOGGER.info('Calculating impact for %s assets (>0) and %s events.',
                    exp_gdf.size, self.hazard.size)

        imp_mat_gen = self.imp_mat_gen(exp_gdf, impf_col)
        ins_mat_gen = self.insured_mat_gen(imp_mat_gen, impf_col)

        if save_mat:
            self.imp_mat = self.stitch_impact_matrix(ins_mat_gen)
            at_event, eai_exp, aai_agg = self.risk_metrics()
        else:
            at_event, eai_exp, aai_agg = self.stitch_risk_metrics(ins_mat_gen)
        return Impact.from_eih(
            self.exposures, self.impfset, self.hazard,
            at_event, eai_exp, aai_agg, self.imp_mat
            )

    def minimal_exp_gdf(self, impf_col):
        """Get minimal exposures geodataframe for impact computation

        Parameters
        ----------
        exposures : climada.entity.Exposures
        hazard : climada.Hazard
        impf_col: stirng
            name of the impact function column in exposures.gdf

        """
        self.exposures.assign_centroids(self.hazard, overwrite=False)

        mask = (
            (self.exposures.gdf.value.values != 0)
            & (self.exposures.gdf[self.hazard.cent_exp_col].values >= 0)
        )
        exp_gdf = pd.DataFrame({
            col: self.exposures.gdf[col].values[mask]
            for col in ['value', impf_col, self.hazard.cent_exp_col]
        })
        if exp_gdf.size == 0:
            LOGGER.warning("No exposures with value >0 in the vicinity of the hazard.")
        return exp_gdf

    def imp_mat_gen(self, exp_gdf, impf_col):
        """
        Geneartor of impact sub-matrices and correspoding exposures indices
        """
        for impf_id in exp_gdf[impf_col].dropna().unique():
            impf = self.impfset.get_func(haz_type=self.hazard.haz_type, fun_id=impf_id)
            idx_exp_impf = (exp_gdf[impf_col].values == impf.id).nonzero()[0]
            exp_step = CONFIG.max_matrix_size.int() // self.hazard.size
            if not exp_step:
                raise ValueError(
                    f'Increase max_matrix_size configuration parameter to > {self.hazard.size}')
            for chk in range(int(idx_exp_impf.size / exp_step) + 1):
                exp_idx = idx_exp_impf[chk * exp_step:(chk + 1) * exp_step]
                exp_values = exp_gdf.value.values[exp_idx]
                cent_idx = exp_gdf[self.hazard.cent_exp_col].values[exp_idx]
                yield (self.impact_matrix(exp_values, cent_idx, impf), exp_idx)

    def insured_mat_gen(self, imp_mat_gen, impf_col):
        """
        Generator of insured impact sub-matrices (with applied cover and deductible)
        and corresponding exposures indices
        """
        for mat, exp_idx in imp_mat_gen:
            impf_id = self.exposures.gdf[impf_col][exp_idx].unique()[0]
            deductible = self.deductible[exp_idx]
            cent_idx = self.exposures.gdf['centr_TC'][exp_idx]
            impf = self.impfset.get_func(haz_type=self.hazard.haz_type, fun_id=impf_id)
            mat = self.apply_deductible_to_mat(mat, deductible, self.hazard, cent_idx, impf)
            cover = self.cover[exp_idx]
            mat = self.apply_cover_to_mat(mat, cover)
            yield (mat, exp_idx)

    def impact_matrix(self, exp_values, cent_idx, impf):
        """
        Compute the impact matrix for given exposure values,
        assigned centroids, a hazard, and one impact function.

        Parameters
        ----------
        exp_values : np.array
            Exposure values
        cent_idx : np.array
            Hazard centroids assigned to each exposure location
        hazard : climada.Hazard
           Hazard object
        impf : climada.entity.ImpactFunc
            one impactfunction comon to all exposure elements in exp_gdf

        Returns
        -------
        scipy.sparse.csr_matrix
            Impact per event (rows) per exposure point (columns)
        """
        n_centroids = cent_idx.size
        mdr = self.hazard.get_mdr(cent_idx, impf)
        fract = self.hazard.get_fraction(cent_idx)
        exp_values_csr = sparse.csr_matrix(
            (exp_values, np.arange(n_centroids), [0, n_centroids]),
            shape=(1, n_centroids))
        return fract.multiply(mdr).multiply(exp_values_csr)

    def stitch_impact_matrix(self, imp_mat_gen):
        """
        Make an impact matrix from an impact sub-matrix generator
        """
        data, row, col = np.hstack([
            (mat.data, mat.nonzero()[0], idx[mat.nonzero()[1]])
            for mat, idx in imp_mat_gen
            ])
        return sparse.csr_matrix(
            (data, (row, col)), shape=(self.n_events, self.n_exp_pnt)
            )

    def stitch_risk_metrics(self, imp_mat_gen):
        """
        Compute the impact metrics from an impact sub-matrix generator
        """
        at_event = np.zeros(self.n_events)
        eai_exp = np.zeros(self.n_exp_pnt)
        for sub_imp_mat, exp_idx in imp_mat_gen:
            at_event += self.at_event_from_mat(sub_imp_mat)
            eai_exp[exp_idx] += self.eai_exp_from_mat(sub_imp_mat, self.hazard.frequency)
        aai_agg = self.aai_agg_from_eai_exp(eai_exp)
        return at_event, eai_exp, aai_agg

    @staticmethod
    def apply_deductible_to_mat(mat, deductible, hazard, cent_idx, impf):
        """
        Apply a deductible per exposure point to an impact matrix at given
        centroid points for given impact function.

        All exposure points must have the same impact function. For different
        impact functions apply use this method repeatedly on the same impact
        matrix.

        Parameters
        ----------
        imp_mat : scipy.sparse.csr_matrix
            impact matrix (events x exposure points)
        deductible : np.array()
            deductible for each exposure point
        hazard : climada.Hazard
            hazard used to compute the imp_mat
        cent_idx : np.array()
            index of centroids associated with each exposure point
        impf : climada.entity.ImpactFunc
            impact function associated with the exposure points

        Returns
        -------
        imp_mat : scipy.sparse.csr_matrix
            impact matrix with applied deductible

        """
        paa = hazard.get_paa(cent_idx, impf)
        mat -= paa.multiply(sparse.csr_matrix(deductible))
        mat.eliminate_zeros()
        return mat

    @staticmethod
    def apply_cover_to_mat(mat, cover):
        """
        Apply cover to impact matrix.

        The impact data is clipped to the range [0, cover]. The cover is defined
        per exposure point.

        Parameters
        ----------
        imp_mat : scipy.sparse.csr_matrix
            impact matrix
        cover : np.array()
            cover per exposures point (columns of imp_mat)

        Returns
        -------
        imp_mat : scipy.sparse.csr_matrix
            impact matrix with applied cover

        """
        mat.data = np.clip(mat.data, 0, cover.to_numpy()[mat.nonzero()[1]])
        mat.eliminate_zeros()
        return mat

    @staticmethod
    def eai_exp_from_mat(mat, freq):
        """
        Compute impact for each exposures from the total impact matrix

        Parameters
        ----------
        imp_mat : sparse.csr_matrix
            matrix num_events x num_exp with impacts.
        frequency : np.array
            annual frequency of events
        Returns
        -------
        eai_exp : np.array
            expected annual impact for each exposure
        """
        n_events = freq.size
        freq_csr = sparse.csr_matrix(
            (freq, np.zeros(n_events), np.arange(n_events + 1)),
            shape=(n_events, 1))
        return mat.multiply(freq_csr).sum(axis=0).A1

    @staticmethod
    def at_event_from_mat(mat):
        """
        Compute impact for each hazard event from the total impact matrix
        Parameters
        ----------
        imp_mat : sparse.csr_matrix
            matrix num_events x num_exp with impacts.
        Returns
        -------
        at_event : np.array
            impact for each hazard event
        """
        return np.squeeze(np.asarray(np.sum(mat, axis=1)))

    @staticmethod
    def aai_agg_from_eai_exp(eai_exp):
        """
        Aggregate impact.eai_exp

        Parameters
        ----------
        eai_exp : np.array
            expected annual impact for each exposure point

        Returns
        -------
        float
            average annual impact aggregated
        """
        return np.sum(eai_exp)


    def risk_metrics(self):
        """
        Compute risk metricss eai_exp, at_event, aai_agg
        for an impact matrix and a frequency vector.

        Parameters
        ----------
        imp_mat : sparse.csr_matrix
            matrix num_events x num_exp with impacts.
        freq : np.array
            array with the frequency per event

        Returns
        -------
        eai_exp: np.array
            expected annual impact at each exposure point
        at_event: np.array()
            total impact for each event
        aai_agg : float
            average annual impact aggregated over all exposure points
        """
        eai_exp = self.eai_exp_from_mat(self.imp_mat, self.hazard.frequency)
        at_event = self.at_event_from_mat(self.imp_mat)
        aai_agg = self.aai_agg_from_eai_exp(eai_exp)
        return at_event, eai_exp, aai_agg

class Impact():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes
    ----------
    tag : dict
        dictionary of tags of exposures, impact functions set and
        hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHazard()}
    event_id :
        np.array id (>0) of each hazard event
    event_name :
        list name of each hazard event
    date : np.array
        date if events as integer date corresponding to the
        proleptic Gregorian ordinal, where January 1 of year 1 has
        ordinal 1 (ordinal format of datetime library)
    coord_exp : np.array
        exposures coordinates [lat, lon] (in degrees)
    eai_exp : np.array
        expected annual impact for each exposure
    at_event : np.array
        impact for each hazard event
    frequency : np.array
        annual frequency of event
    tot_value : float
        total exposure value affected
    aai_agg : float
        average annual impact (aggregated)
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
                 coord_exp=None,
                 crs=DEF_CRS,
                 eai_exp=None,
                 at_event=None,
                 tot_value=0,
                 aai_agg=0,
                 unit='',
                 imp_mat=None,
                 tag=None):

        self.tag = tag or {}
        self.event_id = np.array([], int) if event_id is None else event_id
        self.event_name = event_name or []
        self.date = np.array([], int) if date is None else date
        self.coord_exp = np.ndarray([], float) if coord_exp is None else coord_exp
        self.crs = crs
        self.eai_exp = np.array([], float) if eai_exp is None else eai_exp
        self.at_event = np.array([], float) if at_event is None else at_event
        self.frequency = np.array([],float) if frequency is None else frequency
        self.tot_value = tot_value
        self.aai_agg = aai_agg
        self.unit = unit
        self.imp_mat = sparse.csr_matrix(np.empty((0, 0))) if imp_mat is None else imp_mat

    def calc(self, exposures, impact_funcs, hazard, save_mat=False):
        """This function is deprecated, use Impact.calc_risk and Impact.calc_insured_risk instead.
        """
        impcalc= ImpactCalc(exposures, impact_funcs, hazard)
        if ('deductible' in exposures.gdf) and ('cover' in exposures.gdf) \
            and exposures.gdf.cover.max():
            # LOGGER.warning("To compute the risk transfer value"
            #      "please use Impact.calc_insured_risk")
            self.__dict__ = impcalc.insured_impact(save_mat).__dict__
        else:
            # LOGGER.warning("The use of Impact.calc() is deprecated."
            #      "Please use Impact.calc_risk() or Impact.calc_risk_insured().")
            self.__dict__ = impcalc.impact(save_mat).__dict__

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
        impf_set: climada.entity.ImpactFuncSet
            impact functions set used to compute imp_mat
        hazard : climada.Hazard
            hazard used to compute imp_mat
        imp_mat : sparse.csr_matrix
            matrix num_events x num_exp with impacts.

        Returns
        -------
        climada.engine.impact.Impact
            impact with all risk metrics set based on the given impact matrix
        """
        imp_mat = sparse.csr_matrix(np.empty((0, 0))) if imp_mat is None else imp_mat
        return cls(
            event_id = hazard.event_id,
            event_name = hazard.event_name,
            date = hazard.date,
            frequency = hazard.frequency,
            coord_exp = np.stack([exposures.gdf.latitude.values,
                                  exposures.gdf.longitude.values],
                                 axis=1),
            crs = exposures.crs,
            unit = exposures.value_unit,
            tot_value = exposures.affected_total_value(hazard),
            eai_exp = eai_exp,
            at_event = at_event,
            aai_agg = aai_agg,
            imp_mat = imp_mat,
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
        transfer_at_event : np.array()
            risk transfered per event
        transfer_aai_agg : float
            average  annual risk transfered
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
        residual_at_event : np.array()
            residual risk per event
        residual_aai_agg : float
            average annual residual risk

        See also
        --------
        transfer_risk: compute the transfer risk per portfolio.

        """
        transfer_at_event, _ = self.transfer_risk(attachment, cover)
        residual_at_event = np.maximum(self.at_event - transfer_at_event, 0)
        residual_aai_agg = np.sum(residual_at_event * self.frequency)
        return residual_at_event, residual_aai_agg

#TODO deprecate method
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
            new_imp.imp_mat = sparse.csr_matrix(np.empty((0, 0)))
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
        all_years : boolean
            return values for all years between first and
            last year with event, including years without any events.
        year_range : tuple or list with integers
            start and end year
        Returns
        -------
        year_set: dict
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

    def local_exceedance_imp(self, return_periods=(25, 50, 100, 250)):
        """Compute exceedance impact map for given return periods.
        Requires attribute imp_mat.

        Parameters
        ----------
        return_periods : np.array return periods to consider

        Returns
        -------
        np.array
        """
        LOGGER.info('Computing exceedance impact map for return periods: %s',
                    return_periods)
        try:
            self.imp_mat.shape[1]
        except AttributeError as err:
            raise ValueError('attribute imp_mat is empty. Recalculate Impact'
                             'instance with parameter save_mat=True') from err
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
        ifc = ImpactFreqCurve()
        ifc.tag = self.tag
        # Sort descendingly the impacts per events
        sort_idxs = np.argsort(self.at_event)[::-1]
        # Calculate exceedence frequency
        exceed_freq = np.cumsum(self.frequency[sort_idxs])
        # Set return period and imact exceeding frequency
        ifc.return_per = 1 / exceed_freq[::-1]
        ifc.impact = self.at_event[sort_idxs][::-1]
        ifc.unit = self.unit
        ifc.label = 'Exceedance frequency curve'

        if return_per is not None:
            interp_imp = np.interp(return_per, ifc.return_per, ifc.impact)
            ifc.return_per = return_per
            ifc.impact = interp_imp

        return ifc

    def plot_scatter_eai_exposure(self, mask=None, ignore_zero=True,
                                  pop_name=True, buffer=0.0, extend='neither',
                                  axis=None, adapt_fontsize=True, **kwargs):
        """Plot scatter expected annual impact of each exposure.

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
        axis  : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        adapt_fontsize : bool, optional
                If set to true, the size of the fonts will be adapted to the size of the figure.
                Otherwise the default matplotlib font size is used. Default is True.
        kwargs : optional
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
        axis.set_title('Expected annual impact')
        return axis

    def plot_hexbin_eai_exposure(self, mask=None, ignore_zero=True,
                                 pop_name=True, buffer=0.0, extend='neither',
                                 axis=None, adapt_fontsize=True, **kwargs):
        """Plot hexbin expected annual impact of each exposure.

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
            axis : matplotlib.axes._subplots.AxesSubplot, optional
                axis to use
            kwargs : optional
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
        axis.set_title('Expected annual impact')
        return axis

    def plot_raster_eai_exposure(self, res=None, raster_res=None, save_tiff=None,
                                 raster_f=lambda x: np.log10((np.fmax(x + 1, 1))),
                                 label='value (log10)', axis=None, adapt_fontsize=True,
                                 **kwargs):
        """Plot raster expected annual impact of each exposure.

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
        label : str colorbar label
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        adapt_fontsize : bool, optional
                If set to true, the size of the fonts will be adapted to the size of the figure.
                Otherwise the default matplotlib font size is used. Default is True.
        kwargs : optional
            arguments for imshow matplotlib function

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        axis = eai_exp.plot_raster(res, raster_res, save_tiff, raster_f,
                                   label, axis=axis, adapt_fontsize=adapt_fontsize, **kwargs)
        axis.set_title('Expected annual impact')
        return axis

    def plot_basemap_eai_exposure(self, mask=None, ignore_zero=False, pop_name=True,
                                  buffer=0.0, extend='neither', zoom=10,
                                  url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png',
                                  axis=None, **kwargs):
        """Plot basemap expected annual impact of each exposure.

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
            image source, e.g. ctx.sources.OSM_C
        axis : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        kwargs : optional
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
        axis.set_title('Expected annual impact')
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
        kwargs : optional
            arguments for hexbin matplotlib function
        axis : matplotlib.axes._subplots.AxesSubplot
            optional axis to use
        adapt_fontsize : bool, optional
            If set to true, the size of the fonts will be adapted to the size of the figure.
            Otherwise the default matplotlib font size is used. Default is True.

        Returns
        --------
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if not hasattr(self.imp_mat, "shape") or self.imp_mat.shape[1] == 0:
            raise ValueError('attribute imp_mat is empty. Recalculate Impact'
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
                                     url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png',
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
            image source, e.g. ctx.sources.OSM_C
        axis  : matplotlib.axes._subplots.AxesSubplot, optional axis to use
        kwargs : optional arguments for scatter matplotlib function, e.g.
            cmap='Greys'. Default: 'Wistia'

        Returns
        -------
        cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        if not hasattr(self.imp_mat, "shape") or self.imp_mat.shape[1] == 0:
            raise ValueError('attribute imp_mat is empty. Recalculate Impact'
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
        return_periods : tuple(int), optional
            return periods to consider
        log10_scale : boolean, optional
            plot impact as log10(impact)
        smooth : bool, optional
            smooth plot to plot.RESOLUTIONxplot.RESOLUTION
        kwargs : optional
            arguments for pcolormesh matplotlib function
            used in event plots

        Returns
        --------
        matplotlib.axes._subplots.AxesSubplot,
        np.ndarray (return_periods.size x num_centroids)
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
                             "event_name", "event_date", "event_frequency",
                             "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"])
            csv_data = [[[self.tag['haz'].haz_type], [self.tag['haz'].file_name],
                         [self.tag['haz'].description]],
                        [[self.tag['exp'].file_name], [self.tag['exp'].description]],
                        [[self.tag['impf_set'].file_name], [self.tag['impf_set'].description]],
                        [self.unit], [self.tot_value], [self.aai_agg],
                        self.event_id, self.event_name, self.date,
                        self.frequency, self.at_event,
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
                  "event_name", "event_date", "event_frequency",
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
        write_col(10, imp_ws, self.at_event)
        write_col(11, imp_ws, self.eai_exp)
        write_col(12, imp_ws, self.coord_exp[:, 0])
        write_col(13, imp_ws, self.coord_exp[:, 1])
        write_col(14, imp_ws, [str(self.crs)])

        imp_wb.close()

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
        file_name : str file name

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
        file_name : str absolute path of the file

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
        file_name : str absolute path of the file

        Returns
        -------
        imp : climada.engine.impact.Impact
            Impact from excel file
        """
        LOGGER.info('Reading %s', file_name)
        dfr = pd.read_excel(file_name)
        imp =cls()
        imp.tag['haz'] = TagHaz()
        imp.tag['haz'].haz_type = dfr['tag_hazard'][0]
        imp.tag['haz'].file_name = dfr['tag_hazard'][1]
        imp.tag['haz'].description = dfr['tag_hazard'][2]
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
        impf_set : climada.entity.ImactFuncSet
            impact functions
        haz_list : (list(Hazard))
            every Hazard contains an event; all hazards
            use the same centroids
        file_name : str, optional
            file name to save video, if provided
        writer : matplotlib.animation.*, optional
            video writer. Default: pillow with bitrate=500
        imp_thresh : float
            represent damages greater than threshold
        args_exp : optional
            arguments for scatter (points) or hexbin (raster)
            matplotlib function used in exposures
        args_imp : optional
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
        list(Impact)
        """
        if args_exp is None:
            args_exp = dict()
        if args_imp is None:
            args_imp = dict()
        imp_list = []
        exp_list = []
        imp_arr = np.zeros(len(exp.gdf))
        for i_time, _ in enumerate(haz_list):
            imp_tmp = Impact()
            imp_tmp.calc(exp, impf_set, haz_list[i_time])
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

    def _loc_return_imp(self, return_periods, imp, exc_imp):
        """Compute local exceedence impact for given return period.

        Parameters
        ----------
        return_periods : np.array return periods to consider
        cen_pos (int): centroid position

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

        Note
        ----
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
        event_ids : list[int], optional
            Selection of events by their id. The default is None.
        event_names : list[str], optional
            Selection of events by their name. The default is None.
        dates : tuple(), optional
            (start-date, end-date), events are selected if they are >=
            than start-date and <= than end-date. Dates in same format
            as impact.date (ordinal format of datetime library)
            The default is None.
        coord_exp : np.ndarray, optional
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


class ImpactFreqCurve():
    """Impact exceedence frequency curve.

    Attributes
    ----------
    tag : dict
        dictionary of tags of exposures, impact functions set and
        hazard: {'exp': Tag(), 'impf_set': Tag(), 'haz': TagHazard()}
    return_per : np.array
        return period
    impact : np.array
        impact exceeding frequency
    unit : str
        value unit used (given by exposures unit)
    label : str
        string describing source data
    """
    def __init__(self):
        self.tag = dict()
        self.return_per = np.array([])
        self.impact = np.array([])
        self.unit = ''
        self.label = ''

    def plot(self, axis=None, log_frequency=False, **kwargs):
        """Plot impact frequency curve.

        Parameters
        ----------
        axis  : matplotlib.axes._subplots.AxesSubplot, optional
            axis to use
        log_frequency : boolean, optional
            plot logarithmioc exceedance frequency on x-axis
        kwargs : optional
            arguments for plot matplotlib function, e.g. color='b'

        Returns
        -------
        matplotlib.axes._subplots.AxesSubplot
        """
        if not axis:
            _, axis = plt.subplots(1, 1)
        axis.set_title(self.label)
        axis.set_ylabel('Impact (' + self.unit + ')')
        if log_frequency:
            axis.set_xlabel('Exceedance frequency (1/year)')
            axis.set_xscale('log')
            axis.plot(self.return_per**-1, self.impact, **kwargs)
        else:
            axis.set_xlabel('Return period (year)')
            axis.plot(self.return_per, self.impact, **kwargs)
        return axis
