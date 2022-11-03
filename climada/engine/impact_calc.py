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

Define ImpactCalc class.
"""

__all__ = ['ImpactCalc']

import logging
import numpy as np
from scipy import sparse
import geopandas as gpd

from climada import CONFIG
from climada.engine import Impact

LOGGER = logging.getLogger(__name__)


class ImpactCalc():
    """
    Class to compute impacts from exposures, impact function set and hazard
    """

    def __init__(self,
                 exposures,
                 impfset,
                 hazard):
        """
        ImpactCalc constructor

        The dimension of the imp_mat variable must be compatible with the
        exposures and hazard objects.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures used to compute impacts
        impf_set: climada.entity.ImpactFuncSet
            impact functions set used to compute impacts
        hazard : climada.Hazard
            hazard used to compute impacts

        """

        self.exposures = exposures
        self.impfset = impfset
        self.hazard = hazard
        # exposures index to use for matrix reconstruction
        self._orig_exp_idx = np.arange(self.exposures.gdf.shape[0])

    @property
    def n_exp_pnt(self):
        """Number of exposure points (rows in gdf)"""
        return self.exposures.gdf.shape[0]

    @property
    def n_events(self):
        """Number of hazard events (size of event_id array)"""
        return self.hazard.size

    def impact(self, save_mat=True, assign_centroids=True,
               ignore_cover=False, ignore_deductible=False):
        """Compute the impact of a hazard on exposures.

        Parameters
        ----------
        save_mat : bool, optional
            if true, save the total impact matrix (events x exposures)
            Default: True
        assign_centroids : bool, optional
            indicates whether centroids are assigned to the self.exposures object.
            Centroids assignment is an expensive operation; set this to ``False`` to save
            computation time if the hazards' centroids are already assigned to the exposures
            object.
            Default: True
        ignore_cover : bool, optional
            if set to True, the column 'cover' of the exposures GeoDataFrame, if present, is
            ignored and the impact it not capped by the values in this column.
            Default: False
        ignore_deductible : bool, opotional
            if set to True, the column 'deductible' of the exposures GeoDataFrame, if present, is
            ignored and the impact it not reduced through values in this column.
            Default: False

        Examples
        --------
            >>> haz = Hazard.from_mat(HAZ_DEMO_MAT)  # Set hazard
            >>> impfset = ImpactFuncSet.from_excel(ENT_TEMPLATE_XLS)
            >>> exp = Exposures(pd.read_excel(ENT_TEMPLATE_XLS))
            >>> impcalc = ImpactCal(exp, impfset, haz)
            >>> imp = impcalc.impact(insured=True)
            >>> imp.aai_agg

        See also
        --------
        apply_deductible_to_mat : apply deductible to impact matrix
        apply_cover_to_mat : apply cover to impact matrix
        """
        impf_col = self.exposures.get_impf_column(self.hazard.haz_type)
        exp_gdf = self.minimal_exp_gdf(impf_col, assign_centroids, ignore_cover, ignore_deductible)
        if exp_gdf.size == 0:
            return self._return_empty(save_mat)
        LOGGER.info('Calculating impact for %s assets (>0) and %s events.',
                    exp_gdf.size, self.n_events)
        imp_mat_gen = self.imp_mat_gen(exp_gdf, impf_col)

        insured = ('cover' in exp_gdf and exp_gdf.cover.max() >= 0) \
               or ('deductible' in exp_gdf and exp_gdf.deductible.max() > 0)
        if insured:
            LOGGER.info("cover and/or deductible columns detected,"
                        " going to calculate insured impact")
#TODO: make a better impact matrix generator for insured impacts when
# the impact matrix is already present
            imp_mat_gen = self.insured_mat_gen(imp_mat_gen, exp_gdf, impf_col)

        return self._return_impact(imp_mat_gen, save_mat)

    def _return_impact(self, imp_mat_gen, save_mat):
        """Return an impact object from an impact matrix generator

        Parameters
        ----------
        imp_mat_gen : generator
            Generator of impact matrix and corresponding exposures index
        save_mat : boolean
            if true, save the impact matrix

        Returns
        -------
        Impact
            Impact Object initialize from the impact matrix

        See Also
        --------
        imp_mat_gen : impact matrix generator
        insured_mat_gen: insured impact matrix generator
        """
        if save_mat:
            imp_mat = self.stitch_impact_matrix(imp_mat_gen)
            at_event, eai_exp, aai_agg = \
                self.risk_metrics(imp_mat, self.hazard.frequency)
        else:
            imp_mat = None
            at_event, eai_exp, aai_agg = self.stitch_risk_metrics(imp_mat_gen)
        return Impact.from_eih(
            self.exposures, self.impfset, self.hazard,
            at_event, eai_exp, aai_agg, imp_mat
            )

    def _return_empty(self, save_mat):
        """
        Return empty impact.

        Parameters
        ----------
        save_mat : bool
              If true, save impact matrix

        Returns
        -------
        Impact
            Empty impact object with correct array sizes.
        """
        at_event = np.zeros(self.n_events)
        eai_exp = np.zeros(self.n_exp_pnt)
        aai_agg = 0.0
        if save_mat:
            imp_mat = sparse.csr_matrix((
                self.n_events, self.n_exp_pnt), dtype=np.float64
                )
        else:
            imp_mat = None
        return Impact.from_eih(self.exposures, self.impfset, self.hazard,
                        at_event, eai_exp, aai_agg, imp_mat)

    def minimal_exp_gdf(self, impf_col, assign_centroids, ignore_cover, ignore_deductible):
        """Get minimal exposures geodataframe for impact computation

        Parameters
        ----------
        exposures : climada.entity.Exposures
        hazard : climada.Hazard
        impf_col : str
            Name of the impact function column in exposures.gdf
        assign_centroids : bool
            Indicates whether centroids are re-assigned to the self.exposures object
            or kept from previous impact calculation with a hazard of the same hazard type.
            Centroids assignment is an expensive operation; set this to ``False`` to save
            computation time if the centroids have not changed since the last impact
            calculation.
        include_cover : bool
            if set to True, the column 'cover' of the exposures GeoDataFrame is excluded from the
            returned GeoDataFrame, otherwise it is included if present.
        include_deductible : bool
            if set to True, the column 'deductible' of the exposures GeoDataFrame is excluded from
            the returned GeoDataFrame, otherwise it is included if present.
        """
        if assign_centroids:
            self.exposures.assign_centroids(self.hazard, overwrite=True)
        elif self.hazard.centr_exp_col not in self.exposures.gdf.columns:
            raise ValueError("'assign_centroids' is set to 'False' but no centroids are assigned"
                             f" for the given hazard type ({self.hazard.tag.haz_type})."
                             " Run 'exposures.assign_centroids()' beforehand or set"
                             " 'assign_centroids' to 'True'")
        mask = (
            (self.exposures.gdf.value.values == self.exposures.gdf.value.values)  # value != NaN
            & (self.exposures.gdf.value.values != 0)                              # value != 0
            & (self.exposures.gdf[self.hazard.centr_exp_col].values >= 0)    # centroid assigned
        )

        columns = ['value', impf_col, self.hazard.centr_exp_col]
        if not ignore_cover and 'cover' in self.exposures.gdf:
            columns.append('cover')
        if not ignore_deductible and 'deductible' in self.exposures.gdf:
            columns.append('deductible')
        exp_gdf = gpd.GeoDataFrame(
            {col: self.exposures.gdf[col].values[mask]
            for col in columns},
            )
        if exp_gdf.size == 0:
            LOGGER.warning("No exposures with value >0 in the vicinity of the hazard.")
        self._orig_exp_idx = mask.nonzero()[0]  # update index of kept exposures points in exp_gdf
                                                # within the full exposures
        return exp_gdf

    def imp_mat_gen(self, exp_gdf, impf_col):
        """
        Generator of impact sub-matrices and correspoding exposures indices

        The exposures gdf is decomposed into chunks that fit into the max
        defined memory size. For each chunk, the impact matrix is computed
        and returned, together with the corresponding exposures points index.

        Parameters
        ----------
        exp_gdf : GeoDataFrame
            Geodataframe of the exposures with columns required for impact
            computation.
        impf_col : str
            name of the desired impact column in the exposures.

        Raises
        ------
        ValueError
            if the hazard is larger than the memory limit

        Yields
        ------
        scipy.sparse.crs_matrix, np.ndarray
            impact matrix and corresponding exposures indices for each chunk.

        """

        def _chunk_exp_idx(haz_size, idx_exp_impf):
            '''
            Chunk computations in sizes that roughly fit into memory
            '''
            max_size = CONFIG.max_matrix_size.int()
            if haz_size > max_size:
                raise ValueError(
                    f"Hazard size '{haz_size}' exceeds maximum matrix size '{max_size}'. "
                    "Increase max_matrix_size configuration parameter accordingly."
                )
            n_chunks = np.ceil(haz_size * len(idx_exp_impf) / max_size)
            return np.array_split(idx_exp_impf, n_chunks)

        for impf_id in exp_gdf[impf_col].dropna().unique():
            impf = self.impfset.get_func(
                haz_type=self.hazard.haz_type, fun_id=impf_id
                )
            idx_exp_impf = (exp_gdf[impf_col].values == impf_id).nonzero()[0]
            for exp_idx in _chunk_exp_idx(self.hazard.size, idx_exp_impf):
                exp_values = exp_gdf.value.values[exp_idx]
                cent_idx = exp_gdf[self.hazard.centr_exp_col].values[exp_idx]
                yield (
                    self.impact_matrix(exp_values, cent_idx, impf),
                    exp_idx
                    )

    def insured_mat_gen(self, imp_mat_gen, exp_gdf, impf_col):
        """
        Generator of insured impact sub-matrices (with applied cover and deductible)
        and corresponding exposures indices

        This generator takes a 'regular' impact matrix generator and applies cover and
        deductible onto the impacts. It yields the same sub-matrices as the original
        generator.

        Deductible and cover are taken from the dataframe stored in `exposures.gdf`.

        Parameters
        ----------
        imp_mat_gen : generator of tuples (sparse.csr_matrix, np.array)
            The generator for creating the impact matrix. It returns a part of the full
            matrix and the associated exposure indices.
        exp_gdf : GeoDataFrame
            Geodataframe of the exposures with columns required for impact computation.
        impf_col : str
            Name of the column in 'exp_gdf' indicating the impact function (id)

        Yields
        ------
        mat : scipy.sparse.csr_matrix
            Impact sub-matrix (with applied cover and deductible) with size
            (n_events, len(exp_idx))
        exp_idx : np.array
            Exposure indices for impacts in mat
        """
        for mat, exp_idx in imp_mat_gen:
            impf_id = exp_gdf[impf_col][exp_idx[0]]
            cent_idx = exp_gdf[self.hazard.centr_exp_col].values[exp_idx]
            impf = self.impfset.get_func(
                haz_type=self.hazard.haz_type,
                fun_id=impf_id)
            if 'deductible' in exp_gdf:
                deductible = exp_gdf.deductible.values[exp_idx]
                mat = self.apply_deductible_to_mat(mat, deductible, self.hazard, cent_idx, impf)
            if 'cover' in exp_gdf:
                cover = exp_gdf.cover.values[exp_idx]
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
        n_exp_pnt = len(cent_idx)  # implicitly checks in matrix assignement whether
                                   # len(cent_idx) == len(exp_values)
        mdr = self.hazard.get_mdr(cent_idx, impf)
        exp_values_csr = sparse.csr_matrix(  # vector 1 x exp_size
            (exp_values, np.arange(n_exp_pnt), [0, n_exp_pnt]),
            shape=(1, n_exp_pnt))
        fract = self.hazard._get_fraction(cent_idx)  # pylint: disable=protected-access
        if fract is None:
            return mdr.multiply(exp_values_csr)

        return fract.multiply(mdr).multiply(exp_values_csr)

    def stitch_impact_matrix(self, imp_mat_gen):
        """
        Make an impact matrix from an impact sub-matrix generator
        """
        # rows: events index
        # cols: exposure point index within self.exposures
        data, row, col = np.hstack([
            (mat.data, mat.nonzero()[0], self._orig_exp_idx[idx][mat.nonzero()[1]])
            for mat, idx in imp_mat_gen
            ])
        return sparse.csr_matrix(
            (data, (row, col)), shape=(self.n_events, self.n_exp_pnt)
            )

    def stitch_risk_metrics(self, imp_mat_gen):
        """Compute the impact metrics from an impact sub-matrix generator

        This method is used to compute the risk metrics if the user decided not to store
        the full impact matrix.

        Parameters
        ----------
        imp_mat_gen : generator of tuples (sparse.csr_matrix, np.array)
            The generator for creating the impact matrix. It returns a part of the full
            matrix and the associated exposure indices.

        Returns
        -------
        at_event : np.array
            Accumulated damage for each event
        eai_exp : np.array
            Expected impact within a period of 1/frequency_unit for each exposure point
        aai_agg : float
            Average impact within a period of 1/frequency_unit aggregated
        """
        at_event = np.zeros(self.n_events)
        eai_exp = np.zeros(self.n_exp_pnt)
        for sub_imp_mat, idx in imp_mat_gen:
            at_event += self.at_event_from_mat(sub_imp_mat)
            eai_exp[self._orig_exp_idx[idx]] += \
                self.eai_exp_from_mat(sub_imp_mat, self.hazard.frequency)
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
        mat.data = np.clip(mat.data, 0, cover[mat.nonzero()[1]])
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
            frequency of events within a period of 1/frequency_unit
        Returns
        -------
        eai_exp : np.array
            expected impact within a period of 1/frequency_unit for each exposure
        """
        n_events = freq.size
        freq_csr = sparse.csr_matrix(   #vector n_events x 1
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
        return np.asarray(np.sum(mat, axis=1)).ravel()

    @staticmethod
    def aai_agg_from_eai_exp(eai_exp):
        """
        Aggregate impact.eai_exp

        Parameters
        ----------
        eai_exp : np.array
            expected impact within a period of 1/frequency_unit for each exposure point

        Returns
        -------
        float
            average aggregated impact within a period of 1/frequency_unit
        """
        return np.sum(eai_exp)

    @classmethod
    def risk_metrics(cls, mat, freq):
        """
        Compute risk metricss eai_exp, at_event, aai_agg
        for an impact matrix and a frequency vector.

        Parameters
        ----------
        mat : sparse.csr_matrix
            matrix num_events x num_exp with impacts.
        freq : np.array
            array with the frequency per event

        Returns
        -------
        eai_exp: np.array
            expected impact within a period of 1/frequency_unit at each exposure point
        at_event: np.array()
            total impact for each event
        aai_agg : float
            average impact within a period of 1/frequency_unit aggregated over all exposure points
        """
        eai_exp = cls.eai_exp_from_mat(mat, freq)
        at_event = cls.at_event_from_mat(mat)
        aai_agg = cls.aai_agg_from_eai_exp(eai_exp)
        return at_event, eai_exp, aai_agg
