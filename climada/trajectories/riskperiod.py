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

This modules implements the Snapshot and SnapshotsCollection classes.

"""

import copy
import logging

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from climada.engine.impact_calc import ImpactCalc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.base import Measure
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)


class RiskPeriod:
    """Interpolated impacts between two snapshots.

    This class calculates the interpolated impacts between two snapshots over a specified
    time period. It supports risk transfer modifications and can compute residual impacts.

    Attributes
    ----------
    snapshot0 : Snapshot
        The snapshot starting the period.
    snapshot1 : Snapshot
        The snapshot ending the period.
    start_date : datetime
        The start date of the risk period.
    end_date : datetime
        The end date of the risk period.
    time_frequency : str
        The frequency of the time intervals (e.g., 'YS' for yearly).
        See `pandas freq string documentation <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases>`_.
    date_idx : pd.DatetimeIndex
        The date range index between the start and end dates.
    measure_name : str
        The name of the measure applied to the period. "no_measure" if no measure is applied.
    impfset : object
        The impact function set for the period. If both snapshots do not share the same ImpactFuncSet object,
        they are merged together. Note that if impact functions with the same hazard type and id differ,
        the one from the ending Snapshot takes precedence.
    """

    # Future TODO: make lazy / delayed interpolation and impacts
    # Future TODO: special case where hazard and exposure don't change (no need to interpolate) ?

    def __init__(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        measure_name="no_measure",
        time_freq="YS",
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual=True,
    ):
        LOGGER.debug(
            f"Initializing new RiskPeriod from {snapshot0.date} to {snapshot1.date}, with snapshot0: {id(snapshot0)}, snapshot1: {id(snapshot1)}"
        )
        self.snapshot0 = snapshot0
        self.snapshot1 = snapshot1
        self.start_date = snapshot0.date
        self.end_date = snapshot1.date
        self.time_frequency = time_freq
        self.date_idx = pd.date_range(
            snapshot0.date, snapshot1.date, freq=time_freq, name="date"
        )
        self.measure_name = measure_name
        self.impfset = self._merge_impfset(snapshot0.impfset, snapshot1.impfset)

        # Posterity comment: The following attributes
        # were refered as Victypliers in homage to Victor
        # Watkinsson, the conceptual father of this module
        self._prop_H1 = np.linspace(0, 1, num=len(self.date_idx))
        self._prop_H0 = 1 - self._prop_H1

        self._exp_y0 = snapshot0.exposure
        self._exp_y1 = snapshot1.exposure
        self._haz_y0 = snapshot0.hazard
        self._haz_y1 = snapshot1.hazard

        # Compute impacts once
        LOGGER.debug("Computing snapshots combination impacts")
        imp_E0H0 = ImpactCalc(self._exp_y0, self.impfset, self._haz_y0).impact()
        imp_E1H0 = ImpactCalc(self._exp_y1, self.impfset, self._haz_y0).impact()
        imp_E0H1 = ImpactCalc(self._exp_y0, self.impfset, self._haz_y1).impact()
        imp_E1H1 = ImpactCalc(self._exp_y1, self.impfset, self._haz_y1).impact()

        # Modify the impact matrices if risk transfer is provided
        # TODO: See where this ends up
        imp_E0H0.imp_mat = self.calc_residual_or_risk_transf_imp_mat(
            imp_E0H0.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )
        imp_E1H0.imp_mat = self.calc_residual_or_risk_transf_imp_mat(
            imp_E1H0.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )
        imp_E0H1.imp_mat = self.calc_residual_or_risk_transf_imp_mat(
            imp_E0H1.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )
        imp_E1H1.imp_mat = self.calc_residual_or_risk_transf_imp_mat(
            imp_E1H1.imp_mat, risk_transf_attach, risk_transf_cover, calc_residual
        )

        LOGGER.debug("Interpolating impact matrices between E0H0 and E1H0")
        time_points = len(self.date_idx)
        self._imp_mats_0 = interpolate_imp_mat(imp_E0H0, imp_E1H0, time_points)
        LOGGER.debug("Interpolating impact matrices between E0H1 and E1H1")
        self._imp_mats_1 = interpolate_imp_mat(imp_E0H1, imp_E1H1, time_points)
        LOGGER.debug("Done")

    @staticmethod
    def _merge_impfset(impfs1: ImpactFuncSet, impfs2: ImpactFuncSet):
        if impfs1 == impfs2:
            return impfs1
        else:
            LOGGER.warning(
                "Impact function sets differ. Will update the first one with the second."
            )
            impfs1._data |= impfs2._data  # Merges dictionaries (priority to impfs2)
            return impfs1

    def get_interp(self):
        """Return two lists of interpolated impacts matrices with varying exposure, for starting and ending hazard.

        Returns
        -------

        _imp_mats_0 : np.ndarray
            Interpolated impact matrices varying Exposure from starting snapshot to ending one, using Hazard from starting snapshot.
        _imp_mats_1 : np.ndarray
            Interpolated impact matrices varying Exposure from starting snapshot to ending one, using Hazard from ending snapshot.
        """
        return self._imp_mats_0, self._imp_mats_1

    def apply_measure(self, measure: Measure):
        """Applies measure to RiskPeriod, returns a new object"""
        snapshot0 = self.snapshot0.apply_measure(measure)
        snapshot1 = self.snapshot1.apply_measure(measure)
        return RiskPeriod(snapshot0, snapshot1, measure_name=measure.name)

    @classmethod
    def calc_residual_or_risk_transf_imp_mat(
        cls, imp_mat, attachment=None, cover=None, calc_residual=True
    ):
        """
        Calculate either the residual or the risk transfer impact matrix.

        The impact matrix is adjusted based on the total impact for each event.
        When calculating the residual impact, the result is the total impact minus
        the risk layer. The risk layer is defined as the minimum of the cover and
        the maximum of the difference between the total impact and the attachment.
        If `calc_residual` is False, the function returns the risk layer matrix
        instead of the residual.

        Parameters
        ----------
        imp_mat : scipy.sparse.csr_matrix
            The original impact matrix to be scaled.
        attachment : float, optional
            The attachment point for the risk layer.
        cover : float, optional
            The maximum coverage for the risk layer.
        calc_residual : bool, default=True
            Determines if the function calculates the residual (if True) or the
            risk layer (if False).

        Returns
        -------
        scipy.sparse.csr_matrix
            The adjusted impact matrix, either residual or risk transfer.

        Example
        -------
        >>> calc_residual_or_risk_transf_imp_mat(imp_mat, attachment=100, cover=500, calc_residual=True)
        Residual impact matrix with applied risk layer adjustments.
        """
        if attachment and cover:
            # Make a copy of the impact matrix
            imp_mat = copy.deepcopy(imp_mat)
            # Calculate the total impact per event
            total_at_event = imp_mat.sum(axis=1).A1
            # Risk layer at event
            transfer_at_event = np.minimum(
                np.maximum(total_at_event - attachment, 0), cover
            )
            # Resiudal impact
            residual_at_event = np.maximum(total_at_event - transfer_at_event, 0)

            # Calculate either the residual or transfer impact matrix
            # Choose the denominator to rescale the impact values
            if calc_residual:
                # Rescale the impact values
                numerator = residual_at_event
            else:
                # Rescale the impact values
                numerator = transfer_at_event

            # Rescale the impact values
            rescale_impact_values = np.divide(
                numerator,
                total_at_event,
                out=np.zeros_like(numerator, dtype=float),
                where=total_at_event != 0,
            )

            # The multiplication is broadcasted across the columns for each row
            result_matrix = imp_mat.multiply(rescale_impact_values[:, np.newaxis])

            return result_matrix

        else:

            return imp_mat


def interpolate_imp_mat(imp0, imp1, time_points):
    """
    Interpolate between two impact matrices over a specified time range.

    Parameters
    ----------
    imp0 : ImpactCalc
        The impact calculation for the starting time.
    imp1 : ImpactCalc
        The impact calculation for the ending time.
    time_points:
        The number of points to interpolate.

    Returns
    -------
    list of np.ndarray
        List of interpolated impact matrices for each time points in the specified range.
    """

    def interpolate_sm(mat_start, mat_end, time, time_points):
        """Perform linear interpolation between two matrices for a specified time point."""
        if time > time_points:
            raise ValueError("time point must be within the range")

        ratio = time / (time_points - 1)

        # Convert the input matrices to a format that allows efficient modification of its elements
        mat_start = lil_matrix(mat_start)
        mat_end = lil_matrix(mat_end)

        # Perform the linear interpolation
        mat_interpolated = mat_start + ratio * (mat_end - mat_start)

        return mat_interpolated

    LOGGER.debug(f"imp0: {imp0.imp_mat.data[0]}, imp1: {imp1.imp_mat.data[0]}")
    return [
        interpolate_sm(imp0.imp_mat, imp1.imp_mat, time, time_points)
        for time in range(time_points)
    ]
