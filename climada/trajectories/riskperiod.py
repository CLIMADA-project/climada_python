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
from weakref import WeakValueDictionary

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

from climada.engine.impact_calc import ImpactCalc
from climada.entity.measures.base import Measure
from climada.trajectories.snapshot import Snapshot

LOGGER = logging.getLogger(__name__)


class RiskPeriod:

    # TODO: make lazy / delayed interpolation and impacts
    # TODO: make MeasureRiskPeriod child class (with effective start/end)
    # TODO: special case where hazard and exposure don't change (no need to interpolate) ?

    _instances = WeakValueDictionary()

    def __new__(
        cls,
        snapshot0,
        snapshot1,
        measure_name="no_measure",
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual=True,
    ):
        """Ensure only one instance exists per snapshot pair."""
        key = (id(snapshot0), id(snapshot1), measure_name)
        if key in cls._instances:
            LOGGER.debug("Found existing RiskPeriod")
            return cls._instances[key]

        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        snapshot0: Snapshot,
        snapshot1: Snapshot,
        measure_name="no_measure",
        risk_transf_cover=None,
        risk_transf_attach=None,
        calc_residual=True,
    ):
        if hasattr(self, "_initialized"):
            return  # Avoid re-initialization

        LOGGER.debug(
            f"Initializing new RiskPeriod from {snapshot0.year} to {snapshot1.year}, with snapshot0: {id(snapshot0)}, snapshot1: {id(snapshot1)}"
        )
        self.snapshot0 = snapshot0
        self.snapshot1 = snapshot1
        self.start_year = snapshot0.year
        self.end_year = snapshot1.year
        self.measure_name = measure_name
        self.impfset = snapshot0.impfset

        self._prop_H0, self._prop_H1 = bayesian_viktypliers(
            snapshot0.year, snapshot1.year
        )

        self._exp_y0 = snapshot0.exposure
        self._exp_y1 = snapshot1.exposure
        self._haz_y0 = snapshot0.hazard
        self._haz_y1 = snapshot1.hazard

        # Compute impacts once
        LOGGER.debug("Computing snapshots combination impacts")
        imp_E0H0 = self._compute_impact(self._exp_y0, self._haz_y0)
        imp_E1H0 = self._compute_impact(self._exp_y1, self._haz_y0)
        imp_E0H1 = self._compute_impact(self._exp_y0, self._haz_y1)
        imp_E1H1 = self._compute_impact(self._exp_y1, self._haz_y1)

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
        self.imp_mats_0 = interpolate_imp_mat(
            imp_E0H0, imp_E1H0, snapshot0.year, snapshot1.year
        )
        LOGGER.debug("Interpolating impact matrices between E0H1 and E1H1")
        self.imp_mats_1 = interpolate_imp_mat(
            imp_E0H1, imp_E1H1, snapshot0.year, snapshot1.year
        )
        LOGGER.debug("Done")

        self.year_idx = pd.Index(
            list(range(snapshot0.year, snapshot1.year + 1)), name="year"
        )

        self._initialized = True

        # Store the instance in the cache after initialization
        key = (id(self.snapshot0), id(self.snapshot1), self.measure_name)
        if key not in self._instances:
            LOGGER.debug(f"Created and stored new RiskPeriod {key}")
            self._instances[key] = self

    def _compute_impact(self, exposure, hazard):
        """Compute the impact once per unique exposure-hazard pair."""
        return ImpactCalc(exposure, self.impfset, hazard).impact()

    def get_interp(self):
        return self.imp_mats_0, self.imp_mats_1

    def apply_measure(self, measure: Measure):
        # Apply measure on snapshot and return risk period instance
        snapshot0 = self.snapshot0.apply_measure(measure)
        snapshot1 = self.snapshot1.apply_measure(measure)
        return RiskPeriod(snapshot0, snapshot1, measure_name=measure.name)

    def calc_waterfall_plot(self):

        pass

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


def interpolate_years(year_start, year_end):
    """
    Generate an array of interpolated values between 0 and 1 for a range of years.

    Parameters
    ----------
    year_start : int
        The starting year of interpolation.
    year_end : int
        The ending year of interpolation.

    Returns
    -------
    np.ndarray
        Array of interpolated values between 0 and 1 for each year in the range.
    """
    values = np.linspace(0, 1, num=year_end - year_start + 1)
    return values


def bayesian_viktypliers(year0, year1):
    """
    Calculate the Bayesian interpolation proportions for a given year range.

    Parameters
    ----------
    year0 : int
        Starting year.
    year1 : int
        Ending year.

    Returns
    -------
    tuple of np.ndarray
        Tuple containing:
        - prop_H0 : np.ndarray
            Array of proportions for the H0 hypothesis.
        - prop_H1 : np.ndarray
            Array of proportions for the H1 hypothesis.
    """
    prop_H1 = interpolate_years(year0, year1)
    prop_H0 = 1 - prop_H1
    return prop_H0, prop_H1


def interpolate_imp_mat(imp0, imp1, start_year, end_year):
    """
    Interpolate between two impact matrices over a specified year range.

    Parameters
    ----------
    imp0 : ImpactCalc
        The impact calculation for the starting year.
    imp1 : ImpactCalc
        The impact calculation for the ending year.
    start_year : int
        The starting year for interpolation.
    end_year : int
        The ending year for interpolation.

    Returns
    -------
    list of np.ndarray
        List of interpolated impact matrices for each year in the specified range.
    """

    def interpolate_sm(mat_start, mat_end, year, year_start, year_end):
        """Perform linear interpolation between two matrices for a specified year."""
        if year < year_start or year > year_end:
            raise ValueError("Year must be within the start and end years")

        # Calculate the ratio of the difference between the target year and the start year
        # to the total number of years between the start and end years
        ratio = (year - year_start) / (year_end - year_start)

        # Convert the input matrices to a format that allows efficient modification of its elements
        mat_start = lil_matrix(mat_start)
        mat_end = lil_matrix(mat_end)

        # Perform the linear interpolation
        mat_interpolated = mat_start + ratio * (mat_end - mat_start)

        return mat_interpolated

    LOGGER.debug(f"imp0: {imp0.imp_mat.data[0]}, imp1: {imp1.imp_mat.data[0]}")
    return [
        interpolate_sm(imp0.imp_mat, imp1.imp_mat, year, start_year, end_year)
        for year in range(start_year, end_year + 1)
    ]
