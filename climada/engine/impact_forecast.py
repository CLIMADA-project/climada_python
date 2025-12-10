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

Define Forecast variant of Impact.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np

from ..util import log_level
from ..util.checker import size
from ..util.forecast import Forecast
from .impact import Impact

LOGGER = logging.getLogger(__name__)


class ImpactForecast(Forecast, Impact):
    """An impact object with forecast information"""

    def __init__(
        self,
        *,
        lead_time: np.ndarray | None,
        member: np.ndarray | None,
        **impact_kwargs,
    ):
        """Initialize the impact forecast.

        Parameters
        ----------
        lead_time : np.ndarray, optional
            The lead time associated with each event entry, given as timedelta64 type
        member : np.ndarray, optional
            The ensemble member associated with each event entry, given as integers
        impact_kwargs
            Keyword-arguments passed to ~:py:class`climada.engine.impact.Impact`.
        """
        super().__init__(lead_time=lead_time, member=member, **impact_kwargs)
        self._check_sizes()

    @classmethod
    def from_impact(
        cls, impact: Impact, lead_time: np.ndarray | None, member: np.ndarray | None
    ):
        """Create an impact forecast from an impact object and forecast information.

        Parameters
        ----------
        impact : climada.engine.impact.Impact
            The impact object whose data to use in the forecast object
        lead_time : np.ndarray, optional
            The lead time associated with each event entry, given as timedelta64 type
        member : np.ndarray, optional
            The ensemble member associated with each event entry, given as integers
        """
        with log_level("WARNING", "climada.engine.impact"):
            return cls(
                lead_time=lead_time,
                member=member,
                event_id=impact.event_id,
                event_name=impact.event_name,
                date=impact.date,
                frequency=impact.frequency,
                frequency_unit=impact.frequency_unit,
                coord_exp=impact.coord_exp,
                crs=impact.crs,
                eai_exp=impact.eai_exp,
                at_event=impact.at_event,
                tot_value=impact.tot_value,
                aai_agg=impact.aai_agg,
                unit=impact.unit,
                imp_mat=impact.imp_mat,
                haz_type=impact.haz_type,
            )

    @property
    def at_event(self):
        """Get the total impact for each member/lead_time combination."""
        LOGGER.warning(
            "at_event gives the total impact for one specific combination of member and "
            "lead_time."
        )
        return self._at_event

    @at_event.setter
    def at_event(self, value):
        """Set the total impact for each member/lead_time combination."""
        self._at_event = value

    def local_exceedance_impact(
        self,
        return_periods=(25, 50, 100, 250),
        method="interpolate",
        min_impact=0,
        log_frequency=True,
        log_impact=True,
        bin_decimals=None,
    ):
        """Compution of local exceedance impact for given return periods is not
        implemented for ImpactForecast.

        See Also
        --------
        See :py:meth:`~climada.engine.impact.Impact.local_exceedance_impact`

        Raises
        ------
        NotImplementedError
        """

        LOGGER.error("local_exceedance_impact is not defined for ImpactForecast")
        raise NotImplementedError(
            "local_exceedance_impact is not defined for ImpactForecast"
        )

    def local_return_period(
        self,
        threshold_impact=(1000.0, 10000.0),
        method="interpolate",
        min_impact=0,
        log_frequency=True,
        log_impact=True,
        bin_decimals=None,
    ):
        """Compution of local return period for given impact thresholds is not
        implemented for ImpactForecast.

        See Also
        --------
        See :py:meth:`~climada.engine.impact.Impact.local_return_period`

        Raises
        -------
        NotImplementedError
        """

        LOGGER.error("local_return_period is not defined for ImpactForecast")
        raise NotImplementedError(
            "local_return_period is not defined for ImpactForecast"
        )

    def calc_freq_curve(self, return_per=None):
        """Computation of the impact exceedance frequency curve is not
        implemented for ImpactForecast.

        See Also
        --------
        See :py:meth:`~climada.engine.impact.Impact.calc_freq_curve`

        Raises
        ------
        NotImplementedError
        """

        LOGGER.error("calc_freq_curve is not defined for ImpactForecast")
        raise NotImplementedError("calc_freq_curve is not defined for ImpactForecast")

    @classmethod
    def from_hdf5(cls, file_path: Union[str, Path]):
        """Create an ImpactForecast object from an H5 file.

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
            ├─ lead_time
            ├─ member
            ├─ .attrs/
            │  ├─ aai_agg
            │  ├─ crs
            │  ├─ frequency_unit
            │  ├─ haz_type
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
        imp : ImpactForecast
            ImpactForecast with data from the given file
        """
        return super().from_hdf5(file_path, add_array_attrs={"member", "lead_time"})

    def _check_sizes(self):
        """Check sizes of forecast data vs. impact data.

        Raises
        ------
        ValueError
            If the sizes of the forecast data do not match the
            :py:attr:`~climada.engine.impact.Impact.event_id`
        """
        num_entries = len(self.event_id)
        size(exp_len=num_entries, var=self.member, var_name="Forecast.member")
        size(exp_len=num_entries, var=self.lead_time, var_name="Forecast.lead_time")
