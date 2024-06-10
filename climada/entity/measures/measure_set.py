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

Define MeasureSet class.
"""

__all__ = ['MeasureSet']

import logging
from typing import Iterable


from climada.entity.measures.base import Measure

LOGGER = logging.getLogger(__name__)

class MeasureSet():
    """Contains measures of type Measure.

    Attributes
    ----------
    _data : dict
        Contains Measure objects. This attribute is not suppossed to be accessed directly.
        Use the available methods instead.
    """

    def __init__(
        self,
        measures: Iterable[Measure]
    ):
        """Initialize a new MeasureSet object with specified data.

        Parameters
        ----------
        measures : iterable of Measure objects.
            The measures to include in the MeasureSet

        """
        for meas in measures:
            self.append(meas)

    def append(self, measure):
        """Append an Measure. Override if same name and haz_type.

        Parameters
        ----------
        meas : Measure
            Measure instance

        Raises
        ------
        ValueError
        """
        if not isinstance(measure, Measure):
            raise ValueError("Input value is not of type Measure.")
        if meas.haz_type not in self.measures:
            self._data[meas.haz_type] = dict()
        self._data[meas.haz_type][meas.name] = measure

    def measures(self, haz_type=None, name=None):
        """Get ImpactFunc(s) of input hazard type and/or id.
        If no input provided, all impact functions are returned.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        name : str, optional
            measure name

        Returns
        -------
        Measure (if haz_type and name),
        list(Measure) (if haz_type or name),
        {Measure.haz_type : {Measure.name : Measure}} (if None)
        """
        if (haz_type is not None) and (name is not None):
            try:
                return self._data[haz_type][name]
            except KeyError:
                LOGGER.info("No Measure with hazard %s and id %s.",
                            haz_type, name)
                return list()
        elif haz_type is not None:
            try:
                return list(self._data[haz_type].values())
            except KeyError:
                LOGGER.info("No Measure with hazard %s.", haz_type)
                return list()
        elif name is not None:
            haz_return = self.get_hazard_types(name)
            if not haz_return:
                LOGGER.info("No Measure with name %s.", name)
            meas_return = []
            for haz in haz_return:
                meas_return.append(self._data[haz][name])
            return meas_return
        else:
            return self._data

    def names(self, haz_type=None):
        """Get measures names contained for the hazard type provided.
        Return all names for each hazard type if no input hazard type.

        Parameters
        ----------
        haz_type : str, optional
            hazard type from which to obtain the names

        Returns
        -------
        list(Measure.name) (if haz_type provided),
        {Measure.haz_type : list(Measure.name)} (if no haz_type)
        """
        if haz_type is None:
            out_dict = dict()
            for haz, haz_dict in self._data.items():
                out_dict[haz] = list(haz_dict.keys())
            return out_dict

        try:
            return list(self._data[haz_type].keys())
        except KeyError:
            LOGGER.info("No Measure with hazard %s.", haz_type)
            return list()

    def size(self, haz_type=None, name=None):
        """Get number of measures contained with input hazard type and
        /or id. If no input provided, get total number of impact functions.

        Parameters
        ----------
        haz_type : str, optional
            hazard type
        name : str, optional
            measure name

        Returns
        -------
        int
        """
        if (haz_type is not None) and (name is not None) and \
        (isinstance(self.get_measure(haz_type, name), Measure)):
            return 1
        if (haz_type is not None) or (name is not None):
            return len(self.get_measure(haz_type, name))
        return sum(len(meas_list) for meas_list in self.get_names().values())
