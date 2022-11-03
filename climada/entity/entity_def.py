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

Define Entity Class.
"""

__all__ = ['Entity']

import logging
from typing import Optional
import pandas as pd

from climada.entity.tag import Tag
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.entity.exposures.base import Exposures

LOGGER = logging.getLogger(__name__)

class Entity:
    """Collects exposures, impact functions, measures and discount rates.
    Default values set when empty constructor.

    Attributes
    ----------
    exposures : Exposures
        exposures
    impact_funcs : ImpactFuncSet
        impact functions set
    measures : MeasureSet
        measures
    disc_rates : DiscRates
        discount rates
    def_file : str
        Default file from configuration file
    """

    def __init__(
        self,
        exposures: Optional[Exposures] = None,
        disc_rates: Optional[DiscRates] = None,
        impact_func_set: Optional[ImpactFuncSet] = None,
        measure_set: Optional[MeasureSet] = None
    ):
        """
        Initialize entity

        Parameters
        ----------
        exposures : climada.entity.Exposures, optional
            Exposures of the entity. The default is None (empty Exposures()).
        disc_rates : climada.entity.DiscRates, optional
            Disc rates of the entity. The default is None (empty DiscRates()).
        impact_func_set : climada.entity.ImpactFuncSet, optional
            The impact function set. The default is None (empty ImpactFuncSet()).
        measure_set : climada.entity.Measures, optional
            The measures. The default is None (empty MeasuresSet().
        """
        self.exposures = Exposures() if exposures is None else exposures
        self.disc_rates = DiscRates() if disc_rates is None else disc_rates
        self.impact_funcs = ImpactFuncSet() if impact_func_set is None else impact_func_set
        self.measures = MeasureSet() if measure_set is None else measure_set

    @classmethod
    def from_mat(cls, file_name, description=''):
        """Read MATLAB file of climada.

        Parameters
        ----------
        file_name : str, optional
            file name(s) or folder name
            containing the files to read
        description : str or list(str), optional
            one description of the
            data or a description of each data file

        Returns
        -------
        ent : climada.entity.Entity
            The entity from matlab file
        """
        return cls(
            exposures=Exposures.from_mat(file_name),
            disc_rates=DiscRates.from_mat(file_name, description),
            impact_func_set=ImpactFuncSet.from_mat(file_name, description),
            measure_set=MeasureSet.from_mat(file_name, description)
            )

    def read_mat(self, *args, **kwargs):
        """This function is deprecated, use Entity.from_mat instead."""
        LOGGER.warning("The use of Entity.read_mat is deprecated."
                        "Use Entity.from_mat instead.")
        self.__dict__ = Entity.from_mat(*args, **kwargs).__dict__

    @classmethod
    def from_excel(cls, file_name, description=''):
        """Read csv or xls or xlsx file following climada's template.

        Parameters
        ----------
        file_name : str, optional
            file name(s) or folder name
            containing the files to read
        description : str or list(str), optional
            one description of the
            data or a description of each data file

        Returns
        -------
        ent : climada.entity.Entity
            The entity from excel file
        """

        exp = Exposures(pd.read_excel(file_name))
        exp.tag = Tag()
        exp.tag.file_name = str(file_name)
        exp.tag.description = description

        dr = DiscRates.from_excel(file_name, description)
        impf_set = ImpactFuncSet.from_excel(file_name, description)
        meas_set = MeasureSet.from_excel(file_name, description)

        return cls(
            exposures=exp,
            disc_rates=dr,
            impact_func_set=impf_set,
            measure_set=meas_set
            )

    def read_excel(self, *args, **kwargs):
        """This function is deprecated, use Entity.from_excel instead."""
        LOGGER.warning("The use of Entity.read_excel is deprecated."
                       " Use Entity.from_excel instead.")
        self.__dict__ = Entity.from_excel(*args, **kwargs).__dict__

    def write_excel(self, file_name):
        """Write excel file following template."""
        self.exposures.gdf.to_excel(file_name)
        self.impact_funcs.write_excel(file_name)
        self.measures.write_excel(file_name)
        self.disc_rates.write_excel(file_name)

    def check(self):
        """Check instance attributes.

        Raises
        ------
        ValueError
        """
        self.disc_rates.check()
        self.exposures.check()
        self.impact_funcs.check()
        self.measures.check()

    def __setattr__(self, name, value):
        """Check input type before set"""
        if name == "exposures":
            if not isinstance(value, Exposures):
                raise ValueError("Input value is not (sub)class of Exposures.")
        elif name == "impact_funcs":
            if not isinstance(value, ImpactFuncSet):
                raise ValueError("Input value is not (sub)class of ImpactFuncSet.")
        elif name == "measures":
            if not isinstance(value, MeasureSet):
                raise ValueError("Input value is not (sub)class of MeasureSet.")
        elif name == "disc_rates":
            if not isinstance(value, DiscRates):
                raise ValueError("Input value is not (sub)class of DiscRates.")
        super().__setattr__(name, value)

    def __str__(self):
        return 'Exposures: \n' + self.exposures.tag.__str__() + \
                '\nDiscRates: \n' + self.disc_rates.tag.__str__() + \
                '\nImpactFuncSet: \n' + self.impact_funcs.tag.__str__() + \
                '\nMeasureSet: \n' + self.measures.tag.__str__()

    __repr__ = __str__
