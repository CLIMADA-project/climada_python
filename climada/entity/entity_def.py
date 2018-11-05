"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Entity Class.
"""

__all__ = ['Entity']

import logging

from climada.entity.impact_funcs.impact_func_set  import ImpactFuncSet
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.measure_set import MeasureSet
from climada.entity.exposures.base import Exposures
from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

class Entity(object):
    """Collects exposures, impact functions, measures and discount rates.
    Default values set when empty constructor.

    Attributes:
        exposures (Exposures): exposures
        impact_funcs (ImpactFucs): impact functions
        measures (MeasureSet): measures
        disc_rates (DiscRates): discount rates
        def_file (str): Default file from configuration file
    """

    def_file = CONFIG['local_data']['entity_def']

    def __init__(self, file_name=None, description=''):
        """Fill values from file. Default file used when no file provided.

        Parameters:
            file_name (str or list(str), optional): file name(s) or folder name
                containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises:
            ValueError

        Examples:
            Builds Entity with values obtained from configurable entity_def:

            >>> Entity()

            Builds Entity with the values obtained from file:

            >>> Entity(ENT_TEMPLATE_XLS)

            Builds Entity with exposures and discount rates from configurable
            entity_def file, and the given impact functions and measures:

            >>> Entity(impact_funcs=myimpact_funcs, measures=mymeasures)

        """
        if file_name is None:
            self.exposures = Exposures(self.def_file)
            self.impact_funcs = ImpactFuncSet(self.def_file)
            self.measures = MeasureSet(self.def_file)
            self.disc_rates = DiscRates(self.def_file)
        else:
            self.read(file_name, description)

    def read(self, file_name, description=''):
        """Read and check input file.

        Parameters:
            file_name (str or list(str), optional): file name(s) or folder name
                containing the files to read
            description (str or list(str), optional): one description of the
                data or a description of each data file

        Raises:
            ValueError
        """
        self.exposures = Exposures()
        self.exposures.read(file_name, description)

        self.disc_rates = DiscRates()
        self.disc_rates.read(file_name, description)

        self.impact_funcs = ImpactFuncSet()
        self.impact_funcs.read(file_name, description)

        self.measures = MeasureSet()
        self.measures.read(file_name, description)

    def check(self):
        """Check instance attributes.

        Raises:
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
                LOGGER.error("Input value is not (sub)class of Exposures.")
                raise ValueError
        elif name == "impact_funcs":
            if not isinstance(value, ImpactFuncSet):
                LOGGER.error("Input value is not (sub)class of ImpactFuncSet.")
                raise ValueError
        elif name == "measures":
            if not isinstance(value, MeasureSet):
                LOGGER.error("Input value is not (sub)class of MeasureSet.")
                raise ValueError
        elif name == "disc_rates":
            if not isinstance(value, DiscRates):
                LOGGER.error("Input value is not (sub)class of DiscRates.")
                raise ValueError
        super().__setattr__(name, value)

    def __str__(self):
        return 'Exposures: \n' + self.exposures.tag.__str__() + \
                '\nDiscRates: \n' + self.disc_rates.tag.__str__() + \
                '\nImpactFuncSet: \n' + self.impact_funcs.tag.__str__() + \
                '\nMeasureSet: \n' + self.measures.tag.__str__()

    __repr__ = __str__
