"""
Define Entity Class.
"""

__all__ = ['Entity']

import os

from climada.entity.loader import Loader
from climada.entity.impact_funcs.base  import ImpactFuncs
from climada.entity.impact_funcs.source_excel  import ImpactFuncsExcel
from climada.entity.impact_funcs.source_mat  import ImpactFuncsMat
from climada.entity.disc_rates.base import DiscRates
from climada.entity.disc_rates.source_excel import DiscRatesExcel
from climada.entity.disc_rates.source_mat import DiscRatesMat
from climada.entity.measures.base import Measures
from climada.entity.measures.source_excel import MeasuresExcel
from climada.entity.measures.source_mat import MeasuresMat
from climada.entity.exposures.base import Exposures
from climada.entity.exposures.source_excel import ExposuresExcel
from climada.entity.exposures.source_mat import ExposuresMat
from climada.util.config import ENT_DEF_XLS

class Entity(Loader):
    """Collects exposures, impact functions, measures and discount rates.

    Attributes
    ----------
        exposures (Exposures): exposures
        impact_funcs (ImpactFucs): vulnerability functions
        measures (Measures): measures
        disc_rates (DiscRates): discount rates
        def_file (str): name of the xls file used as default source data
    """

    def_file = ENT_DEF_XLS

    def __init__(self, file_name=None, description=None):
        """Fill values from file. Default file used when no file provided.

        Parameters
        ----------
            file_name (str, optional): name of the source file with supported
                format (xls, xlsx and mat)
            description (str, optional): description of the source data

        Raises
        ------
            ValueError

        Examples
        ---------
            >>> Entity()
            Builds an Entity with the values obtained from ENT_DEF_XLS file.
            >>> Entity(filename)
            Builds an Entity with the values obtained from filename file.
            >>> Entity(impact_funcs=myimpact_funcs, measures=mymeasures)
            Builds an Entity with exposures and discount rates from
            ENT_DEF_XLS file, and the given impact functions and measures.
        """
        if file_name is None:
            self.exposures = ExposuresExcel(self.def_file)
            self.impact_funcs = ImpactFuncsExcel(self.def_file)
            self.measures = MeasuresExcel(self.def_file)
            self.disc_rates = DiscRatesExcel(self.def_file)
        else:
            self.load(file_name, description)

    def read(self, file_name, description=None):
        """Override read Loader method."""
        # Call readers depending on file extension
        extension = os.path.splitext(file_name)[1]
        if extension == '.mat':
            self.exposures = ExposuresMat()
            self.exposures.read(file_name, description)

            self.impact_funcs = ImpactFuncsMat()
            self.impact_funcs.read(file_name, description)

            self.disc_rates = DiscRatesMat()
            self.disc_rates.read(file_name, description)

            self.measures = MeasuresMat()
            self.measures.read(file_name, description)

        elif (extension == '.xlsx') or (extension == '.xls'):
            self.exposures = ExposuresExcel()
            self.exposures.read(file_name, description)

            self.impact_funcs = ImpactFuncsExcel()
            self.impact_funcs.read(file_name, description)

            self.disc_rates = DiscRatesExcel()
            self.disc_rates.read(file_name, description)

            self.measures = MeasuresExcel()
            self.measures.read(file_name, description)

        else:
            raise TypeError('Input file extension not supported: %s.' % \
                            extension)

    def check(self):
        """ Override Loader check."""
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
            if not isinstance(value, ImpactFuncs):
                raise ValueError("Input value is not (sub)class of \
                                 ImpactFuncs.")
        elif name == "measures":
            if not isinstance(value, Measures):
                raise ValueError("Input value is not (sub)class of Measures.")
        elif name == "disc_rates":
            if not isinstance(value, DiscRates):
                raise ValueError("Input value is not (sub)class of DiscRates.")
        super().__setattr__(name, value)
