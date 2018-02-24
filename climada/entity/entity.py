"""
Define Entity Class.
"""

__all__ = ['Entity']

from climada.entity.impact_funcs.base  import ImpactFuncs
from climada.entity.disc_rates.base import DiscRates
from climada.entity.measures.base import Measures
from climada.entity.exposures.base import Exposures
from climada.util.config import ENT_DEF_XLS

class Entity(object):
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
            self.exposures = Exposures(self.def_file)
            self.impact_funcs = ImpactFuncs(self.def_file)
            self.measures = Measures(self.def_file)
            self.disc_rates = DiscRates(self.def_file)
        else:
            self.load(file_name, description)

    def read(self, file_name, description=None):
        """Read input file.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError
        """
        self.exposures = Exposures()
        self.exposures.read(file_name, description)

        self.disc_rates = DiscRates()
        self.disc_rates.read(file_name, description)

        self.impact_funcs = ImpactFuncs()
        self.impact_funcs.read(file_name, description)

        self.measures = Measures()
        self.measures.read(file_name, description)


    def load(self, file_name, description=None):
        """Read, check and save as pkl, if output file name.

        Parameters
        ----------
            file_name (str): name of the source file
            description (str, optional): description of the source data

        Raises
        ------
            ValueError
        """
        self.exposures = Exposures()
        self.exposures.load(file_name, description)

        self.disc_rates = DiscRates()
        self.disc_rates.load(file_name, description)

        self.impact_funcs = ImpactFuncs()
        self.impact_funcs.load(file_name, description)

        self.measures = Measures()
        self.measures.load(file_name, description)

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
