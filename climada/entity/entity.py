"""
Define Entity Class.
"""

import pickle

from climada.entity.impact_funcs.base  import ImpactFuncs
from climada.entity.impact_funcs.source_excel  import ImpactFuncsExcel
from climada.entity.discounts.base import Discounts
from climada.entity.discounts.source_excel import DiscountsExcel
from climada.entity.measures.base import Measures
from climada.entity.measures.source_excel import MeasuresExcel
from climada.entity.exposures.base import Exposures
from climada.entity.exposures.source_excel import ExposuresExcel
from climada.util.config import ENT_DEF_XLS

class Entity(object):
    """Collects exposures, impact functions, measures and discount rates.

    Attributes
    ----------
        exposures (subclass Exposures): exposures
        impact_funcs (subclass ImpactFucs): vulnerability functions
        measures (subclass Measures): measures
        discounts (subclass Discounts): discount rates
        def_file (str): name of the xls file used as default source data
    """

    def_file = ENT_DEF_XLS

    def __init__(self, exposures=None, impact_funcs=None, measures=None,
                 discounts=None):
        """Initialization. Default values are set for the inputs not provided.

        Parameters
        ----------
            exposures (subclass Exposures, optional)
            impact_funcs (subclass ImpactFucs, optional)
            measures (subclass Measures, optional)
            discounts (subclass Discounts, optional)

        Raises
        ------
            ValueError

        Examples
        ---------
            >>> Entity()
            Builds an Entity with the values obtained from ENT_DEF_XLS file.
            >>> Entity(exposures=myexposures)
            Builds an Entity with impact function, measures and discount rates
            from ENT_DEF_XLS file, and the given exposures.
            >>> Entity(impact_funcs=myimpact_funcs, measures=mymeasures)
            Builds an Entity with exposures and discount rates from
            ENT_DEF_XLS file, and the given impact functions and measures.
        """
        if exposures is not None:
            self.exposures = exposures
        else:
            self.exposures = ExposuresExcel(self.def_file)

        if impact_funcs is not None:
            self.impact_funcs = impact_funcs
        else:
            self.impact_funcs = ImpactFuncsExcel(self.def_file)

        if measures is not None:
            self.measures = measures
        else:
            self.measures = MeasuresExcel(self.def_file)

        if discounts is not None:
            self.discounts = discounts
        else:
            self.discounts = DiscountsExcel(self.def_file)

    def tags(self):
        """Return entity tag constructed from its attributes tags."""
        return {self._exposures.tag, self._impact_funcs.tag,
                self._measures.tag, self._discounts.tag}

    def save(self, out_file_name):
        """Save as pkl.

        Parameters
        ----------
            out_file_name (str): output file name to save as pkl
        """
        if out_file_name is not None:
            with open(out_file_name, 'wb') as file:
                pickle.dump(self, file)

    def is_entity(self):
        """ Checks if the attributes contain consistent data.

        Raises
        ------
            ValueError
        """
        self.discounts.is_discounts()
        self.exposures.is_exposures()
        self.impact_funcs.is_impactFuncs()
        self.measures.is_measures()

    @property
    def exposures(self):
        return self._exposures

    @exposures.setter
    def exposures(self, value):
        if not isinstance(value, Exposures):
            raise ValueError("Input value is not subclass of Exposures ABC.")
        self._exposures = value

    @property
    def impact_funcs(self):
        return self._impact_funcs

    @impact_funcs.setter
    def impact_funcs(self, value):
        if not isinstance(value, ImpactFuncs):
            raise ValueError("Input value is not subclass of ImpactFuncs \
                             ABC.")
        self._impact_funcs = value

    @property
    def measures(self):
        return self._measures

    @measures.setter
    def measures(self, value):
        if not isinstance(value, Measures):
            raise ValueError("Input value is not subclass of Measures ABC.")
        self._measures = value

    @property
    def discounts(self):
        return self._discounts

    @discounts.setter
    def discounts(self, value):
        if not isinstance(value, Discounts):
            raise ValueError("Input value is not subclass of Discounts ABC.")
        self._discounts = value
