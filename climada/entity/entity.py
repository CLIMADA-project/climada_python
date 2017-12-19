"""
=====================
entity
=====================

Define Entity Class.
"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Nov 10 10:00:03 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

from climada.entity.impact_funcs.source_excel  import ImpactFuncsExcel
from climada.entity.discounts.source_excel import DiscountsExcel
from climada.entity.measures.source_excel import MeasuresExcel
from climada.entity.exposures.source_excel import ExposuresExcel
from climada.util.config import entity_default

class Entity(object):
    """Contains the definition of one entity"""

    def __init__(self, exposures=None, impact_funcs=None, measures=None,
                 discounts=None):

        if exposures is not None:
            self.exposures = exposures
        else:
            self.exposures = ExposuresExcel(entity_default)

        if impact_funcs is not None:
            self.impact_funcs = impact_funcs
        else:
            self.impact_funcs = ImpactFuncsExcel(entity_default)

        if measures is not None:
            self.measures = measures
        else:
            self.measures = MeasuresExcel(entity_default)

        if discounts is not None:
            self.discounts = discounts
        else:
            self.discounts = DiscountsExcel(entity_default)

    #def interpolate(self, grid, conf):
    #    """ Interpolate the entity assets to a grid using the configuration"""
        #self.assets.interpolate(grid, conf)

    #def calc_future(self, conf):
    #    """ Compute the future assets following the configuration """

    def tags(self):
        """Obtain the entity tag"""
        return {self.exposures.tag, self.impact_funcs.tag,
                self.measures.tag, self.discounts.tag}
