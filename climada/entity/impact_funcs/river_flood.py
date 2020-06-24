"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define impact functions for river flood .
"""

__all__ = ['IFRiverFlood']
import numpy as np
import logging
import pandas as pd
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity import ImpactFuncSet
from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
LOGGER = logging.getLogger(__name__)

DEF_VAR_EXCEL = {'sheet_name': 'damagefunctions',
                 'col_name': {'func_id': 'DamageFunID',
                              'inten': 'Intensity',
                              'mdd': 'MDD',
                              'paa': 'PAA',
                              'mdr': 'MDR',
                              'name': 'name',
                              'peril': 'peril_ID',
                              'unit': 'Intensity_unit'
                              }
                 }


class IFRiverFlood(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'RF'
        self.intensity_unit = 'm'
        self.continent = ''

    def set_RF_IF_Africa(self):
        self.id = 1
        self.name = "Flood Africa JRC Residential noPAA"
        self.continent = 'Africa'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
        self.mdd = np.array([0.0000, 0.2199, 0.3782, 0.5306, 0.6356, 0.8169,
                             0.9034, 0.9572, 1.0000, 1.0000])

        self.mdr = np.array([0.0000, 0.2199, 0.3782, 0.5306, 0.6356, 0.8169,
                             0.9034, 0.9572, 1.0000, 1.0000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_IF_Asia(self):
        self.id = 2
        self.name = "Flood Asia JRC Residential noPAA"
        self.continent = 'Asia'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207, 0.8695,
                             0.9315, 0.9836, 1.0000, 1.0000])

        self.mdr = np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207, 0.8695,
                             0.9315, 0.9836, 1.0000, 1.0000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_IF_Europe(self):
        self.id = 3
        self.name = "Flood Europe JRC Residential noPAA"
        self.continent = 'Europe'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.00, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95,
                             1.00, 1.00])

        self.mdr = np.array([0.000, 0.250, 0.400, 0.500, 0.600, 0.750, 0.850,
                             0.950, 1.000, 1.000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_IF_NorthAmerica(self):
        self.id = 4
        self.name = "Flood North America JRC Residential noPAA"
        self.continent = 'NorthAmerica'
        self.intensity = np.array([0., 0.1, 0.5, 1., 1.5, 2., 3., 4., 5.,
                                   6., 12.])

        self.mdd = np.array([0.0000, 0.2018, 0.4433, 0.5828, 0.6825, 0.7840,
                             0.8543, 0.9237, 0.9585, 1.0000, 1.0000])

        self.mdr = np.array([0.0000, 0.2018, 0.4433, 0.5828, 0.6825, 0.7840,
                             0.8543, 0.9237, 0.9585, 1.0000, 1.0000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_IF_Oceania(self):
        self.id = 5
        self.name = "Flood Oceania JRC Residential noPAA"
        self.continent = 'Oceania'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.00, 0.48, 0.64, 0.71, 0.79, 0.93, 0.97, 0.98,
                             1.00, 1.00])

        self.mdr = np.array([0.000, 0.480, 0.640, 0.710, 0.790, 0.930, 0.970,
                             0.980, 1.000, 1.000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_IF_SouthAmerica(self):
        self.id = 6
        self.name = "Flood South America JRC Residential noPAA"
        self.continent = 'SouthAmerica'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.0000, 0.4908, 0.7112, 0.8420, 0.9494,
                             0.9836, 1.0000, 1.0000, 1.0000, 1.0000])

        self.mdr = np.array([0.0000, 0.4908, 0.7112, 0.8420, 0.9494, 0.9836,
                             1.0000, 1.0000, 1.0000, 1.0000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

def flood_imp_func_set():
    """Builds impact function set for river flood, using standard files"""

    if_set = ImpactFuncSet()

    if_africa = IFRiverFlood()
    if_africa.set_RF_IF_Africa()
    if_set.append(if_africa)

    if_asia = IFRiverFlood()
    if_asia.set_RF_IF_Asia()
    if_set.append(if_asia)

    if_europe = IFRiverFlood()
    if_europe.set_RF_IF_Europe()
    if_set.append(if_europe)

    if_na = IFRiverFlood()
    if_na.set_RF_IF_NorthAmerica()
    if_set.append(if_na)

    if_oceania = IFRiverFlood()
    if_oceania.set_RF_IF_Oceania()
    if_set.append(if_oceania)

    if_sa = IFRiverFlood()
    if_sa.set_RF_IF_SouthAmerica()
    if_set.append(if_sa)

    return if_set


def assign_if_simple(exposure, country):
    info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    if_id = info.loc[info['ISO'] == country, 'if_RF'].values[0]
    exposure['if_RF'] = if_id
