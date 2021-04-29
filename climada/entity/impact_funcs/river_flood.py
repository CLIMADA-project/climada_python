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

Define impact functions for river flood .
"""

__all__ = ['ImpfRiverFlood', 'IFRiverFlood']

import logging
from deprecation import deprecated
import numpy as np
import pandas as pd

from climada.util.constants import RIVER_FLOOD_REGIONS_CSV
from .base import ImpactFunc
from .impact_func_set import ImpactFuncSet

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


class ImpfRiverFlood(ImpactFunc):
    """Impact functions for tropical cyclones."""

    def __init__(self):
        ImpactFunc.__init__(self)
        self.haz_type = 'RF'
        self.intensity_unit = 'm'
        self.continent = ''

    def set_RF_Impf_Africa(self):
        self.id = 1
        self.name = "Flood Africa JRC Residential noPAA"
        self.continent = 'Africa'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])
        self.mdd = np.array([0.0000, 0.2199, 0.3782, 0.5306, 0.6356, 0.8169,
                             0.9034, 0.9572, 1.0000, 1.0000])

        self.mdr = np.array([0.0000, 0.2199, 0.3782, 0.5306, 0.6356, 0.8169,
                             0.9034, 0.9572, 1.0000, 1.0000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_Impf_Asia(self):
        self.id = 2
        self.name = "Flood Asia JRC Residential noPAA"
        self.continent = 'Asia'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207, 0.8695,
                             0.9315, 0.9836, 1.0000, 1.0000])

        self.mdr = np.array([0.000, 0.3266, 0.4941, 0.6166, 0.7207, 0.8695,
                             0.9315, 0.9836, 1.0000, 1.0000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_Impf_Europe(self):
        self.id = 3
        self.name = "Flood Europe JRC Residential noPAA"
        self.continent = 'Europe'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.00, 0.25, 0.40, 0.50, 0.60, 0.75, 0.85, 0.95,
                             1.00, 1.00])

        self.mdr = np.array([0.000, 0.250, 0.400, 0.500, 0.600, 0.750, 0.850,
                             0.950, 1.000, 1.000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_Impf_NorthAmerica(self):
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

    def set_RF_Impf_Oceania(self):
        self.id = 5
        self.name = "Flood Oceania JRC Residential noPAA"
        self.continent = 'Oceania'
        self.intensity = np.array([0., 0.5, 1., 1.5, 2., 3., 4., 5., 6., 12.])

        self.mdd = np.array([0.00, 0.48, 0.64, 0.71, 0.79, 0.93, 0.97, 0.98,
                             1.00, 1.00])

        self.mdr = np.array([0.000, 0.480, 0.640, 0.710, 0.790, 0.930, 0.970,
                             0.980, 1.000, 1.000])

        self.paa = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def set_RF_Impf_SouthAmerica(self):
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

    impf_set = ImpactFuncSet()

    impf_africa = ImpfRiverFlood()
    impf_africa.set_RF_Impf_Africa()
    impf_set.append(impf_africa)

    impf_asia = ImpfRiverFlood()
    impf_asia.set_RF_Impf_Asia()
    impf_set.append(impf_asia)

    impf_europe = ImpfRiverFlood()
    impf_europe.set_RF_Impf_Europe()
    impf_set.append(impf_europe)

    impf_na = ImpfRiverFlood()
    impf_na.set_RF_Impf_NorthAmerica()
    impf_set.append(impf_na)

    impf_oceania = ImpfRiverFlood()
    impf_oceania.set_RF_Impf_Oceania()
    impf_set.append(impf_oceania)

    impf_sa = ImpfRiverFlood()
    impf_sa.set_RF_Impf_SouthAmerica()
    impf_set.append(impf_sa)

    return impf_set


def assign_Impf_simple(exposure, country):
    info = pd.read_csv(RIVER_FLOOD_REGIONS_CSV)
    impf_id = info.loc[info['ISO'] == country, 'Impf_RF'].values[0]
    exposure['Impf_RF'] = impf_id


@deprecated(details="The class name IFRiverFlood is deprecated and won't be supported in a future "
                   +"version. Use ImpfRiverFlood instead")
class IFRiverFlood(ImpfRiverFlood):
    """Is ImpfRiverFlood now"""
