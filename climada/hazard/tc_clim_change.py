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

Define climate change scenarios for tropical cycones.
"""

import os
import numpy as np
import pandas as pd

from climada.util.constants import SYSTEM_DIR

TOT_RADIATIVE_FORCE = os.path.join(SYSTEM_DIR, 'rcp_db.xls')
"""© RCP Database (Version 2.0.5) http://www.iiasa.ac.at/web-apps/tnt/RcpDb.
generated: 2018-07-04 10:47:59."""

def get_knutson_criterion():
    """Fill changes in TCs according to Knutson et al. 2015 Global projections
    of intense tropical cyclone activity for the late twenty-first century from
    dynamical downscaling of CMIP5/RCP4.5 scenarios.

    Returns:
        list(dict) with items 'criteria' (dict with variable_name and list(possible values)),
        'year' (int), 'change' (float), 'variable' (str), 'function' (np function)
    """
    criterion = list()
    # NA
    tmp_chg = {'criteria': {'basin': ['NA'], 'category': [1, 2, 3, 4, 5]},
               'year': 2100, 'change': 1.045, 'variable': 'intensity', 'function': np.multiply}
    criterion.append(tmp_chg)

    # EP
    tmp_chg = {'criteria': {'basin': ['EP'], 'category': [0]},
               'year': 2100, 'change': 1.163, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['EP'], 'category': [1, 2]},
               'year': 2100, 'change': 1.193, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['EP'], 'category': [3]},
               'year': 2100, 'change': 1.837, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['EP'], 'category': [4, 5]},
               'year': 2100, 'change': 3.375, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)

    tmp_chg = {'criteria': {'basin': ['EP'], 'category': [0]},
               'year': 2100, 'change': 1.082, 'variable': 'intensity', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['EP'], 'category': [1, 2, 3, 4, 5]},
               'year': 2100, 'change': 1.078, 'variable': 'intensity', 'function': np.multiply}
    criterion.append(tmp_chg)

    # WP
    tmp_chg = {'criteria': {'basin': ['WP'], 'category': [0]},
               'year': 2100, 'change': 1 - 0.345, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['WP'], 'category': [1, 2]},
               'year': 2100, 'change': 1 - 0.316, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['WP'], 'category': [3, 4, 5]},
               'year': 2100, 'change': 1 - 0.169, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)

    tmp_chg = {'criteria': {'basin': ['WP'], 'category': [0]},
               'year': 2100, 'change': 1.074, 'variable': 'intensity', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['WP'], 'category': [1, 2, 3, 4, 5]},
               'year': 2100, 'change': 1.055, 'variable': 'intensity', 'function': np.multiply}
    criterion.append(tmp_chg)

    # NI
    tmp_chg = {'criteria': {'basin': ['NI'], 'category': [1, 2, 3, 4, 5]},
               'year': 2100, 'change': 1.256, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)

    # SI
    tmp_chg = {'criteria': {'basin': ['SI'], 'category': [0]},
               'year': 2100, 'change': 1 - 0.261, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['SI'], 'category': [1, 2, 3, 4, 5]},
               'year': 2100, 'change': 1 - 0.284, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)

    tmp_chg = {'criteria': {'basin': ['SI'], 'category': [1, 2, 3, 4, 5]},
               'year': 2100, 'change': 1.033, 'variable': 'intensity', 'function': np.multiply}
    criterion.append(tmp_chg)

    # SP
    tmp_chg = {'criteria': {'basin': ['SP'], 'category': [0]},
               'year': 2100, 'change': 1 - 0.366, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['SP'], 'category': [1, 2]},
               'year': 2100, 'change': 1 - 0.406, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['SP'], 'category': [3]},
               'year': 2100, 'change': 1 - 0.506, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)
    tmp_chg = {'criteria': {'basin': ['SP'], 'category': [4, 5]},
               'year': 2100, 'change': 1 - 0.583, 'variable': 'frequency', 'function': np.multiply}
    criterion.append(tmp_chg)

    return criterion

def calc_scale_knutson(ref_year=2050, rcp_scenario=45):
    """Comparison 2081-2100 (i.e., late twenty-first century) and 2001-20
    (i.e., present day). Late twenty-first century effects on intensity and
    frequency per Saffir-Simpson-category and ocean basin is scaled to target
    year and target RCP proportional to total radiative forcing of the respective
    RCP and year.

    Parameters:
        ref_year (int): year between 2000 ad 2100. Default: 2050
        rcp_scenario (int):  26 for RCP 2.6, 45 for RCP 4.5 (default),
            60 for RCP 6.0 and 85 for RCP 8.5.

    Returns:
        float
    """
    # Parameters used in Knutson et al 2015
    base_knu = np.arange(2001, 2021)
    end_knu = np.arange(2081, 2101)
    rcp_knu = 45

    # radiative forcings for each RCP scenario
    rad_force = pd.read_excel(TOT_RADIATIVE_FORCE)
    years = np.array([year for year in rad_force.columns if isinstance(year, int)])
    rad_rcp = np.array([int(float(sce[sce.index('.') - 1:sce.index('.') + 2]) * 10)
                        for sce in rad_force.Scenario if isinstance(sce, str)])

    # mean values for Knutson values
    rf_vals = np.argwhere(rad_rcp == rcp_knu).reshape(-1)[0]
    rf_vals = np.array([rad_force.iloc[rf_vals][year] for year in years])
    rf_base = np.nanmean(np.interp(base_knu, years, rf_vals))
    rf_end = np.nanmean(np.interp(end_knu, years, rf_vals))

    # scale factor for ref_year and rcp_scenario
    rf_vals = np.argwhere(rad_rcp == rcp_scenario).reshape(-1)[0]
    rf_vals = np.array([rad_force.iloc[rf_vals][year] for year in years])
    rf_sel = np.interp(ref_year, years, rf_vals)
    return max((rf_sel - rf_base) / (rf_end - rf_base), 0)
