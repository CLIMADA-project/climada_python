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

Impact function calibration functionalities:
    Optimization and manual calibration
"""

import numpy as np
import pandas as pd
import datetime as dt

from climada.engine import Impact
from climada.entity import ImpactFuncSet

def calib_instance(hazard, exposure, impact_func, df_out=pd.DataFrame(),
                   yearly_impact=False):

    """ calculate one impact instance for the calibration algorithm and write 
        to given DataFrame

        Parameters:
            hazard: hazard set instance
            exposure: exposure set instance
            impact_func: impact function instance
            
        Optional Parameters:
            df_out: Output DataFrame with headers of columns defined and optionally with
                first row (index=0) defined with values. If columns "impact", 
                "event_id", or "year" are not included, they are created here.
                Data like reported impacts or impact function parameters can be
                given here; values are preserved.
            yearly_impact (boolean): if set True, impact is returned per year, 
                not per event

        Returns:
            df_out: DataFrame with modelled impact written to rows for each year
                or event.
    """
    IFS = ImpactFuncSet()
    IFS.append(impact_func)
    impacts = Impact()
    impacts.calc(exposure, IFS, hazard)
    if yearly_impact: # impact per year
        IYS = impacts.calc_impact_year_set(all_years=True)
        # Loop over whole year range:
        for cnt_, year in enumerate(np.sort(list((IYS.keys())))):
            if cnt_ > 0:
                df_out.loc[cnt_] = df_out.loc[0] # copy info from first row
            if year in IYS:
                df_out.loc[cnt_, 'impact'] = IYS[year]
            else:
                df_out.loc[cnt_, 'impact'] = 0
            df_out.loc[cnt_, 'year'] = year

    else: # impact per event
        for cnt_, impact in enumerate(impacts.at_event):
            if cnt_ > 0:
                df_out.loc[cnt_] = df_out.loc[0] # copy info from first row
            df_out.loc[cnt_, 'impact'] = impact
            df_out.loc[cnt_, 'event_id'] = int(impacts.event_id[cnt_])
            df_out.loc[cnt_, 'event_name'] = impacts.event_name[cnt_]
            df_out.loc[cnt_, 'year'] = \
                dt.datetime.fromordinal(impacts.date[cnt_]).year
    return df_out
