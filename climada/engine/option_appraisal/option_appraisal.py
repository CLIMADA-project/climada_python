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

"""

from typing import List

from climada.engine.impact_trajectories import (
    Snapshot,
    SnapshotsCollection,
    interpolate_years,
)
from climada.entity import Measure


def interp_exp(year, exp1, year1, exp2, year2):
    exp = exp1.copy()
    years_mult = interpolate_years(year1, year2)
    exp.gdf.value = (
        years_mult[year - year1] * exp1.gdf.value
        + (1 - years_mult[year - year1]) * exp2.gdf.value
    )
    return exp


class MeasureTrajectory:
    def __init__(self, measures: List[Measure], snapshots: SnapshotsCollection) -> None:
        self.measures = measures
        self.snapshots = snapshots

    def snapshots_lookup(self, measure):
        pass

    def apply_measure(self, measure, snapshot_start: Snapshot, snapshot_end: Snapshot):
        if measure.start > snapshot_start.year or measure.end < snapshot_end.year:
            if measure.start > snapshot_start.year:
                exp_start = interp_exp(
                    measure.start,
                    snapshot_start.exposure,
                    snapshot_start.year,
                    snapshot_end.exposure,
                    snapshot_end.year,
                )
                flag_start = True
            else:
                exp_start = snapshot_start.exposure
                flag_start = False

            if measure.end < snapshot_end.year:
                exp_end = interp_exp(
                    measure.end,
                    snapshot_start.exposure,
                    snapshot_start.year,
                    snapshot_end.exposure,
                    snapshot_end.year,
                )
                flag_end = True
            else:
                exp_end = snapshot_end.exposure
                flag_end = False
        else:
            exp_start, exp_end = snapshot_start.exposure, snapshot_end.exposure

        m_exp_start, m_impfset_start, m_haz_start = measure.apply(
            exp_start, snapshot_start.impfset, snapshot_start.hazard
        )
        exp_end, impfset_end, haz_end = measure.apply(
            exp_end, snapshot_end.impfset, snapshot_end.hazard
        )

        measure_snapshot_start = Snapshot(
            m_exp_start, m_haz_start, m_impfset_start, snapshot_start.year
        )
        measure_snapshot_end = Snapshot(
            exp_end, haz_end, impfset_end, snapshot_end.year
        )

        return measure_snapshot_start, measure_snapshot_end
