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

Define constants for trajectories module.
"""

DEFAULT_TIME_RESOLUTION = "Y"
DATE_COL_NAME = "date"
PERIOD_COL_NAME = "period"
GROUP_COL_NAME = "group"
GROUP_ID_COL_NAME = "group_id"
MEASURE_COL_NAME = "measure"
NO_MEASURE_VALUE = "no_measure"
METRIC_COL_NAME = "metric"
UNIT_COL_NAME = "unit"
RISK_COL_NAME = "risk"
COORD_ID_COL_NAME = "coord_id"

DEFAULT_PERIOD_INDEX_NAME = "date"

DEFAULT_RP = [20, 50, 100]
"""Default return periods to use when computing return period impact estimates."""

DEFAULT_ALLGROUP_NAME = "All"
"""Default string to use to define the exposure subgroup containing all exposure points."""

EAI_METRIC_NAME = "eai"
AAI_METRIC_NAME = "aai"
AAI_PER_GROUP_METRIC_NAME = "aai_per_group"
CONTRIBUTIONS_METRIC_NAME = "risk_contributions"
RETURN_PERIOD_METRIC_NAME = "return_periods"
RP_VALUE_PREFIX = "rp"


CONTRIBUTION_BASE_RISK_NAME = "base risk"
CONTRIBUTION_TOTAL_RISK_NAME = "total risk"
CONTRIBUTION_EXPOSURE_NAME = "exposure contribution"
CONTRIBUTION_HAZARD_NAME = "hazard contribution"
CONTRIBUTION_VULNERABILITY_NAME = "vulnerability contribution"
CONTRIBUTION_INTERACTION_TERM_NAME = "interaction contribution"
