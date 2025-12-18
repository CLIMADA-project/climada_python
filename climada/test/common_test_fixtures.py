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

A set of reusable objects for testing purpose.

The objective of this file is to provide minimalistic, understandable and consistent
default objects for unit and integration testing.

"""

import copy
from unittest import TestCase

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from climada.entity import Exposures, ImpactFunc, ImpactFuncSet, ImpfTropCyclone
from climada.hazard import Centroids, Hazard
from climada.trajectories import InterpolatedRiskTrajectory, StaticRiskTrajectory
from climada.trajectories.snapshot import Snapshot

# ---------------------------------------------------------------------------
# Coordinate system and metadata
# ---------------------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"

# ---------------------------------------------------------------------------
# Exposure attributes
# ---------------------------------------------------------------------------
EXP_DESC = "Test exposure dataset"
EXP_DESC_LATLON = "Test exposure dataset (lat/lon)"
EXPOSURE_REF_YEAR = 2020
EXPOSURE_VALUE_UNIT = "USD"
VALUES = np.array([0, 1000, 2000, 3000])
REGIONS = np.array(["A", "A", "B", "B"])
CATEGORIES = np.array([1, 1, 2, 1])

# Exposure coordinates
EXP_LONS = np.array([4, 4.5, 4, 4.5])
EXP_LATS = np.array([45, 45, 45.5, 45.5])

# ---------------------------------------------------------------------------
# Hazard definition
# ---------------------------------------------------------------------------
HAZARD_TYPE = "TEST_HAZARD_TYPE"
HAZARD_UNIT = "TEST_HAZARD_UNIT"

# Hazard centroid positions
HAZ_JITTER = 0.1  # To test centroid matching
HAZ_LONS = EXP_LONS + HAZ_JITTER
HAZ_LATS = EXP_LATS + HAZ_JITTER

# Hazard events
EVENT_IDS = np.array([1, 2, 3, 4])
EVENT_NAMES = ["ev1", "ev2", "ev3", "ev4"]
DATES = np.array([1, 2, 3, 4])

# Frequency are choosen so that they cumulate nicely
# to correspond to 100, 50, and 20y return periods (for impacts)
FREQUENCY = np.array([0.1, 0.03, 0.01, 0.01])
FREQUENCY_UNIT = "1/year"

# Hazard maximum intensity
# 100 to match 0 to 100% idea
# also in line with linear 1:1 impact function
# for easy mental calculus
HAZARD_MAX_INTENSITY = 100

# ---------------------------------------------------------------------------
# Impact function
# ---------------------------------------------------------------------------
IMPF_ID = 1
IMPF_NAME = "IMPF_1"

# ---------------------------------------------------------------------------
# Future years
# ---------------------------------------------------------------------------
EXPOSURE_FUTURE_YEAR = 2040


def reusable_minimal_exposures(
    values=VALUES,
    regions=REGIONS,
    group_id=None,
    lon=EXP_LONS,
    lat=EXP_LATS,
    crs=CRS_WGS84,
    desc=EXP_DESC,
    ref_year=EXPOSURE_REF_YEAR,
    value_unit=EXPOSURE_VALUE_UNIT,
    assign_impf=IMPF_ID,
    increase_value_factor=1,
) -> Exposures:
    data = gpd.GeoDataFrame(
        {
            "value": values * increase_value_factor,
            "region_id": regions,
            f"impf_{HAZARD_TYPE}": assign_impf,
            "geometry": [Point(lon, lat) for lon, lat in zip(lon, lat)],
        },
        crs=crs,
    )
    if group_id is not None:
        data["group_id"] = group_id
    return Exposures(
        data=data,
        description=desc,
        ref_year=ref_year,
        value_unit=value_unit,
    )


def reusable_intensity_mat(max_intensity=HAZARD_MAX_INTENSITY):
    # Choosen such that:
    # - 1st event has 0 intensity
    # - 2nd event has max intensity in first exposure point (defaulting to 0 value)
    # - 3rd event has 1/2* of max intensity in second centroid
    # - 4th event has 1/4* of max intensity everywhere
    # *: So that you can double intensity of the hazard and expect double impacts
    return csr_matrix(
        [
            [0, 0, 0, 0],
            [max_intensity, 0, 0, 0],
            [0, max_intensity / 2, 0, 0],
            [
                max_intensity / 4,
                max_intensity / 4,
                max_intensity / 4,
                max_intensity / 4,
            ],
        ]
    )


def reusable_minimal_hazard(
    haz_type=HAZARD_TYPE,
    units=HAZARD_UNIT,
    lat=HAZ_LATS,
    lon=HAZ_LONS,
    crs=CRS_WGS84,
    event_id=EVENT_IDS,
    event_name=EVENT_NAMES,
    date=DATES,
    frequency=FREQUENCY,
    frequency_unit=FREQUENCY_UNIT,
    intensity=None,
    intensity_factor=1,
) -> Hazard:
    intensity = reusable_intensity_mat() if intensity is None else intensity
    intensity *= intensity_factor
    return Hazard(
        haz_type=haz_type,
        units=units,
        centroids=Centroids(lat=lat, lon=lon, crs=crs),
        event_id=event_id,
        event_name=event_name,
        date=date,
        frequency=frequency,
        frequency_unit=frequency_unit,
        intensity=intensity,
    )


def reusable_minimal_impfset(
    hazard=None, name=IMPF_NAME, impf_id=IMPF_ID, max_intensity=HAZARD_MAX_INTENSITY
):
    hazard = reusable_minimal_hazard() if hazard is None else hazard
    return ImpactFuncSet(
        [
            ImpactFunc(
                haz_type=hazard.haz_type,
                intensity_unit=hazard.units,
                name=name,
                intensity=np.array([0, max_intensity / 2, max_intensity]),
                mdd=np.array([0, 0.5, 1]),
                paa=np.array([1, 1, 1]),
                id=impf_id,
            )
        ]
    )


def reusable_snapshot(
    hazard_intensity_increase_factor=1,
    exposure_value_increase_factor=1,
    date=EXPOSURE_REF_YEAR,
):
    exposures = reusable_minimal_exposures(
        increase_value_factor=exposure_value_increase_factor
    )
    hazard = reusable_minimal_hazard(intensity_factor=hazard_intensity_increase_factor)
    impfset = reusable_minimal_impfset()
    return Snapshot(exposure=exposures, hazard=hazard, impfset=impfset, date=date)
