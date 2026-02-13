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

A set of reusable fixtures for testing purpose.

The objective of this file is to provide minimalistic, understandable and consistent
default objects for unit and integration testing.

Values are chosen such that:
    - Exposure value of the first points is 0. (First location should always have 0 impacts)
    - Category / Group id of all points is 1, except for third point, valued at 2000 (Impacts on that category are always a share of 2000)
    - Hazard centroids are the exposure centroids shifted by `HAZARD_JITTER` on both lon and lat.
    - There are 4 events, with frequencies == 0.03, 0.01, 0.006, 0.004, 0,
      such that impacts for RP250, 100 and 50 and 20 are at_event,
      (freq sorted cumulate to 1/250, 1/100, 1/50 and 1/20).
    - Hazard intensity is:
        * Event 1: zero everywhere (always no impact)
        * Event 2: max intensity at first centroid (also always no impact (first centroid is 0))
        * Event 3: half max intensity at second centroid (impact == half second centroid)
        * Event 4: quarter max intensity everywhere (impact == 1/4 total value)
        * Event 5: max intensity everywhere (but zero frequency)
      With max intensity set at 100
    - Impact function is the "identity function", x intensity is x% damages
    - Impact values should be round.

"""

import geopandas as gpd
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import Point

from climada.entity import Exposures, ImpactFunc, ImpactFuncSet
from climada.hazard import Centroids, Hazard

# ---------------------------------------------------------------------------
# Coordinate system and metadata
# ---------------------------------------------------------------------------
CRS_WGS84 = "EPSG:4326"

# ---------------------------------------------------------------------------
# Exposure attributes
# ---------------------------------------------------------------------------
EXP_DESC = "Test exposure dataset"
EXPOSURE_REF_YEAR = 2020
EXPOSURE_VALUE_UNIT = "USD"
VALUES = np.array([0, 1000, 2000, 3000, 4000, 5000])
CATEGORIES = np.array([1, 1, 2, 1, 1, 3])

# Exposure coordinates
EXP_LONS = np.array([4, 4.25, 4.5, 4, 4.25, 4.5])
EXP_LATS = np.array([45, 45, 45, 45.25, 45.25, 45.25])

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
EVENT_IDS = np.array([1, 2, 3, 4, 5])
EVENT_NAMES = ["ev1", "ev2", "ev3", "ev4", "ev5"]
DATES = np.array([1, 2, 3, 4, 5])

# Frequency are choosen so that they cumulate nicely
# to correspond to 250, 100, 50, and 20y return periods (for impacts)
FREQUENCY = np.array([0.03, 0.01, 0.006, 0.004, 0.0])
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

# Sanity checks
for const in [VALUES, CATEGORIES, EXP_LONS, EXP_LATS]:
    assert len(const) == len(
        VALUES
    ), "VALUES, REGIONS, CATEGORIES, EXP_LONS, EXP_LATS should all have the same lengths."

for const in [EVENT_IDS, EVENT_NAMES, DATES, FREQUENCY]:
    assert len(const) == len(
        EVENT_IDS
    ), "EVENT_IDS, EVENT_NAMES, DATES, FREQUENCY should all have the same lengths."


@pytest.fixture(scope="session")
def exposure_values():
    return VALUES.copy()


@pytest.fixture(scope="session")
def categories():
    return CATEGORIES.copy()


@pytest.fixture(scope="session")
def exposure_geometry():
    return [Point(lon, lat) for lon, lat in zip(EXP_LONS, EXP_LATS)]


@pytest.fixture(scope="session")
def exposures_factory(
    exposure_values,
    exposure_geometry,
):
    def _make_exposures(
        value_factor=1.0,
        ref_year=EXPOSURE_REF_YEAR,
        hazard_type=HAZARD_TYPE,
        group_id=None,
    ):
        gdf = gpd.GeoDataFrame(
            {
                "value": exposure_values * value_factor,
                f"impf_{hazard_type}": IMPF_ID,
                "geometry": exposure_geometry,
            },
            crs=CRS_WGS84,
        )
        if group_id is not None:
            gdf["group_id"] = group_id

        return Exposures(
            data=gdf,
            description=EXP_DESC,
            ref_year=ref_year,
            value_unit=EXPOSURE_VALUE_UNIT,
        )

    return _make_exposures


@pytest.fixture(scope="session")
def exposures(exposures_factory):
    return exposures_factory()


@pytest.fixture(scope="session")
def hazard_frequency_factory():
    base = FREQUENCY

    def _make_frequency(scale=1.0):
        return base * scale

    return _make_frequency


@pytest.fixture(scope="session")
def hazard_frequency():
    return hazard_frequency_factory()


@pytest.fixture(scope="session")
def hazard_intensity_factory():
    """
    Intensity matrix designed for analytical expectations:
    - Event 1: zero
    - Event 2: max intensity at first centroid
    - Event 3: half max intensity at second centroid
    - Event 4: quarter max intensity everywhere
    """
    base = csr_matrix(
        [
            [0, 0, 0, 0, 0, 0],
            [HAZARD_MAX_INTENSITY, 0, 0, 0, 0, 0],
            [0, HAZARD_MAX_INTENSITY / 2, 0, 0, 0, 0],
            [
                HAZARD_MAX_INTENSITY / 4,
                HAZARD_MAX_INTENSITY / 4,
                HAZARD_MAX_INTENSITY / 4,
                HAZARD_MAX_INTENSITY / 4,
                HAZARD_MAX_INTENSITY / 4,
                HAZARD_MAX_INTENSITY / 4,
            ],
            [
                HAZARD_MAX_INTENSITY,
                HAZARD_MAX_INTENSITY,
                HAZARD_MAX_INTENSITY,
                HAZARD_MAX_INTENSITY,
                HAZARD_MAX_INTENSITY,
                HAZARD_MAX_INTENSITY,
            ],
        ]
    )

    def _make_intensity(scale=1.0):
        return base * scale

    return _make_intensity


@pytest.fixture(scope="session")
def hazard_intensity_matrix(hazard_intensity_factory):
    return hazard_intensity_factory()


@pytest.fixture(scope="session")
def centroids():
    return Centroids(lat=HAZ_LATS, lon=HAZ_LONS, crs=CRS_WGS84)


@pytest.fixture(scope="session")
def hazard_factory(
    hazard_intensity_factory,
    hazard_frequency_factory,
    centroids,
):
    def _make_hazard(
        intensity_scale=1.0,
        frequency_scale=1.0,
        hazard_type=HAZARD_TYPE,
        hazard_unit=HAZARD_UNIT,
    ):
        return Hazard(
            haz_type=hazard_type,
            units=hazard_unit,
            centroids=centroids,
            event_id=EVENT_IDS,
            event_name=EVENT_NAMES,
            date=DATES,
            frequency=hazard_frequency_factory(scale=frequency_scale),
            frequency_unit=FREQUENCY_UNIT,
            intensity=hazard_intensity_factory(scale=intensity_scale),
        )

    return _make_hazard


@pytest.fixture(scope="session")
def hazard(hazard_factory):
    return hazard_factory()


@pytest.fixture(scope="session")
def impf_factory():
    def _make_impf(
        paa_scale=1.0,
        max_intensity=HAZARD_MAX_INTENSITY,
        hazard_type=HAZARD_TYPE,
        hazard_unit=HAZARD_UNIT,
        impf_id=IMPF_ID,
    ):
        return ImpactFunc(
            haz_type=hazard_type,
            intensity_unit=hazard_unit,
            name=IMPF_NAME,
            intensity=np.array([0, max_intensity / 2, max_intensity]),
            mdd=np.array([0, 0.5, 1]),
            paa=np.array([1, 1, 1]) * paa_scale,
            id=impf_id,
        )

    return _make_impf


@pytest.fixture(scope="session")
def linear_impact_function(impf_factory):
    return impf_factory()


@pytest.fixture(scope="session")
def impfset_factory(impf_factory):
    def _make_impfset(
        paa_scale=1.0,
        max_intensity=HAZARD_MAX_INTENSITY,
        hazard_type=HAZARD_TYPE,
        hazard_unit=HAZARD_UNIT,
        impf_id=IMPF_ID,
    ):
        return ImpactFuncSet(
            [impf_factory(paa_scale, max_intensity, hazard_type, hazard_unit, impf_id)]
        )

    return _make_impfset


@pytest.fixture(scope="session")
def impfset(impfset_factory):
    return impfset_factory()
