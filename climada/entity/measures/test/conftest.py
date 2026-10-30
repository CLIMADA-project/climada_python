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
    - Impact values should be:
        * AAI = 18 = 1000*1/2*0.006+(1000+2000+3000+4000+5000)*0.25*0.004
        * RP20 = event1 = 0
        * RP50 = event2 = 0
        * RP100 = event3 = 500 = 1000*1/2
        * RP250 = event4 = 3750 = (1000+2000+3000+4000+5000)*0.25

"""

import geopandas as gpd
import numpy as np
import pytest
from scipy.sparse import csr_matrix

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
EXP_LATS = np.array([33, 33, 33, 33.25, 33.25, 33.25])

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


@pytest.fixture
def exposures_factory():
    def _make_exposures(
        values=VALUES,
        exp_lons=EXP_LONS,
        exp_lats=EXP_LATS,
        value_factor=1.0,
        ref_year=EXPOSURE_REF_YEAR,
        hazard_type=HAZARD_TYPE,
        group_id=None,
        crs=CRS_WGS84,
        impf_id=IMPF_ID,
        description=EXP_DESC,
        value_unit=EXPOSURE_VALUE_UNIT,
        categories=None,
    ):
        gdf = gpd.GeoDataFrame(
            {
                "value": values * value_factor,
                f"impf_{hazard_type}": impf_id,
                "category": categories,
                "geometry": gpd.points_from_xy(exp_lons, exp_lats, crs=crs),
            },
            crs=crs,
        )
        if group_id is not None:
            gdf["group_id"] = group_id

        return Exposures(
            data=gdf,
            description=description,
            ref_year=ref_year,
            value_unit=value_unit,
        )

    return _make_exposures


@pytest.fixture
def exposures(exposures_factory):
    return exposures_factory()


def hazard_frequency_factory(base=FREQUENCY):
    def _make_frequency(scale=1.0):
        return base * scale

    return _make_frequency


def hazard_frequency():
    return hazard_frequency_factory()


def hazard_intensity(max_intensity=HAZARD_MAX_INTENSITY, scale=1.0):
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
            [max_intensity, 0, 0, 0, 0, 0],
            [0, max_intensity / 2, 0, 0, 0, 0],
            [
                max_intensity / 4,
                max_intensity / 4,
                max_intensity / 4,
                max_intensity / 4,
                max_intensity / 4,
                max_intensity / 4,
            ],
            [
                max_intensity,
                max_intensity,
                max_intensity,
                max_intensity,
                max_intensity,
                max_intensity,
            ],
        ]
    )

    return base * scale


@pytest.fixture
def centroids():
    return Centroids(lat=HAZ_LATS, lon=HAZ_LONS, crs=CRS_WGS84)


@pytest.fixture
def hazard_factory():
    def _make_hazard(
        intensity_matrix=None,
        frequency_array=FREQUENCY,
        max_intensity=HAZARD_MAX_INTENSITY,
        centroids=None,
        intensity_scale=1.0,
        frequency_scale=1.0,
        hazard_type=HAZARD_TYPE,
        hazard_unit=HAZARD_UNIT,
        lat=HAZ_LATS,
        lon=HAZ_LONS,
        crs=CRS_WGS84,
        event_id=EVENT_IDS,
        event_name=EVENT_NAMES,
        date=DATES,
        frequency_unit=FREQUENCY_UNIT,
    ):
        if intensity_matrix is None:
            intensity_matrix = hazard_intensity(max_intensity, intensity_scale)

        if centroids is None:
            centroids = Centroids(lat=lat, lon=lon, crs=crs)

        return Hazard(
            haz_type=hazard_type,
            units=hazard_unit,
            centroids=centroids,
            event_id=event_id,
            event_name=event_name,
            date=date,
            frequency=frequency_array * frequency_scale,
            frequency_unit=frequency_unit,
            intensity=intensity_matrix,
        )

    return _make_hazard


@pytest.fixture
def hazard(hazard_factory):
    return hazard_factory()


@pytest.fixture
def impf_factory():
    def _make_impf(
        paa_scale=1.0,
        max_intensity=HAZARD_MAX_INTENSITY,
        hazard_type=HAZARD_TYPE,
        hazard_unit=HAZARD_UNIT,
        impf_id=IMPF_ID,
        negative_intensities=False,
    ):
        intensity = np.array([0, max_intensity / 2, max_intensity])
        mdd = np.array([0, 0.5, 1])
        if negative_intensities:
            intensity = np.flip(intensity) * -1
            mdd = np.flip(mdd)
        return ImpactFunc(
            haz_type=hazard_type,
            intensity_unit=hazard_unit,
            name=IMPF_NAME,
            intensity=intensity,
            mdd=mdd,
            paa=np.array([1, 1, 1]) * paa_scale,
            id=impf_id,
        )

    return _make_impf


@pytest.fixture
def linear_impact_function(impf_factory):
    return impf_factory()


@pytest.fixture
def impfset_factory(impf_factory):
    def _make_impfset(
        paa_scale=1.0,
        max_intensity=HAZARD_MAX_INTENSITY,
        hazard_type=HAZARD_TYPE,
        hazard_unit=HAZARD_UNIT,
        impf_id=IMPF_ID,
        negative_intensities=False,
    ):
        return ImpactFuncSet(
            [
                impf_factory(
                    paa_scale,
                    max_intensity,
                    hazard_type,
                    hazard_unit,
                    impf_id,
                    negative_intensities,
                )
            ]
        )

    return _make_impfset


@pytest.fixture
def impfset(impfset_factory):
    return impfset_factory()
