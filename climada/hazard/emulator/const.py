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

Constants used by the hazard event emulator.
"""

TC_BASIN_GEOM = {
    # Eastern Pacific Basin
    'EP': [
        [-180.0, -75.0, 0.0, 9.0],
        [-180.0, -83.5, 9.0, 15.0],
        [-180.0, -92.0, 15.0, 18.0],
        [-180.0, -99.9, 18.0, 60.0]],
    'EPW': [[-180.0, -135.0, 0.0, 60.0]],
    'EPE': [
        [-135.0, -75.0, 0.0, 9.0],
        [-135.0, -83.5, 9.0, 15.0],
        [-135.0, -92.0, 15.0, 18.0],
        [-135.0, -99.9, 18.0, 60.0]],

    # "Global Basin"
    'GB': [[-179.9, 180.0, -50.0, 60.0]],

    # North Atlantic Basin
    'NA': [
        [-99.0, 13.0, 18.0, 60.0],
        [-91.0, 13.0, 15.0, 18.0],
        [-83.5, 13.0, 9.0, 15.0],
        [-78.0, 13.0, 0.0, 9.0]],
    'NAN': [[-99.0, 13.0, 31.0, 60.0]],
    'NAS': [
        [-99.0, 13.0, 18.0, 31.0],
        [-91.5, 13.0, 15.0, 18.0],
        [-83.5, 13.0, 9.0, 15.0],
        [-78.0, 13.0, 0.0, 9.0]],

    # Northern Indian Basin
    'NI': [[37.0, 99.0, 0.0, 30.0]],
    'NIW': [[37.0, 78.0, 0.0, 30.0]],
    'NIE': [[78.0, 99.0, 0.0, 30.0]],

    # Southern Atlantic
    'SA': [[-65.0, 20.0, -60.0, 0.0]],

    # Southern Indian Basin
    'SI': [[20.0, 135.0, -50.0, 0.0]],
    'SIW': [[20.0, 75.0, -50.0, 0.0]],
    'SIE': [[75.0, 135.0, -50.0, 0.0]],

    # Southern Pacific Basin
    'SP': [
        [135.0, 180.01, -50.0, 0.0],
        [-180.0, -68.0, -50.0, 0.0]],
    'SPW': [[135.0, 172.0, -50.0, 0.0]],
    'SPE': [
        [172.0, 180.01, -50.0, 0.0],
        [-180.0, -68.0, -50.0, 0.0]],

    # Western Pacific Basin
    'WP': [[99.0, 180.0, 0.0, 60.0]],
    'WPN': [[99.0, 180.0, 20.0, 60.0]],
    'WPS': [[99.0, 180.0, 0.0, 20.0]],
}
"""Boundaries of TC (sub-)basins (lon_min, lon_max, lat_min, lat_max)"""

TC_BASIN_GEOM_SIMPL = {
    # Eastern Pacific Basin
    'EP': [[-180.0, -75.0, 0.0, 60.0]],
    'EPW': [[-180.0, -135.0, 0.0, 60.0]],
    'EPE': [[-135.0, -75.0, 0.0, 60.0]],

    # North Atlantic Basin
    'NA': [[-105.0, -30.0, 0.0, 60.0]],
    'NAN': [[-105.0, -30.0, 31.0, 60.0]],
    'NAS': [[-105.0, -30.0, 0.0, 31.0]],

    # Northern Indian Basin
    'NI': [[37.0, 99.0, 0.0, 35.0]],
    'NIW': [[37.0, 78.0, 0.0, 35.0]],
    'NIE': [[78.0, 99.0, 0.0, 35.0]],

    # Southern Indian Basin
    'SI': [[20.0, 135.0, -50.0, 0.0]],
    'SIW': [[20.0, 75.0, -50.0, 0.0]],
    'SIE': [[75.0, 135.0, -50.0, 0.0]],

    # Southern Pacific Basin
    'SP': [[135.0, -60.0, -50.0, 0.0]],
    'SPW': [[135.0, 172.0, -50.0, 0.0]],
    'SPE': [[172.0, -60.0, -50.0, 0.0]],

    # Western Pacific Basin
    'WP': [[99.0, 180.0, 0.0, 60.0]],
    'WPN': [[99.0, 180.0, 20.0, 60.0]],
    'WPS': [[99.0, 180.0, 0.0, 20.0]],
}
"""Simplified boundaries of TC (sub-)basins (lon_min, lon_max, lat_min, lat_max)"""

TC_SUBBASINS = {
    'EP': ['EPW', 'EPE'],
    'NA': ['NAN', 'NAS'],
    'NI': ['NIW', 'NIE'],
    'SA': ['SA'],
    'SI': ['SIW', 'SIE'],
    'SP': ['SPW', 'SPE'],
    'WP': ['WPN', 'WPS'],
}
"""Abbreviated names of TC subbasins for each basin"""

TC_BASIN_SEASONS = {
    'WP': [5, 12],
    'NA': [6, 11],
    'NI': [5, 12],
    'EP': [7, 12],
    'SI': [11, 4],
    'SA': [1, 4],
    'SP': [11, 5],
}
"""Start/end months of hazard seasons in different basins"""

TC_BASIN_NORM_PERIOD = {
    'WP': (1950, 2015),
    'NA': (1950, 2015),
    'EP': (1950, 2015),
    'NI': (1980, 2015),
    'SI': (1980, 2015),
    'SP': (1980, 2015),
    'SA': (1980, 2015),
}
"""TC basin-specific start/end year of norm period (according to IBTrACS data availability)"""

PDO_SEASON = [11, 3]
"""Start/end months of PDO activity"""
