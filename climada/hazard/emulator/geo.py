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

Georegion objects for the hazard event emulator.
"""

import logging
import numpy as np
import geopandas as gpd
import shapely.ops
import shapely.vectorized
from shapely.geometry import Polygon

from climada.hazard import Centroids
import climada.util.coordinates as u_coord
import climada.hazard.emulator.const as const

LOGGER = logging.getLogger(__name__)


class HazRegion():
    """Hazard region for given geo information"""

    def __init__(self, extent=None, geometry=None, country=None, season=(1, 12)):
        """Initialize HazRegion

        If several arguments are passed, the spatial intersection is taken.

        Parameters
        ----------
        extent : tuple (lon_min, lon_max, lat_min, lat_max), optional
        geometry : GeoPandas DataFrame, optional
        country :  str or list of str, optional
            Countries are represented by their ISO 3166-1 alpha-3 identifiers.
            The keyword "all" chooses all countries (i.e., global land areas).
        season : pair of int, optional
            First and last month of hazard-specific season within this region
        """
        self._set_geometry(extent=extent, geometry=geometry, country=country)
        self.geometry['const'] = 0
        self.shape = self.geometry.dissolve(by='const').geometry[0]
        self.season = season


    def _set_geometry(self, extent=None, geometry=None, country=None):
        self.meta = {}

        if extent is not None:
            self.meta['extent'] = extent
        else:
            extent = (-180, 180, -90, 90)

        lon_min, lon_max, lat_min, lat_max = extent
        extent_poly = gpd.GeoSeries(Polygon([
            (lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)
        ]), crs=u_coord.NE_CRS)
        self.geometry = gpd.GeoDataFrame({'geometry': extent_poly}, crs=u_coord.NE_CRS)

        if country is not None:
            self.meta['country'] = country
            if country == "all":
                country = None
            elif not isinstance(country, list):
                country = [country]
            country_geom = u_coord.get_country_geometries(country_names=country)
            self.geometry = gpd.overlay(self.geometry, country_geom, how="intersection")

        if geometry is not None:
            self.meta['geometry'] = repr(geometry)
            self.geometry = gpd.overlay(self.geometry, geometry, how="intersection")


    def centroids(self, latlon=None, res_as=360):
        """Return centroids in this region

        Parameters
        ----------
        latlon : pair (lat, lon), optional
            Latitude and longitude of centroids.
            If not given, values are taken from CLIMADA's base grid (see `res_as`).
        res_as : int, optional
            One of 150 or 360. When `latlon` is not given, choose coordinates from centroids
            according to CLIMADA's base grid of given resolution in arc-seconds. Default: 360.

        Returns
        -------
        centroids : climada.hazard.Centroids object
        """
        if latlon is None:
            centroids = Centroids.from_base_grid(res_as=res_as)
            centroids.set_meta_to_lat_lon()
            lat, lon = centroids.lat, centroids.lon
        else:
            lat, lon = latlon
            centroids = Centroids()
            centroids.set_lat_lon(lat, lon)
        msk = shapely.vectorized.contains(self.shape, lon, lat)
        centroids = centroids.select(sel_cen=msk)
        centroids.id = np.arange(centroids.lon.shape[0])
        return centroids


class TCRegion(HazRegion):
    """Hazard region with support for TC ocean basins"""

    def __init__(self, tc_basin=None, season=None, **kwargs):
        """Initialize TCRegion

        The given geo information must be such that everything is contained in a single
        TC ocean basin.

        Parameters
        ----------
        tc_basin : str
            TC (sub-)basin abbreviated name, such as "SIW". If not given, automatically determined
            from geometry and basin bounds.
        **kwargs : see HazRegion.__init__
        """
        self._set_geometry(**kwargs)
        self.tc_basin = None

        if tc_basin is not None:
            tc_basin_geom = get_tc_basin_geometry(tc_basin)
            self.geometry = gpd.overlay(self.geometry, tc_basin_geom, how="intersection")
            self.meta['tc_basin'] = tc_basin
            self.tc_basin = tc_basin

        self.geometry['const'] = 0
        self.shape = self.geometry.dissolve(by='const').geometry[0]

        if self.tc_basin is None:
            self._determine_tc_basin()
        self.hemisphere = 'S' if const.TC_BASIN_GEOM_SIMPL[self.tc_basin][0][3] <= 0 else 'N'

        if season is None:
            season = const.TC_BASIN_SEASONS[self.tc_basin[:2]]
        self.season = season


    def _determine_tc_basin(self):
        for basin in const.TC_SUBBASINS:
            basin_geom = get_tc_basin_geometry(basin)
            if all(basin_geom.contains(self.shape)):
                self.tc_basin = basin
                break
        if self.tc_basin is None:
            raise ValueError("Region is not contained in a single basin!")
        for tc_basin in const.TC_SUBBASINS[self.tc_basin]:
            tc_basin_geom = get_tc_basin_geometry(tc_basin)
            if all(tc_basin_geom.contains(self.shape)):
                self.tc_basin = tc_basin
                break
        LOGGER.info("Automatically determined TC basin: %s", self.tc_basin)


def get_tc_basin_geometry(tc_basin):
    """Get TC (sub-)basin geometry

    Parameters
    ----------
    tc_basin : str
        TC (sub-)basin abbreviated name, such as "SIW" or "NA".

    Returns
    -------
    df : GeoPandas DataFrame
    """
    polygons = []
    for rect in const.TC_BASIN_GEOM[tc_basin]:
        lonmin, lonmax, latmin, latmax = rect
        polygons.append(Polygon([
            (lonmin, latmin),
            (lonmin, latmax),
            (lonmax, latmax),
            (lonmax, latmin)
        ]))
    polygons = shapely.ops.unary_union(polygons)
    polygons = gpd.GeoSeries(polygons, crs=u_coord.NE_CRS)
    return gpd.GeoDataFrame({'geometry': polygons}, crs=u_coord.NE_CRS)
