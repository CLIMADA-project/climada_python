# -*- coding: utf-8 -*-
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
"""
import logging
import time
from sys import stdout
from pathlib import Path
import numpy as np
import pandas as pd
from pandas_datareader import wb
from scipy import ndimage as nd
from scipy import stats
import geopandas as gpd
import shapefile
from matplotlib import pyplot as plt
from iso3166 import countries as iso_cntry
import gdal
from cartopy.io import shapereader

from climada import CONFIG
from climada.entity.exposures import nightlight
from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, INDICATOR_IF
from climada.entity.exposures import gpw_import
from climada.util import ureg
from climada.util.finance import gdp, income_group, wealth2gdp, world_bank_wealth_account
from climada.util.constants import SYSTEM_DIR, DEF_CRS
from climada.util.coordinates import pts_to_raster_meta, get_resolution

logging.root.setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

"""Define LitPop class."""
# Black Marble nightlight tile names, %i represents the Black Marble reference year
BM_FILENAMES = ['BlackMarble_%i_A1_geo_gray.tif',
                'BlackMarble_%i_A2_geo_gray.tif',
                'BlackMarble_%i_B1_geo_gray.tif',
                'BlackMarble_%i_B2_geo_gray.tif',
                'BlackMarble_%i_C1_geo_gray.tif',
                'BlackMarble_%i_C2_geo_gray.tif',
                'BlackMarble_%i_D1_geo_gray.tif',
                'BlackMarble_%i_D2_geo_gray.tif']
# years with Black Marble Tiles
# https://earthobservatory.nasa.gov/features/NightLights/page3.php
# Update if new years get available!
BM_YEARS = [2016, 2012]  # latest first
# Years with GPW population data available:
GPW_YEARS = [2020, 2015, 2010, 2005, 2000]

NASA_RESOLUTION_DEG = (15 * ureg.arc_second).to(ureg.deg).magnitude

WORLD_BANK_INC_GRP = CONFIG.exposures.litpop.resources.world_bank_inc_group.str()
"""Income group historical data from World bank."""

DEF_RES_NASA_KM = 0.5
"""Default approximate resolution for NASA's nightlights in km."""

DEF_RES_GPW_KM = 1
"""Default approximate resolution for the GPW dataset in km."""

DEF_RES_NASA_ARCSEC = 15
"""Default approximate resolution for NASA's nightlights in arcsec."""

DEF_RES_GPW_ARCSEC = 30
"""Default approximate resolution for the GPW dataset in arcsec."""

DEF_HAZ_TYPE = ''
"""Default hazard type used in impact functions id, i.e. TC"""

class LitPop(Exposures):
    """Defines exposure values from nightlight intensity (NASA), Gridded Population
        data (SEDAC); distributing produced capital (World Bank), GDP (World Bank)
        or non-financial wealth (Global Wealth Databook by the Credit Suisse
        Research Institute.)

        Calling sequence example:
        ent = LitPop()
        country_name = ['Switzerland', 'Austria']
        ent.set_country(country_name)
        ent.plot()
    """

    @property
    def _constructor(self):
        return LitPop

    def clear(self):
        """Appending the base class clear attribute to also delete attributes
            which are only used here.
        """
        Exposures.clear(self)
        try:
            del self.country_data
        except AttributeError:
            pass

    def set_country(self, countries, **args):
        """Get LitPop based exposre for one country or multiple countries
        using values at reference year. If produced capital, GDP, or income
        group, etc. not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (str or list): list of countries or single county as a
                sting. Countries can either be country names ('France') or
                country codes ('FRA'), even a mix is possible in the list.
        args: Keyword arguments. The following keywords are recognised:
            res_km (float, optional): approx resolution in km. Default: 1km.
            res_arcsec (float, optional): resolution in arc-sec. Overrides
                res_km if both are delivered
            check_plot (boolean, optional): choose if a plot is shown at the
                end of the operation.
            exponents (list of two integers, default = [1, 1]) defining
                power with which lit (nightlights) and pop (gpw) go into LitPop.
                To get nightlights^3 alone: [3, 0].
                To use population count alone: [0, 1].
            fin_mode (str, optional): define what total country economic value
                is to be used as an asset base and distributed to the grid:
                - 'gdp': gross-domestic product (Source: World Bank)
                - 'income_group': gdp multiplied by country's income group+1
                - 'nfw': non-financial wealth (Source: Credit Suisse, of households only)
                - 'tw': total wealth (Source: Credit Suisse, of households only)
                - 'pc': produced capital (Source: World Bank), incl. manufactured or
                    built assets such as machinery, equipment, and physical structures
                    (pc is in constant 2014 USD)
                - 'norm': normalized by country
                - 'none': LitPop per pixel is returned unchanged
            admin1_calc (boolean): distribute admin1-level GDP if available?
                (default False)
            conserve_cntrytotal (boolean): given admin1_calc, conserve national
            total asset value (default True)
            reference_year (int)
            adm1_scatter (boolean): produce scatter plot for admin1 validation?
        """
        # TODO: allow for user delivered path
#        self.clear() # clear existing assets (reset)
        start_time = time.time()
        res_km = args.get('res_km', 1)
        res_arcsec = args.get('res_arcsec', [])
        fin_mode = args.get('fin_mode', 'pc')
        exponents = args.get('exponents', [1, 1])
        admin1_calc = args.get('admin1_calc', False)
        adm1_scatter = args.get('adm1_scatter', False)
        conserve_cntrytotal = args.get('conserve_cntrytotal', True)
        reference_year = args.get('reference_year', 2016)
        # bm_year = min(BM_YEARS, key=lambda x:abs(x-reference_year))
#        inherit_admin1_from_admin0 = args.get('inherit_admin1_from_admin0', 1)
        if res_arcsec == []:
            resolution = (res_km / DEF_RES_GPW_KM) * DEF_RES_GPW_ARCSEC
        else:
            resolution = res_arcsec
        _match_target_res(resolution)
        check_plot = args.get('check_plot', 0)
        country_info = dict()
        admin1_info = dict()
        if isinstance(countries, list):  # multiple countries
            list_len = len(countries)
            country_list = countries
            for i, country in enumerate(country_list[::-1]):
                country_new = _get_iso3(country)
                country_list[list_len - 1 - i] =\
                    country_new
                if country_new is None:
                    LOGGER.warning('The country %s could not be found.', country)
                    LOGGER.warning('Country %s is removed from the list.', country)
                    del country_list[list_len - 1 - i]
                else:
                    country_info[country_new], admin1_info[country_new] =\
                        _get_country_info(country_new)
                del country_new
            if not country_list:
                LOGGER.error('No valid country chosen. Operation aborted.')
                raise ValueError
            else:
                all_bbox = [_get_country_shape(countr, 1)[0] for countr in country_list]
                cut_bbox = _bbox_union(all_bbox)
        elif isinstance(countries, str):  # One country
            country_list = list()
            country_list.append(countries)
            country_new = _get_iso3(countries)
            country_list[0] = country_new
            if not _get_country_shape(country_list[0], 1) is None:
                all_bbox = _get_country_shape(country_list[0], 1)[0]
            else:
                LOGGER.error('Country %s could not be found.', countries)
                raise ValueError
            cut_bbox = all_bbox
            country_info[country_list[0]], admin1_info[country_list[0]]\
                = _get_country_info(country_list[0])
        else:
            LOGGER.error('Country parameter data type not recognised. \
                         Operation aborted.')
            raise TypeError
        all_coords = _litpop_box2coords(cut_bbox, resolution, 1)
        # Get LitPop
        LOGGER.info('Generating LitPop data at a resolution of %s arcsec.', str(resolution))
        litpop_data = _get_litpop_box(cut_bbox, resolution, 0, reference_year,
                                      exponents)
        shp_file = shapereader.natural_earth(resolution='10m',
                                             category='cultural',
                                             name='admin_0_countries')
        shp_file = shapereader.Reader(shp_file)

        for cntry_iso, cntry_val in country_info.items():
            if fin_mode == 'pc':
                total_asset_val = world_bank_wealth_account(cntry_iso, reference_year,
                                                            no_land=True)[1]
                # here, total_asset_val is Produced Capital "pc"
                # no_land=True returns value w/o the mark-up of 24% for land value
            elif fin_mode in ['norm', 'none']:
                total_asset_val = 1
            else:
                _, total_asset_val = gdp(cntry_iso, reference_year, shp_file)
            cntry_val.append(total_asset_val)
        _get_gdp2asset_factor(country_info, reference_year, shp_file, fin_mode=fin_mode)

        tag = Tag()
        lp_cntry = list()
        for curr_country in country_list:
            curr_shp = _get_country_shape(curr_country, 0)
            mask = _mask_from_shape(curr_shp, resolution=resolution,
                                    points2check=all_coords)
            litpop_curr = litpop_data[mask.sp_index.indices]
            lon, lat = zip(*np.array(all_coords)[mask.sp_index.indices])
            if fin_mode == 'none':
                LOGGER.info('fin_mode=none --> no downscaling; admin1_calc is ignored')
            elif admin1_calc == 1:
                litpop_curr = _calc_admin1(curr_country,
                                           country_info[curr_country],
                                           admin1_info[curr_country],
                                           litpop_curr, list(zip(lon, lat)),
                                           resolution, adm1_scatter,
                                           conserve_cntrytotal=conserve_cntrytotal,
                                           check_plot=check_plot, masks_adm1=[], return_data=1)
            else:
                litpop_curr = _calc_admin0(litpop_curr,
                                           country_info[curr_country][3],
                                           country_info[curr_country][4])
            lp_cntry.append(self._set_one_country(country_info[curr_country],
                                                  litpop_curr, lon, lat, curr_country))
            tag.description += ('LitPop for %s at %i as, year=%i, financial mode=%s, '
                                'GPW-year=%i, BM-year=%i, exp=[%i, %i]'
                                % (country_info[curr_country][1], resolution, reference_year,
                                   fin_mode,
                                   min(GPW_YEARS, key=lambda x: abs(x - reference_year)),
                                   min(BM_YEARS, key=lambda x: abs(x - reference_year)),
                                   exponents[0], exponents[1]))
        Exposures.__init__(self, gpd.GeoDataFrame(pd.concat(lp_cntry, ignore_index=True)),
                           crs=DEF_CRS)
        self.ref_year = reference_year
        self.tag = tag
        self.value_unit = 'USD'
        try:
            rows, cols, ras_trans = pts_to_raster_meta(
                (self.longitude.min(), self.latitude.min(),
                 self.longitude.max(), self.latitude.max()),
                get_resolution(self.longitude, self.latitude))
            self.meta = {
                'width': cols,
                'height': rows,
                'crs': self.crs,
                'transform': ras_trans,
            }
        except ValueError:
            LOGGER.warning('Could not write attribute meta, because exposure'
                           ' has only 1 data point')
            self.meta = {}
        if check_plot == 1:
            self.plot_log(admin1_plot=0)
        LOGGER.info("Creating the LitPop exposure took %i s",
                    int(round(time.time() - start_time, 2)))
        # self.set_geometry_points()
        self.check()

    @staticmethod
    def _set_one_country(cntry_info, litpop_data, lon, lat, curr_country):
        """Model one country.

        Parameters:
            cntry_info (list): [cntry_id, cnytry_name, cntry_geometry,
                ref_year, gdp, income_group]
            litpop_data (pandas.arrays.SparseArray): LitPop data with the value
                already distributed.
            lon (array): longitudinal coordinates
            lat (array): latudinal coordinates
            curr_country: name or iso3 ID of country
        """
        lp_ent = LitPop()
        lp_ent['value'] = litpop_data.to_numpy()
        lp_ent['latitude'] = lat
        lp_ent['longitude'] = lon
        try:
            lp_ent['region_id'] = np.ones(lp_ent.value.shape, int) \
                    * int(iso_cntry.get(cntry_info[1]).numeric)
        except KeyError:
            lp_ent['region_id'] = np.ones(lp_ent.value.shape, int) \
                    * int(iso_cntry.get(curr_country).numeric)
        lp_ent[INDICATOR_IF + DEF_HAZ_TYPE] = np.ones(lp_ent.value.size, int)
        return lp_ent

    def _append_additional_info(self, cntries_info):
        """Add country information in dictionary attribute country_data.

        Parameters:
            cntries_info (dict): country ISO3, name and shape
        """
        self.country_data = {'ISO3': [], 'name': [], 'shape': []}
        for cntry_iso3, cntry_info in cntries_info.items():
            self.country_data['ISO3'].append(cntry_iso3)
            self.country_data['name'].append(cntry_info[1])
            self.country_data['shape'].append(cntry_info[2])

    def plot_log(self, admin1_plot=1):
        """Plots the LitPop data with the color scale reprenting the values
            in a logarithmic scale.

        Parameters:
            admin1_plot (boolean): whether admin1 borders should be plotted.
                Default=1
        """
        # TODO: plot subplots for the different countries instead of one global
        # one. Countries can be identified by their region id, hence this
        # can be implemented
        import matplotlib.colors as colors
        if not self.value.sum() == 0:
            plt.figure()
#            countr_shape = _get_country_shape(country_iso, 0)
            countr_bbox = np.array((min(self.coord[:, 1]),
                                    min(self.coord[:, 0]),
                                    max(self.coord[:, 1]),
                                    max(self.coord[:, 0])))
            plt.gca().set_xlim(countr_bbox[0]
                               - 0.1 * (countr_bbox[2] - countr_bbox[0]), countr_bbox[2]
                               + 0.1 * (countr_bbox[2] - countr_bbox[0]))
            plt.gca().set_ylim(countr_bbox[1]
                               - 0.1 * (countr_bbox[3] - countr_bbox[1]), countr_bbox[3]
                               + 0.1 * (countr_bbox[3] - countr_bbox[1]))
            plt.scatter(self.coord[:, 1], self.coord[:, 0],
                        c=self.value, marker=',', s=3,
                        norm=colors.LogNorm())
            plt.title('Logarithmic scale LitPop value')
            if hasattr(self, 'country_data') and\
                    self.country_data['shape'] != []:
                for idx, shp in enumerate(self.country_data['shape']):
                    _plot_shape_to_plot(shp)
                    if admin1_plot == 1:
                        _plot_admin1_shapes(self.country_data['ISO3'][idx],
                                            0.6)
            plt.colorbar()
            plt.show()

def _get_litpop_box(cut_bbox, resolution, return_coords=0,
                    reference_year=2016, exponents=None):
    """
    PURPOSE:
        A function which retrieves and calculates the LitPop data within a
        certain bounding box for a given resolution.
    INPUTS:
        cut_bbox (1x4 array-like): Bounding box (ESRI type) of interest.
            The layout of the bounding box corresponds to the bounding box of
            the ESRI shape files and is as follows:
            [minimum longitude, minimum latitude, maximum longitude,
                 maxmimum latitude]
        resolution (scalar): resolution in arc-seconds
        reference_year (int): reference year, population available at:
            2000, 2005, 2010, 2015 (default), 2020
        exponents (list of two integers, default = [1, 1]) defining power with
            which lit (nightlights) and pop (gpw) go into LitPop.
            To get nightlights^3 alone: [3, 0].
            To use population count alone: [0, 1].
    OUTPUT (either one of these lines, depending on option return_coords):
        litpop_data (pandas.arrays.SparseArray): A pandas SparseArray containing the
            raw, unnormalised LitPop data.
        OR
        litpop_data, lon, lat (tuple): if return_coords=1 a tuple in the
            form (lon, lat) with the coordinates falling into the cut_bbox are
            return along with the litpop_data (see above).
    """
    if exponents is None:
        exponents = [1, 1]

    nightlights = _get_box_blackmarble(cut_bbox, reference_year=reference_year,
                                       resolution=resolution, return_coords=0)
    gpw = gpw_import.get_box_gpw(cut_bbox=cut_bbox, resolution=resolution,
                                 return_coords=0, reference_year=reference_year)
    bm_temp = np.ones(nightlights.shape)
    # Lit = Lit + 1 if Population is included, c.f. int(exponents[1]>0):
    bm_temp[nightlights.sp_index.indices] = (np.array(nightlights.sp_values, dtype='uint16')
                                             + int(exponents[1] > 0))
    nightlights = pd.arrays.SparseArray(bm_temp, fill_value=int(exponents[1] > 0))
    del bm_temp

    litpop_data = _LitPop_multiply(nightlights, gpw, exponents=exponents)

    if return_coords == 1:
        lon, lat = _litpop_box2coords(cut_bbox, resolution, 0)
        return litpop_data, lon, lat
    return litpop_data

def _LitPop_multiply(nightlights, gpw, exponents):
    """
    PURPOSE:
        Pixel-wise multiplication of lit (nightlights^exponents[0]) and pop
        (gpw^exponents[1]) to compute LitPop.
        Both factors are included to the power of lit_exp / pop_exp to
        change their weight.
    INPUTS:
        nightlights (SparseArray): gridded nightlights data
        gpw (SparseArray): gridded population data
        exponents (list of two integers): exponents for nightlights and
            population data
    OUTPUT:
        litpop_data (dataframe): gridded resulting LitPop
    """
    litpop_data = pd.arrays.SparseArray(
        np.multiply(nightlights.to_numpy()**exponents[0], gpw.to_numpy()**exponents[1]), fill_value=0)
    return litpop_data

def _litpop_box2coords(box, resolution, point_format=0):
    """
    PURPOSE:
        A function which calculates coordinates arrays explicitly from a
        bounding box for a given resolution
    INPUTS:
        box (1x4 array-like): Bounding box (ESRI type) of coordinates to be
            delivered.
            the layout of the bounding box corresponds to the bounding box of
            the ESRI shape files and is as follows:
            [minimum longitude, minimum latitude, maximum longitude,
                 maxmimum latitude]
        resolution (scalar): resolution in arc-seconds
    OUTPUT (either one of these lines, depending on point_format):
        lon, lat (tuple of arrays): if point_format =0 (default) is selected
            a separate array for the longitude and latitude of each pixel in
            the bounding box is returned.
        coordiates (array): if point_format =1 is selected a tuple entry of
            the form (lon, lat) for each point is returned.
    """
    deg_per_pix = 1 / (3600 / resolution)
    min_col, min_row, max_col, max_row =\
        _litpop_coords_in_glb_grid(box, resolution)
    lon = np.array(np.transpose([np.ones((max_row - min_row + 1,))
                                 * ((-180 + (deg_per_pix / 2)) + l_i * deg_per_pix)
                                 for l_i in range(min_col, (max_col + 1))]))
    lon = lon.flatten(order='F')
    lat = np.array(
        np.transpose(
            [((90 - (deg_per_pix / 2)) - (l_j * deg_per_pix))
             for l_j in range(min_row, (max_row + 1))]
            * np.ones((max_col - min_col + 1, (max_row - min_row + 1)))
        )
    )
    lat = lat.flatten(order='F')
    if point_format == 1:
        return list([(lon, lat) for lon, lat in zip(lon, lat)])
    return lon, lat

def _litpop_coords_in_glb_grid(box, resolution):
    """
    PURPOSE:
        Function which calculates the coordinates from geographic to
    a cartesian coordinate system, where the NE-most point is 0,0.
    INPUTS:
        box (1x4 array-like): Bounding box (ESRI type) of coordinates to be
            delivered.
            the layout of the bounding box corresponds to the bounding box of
            the ESRI shape files and is as follows:
            [minimum longitude, minimum latitude, maximum longitude,
                 maxmimum latitude]
        resolution (scalar): resolution in arc-seconds
    OUTPUTS:
        mincol, minrow, maxcol, maxrow (array): row and col numbers which
            define the box in the cartesian coordinate system.
    """
    minlon, minlat, maxlon, maxlat = box
    deg_per_pix = 1 / (3600 / resolution)
    minlon, maxlon = minlon - (-180), maxlon - (-180)
    minlat, maxlat = -(minlat - (90)), -(maxlat - (90))
    lon_dist = np.ceil(abs(maxlon - minlon) / deg_per_pix)
    lat_dist = np.ceil(abs(maxlat - minlat) / deg_per_pix)
    mincol = int(max(minlon // deg_per_pix, 0))
    maxcol = int(max(mincol + lon_dist - 1, mincol))
    minrow = int(max(maxlat // deg_per_pix, 0))
    maxrow = int(max(minrow + lat_dist - 1, minrow))
    return np.array((mincol, minrow, maxcol, maxrow))

'''
def _litpop_convert_coords(box, resolution):
    """
    PURPOSE:
        A function which fits coordinates to global LitPop grid. Main purpose
        is to keep track of coordinates cut in reading BM and GPW without
        having to store pixel-based coordinates.
    INPUTS:
        box (1x4 array-like): Bounding box (ESRI type) of coordinates to be
            delivered. the layout of the bounding box corresponds to the
            bounding box of the ESRI shape files and is as follows:
                [minimum longitude, minimum latitude, maximum longitude,
                 maxmimum latitude]
        resolution (scalar): resolution in arc-seconds
    OUTPUT:
        box (1x4 array-like): bounding box with coordinates adjusted to global
            LitPop grid.
    """
    minlon, maxlon = box[0], box[2]
    minlat, maxlat = box[1], box[3]
    deg_per_pix = 1/(3600/resolution)
    min_col, max_col = int(max(minlon//deg_per_pix, -180*(1/deg_per_pix))),\
        int(min(maxlon//deg_per_pix, (180-(deg_per_pix/2))*(1/deg_per_pix)))
    min_row, max_row = int(max(minlat//deg_per_pix, -90*(1/deg_per_pix))),\
        int(min(maxlat//deg_per_pix, (90-(deg_per_pix/2))*(1/deg_per_pix)))
    box = np.array((min_col*deg_per_pix+(deg_per_pix/2), min_row*deg_per_pix\
                    +(deg_per_pix/2), max_col*deg_per_pix\
                    +(deg_per_pix/2), max_row*deg_per_pix+(deg_per_pix/2)))
    return box
'''

def _get_country_shape(country_iso, only_geo=0):
    """Retrieves the shape file or coordinate information of a country.

    Parameters:
        country_iso (str): country code of country to get
        only_geo (boolean): Determines the output mode (default =0):
            if =0: returns the entire shape file of the country
            if =1: returns a tuple of values: bbox, lat, lon (see below)

    Returns:
        if only_geo = 0 (default):
            The shape of type shapefile._Shape
        if only_geo = 1
            bbox, lat, lon (tuple of size 3)
                bbox is a 1x4 vector of the bounding box of the country (array)
                lat is a mx1 vector of the latitudinal values of the vertices
                    of the shape (array)
                lon is a mx1 vector of the longitudinal values of the vertices
                    of the shape (array)
    """
    country_iso = country_iso.casefold()
    shp = shapereader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
    shp = shapefile.Reader(shp)
    if len(country_iso) == 3:
        for field_num, field in enumerate(shp.fields[1::]):
            # Skip first (index zero) field, because it is DeletionFlag
            if field[0] == 'ADM0_A3':
                break
    else:
        for field_num, field in enumerate(shp.fields[1::]):
            # Skip first (index zero) field, because it is DeletionFlag
            if field[0] == 'ADMIN':
                break
    del field
    for rec_i, rec in enumerate(shp.records()):
        if rec[field_num].casefold() == country_iso:
            if only_geo != 1:
                return shp.shapes()[rec_i]
            bbox = shp.shapes()[rec_i].bbox
            points = shp.shapes()[rec_i].points
            lat = np.array([x[1] for x in points])
            lon = np.array([x[0] for x in points])
            return bbox, lat, lon

def _match_target_res(target_res='NA'):
    """Checks whether the resolution is compatible with the "legacy"
        resolutions used in Matlab Climada and produces a warning message
        if not.

        Parameters:
            target_res (scalar): Resolution in arc seconds.
    """
    res_list = [30, 60, 120, 300, 600, 3600]
    out_res = min(res_list, key=lambda x: abs(x - target_res))
    if out_res != target_res:
        LOGGER.warning('Not one of the legacy resoultions selected. Consider \
                       adjusting it to %s arc-sec.', out_res)

def _shape_cutter(shape, **opt_args):
    """Checks whether given coordinates are within a shape or not. Can also
        check if a shape possesses enclaves and cuts them out accordingly.
        If no coordinates are supplied, all coordinates in the bounding box
        of the shape under the given resolution are checked.

    Parameters:
        shape (_shape): shape file to check
    Optional:
        opt_args (keyword arguments):
            resolution (scalar): resolution of the points to be checked in
                arcsec. Required if the points need to be created first.
                Defautl = 30.
            check_enclaves (boolean): If activated, enclaves get detected and
                cut out from shapes. Default = 1.
            check_plot (boolean): If activated, a plot with the shap and the
                mask is shown. Default = 0.
            shape_format (str, tuple): colour of the shape if it is plotted.
                Takes any colour format which is recognised by matplotlib.
            enclave_format (str, tuple): colour of the enclaves if it they are
                plotted. Takes any colour format which is recognised by
                matplotlib.
            return_mask (boolean): If activated, the mask is also returned
            points2check (list): a list of points in tuple format (lon, lat)
                for which should be checked whether they are inside the shape.
                if no points are delivered, the points are created for the
                bounding box of the shape.
            point_format (boolean): If activated the points get returned as
                a list in tuple format (lon, lat), otherwise, lon and lat
                get returned separately.

    Returns (depending on the chosen options):
        lon (list): list of longitudinal coordinate data of points inside shape
            (returned if points_format = 0)
        lat (list): list of latitudianl coordinate data of points inside shape
            (returned if points_format = 0)
        incl_coords (list): list of tuples of formate (lon, lat) of points
            inside shape (returned if points_format=1)
        enclave_paths (list): list of detected enclave paths
        mask (pandas.arrays.SparseArray): SparseArray with one where point is
            inside shape and zero otherwise. (only returned if return_mask=1)
        if only_geo = 0 (default):
            The shape of type shapefile._Shape
        if only_geo = 1
            bbox, lat, lon (tuple of size 3)
                bbox is a 1x4 vector of the bounding box of the country (array)
                lat is a mx1 vector of the latitudinal values of the vertices
                    of the shape (array)
                lon is a mx1 vector of the longitudinal values of the vertices
                    of the shape (array)
    """
    from matplotlib import path
    resolution = opt_args.get('resolution', 30)
    check_enclaves = opt_args.get('check_enclaves', 1)
    check_plot = opt_args.get('check_plot', 0)
    return_mask = opt_args.get('return_mask', 1)
    if check_plot == 1:
        shape_format = opt_args.get('shape_format', str(0.3))
        enclave_format = opt_args.get('enclave_format', shape_format)
    points2check = opt_args.get('points2check', [])
    point_format = opt_args.get('point_format', 1)
    if (not hasattr(shape, 'points')) or (not hasattr(shape, 'parts')):
        LOGGER.error('Not a valid shape. Please make sure, the shapefile is \
                     of type from package "shapefile".')
    sub_shapes = len(shape.parts)
    all_coords_shape = [(x, y) for x, y in shape.points]
    LOGGER.debug('Extracting subshapes and detecting enclaves...')
    sub_shape_path = []
    enclave_paths = []
    add2enclave = 0
    if sub_shapes > 1:
        for i in range(0, sub_shapes):
            if i == (sub_shapes - 1):
                end_idx = len(shape.points) - 1
            else:
                end_idx = shape.parts[i + 1] - 1
            if (i > 0) & (check_enclaves == 1):
                temp_path = path.Path(all_coords_shape[shape.parts[i]:end_idx])
                for val in sub_shape_path:
                    if val.contains_point(temp_path.vertices[0]) and \
                    len(temp_path.vertices) > 2 and \
                    val.contains_point(temp_path.vertices[1]) and\
                    val.contains_point(temp_path.vertices[2]):
                        add2enclave = 1
                        break
                if add2enclave == 1:
                    enclave_paths.append(temp_path)
                    temp_path = []
                    add2enclave = 0
                else:
                    sub_shape_path.append(temp_path)
                    temp_path = []
            else:
                sub_shape_path.append(path.Path(all_coords_shape[shape.parts[i]:end_idx]))
        if check_enclaves == 1:
            LOGGER.debug('Detected subshapes: %s, of which subshapes: %s',
                         str(sub_shapes), str(len(enclave_paths)))
        else:
            LOGGER.debug('Detected subshapes: %s. Enclave checking disabled',
                         str(sub_shapes))
    else:
        sub_shape_path.append(path.Path(all_coords_shape))
    del all_coords_shape
    incl_coords = []
    for _, val in enumerate(sub_shape_path):
        add_points = _mask_from_path(val, resolution)
        if add_points is not None:
            [incl_coords.append(point) for point in add_points]
        del add_points
    stdout.write('\n')
    if check_enclaves == 1 and not enclave_paths:
        excl_coords = []
        # LOGGER.debug('Removing enclaves...')
        for _, val in enumerate(enclave_paths):
            temp_excl_points = _mask_from_path(val, resolution)
            if temp_excl_points is not None:
                [excl_coords.append(point) for point in temp_excl_points]
            del temp_excl_points
        excl_coords = set(tuple(row) for row in excl_coords)
        incl_coords = [point for point in incl_coords if point not
                       in excl_coords]
    # LOGGER.debug('Successfully isolated coordinates from shape')
    total_bbox = np.array((min([x[0] for x in shape.points]),
                           min([x[1] for x in shape.points]),
                           max([x[0] for x in shape.points]),
                           max([x[1] for x in shape.points])))
    if points2check == []:
        all_coords = _litpop_box2coords(total_bbox, resolution, 1)
    else:
        all_coords = points2check
        del points2check
    incl_coords = set(incl_coords)
    mask = np.array([(coord in incl_coords) for coord in all_coords])
    mask = pd.arrays.SparseArray(mask, fill_value=0)
    lon, lat = zip(*[all_coords[val] for idx, val
                     in enumerate(mask.sp_index.indices)])
    if check_plot == 1:
        plt.scatter(lon, lat, cmap='plasma', marker=',')
        _plot_shape_to_plot(shape, shape_format)
        if check_enclaves == 1 and not enclave_paths:
            _plot_paths_to_plot(enclave_paths, enclave_format)
    if point_format == 1:
        if return_mask == 1:
#            LOGGER.debug('Cutting the shape took %s s',\
#                         str(round(time.time()-curr_time, 2)))
            return zip(lon, lat), enclave_paths, mask
#        LOGGER.debug('Cutting the shape took %s s',\
#                     str(round(time.time()-curr_time, 2)))
        return incl_coords, enclave_paths

#    LOGGER.debug('Cutting the shape took %s s',\
#                 str(round(time.time()-curr_time, 2)))
    if return_mask == 1:
        return lon, lat, enclave_paths, mask
    lat = [x[1] for x in incl_coords]
    lon = [x[0] for x in incl_coords]
    return lon, lat, enclave_paths

def _mask_from_path(path, resolution=30, return_points=1, return_mask=0):
    curr_bbox = np.array((min([x[0] for x in path.vertices]),
                          min([x[1] for x in path.vertices]),
                          max([x[0] for x in path.vertices]),
                          max([x[1] for x in path.vertices])))
    curr_points2check = _litpop_box2coords(curr_bbox, resolution, 1)
    del curr_bbox
    if curr_points2check == []:
        return None
    temp_mask = pd.arrays.SparseArray(path.contains_points(curr_points2check),
                               fill_value=0)
    points_in = [curr_points2check[val] for idx, val
                 in enumerate(temp_mask.sp_index.indices)]
    if return_points == 1:
        if return_mask == 1:
            return points_in, temp_mask
        return points_in

    lon, lat = [x[0] for x in points_in], [x[1] for x in points_in]
    if return_mask == 1:
        return lon, lat, temp_mask
    return lon, lat

def _mask_from_shape(check_shape, **opt_args):
    """creates a mask from a shape assigning value 1 to points inside and 0
        otherwise.

    Parameters
    ----------
    check_shape : _Shape
        shape file to check
    resolution : scalar, optional
        resolution of the points to be checked in arcsec.
        Required if the points need to be created first.
        Default: 30.
    check_enclaves : boolean, optional
        If activated, enclaves get detected and cut out from shapes. Default: 1.
    check_plot : boolean, optional
        If activated, a plot with the shap and the mask is shown. Default: 0.
    shape_format : str, tuple, optional
        colour of the shape if it is plotted.
        Takes any colour format which is recognised by matplotlib.
    enclave_format : str, tuple, optional
        colour of the enclaves if it they are plotted.
        Takes any colour format which is recognised by matplotlib.
    return_mask : boolean, optional
        If activated, the mask is also returned
    points2check : list, optional
        A list of points in tuple formaat (lon, lat) for which should be
        checked whether they are inside the shape.
        If no points are delivered, the points are created for the bounding box
        of the shape.
    point_format : boolean, optional
        If activated the points get returned as a list in tuple format
        (lon, lat), otherwise, lon and lat get returned separately.

    Returns
    -------
    lon : list
        list of longitudinal coordinate data of points inside shape
        (returned if points_format = 0)
    lat : list
        list of latitudianl coordinate data of points inside shape
        (returned if points_format = 0)
    incl_coords : list
        list of tuples of formate (lon, lat) of points
        inside shape (returned if points_format=1)
    enclave_paths : list
        list of detected enclave paths
    mask : pandas.arrays.SparseArray
        SparseArray with one where point is
        inside shape and zero otherwise. (only returned if return_mask=1)
    """
    from matplotlib import path
    resolution = opt_args.get('resolution', 30)
    check_enclaves = opt_args.get('check_enclaves', 1)
    points2check = opt_args.get('points2check', [])
    if (not hasattr(check_shape, 'points')) or (not hasattr(check_shape, 'parts')):
        LOGGER.error('Not a valid shape. Please make sure, the shapefile is \
                     of type from package "shapefile".')
    sub_shapes = len(check_shape.parts)
    all_coords_shape = [(x, y) for x, y in check_shape.points]
    # LOGGER.debug('Extracting subshapes and detecting enclaves...')
    sub_shape_path = []
    enclave_paths = []
    add2enclave = 0
    if sub_shapes > 1:
        for i in range(0, sub_shapes):
            if i == (sub_shapes - 1):
                end_idx = len(check_shape.points) - 1
            else:
                end_idx = check_shape.parts[i + 1] - 1
            if (i > 0) & (check_enclaves == 1):
                temp_path = path.Path(all_coords_shape[check_shape.parts[i]:end_idx])
                for val in sub_shape_path:
                    if val.contains_point(temp_path.vertices[0]) and \
                            len(temp_path.vertices) > 2 and \
                            val.contains_point(temp_path.vertices[1]) and\
                            val.contains_point(temp_path.vertices[2]):
                        # Only check if the first three vertices of the new shape
                        # is in any of the old shapes for speed
                        add2enclave = 1
                        break
                if add2enclave == 1:
                    enclave_paths.append(temp_path)
                    temp_path = []
                    add2enclave = 0
                else:
                    sub_shape_path.append(temp_path)
                    temp_path = []
            else:
                sub_shape_path.append(path.Path(all_coords_shape[check_shape.parts[i]:end_idx]))
        if check_enclaves == 1:
            LOGGER.debug('Detected subshapes: %s', str(sub_shapes))
            LOGGER.debug('of which detected enclaves: %s', str(len(enclave_paths)))
        else:
            LOGGER.info('Detected subshapes: %s. Enclave checking disabled.',
                        str(sub_shapes))
    else:
        sub_shape_path.append(path.Path(all_coords_shape))
    del all_coords_shape
    incl_coords = []
    for _, val in enumerate(sub_shape_path):
        add_points = _mask_from_path(val, resolution)
        if add_points is not None:
            [incl_coords.append(point) for point in add_points]
        del add_points
    # stdout.write('\n')
    if check_enclaves == 1 and not enclave_paths:
        excl_coords = []
        LOGGER.debug('Removing enclaves...')
        for _, val in enumerate(enclave_paths):
            temp_excl_points = _mask_from_path(val, resolution)
            if temp_excl_points is not None:
                [excl_coords.append(point) for point in temp_excl_points]
            del temp_excl_points
        excl_coords = set(tuple(row) for row in excl_coords)
        incl_coords = [point for point in incl_coords if point not in
                       excl_coords]
    LOGGER.debug('Successfully isolated coordinates from shape')
    total_bbox = np.array((min([x[0] for x in check_shape.points]),
                           min([x[1] for x in check_shape.points]),
                           max([x[0] for x in check_shape.points]),
                           max([x[1] for x in check_shape.points])))
    if points2check == []:
        all_coords = _litpop_box2coords(total_bbox, resolution, 1)
    else:
        all_coords = points2check
        del points2check
    incl_coords = set(incl_coords)
    mask = np.array([(coord in incl_coords) for coord in all_coords])
    mask = pd.arrays.SparseArray(mask, fill_value=0, dtype='bool_')
#    plt.figure()
#    l1, l2 = zip(*[x for n, x in enumerate(all_coords) if mask.values[n]==1])
#    plt.scatter(l1, l2)
#    _plot_shape_to_plot(check_shape)
    return mask

def _get_country_info(iso3):
    """Get country ISO alpha_3, country id (defined as country appearance
    order in natural earth shape file) and country's geometry.

    Parameters
    ----------
    iso3 : str
        country code of country to get

    Returns
    -------
    cntry_info : tuple
        (iso_num, country_name, country_shp)
    country_admin1 : list
        shapres and records of admin1 regions
    """
    shp = shapereader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
    shp = shapefile.Reader(shp)
    for field_num, field in enumerate(shp.fields[1::]):
        # Skip first (index zero) field, because it is DeletionFlag
        if field[0] == 'ADM0_A3':
            break
    del field
    for field_num2, field in enumerate(shp.fields[1::]):
        # Skip first (index zero) field, because it is DeletionFlag
        if field[0] == 'ADMIN':
            break
    del field
    for rec, rec_shp in zip(shp.records(), shp.shapes()):
        if rec[field_num] == iso3:
            country_shp = rec_shp
            country_name = rec[field_num2]
            break

    num_codes = [iso3 for iso3 in wb.country_codes if len(iso3) == 3]

    admin1_file = shapereader.natural_earth(resolution='10m',
                                            category='cultural',
                                            name='admin_1_states_provinces')
    admin1_recs = shapefile.Reader(admin1_file)
    country_admin1 = list()
    for rec, rec_shp in zip(admin1_recs.records(), admin1_recs.shapes()):
        if rec['adm0_a3'] == iso3:
            country_admin1.append([rec_shp, rec])
    try:
        iso_num = num_codes.index(iso3)
    except ValueError:
        iso_num = len(num_codes)
    cntry_info = [iso_num, country_name, country_shp]

    return cntry_info, country_admin1

def _bbox_union(bbox_in):
    bbox = np.zeros((4,))
    bbox[0] = min([val[0] for val in bbox_in])
    bbox[1] = min([val[1] for val in bbox_in])
    bbox[2] = max([val[2] for val in bbox_in])
    bbox[3] = max([val[3] for val in bbox_in])
    return bbox

def _get_iso3(country_name):
    """Find the ISO3 name corresponding to a country name. Can also be used
        to check if an ISO3 exists.

    Parameters:
        country_name (str): the country name to be checked.

    Returns:
        ISO3 (str): if the country name / ISO3 was found OR None (NoneType)
            otherwise
    """
    country_name = country_name.casefold()
    shp = shapereader.natural_earth(resolution='10m',
                                    category='cultural',
                                    name='admin_0_countries')
    shp = shapefile.Reader(shp)
    for field_num, field in enumerate(shp.fields[1::]):
        # Skip first (index zero) field, because it is DeletionFlag
        if field[0] == 'ADM0_A3':
            field_adm = field_num
            if 'field_name' in locals():
                break
        elif field[0] == 'ADMIN':
            field_name = field_num
            if 'field_adm' in locals():
                break
    del field
    for rec in shp.records():
        if len(country_name) == 3 and rec[field_adm].casefold() == country_name:
            return rec[field_adm]
        elif rec[field_name].casefold() == country_name:
            return rec[field_adm]
    return ""

def _get_gdp2asset_factor(cntry_info, ref_year, shp_file, fin_mode='income_group'):
    """Append factor to convert GDP to physcial asset values according to
        the Global Wealth Databook by the Credit Suisse Research Institute.
        Requires a pickled file containg a dictionary with the three letter
        country code as the key. The values are lists, each containg the
        country's name the factor for non-financial assets and the factor
        for financial assets (in this order).

    Parameters:
        cntry_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        ref_year (int): reference year
        fin_mode (str): define what total country economic value
                is to be used as an asset base and distributed to the grid:
                - gdp: gross-domestic product
                - income_group: gdp multiplied by country's income group+1
                - nfw: non-financial wealth (of households only)
                - tw: total wealth (of households only)
                - pc: produced capital
    """
    if fin_mode == 'income_group':
        for cntry_iso, cntry_val in cntry_info.items():
            _, inc_grp = income_group(cntry_iso, ref_year, shp_file)
            cntry_val.append(inc_grp + 1)
    elif fin_mode in ('gdp', 'pc', 'none', 'norm'):
        for cntry_iso, cntry_val in cntry_info.items():
            cntry_val.append(1)
    elif fin_mode in ('nfw', 'tw'):
        for cntry_iso, cntry_val in cntry_info.items():
            _, wealthtogdp_factor = wealth2gdp(cntry_iso, fin_mode == 'nfw', ref_year)
            if np.isnan(wealthtogdp_factor):
                LOGGER.warning("Missing factor for country %s.", cntry_iso)
                LOGGER.warning("Factor to convert GDP to assets will be set to 1.")
                wealthtogdp_factor = 1
            cntry_val.append(wealthtogdp_factor)
    else:
        LOGGER.error("invalid fin_mode")

def _gsdp_read(country_iso3, admin1_shape_data,
               look_folder=SYSTEM_DIR.joinpath('GSDP')):
    """Retrieves the GSDP data for a certain country. It requires an
        excel file in a subfolder "GSDP" in climadas data folder (or in the
        specified folder). The excel file should bear the name
        'ISO3_GSDP.xlsx' (or .xls), where ISO3 is the three letter country
        code. In the excel file, the first sheet should contain a row
        with the title "State_Province" with the name or postal code (two
        letters) of the admin1 unit and a row "GSDP_ref" with either the GDP
        value of the admin1 unit or its share in the national GDP.

    Parameters:
        country_iso3 (string): three letter country code
        admin1_shape_data (list): list containg all admin1 shapes of the
            country.
        look_folder (string): path where to look for file

    Returns:
        out_dict (dictionary): dictionary which contains the GSDP for each
            admin1 unit, where the name of the admin1 unit is the key.
    """
    file_name = _check_excel_exists(look_folder, str(country_iso3 + '_GSDP'))
    if file_name is not None:
        admin1_xls_data = pd.read_excel(file_name)
        if admin1_xls_data.get('State_Province') is None:
            admin1_xls_data = admin1_xls_data.rename(
                columns={admin1_xls_data.columns[0]: 'State_Province'})
        if admin1_xls_data.get('GSDP_ref') is None:
            admin1_xls_data = admin1_xls_data.rename(
                columns={admin1_xls_data.columns[-1]: 'GSDP_ref'})
#        prov = admin1_xls_data['State_Province'].tolist()
        out_dict = dict.fromkeys([nam[1]['name'] for nam in admin1_shape_data])
        postals = [nam[1]['postal'] for nam in admin1_shape_data]
        for subnat_shape in out_dict.keys():
            for idx, subnat_xls in enumerate(admin1_xls_data['State_Province'].tolist()):
                if _compare_strings_nospecchars(subnat_shape, subnat_xls):
                    out_dict[subnat_shape] = admin1_xls_data['GSDP_ref'][idx]
                    break
        # Now a second loop to detect empty ones
        for idx1, country_name in enumerate(out_dict.keys()):
            if out_dict[country_name] is None:
                for idx2, subnat_xls\
                in enumerate(admin1_xls_data['State_Province'].tolist()):
                    if _compare_strings_nospecchars(postals[idx1], subnat_xls):
                        out_dict[country_name] =\
                            admin1_xls_data['GSDP_ref'][idx2]
        return out_dict
    LOGGER.warning('No file for %s could be found in %s.', country_iso3, look_folder)
    LOGGER.warning('No admin1 data is calculated in this case.')
    return None

def _check_excel_exists(file_path, file_name, xlsx_before_xls=1):
    """Checks if an Excel file with the name file_name in the folder
    file_path exists, checking for both xlsx and xls files.

    Parameters:
        file_path (string): path where to look for file
        file_name (string): file name which is checked. Extension is ignored
        xlsx_before_xls (boolean): If set =1, xlsx files are priorised over
            xls files. Default=1.
    """
    try_ext = list()
    if xlsx_before_xls == 1:
        try_ext.append('.xlsx')
        try_ext.append('.xls')
    else:
        try_ext.append('.xls')
        try_ext.append('.xlsx')
    path_name = Path(file_path, file_name).stem
    for i in try_ext:
        if Path(file_path, path_name + i).is_file():
            return str(Path(file_path, path_name + i))
    return None

def _compare_strings_nospecchars(str1, str2):
    """Compares strings while ignoring non-alphanumeric and special
        characters.

    Parameters:
        str1 (string): string to be compared to str2
        str2 (string): string to be compared to str1
    Returns
        Boolean: True if the strings are the same, False otherwise.
    """
    import re
    if not isinstance(str1, str) or not isinstance(str2, str):
        LOGGER.warning('Invalid datatype (not strings), which cannot be '
                       'compared. Function will return exit and return false.')
        return False
    pattern = re.compile('[^a-z|A-Z|0-9| ]')  # ignore special
    cstr1 = re.sub(pattern, '', str1).casefold()
    cstr2 = re.sub(pattern, '', str2).casefold()
    return bool(cstr1 == cstr2)

def _plot_shape_to_plot(shp, gray_val=str(0.3)):
    """Plots a shape file to a pyplot.

    Parameters:
        shp (shapefile._Shape): shapefile to be plotted
        gray_val: (scalar): grayscale value of color line between zero and one.
            A value of zero corresponds to black and one to white.
    """
    gray_val = str(gray_val)
    parts = np.array(shp.parts)
    for i in range(0, len(parts) - 1):
        x_arr = np.array([x[0] for x in shp.points[parts[i]:parts[i + 1]]])
        y_arr = np.array([x[1] for x in shp.points[parts[i]:parts[i + 1]]])
        plt.plot(x_arr, y_arr, gray_val)
    x_arr = np.array([x[0] for x in shp.points[parts[len(parts) - 1]:]])
    y_arr = np.array([x[1] for x in shp.points[parts[len(parts) - 1]:]])
    plt.plot(x_arr, y_arr, gray_val)
    plt.show()

def _plot_paths_to_plot(list_of_paths, gray_val=str(0.3)):
    """Plot a path or paths to a pyplot

    Parameters:
        list of paths (list): paths to be plotted
        gray_val: (scalar): grayscale value of color line between zero and one.
            A value of zero corresponds to black and one to white.
    """
    gray_val = str(gray_val)
    for i in range(0, len(list_of_paths)):
        x_arr = np.array([x[0] for x in list_of_paths[i].vertices])
        y_arr = np.array([x[1] for x in list_of_paths[i].vertices])
        plt.plot(x_arr, y_arr, gray_val)
    plt.show()

def _plot_admin1_shapes(adm0_a3, gray_val=str(0.3)):
    """Retrieves the shape file or coordinate information of a country.

    Parameters:
        adm0_a3 (str): iso3 country code of country to get
        only_geo (boolean): Determines the output mode (default =0):
            if =0: returns the entire shape file of the country
            if =1: returns a tuple of values: bbox, lat, lon (see below)

    Returns:
        if only_geo = 0 (default):
            The shape of type shapefile._Shape
        if only_geo = 1
            bbox, lat, lon (tuple of size 3)
                bbox is a 1x4 vector of the bounding box of the country (array)
                lat is a mx1 vector of the latitudinal values of the vertices
                    of the shape (array)
                lon is a mx1 vector of the longitudinal values of the vertices
                    of the shape (array)
    """
    shp_file = shapereader.natural_earth('10m', category='cultural',
                                         name='admin_1_states_provinces')
    shp = shapefile.Reader(shp_file)
    del shp_file
    for field_num, field in enumerate(shp.fields[1::]):
        # Skip first (index zero) field, because it is DeletionFlag
        if field[0].casefold() == 'ADM0_A3'.casefold():
            break
    del field
    adm1_shapes = []
    for rec_i, rec in enumerate(shp.records()):
        if rec[field_num] == adm0_a3:
            adm1_shapes.append(shp.shapes()[rec_i])
    for i in adm1_shapes:
        _plot_shape_to_plot(i, gray_val=gray_val)

def _calc_admin1(curr_country, country_info, admin1_info, litpop_data,
                 coords, resolution, adm1_scatter, conserve_cntrytotal=1,
                 check_plot=1, masks_adm1=None, return_data=1):
    """Calculates the LitPop on admin1 level for provinces/states where such
        information is available (i.e. GDP is distributed on a subnational
        instead of a national level). Requires excel files in a subfolder
        "GSDP" in climadas data folder. The excel files should contain a row
        with the title "State_Province" with the name or postal code (two
        letters) of the admin1 unit and a row "GSDP_ref" with either the GDP
        value or the share of the state in the national GDP.
        If only for certain states admin1 info is found, the rest of the
        country is assigned value according to the admin0 method.

    Parameters:
        curr_country (str): country code of country to get
        country_info (list): a list which contains information about the
            country (is produced in the .set_country procedure). GDP should be
            stored in index 3 and the factor to convert GDP to physical asset
            values is stored in position index 4.
        admin1_info (list): a list which contains information about the admin1
            level of the country (is produced in the .set_country procedure).
            It contains Shape files among others.
        litpop_data (pandas.arrays.SparseArray): The raw litpop_data to which the
            admin1 based value should be assinged.
        coords (list): a list containing all the coordinates of the country in
            the format (lon, lat)
        resolution (scalar): the desired resolution in arc-seconds.
        adm1_scatter (boolean): whether a scatter plot and correlation comparing admin0 and
            admin1 results should be produced.
        conserve_cntrytotal (boolean): if True, final LitPop is normalized with country value

    Returns:
        litpop_data (pandas.arrays.SparseArray): The litpop_data the sum of which
            corresponds to the GDP multiplied by the GDP2Asset conversion
            factor.
    """
# TODO: if a state/province has GSDP value, but no coordinates inside,
#    the final total value is off (e.g. Basel Switzerland at 300 arcsec).
#    Potential fix: normalise the value in the end
    gsdp_data = _gsdp_read(curr_country, admin1_info)
    litpop_data = _normalise_litpop(litpop_data)
    if gsdp_data is not None:
        sum_vals = sum(filter(None, gsdp_data.values()))
        gsdp_data = {key: (value / sum_vals if value is not None else None)
                     for (key, value) in gsdp_data.items()}
        if None not in gsdp_data.values():
            # standard loop if all GSDP data is available
            temp_adm1 = {'adm0_LitPop_share': [], 'adm1_LitPop_share': []}
            for idx3, adm1_shp in enumerate(admin1_info):
                # start_time = time.time()
                LOGGER.debug('Caclulating admin1 for %s.', adm1_shp[1]['name'])
                if not masks_adm1:
                    mask_adm1 = _mask_from_shape(adm1_shp[0],
                                                 resolution=resolution,
                                                 points2check=coords)
                    shr_adm0 = sum(litpop_data[mask_adm1])
                else:
                    shr_adm0 = sum(litpop_data[masks_adm1[idx3]])
                temp_adm1['adm0_LitPop_share'].append(shr_adm0)
                temp_adm1['adm1_LitPop_share'].append(list(gsdp_data.values())[idx3])
                # LitPop in the admin1-unit is scaled by ratio of admin
                if shr_adm0 > 0:
                    mult = country_info[3]\
                        * country_info[4]\
                        * gsdp_data[adm1_shp[1]['name']] / shr_adm0
                else:
                    mult = 0
                if return_data:
                    if not masks_adm1:
                        litpop_data = pd.arrays.SparseArray(
                            [val * mult if mask_adm1[idx] == 1 else val
                             for idx, val in enumerate(litpop_data.to_numpy())],
                            fill_value=0)
                    else:
                        litpop_data = pd.arrays.SparseArray(
                            [val * mult if masks_adm1[idx3][idx] == 1 else val
                             for idx, val in enumerate(litpop_data.to_numpy())],
                            fill_value=0)
        else:
            temp_adm1 = {'mask': [], 'adm0_LitPop_share': [],
                         'adm1_LitPop_share': [], 'LitPop_sum': []}
            litpop_data = _calc_admin0(litpop_data, country_info[3],
                                       country_info[4])
            sum_litpop = sum(litpop_data.sp_values)
            for idx3, adm1_shp in enumerate(admin1_info):
                if not masks_adm1:
                    mask_adm1 = _mask_from_shape(adm1_shp[0],
                                                 resolution=resolution,
                                                 points2check=coords)
                else:
                    mask_adm1 = masks_adm1[idx3]
                temp_adm1['mask'].append(mask_adm1)
                temp_adm1['LitPop_sum'].append(sum(litpop_data[mask_adm1]))
                temp_adm1['adm0_LitPop_share'].append(sum(litpop_data[mask_adm1])
                                                      / sum_litpop)
            del mask_adm1
            sum_litpop_adm1 = sum([
                sum(litpop_data[temp_adm1['mask'][n1]])
                for n1, val in enumerate(gsdp_data.values()) if val is not None
            ])
            admin1_share = sum_litpop_adm1 / sum_litpop
            for idx2, val in enumerate(gsdp_data.values()):
                if val is not None:
                    LOGGER.debug('Calculating admin1 data for %s.',
                                 admin1_info[1][idx2].attributes['name'])
                    mult = (val * admin1_share * (country_info[3] * country_info[4])
                            / temp_adm1['LitPop_sum'][idx2])
                    temp_mask = temp_adm1['mask'][idx2]
                    if return_data:
                        litpop_data = pd.arrays.SparseArray(
                            [val1 * mult if temp_mask[idx] == 1 else val1
                             for idx, val1 in enumerate(litpop_data.to_numpy())])

                else:
                    LOGGER.warning('No admin1 data found for %s.',
                                   admin1_info[1][idx2].attributes['name'])
                    LOGGER.warning('Only admin0 data is calculated in this case.')
            for idx5, _ in enumerate(admin1_info):
                temp_adm1['adm1_LitPop_share'].append(list(gsdp_data.values())[idx5])
                # LP_sum = sum(litpop_data.values)
                # temp_adm1['adm1_LitPop_share'].append(sum(litpop_data.values\
                #        [temp_adm1['mask'][idx5].values])/LP_sum)
        if adm1_scatter == 1:
            pearsonr, spearmanr, rmse, rmsf = _litpop_scatter(temp_adm1['adm0_LitPop_share'],
                                                              temp_adm1['adm1_LitPop_share'],
                                                              admin1_info, check_plot)
    elif return_data:
        litpop_data = _calc_admin0(litpop_data, country_info[3],
                                   country_info[4])
    if conserve_cntrytotal and return_data:
        litpop_data = _normalise_litpop(litpop_data) * country_info[3] * country_info[4]
    if not return_data:
        litpop_data = []
    if adm1_scatter:
        return litpop_data, [pearsonr, spearmanr, rmse, rmsf], \
                temp_adm1['adm0_LitPop_share'], temp_adm1['adm1_LitPop_share']
    return litpop_data

def _calc_admin0(litpop_data, total_asset_val, gdptoasset_factor):
    """Calculates the LitPop on a national level. The total value distributed
        corresponds to GDP times the factor to convert GDP to assets from
        the Gloabl Wealth Databook by the Credit Suisse Research Institute.

    Parameters:
        litpop_data (pandas.arrays.SparseArray): The raw litpop_data to which the
            admin0 based value should be assinged.
        total_asset_val (scalar): The total asset value of the country.
        gdptoasset_factor (scalar): The factor with which GDP can be converted
            to physical asset value.

    Returns:
        litpop_data (pandas.arrays.SparseArray): The litpop_data the sum of which
            corresponds to the GDP multiplied by the GDP2Asset conversion
            factor.
    """
    return _normalise_litpop(litpop_data) * total_asset_val * gdptoasset_factor

def _normalise_litpop(litpop_data):
    """Normailses LitPop data, such that its total sum equals to one.

    Parameters:
        litpop_data (pandas.arrays.SparseArray): The litpop_data which sjould be
            normalised.

    Returns:
        litpop_data (pandas.arrays.SparseArray): The litpop_data the sum of which
            corresponds to one.
    """
    if not isinstance(litpop_data, pd.arrays.SparseArray):
        LOGGER.error('LitPop data is not of expected type (Pandas '
                     'SparseArray). Operation aborted.')
        raise TypeError

    sum_all = sum(litpop_data.sp_values)
    return litpop_data / sum_all

def _check_bbox_country_cut_mode(country_cut_mode, cut_bbox, country_adm0):
    """Checks whether a bounding box is valid an compatible with the chosen
        country cut mode.

        Parameters:
            country_cut_mode (scalar): the chosen country cut mode.
            cut_bbox (4x1 array): the bounding box, ESRI style.
            country_adm0 (string): three letter country code.

        Returns:
            cut_bbox (4x1 array): the bounding box, corrected if necessary.
    """
    if (country_adm0 is not None) & (country_cut_mode == 1) & (cut_bbox is not None):
        cut_bbox = _get_country_shape(country_adm0, 1)[0]
        LOGGER.warning('Custom bounding box overwritten in chosen country cut mode.')
    elif (country_adm0 is not None) & (country_cut_mode == 1) & (cut_bbox is None):
        cut_bbox = _get_country_shape(country_adm0, 1)[0]
    if (country_cut_mode != 1) & (cut_bbox is not None):
        try:
            cut_bbox = np.array(cut_bbox)
            if not isinstance(cut_bbox, np.ndarray) and not np.size(cut_bbox) == 4:
                LOGGER.warning('Invalid bounding box provided. \
                               Bounding box ignored. Please ensure the \
                               bounding box is an array like type of \
                               dimension 1x4')
                cut_bbox = None
            else:
                if (cut_bbox[0] > cut_bbox[2]) or \
                                                (cut_bbox[1] > cut_bbox[3]):
                    LOGGER.warning('Invalid bounding box provided. \
                                   Bounding box ignored. Please make sure \
                                   that the layout of the bounding box is \
                                   (Min_Longitude, Min_Latitude, \
                                    Max_Longitude, Max_Latitude).')
                    cut_bbox = None
        except TypeError:
            LOGGER.warning('Invalid bounding box provided. Bounding box \
                           ignored. Please ensure the bounding box is an \
                           array like type.')
            cut_bbox = None
    return cut_bbox

def _litpop_scatter(adm0_data, adm1_data, adm1_info, check_plot=True):
    """Plots the admin0 share of the states and provinces against the admin1
        shares.

    Parameters:
        adm0_data (list): list containing the admin0 shares
        adm1_data (list): list containing the admin1 shares
        adm1_info (list): list containing the shape files of the admin1 items.
    """
    adm0_data = np.array(adm0_data)
    adm1_data = np.array(adm1_data)
    inter = np.intersect1d(np.nonzero(adm1_data), np.nonzero(adm0_data))
    adm1_data = adm1_data[inter].astype(float)
    adm0_data = adm0_data[inter].astype(float)
    # Correlation coefficients:
    spearmanr = stats.spearmanr(adm0_data, adm1_data)[0]
    pearsonr = stats.pearsonr(adm0_data, adm1_data)[0]
    # Root mean square error:
    rmse = (sum((adm0_data - adm1_data)**2))**.5
    # Relative root mean square error:
    # rrmse = (sum(((adm0_data-adm1_data)/adm1_data)**2))**.5
    # Root mean squared fraction:
    rmsf = np.exp(np.sqrt(np.sum((np.log(adm0_data / adm1_data))**2) / adm0_data.shape[0]))
    if check_plot:
        plt.figure()
        plt.scatter(adm1_data, adm0_data, c=(0.1, 0.1, 0.3))
#        plt.suptitle('Comparison of admin0 and admin1 LitPop data for '\
#                  + adm1_info[0].attributes['admin'])
        plt.plot([0, np.max([plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]])],
                 [0, np.max([plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]])],
                 ls="--", c=".3")
    #    plt.annotate(label, xy=(adm0_data, adm1_data), xytext=(-20, 20),
    #        textcoords='offset points', ha='right', va='bottom')
        plt.suptitle(adm1_info[1][0].attributes['admin'] + ': rp='
                     + format(pearsonr, '.2f') + ', rs='
                     + format(spearmanr, '.2f'), fontsize=18)
        plt.xlabel('Reference GDP share')
        plt.ylabel('Modelled GDP share')
        plt.show()
    return pearsonr, spearmanr, rmse, rmsf

def read_bm_file(bm_path, filename):
    """Reads a single NASA BlackMarble GeoTiff and returns the data. Run all
        required checks first.

    Parameters
    ----------
    bm_path : str
        absolute path where files are stored.
    filename : str
        filename of the file to be read.

    Returns
    -------
    arr1 : array
        Raw BM data
    curr_file : gdal GeoTiff File
        Additional info from which coordinates can be calculated.
    """
    try:
        LOGGER.debug('Importing %s.', Path(bm_path, filename))
        curr_file = gdal.Open(str(Path(bm_path, filename)))
        band1 = curr_file.GetRasterBand(1)
        arr1 = band1.ReadAsArray()
        del band1
        return arr1, curr_file
    except:
        LOGGER.error('Failed: Importing %s', str(curr_file))
        raise

def get_bm(required_files=np.ones(np.count_nonzero(BM_FILENAMES),),
           **parameters):
    """Reads data from NASA GeoTiff files and cuts out the data along a chosen
        bounding box. Call this after the functions
        nightlight.required_nl_files and nightlight.check_nl_local_file_exists
        have ensured which files are required and which ones exist and missing
        files have been donwloaded.

    Parameters
    ----------
    required_files : 8x1 array
        boolean values which designates which BM files are required.
        Can be generated by the function nightlight.check_required_nl_files
    cut_bbox : 1x4 array-like, optional
        Bounding box (ESRI type) to be cut out.
        The layout of the bounding box corresponds to the bounding box
        of the ESRI shape files and is as follows:
        [minimum longitude, minimum latitude, maximum longitude, maxmimum latitude]
    country_adm0 : str, optional
        Country Code of the country of interest.
    country_cut_mode : int, optional
        Defines how the country is cut out:
        if 0: the country is only cut out with a bounding box
        if 1: the country is cut out along it's borders
        Default: = 1
    bm_path : str, optional
        absolute path where files are stored.
    resolution : int, optional
        the resolution in arcsec in which the data output
        is created.
    return_coords : boolean, optional
        Determines whether latitude and longitude
        are delievered along with gpw data (1) or only bm_data is
        returned (1) (Default: 0)
    reference_year : int
        Default = 2016

    Returns
    -------
    nightlight_intensity : pandas.arrays.SparseArray
        BM data
    lon : list
        list with longitudinal infomation on the GPW data. Same
        dimensionality as tile_temp (only returned if return_coords=1)
    lat : list
        list with latitudinal infomation on the GPW data. Same
        dimensionality as tile_temp (only returned if return_coords=1)
    """
    # Potential TODO: put cutting before zooming (faster), but with expanding
    # bbox in order to preserve additional pixels for interpolation...

    bm_path = parameters.get('file_path', SYSTEM_DIR)
    resolution = parameters.get('resolution', 30)
    reference_year = parameters.get('reference_year', 30)
    _match_target_res(resolution)
    cut_bbox = parameters.get('cut_bbox')
    country_adm0 = parameters.get('country_adm0')
    if country_adm0 is None:
        country_crop_mode = 0
    else:
        country_crop_mode = parameters.get('country_crop_mode', 1)
    return_coords = parameters.get('return_coords', 0)
    cut_bbox = _check_bbox_country_cut_mode(country_crop_mode, cut_bbox, country_adm0)
    nightlight_temp = None
    file_count = 0
    zoom_factor = 15 / resolution  # Orignal resolution is 15 arc-seconds
    for num_i, _ in enumerate(BM_FILENAMES[::2]):
        # Due to concat, we have to anlayse the tiles in pairs otherwise the
        # data is concatenated in the wrong order
        arr1 = [None] * 2  # Prepopulate list
        for j in range(0, 2):
            # Loop which cycles through the two tiles in each "column"
            if required_files[num_i * 2 + j] == 0:
                continue
            else:
                file_count = file_count + 1
                arr1[j], curr_file = read_bm_file(
                    bm_path,
                    BM_FILENAMES[num_i * 2 + j] % min(BM_YEARS,
                                                      key=lambda x: abs(x - reference_year)))
                if zoom_factor != 1:
#                    LOGGER.debug('Resizing image according to chosen '\
#                                + 'resolution')
                    arr1[j] = to_sparse_dataframe(nd.zoom(arr1[j], zoom_factor, order=1))
                else:
                    arr1[j] = to_sparse_dataframe(arr1[j])
                if cut_bbox is not None:
                    arr1[j] = _bm_bbox_cutter(arr1[j], (num_i * 2) + j, cut_bbox, resolution)
                if file_count == 1:
                    # Now get the coordinates
                    geo_t = curr_file.GetGeoTransform()
                    rastsize_x, rastsize_y = curr_file.RasterXSize,\
                        curr_file.RasterYSize
                    minlon = geo_t[0]
                    minlat = geo_t[3] + rastsize_x * geo_t[4] + rastsize_y * geo_t[5]
                    maxlon = geo_t[0] + rastsize_x * geo_t[1] + rastsize_y * geo_t[2]
                    maxlat = geo_t[3]
                else:
                    geo_t = curr_file.GetGeoTransform()
                    # Now get the coordinates
                    rastsize_x, rastsize_y = curr_file.RasterXSize,\
                        curr_file.RasterYSize
                    minlon = min(minlon, geo_t[0])
                    # Only add if they extend the current bbox
                    minlat = min(minlat, geo_t[3] + rastsize_x * geo_t[4]
                                 + rastsize_y * geo_t[5])
                    maxlon = max(maxlon, geo_t[0] + rastsize_x * geo_t[1]
                                 + rastsize_y * geo_t[2])
                    maxlat = max(maxlat, geo_t[3])
                del curr_file
        if (arr1[0] is None) & (arr1[1] is None):
            continue
        elif (not arr1[0] is None) & (arr1[1] is None):
            arr1 = arr1[0]
        elif (arr1[0] is None) & (not arr1[1] is None):
            arr1 = arr1[1]
        elif (not arr1[0] is None) & (not arr1[1] is None):
            arr1 = pd.concat(arr1, 0)
        if nightlight_temp is None:
            nightlight_temp = arr1
        else:
            nightlight_temp = pd.concat((nightlight_temp, arr1), 1)
        del arr1
    # LOGGER.debug('Reducing to one dimension...')
    nightlight_intensity = pd.arrays.SparseArray(nightlight_temp.values
                                          .reshape((-1,), order='F'),
                                          dtype='float')
    del nightlight_temp
    if return_coords == 1:
        if cut_bbox is None:
            temp_bbox = np.array((minlon, minlat, maxlon, maxlat))
            lon, lat = _litpop_box2coords(temp_bbox, resolution)
        else:
            lon, lat = _litpop_box2coords(cut_bbox, resolution)
    if return_coords == 1:
        return nightlight_intensity, lon, lat
    else:
        try:
            out_bbox = np.array((minlon, minlat, maxlon, maxlat))
            return nightlight_intensity, out_bbox
        except NameError:
            return nightlight_intensity, None

def _bm_bbox_cutter(bm_data, curr_file, bbox, resolution):
    """Crops the imported blackmarble data to the bounding box to reduce
        memory foot print during import This is done for each of the eight
        Blackmarble tiles seperately, therefore, the function needs to know
        which file is currenlty being treated (curr_file).

    Optional parameters:
        bm_data (pandas.arrays.SparseArray or array): Imported BM data in gridded
            format
        curr_file (integer): the file which is currenlty being imported (out
            of all the eignt BM files) in zero indexing.
        bbox (array 4x1): Bounding box to which the data should be cropped.
        resolution (int): The resolution in arcsec with which the data is
            being imported.

    Returns:
        bm_data (pandas.arrays.SparseArray): Cropped BM data
    """
    fixed_source_resolution = resolution
    deg_per_pix = 1 / (3600 / fixed_source_resolution)
    minlat, maxlat, minlon, maxlon = bbox[1], bbox[3], bbox[0], bbox[2]
    minlat_tile, maxlat_tile, minlon_tile, maxlon_tile =\
        (-90) + (curr_file // 2 == curr_file / 2) * (90),\
        0 + (curr_file // 2 == curr_file / 2) * 90,\
        (-180) + (curr_file // 2) * 90, (-90) + (curr_file // 2) * 90
    if minlat > maxlat_tile or maxlat < minlat_tile\
            or minlon > maxlon_tile or maxlon < minlon_tile:
        LOGGER.warning('This tile does not contain any relevant data. \
                       Skipping file.')
        return pd.DataFrame()
    bbox_conv = np.array((minlon, minlat, maxlon, maxlat))
    col_min, row_min, col_max, row_max = \
        _litpop_coords_in_glb_grid(bbox_conv, resolution)
    minrow_tile, maxrow_tile, mincol_tile, maxcol_tile =\
        (curr_file // 2 != curr_file / 2) * 90 * (3600 / resolution),\
        90 * (3600 / resolution) + (curr_file // 2 != curr_file / 2) * 90\
        * (3600 / resolution), (curr_file // 2) * 90 * (3600 / resolution),\
        (3600 / resolution) * 90 + (curr_file // 2) * 90 * (3600 / resolution)
    row_min = max(row_min, minrow_tile) -\
        (curr_file // 2 != curr_file / 2) * (90) * (3600 / resolution)
    row_max = min(row_max, maxrow_tile) -\
        (curr_file // 2 != curr_file / 2) * (90) * (3600 / resolution)
    col_min = max(col_min, mincol_tile) - (curr_file // 2) * (90) * (3600 / resolution)
    col_max = min(col_max, maxcol_tile) - (curr_file // 2) * (90) * (3600 / resolution)

    if isinstance(bm_data, pd.DataFrame):
        bm_data = to_sparse_dataframe(bm_data.loc[row_min:row_max, col_min:col_max].values)
    else:
        row_max = min(row_max + 1, ((maxlat_tile - minlat_tile)
                                    - (deg_per_pix / 2)) * (1 / deg_per_pix))
        col_max = min(col_max + 1, ((maxlon_tile - minlon_tile)
                                    - (deg_per_pix / 2)) * (1 / deg_per_pix))
        bm_data = bm_data[row_min:row_max, col_min:col_max]
    return bm_data

def _get_box_blackmarble(cut_bbox, **args):
    """Reads data from NASA GeoTiff files and cuts out the data along a chosen
        bounding box.

    Parameters
    ----------
    cut_bbox : 1x4 array-like
        Bounding box (ESRI type) to be cut out.
        the layout of the bounding box corresponds to the bounding box
        of the ESRI shape files and is as follows:
        [minimum longitude, minimum latitude, maximum longitude, maxmimum latitude]
    gpw_path : str, optional
        absolute path where files are stored. If the files
        dont exist, they get saved there. Default: SYSTEM_DIR
    resolution : int, optional
        the resolution in arcsec in which the data output
        is created.
    return_coords : boolean, optional
        Determines whether latitude and longitude
        are delievered along with gpw data (1) or only bm_data is
        returned (1) (Default: 0)
    reference_year : int, optional
        Default: 2016

    Returns
    -------
    nightlight_intensity : pandas.arrays.SparseArray
        BM data
    lon : list
        list with longitudinal infomation on the GPW data. Same
        dimensionality as tile_temp (only returned if return_coords=1)
    lat : list
        list with latitudinal infomation on the GPW data. Same
        dimensionality as tile_temp (only returned if return_coords=1)
    """
    resolution = args.get('resolution', 30)
    reference_year = args.get('reference_year', 2016)
    return_coords = args.get('return_coords', 0)
    bm_path = args.get('bm_path', SYSTEM_DIR)
    # Determine required satellite files
    req_sat_files = nightlight.check_required_nl_files(cut_bbox)
    # Check existence of necessary files for BM-year:
    files_exist = nightlight.check_nl_local_file_exists(
        req_sat_files, bm_path, min(BM_YEARS, key=lambda x: abs(x - reference_year)))[0]
    # Download necessary files:
    if not np.array_equal(req_sat_files, files_exist):
        try:
            LOGGER.debug('Downloading %s', str(int(sum(req_sat_files) - sum(files_exist))))
            nightlight.download_nl_files(req_sat_files, files_exist,
                                         dwnl_path=bm_path,
                                         year=min(BM_YEARS, key=lambda x: abs(x - reference_year)))
        except:
            LOGGER.error('Could not download missing satellite data files. \
                     Operation aborted.')
            raise
    # Read corresponding files
    # LOGGER.debug('Reading and cropping necessary BM files.')
    nightlight_intensity = get_bm(req_sat_files, resolution=resolution,
                                  return_coords=0, cut_bbox=cut_bbox,
                                  bm_path=bm_path, reference_year=reference_year)[0]
    if return_coords == 1:
        lon = tuple((cut_bbox[0], 1 / (3600 / resolution)))
        lat = tuple((cut_bbox[1], 1 / (3600 / resolution)))
        return nightlight_intensity, lon, lat
    # TODO: ensure function is efficient if no coords are returned
    return nightlight_intensity

def admin1_validation(country, methods, exponents, **args):
    """Get LitPop based exposre for one country or multiple countries
    using values at reference year. If GDP or income
    group not available for that year, consider the value of the closest
    available year.

    Parameters:
        country (str): list of countries or single county as a
            string. Countries can either be country names ('France') or
            country codes ('FRA'), even a mix is possible in the list.
        methods_name (list of str), i.e.:
            - ['LitPop' for LitPop,
            - ['Lit', 'Pop'] for Lit and Pop,
            - ['Lit3'] for cube of night lights (Lit3)
        exponents (list of 2-vectors), same length as methods_name i.e.:
            - [[1, 1]] for LitPop,
            - [[1, 0], [0, 1]] for Lit and Pop,
            - [[3, 0]] for cube of night lights (Lit3)
    args: Keyword arguments. The following keywords are recognised:
        res_km (float, optional): approx resolution in km. Default: 1km.
        res_arcsec (float, optional): resolution in arc-sec. Overrides
            res_km if both are delivered
        check_plot (boolean, optional): choose if a plot is shown at the
            end of the operation.
    """
    res_km = args.get('res_km', 1)
    res_arcsec = args.get('res_arcsec', [])
    check_plot = args.get('check_plot', True)
    fin_mode = 'gdp'
    reference_year = 2015
    #        inherit_admin1_from_admin0 = args.get('inherit_admin1_from_admin0', 1)
    if res_arcsec == []:
        resolution = (res_km / DEF_RES_GPW_KM) * DEF_RES_GPW_ARCSEC
    else:
        resolution = res_arcsec
    _match_target_res(resolution)
    country_info = dict()
    admin1_info = dict()
    LOGGER.info('Preparing coordinates, nightlights, and gpw data at %s arcsec.',
                str(resolution))
    if isinstance(country, list):  # multiple countries
        LOGGER.error('No valid country chosen. Give country as string.')
        raise TypeError
    elif isinstance(country, str):  # One country
        country_list = list()
        country_list.append(country)
        country_new = _get_iso3(country)
        country_list[0] = country_new
        cut_bbox = _get_country_shape(country_list[0], 1)[0]

        country_info[country_list[0]], admin1_info[country_list[0]]\
            = _get_country_info(country_list[0])
    else:
        LOGGER.error('Country parameter data type not recognised. Operation aborted.')
        raise TypeError
    shp_file = shapereader.natural_earth(resolution='10m',
                                         category='cultural',
                                         name='admin_0_countries')
    shp_file = shapereader.Reader(shp_file)

    for cntry_iso, cntry_val in country_info.items():  # get GDP value for country
        _, gdp_val = gdp(cntry_iso, reference_year, shp_file)
        cntry_val.append(gdp_val)
    _get_gdp2asset_factor(country_info, reference_year, shp_file, fin_mode=fin_mode)
    curr_shp = _get_country_shape(country_list[0], 0)
    all_coords = _litpop_box2coords(cut_bbox, resolution, 1)
    mask = _mask_from_shape(curr_shp, resolution=resolution,
                            points2check=all_coords)

    # Get LitPop, Lit and Pop, etc:
    nightlights = _get_box_blackmarble(cut_bbox, reference_year=reference_year,
                                       resolution=resolution, return_coords=0)

    bm_temp = np.ones(nightlights.shape)
    # Lit = Lit + 1 if Population is included, c.f. int(exponents[1]>0):
    bm_temp[nightlights.sp_index.indices] = (np.array(nightlights.sp_values, dtype='uint16'))
    del nightlights

    nightlights0 = pd.arrays.SparseArray(bm_temp, fill_value=0)
    nightlights0 = nightlights0[mask.sp_index.indices]
    nightlights1 = pd.arrays.SparseArray(bm_temp + 1, fill_value=1)
    del bm_temp
    nightlights1 = nightlights1[mask.sp_index.indices]

    gpw = gpw_import.get_box_gpw(cut_bbox=cut_bbox, resolution=resolution,
                                 return_coords=0, reference_year=reference_year)
    gpw = gpw[mask.sp_index.indices]

    lon, lat = zip(*np.array(all_coords)[mask.sp_index.indices])
    LOGGER.debug('Caclulating admin1 masks...')
    masks_adm1 = dict()
    for idx, adm1_shp in enumerate(admin1_info[country_list[0]]):
        masks_adm1[idx] = _mask_from_shape(adm1_shp[0], resolution=resolution,
                                           points2check=list(zip(lon, lat)))
    n_scores = 4
    rho = np.zeros(len(methods) * n_scores)
    adm0 = dict()
    adm1 = dict()
    LOGGER.info('Loop through methods...')
    for i in np.arange(0, len(methods)):
        LOGGER.info('%s :', methods[i])
        if exponents[i][1] == 0:  # Lit only, use Lit in [0, 255]
            _data = _LitPop_multiply(nightlights0, gpw, exponents=exponents[i])
        else:  # Pop is used, use Lit+1 in [1, 256]
            _data = _LitPop_multiply(nightlights1, gpw, exponents=exponents[i])
        _, rho[i * n_scores:(i * n_scores) + n_scores], adm0[methods[i]], adm1[methods[i]] = \
            _calc_admin1(country_list[0], country_info[country_list[0]],
                         admin1_info[country_list[0]], _data, list(zip(lon, lat)),
                         resolution, True, conserve_cntrytotal=0,
                         check_plot=check_plot, masks_adm1=masks_adm1, return_data=0)
    return rho, adm0, adm1


def exposure_set_admin1(exposure, res_arcsec):
    """add admin1 ID and name to exposure dataframe.

    Parameters:
        exposure: exposure instance
        res_arcsec: resolution in arc seconds, needs to match exposure resolution

    Returns:
        exposure: exposure instance with 2 extra columns: admin1 & admin1_ID
    """

    exposure['admin1'] = pd.Series()
    exposure['admin1_ID'] = pd.Series()
    count = 0
    for cntry in np.unique(exposure.region_id):
        _, admin1_info = _get_country_info(iso_cntry.get(cntry).alpha3)
        for adm1_shp in admin1_info:
            count = count + 1
            LOGGER.debug('Extracting admin1 for %s.', adm1_shp[1]['name'])
            mask_adm1 = _mask_from_shape(
                adm1_shp[0], resolution=res_arcsec,
                points2check=list(zip(exposure.longitude, exposure.latitude)))
            exposure.admin1_ID[mask_adm1] = adm1_shp[1][3]
            exposure.admin1[mask_adm1] = adm1_shp[1]['name']
    return exposure


def to_sparse_dataframe(ndarr):
    """Turns a 2-dim ndarray into a DataFrame with little memory footprint.

    Parameters
    ----------
    ndarr : numpy.ndarray
        2 dimensional
    
    Returns
    -------
    sparse dataframe : pandas.DataFrame
    """

    # in order to retain the low memory consumption of SparseArrays
    # it seems to be necessary to build the data frame from a dictionary of columns
    # and not just a mere list
    return pd.DataFrame(
        dict([
            (i, pd.arrays.SparseArray(ndarr[:,i]))
            for i in range(ndarr.shape[1])
        ])
    )
