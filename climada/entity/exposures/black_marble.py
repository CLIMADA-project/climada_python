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

Define BlackMarble class.
"""

__all__ = ['BlackMarble']

import logging
import math
import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy import ndimage
import shapely.vectorized
from cartopy.io import shapereader

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, INDICATOR_IMPF
from climada.entity.exposures.litpop import nightlight as nl_utils
from climada.util.constants import SYSTEM_DIR, DEF_CRS
from climada.util.finance import gdp, income_group
import climada.util.coordinates as u_coord

LOGGER = logging.getLogger(__name__)

DEF_RES_NOAA_KM = 1
"""Default approximate resolution for NOAA NGDC nightlights in km."""

DEF_RES_NASA_KM = 0.5
"""Default approximate resolution for NASA's nightlights in km."""

DEF_HAZ_TYPE = 'TC'
"""Default hazard type used in impact functions id."""

DEF_POLY_VAL = [0, 0, 1]
"""Default polynomial transformation used."""

class BlackMarble(Exposures):
    """Defines exposures from night light intensity, GDP and income group.
    Attribute region_id is defined as:
    - United Nations Statistics Division (UNSD) 3-digit equivalent numeric code
    - 0 if country not found in UNSD.
    - -1 for water
    """

    def set_countries(self, countries, ref_year=2016, res_km=None, from_hr=None,
                      admin_file='admin_0_countries', **kwargs):
        """ Model countries using values at reference year. If GDP or income
        group not available for that year, consider the value of the closest
        available year.

        Parameters:
            countries (list or dict): list of country names (admin0 or subunits)
                or dict with key = admin0 name and value = [admin1 names]
            ref_year (int, optional): reference year. Default: 2016
            res_km (float, optional): approx resolution in km. Default:
                nightlights resolution.
            from_hr (bool, optional): force to use higher resolution image,
                independently of its year of acquisition.
            admin_file (str): file name, admin_0_countries or admin_0_map_subunits
            kwargs (optional): 'gdp' and 'inc_grp' dictionaries with keys the
                country ISO_alpha3 code. 'poly_val' list of polynomial coefficients
                [1,x,x^2,...] to apply to nightlight (DEF_POLY_VAL used if not
                provided). If provided, these are used.
        """
        admin_key_dict = {'admin_0_countries': ['ADMIN', 'ADM0_A3'],
                          'admin_0_map_subunits': ['SUBUNIT', 'SU_A3']}

        shp_file = shapereader.natural_earth(resolution='10m',
                                             category='cultural',
                                             name=admin_file)
        shp_file = shapereader.Reader(shp_file)

        cntry_info, cntry_admin1 = country_iso_geom(countries, shp_file,
                                                    admin_key_dict[admin_file])
        fill_econ_indicators(ref_year, cntry_info, shp_file, **kwargs)

        nightlight, coord_nl, fn_nl, res_fact, res_km = get_nightlight(
            ref_year, cntry_info, res_km, from_hr)

        tag = Tag(file_name=fn_nl)
        bkmrbl_list = []

        for cntry_iso, cntry_val in cntry_info.items():

            bkmrbl_list.append(
                self._set_one_country(cntry_val, nightlight, coord_nl, res_fact, res_km,
                                      cntry_admin1[cntry_iso], **kwargs).gdf)
            tag.description += ("{} {:d} GDP: {:.3e} income group: {:d} \n").\
                format(cntry_val[1], cntry_val[3], cntry_val[4], cntry_val[5])

        Exposures.__init__(
            self,
            data=Exposures.concat(bkmrbl_list).gdf,
            crs=DEF_CRS,
            ref_year=ref_year,
            tag=tag,
            value_unit='USD'
        )

        rows, cols, ras_trans = u_coord.pts_to_raster_meta(
            (self.gdf.longitude.min(), self.gdf.latitude.min(),
             self.gdf.longitude.max(), self.gdf.latitude.max()),
            (coord_nl[0, 1], -coord_nl[0, 1])
        )
        self.meta = {'width': cols, 'height': rows, 'crs': self.crs, 'transform': ras_trans}

    @staticmethod
    def _set_one_country(cntry_info, nightlight, coord_nl, res_fact,
                         res_km, admin1_geom, **kwargs):
        """Model one country.

        Parameters:
            cntry_info (lsit): [cntry_id, cnytry_name, cntry_geometry,
                ref_year, gdp, income_group]
            nightlight (np.array): nightlight in 30arcsec ~ 1km resolution.
                Row latitudes, col longitudes
            coord_nl (np.array): nightlight coordinates: [[min_lat, lat_step],
                [min_lon, lon_step]]
            res_fact (float): resampling factor
            res_km (float): wished resolution in km
            admin1_geom (list): list of admin1 geometries to filter
            poly_val (list): list of polynomial coefficients to apply to
                nightlight
        """
        LOGGER.info('Processing country %s.', cntry_info[1])

        if 'poly_val' in kwargs:
            poly_val = kwargs['poly_val']
        else:
            poly_val = DEF_POLY_VAL
        geom = cntry_info[2]

        nightlight_reg, lat_reg, lon_reg, on_land = _cut_country(geom, nightlight, coord_nl)
        nightlight_reg = _set_econ_indicators(nightlight_reg, cntry_info[4],
                                              cntry_info[5], poly_val)
        if admin1_geom:
            nightlight_reg, lat_reg, lon_reg, geom, on_land = _cut_admin1(
                nightlight_reg, lat_reg, lon_reg, admin1_geom, coord_nl, on_land)

        LOGGER.info('Generating resolution of approx %s km.', res_km)
        nightlight_reg, lat_reg, lon_reg = _resample_land(geom, nightlight_reg,
                                                          lat_reg, lon_reg, res_fact, on_land)

        exp_bkmrb = BlackMarble(data={
            'value': np.asarray(nightlight_reg).reshape(-1,),
            'latitude': lat_reg,
            'longitude': lon_reg,
        })
        exp_bkmrb.gdf['region_id'] = cntry_info[0]
        exp_bkmrb.gdf[INDICATOR_IMPF] = 1

        return exp_bkmrb

def country_iso_geom(countries, shp_file, admin_key=['ADMIN', 'ADM0_A3']):
    """ Get country ISO alpha_3, country id (defined as the United Nations
    Statistics Division (UNSD) 3-digit equivalent numeric codes and 0 if
    country not found) and country's geometry shape.

    Parameters
    ----------
    countries : list or dict
        list of country names (admin0) or dict with key = admin0 name
        and value = [admin1 names]
    shp_file : cartopy.io.shapereader.Reader
        shape file
    admin_key: str
        key to find admin0 or subunit name

    Returns
    -------
    cntry_info : dict
        key = ISO alpha_3 country, value = [country id, country name, country geometry],
    cntry_admin1 : dict
        key = ISO alpha_3 country, value = [admin1 geometries]

    """
    countries_shp = {}
    list_records = list(shp_file.records())
    for info_idx, info in enumerate(list_records):
        countries_shp[info.attributes[admin_key[0]].title()] = info_idx

    cntry_info = dict()
    cntry_admin1 = dict()
    if isinstance(countries, list):
        countries = {cntry: [] for cntry in countries}
        admin1_rec = list()
    else:
        admin1_rec = shapereader.natural_earth(resolution='10m',
                                               category='cultural',
                                               name='admin_1_states_provinces')
        admin1_rec = shapereader.Reader(admin1_rec)
        admin1_rec = list(admin1_rec.records())

    for country_name, prov_list in countries.items():
        country_idx = countries_shp.get(country_name.title())
        if country_idx is None:
            options = [country_opt for country_opt in countries_shp
                       if country_name.title() in country_opt]
            if not options:
                options = list(countries_shp.keys())
            raise ValueError('Country %s not found. Possible options: %s'
                             % (country_name, options))
        iso3 = list_records[country_idx].attributes[admin_key[1]]
        try:
            cntry_id = u_coord.country_to_iso(iso3, "numeric")
        except LookupError:
            cntry_id = 0
        cntry_info[iso3] = [cntry_id, country_name.title(),
                            list_records[country_idx].geometry]
        cntry_admin1[iso3] = _fill_admin1_geom(iso3, admin1_rec, prov_list)

    return cntry_info, cntry_admin1

def fill_econ_indicators(ref_year, cntry_info, shp_file, **kwargs):
    """Get GDP and income group per country in reference year, or it closest
    one. Source: world bank. Natural earth repository used when missing data.
    Modifies country info with values [country id, country name,
    country geometry, ref_year, gdp, income_group].

    Parameters:
        ref_year (int): reference year
        cntry_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        kwargs (optional): 'gdp' and 'inc_grp' dictionaries with keys the
            country ISO_alpha3 code. If provided, these are used
    """
    for cntry_iso, cntry_val in cntry_info.items():
        cntry_val.append(ref_year)
        if 'gdp' in kwargs and kwargs['gdp'][cntry_iso] != '':
            gdp_val = kwargs['gdp'][cntry_iso]
        else:
            _, gdp_val = gdp(cntry_iso, ref_year, shp_file)
        cntry_val.append(gdp_val)

        if 'inc_grp' in kwargs and kwargs['inc_grp'][cntry_iso] != '':
            inc_grp = kwargs['inc_grp'][cntry_iso]
        else:
            _, inc_grp = income_group(cntry_iso, ref_year, shp_file)
        cntry_val.append(inc_grp)

def get_nightlight(ref_year, cntry_info, res_km=None, from_hr=None):
    """Obtain nightlight from different sources depending on reference year.
    Compute resolution factor used at resampling depending on source.

    Parameters:
        ref_year (int): reference year
        cntry_info (dict): key = ISO alpha_3 country, value = [country id,
            country name, country geometry]
        res_km (float): approx resolution in km.
        from_hr (bool, optional):
    Returns:
        nightlight (sparse.csr_matrix), coord_nl (np.array), fn_nl (str),
        res_fact (float)
    """
    if from_hr is None and ref_year > 2013:
        from_hr = True
    elif from_hr is None and ref_year <= 2013:
        from_hr = False

    if from_hr:
        if not res_km:
            res_km = 0.5
        nl_year = ref_year
        if ref_year > 2013:
            nl_year = 2016
        else:
            nl_year = 2012
        LOGGER.info("Nightlights from NASA's earth observatory for year %s.",
                    str(nl_year))
        res_fact = DEF_RES_NASA_KM / res_km
        geom = [info[2] for info in cntry_info.values()]
        geom = shapely.ops.cascaded_union(geom)
        req_files = nl_utils.get_required_nl_files(geom.bounds)
        files_exist = nl_utils.check_nl_local_file_exists(req_files,
                                                             SYSTEM_DIR, nl_year)
        nl_utils.download_nl_files(req_files, files_exist, SYSTEM_DIR, nl_year)
        # nightlight intensity with 15 arcsec resolution
        nightlight, coord_nl = nl_utils.load_nightlight_nasa(geom.bounds,
                                                             req_files, nl_year)
        fn_nl = [file.replace('*', str(nl_year)) for idx, file
                 in enumerate(nl_utils.BM_FILENAMES) if req_files[idx]]
        fn_nl = ' + '.join(fn_nl)
    else:
        if not res_km:
            res_km = 1.0
        nl_year = ref_year
        if ref_year < 1992:
            nl_year = 1992
        elif ref_year > 2013:
            nl_year = 2013
        LOGGER.info("Nightlights from NOAA's earth observation group for year %s.",
                    str(nl_year))
        res_fact = DEF_RES_NOAA_KM / res_km
        # nightlight intensity with 30 arcsec resolution
        nightlight, coord_nl, fn_nl = nl_utils.load_nightlight_noaa(nl_year)

    return nightlight, coord_nl, fn_nl, res_fact, res_km

def _fill_admin1_geom(iso3, admin1_rec, prov_list):
    """Get admin1 polygons for each input province of country iso3.

    Parameters:
        iso3 (str): admin0 country name in alpha3
        admin1_rec (list): list of admin1 records
        prov_list (list): province names
    Returns:
        list(geometry)
    """
    prov_geom = list()

    for prov in prov_list:
        found = False
        for rec in admin1_rec:
            if prov == rec.attributes['name'] and \
            rec.attributes['adm0_a3'] == iso3:
                found = True
                prov_geom.append(rec.geometry)
                break
        if not found:
            options = [rec.attributes['name'] for rec in admin1_rec
                       if rec.attributes['adm0_a3'] == iso3]
            raise ValueError('%s not found. Possible provinces of %s are: %s'
                             % (prov, iso3, options))

    return prov_geom

def _cut_admin1(nightlight, lat, lon, admin1_geom, coord_nl, on_land):
    """Cut nightlight image on box containing all the admin1 territories.

    Parameters:
        nightlight (np.array): nightlight values
        lat (np.array): latitude values in meshgrid
        lon (np.array): longitude values in meshgrid
        admin1_geom (list(shapely.geometry)): all admin1 geometries
        coord_nl (np.array): nightlight coordinates: [[min_lat, lat_step],
            [min_lon, lon_step]]
        on_land (np.array): array with true values in land points. same size
            as nightlight, lat, lon

    Returns:
        nightlight_reg, lat_reg, lon_reg (2d arrays with nightlight values,
        and coordinates in a square containing the admin1)
        on_land_reg (2d array of same size as previous with True values on land
        points)


    """
    all_geom = shapely.ops.cascaded_union(admin1_geom)

    in_lat = (math.floor((all_geom.bounds[1] - lat[0, 0]) / coord_nl[0, 1]),
              math.ceil((all_geom.bounds[3] - lat[0, 0]) / coord_nl[0, 1]))
    in_lon = (math.floor((all_geom.bounds[0] - lon[0, 0]) / coord_nl[1, 1]),
              math.ceil((all_geom.bounds[2] - lon[0, 0]) / coord_nl[1, 1]))

    nightlight_reg = nightlight[in_lat[0]:in_lat[-1] + 1, :][:, in_lon[0]:in_lon[-1] + 1]
    nightlight_reg[nightlight_reg < 0.0] = 0.0

    lat_reg, lon_reg = np.mgrid[lat[0, 0] + in_lat[0] * coord_nl[0, 1]:
                                lat[0, 0] + in_lat[1] * coord_nl[0, 1]:
                                complex(0, nightlight_reg.shape[0]),
                                lon[0, 0] + in_lon[0] * coord_nl[1, 1]:
                                lon[0, 0] + in_lon[1] * coord_nl[1, 1]:
                                complex(0, nightlight_reg.shape[1])]

    on_land_reg = on_land[in_lat[0]:in_lat[-1] + 1, :][:, in_lon[0]:in_lon[-1] + 1]

    return nightlight_reg, lat_reg, lon_reg, all_geom, on_land_reg

def _cut_country(geom, nightlight, coord_nl):
    """Cut nightlight image on box containing all the land.

    Parameters:
        geom (shapely.geometry): geometry of the region to consider
        nightlight (sparse.csr_matrix): nightlight values
        coord_nl (np.array): nightlight coordinates: [[min_lat, lat_step],
            [min_lon, lon_step]]

    Returns:
        nightlight_reg, lat_reg, lon_reg (2d arrays with nightlight values,
        and coordinates in a square containing the country)
        on_land_reg (2d array of same size as previous with True values on land
        points)
    """
    in_lat = (math.floor((geom.bounds[1] - coord_nl[0, 0]) / coord_nl[0, 1]),
              math.ceil((geom.bounds[3] - coord_nl[0, 0]) / coord_nl[0, 1]))
    in_lon = (math.floor((geom.bounds[0] - coord_nl[1, 0]) / coord_nl[1, 1]),
              math.ceil((geom.bounds[2] - coord_nl[1, 0]) / coord_nl[1, 1]))

    nightlight_reg = nightlight[in_lat[0]:in_lat[1] + 1, in_lon[0]:in_lon[-1] + 1] \
        .toarray()
    lat_reg, lon_reg = np.mgrid[coord_nl[0, 0] + in_lat[0] * coord_nl[0, 1]:
                                coord_nl[0, 0] + in_lat[1] * coord_nl[0, 1]:
                                complex(0, nightlight_reg.shape[0]),
                                coord_nl[1, 0] + in_lon[0] * coord_nl[1, 1]:
                                coord_nl[1, 0] + in_lon[1] * coord_nl[1, 1]:
                                complex(0, nightlight_reg.shape[1])]

    on_land_reg = np.zeros(lat_reg.shape, bool)
    try:
        iter(geom)
    except TypeError:
        geom = [geom]
    for poly in geom:
        in_lat = (math.floor((poly.bounds[1] - lat_reg[0, 0]) / coord_nl[0, 1]),
                  math.ceil((poly.bounds[3] - lat_reg[0, 0]) / coord_nl[0, 1]))
        in_lon = (math.floor((poly.bounds[0] - lon_reg[0, 0]) / coord_nl[1, 1]),
                  math.ceil((poly.bounds[2] - lon_reg[0, 0]) / coord_nl[1, 1]))
        on_land_reg[in_lat[0]:in_lat[1] + 1, in_lon[0]:in_lon[1] + 1] = (
            on_land_reg[in_lat[0]:in_lat[1] + 1, in_lon[0]:in_lon[1] + 1]
            | shapely.vectorized.contains(
                poly, lon_reg[in_lat[0]:in_lat[1] + 1, in_lon[0]:in_lon[1] + 1],
                lat_reg[in_lat[0]:in_lat[1] + 1, in_lon[0]:in_lon[1] + 1]))

    # put zero values outside country
    nightlight_reg[~on_land_reg] = 0.0

    return nightlight_reg, lat_reg, lon_reg, on_land_reg

def _resample_land(geom, nightlight, lat, lon, res_fact, on_land):
    """Model land exposures from nightlight intensities and normalized
    to GDP * (income_group + 1).

    Parameters:
        geom (shapely.geometry): geometry of the region to consider
        nightlight (np.array): nightlight values
        lat (np.array): latitude values in meshgrid
        lon (np.array): longitude values in meshgrid
        res_fact (float): resampling factor
        on_land (np.array): array with true values in land points. same size
            as nightlight, lat, lon

    Returns:
        nightlight_res, lat_res, lon_res (1d arrays with nightlight on land
        values and coordinates)
    """
    nightlight_res, lat_res, lon_res = nightlight, lat, lon
    if res_fact != 1.0:
        sum_val = nightlight.sum()
        nightlight_res = ndimage.zoom(nightlight, res_fact, mode='nearest')
        nightlight_res[nightlight_res < 0.0] = 0.0

        lat_res, lon_res = np.mgrid[
            lat[0, 0]: lat[-1, 0]: complex(0, nightlight_res.shape[0]),
            lon[0, 0]: lon[0, -1]: complex(0, nightlight_res.shape[1])]

        on_land = shapely.vectorized.contains(geom, lon_res, lat_res)

        nightlight_res[~on_land] = 0.0
        nightlight_res = nightlight_res / nightlight_res.sum() * sum_val

    return nightlight_res[on_land].ravel(), lat_res[on_land], lon_res[on_land]

def _set_econ_indicators(nightlight, gdp_val, inc_grp, poly_val):
    """Model land exposures from nightlight intensities and normalized
    to GDP * (income_group + 1).

    Parameters:
        nightlight (np.matrix): nightlight values
        gdp (float): GDP to interpolate in the region
        inc_grp (float): index to weight exposures in the region
        poly_val (list): list of polynomial coefficients to apply to nightlight

    Returns:
        np.array
    """
    if nightlight.sum() > 0:
        nightlight = polyval(np.asarray(nightlight), poly_val)
        nightlight = nightlight / nightlight.sum() * gdp_val * (inc_grp + 1)

    return nightlight
