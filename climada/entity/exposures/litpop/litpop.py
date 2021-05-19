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

Define LitPop class.
"""
import logging
import numpy as np
import rasterio
import geopandas
from shapely.geometry import Polygon, MultiPolygon

import climada.util.coordinates as u_coord
from climada.util.finance import gdp, income_group, wealth2gdp, world_bank_wealth_account
from climada.entity.tag import Tag
from climada.entity.exposures.litpop import nightlight as nl_util
from climada.entity.exposures.litpop import gpw_population as pop_util
from climada.entity.exposures.base import Exposures, INDICATOR_IMPF

LOGGER = logging.getLogger(__name__)

class LitPop(Exposures):
    """Defines exposure values from nightlight intensity (NASA), Gridded Population
        data (SEDAC); distributing produced capital (World Bank), population count,
        GDP (World Bank), or non-financial wealth (Global Wealth Databook by Credit Suisse
        Research Institute.)

        Calling sequence example:
        ent = LitPop()
        country_names = ['CHE', 'Austria']
        ent.set_countries(country_names)
        ent.plot()
    """

    def set_countries(self, countries, res_km=1, res_arcsec=None,
                    exponents=(1,1), fin_mode='pc', total_value=None,
                    admin1_calc=False, conserve_cntrytotal=True,
                    reference_year=2020, gpw_version=None, data_dir=None,
                    resample_first=True):
        """init LitPop exposure object for a list of countries (admin 0).
        Alias: set_country()

        Parameters
        ----------
        countries : list with str or int
            list containing country identifiers:
            iso3alpha (e.g. 'JPN'), iso3num (e.g. 92) or name (e.g. 'Togo')
        res_km : float (optional)
            Approx. horizontal resolution in km. Default: 1
        res_arcsec : float (optional)
            Horizontal resolution in arc-sec. Overrides res_km if both are provided.
            The default 30 arcsec corresponds to roughly 1 km.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop. To get
            nightlights^3 without population count: (3, 0).
            To use population count alone: (0, 1).
            Default: (1, 1)
        fin_mode : str, optional
            Socio-economic value to be used as an asset base that is disaggregated
            to the grid points within the country
            * 'pc': produced capital (Source: World Bank), incl. manufactured or
                    built assets such as machinery, equipment, and physical structures
                    (pc is in constant 2014 USD)
            * 'pop': population count (source: GPW, same as gridded population)
            * 'gdp': gross-domestic product (Source: World Bank)
            * 'income_group': gdp multiplied by country's income group+1
            * 'nfw': non-financial wealth (Source: Credit Suisse, of households only)
            * 'tw': total wealth (Source: Credit Suisse, of households only)
            * 'norm': normalized by country
            * 'none': LitPop per pixel is returned unchanged
            The default is 'pc'.
        total_value : list contaiuning numerics, same length as countries (optional)
            Total values to be disaggregated to grid in each country.
            The default is None. If None, the total number is extracted from other
            sources depending on the value of fin_mode.
        admin1_calc : boolean, optional
            If True, distribute admin1-level GDP (if available). Default: False
        conserve_cntrytotal : boolean, optional
            Given admin1_calc, conserve national total asset value. Default: True
        reference_year : int, optional
            Reference year for data sources. Default: 2020
        gpw_version : int (optional)
            Version number of GPW population data.
            The default is None. If None, the default is set in gpw_population module.
        data_dir : Path (optional)
            redefines path to input data directory. The default is SYSTEM_DIR.
        resample_first : boolean
            First resample nightlight (Lit) and population (Pop) data to target
            resolution before combining them as Lit^m * Pop^n?
            The default is True. Warning: Setting this to False affects the
            disaggregation results - expert choice only

        Raises
        ------
        ValueError
        """
        if isinstance(countries, str) or isinstance(countries, int):
            countries = [countries] # for backward compatibility

        # set res_arcsec if not given:
        if res_arcsec is None:
            if res_km is None:
                res_arcsec = 30
            else:
                res_arcsec = 30 * res_km
            LOGGER.info('Resolution is set to %i arcsec.', res_arcsec)
        if fin_mode is None:
            fin_mode = 'pc'
        # value_unit is set depending on fin_mode:
        if fin_mode in [None, 'none', 'norm']:
            value_unit = ''
        if fin_mode == 'pop':
            value_unit='people'
        else:
            value_unit = 'USD'
        if total_value is None:
            total_value = [None] * len(countries)
        elif len(total_value) != len(countries):
            LOGGER.error("'countries' and 'total_value' must be lists of same length")
            raise ValueError("'countries' and 'total_value' must be lists of same length")
        tag = Tag()

        # loop over countries: litpop is initiated for each individual polygon
        # within each country and combined at the end.
        litpop_list = \
            [self._set_one_country(country,
                                   res_arcsec=res_arcsec,
                                   exponents=exponents,
                                   fin_mode=fin_mode,
                                   total_value=total_value[idc],
                                   admin1_calc=admin1_calc,
                                   conserve_cntrytotal=conserve_cntrytotal,
                                   reference_year=reference_year,
                                   gpw_version=gpw_version,
                                   data_dir=data_dir,
                                   resample_first=resample_first)
             for idc, country in enumerate(countries)]
        # identified countries and those ignored:
        countries_in = \
            [country for i, country in enumerate(countries) if litpop_list[i] is not None]
        countries_out = \
            [country for i, country in enumerate(countries) if litpop_list[i] is None]
        if not countries_in:
            LOGGER.error('No valid country identified in %s, aborting.' % countries)
            raise ValueError('No valid country identified in %s, aborting.' % countries)
        if countries_out:
            LOGGER.warning('Some countries could not be identified and are ignored: '
                           '%s. Litpop only initiated for: %s'
                           % (countries_out, countries_in))

        tag.description = ('LitPop Exposure for %s at %i as, year: %i, financial mode: %s, '
                           'exp: [%i, %i]'
                           % (countries_in, res_arcsec, reference_year, fin_mode,
                              exponents[0], exponents[1]))

        Exposures.__init__(
            self,
            data=Exposures.concat(litpop_list).gdf,
            crs=litpop_list[0].crs,
            ref_year=reference_year,
            tag=tag,
            value_unit=value_unit
        )

        try:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (self.gdf.longitude.min(), self.gdf.latitude.min(),
                 self.gdf.longitude.max(), self.gdf.latitude.max()),
                u_coord.get_resolution(self.gdf.longitude, self.gdf.latitude))
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
        self.check()

    def set_custom_shape(self, shape, res_arcsec=30, exponents=(1,1), fin_mode=None,
                         total_value_abs=None, rel_value_share=1., in_countries=None,
                         region_id = None,
                         reference_year=2020, gpw_version=None, data_dir=None,
                         resample_first=True):
        """init LitPop exposure object for a custom shape.
        Requires user input regarding total value, either absolute ore relative
        combined with info which countries it is in or overlapping with.

        This method can be used to initiated LitPop Exposure for sub-national
        regions such as states, districts, cantons, cities, ... but shapes
        and info regarding total value need to be provided manually.
        If these required input parameters are not known / available,
        initiate Exposure for entire country and extract shape afterwards.

        Parameters
        ----------
        shape : shapely.geometry.Polygon or MultiPolygon
            geographical shape for which LitPop Exposure is to be initaited
        res_arcsec : float (optional)
            Horizontal resolution in arc-sec.
            The default 30 arcsec corresponds to roughly 1 km.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop.
        fin_mode : str, optional
            Socio-economic value to be used as an asset base that is disaggregated
            to the grid points within the country
            * 'pc': produced capital (Source: World Bank), incl. manufactured or
                    built assets such as machinery, equipment, and physical structures
                    (pc is in constant 2014 USD)
            * 'pop': population count (source: GPW, same as gridded population)
            * 'gdp': gross-domestic product (Source: World Bank)
            * 'income_group': gdp multiplied by country's income group+1
            * 'nfw': non-financial wealth (Source: Credit Suisse, of households only)
            * 'tw': total wealth (Source: Credit Suisse, of households only)
            * 'norm': normalized by country
            * 'none': LitPop per pixel is returned unchanged
            The default is 'pc'.
        total_value_abs : numeric (optional)
            Total value to be disaggregated to grid in shape.
            The default is None. If None, the total number is extracted from other
            sources depending on the value of fin_mode and in_countries.
        rel_value_share : float between 0. and 1. (optional)
            the relative share of the total value within the shape.
            e.g. if total_value_abs = 1000 and rel_value_share = 0.2, the total
            value that is disagreggated within the shape is 200 (= 1000 * 0.2).
            Important: This should be set manually by the user if total_value_abs
            is extracted from in_countries.
            The default is 1 (=100%)
        in_countries : str or list (optional)
            country or list of countries the shape is in or overlapping with.
            Porvide countries as ISO3alpha code, e.g. 'JPN' for Japan.
            If no total_value_abs is provided, total_value_abs is set as total 
            value of these countries. Required if total_value_abs is None.
            The default is None.
        region_id : int (optional)
            if given, sets region_id of Exposure manually, overwrites countries_in
            The default is None.
        reference_year : int, optional
            Reference year for data sources. Default: 2020
        gpw_version : int (optional)
            Version number of GPW population data.
            The default is None. If None, the default is set in gpw_population module.
        data_dir : Path (optional)
            redefines path to input data directory. The default is SYSTEM_DIR.
        resample_first : boolean
            First resample nightlight (Lit) and population (Pop) data to target
            resolution before combining them as Lit^m * Pop^n?
            The default is True. Warning: Setting this to False affects the
            disaggregation results.

        Raises
        ------
        ValueError
        NotImplementedError

        Returns
        -------
        LitPop Exposure instance
        """
        if total_value_abs is None and in_countries is None and fin_mode != 'pop':
            raise ValueError("Either total_value_abs or in_countries needs to "
                             "be provided (except if fin_mode is 'pop').")

        if isinstance(shape, MultiPolygon):
            shape_list = list(shape)
        elif isinstance(shape, Polygon):
            shape_list = [shape]
        elif isinstance(shape, list):
            shape_list = shape

        iso3a = list()
        iso3n = list()
        if isinstance(in_countries, str) or isinstance(in_countries, int):
            in_countries = [in_countries]
        if isinstance(in_countries, list):
            for country in in_countries:
                # Determine ISO 3166 representation of countries
                try:
                    iso3a.append(u_coord.country_to_iso(country, representation="alpha3"))
                    iso3n.append(u_coord.country_to_iso(country, representation="numeric"))
                except LookupError:
                    LOGGER.warning('Country not identified (ignored): %s.', country)
                    iso3n.append(0)

        if (total_value_abs is not None) and (in_countries is not None):
            LOGGER.info('total_value_abs is provided. Thus, in_countries is ignored '
                        'for calculation of total_value.')
            in_countries = None
        if fin_mode is None:
            fin_mode = 'pc'
        if fin_mode in [None, 'none', 'norm']:
            value_unit = ''
        elif fin_mode == 'pop':
            value_unit='people'
        else:
            value_unit = 'USD'

        tag = Tag()
        # set nightlight offset (delta) to 1 in case n>0, c.f. delta in Eq. 1 of paper:
        if exponents[1] == 0:
            offsets = (0, 0)
        else:
            offsets = (1, 0)

        # init LitPop GeoDataFrame for shape:
        total_population = 0
        litpop_gdf = geopandas.GeoDataFrame()
        for idx, polygon in enumerate(shape_list):
            # get litpop data for each polygon and combine into GeoDataFrame:
            _, meta_tmp, gdf_tmp = _get_litpop_single_polygon(polygon, reference_year,
                                                    res_arcsec, data_dir,
                                                    gpw_version, resample_first,
                                                    offsets, exponents,
                                                    return_gdf=True,
                                                    verbatim=not bool(idx),
                                                    )
            total_population += meta_tmp['total_population']
            litpop_gdf = litpop_gdf.append(gdf_tmp)
        litpop_gdf.crs = meta_tmp['crs']


        if fin_mode == 'pop' and total_value_abs is None:
            LOGGER.info('Total population in shape is extracted directly'
                        ' from pop data. Thus, in_countries is ignored.')
            in_countries = None
            total_value_abs = total_population
            if rel_value_share != 1:
                LOGGER.warning('Total population in shape is extracted directly'
                        ' from pop data. Thus, rel_value_share != 1 could be unintended(?)')
        # if required, total value is extracted for countries:
        elif isinstance(in_countries, list):
            total_value_abs = 0
            for country in iso3a:
                # add up total value of countries
                total_value_abs += \
                    get_total_value_per_country(country, fin_mode,
                                                    reference_year, 0)

        # the total value for disaggregation is computed by multiplying
        # absolute value and relative share – be careful setting rel_value_share:
        total_value = total_value_abs * rel_value_share

        # disaggregate total value proportional to LitPop values:
        if isinstance(total_value, float) or isinstance(total_value, int):
            litpop_gdf['value'] = np.divide(litpop_gdf['value'],
                                            litpop_gdf['value'].sum()) * total_value
        elif total_value is not None:
            raise TypeError("total_val_rescale must be int or float.")

        tag.description = ('LitPop Exposure for custom shape at %i as, year: %i, financial mode: %s, '
                           'exp: [%i, %i]'
                           % (res_arcsec, reference_year, fin_mode,
                              exponents[0], exponents[1]))

        litpop_gdf[INDICATOR_IMPF] = 1
        if region_id is not None:
            litpop_gdf['region_id'] = region_id
        elif len(iso3n) == 1:
            # single country
            litpop_gdf['region_id'] = iso3n[0]
        else:
            LOGGER.warning('region_id not set, please set manually')

        Exposures.__init__(
            self,
            data=litpop_gdf,
            crs=litpop_gdf.crs,
            ref_year=reference_year,
            tag=tag,
            value_unit=value_unit
        )

        try:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (self.gdf.longitude.min(), self.gdf.latitude.min(),
                 self.gdf.longitude.max(), self.gdf.latitude.max()),
                u_coord.get_resolution(self.gdf.longitude, self.gdf.latitude))
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
        self.check()

    @staticmethod
    def _set_one_country(country, res_arcsec=30, exponents=(1,1), fin_mode=None,
                         total_value=None, admin1_calc=False,  conserve_cntrytotal=True,
                         reference_year=2020, gpw_version=None, data_dir=None,
                         resample_first=True):
        """init LitPop exposure object for one single country

        Parameters
        ----------
        country : str or int
            country identifier:
            iso3alpha (e.g. 'JPN'), iso3num (e.g. 92) or name (e.g. 'Togo')
        res_arcsec : float (optional)
            Horizontal resolution in arc-sec.
            The default 30 arcsec corresponds to roughly 1 km.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop. To get
            nightlights^3 without population count: (3, 0). To use population count alone: (0, 1).
            Default: (1, 1)
        fin_mode : str, optional
            Socio-economic value to be used as an asset base that is disaggregated
            to the grid points within the country
            * 'pc': produced capital (Source: World Bank), incl. manufactured or
                    built assets such as machinery, equipment, and physical structures
                    (pc is in constant 2014 USD)
            * 'pop': population count (source: GPW, same as gridded population)
            * 'gdp': gross-domestic product (Source: World Bank)
            * 'income_group': gdp multiplied by country's income group+1
            * 'nfw': non-financial wealth (Source: Credit Suisse, of households only)
            * 'tw': total wealth (Source: Credit Suisse, of households only)
            * 'norm': normalized by country
            * 'none': LitPop per pixel is returned unchanged
            The default is 'pc'.
        total_value : numeric (optional)
            Total value to be disaggregated to grid in country.
            The default is None. If None, the total number is extracted from other
            sources depending on the value of fin_mode.
        admin1_calc : boolean, optional
            If True, distribute admin1-level GDP (if available). Default: False
        conserve_cntrytotal : boolean, optional
            Given admin1_calc, conserve national total asset value. Default: True
        reference_year : int, optional
            Reference year for data sources. Default: 2020
        gpw_version : int (optional)
            Version number of GPW population data.
            The default is None. If None, the default is set in gpw_population module.
        data_dir : Path (optional)
            redefines path to input data directory. The default is SYSTEM_DIR.
        resample_first : boolean
            First resample nightlight (Lit) and population (Pop) data to target
            resolution before combining them as Lit^m * Pop^n?
            The default is True. Warning: Setting this to False affects the
            disaggregation results.

        Raises
        ------
        ValueError

        Returns
        -------
        LitPop Exposure instance
        """
        # set nightlight offset (delta) to 1 in case n>0, c.f. delta in Eq. 1 of paper:
        if exponents[1] == 0:
            offsets = (0, 0)
        else:
            offsets = (1, 0)
    
        # Determine ISO 3166 representation of country and get geometry:
        try:
            iso3a = u_coord.country_to_iso(country, representation="alpha3")
            iso3n = u_coord.country_to_iso(country, representation="numeric")
        except LookupError:
            LOGGER.error('Country not identified: %s.', country)
            return None
        country_geometry = u_coord.get_land_geometry([iso3a])
        if not country_geometry.bounds: # check for empty shape
            LOGGER.error('No geometry found for country: %s.', country)
            return None
        LOGGER.info('LitPop: Init Exposure for country: %s, %i ...\n ' %(iso3a, iso3n))
        litpop_gdf = geopandas.GeoDataFrame()
        total_population = 0

        # for countries with multiple sperated shapes (e.g., islands), data
        # is initiated for each shape seperately and 0 values (e.g. on sea)
        # removed before combination, to save memory.
        # loop over single polygons in country shape object:
        for idx, polygon in enumerate(list(country_geometry)):
            # get litpop data for each polygon and combine into GeoDataFrame:
            _, meta_tmp, gdf_tmp = _get_litpop_single_polygon(polygon, reference_year,
                                                    res_arcsec, data_dir,
                                                    gpw_version, resample_first,
                                                    offsets, exponents,
                                                    return_gdf=True,
                                                    verbatim=not bool(idx),
                                                    )
            total_population += meta_tmp['total_population']
            litpop_gdf = litpop_gdf.append(gdf_tmp)
        litpop_gdf.crs = meta_tmp['crs']
        # set total value for disaggregation if not provided:
        if total_value is None: # default, no total value provided...
            total_value = get_total_value_per_country(iso3a, fin_mode,
                                                      reference_year, total_population)

        # disaggregate total value proportional to LitPop values:
        if isinstance(total_value, float) or isinstance(total_value, int):
            litpop_gdf['value'] = np.divide(litpop_gdf['value'],
                                            litpop_gdf['value'].sum()) * total_value
        elif total_value is not None:
            raise TypeError("total_val_rescale must be int or float.")

        exp_country = LitPop()
        exp_country.gdf = litpop_gdf
        exp_country.gdf[INDICATOR_IMPF] = 1
        exp_country.gdf['region_id'] = iso3n

        return exp_country

    # Alias method names for backward compatibility:
    set_country = set_countries

def _get_litpop_single_polygon(polygon, reference_year, res_arcsec, data_dir,
                               gpw_version, resample_first, offsets, exponents,
                               return_gdf=True, verbatim=False):
    """load nightlight (nl) and population (pop) data in rastered 2d arrays
    and apply rescaling (resolution reprojection) and LitPop core calculation.

    Parameters
    ----------
    polygon : Polygon object
        single shape to be extracted
    res_arcsec : float (optional)
        Horizontal resolution in arc-seconds.
    exponents : tuple of two integers, optional
        Defining power with which lit (nightlights) and pop (gpw) go into LitPop. To get
        nightlights^3 without population count: (3, 0). To use population count alone: (0, 1).
    reference_year : int, optional
        Reference year for data sources. Default: 2020
    gpw_version : int (optional)
        Version number of GPW population data.
        The default is None. If None, the default is set in gpw_population module.
    data_dir : Path (optional)
        redefines path to input data directory. The default is SYSTEM_DIR.
    resample_first : boolean
        First resample nightlight (Lit) and population (Pop) data to target
        resolution before combining them as Lit^m * Pop^n?
        The default is True. Warning: Setting this to False affects the
        disaggregation results.
    offsets : tuple of numbers
        offsets to be added to input data, required for LitPop core calculation

    Returns
    -------
    litpop_array : 2d np.ndarray
        resulting gridded data for Lit^m * Pop^n in bbox around polygon,
        data points outside the polygon are set to zero.
    meta_out : dict
        raster meta info for gridded data in litpop_array, additionally the field
        'total_population' contains the sum of population in the polygon.
    litpop_gdf : GeoDataframe
        resulting gridded data for Lit^m * Pop^n inside polygon,
        data points outside the polygon and equal to zero are not returned.
        litpop_gdf is None if return_gdf == False
    """
    # import population data (2d array), meta data, and global grid info,
    # global_transform defines the origin (corner points) of the global traget grid:
    pop, meta_pop, global_transform = \
        pop_util.load_gpw_pop_shape(polygon,
                                    reference_year,
                                    gpw_version=gpw_version,
                                    data_dir=data_dir,
                                    verbatim=verbatim,
                                    )
    total_population = pop.sum()
    # import nightlight data (2d array) and associated meta data:
    nl, meta_nl = nl_util.load_nasa_nl_shape(polygon,
                                             reference_year,
                                             data_dir=data_dir,
                                             )

    if resample_first: # default is True
        # --> resampling to target res. before core calculation
        target_res_arcsec = res_arcsec
    else: # resolution of pop is used for first resampling (degree to arcsec)
        target_res_arcsec = np.abs(meta_pop['transform'][0]) * 3600
    # resample Lit and Pop input data to same grid:
    [pop, nl], meta_out = resample_input_data([pop, nl],
                                              [meta_pop, meta_nl],
                                              i_ref=0, # pop defines grid
                                              target_res_arcsec=target_res_arcsec,
                                              global_origins=(global_transform[2], # lon
                                                              global_transform[5]), # lat
                                              )

    # calculate Lit^m * Pop^n (but not yet disaggregating any total value to grid):
    litpop_array = gridpoints_core_calc([nl, pop],
                                        offsets=offsets,
                                        exponents=exponents,
                                        total_val_rescale=None)
    if not resample_first: 
        # alternative option: resample to target resolution after core calc.:
        [litpop_array], meta_out = resample_input_data([litpop_array],
                                                       [meta_out],
                                                       target_res_arcsec=res_arcsec,
                                                       global_origins=(global_transform[2],
                                                                       global_transform[5]),
                                                       )
    meta_out['total_population'] = total_population
    if return_gdf:
        lon, lat = u_coord.raster_to_meshgrid(meta_out['transform'],
                                              meta_out['width'],
                                              meta_out['height'])
        gdf = geopandas.GeoDataFrame({'value': litpop_array.flatten()}, crs=meta_out['crs'],
                                     geometry=geopandas.points_from_xy(lon.flatten(),
                                                                       lat.flatten()))
        gdf['latitude'] = lat.flatten()
        gdf['longitude'] = lon.flatten()
        return litpop_array, meta_out, gdf[gdf['value'] != 0]
    return litpop_array, meta_out, None


def get_total_value_per_country(cntry_iso3a, fin_mode, reference_year, total_population=None):
    """
    Get total value for disaggregation, e.g., total asset value or population
    for a country, depending on unser choice (fin_mode).

    Parameters
    ----------
    cntry_iso3a : str
        country iso3 code alphabetic, e.g. 'JPN' for Japan
    fin_mode : str
        Socio-economic value to be used as an asset base that is disaggregated
        to the grid points within the country
        * 'pc': produced capital (Source: World Bank), incl. manufactured or
                built assets such as machinery, equipment, and physical structures
                (pc is in constant 2014 USD)
        * 'pc_land': produced capital (Source: World Bank), incl. manufactured or
                built assets such as machinery, equipment, physical structures,
                and land value for built-up land.
                (pc is in constant 2014 USD)
        * 'pop': population count (source: GPW, same as gridded population)
        * 'gdp': gross-domestic product (Source: World Bank)
        * 'income_group': gdp multiplied by country's income group+1
        * 'nfw': non-financial wealth (Source: Credit Suisse, of households only)
        * 'tw': total wealth (Source: Credit Suisse, of households only)
        * 'norm': normalized by country
        * 'none': LitPop per pixel is returned unscaled
        The default is 'pc'
    reference_year : int
        reference year for data extraction
    total_population : number, optional
        total population number, only required for fin_mode 'pop'.
        The default is None.

    Returns
    -------
    total_value : float
    """
    if fin_mode == 'none':
        return None
    elif fin_mode == 'pop':
        return total_population
    elif fin_mode == 'pc':
        return(world_bank_wealth_account(cntry_iso3a, reference_year, no_land=True)[1])
        # here, total_asset_val is Produced Capital "pc"
        # no_land=True returns value w/o the mark-up of 24% for land value
    elif fin_mode == 'pc_land':
        return(world_bank_wealth_account(cntry_iso3a, reference_year, no_land=False)[1])
        # no_land=False returns pc value incl. the mark-up of 24% for land value
    elif fin_mode == 'norm':
        return 1
    else: # GDP based total values:
        gdp_value = gdp(cntry_iso3a, reference_year)[1]
        if fin_mode == 'income_group': # gdp * (income group + 1)
            return gdp_value * (income_group(cntry_iso3a, reference_year)[1] + 1)
        elif fin_mode in ('nfw', 'tw'):
            wealthtogdp_factor = wealth2gdp(cntry_iso3a, fin_mode == 'nfw', reference_year)[1]
            if np.isnan(wealthtogdp_factor):
                LOGGER.warning("Missing wealth-to-gdp factor for country %s.", cntry_iso3a)
                LOGGER.warning("Using GDP instead as total value.")
                return gdp_value
            return gdp_value * wealthtogdp_factor
    raise ValueError(f"Unsupported fin_mode: {fin_mode}")

def resample_input_data(data_array_list, meta_list,
                        i_ref=0,
                        target_res_arcsec=None,
                        global_origins=(-180.0, 89.99999999999991),
                        target_crs=None,
                        resampling=None,
                        conserve=None):
    """
    Resamples all arrays in data_arrays to a given resolution –
    all based on the population data grid.

    Parameters
    ----------
    data_array_list : list or array of numpy arrays containing numbers
        Data to be resampled, i.e. list containing N (min. 1) 2D-arrays.
        The data with the reference grid used to define the global destination
        grid should be first in the list, e.g., pop (GPW population data)
        for LitPop.
    meta_list : list of dicts
        meta data dictionaries of data arrays in same order as data_array_list.
        Required fields in each dict are 'dtype,', 'width', 'height', 'crs', 'transform'.
        Example:
            {'driver': 'GTiff',
             'dtype': 'float32',
             'nodata': 0,
             'width': 2702,
             'height': 1939,
             'count': 1,
             'crs': CRS.from_epsg(4326),
             'transform': Affine(0.00833333333333333, 0.0, -18.175000000000068,
                                 0.0, -0.00833333333333333, 43.79999999999993)}
        The meta data with the reference grid used to define the global destination
        grid should be first in the list, e.g., GPW population data for LitPop.
    i_ref : int (optional)
        Index/Position of data set used to define the reference grid.
        The default is 0.
    target_res_arcsec : int (optional)
        target resolution in arcsec. The default is None, i.e. same resolution
        as reference data.
    global_origins : tuple with two numbers (lat, lon) (optional)
        global lon and lat origins as basis for destination grid.
        The default is the same as for GPW population data:
            (-180.0, 89.99999999999991)
    target_crs : rasterio.crs.CRS
        destination CRS
        The default is None, implying crs of reference data is used
        (e.g., CRS.from_epsg(4326) for GPW pop)
    resampling : resampling function (optional)
        The default is rasterio.warp.Resampling.bilinear
    conserve : str (optional), either 'mean' or 'sum'
        Conserve mean or sum of data? The default is None (no conservation).

    Returns
    -------
    data_array_list : list
        contains resampled data sets
    meta : dict
        contains meta data of new grid (same for all arrays)
    """

    # target resolution in degree lon,lat:
    if target_res_arcsec is None:
        res_degree = meta_list[i_ref]['transform'][0] # reference grid
    else:
        res_degree = target_res_arcsec / 3600 

    # loop over data arrays, do transformation where required:
    data_out_list = [None] * len(data_array_list)
    for idx, data in enumerate(data_array_list):
        # if target resolution corresponds to reference data resolution,
        # the reference data is not transformed:
        if idx==i_ref and ((target_res_arcsec is None) or (np.round(meta_list[i_ref]['transform'][0],
                            decimals=7)==np.round(res_degree, decimals=7))):
            data_out_list[idx] = data
            continue
        # reproject data grid:
        data_out_list[idx], meta = \
            u_coord.reproject_2d_grid(data, meta_list[idx], meta_list[i_ref], 
                              res_arcsec_out=target_res_arcsec,
                              global_origins=global_origins,
                              crs_out=target_crs,
                              resampling=rasterio.warp.Resampling.bilinear,
                              conserve=conserve,
                              buffer=0)
    return data_out_list, meta

def gridpoints_core_calc(data_arrays, offsets=None, exponents=None,
                         total_val_rescale=None):
    """
    Combines N dense numerical arrays by point-wise multipilcation and
    optionally rescales to new total value:
    (1) An offset (1 number per array) is added to all elements in
        the corresponding data array in data_arrays (optional).
    (2) Numbers in each array are taken to the power of the corresponding
        exponent (optional).
    (3) Arrays are multiplied element-wise.
    (4) if total_val_rescale is provided,
        results are normalized and re-scaled with total_val_rescale.
    (5) One array with results is returned.

    Parameters
    ----------
    data_arrays : list or array of numpy arrays containing numbers
        Data to be combined, i.e. list containing N (min. 1) arrays of same shape.
    total_val_rescale : float or int (optional)
        Total value for optional rescaling of resulting array. All values in result_array
        are skaled so that the sum is equal to total_val_rescale.
        The default (None) implies no rescaling.
    offsets: list or array containing N numbers >= 0 (optional)
        One numerical offset per array that is added (sum) to the
        corresponding array in data_arrays.
        The default (None) corresponds to np.zeros(N).
    exponents: list or array containing N numbers >= 0 (optional)
        One exponent per array used as power for the corresponding array.
        The default (None) corresponds to np.ones(N).

    Raises
    ------
    ValueError
        If input lists don't have the same number of elements.
        Or: If arrays in data_arrays do not have the same shape.

    Returns
    -------
    result_array : np.array of same shape as arrays in data_arrays
        Results from calculation described above.
    """
    # convert input data to arrays if proivided as lists
    # check integrity of data_array input (length and type):
    try:
        if isinstance(data_arrays[0], list):
            data_arrays[0] = np.array(data_arrays[0])
        for i_arr in np.arange(len(data_arrays)-1):
            if isinstance(data_arrays[i_arr+1], list):
                data_arrays[i_arr+1] = np.array(data_arrays[i_arr+1])
            if data_arrays[i_arr].shape != data_arrays[i_arr+1].shape:
                raise ValueError("Elements in data_arrays don't agree in shape.")
                
    except AttributeError as err:
            raise TypeError("data_arrays or contained elements have wrong type.") from err

    # if None, defaults for offsets and exponents are set:
    if offsets is None:
        offsets = np.zeros(len(data_arrays))
    if exponents is None:
        exponents = np.ones(len(data_arrays))
    if np.min(offsets) < 0:
        raise ValueError("offset values < 0 not are allowed.")
    if np.min(exponents) < 0:
        raise ValueError("exponents < 0 not are allowed.")
    # Steps 1-3: arrays are multiplied after application of offets and exponents:
    #       (arrays are converted to float to prevent ValueError:
    #       "Integers to negative integer powers are not allowed.")
    result_array = np.power(np.array(data_arrays[0]+offsets[0], dtype=float),
                            exponents[0])
    for i in np.arange(1, len(data_arrays)):
        result_array = np.multiply(result_array,
                                   np.power(np.array(data_arrays[i]+offsets[i], dtype=float),
                                            exponents[i])
                                   )

    # Steps 4+5: if total value for rescaling is provided, result_array is normalized and
    # scaled with this total value (total_val_rescale):
    if isinstance(total_val_rescale, float) or isinstance(total_val_rescale, int):
        return np.divide(result_array, result_array.sum()) * total_val_rescale
    elif total_val_rescale is not None:
        raise TypeError("total_val_rescale must be int or float.")

    return result_array