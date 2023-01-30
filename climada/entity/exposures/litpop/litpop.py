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
from pathlib import Path
import numpy as np
import rasterio
import geopandas
from shapefile import Shape
import shapely
import pandas as pd

import climada.util.coordinates as u_coord
import climada.util.finance as u_fin

from climada.entity.tag import Tag
from climada.entity.exposures.litpop import nightlight as nl_util
from climada.entity.exposures.litpop import gpw_population as pop_util
from climada.entity.exposures.base import Exposures, INDICATOR_IMPF, DEF_REF_YEAR
from climada.util.constants import SYSTEM_DIR
from climada import CONFIG
LOGGER = logging.getLogger(__name__)

GPW_VERSION = CONFIG.exposures.litpop.gpw_population.gpw_version.int()
"""Version of Gridded Population of the World (GPW) input data. Check for updates."""

class LitPop(Exposures):
    """
    Holds geopandas GeoDataFrame with metada and columns (pd.Series) defined in
    Attributes of Exposures class.
    LitPop exposure values are disaggregated proportional to a combination of
    nightlight intensity (NASA) and Gridded Population data (SEDAC).
    Total asset values can be produced capital, population count,
    GDP, or non-financial wealth.

    Calling sequence example:
    country_names = ['CHE', 'Austria']
    exp = LitPop.from_countries(country_names)
    exp.plot()

    Attributes
    ----------
    exponents : tuple of two integers, optional
        Defining powers (m, n) with which lit (nightlights) and pop (gpw)
        go into Lit**m * Pop**n. The default is (1,1).
    fin_mode : str, optional
        Socio-economic value to be used as an asset base that is disaggregated.
        The default is 'pc'.
    gpw_version : int, optional
        Version number of GPW population data, e.g. 11 for v4.11.
        The default is defined in GPW_VERSION.
    """
    _metadata = Exposures._metadata + ['exponents', 'fin_mode', 'gpw_version']

    def set_countries(self, *args, **kwargs):
        """This function is deprecated, use LitPop.from_countries instead."""
        LOGGER.warning("The use of LitPop.set_countries is deprecated."
                       "Use LitPop.from_countries instead.")
        self.__dict__ = LitPop.from_countries(*args, **kwargs).__dict__

    @classmethod
    def from_countries(cls, countries, res_arcsec=30, exponents=(1,1),
                       fin_mode='pc', total_values=None, admin1_calc=False,
                       reference_year=DEF_REF_YEAR, gpw_version=GPW_VERSION,
                       data_dir=SYSTEM_DIR):
        """Init new LitPop exposure object for a list of countries (admin 0).

        Sets attributes `ref_year`, `tag`, `crs`, `value`, `geometry`, `meta`,
        `value_unit`, `exponents`,`fin_mode`, `gpw_version`, and `admin1_calc`.

        Parameters
        ----------
        countries : list with str or int
            list containing country identifiers:
            iso3alpha (e.g. 'JPN'), iso3num (e.g. 92) or name (e.g. 'Togo')
        res_arcsec : float, optional
            Horizontal resolution in arc-sec.
            The default is 30 arcsec, this corresponds to roughly 1 km.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop. To get
            nightlights^3 without population count: (3, 0).
            To use population count alone: (0, 1).
            Default: (1, 1)
        fin_mode : str, optional
            Socio-economic value to be used as an asset base that is disaggregated
            to the grid points within the country:

            * 'pc': produced capital (Source: World Bank), incl. manufactured or
              built assets such as machinery, equipment, and physical structures
              `pc` is in constant 2014 USD.
            * 'pop': population count (source: GPW, same as gridded population).
              The unit is 'people'.
            * 'gdp': gross-domestic product (Source: World Bank) [USD]
            * 'income_group': gdp multiplied by country's income group+1 [USD].
              Income groups are 1 (low) to 4 (high income).
            * 'nfw': non-financial wealth (Source: Credit Suisse, of households only) [USD]
            * 'tw': total wealth (Source: Credit Suisse, of households only) [USD]
            * 'norm': normalized by country (no unit)
            * 'none': LitPop per pixel is returned unchanged (no unit)

            Default: 'pc'
        total_values : list containing numerics, same length as countries, optional
            Total values to be disaggregated to grid in each country.
            The default is None. If None, the total number is extracted from other
            sources depending on the value of fin_mode.
        admin1_calc : boolean, optional
            If True, distribute admin1-level GDP (if available). Default: False
        reference_year : int, optional
            Reference year. Default: CONFIG.exposures.def_ref_year.
        gpw_version : int, optional
            Version number of GPW population data.
            The default is GPW_VERSION
        data_dir : Path, optional
            redefines path to input data directory. The default is SYSTEM_DIR.

        Raises
        ------
        ValueError

        Returns
        -------
        exp : LitPop
            LitPop instance with exposure for given countries
        """
        if isinstance(countries, (int, str)):
            countries = [countries] # for backward compatibility

        if total_values is None: # init list with total values per countries
            total_values = [None] * len(countries)
        elif len(total_values) != len(countries):
            raise ValueError("'countries' and 'total_values' must be lists of same length")
        tag = Tag()

        # litpop_list is initiated, a list containing one Exposure instance per
        # country and None for countries that could not be identified:
        if admin1_calc: # each admin 1 region is initiated separately,
                        # with total value share based on subnational GDP share.
                        # This requires GRP (Gross Regional Product) data in the
                        # GSDP data folder.
            litpop_list = [_calc_admin1_one_country(country, res_arcsec, exponents,
                                                    fin_mode, tot_value, reference_year,
                                                    gpw_version, data_dir,
                                                    )
                           for tot_value, country in zip(total_values, countries)]

        else: # else, as default, country is initiated as a whole:
            # loop over countries: litpop is initiated for each individual polygon
            # within each country and combined at the end.
            litpop_list = \
                [cls._from_country(country,
                                   res_arcsec=res_arcsec,
                                   exponents=exponents,
                                   fin_mode=fin_mode,
                                   total_value=total_values[idc],
                                   reference_year=reference_year,
                                   gpw_version=gpw_version,
                                   data_dir=data_dir)
                 for idc, country in enumerate(countries)]
        # make lists of countries with Exposure initaited and those ignored:
        countries_in = \
            [country for lp, country in zip(litpop_list, countries) if lp is not None]
        countries_out = \
            [country for lp, country in zip(litpop_list, countries) if lp is None]
        if not countries_in:
            raise ValueError('No valid country identified in %s, aborting.' % countries)
        litpop_list = [exp for exp in litpop_list if exp is not None]
        if countries_out:
            LOGGER.warning('Some countries could not be identified and are ignored: '
                           '%s. Litpop only initiated for: %s', countries_out, countries_in)

        tag.description = f'LitPop Exposure for {countries_in} at {res_arcsec} as, ' \
                          f'year: {reference_year}, financial mode: {fin_mode}, ' \
                          f'exp: {exponents}, admin1_calc: {admin1_calc}'

        exp = cls(
            data=Exposures.concat(litpop_list).gdf,
            crs=litpop_list[0].crs,
            ref_year=reference_year,
            tag=tag,
            value_unit=get_value_unit(fin_mode),
            exponents = exponents,
            gpw_version = gpw_version,
            fin_mode = fin_mode,
        )

        try:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (exp.gdf.longitude.min(), exp.gdf.latitude.min(),
                 exp.gdf.longitude.max(), exp.gdf.latitude.max()),
                u_coord.get_resolution(exp.gdf.longitude, exp.gdf.latitude))
            exp.meta = {
                'width': cols,
                'height': rows,
                'crs': exp.crs,
                'transform': ras_trans,
            }
        except ValueError:
            LOGGER.warning('Could not write attribute meta, because exposure'
                           ' has only 1 data point')
            exp.meta = {'crs': exp.crs}
        exp.check()
        return exp

    def set_nightlight_intensity(self, *args, **kwargs):
        """This function is deprecated, use LitPop.from_nightlight_intensity instead."""
        LOGGER.warning("The use of LitPop.set_nightlight_intensity is deprecated."
                       "Use LitPop.from_nightlight_intensity instead.")
        self.__dict__ = LitPop.from_nightlight_intensity(*args, **kwargs).__dict__

    @classmethod
    def from_nightlight_intensity(cls, countries=None, shape=None, res_arcsec=15,
                        reference_year=DEF_REF_YEAR, data_dir=SYSTEM_DIR):
        """
        Wrapper around `from_countries` / `from_shape`.

        Initiate exposures instance with value equal to the original BlackMarble
        nightlight intensity resampled to the target resolution `res_arcsec`.

        Provide either `countries` or `shape`.

        Parameters
        ----------
        countries : list or str, optional
            list containing country identifiers (name or iso3)
        shape : Shape, Polygon or MultiPolygon, optional
            geographical shape of target region, alternative to `countries`.
        res_arcsec : int, optional
            Resolution in arc seconds. The default is 15.
        reference_year : int, optional
            Reference year. The default is CONFIG.exposures.def_ref_year.
        data_dir : Path, optional
            data directory. The default is None.

        Raises
        ------
        ValueError

        Returns
        -------
        exp : LitPop
            Exposure instance with values representing pure nightlight intensity
            from input nightlight data (BlackMarble)
        """
        if countries is None and shape is None:
            raise ValueError("Either `countries` or `shape` required. Aborting.")
        if countries is not None and shape is not None:
            raise ValueError("Not allowed to set both `countries` and `shape`. Aborting.")
        if countries is not None:
            exp = cls.from_countries(countries, res_arcsec=res_arcsec,
                                     exponents=(1,0), fin_mode='none',
                                     reference_year=reference_year, gpw_version=GPW_VERSION,
                                     data_dir=data_dir)
        else:
            exp = cls.from_shape(shape, None, res_arcsec=res_arcsec,
                                 exponents=(1,0), value_unit='',
                                 reference_year=reference_year,
                                 gpw_version=GPW_VERSION, data_dir=SYSTEM_DIR)
        LOGGER.warning("Note: set_nightlight_intensity sets values to raw nightlight intensity, "
                       "not to USD. "
                       "To disaggregate asset value proportionally to nightlights^m, "
                       "call from_countries or from_shape with exponents=(m,0).")
        return exp

    def set_population(self, *args, **kwargs):
        """This function is deprecated, use LitPop.from_population instead."""
        LOGGER.warning("The use of LitPop.set_population is deprecated."
                       "Use LitPop.from_population instead.")
        self.__dict__ = LitPop.from_population(*args, **kwargs).__dict__

    @classmethod
    def from_population(cls, countries=None, shape=None, res_arcsec=30,
                        reference_year=DEF_REF_YEAR, gpw_version=GPW_VERSION,
                        data_dir=SYSTEM_DIR):
        """
        Wrapper around `from_countries` / `from_shape`.

        Initiate exposures instance with value equal to GPW population count.
        Provide either `countries` or `shape`.

        Parameters
        ----------
        countries : list or str, optional
            list containing country identifiers (name or iso3)
        shape : Shape, Polygon or MultiPolygon, optional
            geographical shape of target region, alternative to `countries`.
        res_arcsec : int, optional
            Resolution in arc seconds. The default is 30.
        reference_year : int, optional
            Reference year (closest available GPW data year is used)
            The default is CONFIG.exposures.def_ref_year.
        gpw_version : int, optional
            specify GPW data verison. The default is 11.
        data_dir : Path, optional
            data directory. The default is None.
            Either countries or shape is required.

        Raises
        ------
        ValueError

        Returns
        -------
        exp : LitPop
            Exposure instance with values representing population count according
            to Gridded Population of the World (GPW) input data set.
        """
        if countries is None and shape is None:
            raise ValueError("Either `countries` or `shape` required. Aborting.")
        if countries is not None and shape is not None:
            raise ValueError("Not allowed to set both `countries` and `shape`. Aborting.")
        if countries is not None:
            exp = cls.from_countries(countries, res_arcsec=res_arcsec,
                                     exponents=(0,1), fin_mode='pop',
                                     reference_year=reference_year, gpw_version=gpw_version,
                                     data_dir=data_dir)
        else:
            exp = cls.from_shape(shape, None, res_arcsec=res_arcsec, exponents=(0,1),
                                 value_unit='people', reference_year=reference_year,
                                 gpw_version=gpw_version, data_dir=data_dir)
        return exp

    def set_custom_shape_from_countries(self, *args, **kwargs):
        """This function is deprecated, use LitPop.from_shape_and_countries instead."""
        LOGGER.warning("The use of LitPop.set_custom_shape_from_countries is deprecated."
                       "Use LitPop.from_shape_and_countries instead.")
        self.__dict__ = LitPop.from_shape_and_countries(*args, **kwargs).__dict__

    @classmethod
    def from_shape_and_countries(cls, shape, countries, res_arcsec=30, exponents=(1,1),
                                 fin_mode='pc', admin1_calc=False, reference_year=DEF_REF_YEAR,
                                 gpw_version=GPW_VERSION, data_dir=SYSTEM_DIR):
        """
        create LitPop exposure for `country` and then crop to given shape.

        Parameters
        ----------
        shape : shapely.geometry.Polygon, MultiPolygon, shapereader.Shape,
            or GeoSeries or list containg either Polygons or Multipolygons.
            Geographical shape for which LitPop Exposure is to be initiated.
        countries : list with str or int
            list containing country identifiers:
            iso3alpha (e.g. 'JPN'), iso3num (e.g. 92) or name (e.g. 'Togo')
        res_arcsec : float, optional
            Horizontal resolution in arc-sec.
            The default is 30 arcsec, this corresponds to roughly 1 km.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop.
            Default: (1, 1)
        fin_mode : str, optional
            Socio-economic value to be used as an asset base that is disaggregated
            to the grid points within the country:

            * 'pc': produced capital (Source: World Bank), incl. manufactured or
              built assets such as machinery, equipment, and physical structures
              (pc is in constant 2014 USD)
            * 'pop': population count (source: GPW, same as gridded population).
              The unit is 'people'.
            * 'gdp': gross-domestic product (Source: World Bank) [USD]
            * 'income_group': gdp multiplied by country's income group+1 [USD]
              Income groups are 1 (low) to 4 (high income).
            * 'nfw': non-financial wealth (Source: Credit Suisse, of households only) [USD]
            * 'tw': total wealth (Source: Credit Suisse, of households only) [USD]
            * 'norm': normalized by country
            * 'none': LitPop per pixel is returned unchanged

            Default: 'pc'
        admin1_calc : boolean, optional
            If True, distribute admin1-level GDP (if available). Default: False
        reference_year : int, optional
            Reference year for data sources. Default: 2020
        gpw_version : int, optional
            Version number of GPW population data.
            The default is GPW_VERSION
        data_dir : Path, optional
            redefines path to input data directory. The default is SYSTEM_DIR.

        Raises
        ------
        NotImplementedError

        Returns
        -------
        exp : LitPop
            The exposure LitPop within shape
        """
        # init countries' exposure:
        exp = cls.from_countries(countries, res_arcsec=res_arcsec, exponents=exponents,
                                fin_mode=fin_mode, reference_year=reference_year,
                                gpw_version=gpw_version, data_dir=data_dir)

        if isinstance(shape, Shape):
            # get gdf with geometries of points within shape:
            shape_gdf, _ = _get_litpop_single_polygon(shape, reference_year,
                                                      res_arcsec, data_dir,
                                                      gpw_version, exponents,
                                                      )
            shape_gdf = shape_gdf.drop(
                columns=shape_gdf.columns[shape_gdf.columns != 'geometry'])
            # extract gdf with data points within shape:
            gdf = geopandas.sjoin(exp.gdf, shape_gdf, how='right')
            gdf = gdf.drop(columns=['index_left'])
        elif isinstance(shape, (shapely.geometry.MultiPolygon, shapely.geometry.Polygon)):
            # works if shape is Polygon or MultiPolygon
            gdf = exp.gdf.loc[exp.gdf.geometry.within(shape)]
        elif isinstance(shape, (geopandas.GeoSeries, list)):
            gdf = geopandas.GeoDataFrame(columns=exp.gdf.columns)
            for shp in shape:
                if isinstance(shp, (shapely.geometry.MultiPolygon,
                                    shapely.geometry.Polygon)):
                    gdf = gdf.append(exp.gdf.loc[exp.gdf.geometry.within(shp)])
                else:
                    raise NotImplementedError('Not implemented for list or GeoSeries containing '
                                              f'objects of type {type(shp)} as `shape`')
        else:
            raise NotImplementedError('Not implemented for `shape` of type {type(shape)}')

        tag = Tag()
        tag.description = f'LitPop Exposure for custom shape in {countries} at ' \
                          f'{res_arcsec} as, year: {reference_year}, financial mode: ' \
                          f'{fin_mode}, exp: {exponents}, admin1_calc: {admin1_calc}'
        exp.set_gdf(gdf.reset_index())

        try:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (exp.gdf.longitude.min(), exp.gdf.latitude.min(),
                 exp.gdf.longitude.max(), exp.gdf.latitude.max()),
                u_coord.get_resolution(exp.gdf.longitude, exp.gdf.latitude))
            exp.meta = {
                'width': cols,
                'height': rows,
                'crs': exp.crs,
                'transform': ras_trans,
            }
        except ValueError as err:
            LOGGER.warning('Could not write attribute meta with ValueError: ')
            LOGGER.warning(err.args[0])
            exp.meta = {'crs': exp.crs}
        return exp

    def set_custom_shape(self, *args, **kwargs):
        """This function is deprecated, use LitPop.from_shape instead."""
        LOGGER.warning("The use of LitPop.set_custom_shape is deprecated."
                       "Use LitPop.from_shape instead.")
        self.__dict__ = LitPop.from_shape(*args, **kwargs).__dict__

    @classmethod
    def from_shape(cls, shape, total_value, res_arcsec=30, exponents=(1,1), value_unit='USD',
                   reference_year=DEF_REF_YEAR, gpw_version=GPW_VERSION, data_dir=SYSTEM_DIR):
        """init LitPop exposure object for a custom shape.
        Requires user input regarding the total value to be disaggregated.

        Sets attributes `ref_year`, `tag`, `crs`, `value`, `geometry`, `meta`,
        `value_unit`, `exponents`,`fin_mode`, `gpw_version`, and `admin1_calc`.

        This method can be used to initiated LitPop Exposure for sub-national
        regions such as states, districts, cantons, cities, ... but shapes
        and total value need to be provided manually.
        If these required input parameters are not known / available,
        better initiate Exposure for entire country and extract shape afterwards.

        Parameters
        ----------
        shape : shapely.geometry.Polygon or MultiPolygon or shapereader.Shape.
            Geographical shape for which LitPop Exposure is to be initiated.
        total_value : int, float or None type
            Total value to be disaggregated to grid in shape.
            If None, no value is disaggregated.
        res_arcsec : float, optional
            Horizontal resolution in arc-sec.
            The default 30 arcsec corresponds to roughly 1 km.
        exponents : tuple of two integers, optional
            Defining power with which lit (nightlights) and pop (gpw) go into LitPop.
        value_unit : str
            Unit of exposure values. The default is USD.
        reference_year : int, optional
            Reference year for data sources. Default: CONFIG.exposures.def_ref_year
        gpw_version : int, optional
            Version number of GPW population data.
            The default is set in CONFIG.
        data_dir : Path, optional
            redefines path to input data directory. The default is SYSTEM_DIR.

        Raises
        ------
        NotImplementedError
        ValueError
        TypeError

        Returns
        -------
        exp : LitPop
            The exposure LitPop within shape
        """
        if isinstance(shape, (geopandas.GeoSeries, list)):
            raise NotImplementedError('Not implemented for `shape` of type list or '
                                      'GeoSeries. Loop over elements of series outside method.')

        tag = Tag()
        litpop_gdf, _ = _get_litpop_single_polygon(shape, reference_year,
                                                          res_arcsec, data_dir,
                                                          gpw_version, exponents,
                                                          )

        # disaggregate total value proportional to LitPop values:
        if isinstance(total_value, (float, int)):
            litpop_gdf['value'] = np.divide(litpop_gdf['value'],
                                            litpop_gdf['value'].sum()) * total_value
        elif total_value is not None:
            raise TypeError("total_value must be int, float or None.")

        tag.description = f'LitPop Exposure for custom shape at {res_arcsec} as, ' \
                          f'year: {reference_year}, exp: {exponents}'


        litpop_gdf[INDICATOR_IMPF] = 1

        exp = cls(
                  data=litpop_gdf,
                  crs=litpop_gdf.crs,
                  ref_year=reference_year,
                  tag=tag,
                  value_unit=value_unit,
                  exponents = exponents,
                  gpw_version = gpw_version,
                  fin_mode = None,
                  )

        if min(len(exp.gdf.latitude.unique()), len(exp.gdf.longitude.unique())) > 1:
        #if exp.gdf.shape[0] > 1 and len(exp.gdf.latitude.unique()) > 1:
            rows, cols, ras_trans = u_coord.pts_to_raster_meta(
                (exp.gdf.longitude.min(), exp.gdf.latitude.min(),
                 exp.gdf.longitude.max(), exp.gdf.latitude.max()),
                u_coord.get_resolution(exp.gdf.longitude, exp.gdf.latitude))
            exp.meta = {
                'width': cols,
                'height': rows,
                'crs': exp.crs,
                'transform': ras_trans,
            }
        else:
            LOGGER.warning('Could not write attribute meta because coordinates'
                           ' are either only one point or do not extend in lat and lon')
            exp.meta = {'crs': exp.crs}
        return exp

    @staticmethod
    def _from_country(country, res_arcsec=30, exponents=(1,1), fin_mode=None,
                         total_value=None, reference_year=DEF_REF_YEAR,
                         gpw_version=GPW_VERSION, data_dir=SYSTEM_DIR):
        """init LitPop exposure object for one single country
        See docstring of from_countries() for detailled description of parameters.

        Parameters
        ----------
        country : str or int
            country identifier as iso3alpha, iso3num or name.
        res_arcsec : float, optional
            horizontal resolution in arc-sec.
        exponents : tuple of two integers, optional
        fin_mode : str, optional
        total_value : numeric, optional
        reference_year : int, optional
        gpw_version : int, optional
        data_dir : Path, optional
            redefines path to input data directory. The default is SYSTEM_DIR.

        Raises
        ------
        ValueError

        Returns
        -------
        exp : LitPop
            LitPop Exposure instance for the country
        """
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
        LOGGER.info('\n LitPop: Init Exposure for country: %s (%i)...\n',
                    iso3a, iso3n)
        litpop_gdf = geopandas.GeoDataFrame()
        total_population = 0

        # for countries with multiple sperated shapes (e.g., islands), data
        # is initiated for each shape separately and 0 values (e.g. on sea)
        # removed before combination, to save memory.
        # loop over single polygons in country shape object:
        for idx, polygon in enumerate(country_geometry.geoms):
            # get litpop data for each polygon and combine into GeoDataFrame:
            gdf_tmp, meta_tmp, = \
                _get_litpop_single_polygon(polygon, reference_year,
                                           res_arcsec, data_dir,
                                           gpw_version, exponents,
                                           verbose=(idx > 0),
                                           region_id=iso3n
                                           )
            if gdf_tmp is None:
                LOGGER.debug(f'Skipping polygon with index {idx} for' +
                               f' country {iso3a}.')
                continue
            total_population += meta_tmp['total_population']
            litpop_gdf = pd.concat([litpop_gdf, gdf_tmp])
        litpop_gdf.crs = meta_tmp['crs']

        # set total value for disaggregation if not provided:
        if total_value is None and fin_mode == 'pop':
            total_value = total_population # population count is taken from pop-data.
        elif total_value is None:
            total_value = _get_total_value_per_country(iso3a, fin_mode, reference_year)

        # disaggregate total value proportional to LitPop values:
        if isinstance(total_value, (float, int)):
            litpop_gdf['value'] = np.divide(litpop_gdf['value'],
                                            litpop_gdf['value'].sum()) * total_value
        elif total_value is not None:
            raise TypeError("total_value must be int or float.")

        exp = LitPop()
        exp.set_gdf(litpop_gdf)
        exp.gdf[INDICATOR_IMPF] = 1
        return exp

    # Alias method names for backward compatibility:
    set_country = set_countries

def _get_litpop_single_polygon(polygon, reference_year, res_arcsec, data_dir,
                               gpw_version, exponents, region_id=None, verbose=False):
    """load nightlight (nl) and population (pop) data in rastered 2d arrays
    and apply rescaling (resolution reprojection) and LitPop core calculation,
    i.e. combination of nl and pop per grid cell.

    NB: In this method, there is no disaggregation taking place, yet. This is
    done in a subsequent step outside this method.

    Parameters
    ----------
    polygon : Polygon object
        single polygon to be extracted
    reference_year : int, optional
        Reference year for data sources.
    res_arcsec : float
        Horizontal resolution in arc-seconds.
    data_dir : Path
        redefines path to input data directory. The default is SYSTEM_DIR.
    gpw_version : int
        Version number of GPW population data.
        The default is None. If None, the default is set in gpw_population module.
    exponents : tuple of two integers
        Defining power with which lit (nightlights) and pop (gpw) go into LitPop. To get
        nightlights^3 without population count: (3, 0). To use population count alone: (0, 1).
    region_id : int, optional
        if provided, region_id of gdf is set to value.
        The default is None, this implies that region_id is not set.
    verbose : bool, optional
        Enable verbose logging about the used GPW version and reference year as well as about the
        boundary case where no grid points from the GPW grid are contained in the specified
        polygon. Default: False.

    Returns
    -------
    litpop_gdf : GeoDataframe
        resulting gridded data for Lit^m * Pop^n inside polygon,
        data points outside the polygon and equal to zero are not returned.
    meta_out : dict
        raster meta info for gridded data in litpop_array, additionally the field
        'total_population' contains the sum of population in the polygon.

    """
    # set nightlight offset (delta) to 1 in case n>0, c.f. delta in Eq. 1 of paper:
    if exponents[1] == 0:
        offsets = (0, 0)
    else:
        offsets = (1, 0)
    # import population data (2d array), meta data, and global grid info,
    # global_transform defines the origin (corner points) of the global traget grid:
    pop, meta_pop, global_transform = \
        pop_util.load_gpw_pop_shape(polygon,
                                    reference_year,
                                    gpw_version=gpw_version,
                                    data_dir=data_dir,
                                    verbose=verbose,
                                    )
    total_population = pop.sum()
    # import nightlight data (2d array) and associated meta data:
    nlight, meta_nl = nl_util.load_nasa_nl_shape(polygon,
                                                 reference_year,
                                                 data_dir=data_dir,
                                                 dtype=float
                                                 )

    # if resolution is the same as for lit (15 arcsec), set grid same as lit:
    if res_arcsec==15:
        i_align = 1
        global_origins = (meta_nl['transform'][2], # lon
                          meta_nl['transform'][5]) # lat
    else: # align grid for resampling to grid of population data (pop)
        i_align = 0
        global_origins=(global_transform[2],
                        global_transform[5])

    # reproject Lit and Pop input data to aligned grid with target resolution:
    try:
        [pop, nlight], meta_out = reproject_input_data([pop, nlight],
                                                       [meta_pop, meta_nl],
                                                       i_align=i_align, # pop defines grid
                                                       target_res_arcsec=res_arcsec,
                                                       global_origins=global_origins,
                                                       )
    except ValueError as err:
        if "height must be > 0" in err.args[0] or "width must be > 0" in err.args[0]:
            # no grid point within shape after reprojection, None is returned.
            if verbose:
                LOGGER.info('No data point on destination grid within polygon.')
            return None, {'crs': meta_pop['crs']}
        raise err

    # calculate Lit^m * Pop^n (but not yet disaggregate any total value to grid):
    litpop_array = gridpoints_core_calc([nlight, pop],
                                        offsets=offsets,
                                        exponents=exponents,
                                        total_val_rescale=None)

    # mask entries outside polygon (set to NaN) and set total population:
    litpop_array = u_coord.mask_raster_with_geometry(litpop_array, meta_out['transform'],
                                                     [polygon], nodata=np.nan)
    meta_out['total_population'] = total_population

    # extract coordinates as meshgrid arrays:
    lon, lat = u_coord.raster_to_meshgrid(meta_out['transform'],
                                          meta_out['width'],
                                          meta_out['height'])
    # init GeoDataFrame from data and coordinates:
    gdf = geopandas.GeoDataFrame({'value': litpop_array.flatten()}, crs=meta_out['crs'],
                                 geometry=geopandas.points_from_xy(
                                     np.round_(lon.flatten(), decimals = 8, out = None),
                                     np.round_(lat.flatten(), decimals = 8, out = None)
                                     )
                                 )
    gdf['latitude'] = np.round_(lat.flatten(), decimals = 8, out = None)
    gdf['longitude'] = np.round_(lon.flatten(), decimals = 8, out = None)
    if region_id is not None: # set region_id
        gdf['region_id'] = region_id
    else:
        gdf['region_id'] = u_coord.get_country_code(gdf.latitude, gdf.longitude)
    # remove entries outside polygon with `dropna` and return GeoDataFrame:
    return gdf.dropna(), meta_out

def get_value_unit(fin_mode):
    """get `value_unit` depending on `fin_mode`

    Parameters
    ----------
    fin_mode : Socio-economic value to be used as an asset base

    Returns
    -------
    value_unit : str

    """
    if fin_mode in ['none', 'norm']:
        return ''
    if fin_mode == 'pop':
        return 'people'
    return 'USD'

def _get_total_value_per_country(cntry_iso3a, fin_mode, reference_year):
    """
    Get total value for disaggregation, e.g., total asset value
    for a country, depending on user choice (fin_mode).

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
            * 'gdp': gross-domestic product (Source: World Bank) [USD]
            * 'income_group': gdp multiplied by country's income group+1 [USD]
              Income groups are 1 (low) to 4 (high income).
            * 'nfw': non-financial wealth (Source: Credit Suisse, of households only) [USD]
            * 'tw': total wealth (Source: Credit Suisse, of households only) [USD]
            * 'norm': normalized by country
            * 'none': LitPop per pixel is returned unscaled
        The default is 'pc'
    reference_year : int
        reference year for data extraction

    Returns
    -------
    total_value : float
    """
    if fin_mode == 'none':
        return None
    if fin_mode == 'pop':
        raise NotImplementedError("`_get_total_value_per_country` is not "
                                  "implemented for `fin_mode` == 'pop'.")
    if fin_mode == 'pc':
        return(u_fin.world_bank_wealth_account(cntry_iso3a, reference_year,
                                               no_land=True)[1])
        # here, total_asset_val is Produced Capital "pc"
        # no_land=True returns value w/o the mark-up of 24% for land value
    if fin_mode == 'pc_land':
        return(u_fin.world_bank_wealth_account(cntry_iso3a, reference_year,
                                               no_land=False)[1])
        # no_land=False returns pc value incl. the mark-up of 24% for land value
    if fin_mode == 'norm':
        return 1
    # GDP based total values:
    gdp_value = u_fin.gdp(cntry_iso3a, reference_year)[1]
    if fin_mode == 'gdp':
        return gdp_value
    if fin_mode == 'income_group': # gdp * (income group + 1)
        return gdp_value*(u_fin.income_group(cntry_iso3a, reference_year)[1]+1)
    if fin_mode in ('nfw', 'tw'):
        wealthtogdp_factor = u_fin.wealth2gdp(cntry_iso3a, fin_mode == 'nfw',
                                              reference_year)[1]
        if np.isnan(wealthtogdp_factor):
            LOGGER.warning("Missing wealth-to-gdp factor for country %s.", cntry_iso3a)
            LOGGER.warning("Using GDP instead as total value.")
            return gdp_value
        return gdp_value * wealthtogdp_factor
    raise ValueError(f"Unsupported fin_mode: {fin_mode}")

def reproject_input_data(data_array_list, meta_list,
                        i_align=0,
                        target_res_arcsec=None,
                        global_origins=(-180.0, 89.99999999999991),
                        resampling=rasterio.warp.Resampling.bilinear,
                        conserve=None):
    """
    LitPop-sepcific wrapper around u_coord.align_raster_data.

    Reprojects all arrays in data_arrays to a given resolution –
    all based on the population data grid.

    Parameters
    ----------
    data_array_list : list or array of numpy arrays containing numbers
        Data to be reprojected, i.e. list containing N (min. 1) 2D-arrays.
        The data with the reference grid used to align the global destination
        grid to should be first data_array_list[i_align],
        e.g., pop (GPW population data) for LitPop.
    meta_list : list of dicts
        meta data dictionaries of data arrays in same order as data_array_list.
        Required fields in each dict are 'dtype,', 'width', 'height', 'crs', 'transform'.
        Example:

        >>> {
        ...     'driver': 'GTiff',
        ...     'dtype': 'float32',
        ...     'nodata': 0,
        ...     'width': 2702,
        ...     'height': 1939,
        ...     'count': 1,
        ...     'crs': CRS.from_epsg(4326),
        ...     'transform': Affine(0.00833333333333333, 0.0, -18.175000000000068,
        ...                         0.0, -0.00833333333333333, 43.79999999999993),
        ... }

        The meta data with the reference grid used to define the global destination
        grid should be first in the list, e.g., GPW population data for LitPop.
    i_align : int, optional
        Index/Position of meta in meta_list to which the global grid of the destination
        is to be aligned to (c.f. u_coord.align_raster_data)
        The default is 0.
    target_res_arcsec : int, optional
        target resolution in arcsec. The default is None, i.e. same resolution
        as reference data.
    global_origins : tuple with two numbers (lat, lon), optional
        global lon and lat origins as basis for destination grid.
        The default is the same as for GPW population data: ``(-180.0, 89.99999999999991)``
    resampling : resampling function, optional
        The default is rasterio.warp.Resampling.bilinear
    conserve : str, optional, either 'mean' or 'sum'
        Conserve mean or sum of data? The default is None (no conservation).

    Returns
    -------
    data_array_list : list
        contains reprojected data sets
    meta_out : dict
        contains meta data of new grid (same for all arrays)
    """

    # target resolution in degree lon,lat:
    if target_res_arcsec is None:
        res_degree = meta_list[i_align]['transform'][0] # reference grid
    else:
        res_degree = target_res_arcsec / 3600

    dst_crs = meta_list[i_align]['crs']
    # loop over data arrays, do transformation where required:
    data_out_list = [None] * len(data_array_list)
    meta_out = {'dtype': meta_list[i_align]['dtype'],
                'nodata': meta_list[i_align]['nodata'],
                'crs': dst_crs}

    for idx, data in enumerate(data_array_list):
        # if target resolution corresponds to reference data resolution,
        # the reference data is not transformed:
        if idx==i_align and ((target_res_arcsec is None) or \
                           (np.round(meta_list[i_align]['transform'][0],
                            decimals=7)==np.round(res_degree, decimals=7))):
            data_out_list[idx] = data
            continue
        # reproject data grid:
        dst_bounds = rasterio.transform.array_bounds(meta_list[i_align]['height'],
                                                     meta_list[i_align]['width'],
                                                     meta_list[i_align]['transform'])
        data_out_list[idx], meta_out['transform'] = \
            u_coord.align_raster_data(data_array_list[idx], meta_list[idx]['crs'],
                                      meta_list[idx]['transform'],
                                      dst_crs=dst_crs,
                                      dst_resolution=(res_degree, res_degree),
                                      dst_bounds=dst_bounds,
                                      global_origin=global_origins,
                                      resampling=resampling,
                                      conserve=conserve)
    meta_out['height'] = data_out_list[-1].shape[0]
    meta_out['width'] = data_out_list[-1].shape[1]
    return data_out_list, meta_out

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
    total_val_rescale : float or int, optional
        Total value for optional rescaling of resulting array. All values in result_array
        are skaled so that the sum is equal to total_val_rescale.
        The default (None) implies no rescaling.
    offsets: list or array containing N numbers >= 0, optional
        One numerical offset per array that is added (sum) to the
        corresponding array in data_arrays.
        The default (None) corresponds to np.zeros(N).
    exponents: list or array containing N numbers >= 0, optional
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
    # convert input data to arrays
    # check integrity of data_array input (length and type):
    try:
        data_arrays = [np.array(data) for data in data_arrays]
        if len(list({data.shape for data in data_arrays})) > 1:
            raise ValueError("Elements in data_arrays don't agree in shape.")
    except AttributeError as err:
        raise TypeError("data_arrays or contained elements have wrong type.") from err

    # if None, defaults for offsets and exponents are set:
    if offsets is None:
        offsets = np.zeros(len(data_arrays))
    if exponents is None:
        exponents = np.ones(len(data_arrays))
    if np.min(offsets) < 0:
        raise ValueError("offset values < 0 are not allowed, because negative offset "
                         "values produce undesired negative disaggregation values.")
    if np.min(exponents) < 0:
        raise ValueError("exponents < 0 are not allowed, because negative exponents "
                         "produce 'inf' values where the initial value is 0.")
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
    if isinstance(total_val_rescale, (float, int)):
        return np.divide(result_array, result_array.sum()) * total_val_rescale
    if total_val_rescale is not None:
        raise TypeError("total_val_rescale must be int or float.")
    return result_array

# The following functions are only required if calc_admin1 is True,
# not for core LitPop. They are maintained here mainly for backward compatibility
def _check_excel_exists(file_path, file_name, xlsx_before_xls=True):
    """Checks if an Excel file with the name file_name in the folder file_path exists, checking for
    both xlsx and xls files.

    Parameters
    ----------
    file_path : string
        path where to look for file
    file_name : string
        file name which is checked. Extension is ignored
    xlsx_before_xls : boolean, optional
        If True, xlsx files are priorised over xls files. Default: True.

    Returns
    -------
    path: Path instance
    """
    try_ext = []
    if xlsx_before_xls:
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

def _grp_read(country_iso3, admin1_info=None, data_dir=SYSTEM_DIR):
    """Retrieves the Gross Regional Product (GRP) aka Gross State Domestic Product (GSDP)
    data for a certain country. It requires an excel file in a subfolder
    "GSDP" in climadas data folder (or in the specified folder). The excel file should bear the
    name 'ISO3_GSDP.xlsx' (or .xls), where ISO3 is the three letter country code. In the excel
    file, the first sheet should contain a row with the title "State_Province" with the name or
    postal code (two letters) of the admin1 unit and a row "GSDP_ref" with either the GDP value of
    the admin1 unit or its share in the national GDP.

    Parameters
    ----------
    country_iso3 : str
        alphabetic three letter country code ISO3a
    admin1_info : list, optional
        list containg all admin1 records for country.
        if not provided, info is set retrieved automatically
    data_dir : str, optional
        path where to look for file

    Returns
    -------
    out_dict : dictionary
        GRP for each admin1 unit name.
    """
    if admin1_info is None:
        admin1_info, _ = u_coord.get_admin1_info(country_iso3)
        admin1_info = admin1_info[country_iso3]
    file_name = _check_excel_exists(data_dir.joinpath('GSDP'), str(country_iso3 + '_GSDP'))
    if file_name is not None:
        # open spreadsheet and identify relevant columns:
        admin1_xls_data = pd.read_excel(file_name)
        if admin1_xls_data.get('State_Province') is None:
            admin1_xls_data = admin1_xls_data.rename(
                columns={admin1_xls_data.columns[0]: 'State_Province'})
        if admin1_xls_data.get('GSDP_ref') is None:
            admin1_xls_data = admin1_xls_data.rename(
                columns={admin1_xls_data.columns[-1]: 'GSDP_ref'})

        # initiate dictionary with admin 1 names as keys:
        out_dict = dict.fromkeys([record['name'] for record in admin1_info])
        postals = [record['postal'] for record in admin1_info]
        # first nested loop. outer loop over region names in admin1_info:
        for record_name in out_dict:
            # inner loop over region names in spreadsheet, find matches
            for idx, xls_name in enumerate(admin1_xls_data['State_Province'].tolist()):
                subnat_shape_str = [c for c in record_name if c.isalpha() or c.isnumeric()]
                subnat_xls_str = [c for c in xls_name if c.isalpha()]
                if subnat_shape_str == subnat_xls_str:
                    out_dict[record_name] = admin1_xls_data['GSDP_ref'][idx]
                    break
        # second nested loop to detect matched empty entries
        for idx1, country_name in enumerate(out_dict.keys()):
            if out_dict[country_name] is None:
                for idx2, xls_name in enumerate(admin1_xls_data['State_Province'].tolist()):
                    subnat_xls_str = [c for c in xls_name if c.isalpha()]
                    postals_str = [c for c in postals[idx1] if c.isalpha()]
                    if subnat_xls_str == postals_str:
                        out_dict[country_name] = admin1_xls_data['GSDP_ref'][idx2]
        return out_dict
    LOGGER.warning('No file for %s could be found in %s.', country_iso3, data_dir)
    LOGGER.warning('No admin1 data is calculated in this case.')
    return None

def _calc_admin1_one_country(country, res_arcsec, exponents, fin_mode, total_value,
                 reference_year, gpw_version, data_dir):
    """
    Calculates the LitPop on admin1 level for provinces/states where such information are
    available (i.e. GDP is distributed on a subnational instead of a national level). Requires
    excel files in a subfolder "GSDP" in climadas data folder. The excel files should contain a row
    with the title "State_Province" with the name or postal code (two letters) of the admin1 unit
    and a row "GSDP_ref" with either the GDP value or the share of the state in the national GDP.
    If only for certain states admin1 info is found, the rest of the country is assigned value
    according to the admin0 method.

    See from_countries() for description of parameters.

    Parameters
    ----------
    country : str
    res_arcsec : int
    exponents : tuple
    fin_mode : str
    total_value :int or float
    reference_year : int
    gpw_version: int
    data_dir : Path

    Returns
    -------
    Exposure instance

    """
    if fin_mode == 'pop':
        raise NotImplementedError('`_calc_admin1_one_country` not implemented for '+
                                          "`fin_mode` == 'pop'.")
    # Determine ISO 3166 representation of country and get geometry:
    try:
        iso3a = u_coord.country_to_iso(country, representation="alpha3")
    except LookupError:
        LOGGER.error('Country not identified: %s. Skippig.', country)
        return None
    # get records and shapes on admin 1 level:
    admin1_info, admin1_shapes = u_coord.get_admin1_info(iso3a)
    admin1_info = admin1_info[iso3a]
    admin1_shapes = admin1_shapes[iso3a]
    # get subnational Gross Regional Product (GRP) data for country:
    grp_values = _grp_read(iso3a, admin1_info=admin1_info, data_dir=data_dir)
    if grp_values is None:
        LOGGER.error("No subnational GRP data found for calc_admin1"
                         " for country %s. Skipping.", country)
        return None
    # normalize GRP values:
    sum_vals = sum(filter(None, grp_values.values())) # get total
    grp_values = {key: (value / sum_vals if value is not None else None)
                 for (key, value) in grp_values.items()}

    # get total value of country:
    total_value = _get_total_value_per_country(iso3a, fin_mode, reference_year)
    exp_list = []
    for idx, record in enumerate(admin1_info):
        if grp_values[record['name']] is None:
            continue
        LOGGER.info(record['name'])
        # init exposure for province and add to list
        # total value is defined from country multiplied by grp_share:
        exp_list.append(LitPop.from_shape(admin1_shapes[idx],
                                          total_value * grp_values[record['name']],
                                          res_arcsec=res_arcsec,
                                          exponents=exponents,
                                          reference_year=reference_year,
                                          gpw_version=gpw_version,
                                          data_dir=data_dir)
                        )
        exp_list[-1].gdf['admin1'] = record['name']

    return Exposures.concat(exp_list)
