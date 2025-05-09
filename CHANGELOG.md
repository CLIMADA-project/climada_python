# Changelog

## 6.0.1

Release date: 2025-03-13

### Fixed

- bug in `climada.util.coordinates.bounding_box_from_countries` occurring if the country is a polygon and not a multipolygon
[#1018](https://github.com/CLIMADA-project/climada_python/pull/1018)

## 6.0.0

Release date: 2025-03-03

### Dependency Changes

Added:

- `osm-flex` >=1.1

Updated:

- `cartopy` >=0.23 &rarr; >=0.24
- `cfgrib` >=0.9.9,<0.9.10 &rarr; >=0.9
- `dask` >=2024.2,<2024.3 &rarr; >=2025.2
- `eccodes` >=2.27,<2.28 &rarr; >=2.40
- `gdal` >=3.6 &rarr; >=3.10
- `geopandas` >=0.14 &rarr; >=0.14,<1.0
- `h5py` >=3.8 &rarr; >=3.12
- `haversine` >=2.8 &rarr; >=2.9
- `matplotlib-base` >=3.9 &rarr; >=3.10
- `netcdf4` >=1.6 &rarr; >=1.7
- `numba` >=0.60 &rarr; >=0.61
- `pillow` =9.4 &rarr; =11.1
- `pyproj` >=3.5 &rarr; >=3.7
- `pytables` >=3.7 &rarr; >=3.10
- `python` =3.9 &rarr; =3.11
- `rasterio` >=1.3 &rarr; >=1.4
- `scikit-learn` >=1.5 &rarr; >=1.6
- `scipy` >=1.13 &rarr; >=1.14,<1.15
- `tqdm` >=4.66 &rarr; >=4.67
- `xarray` >=2024.6 &rarr; >=2025.1
- `xlsxwriter` >=3.1 &rarr; >=3.2

Removed:

- `pyepsg`

### Added

- `climada.hazard.tc_tracks.TCTracks.from_FAST` function, add Australia basin (AU) [#993](https://github.com/CLIMADA-project/climada_python/pull/993)
- Add `osm-flex` package to CLIMADA core [#981](https://github.com/CLIMADA-project/climada_python/pull/981)
- `doc.tutorial.climada_entity_Exposures_osm.ipynb` tutorial explaining how to use `osm-flex`with CLIMADA
- `climada.util.coordinates.bounding_box_global` function [#980](https://github.com/CLIMADA-project/climada_python/pull/980)
- `climada.util.coordinates.bounding_box_from_countries` function [#980](https://github.com/CLIMADA-project/climada_python/pull/980)
- `climada.util.coordinates.bounding_box_from_cardinal_bounds` function [#980](https://github.com/CLIMADA-project/climada_python/pull/980)
- `climada.engine.impact.Impact.local_return_period` method [#971](https://github.com/CLIMADA-project/climada_python/pull/971)
- `doc.tutorial.climada_util_local_exceedance_values.ipynb` tutorial explaining `Hazard.local_exceedance_intensity`, `Hazard.local_return_period`, `Impact.local_exceedance_impact`, and `Impact.local_return_period` methods [#971](https://github.com/CLIMADA-project/climada_python/pull/971)
- `Hazard.local_exceedance_intensity`, `Hazard.local_return_period` and `Impact.local_exceedance_impact`, that all use the `climada.util.interpolation` module [#918](https://github.com/CLIMADA-project/climada_python/pull/918)
- `climada.util.interpolation` module for inter- and extrapolation util functions used in local exceedance intensity and return period functions [#930](https://github.com/CLIMADA-project/climada_python/pull/930)
- `climada.exposures.exposures.Exposures.geometry` property
- `climada.exposures.exposures.Exposures.latitude` property
- `climada.exposures.exposures.Exposures.longitude` property
- `climada.exposures.exposures.Exposures.value` property
- `climada.exposures.exposures.Exposures.region_id` property
- `climada.exposures.exposures.Exposures.category_id` property
- `climada.exposures.exposures.Exposures.cover` property
- `climada.exposures.exposures.Exposures.hazard_impf` method
- `climada.exposures.exposures.Exposures.hazard_centroids` method

### Changed

- `Centroids.append` now takes multiple arguments and provides a performance boost when doing so [#989](https://github.com/CLIMADA-project/climada_python/pull/989)
- `climada.util.coordinates.get_country_geometries` function: Now throwing a ValueError if unregognized ISO country code is given (before, the invalid ISO code was ignored) [#980](https://github.com/CLIMADA-project/climada_python/pull/980)
- Improved scaling factors implemented in `climada.hazard.trop_cyclone.apply_climate_scenario_knu` to model the impact of climate changes to tropical cyclones [#734](https://github.com/CLIMADA-project/climada_python/pull/734)
- In `climada.util.plot.geo_im_from_array`, NaNs are plotted in gray while cells with no centroid are not plotted [#929](https://github.com/CLIMADA-project/climada_python/pull/929)
- Renamed `climada.util.plot.subplots_from_gdf` to `climada.util.plot.plot_from_gdf` [#929](https://github.com/CLIMADA-project/climada_python/pull/929)
- `Hazard.local_exceedance_inten`, `Hazard.local_return_period`, and `Impact.local_exceedance_imp` call the corresponding new functions and a deprecation warning is added [#918](https://github.com/CLIMADA-project/climada_python/pull/918). Some inconsistencies in the previous versions are removed and the default method is changed. To reconstruct results from the previous versions, use CLIMADA v5.0.0 or less.
- elements of `event_name` are now explicitly converted to `str` in `from_raster`, `from_xarray_raster`, `from_excel` and `from_csv`. [#951](https://github.com/CLIMADA-project/climada_python/pull/951), [#910](https://github.com/CLIMADA-project/climada_python/issues/910)
- `event_id` and `event_name` are now explicitly converted to respectively a `np.ndarray` (`event_id`), a `list` (`event_name`) in readers. [#951](https://github.com/CLIMADA-project/climada_python/pull/951), [#950](https://github.com/CLIMADA-project/climada_python/issues/950)
- Exposures complete overhaul. Notably
- the _geometry_ column of the inherent `GeoDataFrame` is set up at initialization
- latitude and longitude column are no longer present there (the according arrays can be retrieved as properties of the Exposures object: `exp.latitude` instead of `exp.gdf.latitude.values`).
- `Exposures.gdf` has been renamed to `Exposures.data` (it still works though, as it is a property now pointing to the latter)
- the `check` method does not add a default "IMPF_" column to the GeoDataFrame anymore
- Updated IBTrACS version from v4.0 to v4.1 ([#976](https://github.com/CLIMADA-project/climada_python/pull/976)
- Fix xarray future warning in TCTracks for .dims to .sizes
- Fix hazard.concatenate type test for pathos pools

### Fixed

- Resolved an issue where windspeed computation was much slower than in Climada v3 [#989](https://github.com/CLIMADA-project/climada_python/pull/989)
- File handles are being closed after reading netcdf files with `climada.hazard` modules [#953](https://github.com/CLIMADA-project/climada_python/pull/953)
- Avoids a ValueError in the impact calculation for cases with a single exposure point and MDR values of 0, by explicitly removing zeros in `climada.hazard.Hazard.get_mdr` [#933](https://github.com/CLIMADA-project/climada_python/pull/948)

### Deprecated

- `climada.hazard.trop_cyclone.trop_cyclone_windfields.compute_angular_windspeeds.cyclostrophic` argument
- `climada.entity.exposures.Exposures.meta` attribute
- `climada.entity.exposures.Exposures.set_lat_lon` method
- `climada.entity.exposures.Exposures.set_geometry_points` method
- `climada.hazard.Hazard.local_exceedance_inten` method
- `climada.hazard.Hazard.plot_rp_intensity` method
- `climada.engine.impact.Impact.local_exceedance_imp` method
- `climada.engine.impact.Impact.plot_rp_imp` method

## 5.0.0

Release date: 2024-07-19

### Dependency Changes

Added:

- `bayesian-optimization`
- `seaborn` >=0.13

Updated:

- `bottleneck` >=1.3 &rarr; >=1.4
- `cartopy` >=0.22 &rarr; >=0.23
- `contextily` >=1.5 &rarr; >=1.6
- `dask` >=2024.1,<2024.3 &rarr; >=2024.2,<2024.3
- `matplotlib-base` >=3.8 &rarr; >=3.9
- `numba` >=0.59 &rarr; >=0.60
- `numexpr` >=2.9 &rarr; >=2.10
- `pint` >=0.23 &rarr; >=0.24
- `pycountry` >=22.3 &rarr; >=24.6
- `requests` >=2.31 &rarr; >=2.32
- `salib` >=1.4 &rarr; >=1.5
- `scikit-learn` >=1.4 &rarr; >=1.5
- `scipy` >=1.12 &rarr; >=1.13
- `xarray` >=2024.2 &rarr; >=2024.6

### Added

- GitHub actions workflow for CLIMADA Petals compatibility tests [#855](https://github.com/CLIMADA-project/climada_python/pull/855)
- `climada.util.calibrate` module for calibrating impact functions [#692](https://github.com/CLIMADA-project/climada_python/pull/692)
- Method `Hazard.check_matrices` for bringing the stored CSR matrices into "canonical format" [#893](https://github.com/CLIMADA-project/climada_python/pull/893)
- Generic s-shaped impact function via `ImpactFunc.from_poly_s_shape` [#878](https://github.com/CLIMADA-project/climada_python/pull/878)
- climada.hazard.centroids.centr.Centroids.get_area_pixel
- climada.hazard.centroids.centr.Centroids.get_dist_coast
- climada.hazard.centroids.centr.Centroids.get_elevation
- climada.hazard.centroids.centr.Centroids.get_meta
- climada.hazard.centroids.centr.Centroids.get_pixel_shapes
- climada.hazard.centroids.centr.Centroids.to_crs
- climada.hazard.centroids.centr.Centroids.to_default_crs
- climada.hazard.centroids.centr.Centroids.write_csv
- climada.hazard.centroids.centr.Centroids.write_excel
- climada.hazard.local_return_period [#898](https://github.com/CLIMADA-project/climada_python/pull/898)
- climada.util.plot.subplots_from_gdf [#898](https://github.com/CLIMADA-project/climada_python/pull/898)

### Changed

- Use Geopandas GeoDataFrame.plot() for centroids plotting function [896](https://github.com/CLIMADA-project/climada_python/pull/896)
- Update SALib sensitivity and sampling methods from newest version (SALib 1.4.7) [#828](https://github.com/CLIMADA-project/climada_python/issues/828)
- Allow for computation of relative and absolute delta impacts in `CalcDeltaClimate`
- Remove content tables and make minor improvements (fix typos and readability) in
CLIMADA tutorials. [#872](https://github.com/CLIMADA-project/climada_python/pull/872)
- Centroids complete overhaul. Most function should be backward compatible. Internal data is stored in a geodataframe attribute. Raster are now stored as points, and the meta attribute is removed. Several methds were deprecated or removed. [#787](https://github.com/CLIMADA-project/climada_python/pull/787)
- Improved error messages produced by `ImpactCalc.impact()` in case impact function in the exposures is not found in impf_set [#863](https://github.com/CLIMADA-project/climada_python/pull/863)
- Update the Holland et al. 2010 TC windfield model and introduce `model_kwargs` parameter to adjust model parameters [#846](https://github.com/CLIMADA-project/climada_python/pull/846)
- Changed module structure: `climada.hazard.Hazard` has been split into the modules `base`, `io` and `plot` [#871](https://github.com/CLIMADA-project/climada_python/pull/871)
- Ensure `csr_matrix` stored in `climada.hazard.Hazard` have consistent data format and store no explicit zeros when initializing `ImpactCalc` [#893](https://github.com/CLIMADA-project/climada_python/pull/893)
- `Impact.from_hdf5` now calls `str` on `event_name` data that is not strings, and issue a warning then [#894](https://github.com/CLIMADA-project/climada_python/pull/894)
- `Impact.write_hdf5` now throws an error if `event_name` is does not contain strings exclusively [#894](https://github.com/CLIMADA-project/climada_python/pull/894)
- Split `climada.hazard.trop_cyclone` module into smaller submodules without affecting module usage [#911](https://github.com/CLIMADA-project/climada_python/pull/911)

### Fixed

- Avoid an issue where a Hazard subselection would have a fraction matrix with only zeros as entries by throwing an error [#866](https://github.com/CLIMADA-project/climada_python/pull/866)
- Allow downgrading the Python bugfix version to improve environment compatibility [#900](https://github.com/CLIMADA-project/climada_python/pull/900)
- Fix broken links in `CONTRIBUTING.md` [#900](https://github.com/CLIMADA-project/climada_python/pull/900)
- When writing `TCTracks` to NetCDF, only apply compression to `float` or `int` data types. This fixes a downstream issue, see [climada_petals#135](https://github.com/CLIMADA-project/climada_petals/issues/135) [#911](https://github.com/CLIMADA-project/climada_python/pull/911)

### Deprecated

- climada.hazard.centroids.centr.Centroids.from_lat_lon
- climada.hazard.centroids.centr.Centroids.def set_area_pixel
- climada.hazard.centroids.centr.Centroids.def set_area_approx
- climada.hazard.centroids.centr.Centroids.set_dist_coast
- climada.hazard.centroids.centr.Centroids.empty_geometry_points
- climada.hazard.centroids.centr.Centroids.set_meta_to_lat_lon
- climada.hazard.centroids.centr.Centroids.set_lat_lon_to_meta
- `scheduler` parameter in `climada.util.coordinates.set_df_geometry_points`, as dask is not used anymore, leaving all calculation to shapely [#912](https://github.com/CLIMADA-project/climada_python/pull/912)

### Removed

- climada.hazard.base.Hazard.clear
- climada.hazard.base.Hazard.from_mat
- climada.hazard.base.Hazard.raster_to_vector
- climada.hazard.base.Hazard.read_mat
- climada.hazard.base.Hazard.reproject_raster
- climada.hazard.base.Hazard.set_vector
- climada.hazard.base.Hazard.vector_to_raster
- climada.hazard.centroids.centr.Centroids.calc_pixels_polygons
- climada.hazard.centroids.centr.Centroids.check
- climada.hazard.centroids.centr.Centroids.clear
- climada.hazard.centroids.centr.Centroids.equal
- climada.hazard.centroids.centr.Centroids.from_mat
- climada.hazard.centroids.centr.Centroids.from_base_grid
- climada.hazard.centroids.centr.Centroids.read_excel
- climada.hazard.centroids.centr.Centroids.read_hdf5
- climada.hazard.centroids.centr.Centroids.read_mat
- climada.hazard.centroids.centr.Centroids.set_elevation
- climada.hazard.centroids.centr.Centroids.set_geometry_points
- climada.hazard.centroids.centr.Centroids.set_lat_lon
- climada.hazard.centroids.centr.Centroids.set_raster_file
- climada.hazard.centroids.centr.Centroids.set_raster_from_pnt_bounds
- climada.hazard.centroids.centr.Centroids.set_vector_file
- climada.hazard.centroids.centr.Centroids.values_from_raster_files
- climada.hazard.centroids.centr.Centroids.values_from_vector_files
- climada.hazard.centroids.centr.generate_nat_earth_centroids
- `requirements/env_docs.yml`. The regular environment specs are now used to build the online documentation [#687](https://github.com/CLIMADA-project/climada_python/pull/687)

## 4.1.1

Release date: 2024-02-21

### Fixed

- Fix `util.coordinates.latlon_bounds` for cases where the specified buffer is very large so that the bounds cover more than the full longitudinal range `[-180, 180]` [#839](https://github.com/CLIMADA-project/climada_python/pull/839)
- Fix `climada.hazard.trop_cyclone` for TC tracks crossing the antimeridian [#839](https://github.com/CLIMADA-project/climada_python/pull/839)

## 4.1.0

Release date: 2024-02-14

### Dependency Changes

Added:

- `pyproj` >=3.5
- `numexpr` >=2.9

Updated:

- `contextily` >=1.3 &rarr; >=1.5
- `dask` >=2023 &rarr; >=2024
- `numba` >=0.57 &rarr; >=0.59
- `pandas` >=2.1 &rarr; >=2.1,<2.2
- `pint` >=0.22 &rarr; >=0.23
- `scikit-learn` >=1.3 &rarr; >=1.4
- `scipy` >=1.11 &rarr; >=1.12
- `sparse` >=0.14 &rarr; >=0.15
- `xarray` >=2023.8 &rarr; >=2024.1
- `overpy` =0.6 &rarr; =0.7
- `peewee` =3.16.3 &rarr; =3.17.1

Removed:

- `proj` (in favor of `pyproj`)

### Added

- Convenience method `api_client.Client.get_dataset_file`, combining `get_dataset_info` and `download_dataset`, returning a single file objet. [#821](https://github.com/CLIMADA-project/climada_python/pull/821)
- Read and Write methods to and from csv files for the `DiscRates` class. [#818](ttps://github.com/CLIMADA-project/climada_python/pull/818)
- Add `CalcDeltaClimate` to unsequa module to allow uncertainty and sensitivity analysis of impact change calculations [#844](https://github.com/CLIMADA-project/climada_python/pull/844)
- Add function `safe_divide` in util which handles division by zero and NaN values in the numerator or denominator [#844](https://github.com/CLIMADA-project/climada_python/pull/844)
- Add reset_frequency option for the impact.select() function. [#847](https://github.com/CLIMADA-project/climada_python/pull/847)

### Changed

- Update Developer and Installation Guides for easier accessibility by new developers. [808](https://github.com/CLIMADA-project/climada_python/pull/808)
- Add `shapes` argument to `geo_im_from_array` to allow flexible turning on/off of plotting coastline in `plot_intensity`. [#805](https://github.com/CLIMADA-project/climada_python/pull/805)
- Update `CONTRIBUTING.md` to better explain types of contributions to this repository [#797](https://github.com/CLIMADA-project/climada_python/pull/797)
- The default tile layer in Exposures maps is not Stamen Terrain anymore, but [CartoDB Positron](https://github.com/CartoDB/basemap-styles). Affected methods are `climada.engine.Impact.plot_basemap_eai_exposure`,`climada.engine.Impact.plot_basemap_impact_exposure` and `climada.entity.Exposures.plot_basemap`. [#798](https://github.com/CLIMADA-project/climada_python/pull/798)
- Recommend using Mamba instead of Conda for installing CLIMADA [#809](https://github.com/CLIMADA-project/climada_python/pull/809)
- `Hazard.from_xarray_raster` now allows arbitrary values as 'event' coordinates [#837](https://github.com/CLIMADA-project/climada_python/pull/837)
- `climada.test.get_test_file` now compares the version of the requested test dataset with the version of climada itself and selects the most appropriate dataset. In this way a test file can be updated without the need of changing the code of the unittest. [#822](https://github.com/CLIMADA-project/climada_python/pull/822)
- Explicitly require `pyproj` instead of `proj` (the latter is now implicitly required) [#845](https://github.com/CLIMADA-project/climada_python/pull/845)

### Fixed

- `Hazard.from_xarray_raster` now stores strings as default values for `Hazard.event_name` [#795](https://github.com/CLIMADA-project/climada_python/pull/795)
- Fix the dist_approx util function when used with method="geosphere" and log=True and points that are very close. [#792](https://github.com/CLIMADA-project/climada_python/pull/792)
- `climada.util.yearsets.sample_from_poisson`: fix a bug ([#819](https://github.com/CLIMADA-project/climada_python/issues/819)) and inconsistency that occurs when lambda events per year (`lam`) are set to 1. [[#823](https://github.com/CLIMADA-project/climada_python/pull/823)]
- In the TropCyclone class in the Holland model 2008 and 2010 implementation, a doublecounting of translational velocity is removed [#833](https://github.com/CLIMADA-project/climada_python/pull/833)
- `climada.util.test.test_finance` and `climada.test.test_engine` updated to recent input data from worldbank [#841](https://github.com/CLIMADA-project/climada_python/pull/841)
- Set `nodefaults` in Conda environment specs because `defaults` are not compatible with conda-forge [#845](https://github.com/CLIMADA-project/climada_python/pull/845)
- Avoid redundant calls to `np.unique` in `Impact.impact_at_reg` [#848](https://github.com/CLIMADA-project/climada_python/pull/848)

## 4.0.1

Release date: 2023-09-27

### Dependency Changes

Added:

- `matplotlib-base` None &rarr; >=3.8

Changed:

- `geopandas` >=0.13 &rarr; >=0.14
- `pandas` >=1.5,<2.0 &rarr; >=2.1
- `salib` >=1.3.0 &rarr; >=1.4.7

Removed:

- `matplotlib` >=3.7

### Changed

- Rearranged file-system structure: `data` directory moved into `climada` package directory. [#781](https://github.com/CLIMADA-project/climada_python/pull/781)

### Fixed

- `climada.util.coordinates.get_country_code` bug, occurring with non-standard longitudinal coordinates around the anti-meridian. [#770](https://github.com/CLIMADA-project/climada_python/issues/770)

## 4.0.0

Release date: 2023-09-01

### Dependency Updates

Added:

- `pytest` [#726](https://github.com/CLIMADA-project/climada_python/pull/726)
- `pytest-cov` [#726](https://github.com/CLIMADA-project/climada_python/pull/726)
- `pytest-subtests` [#726](https://github.com/CLIMADA-project/climada_python/pull/726)
- `unittest-xml-reporting`

Changed:

- `cartopy` >=0.20.0,<0.20.3 &rarr; >=0.21
- `cfgrib` >=0.9.7,<0.9.10 &rarr; =0.9.9
- `contextily` >=1.0 &rarr; >=1.3
- `dask` >=2.25 &rarr; >=2023
- `eccodes` [auto] &rarr; =2.27
- `gdal` !=3.4.1 &rarr; >=3.6
- `geopandas` >=0.8 &rarr; >=0.13
- `h5py` >=2.10 &rarr; >=3.8
- `haversine` >=2.3 &rarr; >=2.8
- `matplotlib` >=3.2,< 3.6 &rarr; >=3.7
- `netcdf4` >=1.5 &rarr; >=1.6
- `numba` >=0.51,!=0.55.0 &rarr; >=0.57
- `openpyxl` >=3.0 &rarr; >=3.1
- `pandas-datareader` >=0.9 &rarr; >=0.10
- `pathos` >=0.2 &rarr; >=0.3
- `pint` >=0.15 &rarr; >=0.22
- `proj` !=9.0.0 &rarr; >=9.1
- `pycountry` >=20.7 &rarr; >=22.3
- `pytables` >=3.6 &rarr; >=3.7
- `rasterio` >=1.2.7,<1.3 &rarr; >=1.3
- `requests` >=2.24 &rarr; >=2.31
- `salib` >=1.3.0 &rarr; >=1.4
- `scikit-learn` >=1.0 &rarr; >=1.2
- `scipy` >=1.6 &rarr; >=1.10
- `sparse` >=0.13 &rarr; >=0.14
- `statsmodels` >=0.11 &rarr; >=0.14
- `tabulate` >=0.8 &rarr; >=0.9
- `tqdm` >=4.48 &rarr; >=4.65
- `xarray` >=0.13 &rarr; >=2023.5
- `xlrd` >=1.2 &rarr; >=2.0
- `xlsxwriter` >=1.3 &rarr; >=3.1

Removed:

- `nbsphinx` [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- `pandoc` [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- `xmlrunner`

### Added

- `Impact.impact_at_reg` method for aggregating impacts per country or custom region [#642](https://github.com/CLIMADA-project/climada_python/pull/642)
- `Impact.match_centroids` convenience method for matching (hazard) centroids to impact objects [#602](https://github.com/CLIMADA-project/climada_python/pull/602)
- `climada.util.coordinates.match_centroids` method for matching (hazard) centroids to GeoDataFrames [#602](https://github.com/CLIMADA-project/climada_python/pull/602)
- 'Extra' requirements `doc`, `test`, and `dev` for Python package [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- Added method `Exposures.centroids_total_value` to replace the functionality of `Exposures.affected_total_value`. This method is temporary and deprecated. [#702](https://github.com/CLIMADA-project/climada_python/pull/702)
- New method `climada.util.api_client.Client.purge_cache`: utility function to remove outdated files from the local file system to free disk space.
([#737](https://github.com/CLIMADA-project/climada_python/pull/737))
- New attribute `climada.hazard.Hazard.haz_type`: used for assigning impacts to hazards. In previous versions this information was stored in the now removed `climada.hazard.tag.Tag` class. [#736](https://github.com/CLIMADA-project/climada_python/pull/736)
- New attribute `climada.entity.exposures.Exposures.description`: used for setting the default title in plots from plotting mathods `plot_hexbin` and `plot_scatter`. In previous versions this information was stored in the deprecated `climada.entity.tag.Tag` class. [#756](https://github.com/CLIMADA-project/climada_python/pull/756)
- Added advanced examples in unsequa tutorial for coupled input variables and for handling efficiently the loading of multiple large files [#766](https://github.com/CLIMADA-project/climada_python/pull/766)

### Changed

- Improved error messages from `climada.CONFIG` in case of missing configuration values [#670](https://github.com/CLIMADA-project/climada_python/pull/670)
- Refactored `Exposure.assign_centroids` using a new util function `u_coord.match_centroids` [#602](https://github.com/CLIMADA-project/climada_python/pull/602)
- Renamed `climada.util.coordinate.assign_grid_points` to `match_grid_points` and `climada.util.coordinates.assign_coordinates` to `match_coordinates`
[#602](https://github.com/CLIMADA-project/climada_python/pull/602)
- Modified the method to disaggregate lines in the `lines_polys_handler` utility module in order to better conserve the total length of all lines on average [#679](https://github.com/CLIMADA-project/climada_python/pull/679).
- Added test for non-default impact function id in the `lines_polys_handler` [#676](https://github.com/CLIMADA-project/climada_python/pull/676)
- The sigmoid and step impact functions now require the user to define the hazard type. [#675](https://github.com/CLIMADA-project/climada_python/pull/675)
- Improved error messages produced by `ImpactCalc.impact()` in case hazard type is not found in exposures/impf_set [#691](https://github.com/CLIMADA-project/climada_python/pull/691)
- Tests with long runtime were moved to integration tests in `climada/test` [#709](https://github.com/CLIMADA-project/climada_python/pull/709)
- Use `myst-nb` for parsing Jupyter Notebooks for the documentation instead of `nbsphinx` [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- Installation guide now recommends installing CLIMADA directly via `conda install` [#714](https://github.com/CLIMADA-project/climada_python/pull/714)
- `Exposures.affected_total_value` now takes a hazard intensity threshold as argument. Affected values are only those for which at least one event exceeds the threshold. (previously, all exposures points with an assigned centroid were considered affected). By default the centroids are reassigned. [#702](https://github.com/CLIMADA-project/climada_python/pull/702) [#730](https://github.com/CLIMADA-project/climada_python/pull/730)
- Add option to pass region ID to `LitPop.from_shape` [#720](https://github.com/CLIMADA-project/climada_python/pull/720)
- Slightly improved performance on `LitPop`-internal computations [#720](https://github.com/CLIMADA-project/climada_python/pull/720)
- Use `pytest` for executing tests [#726](https://github.com/CLIMADA-project/climada_python/pull/726)
- Users can opt-out of the climada specific logging definitions and freely configure logging to their will, by setting the config value `logging.managed` to `false`. [#724](https://github.com/CLIMADA-project/climada_python/pull/724)
- Add option to read additional variables from IBTrACS when using `TCTracks.from_ibtracs_netcdf` [#728](https://github.com/CLIMADA-project/climada_python/pull/728)
- New file format for `TCTracks` I/O with better performance. This change is not backwards compatible: If you stored `TCTracks` objects with `TCTracks.write_hdf5`, reload the original data and store them again. [#735](https://github.com/CLIMADA-project/climada_python/pull/735)
- Add option to load only a subset when reading TC tracks using `TCTracks.from_simulations_emanuel`. [#741](https://github.com/CLIMADA-project/climada_python/pull/741)
- Set `save_mat` to `False` in the `unsequa` module [#746](https://github.com/CLIMADA-project/climada_python/pull/746)
- `list_dataset_infos` from `climada.util.api_client.Client`: the `properties` argument, a `dict`, can now have `None` as values. Before, only strings and lists of strings were allowed. Setting a particular property to `None` triggers a search for datasets where this property is not assigned. [#752](https://github.com/CLIMADA-project/climada_python/pull/752)
- Reduce memory requirements of `TropCyclone.from_tracks` [#749](https://github.com/CLIMADA-project/climada_python/pull/749)
- Support for different wind speed and pressure units in `TCTracks` when running `TropCyclone.from_tracks` [#749](https://github.com/CLIMADA-project/climada_python/pull/749)
- The title of plots created by the `Exposures` methods `plot_hexbin` and `plot_scatter` can be set as a method argument. [#756](https://github.com/CLIMADA-project/climada_python/pull/756)
- Changed the parallel package from Pathos to Multiproess in the unsequa module [#763](https://github.com/CLIMADA-project/climada_python/pull/763)
- Updated installation instructions to use conda for core and petals [#776](https://github.com/CLIMADA-project/climada_python/pull/776)

### Fixed

- `util.lines_polys_handler` solve polygon disaggregation issue in metre-based projection [#666](https://github.com/CLIMADA-project/climada_python/pull/666)
- Problem with `pyproj.CRS` as `Impact` attribute, [#706](https://github.com/CLIMADA-project/climada_python/issues/706). Now CRS is always stored as `str` in WKT format.
- Correctly handle assertion errors in `Centroids.values_from_vector_files` and fix the associated test [#768](https://github.com/CLIMADA-project/climada_python/pull/768/)
- Text in `Forecast` class plots can now be adjusted [#769](https://github.com/CLIMADA-project/climada_python/issues/769)
- `Impact.impact_at_reg` now supports impact matrices where all entries are zero [#773](https://github.com/CLIMADA-project/climada_python/pull/773)
- upgrade pathos 0.3.0 -> 0.3.1 issue [#761](https://github.com/CLIMADA-project/climada_python/issues/761) (for unsequa module [#763](https://github.com/CLIMADA-project/climada_python/pull/763))
- Fix bugs with pandas 2.0 (iteritems -> items, append -> concat) (fix issue [#700](https://github.com/CLIMADA-project/climada_python/issues/700) for unsequa module) [#763](https://github.com/CLIMADA-project/climada_python/pull/763))
- Remove matplotlib styles in unsequa module (fixes issue [#758](https://github.com/CLIMADA-project/climada_python/issues/758)) [#763](https://github.com/CLIMADA-project/climada_python/pull/763)

### Deprecated

- `Centroids.from_geodataframe` and `Centroids.from_pix_bounds` [#721](https://github.com/CLIMADA-project/climada_python/pull/721)
- `Impact.tot_value`: Use `Exposures.affected_total_value` to compute the total value affected by a hazard intensity above a custom threshold [#702](https://github.com/CLIMADA-project/climada_python/pull/702)
- `climada.entity.tag.Tag`. [#779](https://github.com/CLIMADA-project/climada_python/pull/779). The class is not used anymore but had to be kept for reading Exposures HDF5 files that were created with previous versions of CLIMADA.

### Removed

- `Centroids.set_raster_from_pix_bounds` [#721](https://github.com/CLIMADA-project/climada_python/pull/721)
- `requirements/env_developer.yml` environment specs. Use 'extra' requirements when installing the Python package instead [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- The `climada.entitity.tag.Tag` class, together with `Impact.tag`, `Exposures.tag`, `ImpactFuncSet.tag`, `MeasuresSet.tag`, `Hazard.tag` attributes.
This may break backwards-compatibility with respect to the files written and read by the `Impact` class.
[#736](https://github.com/CLIMADA-project/climada_python/pull/736),
[#743](https://github.com/CLIMADA-project/climada_python/pull/743),
[#753](https://github.com/CLIMADA-project/climada_python/pull/753),
[#754](https://github.com/CLIMADA-project/climada_python/pull/754),
[#756](https://github.com/CLIMADA-project/climada_python/pull/756),
[#767](https://github.com/CLIMADA-project/climada_python/pull/767),
[#779](https://github.com/CLIMADA-project/climada_python/pull/779)
- `impact.tot_value` attribute removed from unsequa module [#763](https://github.com/CLIMADA-project/climada_python/pull/763)

## v3.3.2

Release date: 2023-03-02

### Dependency Updates

Removed:

- `pybufrkit` [#662](https://github.com/CLIMADA-project/climada_python/pull/662)

## v3.3.1

Release date: 2023-02-27

### Description

Patch-relaese with altered base config file so that the basic installation test passes.

### Changed

- The base config file `climada/conf/climada.conf` has an entry for `CONFIG.hazard.test_data`.

## v3.3.0

Release date: 2023-02-17

### Dependency Changes

new:

- sparse (>=0.13) for [#578](https://github.com/CLIMADA-project/climada_python/pull/578)

updated:

- **python 3.9** - python 3.8 will still work, but python 3.9 is now the default version for [installing climada](https://climada-python.readthedocs.io/en/latest/tutorial/climada_installation_step_by_step.html) ([#614](https://github.com/CLIMADA-project/climada_python/pull/614))
- contextily >=1.0 (no longer restricted to <1.2 as `contextily.sources` has been replaced in [#517](https://github.com/CLIMADA-project/climada_python/pull/517))
- cartopy >=0.20.0,<0.20.3 (>=0.20.3 has an issue with geographic crs in plots)
- matplotlib >=3.2,<3.6 (3.6 depends on cartopy 0.21)

### Added

- `climada.hazard.Hazard.from_xarray_raster(_file)` class methods for reading `Hazard` objects from an `xarray.Dataset`, or from a file that can be read by `xarray`.
[#507](https://github.com/CLIMADA-project/climada_python/pull/507),
[#589](https://github.com/CLIMADA-project/climada_python/pull/589),
[#652](https://github.com/CLIMADA-project/climada_python/pull/652).
- `climada.engine.impact.Impact` objects have new methods `from_hdf5` and `write_hdf5` for reading their data from, and writing it to, H5 files [#606](https://github.com/CLIMADA-project/climada_python/pull/606)
- `climada.engine.impact.Impact` objects has a new class method `concat` for concatenation of impacts based on the same exposures [#529](https://github.com/CLIMADA-project/climada_python/pull/529).
- `climada.engine.impact_calc`: this module was separated from `climada.engine.impact` and contains the code that dealing with impact _calculation_ while the latter focuses on impact _data_ [#560](https://github.com/CLIMADA-project/climada_python/pull/560).
- The classes `Hazard`, `Impact` and `ImpactFreqCurve` have a novel attribute `frequency_unit`. Before it was implicitly set to annual, now it can be specified and accordingly displayed in plots.
[#532](https://github.com/CLIMADA-project/climada_python/pull/532).
- CONTRIBUTING.md [#518](https://github.com/CLIMADA-project/climada_python/pull/518).
- Changelog based on the CLIMADA release overview and https://keepachangelog.com template [#626](https://github.com/CLIMADA-project/climada_python/pull/626).

### Changed

- The `Impact` calculation underwent a major refactoring. Now the suggested way to run an impact calculation is by `climada.engine.impact_calc.ImpactCalc.impact()`.
[#436](https://github.com/CLIMADA-project/climada_python/pull/436),
[#527](https://github.com/CLIMADA-project/climada_python/pull/527).
- Addition of uncertainty helper methods variables: list of hazard, list of impact function sets, and hazard fraction. This allows to pre-compute hazards or impact function sets from different sources from which one can then sample uniformly. [#513](https://github.com/CLIMADA-project/climada_python/pull/513)
- Full initialization of most Climada objects is now possible (and suggested!) in one step, by simply calling the constructor with all arguments required for coherently filling the object with data:
[#560](https://github.com/CLIMADA-project/climada_python/pull/560),
[#553](https://github.com/CLIMADA-project/climada_python/pull/553),
[#550](https://github.com/CLIMADA-project/climada_python/pull/550),
[#564](https://github.com/CLIMADA-project/climada_python/pull/564),
[#563](https://github.com/CLIMADA-project/climada_python/pull/563),
[#565](https://github.com/CLIMADA-project/climada_python/pull/565),
[#573](https://github.com/CLIMADA-project/climada_python/pull/573),
[#569](https://github.com/CLIMADA-project/climada_python/pull/569),
[#570](https://github.com/CLIMADA-project/climada_python/pull/570),
[#574](https://github.com/CLIMADA-project/climada_python/pull/574),
[#559](https://github.com/CLIMADA-project/climada_python/pull/559),
[#571](https://github.com/CLIMADA-project/climada_python/pull/571),
[#549](https://github.com/CLIMADA-project/climada_python/pull/549),
[#567](https://github.com/CLIMADA-project/climada_python/pull/567),
[#568](https://github.com/CLIMADA-project/climada_python/pull/568),
[#562](https://github.com/CLIMADA-project/climada_python/pull/562).
- It is possible now to set the `fraction` of a `Hazard` object to `None` which will have the same effect as if it were `1` everywhere. This saves a lot of memory and calculation time, [#541](https://github.com/CLIMADA-project/climada_python/pull/541).
- The online documentation has been completely overhauled:
[#597](https://github.com/CLIMADA-project/climada_python/pull/597),
[#600](https://github.com/CLIMADA-project/climada_python/pull/600),
[#609](https://github.com/CLIMADA-project/climada_python/pull/609),
[#620](https://github.com/CLIMADA-project/climada_python/pull/620),
[#615](https://github.com/CLIMADA-project/climada_python/pull/615),
[#617](https://github.com/CLIMADA-project/climada_python/pull/617),
[#622](https://github.com/CLIMADA-project/climada_python/pull/622),
[#656](https://github.com/CLIMADA-project/climada_python/pull/656).
- Updated installation instructions [#644](https://github.com/CLIMADA-project/climada_python/pull/644)

### Fixed

- several antimeridian issues:
[#524](https://github.com/CLIMADA-project/climada_python/pull/524),
[#551](https://github.com/CLIMADA-project/climada_python/pull/551),
[#613](https://github.com/CLIMADA-project/climada_python/pull/613).
- bug in `climada.hazard.Centroids.set_on_land()` when coordinates go around the globe:
[#542](https://github.com/CLIMADA-project/climada_python/pull/542),
[#543](https://github.com/CLIMADA-project/climada_python/pull/543).
- bug in `climada.util.coordinates.get_country_code()` when all coordinates are on sea.
- suppress pointless warnings in plotting functions, [#520](https://github.com/CLIMADA-project/climada_python/pull/520).
- test coverage improved:
[#583](https://github.com/CLIMADA-project/climada_python/pull/583),
[#594](https://github.com/CLIMADA-project/climada_python/pull/594),
[#608](https://github.com/CLIMADA-project/climada_python/pull/608),
[#616](https://github.com/CLIMADA-project/climada_python/pull/616),
[#637](https://github.com/CLIMADA-project/climada_python/pull/637).
- deprecated features removoed:
[#517](https://github.com/CLIMADA-project/climada_python/pull/517),
[#535](https://github.com/CLIMADA-project/climada_python/pull/535),
[#566](https://github.com/CLIMADA-project/climada_python/pull/566),

### Deprecated

- `climada.enginge.impact.Impact.calc()` and `climada.enginge.impact.Impact.calc_impact_yearset()`
[#436](https://github.com/CLIMADA-project/climada_python/pull/436).
