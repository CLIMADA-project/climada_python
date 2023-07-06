# Changelog

## Unreleased

Release date: YYYY-MM-DD

Code freeze date: YYYY-MM-DD

### Dependency Updates

Added:

- `pytest` [#726](https://github.com/CLIMADA-project/climada_python/pull/726)
- `pytest-cov` [#726](https://github.com/CLIMADA-project/climada_python/pull/726)
- `pytest-subtests` [#726](https://github.com/CLIMADA-project/climada_python/pull/726)

Changed:

Removed:

- `nbsphinx` [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- `pandoc` [#712](https://github.com/CLIMADA-project/climada_python/pull/712)

### Added

- `Impact.impact_at_reg` method for aggregating impacts per country or custom region [#642](https://github.com/CLIMADA-project/climada_python/pull/642)
- `Impact.match_centroids` convenience method for matching (hazard) centroids to impact objects [#602](https://github.com/CLIMADA-project/climada_python/pull/602)
- `climada.util.coordinates.match_centroids` method for matching (hazard) centroids to GeoDataFrames [#602](https://github.com/CLIMADA-project/climada_python/pull/602)
- 'Extra' requirements `doc`, `test`, and `dev` for Python package [#712](https://github.com/CLIMADA-project/climada_python/pull/712)
- Added method `Exposures.centroids_total_value` to replace the functionality of `Exposures.affected_total_value`. This method is temporary and deprecated. [#702](https://github.com/CLIMADA-project/climada_python/pull/702)
- New method `climada.util.api_client.Client.purge_cache`: utility function to remove outdated files from the local file system to free disk space.
  ([#737](https://github.com/CLIMADA-project/climada_python/pull/737))

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
- The `haz_type` attribute has been moved from `climada.hazard.tag.Tag` to `climada.hazard.Hazard` itself. [#736](https://github.com/CLIMADA-project/climada_python/pull/736)
- New file format for `TCTracks` I/O with better performance. This change is not backwards compatible: If you stored `TCTracks` objects with `TCTracks.write_hdf5`, reload the original data and store them again. [#735](https://github.com/CLIMADA-project/climada_python/pull/735)
- Add option to load only a subset when reading TC tracks using `TCTracks.from_simulations_emanuel`. [#741](https://github.com/CLIMADA-project/climada_python/pull/741)
- Set `save_mat` to `False` in the `unsequa` module [#746](https://github.com/CLIMADA-project/climada_python/pull/746)

### Fixed

- `util.lines_polys_handler` solve polygon disaggregation issue in metre-based projection [#666](https://github.com/CLIMADA-project/climada_python/pull/666)
- Problem with `pyproj.CRS` as `Impact` attribute, [#706](https://github.com/CLIMADA-project/climada_python/issues/706). Now CRS is always stored as `str` in WKT format.

### Deprecated

- `Centroids.from_geodataframe` and `Centroids.from_pix_bounds` [#721](https://github.com/CLIMADA-project/climada_python/pull/721)
- `Impact.tot_value`: Use `Exposures.affected_total_value` to compute the total value affected by a hazard intensity above a custom threshold [#702](https://github.com/CLIMADA-project/climada_python/pull/702)
- `climada.hazard.tag.Tag` and `climada.entity.tag.Tag`. [#736](https://github.com/CLIMADA-project/climada_python/pull/736). They were unified into `climada.util.tag.Tag`. Note: the latter is to be deprecated and removed in a future version as well.

### Removed

- `Centroids.set_raster_from_pix_bounds` [#721](https://github.com/CLIMADA-project/climada_python/pull/721)

- `requirements/env_developer.yml` environment specs. Use 'extra' requirements when installing the Python package instead [#712](https://github.com/CLIMADA-project/climada_python/pull/712)

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

Code freeze date: 2023-02-05

### Description

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

### Removed
