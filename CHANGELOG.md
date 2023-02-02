# Changelog

## v3.3.0 (upcoming)

Release date: 2023-02-15

Code freeze date: 2023-02-05

### Description

### Dependency Changes

new:
- sparse (>=0.13) for [#578](https://github.com/CLIMADA-project/climada_python/pull/578)

updated:
- contextily >=1.0 (no longer restricted to <1.2 as `contextily.sources` has been replaced in [#517](https://github.com/CLIMADA-project/climada_python/pull/517))
- cartopy >=0.20.0,<0.20.3 (>=0.20.3 has an issue with geographic crs in plots)
- matplotlib >=3.2,<3.6 (3.6 depends on cartopy 0.21)

### Added

- `climada.engine.impact.Impact` objects have new methods `from_hdf5` and `write_hdf5` for reading their data from, and writing it to, H5 files [#606](https://github.com/CLIMADA-project/climada_python/pull/606)
- `climada.engine.impact_calc`: this module was separated from `climada.engine.impact` and contains the code that dealing with impact _calculation_ while the latter focuses on impact _data_ [#560](https://github.com/CLIMADA-project/climada_python/pull/560).
- The classes `Hazard`, `Impact` and `ImpactFreqCurve` have a novel attribute `frequency_unit`. Before it was implicitly set to annual, now it can be specified and accordingly displayed in plots.
[#532](https://github.com/CLIMADA-project/climada_python/pull/532).
- CONTRIBUTING.md [#518](https://github.com/CLIMADA-project/climada_python/pull/518)
- Changelog based on the CLIMADA release overview and https://keepachangelog.com template [#626](https://github.com/CLIMADA-project/climada_python/pull/626).

### Changed

- The `Impact` calculation underwent a major refactoring. Now the suggested way to run an impact calculation is by `climada.engine.impact_calc.ImpactCalc.impact()`.
[#436](https://github.com/CLIMADA-project/climada_python/pull/436)
- Full initialization of most Climada objects is now possible (and suggested!) in one step, by simply calling the constructor with all arguments required for coherently filling the object with data:
[#560](https://github.com/CLIMADA-project/climada_python/pull/560)
[#553](https://github.com/CLIMADA-project/climada_python/pull/553)
[#550](https://github.com/CLIMADA-project/climada_python/pull/550)
[#564](https://github.com/CLIMADA-project/climada_python/pull/564)
[#563](https://github.com/CLIMADA-project/climada_python/pull/563)
[#565](https://github.com/CLIMADA-project/climada_python/pull/565)
[#573](https://github.com/CLIMADA-project/climada_python/pull/573)
[#569](https://github.com/CLIMADA-project/climada_python/pull/569)
[#570](https://github.com/CLIMADA-project/climada_python/pull/570)
[#574](https://github.com/CLIMADA-project/climada_python/pull/574)
[#559](https://github.com/CLIMADA-project/climada_python/pull/559)
[#571](https://github.com/CLIMADA-project/climada_python/pull/571)
[#549](https://github.com/CLIMADA-project/climada_python/pull/549)
[#567](https://github.com/CLIMADA-project/climada_python/pull/567)
[#568](https://github.com/CLIMADA-project/climada_python/pull/568)
[#562](https://github.com/CLIMADA-project/climada_python/pull/562).
- The online documentation has been completely overhauled:
[#597](https://github.com/CLIMADA-project/climada_python/pull/597)
[#600](https://github.com/CLIMADA-project/climada_python/pull/600)
[#609](https://github.com/CLIMADA-project/climada_python/pull/609)
[#620](https://github.com/CLIMADA-project/climada_python/pull/620)
[#615](https://github.com/CLIMADA-project/climada_python/pull/615)
[#617](https://github.com/CLIMADA-project/climada_python/pull/617)

### Fixed

- several antimeridian issues:
[#524](https://github.com/CLIMADA-project/climada_python/pull/524)
[#551](https://github.com/CLIMADA-project/climada_python/pull/551)
[#613](https://github.com/CLIMADA-project/climada_python/pull/613)

### Deprecated

- `climada.enginge.impact.Impact.calc()` and `climada.enginge.impact.Impact.calc_impact_yearset()`
[#436](https://github.com/CLIMADA-project/climada_python/pull/436)

### Removed
