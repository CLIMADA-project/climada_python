[![DOI](https://zenodo.org/badge/112729129.svg)](https://zenodo.org/badge/latestdoi/112729129)
[![Build Status](http://ied-wcr-jenkins.ethz.ch/buildStatus/icon?job=climada_branches/develop)](http://ied-wcr-jenkins.ethz.ch/job/climada_branches/)
[![Documentation build status](https://img.shields.io/readthedocs/climada-python.svg?style=flat-square)](https://readthedocs.org/projects/climada-python/builds/)

# CLIMADA

[CLIMADA](https://climada.ethz.ch/climada/) (CLIMate ADAptation) is a free and open-source software framework for climate risk assessment and adaptation
option appraisal. Designed by a large scientific community, it helps reasearchers, policymakers, and businesses analyse the impacts of natural hazards and
explore adaptation strategies.

As of today, CLIMADA provides global coverage of major climate-related extreme-weather hazards at high resolution (4x4km) via a [data API](https://climada.ethz.ch/data-api/v1/docs) For select hazards, historic and probabilistic events sets, for past, present and future climate exist at distinct time horizons.
You will find a repository containing scientific peer-reviewed articles that explain software components implemented in CLIMADA [here](https://github.com/CLIMADA-project/climada_papers).

CLIMADA is divided into two parts (two repositories):

1. the core [climada_python](https://github.com/CLIMADA-project/climada_python) contains all the modules necessary for the probabilistic impact, the averted damage, uncertainty and forecast calculations. Data for hazard, exposures and impact functions can be obtained from the [data API](https://github.com/CLIMADA-project/climada_python/blob/main/doc/tutorial/climada_util_api_client.ipynb). [Litpop](https://github.com/CLIMADA-project/climada_python/blob/main/doc/tutorial/climada_entity_LitPop.ipynb) is included as demo Exposures module, and [Tropical cyclones](https://github.com/CLIMADA-project/climada_python/blob/main/doc/tutorial/climada_hazard_TropCyclone.ipynb) is included as a demo Hazard module.
2. the petals [climada_petals](https://github.com/CLIMADA-project/climada_petals) contains all the modules for generating data (e.g., TC_Surge, WildFire, OpenStreeMap, ...). Most development is done here. The petals builds-upon the core and does not work as a stand-alone.

For new users, we recommend to begin with the core (1) and the [tutorials](https://github.com/CLIMADA-project/climada_python/tree/main/doc/tutorial) therein.

This is the Python version of CLIMADA - please see [here](https://github.com/davidnbresch/climada) for backward compatibility with the MATLAB version.

## Getting started

CLIMADA runs on Windows, macOS and Linux.
The released versions of CLIMADA are available from [conda-forge](https://anaconda.org/conda-forge/climada).
Use the [Mamba](https://mamba.readthedocs.io/en/latest/) package manager to install it:

```shell
mamba install -c conda-forge climada
```

It is **highly recommended** to install CLIMADA into a **separate** Conda environment.
See the [installation guide](https://climada-python.readthedocs.io/en/latest/guide/install.html) for further information.

Follow the [tutorials](https://climada-python.readthedocs.io/en/stable/tutorial/1_main_climada.html) in a Jupyter Notebook to see what can be done with CLIMADA and how.

## New: physrisk Vulnerability Curves (Heat & Wildfire)

This repository now includes **empirically-calibrated vulnerability curves** for integration with the [OS-Climate physrisk](https://github.com/os-climate/physrisk) physical climate risk assessment framework.

### What's Included

**18 Vulnerability Curves** covering:

1. **WBGT (Heat Stress)** - 12 curves
   - 7 ISO 7243:2017 standard functions (occupational heat stress)
   - 5 empirical calibrations from peer-reviewed research
   - Asset types: Construction, manufacturing, mining, services, energy sector
   - Acclimatized and unacclimatized worker profiles

2. **FFDI (Wildfire)** - 6 curves
   - Australian Forest Fire Danger Index (McArthur 1967)
   - Calibrated from 54 bushfires, 8,256 houses (1957-2009)
   - Asset types: Residential (standard/bushfire-prone), commercial, forestry, infrastructure
   - Validated against Black Saturday 2009 (FFDI 160-190)

### Quick Start

Generate all vulnerability curves:

```bash
python vulnerability_curves_physrisk/generate_all_curves.py
```

**Output**: 18 JSON files in physrisk `VulnerabilityCurve` format, ready for climate risk assessment.

### Empirical Calibration Sources

All curves are calibrated from peer-reviewed research:

- **ISO 7243:2017** - International standard for occupational heat stress
- **Dunne et al. (2013)** *Nature Climate Change* - Labor productivity under heat
- **Kjellstrom et al. (2018)** *Int. J. Biometeorology* - Epidemiological heat studies
- **Blanchi et al. (2010)** *Int. J. Wildland Fire* - 54 bushfires, 8,256 houses
- **Krix et al. (2025)** *Int. J. Wildland Fire* - AFDRS impact index (r²=0.71)
- **Black Saturday 2009** - Victorian Bushfires Royal Commission data

### Documentation

**Comprehensive methodology documentation** (250+ pages with APA citations):

#### WBGT (Heat Stress)
- [`doc/user-guide/wbgt_impact_functions_empirical_calibration.md`](doc/user-guide/wbgt_impact_functions_empirical_calibration.md)
- [`doc/user-guide/iso7243_wbgt_standard_implementation.md`](doc/user-guide/iso7243_wbgt_standard_implementation.md)
- [`doc/user-guide/wbgt_physrisk_integration_methods.md`](doc/user-guide/wbgt_physrisk_integration_methods.md)

#### FFDI (Wildfire)
- [`doc/user-guide/ffdi_impact_functions_methodology.md`](doc/user-guide/ffdi_impact_functions_methodology.md)

#### General
- [`vulnerability_curves_physrisk/README.md`](vulnerability_curves_physrisk/README.md) - Usage guide
- [`doc/user-guide/climada_entity_physrisk_export.md`](doc/user-guide/climada_entity_physrisk_export.md) - Export workflow

### Key Features

- **physrisk converter**: Automatic CLIMADA → physrisk JSON export ([`climada/entity/impact_funcs/physrisk_converter.py`](climada/entity/impact_funcs/physrisk_converter.py))
- **Empirical validation**: All curves validated against historical events
- **Asset-specific profiles**: Different vulnerability by building type, work intensity, acclimatization
- **Uncertainty quantification**: Documented uncertainty bounds (±10-40% depending on hazard)
- **Climate scenario ready**: Direct integration with physrisk for RCP projections

### Generation Scripts

Individual curve generation scripts in [`script/applications/`](script/applications/):

- `create_wbgt_iso7243_impact_functions.py` - ISO 7243 standard curves
- `create_wbgt_impact_function.py` - Empirical WBGT calibrations
- `create_ffdi_impact_functions.py` - FFDI wildfire curves
- `create_heat_fire_impact_functions.py` - Generic heat/fire templates

### Location

All files are organized in [`vulnerability_curves_physrisk/`](vulnerability_curves_physrisk/):
- Generated JSON curves (18 files)
- `README.md` with complete usage instructions
- `generate_all_curves.py` for automated generation

---

## Documentation

The online documentation is available on [Read the Docs](https://climada-python.readthedocs.io/en/stable/).The documentation of each release version of CLIMADA can be accessed separately through the drop-down menu at the bottom of the left sidebar. Additionally, the version 'stable' refers to the most recent release (installed via `conda`), and 'latest' refers to the latest unstable development version (the `develop` branch).

CLIMADA python:

- [online (recommended)](https://climada-python.readthedocs.io/en/latest/)
- [PDF file](https://climada-python.readthedocs.io/_/downloads/en/stable/pdf/)
- [core Tutorials on GitHub](https://github.com/CLIMADA-project/climada_python/tree/main/doc/tutorial)

CLIMADA petals:

- [online (recommended)](https://climada-petals.readthedocs.io/en/latest/)
- [PDF file](https://climada-petals.readthedocs.io/_/downloads/en/stable/pdf/)
- [petals Tutorials on GitHub](https://github.com/CLIMADA-project/climada_petals/tree/main/doc/tutorial)

The documentation can also be [built locally](https://climada-python.readthedocs.io/en/latest/README.html).

## Citing CLIMADA

See the [Citation Guide](https://climada-python.readthedocs.io/en/latest/misc/citation.html).

Please use the following logo if you are presenting results obtained with or through CLIMADA:

![https://github.com/CLIMADA-project/climada_python/blob/main/doc/guide/img/CLIMADA_logo_QR.png](https://github.com/CLIMADA-project/climada_python/blob/main/doc/guide/img/CLIMADA_logo_QR.png?raw=true)

## Contributing

We welcome any contribution to this repository, be it bugfixes and other code changes and additions, documentation improvements, or tutorial updates.

If you would like to contribute, please refer to our [Contribution Guide](CONTRIBUTING.md).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [releases on this repository](https://github.com/CLIMADA-project/climada_python/releases).

## License

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License Version 3, 29 June 2007 as published by the Free Software Foundation, <https://www.gnu.org/licenses/gpl-3.0.html>

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: <https://www.gnu.org/licenses/gpl-3.0.html>
