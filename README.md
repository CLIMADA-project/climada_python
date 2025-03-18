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

It is recommend for new users to begin with the core (1) and the [tutorials](https://github.com/CLIMADA-project/climada_python/tree/main/doc/tutorial) therein.

This is the Python (3.9+) version of CLIMADA - please see [here](https://github.com/davidnbresch/climada) for backward compatibility with the MATLAB version.

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
