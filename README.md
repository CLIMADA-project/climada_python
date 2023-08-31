[![DOI](https://zenodo.org/badge/112729129.svg)](https://zenodo.org/badge/latestdoi/112729129)
[![Build Status](http://ied-wcr-jenkins.ethz.ch/buildStatus/icon?job=climada_branches/develop)](http://ied-wcr-jenkins.ethz.ch/job/climada_branches/)
[![Documentation build status](https://img.shields.io/readthedocs/climada-python.svg?style=flat-square)](https://readthedocs.org/projects/climada-python/builds/)

# CLIMADA

CLIMADA stands for **CLIM**ate **ADA**ptation and is a probabilistic natural catastrophe impact model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.).

As of today, CLIMADA provides global coverage of major climate-related extreme-weather hazards at high resolution via a [data API](https://climada.ethz.ch/data-api/v1/docs) such as tropical cyclones, river flood, European winter storms, earthquakes, ..., all at 4km spatial resolution - wildfire to be added soon. For all hazards, historic and probabilistic event sets exist, for some also under select climate forcing scenarios (RCPs) at distinct time horizons (e.g. 2040). See also [papers](https://github.com/CLIMADA-project/climada_papers) for details.

CLIMADA is divided into two parts (two repositories):

1. the core [climada_python](https://github.com/CLIMADA-project/climada_python) contains all the modules necessary for the probabilistic impact, the averted damage, uncertainty and forecast calculations. Data for hazard, exposures and impact functions can be obtained from the [data API](https://github.com/CLIMADA-project/climada_python/blob/main/doc/tutorial/climada_util_api_client.ipynb). [Litpop](https://github.com/CLIMADA-project/climada_python/blob/main/doc/tutorial/climada_entity_LitPop.ipynb) is included as demo Exposures module, and [Tropical cyclones](https://github.com/CLIMADA-project/climada_python/blob/main/doc/tutorial/climada_hazard_TropCyclone.ipynb) is included as a demo Hazard module.
2. the petals [climada_petals](https://github.com/CLIMADA-project/climada_petals) contains all the modules for generating data (e.g., TC_Surge, WildFire, OpenStreeMap, ...). Most development is done here. The petals builds-upon the core and does not work as a stand-alone.

It is recommend for new users to begin with the core (1) and the [tutorials](https://github.com/CLIMADA-project/climada_python/tree/main/doc/tutorial) therein.

This is the Python (3.8+) version of CLIMADA - please see [https://github.com/davidnbresch/climada](https://github.com/davidnbresch/climada) for backward compatibility (MATLAB).

## Getting started

CLIMADA runs on Windows, macOS and Linux.
The released versions of the CLIMADA core can be installed directly through Anaconda:

```shell
conda install -c conda-forge climada
```

It is **highly recommended** to install CLIMADA into a **separate** Anaconda environment.
See the [installation guide](https://climada-python.readthedocs.io/en/latest/guide/install.html) for further information.

Follow the [tutorial](https://climada-python.readthedocs.io/en/stable/tutorial/1_main_climada.html) `climada_python-x.y.z/doc/tutorial/1_main_climada.ipynb` in a Jupyter Notebook to see what can be done with CLIMADA and how.

## Documentation

The online documentation is available on Read the Docs: https://climada-python.readthedocs.io/en/stable/

The documentation of each release version of CLIMADA can be accessed separately through the drop-down menu at the bottom of the left sidebar. Additionally, the version 'stable' refers to the most recent release (installed via `conda`), and 'latest' refers to the latest unstable development version (the `develop` branch).


CLIMADA python:

* [online (recommended)](https://climada-python.readthedocs.io/en/latest/)
* [PDF file](https://climada-python.readthedocs.io/_/downloads/en/stable/pdf/)
* [core Tutorials on GitHub](https://github.com/CLIMADA-project/climada_python/tree/main/doc/tutorial)

CLIMADA petals:

* [online (recommended)](https://climada-petals.readthedocs.io/en/latest/)
* [PDF file](https://climada-petals.readthedocs.io/_/downloads/en/stable/pdf/)
* [petals Tutorials on GitHub](https://github.com/CLIMADA-project/climada_petals/tree/main/doc/tutorial)

The documentation can also be [built locally](https://climada-python.readthedocs.io/en/latest/README.html).

## Citing CLIMADA

If you use CLIMADA please cite (in general, in particular for academic work) :

The [used version](https://zenodo.org/search?page=1&size=20&q=climada) and/or the following published articles depending on which functionalities you made use of:

- *Impact calculations*: Aznar-Siguan, G. and Bresch, D. N., 2019: CLIMADA v1: a global weather and climate risk assessment platform, Geosci. Model Dev., 12, 3085–3097, [https://doi.org/10.5194/gmd-12-3085-2019](https://doi.org/10.5194/gmd-14-351-2021
)

- *Cost-benefit analysis*: Bresch, D. N. and Aznar-Siguan, G., 2021: CLIMADA v1.4.1: towards a globally consistent adaptation options appraisal tool, Geosci. Model Dev., 14, 351-363, [https://doi.org/10.5194/gmd-14-351-2021](https://doi.org/10.5194/gmd-14-351-2021
)

- *Uncertainty and sensitivity analysis (unsequa)*:  Kropf, C. M. et al., 2022: Uncertainty and sensitivity analysis for probabilistic weather and climate-risk modelling: an implementation in CLIMADA v.3.1.0. Geoscientific Model Development 15, 7177–7201, [https://doi.org/10.5194/gmd-15-7177-2022](https://doi.org/10.5194/gmd-15-7177-2022
)

- *LitPop exposures* : Eberenz, S., et al., D. N. (2020): Asset exposure data for global physical risk assessment. Earth System Science Data 12, 817–833, [https://doi.org/10.3929/ethz-b-000409595](https://doi.org/10.3929/ethz-b-000409595)


Please see all CLIMADA-related scientific publications in our [repository of scientific publications](https://github.com/CLIMADA-project/climada_papers) and cite according to your use of select features, be it hazard set(s), exposure(s) ...

In presentations or other graphical material, as well as in reports etc., where applicable, please add the logo as follows:\
![https://github.com/CLIMADA-project/climada_python/blob/main/doc/guide/img/CLIMADA_logo_QR.png](https://github.com/CLIMADA-project/climada_python/blob/main/doc/guide/img/CLIMADA_logo_QR.png?raw=true)

As key link, please use https://wcr.ethz.ch/research/climada.html, as it will last and provides a bit of an intro, especially for those not familiar with GitHub - plus a nice CLIMADA infographic towards the bottom of the page

## Contributing

See the [Contribution Guide](CONTRIBUTING.md).

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [releases on this repository](https://github.com/CLIMADA-project/climada_python/releases).

## License

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License Version 3, 29 June 2007 as published by the Free Software Foundation, https://www.gnu.org/licenses/gpl-3.0.html

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details: https://www.gnu.org/licenses/gpl-3.0.html
