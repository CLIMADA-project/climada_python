[![Build Status](http://ied-wcr-jenkins.ethz.ch/buildStatus/icon?job=climada_branches/develop)](http://ied-wcr-jenkins.ethz.ch/job/climada_branches/)
[![Documentation build status](https://img.shields.io/readthedocs/climada-python.svg?style=flat-square)](https://readthedocs.org/projects/climada-python/builds/)
![Jenkins Coverage](https://img.shields.io/jenkins/coverage/cobertura/http/ied-wcr-jenkins.ethz.ch/climada_ci_night.svg)

# CLIMADA

CLIMADA stands for **CLIM**ate **ADA**ptation and is a probabilistic natural catastrophe impact model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.).

This is the Python (3.6+) version of CLIMADA - please see https://github.com/davidnbresch/climada for backward compatibility (MATLAB).

## Getting started

CLIMADA runs on Windows, macOS and Linux. Download the [latest release](https://github.com/CLIMADA-project/climada_python/releases). Install CLIMADA's dependencies specified in  the downloaded file `climada_python-x.y.z/requirements/env_climada.yml` with conda. See the documentation for more [information on installing](https://climada-python.readthedocs.io/en/latest/guide/install.html).

Follow the [tutorial](https://climada-python.readthedocs.io/en/latest/guide/tutorial.html) `climada_python-x.y.z/doc/tutorial/1_main_climada.ipynb` in a Jupyter Notebook to see what can be done with CLIMADA and how.

## Documentation

Documentation is available on Read the Docs:

* [online (recommended)](https://climada-python.readthedocs.io/en/latest/)
* [PDF file](https://buildmedia.readthedocs.org/media/pdf/climada-python/latest/climada-python.pdf)

## Citing CLIMADA

If you use CLIMADA for academic work please cite:

Aznar-Siguan, G. and Bresch, D. N., 2019: CLIMADA v1: a global weather and climate risk assessment platform, Geosci. Model Dev. Discuss., https://doi.org/10.5194/gmd-12-3085-2019

Please see all CLIMADA's related scientific publications in our [repository of scientific publications](https://github.com/CLIMADA-project/climada_papers).

## Contributing

To contribute follow these steps:

1. Fork the project on GitHub.
2. Create a local clone of the develop branch (`git clone https://github.com/YOUR-USERNAME/climada_python.git -b develop`)
3. Install the packages in `climada_python/requirements/env_climada.yml` and `climada_python/requirements/env_developer.yml`.
4. Make well commented and clean commits to your repository.
5. Make unit and integration tests on your code, preferably during development.
6. Perform a static code analysis of your code with CLIMADA's configuration `.pylintrc`.
7. Add your name to the AUTHORS file.
8. Push the changes to GitHub (`git push origin develop`).
9. On GitHub, create a new pull request onto the develop branch of CLIMADA-project/climada_python.

See our [contribution guidelines](https://climada-python.readthedocs.io/en/latest/guide/developer.html) for more information.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [releases on this repository](https://github.com/CLIMADA-project/climada_python/releases).

## License

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details:

<https://www.gnu.org/licenses/lgpl.html>
