[![Build Status](http://ied-wcr-jenkins.ethz.ch/buildStatus/icon?job=climada_ci)](http://ied-wcr-jenkins.ethz.ch/job/climada_ci/)
[![Documentation build status](https://img.shields.io/readthedocs/climada-python.svg?style=flat-square)](https://readthedocs.org/projects/climada-python/builds/)
![Jenkins Coverage](https://img.shields.io/jenkins/coverage/cobertura/http/ied-wcr-jenkins.ethz.ch/climada_ci_night.svg)

CLIMADA
=======
Python (3.6+) version of CLIMADA

Latest CLIMADA in Python - please see https://github.com/davidnbresch/climada for backard compatibility (MATLAB)

Authors: David N. Bresch <dbresch@ethz.ch>, Gabriela Aznar Siguan <aznarsig@ethz.ch>

Date: 2019-04-08

Version: 1.2.3+

See [documentation](http://climada-python.readthedocs.io/en/latest/) and [tutorial](https://github.com/davidnbresch/climada_python/tree/master/script/tutorial/1_main_climada.ipynb).

Introduction
------------

CLIMADA stands for **clim**ate **ada**ptation and is a probabilistic natural catastrophe damage model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.).

Installation
------------

Follow the [Installation](https://climada-python.readthedocs.io/en/latest/install.html) instructions to install climada's development version and climada's stable version.

Data dependencies
-----------------

CLIMADA relies on several open data available through APIs (e.g. World Bank, Natural Earth data). Other data need to be downloaded manually by the user. The following table shows these last data sources, their version used, its current availabilty and where they are used within CLIMADA:

| Availability | Name | Version | Link | CLIMADA class | CLIMADA version | CLIMADA tutorial reference |
|--------------|:----:|--------:|------|:-------------:|----------------:|----------------:|
| OK | Gridded Population of the World (GPW)  | v4.11 | [GPW v4.11](http://sedac.ciesin.org/data/set/gpw-v4-population-count-rev11) | LitPop | > v1.2.3 | climada_entity_LitPop.ipynb |
| FAILED | Gridded Population of the World (GPW)  | v4.10 | [GPW v4.10](http://sedac.ciesin.org/data/set/gpw-v4-population-count-rev10) | LitPop | >= v1.2.0 | climada_entity_LitPop.ipynb |
| OK| International Best Track Archive for Climate Stewardship (IBTrACS) | v04r00 | [IBTrACS v04r00](ftp://eclipse.ncdc.noaa.gov/pub/ibtracs//v04r00/provisional/netcdf/) | TCTracks | >= v1.2.0 | climada_hazard_TropCyclone.ipynb |

Configuration options
---------------------

The program searches for a local configuration file located in the current 
working directory. A static default configuration file is supplied by the package 
and used as fallback. The local configuration file needs to be called 
``climada.conf``. All other files will be ignored.

The climada configuration file is a JSON file and consists of the following values:

- ``local_data``
- ``global``
- ``trop_cyclone``

A minimal configuration file looks something like this:

```javascript
{
    "local_data":
    {
        "save_dir": "./results/"
    },

    "global":
    {
        "log_level": "INFO",
        "max_matrix_size": 1.0e8
    },

    "trop_cyclone":
    {
        "random_seed": 54
    }
}
```

### local_data
Configuration values related to local data location.

| Option | Description | Default |
| ------ | ----------- | ------- |
| ``save_dir`` | Folder were the variables are saved through the ``save`` command when no absolute path provided. | "./results" |

### global
| Option | Description | Default |
| ------ | ----------- | ------- |
| ``log_level`` | Minimum log level showed by logging: DEBUG, INFO, WARNING, ERROR or CRITICAL. | "INFO" |
| ``max_matrix_size`` | Maximum matrix size that can be used. Set a lower value if memory issues. | 1.0e8 |

### trop_cyclone
Configuration values related to tropical cyclones.

| Option | Description | Default |
| ------ | ----------- | ------- |
| ``random_seed`` | Seed used for the stochastic tracks generation. | 54 |

