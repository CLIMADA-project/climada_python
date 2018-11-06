climada_python
==============
Python (3.6+) version of CLIMADA

Latest CLIMADA in Python - please see https://github.com/davidnbresch/climada for backard compatibility (MATLAB)

Authors: David N. Bresch <dbresch@ethz.ch>, Gabriela Aznar Siguan <aznarsig@ethz.ch>

Date: 2018-11-06

Version: 0.1.0

See [technical documentation](http://climada-python.readthedocs.io/en/latest/) and [tutorial](https://github.com/davidnbresch/climada_python/tree/master/script/tutorial).

Introduction
------------

CLIMADA stands for **clim**ate **ada**ptation and is a probabilistic natural catastrophe damage model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.).

Installation
------------

Follow the [Installation](https://github.com/davidnbresch/climada_python/blob/master/doc/source/install.rst) instructions to install climada's development version and climada's stable version.

Configuration options
---------------------

The program searches for a local configuration file located in the current 
working directory. A static default configuration file is supplied by the package 
and used as fallback. The local configuration file needs to be called 
``climada.conf``. All other files will be ignored.

The climada configuration file is a JSON file and consists of the following values:

- ``local_data``
- ``entity``
- ``trop_cyclone``
- ``log_level``

A minimal configuration file looks something like this:

```javascript
{
    "local_data":
    {
        "save_dir": "./results/",
        "entity_def" : "",
        "repository": ""
    },

    "log_level": "INFO",
    
    "entity":
    {
        "present_ref_year": 2016,
        "future_ref_year": 2030
    },

    "trop_cyclone":
    {
        "time_step_h": 1,
        "random_seed": 54
    }
}
```


### local_data

| Option | Description | Default |
| ------ | ----------- | ------- |
| ``save_dir`` | Folder were the variables are saved through the ``save`` command. An absolut path is safer. | "./results" |
| ``entity_def`` | Entity to be used as default. If not provided, the static entity_template.xlsx is used. | "" |
| ``repository`` | Absolute path of climada's data repository. No default path provided. | "" |


### entity
Configuration values related to an Entity.

### trop_cyclone
Configuration values related to tropical cyclones.

### log_level
Minimum log level showed by logging. DEBUG, INFO, WARNING, ERROR and CRITICAL are the different levels.

