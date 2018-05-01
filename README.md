climada_python
==============
Python (3.6+) version of CLIMADA

TEST version of core CLIMADA in Python - please use https://github.com/davidnbresch/climada for the time being unless you really know what you're doing. dbresch@ethz.ch

Authors: David N. Bresch <dbresch@ethz.ch>, Gabriela Aznar Siguan <aznarsig@ethz.ch>

Date: 2018-03-02

Version: 0.0.1

See [Technical documentation](http://climada-python.readthedocs.io/en/latest/).

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
- ``present_ref_year``
- ``future_ref_year``
- ``tc_time_step_h``
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
    
    "present_ref_year": 2016,
    
    "future_ref_year": 2030,

    "tc_time_step_h": 1,

    "log_level": "INFO"
}
```


### local_data

| Option | Description | Default |
| ------ | ----------- | ------- |
| ``save_dir`` | Folder were the variables are saved through the ``save`` command. An absolut path is safer. | "./results" |
| ``entity_def`` | Entity to be used as default. If not provided, the static entity_template.xlsx is used. | "" |
| ``repository`` | Absolute path of climada's data repository. No default path provided. | "" |


### present_ref_year
Default present reference year used in the entity.

### future_ref_year
Default future reference year used in the entity.

### tc_time_step_h
Time step (in hours) used to interpolate tropical cyclone tracks.

### log_level
Minimum log level showed by logging. DEBUG, INFO, WARNING, ERROR and CRITICAL are the different levels.

