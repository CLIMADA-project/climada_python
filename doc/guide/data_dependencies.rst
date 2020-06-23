.. _Data dependencies:

Data dependencies
=================

Web APIs
--------
CLIMADA relies on open data available through web APIs such as those of the World Bank, Natural Earth, NASA and NOAA.
You might execute the test ``climada_python-x.y.z/test_data_api.py`` to check that all the APIs used are active.
If any is out of service (temporarily or permanently), the test will indicate which one.

Manual download
---------------
As indicated in the software and tutorials, other data might need to be downloaded manually by the user. The following table shows these last data sources, their version used, its current availabilty and where they are used within CLIMADA:



+--------------+--------------------------------------------------------------------+-----------+-----------------------------------------------------------------------------------------+---------------+-----------------+-----------------------------------+
| Availability |                          Name                                      |  Version  |                                      Link                                               | CLIMADA class | CLIMADA version | CLIMADA tutorial reference        |
+==============+====================================================================+===========+=========================================================================================+===============+=================+===================================+
|     OK       | Fire Information for Resource Management System                    |    -      |  `FIRMS <https://firms.modaps.eosdis.nasa.gov/download/>`_                              | BushFire      | > v1.2.5       | climada_hazard_BushFire.ipynb      |
+--------------+--------------------------------------------------------------------+-----------+-----------------------------------------------------------------------------------------+---------------+-----------------+-----------------------------------+
|     OK       | Gridded Population of the World (GPW)                              |    v4.11  |  `GPW v4.11 <http://sedac.ciesin.org/data/set/gpw-v4-population-count-rev11>`_          | LitPop        | > v1.2.3        | climada_entity_LitPop.ipynb       |
+--------------+--------------------------------------------------------------------+-----------+-----------------------------------------------------------------------------------------+---------------+-----------------+-----------------------------------+
|   FAILED     | Gridded Population of the World (GPW)                              |    v4.10  |  `GPW v4.10 <http://sedac.ciesin.org/data/set/gpw-v4-population-count-rev10>`_          | LitPop        | >= v1.2.0       | climada_entity_LitPop.ipynb       |
+--------------+--------------------------------------------------------------------+-----------+-----------------------------------------------------------------------------------------+---------------+-----------------+-----------------------------------+
