.. _Configuration options:

Configuration options
=====================

CLIMADA searches for a local configuration file located in the current
working directory. A static default configuration file is supplied by the package
and used as fallback. The local configuration file needs to be called
``climada.conf``. All other files will be ignored.

The climada configuration file is a JSON file and consists of the following values:

- ``local_data``
- ``global``
- ``trop_cyclone``

A minimal configuration file looks something like this::

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


local_data
----------
Configuration parameters related to local data location.

+---------------+--------------------------------------------------------------------------------------------------+-------------+
|     Option    |                                Description                                                       |   Default   |
+===============+==================================================================================================+=============+
| ``save_dir``  | Folder were the variables are saved through the ``save`` command when no absolute path provided. | "./results" |
+---------------+--------------------------------------------------------------------------------------------------+-------------+

global
------
Configuration parameters with global scope.

+---------------------+--------------------------------------------------------------------------------------------------+-------------+
|     Option          |                                Description                                                       |   Default   |
+=====================+==================================================================================================+=============+
| ``log_level``       | Minimum log level showed by logging: DEBUG, INFO, WARNING, ERROR or CRITICAL.                    | "INFO"      |
+---------------------+--------------------------------------------------------------------------------------------------+-------------+
| ``max_matrix_size`` | Maximum matrix size that can be used. Set a lower value if memory issues.                        | 1.0E8       |
+---------------------+--------------------------------------------------------------------------------------------------+-------------+

trop_cyclone
------------
Configuration parameters related to tropical cyclones.

+---------------------+--------------------------------------------------------------------------------------------------+-------------+
|     Option          |                                Description                                                       |   Default   |
+=====================+==================================================================================================+=============+
| ``random_seed``     | Seed used for the stochastic tracks generation.                                                  | 54          |
+---------------------+--------------------------------------------------------------------------------------------------+-------------+

