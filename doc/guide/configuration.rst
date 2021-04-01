.. _Configuration options:

Configuration options
=====================

CLIMADA searches for local configuration files located in the current
working directory. A static default configuration file is supplied by the package
and used as fallback. The local configuration files needs to be called
``climada.conf`` and ``climada_test.conf``. All other files will be ignored.

Configuration values that are used only for test purposes are defined in ``climada_test.conf``.
All other configuration values are in ``climada.conf``.

The climada configuration files are JSON files and has the following values on the
toplevel:

- ``global``
- ``<module>`` where module can be any *climada* module.

Each entry may have the conventional items ``resource`` and ``data`` on the second level.

- ``resource`` is meant to contain any access information for external resources.
- ``data`` is for defining directories in the local filesystem that are used for reading and writing files.

Values can be composed of other configuration values by surrounding them with curly brackets,
or of system variables by surrounding them with square brackets.
Literal brackets ``[``, ``]``, ``{`` and ``}`` are denoted as ``[[``, ``]]``, ``{{`` and ``}}`` respectively.

A configuration file looks something like this::

  {
    "global":
    {
      "data": {
        "root": "./[CLIMADA_DATA]/data",
        "results": "{global.data.root}/results",
        "system": "{global.data.root}/system"
      }
      "log_level": "INFO",
      "max_matrix_size": 1.0e8
    },

    "climada.hazard.drought":
    {
      "resource": {
        "spei_url": "http://digital.csic.es/bitstream/10261/153475/8"
      }
    },

    "climada.hazard.trop_cyclone":
    {
      "random_seed": 54
    }
  }

Configuration values can be accessed like this:

.. code::

   from climada.util import config
   SYSTEM_DATA_DIR = config.get("climada.global.data.system")

   from climada.test import test_config
   DROUGHT_TEST_DATA_DIR = test_config.get("climada.hazard.drought.data")
