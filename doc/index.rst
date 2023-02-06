===================
Welcome to CLIMADA!
===================

.. image:: guide/img/CLIMADA_logo_QR.png
   :align: center
   :alt: CLIMADA Logo

CLIMADA stands for CLIMate ADAptation and is a probabilistic natural catastrophe impact model, that also calculates averted damage (benefit) thanks to adaptation measures of any kind (from grey to green infrastructure, behavioural, etc.).

CLIMADA is primarily developed and maintained by the `Weather and Climate Risks Group <https://wcr.ethz.ch/>`_ at `ETH Zürich <https://ethz.ch/en.html>`_.

This is the documentation of the CLIMADA core module which contains all functionalities necessary for performing climate risk analysis and appraisal of adaptation options. Modules for generating different types of hazards and other specialized applications can be found in the `CLIMADA Petals <https://climada-petals.readthedocs.io/en/stable/>`_ module.

Jump right in:

* :doc:`README <misc/README>`
* :doc:`Getting Started <guide/Guide_get_started>`
* :doc:`Installation <guide/Guide_Installation>`
* :doc:`Overview <tutorial/1_main_climada>`
* `GitHub Repository <https://github.com/CLIMADA-project/climada_python>`_
* :doc:`Module Reference <climada/climada>`

.. ifconfig:: readthedocs

   .. hint::

      ReadTheDocs hosts multiple versions of this documentation.
      Use the drop-down menu on the bottom left to switch versions.
      ``stable`` refers to the most recent release, whereas ``latest`` refers to the latest development version.

.. admonition:: Copyright Notice

   Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in :doc:`AUTHORS.md <misc/AUTHORS>`.

   CLIMADA is free software: you can redistribute it and/or modify it under the
   terms of the GNU General Public License as published by the Free
   Software Foundation, version 3.

   CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
   PARTICULAR PURPOSE.  See the GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with CLIMADA. If not, see https://www.gnu.org/licenses/.


.. toctree::
   :hidden:

   GitHub Repositories <https://github.com/CLIMADA-project>
   CLIMADA Petals <https://climada-petals.readthedocs.io/en/stable/>
   Weather and Climate Risks Group <https://wcr.ethz.ch/>


.. toctree::
   :maxdepth: 1
   :caption: User Guide
   :hidden:

   guide/Guide_Introduction
   Getting Started <guide/Guide_get_started>
   guide/Guide_Installation
   Very Easy Installation <tutorial/climada_installation_step_by_step>
   Running CLIMADA on Euler <guide/Guide_Euler>


.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   Overview <tutorial/1_main_climada>
   Python Introduction <tutorial/0_intro_python>
   Hazard <tutorial/hazard>
   Exposures <tutorial/exposures>
   Impact <tutorial/impact>
   Uncertainty Quantification <tutorial/unsequa>
   tutorial/climada_engine_Forecast
   Google Earth Engine <tutorial/climada_util_earth_engine>
   tutorial/climada_util_api_client


.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   :hidden:

   Development with Git <guide/Guide_Git_Development>
   guide/Guide_CLIMADA_Tutorial
   guide/Guide_Configuration
   guide/Guide_Continuous_Integration_and_Testing
   guide/Guide_Reviewer_Checklist
   guide/Guide_PythonDos-n-Donts
   Performance and Best Practices <guide/Guide_Py_Performance>
   Coding Conventions <guide/Guide_Miscellaneous>
   Building the Documentation <README>


.. toctree::
   :caption: Miscellaneous
   :hidden:

   Python modules <climada/climada>
   README <misc/README>
   Changelog <misc/CHANGELOG>
   List of Authors <misc/AUTHORS>
   Contribution Guide <misc/CONTRIBUTING>
