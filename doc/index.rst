===================
Welcome to CLIMADA!
===================

.. image:: guide/img/CLIMADA_logo_QR.png
   :align: center
   :alt: CLIMADA Logo

`CLIMADA <https://climada.ethz.ch>`_ (CLIMate ADAptation) is a free and open-source software framework for
comprehensive climate risk assessment. Designed by a large scientific community,
CLIMADA offers a robust and flexible platform to analyse the impacts of natural
hazards and explore adaptation strategies, and it can be used by researchers,
policy and decision-makers.

CLIMADA is primarily developed and maintained by the `Weather and Climate Risks
Group <https://wcr.ethz.ch/>`_ at `ETH Zürich <https://ethz.ch/en.html>`_.

If you use CLIMADA for your own scientific work, please reference the
appropriate publications according to the :doc:`misc/citation`.

This is the documentation of the CLIMADA core module which contains all
functionalities necessary for performing climate risk analysis and appraisal of
adaptation options. Modules for generating different types of hazards and other
specialized applications can be found in the `CLIMADA Petals
<https://climada-petals.readthedocs.io/en/stable/>`_ module.

.. grid:: 1 2 2 2
    :gutter: 4
    :padding: 2 2 0 0
    :class-container: sd-text-center

    .. grid-item-card:: Getting Started
        :shadow: md

        Getting started with CLIMADA: How to install?
        What are the basic concepts and functionalities?

        +++

        .. button-ref:: getting-started/index
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:


    .. grid-item-card:: User Guide
        :shadow: md

        Want to go more in depth? Check out the User guide. It contains detailed
        tutorials on the different concepts, modules and possible usage of CLIMADA.

        +++

        .. button-ref:: user-guide/index
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the user guide!



    .. grid-item-card::  Implementation API reference
        :shadow: md

        The reference guide contains a detailed description of
        the CLIMADA API. The API reference describes each module, class,
        methods and functions.

        +++

        .. button-ref:: api/index
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the reference guide!

    .. grid-item-card::  Developer guide
        :shadow: md

        Saw a typo in the documentation? Want to improve
        existing functionalities? Want to extend them?
        The contributing guidelines will guide you through
        the process of improving CLIMADA.

        +++

        .. button-ref:: development/index
            :ref-type: doc
            :click-parent:
            :color: secondary
            :expand:

            To the development guide!

.. ifconfig:: readthedocs

   .. hint::

      ReadTheDocs hosts multiple versions of this documentation.
      Use the drop-down menu on the bottom left to switch versions.
      ``stable`` refers to the most recent release, whereas ``latest`` refers to the latest development version.

**Date**: |today| **Version**: |version|

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
   :maxdepth: 1
   :hidden:

   Getting started <getting-started/index>
   User Guide <user-guide/index>
   Developer Guide <development/index>
   API Reference <api/index>
   About <misc/AUTHORS>
   Changelog <misc/CHANGELOG>
   CLIMADA Petals <https://climada-petals.readthedocs.io/en/stable/>
   WCR Group <https://wcr.ethz.ch/>
   CLIMADA Website <https://climada.ethz.ch>
