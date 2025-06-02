===================
Getting started
===================

Quick Installation
--------------------



Are you already working with mamba or conda? proceed to install CLIMADA by executing the following line in the terminal::

    mamba create -n climada_env -c conda-forge climada

Each time you will want to work with CLIMADA, simply activate the environnment::

    mamba activate climada_env

You are good to go!

.. seealso::

   You don't have mamba or conda installed or you are looking for advanced installation instructions? Look up our :doc:`detailed instructions <install>` on CLIMADA installation.

Climada in a Nutshell
---------------------

.. dropdown:: How does CLIMADA compute impacts ?
   :color: primary

   CLIMADA follows the IPCC risk framework to compute impacts by combining hazard intensity, exposure, and vulnerability.
   It models hazards intensity (e.g., tropical cyclones, floods) using
   historical event sets or stochastic simulations, overlaying them with spatial exposure data
   (e.g., population, infrastructure), and applies vulnerability functions that estimate damage
   given the hazard intensity. By aggregating these results, CLIMADA calculates expected
   impacts, such as economic losses or affected populations. See the dedicated :doc:`impact tutorial </user-guide/climada_engine_Impact>`
   for more informations.

   .. image:: /user-guide/img/risk_framework.png
      :width: 400
      :alt: Alternative text
      :align: center

.. dropdown:: How do you create a Hazard ?
   :color: primary

   From a risk perspective, the intersting aspect of a natural hazard is its location and intensity. For such,
   CLIMADA allows you to load your own :doc:`hazard </user-guide/climada_hazard_Hazard>` data or to directly define it in the platform. As an example,
   users can easily load historical tropical cyclone tracks (IBTracks) and apply stochastic methods to generate
   a larger ensemble of tracks from the historical ones, from which they can easily compute the maximal windspeed,
   the hazard intensity.

   .. image:: /user-guide/img/tc-tracks.png
      :width: 500
      :alt: Alternative text
      :align: center

.. dropdown:: How do we define an exposure ?
   :color: primary

   Exposure is defined as the entity that could potentially be damaged by a hazard: it can be people, infrastructures,
   assests, ecosystems or more. A CLIMADA user is given the option to load its own exposure data into the platform,
   or to use CLIMADA to define it. One common way of defining assets' exposure is through :doc:`LitPop </user-guide/climada_entity_LitPop>`. LitPop dissagrate a
   financial index, as the country GDP for instance, to a much finer resolution proportionally to population
   density and nighlight intensity.

   .. image:: /user-guide/img/exposure.png
      :width: 500
      :align: center

.. dropdown:: How do we model vulnerability ?
   :color: primary

   Vulnerability curves, also known as impact functions, tie the link between hazard intensity and damage.
   CLIMADA offers built-in sigmoidal or step-wise vulnerability curves, and allows you to calibrate your own
   impact functions with damage and hazard data through the :doc:`calibration module </user-guide/climada_util_calibrate>`.


   .. image:: /user-guide/img/impact-function.png
      :width: 400
      :align: center

.. dropdown:: Do you want to quantify uncertainties ?
   :color: primary

   CLIMADA provides a dedicated module :doc:`unsequa </user-guide/climada_engine_unsequa>` for conducting uncertainty and sensitivity analyses.
   This module allows you to define a range of input parameters and evaluate their influence on the output,
   helping you quantify the sensitivity of the modeling chain as well as the uncertainties in your results.

   .. image:: /user-guide/img/sensitivity.png
      :width: 500
      :align: center

.. dropdown:: Compare adaptation measures and assess their cost-effectiveness
   :color: primary

   Is there an adaptation measure that will decrease the impact? Does the cost needed to implement such
   measure outweight the gains? All these questions can be asnwered using the :doc:`cost-benefit </user-guide/climada_engine_CostBenefit>` and
   :doc:`adaptation module </user-guide/climada_entity_MeasureSet>`.
   With this module, users can define and compare adaptation measures to establish their cost-effectiveness.

   .. image:: /user-guide/img/cost-benefit.png
      :width: 400
      :align: center

.. toctree::
   :maxdepth: 1
   :hidden:

   Navigate this documentation <Guide_get_started>
   Introduction <Guide_Introduction>
   Installation instructions <install>
   Python introduction <0_intro_python>
