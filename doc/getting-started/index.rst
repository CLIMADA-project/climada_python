===================
Getting started
===================

Installation
-------------------



Are you already working with conda ? proceed to install CLIMADA by executing the following line in the terminal::

    conda create -n climada_env -c conda-forge climada

Each time you will want to work with CLIMADA, simply activate the environnment::

    conda activate climada_env

You are good to go!


.. seealso::

    You don't have conda installed or you are looking for advaced installation instructions ? Look up our `detailed instructions <http://www.python.org>`__  on CLIMADA installation.


.. dropdown:: How does CLIMADA compute impacts ?
   :color: primary

   CLIMADA computes impacts following the IPCC risk framework by combining hazard intensity, exposure, and vulnerability
   data. It models hazards intensity (e.g., tropical cyclones, floods) using
   historical event sets or stochastic simulations, overlaying them with spatial exposure data
   (e.g., population, infrastructure), and applies vulnerability functions that estimate damage or
   loss, given the hazard intensity. By aggregating these results, CLIMADA calculates expected
   impacts, such as economic losses or affected populations.
.. dropdown:: How do you create a Hazard ?
   :color: primary

   From a risk perspective, the intersting aspect of a natural hazard is its location and intensity. For such,
   CLIMADA allows you to load your own hazard data or to directly define it using the platform. As an example,
   Users can easily load historical tropical cyclone tracks (IBTracks) and apply stochastic methods to generate
   a larger ensemble of tracks from the historical ones, from which they can easily compute the maximal windspeed.

.. dropdown:: How do we define an exposure ?
   :color: primary

   Exposure is defined as the entity that could potentially be damaged by a hazard: it can be people, infrastructures,
   assests, ecosystems or others. The CLIMADA user is given the option to load its own exposure data into the platform,
   or to use CLIMADA to define it. One common way of defining assets' exposure is through LitPop (link). LitPop dissagrate a
   financial index, as the GDP of a country for instance, to a much finer resolution proportionally to population
   density and nighlight intensity.

.. dropdown:: What are centroids ?
   :color: primary

   How can you compute the impact of a hazard on an exposure if their locations differs ? Well, you can't.
   This is what cetroids are for. Centroids are a grid of points defined by the users, in which both the exposure value
   and hazard intensity are calculated, allowing you to obtain the asset value and the hazard intensity im those
   defined points.

.. dropdown:: How do we model vulnerability ?
   :color: primary

   Vulnerability curves, also known as impact functions, tie the link between hazard intensity and damage.
   CLIMADA offers built-in sigmoidal or step-wise vulnerability curves, or allows you to calibrate your own
   impact functions with damage and hazard data through the calibration module (link).

   (image many impact functions and optimal)

.. dropdown:: Do you want to quantify the uncertainties ?
   :color: primary

   CLIMADA provides a dedicated module ([unsequa link]) for conducting uncertainty and sensitivity analyses.
   This module allows you to define a range of input parameters and evaluate their influence on the output,
   helping you quantify the sensitivity of the modeling chain as well as the uncertainties in your results.

.. dropdown:: Compare adaptation measures and assess their cost-effectiveness
   :color: primary

   Is there an adaptation measure that will decrease the impact? Does the cost needed to implement such
   measure outweight the gains? All these questions can be asnwered using the cost-benefit module (link adaptation).
   With this module, users can define and compare adaptation measures to establish their cost-effectiveness.
