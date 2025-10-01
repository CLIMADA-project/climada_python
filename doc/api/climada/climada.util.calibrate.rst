.. _calibration-module:

==================================
Impact Function Calibration Module
==================================

What's New?
-----------

Since CLIMADA v6.0.1, some functionality of this module has been changed.
When upgrading to a newer version of CLIMADA, please mind the following changes:

* :py:class:`~climada.util.calibrate.base.Input` received additional attributes.
  We now support optional weights that are passed to the cost function.
  Therefore, the cost function must support an additional, optional argument.
* :py:attr:`~climada.util.calibrate.base.Input.cost_func` now receives numpy arrays.
  An additional attribute :py:attr:`~climada.util.calibrate.base.Input.df_to_numpy` was
  added to transform ``pandas.DataFrame`` objects to ``np.ndarray``.
  By default, it returns a flattened array.
* This module now exports cost functions that support optional weights, see
  :py:mod:`climada.util.calibrate.cost_func`.
* :ref:`Ensemble optimizers <ensemble-optimizers>` have been addded.

Base Classes
------------

Generic classes for defining the data structures of this module.

.. automodule:: climada.util.calibrate.base
   :members:
   :private-members:

.. automodule:: climada.util.calibrate.cost_func
   :members:
   :private-members:

Bayesian Optimizer
------------------

Calibration based on Bayesian optimization.

.. automodule:: climada.util.calibrate.bayesian_optimizer
   :members:
   :show-inheritance:
   :inherited-members:  abc.ABC

Scipy Optimizer
---------------

Calibration based on the ``scipy.optimize`` module.

.. automodule:: climada.util.calibrate.scipy_optimizer
   :members:
   :show-inheritance:
   :inherited-members:  abc.ABC

.. _ensemble-optimizers:

Ensemble Optimizers
-------------------

Ensemble optimizers calibrate an ensemble of optimized parameter sets from subsets of the original input by employing multiple instances of the above "default" optimizers.
This gives a better sense of uncertainty in the calibration results:
By only selecting a subset of events to calibrate on, and by repeating this process for several times, one receives a varying set of impact functions that may spread considerably, as some events might dominate the calibration.
We distinguish two cases:
The :py:class:`~climada.util.calibrate.ensemble.AverageEnsembleOptimizer` samples a subset of all events with or without replacement.
The resulting "average ensemble" contains uncertainty information on the average impact function for all events.
The :py:class:`~climada.util.calibrate.ensemble.TragedyEnsembleOptimizer` calibrates one impact function for each single event.
The resulting "ensemble of tragedies" encodes the inter-event uncertainty.

.. automodule:: climada.util.calibrate.ensemble
   :members:
   :show-inheritance:
   :inherited-members:  abc.ABC
