==================================
Impact Function Calibration Module
==================================

Base Classes
------------

Generic classes for defining the data structures of this module.

.. automodule:: climada.util.calibrate.base
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

Ensemble Optimizers
-------------------

Calibration creating ensembles of impact functions.

.. automodule:: climada.util.calibrate.cross_calibrate
   :members:
   :show-inheritance:
   :inherited-members:  abc.ABC
