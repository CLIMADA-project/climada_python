=================
GitHub Actions CI
=================

CLIMADA has been using a private Jenkins instance for automated testing (Continuous Integration, CI), see :doc:`Guide_Continuous_Integration_and_Testing`.
We recently adopted `GitHub Actions <https://docs.github.com/en/actions>`_ for automated unit testing.
GitHub Actions is a service provided by GitHub, which lets you configure CI/CD pipelines based on YAML configuration files.
GitHub provides servers which ample computational resources to create software environments, install software, test it, and deploy it.
See the `GitHub Actions Overview <https://docs.github.com/en/actions/learn-github-actions/understanding-github-actions>`_ for a technical introduction, and the `Workflow Syntax <https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions>`_ for a reference of the pipeline definitions.

The CI results for each pull request can be inspected in the "Checks" tab.
For GitHub Actions, users can inspect the logs of every step for every job.

.. note::

   As of CLIMADA v4.0, the default CI technology remains Jenkins.
   GitHub Actions CI is currently considered experimental for CLIMADA development.

---------------------
Unit Testing Pipeline
---------------------

This pipeline is defined by the ``.github/workflows/ci.yml`` file.
It contains a single job which will create a CLIMADA environment with Mamba for multiple Python versions, install CLIMADA, run the unit tests, and report the test coverage as well as the simplified test results.
The job has a `strategy <https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions#jobsjob_idstrategy>`_ which runs it for multiple times for different Python versions.
This way, we make sure that CLIMADA is compatible with all currently supported versions of Python.

The coverage reports in HTML format will be uploaded as job artifacts and can be downloaded as ZIP files.
The test results are simple testing summaries that will appear as individual checks/jobs after the respective job completed.
