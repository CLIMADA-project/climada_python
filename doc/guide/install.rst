============
Installation
============

The following sections will guide you through the installation of CLIMADA and its dependencies.

.. attention::

    CLIMADA has a complicated set of dependencies that cannot be installed with ``pip`` alone.
    Please follow the installation instructions carefully!
    We recommend to use `Anaconda`_ for creating a suitable software environment to execute CLIMADA.

All following instructions should work on any operating system (OS) that is supported by `Anaconda`_, including in particular: **Windows**, **macOS**, and **Linux**.

.. note:: When mentioning the terms "terminal" or "command line" in the following, we are referring to the "Terminal" apps on macOS or Linux and the "Anaconda Prompt" on Windows.

-------------
Prerequisites
-------------

* Make sure you are using the **latest version** of your OS. Install any outstanding **updates**.
* Free up at least 10 GB of **free storage space** on your machine.
  Anaconda and the CLIMADA dependencies will require around 5 GB of free space, and you will need at least that much additional space for storing the input and output data of CLIMADA.
* Ensure a **stable internet connection** for the installation procedure.
  All dependencies will be downloaded from the internet.
  Do **not** use a metered, mobile connection!
* Install `Anaconda`_, following the `installation instructions <https://docs.anaconda.com/anaconda/install/>`_ for your OS.

.. hint:: If you need help with the vocabulary used on this page, refer to the :ref:`Glossary <install-glossary>`.

.. _install-choice:

Decide on Your Entry Level!
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Depening on your level of expertise, we provide two different approaches:

* If you have never worked with a command line, or if you just want to give CLIMADA a try, follow the :ref:`simple instructions <install-simple>`.
* If you want to use the very latest development version of CLIMADA or even develop new CLIMADA code, follow the :ref:`advanced instructions <install-advanced>`.
  If you want to install `CLIMADA Petals`_, also follow these.

Both approaches are not mutually exclusive.
After successful installation, you may switch your setup at any time.

.. _install-simple:

-------------------
Simple Instructions
-------------------

These instructions will install the most recent stable version of CLIMADA without cloning its repository.

#. Open the command line.
   Create a new conda environment with CLIMADA by executing

   .. code-block:: shell

      conda create -n climada_env -c conda-forge climada

#. Activate the environment:

   .. code-block:: shell

      conda activate climada_env

   You should now see ``(climada_env)`` appear in the beginning of your command prompt.
   This means the environment is activated.

#. Verify that everything is installed correctly by executing a single test:

   .. code-block:: shell

      python -m unittest climada.engine.test.test_impact

   Executing CLIMADA for the first time will take some time because it will generate a directory tree in your home/user directory.
   After a while, some text should appear in your terminal.
   In the end, you should see an "Ok".
   If so, great! You are good to go.

.. _install-advanced:

---------------------
Advanced Instructions
---------------------

For advanced Python users or developers of CLIMADA, we recommed cloning the CLIMADA repository and installing the package from source.

#. If you are using a **Linux** OS, make sure you have ``git`` installed
   (Windows and macOS users are good to go once Anaconda is installed).
   On Ubuntu and Debian, you may use APT:

   .. code-block:: shell

      apt update
      apt install git

   Both commands will probably require administrator rights, which can be enabled by prepending ``sudo``.

#. Create a **workspace directory**.
   To make sure that your user can manipulate it without special privileges, use a subdirectory of your user/home directory.
   Do **not** use a directory that is synchronized by cloud storage systems like OneDrive, iCloud or Polybox!

#. Open the command line and navigate to the workspace directory you created using ``cd``.
   Replace ``<path/to/workspace>`` with the path of the directory that contains the workspace folder:

   .. code-block:: shell

      cd <path/to/workspace>

#. Clone CLIMADA from its `GitHub repository <https://github.com/CLIMADA-project/climada_python>`_.
   Enter the directory and check out the branch of your choice.
   The latest development version will be available under the branch ``develop``.

   .. code-block:: shell

      git clone https://github.com/CLIMADA-project/climada_python.git
      cd climada_python
      git checkout develop

#. Create an Anaconda environment called ``climada_env`` for installing CLIMADA.
   Use the default environment specs in ``env_climada.yml`` to create it.
   Then activate the environment:

   .. code-block:: shell

      conda env create -n climada_env -f requirements/env_climada.yml
      conda activate climada_env

#. Install the local CLIMADA source files as Python package using ``pip``:

   .. code-block:: shell

      python -m pip install -e ./

   .. hint::

      Using a path ``./`` (referring to the path you are currently located at) will instruct ``pip`` to install the local files instead of downloading the module from the internet.
      The ``-e`` (for "editable") option further instructs ``pip`` to link to the source files instead of copying them during installation.
      This means that any changes to the source files will have immediate effects in your environment, and re-installing the module is never required.

#. Verify that everything is installed correctly by executing a single test:

   .. code-block:: shell

      python -m unittest climada.engine.test.test_impact

   Executing CLIMADA for the first time will take some time because it will generate a directory tree in your home/user directory.
   If this test passes, great!
   You are good to go.

Install Developer Dependencies (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building the documentation and running the entire test suite of CLIMADA requires additional dependencies which are not installed by default.
They are also not needed for using CLIMADA.
However, if you want to develop CLIMADA, we recommend you install them.

With the ``climada_env`` activated, enter the workspace directory and then the CLIMADA repository as above.
Then, add the ``dev`` extra specification to the ``pip install`` command (**mind the quotation marks**, and see also `pip install examples <https://pip.pypa.io/en/stable/cli/pip_install/#examples>`_):

.. code-block:: shell

   python -m pip install -e "./[dev]"

The CLIMADA Python package defines the following `extras <https://peps.python.org/pep-0508/#extras>`_:

.. list-table::
   :header-rows: 1
   :widths: 1 5

   * - Extra
     - Includes Dependencies...
   * - ``doc``
     - for building documentation
   * - ``test``
     - for running and evaluating tests
   * - ``dev``
     - combination of ``doc`` and ``test``

Install CLIMADA Petals (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CLIMADA is divided into two repositories, CLIMADA Core (`climada_python <https://github.com/CLIMADA-project/climada_python>`_) and CLIMADA Petals (`climada_petals <https://github.com/CLIMADA-project/climada_petals>`_).
The Core contains all the modules necessary for probabilistic impact, averted damage, uncertainty and forecast calculations.
Data for hazard, exposures and impact functions can be obtained from the :doc:`CLIMADA Data API </tutorial/climada_util_api_client>`.
Hazard and Exposures subclasses are included as demonstrators only.

.. attention:: CLIMADA Petals is **not** a standalone module and requires CLIMADA Core to be installed!

CLIMADA Petals contains all the modules for generating data (e.g., ``TC_Surge``, ``WildFire``, ``OpenStreeMap``, ...).
New modules are developed and tested here.
Some data created with modules from Petals is available to download from the :doc:`Data API </tutorial/climada_util_api_client>`.
This works with just CLIMADA Core installed.
CLIMADA Petals can be used to generate additional data of this type, or to have a look at the tutorials for all data types available from the API.

To install CLIMADA Petals, we assume you have already installed CLIMADA Core with the :ref:`advanced instructions <install-advanced>` above.

#. Open the command line and navigate to the workspace directory.
#. Clone CLIMADA Petals from its `repository <https://github.com/CLIMADA-project/climada_petals>`_.
   Enter the directory and check out the branch of your choice.
   The latest development version will be available under the branch ``develop``.

   .. code-block:: shell

      git clone https://github.com/CLIMADA-project/climada_petals.git
      cd climada_petals
      git checkout develop

#. Update the Anaconda environment with the specifications from Petals and activate it:

   .. code-block:: shell

      conda env update -n climada_env -f requirements/env_climada.yml
      conda env update -n climada_env -f requirements/env_developer.yml
      conda activate climada_env

#. Install the CLIMADA Petals package:

   .. code-block:: shell

      python -m pip install -e ./

------------------------------
Apps for Programming in Python
------------------------------

To work with CLIMADA, you will need an application that supports Jupyter Notebooks.
There are plugins available for nearly every code editor or IDE, but if you are unsure about which to choose, we recommend `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_, `Visual Studio Code <https://code.visualstudio.com/>`_ or `Spyder <https://www.spyder-ide.org/>`_.

JupyterLab
^^^^^^^^^^

#. Install JupyterLab into the Anaconda environment:

   .. code-block:: shell

      conda install -n climada_env -c conda-forge jupyterlab

#. Make sure that the ``climada_env`` is activated (see above) and then start JupyterLab:

   .. code-block:: shell

      conda env activate climada_env
      jupyter-lab

   JupyterLab will open in a new window of your default browser.

Visual Studio Code (VSCode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Basic Setup
"""""""""""

#. Download and install VSCode following the instructions on https://code.visualstudio.com/.

#. Install the Python and Jupyter extensions.
   In the left sidebar, select the "Extensions" symbol, enter "Python" in the search bar and click *Install* next to the "Python" extension.
   Repeat this process for "Jupyter".

#. Open a Jupyter Notebook or create a new one.
   On the top right, click on *Select Kernel*, select *Python Environments...* and then choose the Python interpreter from the ``climada_env``.

See the VSCode docs on `Python <https://code.visualstudio.com/docs/python/python-tutorial>`_ and `Jupyter Notebooks <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>`_ for further information.

Workspace Setup
"""""""""""""""

Setting up a workspace for the CLIMADA source code is only available for :ref:`advanced installations <install-advanced>`.

#. Open a new VSCode window.
   Below *Start*, click *Open...*, select the ``climada_python`` repository folder in your workspace directory, and click on *Open* on the bottom right.

#. Click *File* > *Save Workspace As...* and store the workspace settings file next to (**not** in!) the ``climada_python`` folder.
   This will enable you to load the workspace and all its specific settings in one go.

#. Open the Command Palette by clicking *View* > *Command Palette* or by using the shortcut keys ``Ctrl+Shift+P`` (Windows, Linux) / ``Cmd+Shift+P`` (macOS).
   Start typing "Python: Select Interpreter" and select it from the dropdown menu.
   If prompted, choose the option to set the interpreter for the workspace, not just the current folder.
   Then, choose the Python interpreter from the ``climada_env``.

For further information, refer to the VSCode docs on `Workspaces <https://code.visualstudio.com/docs/editor/workspaces>`_.

Test Explorer Setup
"""""""""""""""""""

After you set up a workspace, you might want to configure the test explorer for easily running the CLIMADA test suite within VSCode:

#. In the left sidebar, select the "Testing" symbol, and click on *Configure Python Tests*.

#. Select "unittest" as test framework and then select the ``test*`` pattern for test discovery.

#. The "Test Explorer" will display the tree structure of modules, files, test classes and individuals tests.
   You can run individual tests or test subtrees by clicking the Play buttons next to them.

#. By default, the test explorer will show test output for failed tests when you click on them.
   To view the logs for any test, click on *View* > *Output*, and select "Python Test Log" from the dropdown menu in the view that just opened.
   If there are errors during test discovery, you can see what's wrong in the "Python" output.

For further information, see the VSCode docs on `Python Testing <https://code.visualstudio.com/docs/python/testing>`_.

Spyder
^^^^^^

Installing Spyder into the existing Anaconda environment for CLIMADA might fail depending on the exact versions of dependencies installed.
Therefore, we recommend installing Spyder in a *separate* environment, and then connecting it to a kernel in the original ``climada_env``.

#. Follow the `Spyder installation instructions <https://docs.spyder-ide.org/current/installation.html#installing-with-conda>`_.
   Make sure you install it with ``conda``!

#. Check the version of the Spyder kernel in the new environment:

   .. code-block:: shell

      conda env export -n spyder-env | grep spyder-kernels

   This will return a line like this:

   .. code-block:: shell

      - spyder-kernels=X.Y.Z=<hash>

   Copy the part ``spyder-kernels=X.Y.Z`` (until the second ``=``) and paste it into the following command to install the same kernel version into the ``climada_env``:

   .. code-block:: shell

      conda install -n climada_env spyder-kernels=X.Y.Z

#. Obtain the path to the Python interpreter of your ``climada_env``.
   Execute the following commands:

   .. code-block:: shell

      conda activate climada_env
      python -c "import sys; print(sys.executable)"

   Copy the resulting path.

#. Open Spyder.
   You can do so from the Anaconda Navigator, or by activating the new environment and launching it through the command line:

   .. code-block:: shell

      conda activate spyder-env
      spyder

#. Set the Python interpreter used by Spyder to the one of ``climada_env``.
   Select *Preferences* > *Python Interpreter* > *Use the following interpreter* and paste the iterpreter path you copied from the ``climada_env``.

----
FAQs
----

Answers to frequently asked questions.

.. _update-climada:

Updating CLIMADA
^^^^^^^^^^^^^^^^

We recommend keeping CLIMADA up-to-date.
To update, follow the instructions based on your :ref:`installation type <install-choice>`:

* **Simple Instructions:** Update CLIMADA using ``conda``:

  .. code-block:: shell

     conda update -n climada_env -c conda-forge climada

* **Advanced Instructions:** Move into your local CLIMADA repository and pull the latest version of your respective branch:

  .. code-block:: shell

     cd <path/to/workspace>/climada_python
     git pull

  Then, update the environment:

  .. code-block:: shell

     conda env update -n climada_env -f requirements/env_climada.yml
     conda env update -n climada_env -f requirements/env_developer.yml

  The same instructions apply for CLIMADA Petals.

.. _install-more-packages:

Installing More Packages
^^^^^^^^^^^^^^^^^^^^^^^^

You might use CLIMADA in code that requires more packages than the ones readily available in the CLIMADA Anaconda environment.
If so, **prefer installing these packages via Anaconda**, and only rely on ``pip`` if that fails.
The default channels of Anaconda sometimes contain outdated versions.
Therefore, use the ``conda-forge`` channel:

.. code-block:: shell

   conda install -n climada_env -c conda-forge <package>

Only if the desired package (version) is not available, go for ``pip``:

.. code-block:: shell

   conda activate climada_env
   python -m pip install <package>

Verifying Your Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you followed the installation instructions, you already executed a single unit test.
This test, however, will not cover all issues that could occur within your installation setup.
If you are unsure if everything works as intended, try running all unit tests.
This is only available for :ref:`advanced setups <install-advanced>`!
Move into the CLIMADA repository, activate the environment and then execute the tests:

.. code-block:: shell

   cd <path/to/workspace>/climada_python
   conda activate climada_env
   python -m unittest discover -s climada -p "test*.py"

Error: ``ModuleNotFoundError``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Something is wrong with the environment you are using.
After **each** of the following steps, check if the problem is solved, and only continue if it is **not**:

#. Make sure you are working in the CLIMADA environment:

   .. code-block:: shell

      conda activate climada_env

#. :ref:`Update the conda environment and CLIMADA <update-climada>`.

#. Anaconda will notify you if it is not up-to-date.
   In this case, follow its instructions to update it.
   Then, repeat the last step and update the environment and CLIMADA (again).

#. Install the missing package manually.
   Follow the instructions for :ref:`installing more packages <install-more-packages>`.

#. If you reached this point, something is severely broken.
   The last course of action is to delete your CLIMADA environment:

   .. code-block:: shell

      conda deactivate
      conda env remove -n climada_env

   Now repeat the :ref:`installation process <install-choice>`.

#. Still no good?
   Please raise an `issue on GitHub <https://github.com/CLIMADA-project/climada_python/issues>`_ to get help.

Changing the Logging Level
^^^^^^^^^^^^^^^^^^^^^^^^^^

By default the logging level is set to ``INFO``, which is quite verbose.
You can change this setting in multiple ways:

* Adjust the :doc:`configuration file <Guide_Configuration>` ``climada.conf`` by setting a the value of the ``global.log_level`` property.

* Set a global logging level in your Python script:

  .. code-block:: python

     from climada.util.config import LOGGER
     from logging import WARNING
     LOGGER.setLevel(WARNING)

* Set a local logging level in a context manager:

  .. code-block:: python

     from climada.util import log_level
     with log_level(level="WARNING"):
         # This code only emits log levels 'WARNING' or higher
         foo()

     # Default logging level again
     bar()

All of these approaches can also be combined.

`Mamba <https://mamba.readthedocs.io/en/latest/>`_ Instead of Anaconda
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer using Mamba, you should be able to simply replace all ``conda`` commands with ``mamba``.
Note that we can only provide **limited support** for Mamba installations!

Error: ``operation not permitted``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda might report a permission error on macOS Mojave.
Carefully follow these instructions: https://github.com/conda/conda/issues/8440#issuecomment-481167572

No ``impf_TC`` Column in ``GeoDataFrame``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This may happen when a demo file from CLIMADA was not updated after the change in the impact function naming pattern from ``if_`` to ``impf_`` when `CLIMADA v2.2.0 <https://github.com/CLIMADA-project/climada_python/releases/tag/v2.2.0>`_ was released.
Execute

.. code-block:: shell

   conda activate climada_env
   python -c "import climada; climada.setup_climada_data(reload=True)"

.. _install-glossary:

------------------------
The What Now? (Glossary)
------------------------

You might have become confused about all the names thrown at you.
Let's clear that up:

Terminal, Command Line
    A text-only program for interacting with your computer (the old fashioned way).

`Anaconda`_, conda
    The program that installs all requirements and creates a suitable environment for CLIMADA.

Environment (Programming)
    A setup where only a specific set of modules and programs can interact.
    This is especially useful if you want to install programs with mutually incompatible requirements.

`pip <https://pip.pypa.io/en/stable/index.html>`_
    The Python package installer.

`git <https://git-scm.com/>`_
    A popular version control software for programming code (or any text-based set of files).

`GitHub <https://github.com/>`_
    A website that publicly hosts git repositories.

git Repository
    A collection of files and their entire revision/version history, managed by git.

Cloning
    The process and command (``git clone``) for downloading a git repository.

IDE
    Integrated Development Environment.
    A fancy source code editor tailored for software development and engineering.


.. _Anaconda: https://www.anaconda.com/
.. _CLIMADA Petals: https://climada-petals.readthedocs.io/en/latest/
