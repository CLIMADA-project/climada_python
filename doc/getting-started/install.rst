============
Installation
============

The following sections will guide you through the installation of CLIMADA and its dependencies.

.. attention::

    We recommend to use a ``conda``-based python environment manager such as `Mamba`_ or `Conda`_ for creating a suitable software environment to execute CLIMADA.

All following instructions should work on any operating system (OS), including in particular: **Windows**, **macOS**, and **Linux**.

.. hint:: If you need help with the vocabulary used on this page, refer to the :ref:`Glossary <install-glossary>`.

.. _install-choice:

---------------------------
Decide on Your Entry Level!
---------------------------

.. hint:: When mentioning the terms "terminal" or "command line" in the following, we are referring to the "Terminal" apps on macOS or Linux and the "Miniforge Prompt" on Windows.

Depening on your level of expertise, we provide two different approaches:

* If you have never worked with a command line, or if you just want to give CLIMADA a try, follow the :ref:`simple instructions <install-simple>`.
* If you want to use the very latest development version of CLIMADA or even develop new CLIMADA code, follow the :ref:`advanced instructions <install-advanced>`.

Both approaches are **not** mutually exclusive.
After successful installation, you may switch your setup at any time.

.. _petals-notes:

Notes on the CLIMADA Petals Package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CLIMADA is divided into two packages, CLIMADA Core (`climada_python <https://github.com/CLIMADA-project/climada_python>`_) and CLIMADA Petals (`climada_petals <https://github.com/CLIMADA-project/climada_petals>`_).
The Core contains all the modules necessary for probabilistic impact, averted damage, uncertainty and forecast calculations.
Data for hazard, exposures and impact functions can be obtained from the :doc:`CLIMADA Data API </user-guide/climada_util_api_client>`.
Hazard and Exposures subclasses are included as demonstrators only.

.. attention:: CLIMADA Petals is **not** a standalone module and requires CLIMADA Core to be installed!

CLIMADA Petals contains all the modules for generating data (e.g., ``TC_Surge``, ``WildFire``, ``OpenStreeMap``, ...).
New modules are developed and tested here.
Some data created with modules from Petals is available to download from the :doc:`Data API </user-guide/climada_util_api_client>`.
This works with just CLIMADA Core installed.
CLIMADA Petals can be used to generate additional data of this type, or to have a look at the tutorials for all data types available from the API.

Both :ref:`installation approaches <install-choice>` mentioned above support CLIMADA Petals.
If you are unsure whether you need Petals, you can install the Core first and later add Petals in both approaches.

.. warning::

   At the moment, not all package managers support both Core and Petals in both :ref:`installation approaches <install-choice>`.
   See the list below.

.. _install-manager:

---------------------------
Install Environment Manager
---------------------------

You will need to choose a suitable Python environment manager.
If you have Python installed, Pip should be installed alongside it.


.. list-table::
   :header-rows: 1
   :width: 60%

   * - Manager
     - CLIMADA Core
     - CLIMADA Petals
   * - `Mamba`_
     - ✅
     - ✅
   * - `Pixi`_
     - :ref:`Simple <install-simple>` only
     - :ref:`Simple <install-simple>` only
   * - `uv`_
     - ✅
     - ❌
   * - Pip
     - ✅
     - ❌

**We highly recommend `Mamba`_ or `Conda`_, as this supports both simple and advanced installations for both Core and Petals**, and installs a useful shell on Windows.
The tool ``mamba`` (see :ref:`conda-instead-of-mamba`) is available in the installer `Miniforge`_:

.. tabs::

   .. group-tab:: macOS & Linux

      * Open the "Terminal" app, copy-paste the two commands below, and hit enter:

      .. code-block:: shell

         curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
         bash Miniforge3-$(uname)-$(uname -m).sh

      * Accept the license terms.
      * You can confirm the default location.
      * Answer 'yes' when asked if if you wish to update your shell profile to automatically initialize conda. **Do not just hit ENTER but first type 'yes'**
      * If at some point you encounter ``command not found: mamba``, open a new terminal window.
      * If you encounter ``Run 'mamba init' to be able to run mamba activate/deactivate ...``, please run ``mamba init zsh`` or ``mamba init``.

   .. group-tab:: Windows

      * Download the Windows installer at the Install section from `Miniforge`_.
      * Execute the installer. This will install Mamba and provide the "Miniforge Prompt" program as a command line replacement.

.. _python-versions:

.. admonition:: Python Versions

   CLIMADA is primarily tested against a **supported** Python version, but is allowed to run with others.
   If you follow the installation instructions exactly, you will create an environment with the supported version.
   Depending on your setup, you are free to choose another allowed version, but we recommend the supported one.

   .. list-table::
      :width: 60%

      * - **Supported Version**
        - ``3.11``
      * - Allowed Versions
        - ``3.11``, ``3.12``, ``3.13``


.. _install-simple:
.. _install-pixi:

-------------------
Simple Instructions
-------------------

These instructions will install the most recent stable version of CLIMADA without cloning its repository.

.. tabs::

   .. group-tab:: Mamba

      #. Open the command line.
         Create a new Conda environment with CLIMADA by executing

         .. code-block:: shell

            mamba create -n climada_env -c conda-forge climada

      #. Activate the environment:

         .. code-block:: shell

            mamba activate climada_env

         You should now see ``(climada_env)`` appear in the beginning of your command prompt.
         This means the environment is activated.

      #. Verify that everything is installed correctly by executing a single test:

         .. code-block:: shell

            python -m unittest climada.engine.test.test_impact

         Executing CLIMADA for the first time will take some time because it will generate a directory tree in your home/user directory.
         After a while, some text should appear in your terminal.
         In the end, you should see an "Ok".
         If so, great! You are good to go.

      #. *Optional:* Install CLIMADA Petals into the environment:

         .. code-block:: shell

            mamba install -n climada_env -c conda-forge climada-petals

         .. attention::

            On Windows, installing ``climada-petals`` with Conda/Mamba may be prohibited as it takes forever to resolve the dependency tree.
            In these cases, `Pixi`_ is often the only reliable fallback.

   .. group-tab:: Pixi

      As an alternative to mamba `Pixi`_ can be used to install climada from conda-forge.
      Pixi is a modern package manager, based on Conda, that offers much faster dependency resolution.

      #. Open the command line and create a new folder for your Pixi project:

         .. code-block:: shell

            pixi init climada_pixi

      #. Add CLIMADA from ``conda-forge``:

         .. code-block:: shell

            pixi add --manifest-path climada_pixi/pixi.toml climada

      #. Verify that everything is installed correctly by executing a single test:

         .. code-block:: shell

            # the pixi way to activate an environment
            pixi shell --manifest-path climada_pixi/pixi.toml
            # execute the test
            python -m unittest climada.engine.test.test_impact
            # the pixi way to deactivate an environment
            exit

         Executing CLIMADA for the first time in a pixi environment will take quite some time because it will generate
         a directory tree in your home/user directory. If this test passes, great! You are good to go.

      #. *Optional:* Install CLIMADA Petals into the same Pixi environment:

         .. code-block:: shell

            pixi add --manifest-path climada_pixi/pixi.toml climada-petals

   .. group-tab:: uv

      You can `execute a script or program <https://docs.astral.sh/uv/guides/scripts/>`_ and include CLIMADA in the current environment like so:

      .. code-block:: shell

         uv run --with climada <script.py>

      Start a JupyterLab session with CLIMADA installed by executing

      .. code-block:: shell

         uv run -w climada -w jupyterlab jupyter lab

   .. group-tab:: Pip

      #. Open the command line.
         It usually makes sense to install CLIMADA in a new virtual environment (venv).
         Choose a suitable ``path/to/climada_venv``, and create one there:

         .. code-block:: shell

            python -m venv path/to/climada_venv

      #. Activate the virtual environment.

         .. code-block:: shell

            source path/to/climada_venv/bin/activate

         You should see ``(climada_venv)`` at the beginning of your command line now.

      #. Install CLIMADA into the venv.

         .. code-block:: shell

            pip install climada

      .. note::

         To deactivate the venv, simply execute

         .. code-block:: shell

            deactivate

.. _install-advanced:

---------------------------------------------
Advanced Instructions: Installing from Source
---------------------------------------------

For advanced Python users or developers of CLIMADA, cloning the CLIMADA repository and installing the package from source.
Installing from source also allows you to :ref:`switch branches <switch-branch>`.

#. If you are using a **Linux** OS, make sure you have ``git`` installed
   (Windows and macOS users are good to go once Conda is installed).
   On Ubuntu and Debian, you may use APT:

   .. code-block:: shell

      apt update
      apt install git

   Both commands will probably require administrator rights, which can be enabled by prepending ``sudo``.

#. Create a folder for your code.
   We will call it **workspace directory**.
   To make sure that your user can manipulate it without special privileges, use a subdirectory of your user/home directory.
   Do **not** use a directory that is synchronized by cloud storage systems like OneDrive, iCloud or Polybox!

#. Open the command line and navigate to the workspace directory you created using ``cd``.
   Replace ``<path/to/workspace>`` with the path of the workspace directory:

   .. code-block:: shell

      cd <path/to/workspace>

#. Clone CLIMADA from its `GitHub repository <https://github.com/CLIMADA-project/climada_python>`_.
   Enter the directory and check out the branch of your choice.
   The latest development version will be available under the branch ``develop``.

   .. code-block:: shell

      git clone https://github.com/CLIMADA-project/climada_python.git
      cd climada_python
      git checkout develop

The following steps depend on your choice of :ref:`package manager <install-manager>`:

.. tabs::

   .. group-tab:: Mamba

      .. warning::

         If you followed the :ref:`install-simple` before, make sure you **either** remove the environment with:

         .. code-block:: shell

            mamba env remove -n climada_env

         before you continue, **or** you use a **different** environment name for the following instructions (e.g. ``climada_dev`` instead of ``climada_env``).

      #. Create an Conda environment called ``climada_env`` for installing CLIMADA:

         .. code-block:: shell

            mamba create -n climada_env "python=3.11.*"

         .. hint::

            Use the wildcard ``.*`` at the end to allow a downgrade of the bugfix version of Python.
            This increases compatibility when installing the requirements in the next step.

         .. note:: You may choose any of the  :ref:`allowed Python versions <python-versions>` from the list above.

      #. Use the default environment specs in ``env_climada.yml`` to install all dependencies.
         Then activate the environment:

         .. code-block:: shell

            mamba env update -n climada_env -f requirements/env_climada.yml
            mamba activate climada_env

      #. Install the local CLIMADA source files as Python package using ``pip``:

         .. code-block:: shell

            python -m pip install -e ./

         .. hint::

            Using a path ``./`` (referring to the path you are currently located at) will instruct ``pip`` to install the local files instead of downloading the module from the internet.
            The ``-e`` (for "editable") option further instructs ``pip`` to link to the source files instead of copying them during installation.
            This means that any changes to the source files will have immediate effects in your environment, and re-installing the module is only required when the dependencies change.

      #. Verify that everything is installed correctly by executing a single test:

         .. code-block:: shell

            python -m unittest climada.engine.test.test_impact

         Executing CLIMADA for the first time will take some time because it will generate a directory tree in your home/user directory.
         If this test passes, great!
         You are good to go.

   .. group-tab:: uv

      #. Create a local venv and synchronize it:

         .. code-block:: shell

            uv sync

         .. note:: uv will automatically install the local package in editable mode.

   .. group-tab:: Pip

      #. Create a virtual environment in the repository folder and activate it:

         .. code-block:: shell

            python -m venv .venv
            source .venv/bin/activate

      #. Install CLIMADA into the environment:

         .. code-block:: shell

            python -m pip install -e ./

         .. hint::

            Using a path ``./`` (referring to the path you are currently located at) will instruct ``pip`` to install the local files instead of downloading the module from the internet.
            The ``-e`` (for "editable") option further instructs ``pip`` to link to the source files instead of copying them during installation.
            This means that any changes to the source files will have immediate effects in your environment, and re-installing the module is only required when the dependencies change.

.. _devdeps:

Install Developer Dependencies (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Building the documentation and running the entire test suite of CLIMADA requires additional dependencies () which are not installed by default.
They are also not needed for using CLIMADA.
However, if you want to develop CLIMADA, we strongly recommend you install them.

Dependency Groups
"""""""""""""""""

The CLIMADA Python package defines the following `dependency groups <https://packaging.python.org/en/latest/specifications/dependency-groups/>`_:

.. list-table::
   :header-rows: 1

   * - Dependency Group
     - Includes Dependencies...
   * - ``dev``
     - for all developer needs (includes all others)
   * - ``doc``
     - for building the documentation files from in-source pydoc annotations
   * - ``test``
     - for executing the tests

Installation works as follows:

.. tabs::

   .. group-tab:: Mamba

      With the ``climada_env`` activated, enter the workspace directory and then the CLIMADA repository as above.
      Then, tell Pip to install the ``dev`` dependency group:

      .. code-block:: shell

         python -m pip install --group dev

   .. group-tab:: uv

      No action required, ``uv sync`` installs the ``dev`` dependency group by default.
      To request (only) a specific group, execute

      .. code-block:: shell

         uv sync --group <group>

   .. group-tab:: Pip

      Activate the venv, and enter the workspace directory and then the CLIMADA repository as above.
      Then, tell Pip to install the ``dev`` dependency group:

      .. code-block:: shell

         python -m pip install --group dev

      .. note::

         If your ``pip`` does not recognize the ``--group`` option, make sure you update to the latest version:

         .. code-block:: shell

            python -m pip install --upgrade pip

pre-commit
""""""""""

We use `pre-commit <https://pre-commit.com/#intro>`_ to install and run automated, so-called pre-commit hooks before a new commit.
These ensure, e.g., that files are formatted correctly.
Please refer to the :ref:`guide on pre-commit hooks <dev-pre-commit-hooks>` for information on how to use this tool.

In order to use the hooks defined in ``.pre-commit-config.yaml``, you need to install pre-commit and its hooks first.

.. tabs::

   .. group-tab:: Mamba

      With the ``climada_env`` activated, move to the repository and execute

      .. code-block:: shell

         python -m pip install pre-commit
         pre-commit install

   .. group-tab:: uv

      uv supports running `tools <https://docs.astral.sh/uv/guides/tools/>`_ that are not part of the local Python environment.
      These tools can be downloaded and run when invoked, or `installed <https://docs.astral.sh/uv/guides/tools/#installing-tools>`_ when needed repeatedly.
      We use this function for installing pre-commit:

      .. code-block:: shell

         uv tool install pre-commit --with pre-commit-uv

      pre-commit is now available as tool in your shell.
      Move to the repository and install the hooks:

      .. code-block:: shell

         pre-commit install

   .. group-tab:: Pip

      With the ``climada_venv`` activated, move to the repository and execute

      .. code-block:: shell

         python -m pip install pre-commit
         pre-commit install

make
""""

Instructions for running the test scripts can be found in the :doc:`Testing Guide </development/Guide_Testing>`.
For executing the pre-defined test scripts in exactly the same way as they are executed by the automated CI pipeline, you will need ``make`` to be installed.

.. tabs::

   .. group-tab:: macOS & Linux

      No action required.
      ``make`` is installed by default.

   .. group-tab:: Windows

      .. tabs::

         .. group-tab:: Mamba

            .. code-block:: shell

               mamba install -n climada_env make

         .. group-tab:: uv

            *tbd*

         .. group-tab:: Pip

            *tbd*

Install CLIMADA Petals (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are unsure whether you need Petals, see the :ref:`notes above <petals-notes>`.

To install CLIMADA Petals, we assume you have already installed CLIMADA Core with the :ref:`advanced instructions <install-advanced>` above.

#. Open the command line and navigate to the workspace directory.
#. Clone CLIMADA Petals from its `repository <https://github.com/CLIMADA-project/climada_petals>`_.
   Enter the directory and check out the branch of your choice.
   The latest development version will be available under the branch ``develop``.

   .. code-block:: shell

      git clone https://github.com/CLIMADA-project/climada_petals.git
      cd climada_petals
      git checkout develop

The following steps depend on your choice of :ref:`package manager <install-manager>`:

.. tabs::

   .. group-tab:: Mamba

      #. Update the Conda environment with the specifications from Petals and activate it:

         .. code-block:: shell

            mamba env update -n climada_env -f requirements/env_climada.yml
            mamba activate climada_env

      #. Install the CLIMADA Petals package:

         .. code-block:: shell

            python -m pip install -e ./

   .. group-tab:: uv

      *Petals currently cannot be installed from source with uv.*

   .. group-tab:: Pip

      *Petals currently cannot be installed from source with Pip.*

------------
Code Editors
------------

JupyterLab
^^^^^^^^^^

.. tabs::

   .. group-tab:: Mamba

      #. Install JupyterLab into the Conda environment:

         .. code-block:: shell

            mamba install -n climada_env -c conda-forge jupyterlab

      #. Make sure that the ``climada_env`` is activated (see above) and then start JupyterLab:

         .. code-block:: shell

            mamba activate climada_env
            jupyter lab

         JupyterLab will open in a new window of your default browser.

   .. group-tab:: uv

      Start JupyterLab in the local environment without adding it to the dependencies:

      .. code-block:: shell

         uv run --with jupyterlab jupyter lab

   .. group-tab:: Pip

      #. When inside the CLIMADA repository folder, activate the venv and install JupyterLab:

         .. code-block:: shell

            source ./venv/bin/activate
            python -m pip install jupyterlab

      #. Start JupyterLab:

         .. code-block:: shell

            jupyter lab

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

   If you are **not** using Mamba, your Python environment is not known system-wide.
   To find your interpreter, open the VSCode in the folder where your repository or venv is located.
   VSCode should detect it then.

See the VSCode docs on `Python <https://code.visualstudio.com/docs/python/python-tutorial>`_ and `Jupyter Notebooks <https://code.visualstudio.com/docs/datascience/jupyter-notebooks>`_ for further information.

.. hint::

   Both of the following setup instructions work analogously for Core and Petals.
   The specific instructions for Petals are shown in square brackets: []

Workspace Setup
"""""""""""""""

Setting up a workspace for the CLIMADA source code is only available for :ref:`advanced installations <install-advanced>`.

#. Open a new VSCode window.
   Below *Start*, click *Open...*, select the ``climada_python`` [``climada_petals``] repository folder in your workspace directory, and click on *Open* on the bottom right.

#. Click *File* > *Save Workspace As...* and store the workspace settings file next to (**not** in!) the ``climada_python`` [``climada_petals``] folder.
   This will enable you to load the workspace and all its specific settings in one go.

#. Open the Command Palette by clicking *View* > *Command Palette* or by using the shortcut keys ``Ctrl+Shift+P`` (Windows, Linux) / ``Cmd+Shift+P`` (macOS).
   Start typing "Python: Select Interpreter" and select it from the dropdown menu.
   If prompted, choose the option to set the interpreter for the workspace, not just the current folder.
   Then, choose the Python interpreter from the ``climada_env`` (or venv).

For further information, refer to the VSCode docs on `Workspaces <https://code.visualstudio.com/docs/editor/workspaces>`_.

Test Explorer Setup
"""""""""""""""""""

After you set up a workspace, you might want to configure the test explorer for easily running the CLIMADA test suite within VSCode.

.. note:: Please install the additional :ref:`test dependencies <devdeps>` before proceeding.

#. In the left sidebar, select the "Testing" symbol, and click on *Configure Python Tests*.

#. Select "pytest" as test framework and then select ``climada`` [``climada_petals``] as the directory containing the test files.

#. Select "Testing" in the Activity Bar on the left or through *View* > *Testing*.
   The "Test Explorer" in the left sidebar will display the tree structure of modules, files, test classes and individual tests.
   You can run individual tests or test subtrees by clicking the Play buttons next to them.

#. By default, the test explorer will show test output for failed tests when you click on them.
   To view the logs for any test, click on *View* > *Output*, and select "Python Test Log" from the dropdown menu in the view that just opened.
   If there are errors during test discovery, you can see what's wrong in the "Python" output.

For further information, see the VSCode docs on `Python Testing <https://code.visualstudio.com/docs/python/testing>`_.

Spyder
^^^^^^

.. dropdown:: Spyter Setup (outdated)

   Installing Spyder into the existing Conda environment for CLIMADA might fail depending on the exact versions of dependencies installed.
   Therefore, we recommend installing Spyder in a *separate* environment, and then connecting it to a kernel in the original ``climada_env``.

   #. Follow the `Spyder installation instructions <https://docs.spyder-ide.org/current/installation.html#installing-with-conda>`_.
      You can follow the "Conda" installation instructions.
      Keep in mind you are using ``mamba``, though!

   #. Check the version of the Spyder kernel in the new environment:

      .. code-block:: shell

         mamba env export -n spyder-env | grep spyder-kernels

      This will return a line like this:

      .. code-block:: shell

         - spyder-kernels=X.Y.Z=<hash>

      Copy the part ``spyder-kernels=X.Y.Z`` (until the second ``=``) and paste it into the following command to install the same kernel version into the ``climada_env``:

      .. code-block:: shell

         mamba install -n climada_env spyder-kernels=X.Y.Z

   #. Obtain the path to the Python interpreter of your ``climada_env``.
      Execute the following commands:

      .. code-block:: shell

         mamba activate climada_env
         python -c "import sys; print(sys.executable)"

      Copy the resulting path.

   #. Open Spyder through the command line:

      .. code-block:: shell

         mamba activate spyder-env
         spyder

   #. Set the Python interpreter used by Spyder to the one of ``climada_env``.
      Select *Preferences* > *Python Interpreter* > *Use the following interpreter* and paste the iterpreter path you copied from the ``climada_env``.

-----------------------------
Apps for Working with CLIMADA
-----------------------------

To work with CLIMADA, you will need an application that supports Jupyter Notebooks.
There are plugins available for nearly every code editor or IDE, but if you are unsure about which to choose, we recommend `JupyterLab <https://jupyterlab.readthedocs.io/en/stable/>`_, `Visual Studio Code <https://code.visualstudio.com/>`_ or `Spyder <https://www.spyder-ide.org/>`_.
It is easy to get confused by all the different softwares and their uses so here is an overview of which tools we use for what:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Use
     - Tools
     - Description
     - Useful for
   * - Distribution / manage virtual environment & packages
     - **Recommended:**
       Mamba
       **Alternatives:**
       Anaconda
     - - Install climada, manage & use the climada virtual environment, install packages
       - Anaconda includes Anaconda Navigator, which is a desktop GUI and can be used to launch applications like Jupyter Notebook, Spyder, etc.
     - Climada Users
       & Developers
   * - IDE (Integrated Development Environment)
     - **Recommended:**
       VSCode
       **Alternatives:**
       Spyder, JupyterLab, PyCharm, & many more
     - - Write and run code
       - Useful for Developers:
         - VSCode also has a GUI to commit changes to Git (similar to GitHub Desktop, but in the same place as your code)
         - VSCode test explorer shows results for individual tests & any classes and files containing those tests (folders display a failure or pass icon)
     - Climada Users
       & Developers
   * - Git GUI (Graphical User Interface)
     - GitHub Desktop, GitKraken
     - - Provides an interface which keeps track of the branch you’re working on, changes you made, etc.
       - Allows you to commit changes, push to GitHub, etc. without having to use the command line
       - The code itself is not written using these applications but with your IDE of choice (see above)
     - Climada Developers
   * - Continuous integration (CI) server
     - Jenkins
     - - Automatically checks code changes in GitHub repositories, e.g., when you create a pull request for the develop branch
       - Performs static code analysis using pylint
       - You don't need to do any installations yourself; this runs automatically when you push new code to GitHub
       - See `Continuous Integration and GitHub Actions <../development/Guide_continuous_integration_GitHub_actions.ipynb>`_
     - Climada Developers

----
FAQs
----

Answers to frequently asked questions.

.. _update-climada:

How to update CLIMADA?
^^^^^^^^^^^^^^^^^^^^^^

We recommend keeping CLIMADA up-to-date.
To update, follow the instructions based on your :ref:`installation type <install-choice>`:

Simple Instructions
"""""""""""""""""""

.. tabs::

   .. group-tab:: Mamba

      .. code-block:: shell

         mamba update -n climada_env -c conda-forge climada

      If Petals is installed, you can also update it:

      .. code-block:: shell

         mamba update -n climada_env -c conda-forge climada_petals

   .. group-tab:: uv

      No action required.
      If you have not `locked <https://docs.astral.sh/uv/guides/scripts/#locking-dependencies>`_ your script dependencies, uv will always use the latest version.

   .. group-tab:: Pip

      With the venv activated, upgrade CLIMADA:

      .. code-block:: shell

         python -m pip install --upgrade climada

.. _update-climada-advanced:

Advanced Instructions
"""""""""""""""""""""

Move into your local CLIMADA repository and pull the latest version of your respective branch:

.. code-block:: shell

   cd <path/to/workspace>/climada_python
   git pull

Then, update the environment and reinstall the package:

.. tabs::

   .. group-tab:: Mamba

      .. code-block:: shell

         mamba env update -n climada_env -f requirements/env_climada.yml
         mamba activate climada_env
         python -m pip install -e ./

      The same instructions apply for CLIMADA Petals.

      .. note::

         In rare cases, updating the environment might fail after switching a branch because the dependency updates are incompatible.
         If this happens, create a new environment (see the :ref:`advanced instructions <install-advanced>`).

   .. group-tab:: uv

      .. code-block:: shell

         uv sync

   .. group-tab:: Pip

      Activate the venv first.

      .. code-block:: shell

         python -m pip install -e ./


.. _switch-branch:

How to switch branches?
^^^^^^^^^^^^^^^^^^^^^^^

This is only available for :ref:`advanced installations <install-advanced>` from source.

If you want to check out or review a specific feature, you may want to switch to a `Git branch <https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell>`_ other than ``develop``.
To switch to a new branch, open a terminal and move to the CLIMADA repository.
Then fetch the latest changes from the remote, switch to the target branch, and update your local version:

.. code-block:: shell

   cd <path/to/workspace>/climada_python
   git fetch
   git checkout <branch>
   git pull

After switching, it is good practice to :ref:`update your environment <update-climada-advanced>`, as the dependency definitions might have been changed.

The above instructions also apply for CLIMADA Petals, but you have to ``cd`` into the ``climada_petals`` repository in the first step, of course.

.. _install-more-packages:

Installing More Packages
^^^^^^^^^^^^^^^^^^^^^^^^

You might use CLIMADA in code that requires more packages than the ones readily available in the CLIMADA environment.

.. tabs::

   .. group-tab:: Mamba

      **Prefer installing these packages via Conda**, and only rely on ``pip`` if that fails.
      The default channels of Conda sometimes contain outdated versions.
      Therefore, use the ``conda-forge`` channel:

      .. code-block:: shell

         mamba install -n climada_env -c conda-forge <package>

      Only if the desired package (version) is not available, go for ``pip``:

      .. code-block:: shell

         mamba activate climada_env
         python -m pip install <package>

   .. group-tab:: uv

      If you need additional dependencies for a script or program, use ``-w/--with`` (possibly multiple times):

      .. code-block:: shell

         uv run --with <package> <script.py>

      If you manage a downstream project, you can `add a new dependency <https://docs.astral.sh/uv/guides/projects/#managing-dependencies>`_.

   .. group-tab:: Pip

      Activate the venv.
      Then use pip:

      .. code-block:: shell

         python -m pip install <package>

Verifying Your Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are unsure if everything works as intended, try running all tests.
This is only available for :ref:`advanced setups <install-advanced>`!

.. tabs::

   .. group-tab:: Mamba

      Move into the CLIMADA repository, activate the environment and then execute the tests:

      .. code-block:: shell

         cd <path/to/workspace>/climada_python
         mamba activate climada_env
         make unit_test
         make integ_test

   .. group-tab:: uv

      Move into the CLIMADA repository and then execute the tests:

      .. code-block:: shell

         uv run -- make unit_test
         uv run -- make integ_test

   .. group-tab:: Pip

      Move into the CLIMADA repository, activate the venv and then execute the tests:

      .. code-block:: shell

         cd <path/to/workspace>/climada_python
         source .venv/bin/activate
         make unit_test
         make integ_test

Error: ``ModuleNotFoundError``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Something is wrong with the environment you are using.
After **each** of the following steps, check if the problem is solved, and only continue if it is **not**:

.. tabs::

   .. group-tab:: Mamba

      #. Make sure you are working in the CLIMADA environment:

         .. code-block:: shell

            mamba activate climada_env

      #. :ref:`Update the Conda environment and CLIMADA <update-climada>`.

      #. Conda will notify you if it is not up-to-date.
         In this case, follow its instructions to update it.
         Then, repeat the last step and update the environment and CLIMADA (again).

      #. Install the missing package manually.
         Follow the instructions for :ref:`installing more packages <install-more-packages>`.

      #. If you reached this point, something is severely broken.
         The last course of action is to delete your CLIMADA environment:

         .. code-block:: shell

            mamba deactivate
            mamba env remove -n climada_env

         Now repeat the :ref:`installation process <install-choice>`.

      #. Still no good?
         Please raise an `issue on GitHub <https://github.com/CLIMADA-project/climada_python/issues>`_ to get help.

   .. group-tab:: uv

      *todo*

   .. group-tab:: Pip

      *todo*

Logging Configuration
^^^^^^^^^^^^^^^^^^^^^

Climada makes use of the standard `logging <https://docs.python.org/3/howto/logging.html>`_ package.
By default, the "climada"-``Logger`` is detached from ``logging.root``, logging to `stdout` with
the level set to ``WARNING``.

If you prefer another logging configuration, e.g., for using Climada embedded in another application,
you can opt out of the default pre-configuration by setting the config value for
``logging.climada_style`` to ``false`` in the :doc:`configuration file </development/Guide_Configuration>`
``climada.conf``.

Changing the logging level can be done in multiple ways:

* Adjust the :doc:`configuration file </development/Guide_Configuration>` ``climada.conf`` by setting a the value of the ``global.log_level`` property.
  This only has an effect if the ``logging.climada_style`` is set to ``true`` though.

* Set a global logging level in your Python script:

  .. code-block:: python

     import logging
     logging.getLogger('climada').setLevel(logging.ERROR)  # to silence all warnings

* Set a local logging level in a context manager:

  .. code-block:: python

     from climada.util import log_level
     with log_level(level="INFO"):
         # This also emits all info log messages
         foo()

     # Default logging level again
     bar()

All three approaches can also be combined.

.. _conda-instead-of-mamba:

Conda as Alternative to Mamba
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We experienced several issues with the default ``conda`` package manager lately.
This is likely due to the large dependency set of CLIMADA, which makes solving the environment a tedious task.
We therefore switched to the more performant ``mamba`` and recommend using it.

.. caution::

   In theory, you could also use an `Anaconda <https://docs.anaconda.com/free/anaconda/>`_ or `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ distribution and replace every ``mamba`` command in this guide with ``conda``.
   In practice, however, ``conda`` is often unable to solve an environment that ``mamba`` solves without issues in few seconds.

Error: ``operation not permitted``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conda might report a permission error on macOS Mojave.
Carefully follow these instructions: https://github.com/conda/conda/issues/8440#issuecomment-481167572

No ``impf_TC`` Column in ``GeoDataFrame``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This may happen when a demo file from CLIMADA was not updated after the change in the impact function naming pattern from ``if_`` to ``impf_`` when `CLIMADA v2.2.0 <https://github.com/CLIMADA-project/climada_python/releases/tag/v2.2.0>`_ was released.
Execute

.. code-block:: shell

   mamba activate climada_env
   python -c "import climada; climada.setup_climada_data(reload=True)"

.. _install-glossary:

------------------------
The What Now? (Glossary)
------------------------

You might have become confused about all the names thrown at you.
Let's clear that up:

Terminal, Command Line
    A text-only program for interacting with your computer (the old fashioned way).
    If you are using `Miniforge`_ on Windows, the program is called "Miniforge Prompt".

`Conda`_
    A cross-platform package management system. Comes in different varieties (distributions).

`Mamba`_
    The faster reimplementation of the ``conda`` package manager.

`Pixi`_
    The much faster modern package manager based on ``conda``.

`uv`_
    A modern, fast Python package manager.

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


.. _Conda: https://docs.conda.io/en/latest/
.. _Mamba: https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
.. _Miniforge: https://github.com/conda-forge/miniforge
.. _Pixi: https://pixi.sh/
.. _CLIMADA Petals: https://climada-petals.readthedocs.io/en/latest/
.. _uv: https://docs.astral.sh/uv/
