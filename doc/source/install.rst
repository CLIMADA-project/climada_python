.. _Installation:

Installation
************

Download
========
Download or clone the GitHub repository `climada_python <https://github.com/davidnbresch/climada_python.git>`_. 
To clone it, you might want to use the `GitHub Desktop <https://desktop.github.com>`_ or the command line with the provided URL::

  git clone https://github.com/davidnbresch/climada_python.git

Developers with Unix
====================

.. _Install environment with Anaconda:

Install environment with Anaconda
---------------------------------
1. **Anaconda**: Download or update to the latest version of `Anaconda <https://www.anaconda.com/>`_. Execute it.

2. **Install dependencies**: In the *Environments* section, use the *Import* box to create a new virtual environment from a yml file. A dialogue box will ask you for the location of the file. Provide first the path of climada's ``requirements/env_climada.yml``. The default name of the environment, *climada_env*, appears. Click the *Import* button to start the installation. 

  The installation of the packages will take some minutes. No dialogue box should appear in the meantime. If an error happens, try to solve it looking into the details description.

  To include *climada_python* in the environment's path, do the following. In your environments folder, for example */home/user/anaconda3/*::
   
   cd envs/climada_env/lib/python3.6/site-packages
   echo '/your/path/to/climada_python/' > climada_env_path.pth

3. **Test installation**: Before leaving the *Environments* section of Anaconda, make sure that the climada environment, *climada_env* is selected. Go to the *Home* section of Anaconda and launch Spyder. Open the file containing all the unit tests, ``tests_runner.py`` in ``climada_python`` folder. If the installation has been successful, an OK will appear at the end of the execution.

Install environment with Miniconda
----------------------------------
1. **Miniconda**: Download or update to the latest version of `Miniconda <https://conda.io/miniconda.html>`_.

2. **Install dependencies**: Create the virtual environment *climada_env* with climada's dependencies::

    conda env create -f requirements/env_climada.yml --name climada_env 

   To include *climada_python* in the environment's path, do the following. In your environments folder, for example */home/user/miniconda3/*::
   
    cd envs/climada_env/lib/python3.6/site-packages
    echo '/your/path/to/climada_python/' > climada_env_path.pth

3. **Test installation**: Activate the environment, execute the unit tests and deactivate the environment when finished using climada::

    source activate climada_env
    python3 tests_runner.py
    source deactivate
  
 If the installation has been successful, an OK will appear at the end of the execution.

Update climada's environment
----------------------------
Before using climada's code in development, remember to update your code as well as climada's environment. The requirements in ``requirements/env_developer.yml`` contain all the packages which are necessary to execute the continuous integration of climada. These can be therefore useful for climada's contributors. 

If you use conda, you might use the following commands to update the environments::

    cd climada_python
    git pull
    source activate climada_env
    conda env update --file requirements/env_climada.yml
    conda env update --file requirements/env_developer.yml
    
If any problem occurs during this process, consider reinstalling everything from scratch following the `Installation`_ instructions. 
You can find more information about virtual environments with conda `here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.

Developers with Windows
=======================

Install environment with Anaconda
---------------------------------

See `Install environment with Anaconda`_.


Package Installation
====================

.. _Pre-requisites:

Pre-requisites
--------------

Following libraries need to be installed:

* GEOS >= 3.5.0

* PROJ4 = 4.9.3

* python = 3.6

Install distribution
--------------------

1. **Download distribution**: Download the desired distribution from the GitHub repository `climada_python <https://github.com/davidnbresch/climada_python.git>`_ in the ``releases`` section.

2. **Install climada**: Specifying the correct distribution, install it as follows::

    pip install climada-*.tar.gz

  If errors, try installing Cartopy first. With pip::

    pip install --global-option=build_ext --global-option="-I/path_to/proj/4.9.3/x86_64/include/:/path_to/geos/3.5.0/x86_64/include/" --global-option="-L/path_to/proj/4.9.3/x86_64/lib/" Cartopy 

  or with conda::

    conda install cartopy

3. **Test installation**: Execute the unit tests to ensure the installation has been successful::

    PYTHONPATH=. python3 tests_runner.py

 If the installation has been successful, an OK will appear at the end of the execution.

