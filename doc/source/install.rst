.. _Installation:

Installation
************

Please execute the instructions of the following text boxes in a Terminal or Anaconda Prompt.

Download
========
**Version in development**: Download or clone the GitHub repository `climada_python <https://github.com/CLIMADA-project/climada_python.git>`_.

To clone the repository, you need to first `install git <https://www.linode.com/docs/development/version-control/how-to-install-git-on-linux-mac-and-windows/>`_ or install `GitHub Desktop <https://desktop.github.com>`_. Afterwards you can use the following command line or GitHub Desktop options (depending on your git installation choice) with climada_python's URL::

  git clone https://github.com/CLIMADA-project/climada_python.git

**Stable version**: Download the last CLIMADA release available in `climada releases <https://github.com/CLIMADA-project/climada_python/releases>`_ as a zip or tar.gz file.

Unix Operating System
=====================

.. _Install environment with Anaconda:

Install environment with Anaconda
---------------------------------
1. **Anaconda**: Download or update to the latest version of `Anaconda <https://www.anaconda.com/>`_. Execute it.

2. **Install dependencies**: In the *Environments* section, use the *Import* box to create a new virtual environment from a yml file. A dialogue box will ask you for the location of the file. Provide first the path of climada's ``requirements/env_climada.yml``. The default name of the environment, *climada_env*, appears. Click the *Import* button to start the installation. 

  The installation of the packages will take some minutes. No dialogue box should appear in the meantime. If an error happens, try to solve it looking into the details description.

  To include *climada_python* in the environment's path, do the following. In your environments folder, for example */home/user/anaconda3/*::
   
   cd envs/climada_env/lib/python3.6/site-packages
   echo '/your/path/to/climada_python/' > climada_env_path.pth

3. **Test installation**: Before leaving the *Environments* section of Anaconda, make sure that the climada environment, *climada_env* is selected. Go to the *Home* section of Anaconda and install and launch Spyder (or your preferred editor). Open the file containing all the unit tests, ``tests_runner.py`` in ``climada_python`` folder and execute it. If the installation has been successful, an OK will appear at the end (the execution should last less than 5min).

4. **Run tutorials**: In the *Home* section of Anaconda, with *climada_env* selected, install and launch *jupyter notebook*. A browser window will show up. Navigate to your ``climada_python`` repository and open ``climada_python/script/tutorial/1_main_climada.ipynb``. This is the tutorial which will guide you through all climada's functionalities. Execute each code cell to see the results, you might also edit the code cells before executing.

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

4. **Run tutorials**: Install and launch *jupyter notebook*::

    jupyter notebook --notebook-dir /path/to/climada_python

 A browser window will show up. Open ``climada_python/script/tutorial/1_main_climada.ipynb``. This is the tutorial which will guide you through all climada's functionalities. Execute each code cell to see the results, you might also edit the code cells before executing.

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

Windows Operating System
========================

Install environment with Anaconda
---------------------------------

See `Install environment with Anaconda`_.

FAQs
====
* ModuleNotFoundError; climada libraries are not found. Try to include *climada_python* path in the environment *climada_env* path as suggested in Section 2 of `Install environment with Anaconda`_. If it does not work you can always include the path manually before executing your code::

    import sys
    sys.path.append('path/to/climada_python')

* ModuleNotFoundError; some python library is not found. It might happen that the pip dependencies of *env_climada.yml* (the ones specified after ``pip:``) have not been installed in the environment *climada_env*. You can then install them manually one by one as follows::

    source activate climada_env
    pip install library_name

  where ``library_name`` is the missing library.
