.. _Installation:

Installation
************

Please execute the instructions of the following text boxes in a Terminal or Anaconda Prompt.

Download
========

Download the last CLIMADA release available in `climada releases <https://github.com/CLIMADA-project/climada_python/releases>`_ as a zip or tar.gz file. Uncompress it to your local computer. Hereinafter ``climada_python-x.y.z`` refers to the downloaded folder of CLIMADA version x.y.z.

Unix Operating System
=====================

.. _Install environment with Anaconda:

Install environment with Anaconda
---------------------------------
1. **Anaconda**: Download or update to the latest version of `Anaconda <https://www.anaconda.com/>`_. Execute it.

2. **Install dependencies**: In the *Environments* section, use the *Import* box to create a new virtual environment from a yml file. A dialogue box will ask you for the location of the file. Provide first the path of climada's ``climada_python-x.y.z/requirements/env_climada.yml``. The default name of the environment, *climada_env*, appears. Click the *Import* button to start the installation. 

  The installation of the packages will take some minutes. No dialogue box should appear in the meantime. If an error happens, try to solve it looking into the details description.

  *Optional*: To include *climada_python-x.y.z* in the environment's path, do the following. In your environments folder, for example */home/user/anaconda3/*::
   
   cd envs/climada_env/lib/python3.6/site-packages
   echo '/your/path/to/climada_python-x.y.z/' > climada_env_path.pth

3. **Test installation**: Before leaving the *Environments* section of Anaconda, make sure that the climada environment, *climada_env* is selected. Go to the *Home* section of Anaconda and install and launch Spyder (or your preferred editor). Open the file containing all the installation tests, ``tests_install.py`` in ``climada_python-x.y.z`` folder and execute it. If the installation has been successful, an OK will appear at the end (the execution should last less than 2min).

4. **Run tutorials**: In the *Home* section of Anaconda, with *climada_env* selected, install and launch *jupyter notebook*. A browser window will show up. Navigate to your ``climada_python-x.y.z`` repository and open ``doc/tutorial/1_main_climada.ipynb``. This is the tutorial which will guide you through all climada's functionalities. Execute each code cell to see the results, you might also edit the code cells before executing. See :doc:`tutorial` for more information.

Install environment with Miniconda
----------------------------------
1. **Miniconda**: Download or update to the latest version of `Miniconda <https://conda.io/miniconda.html>`_.

2. **Install dependencies**: Create the virtual environment *climada_env* with climada's dependencies::

    cd climada_python-x.y.z
    conda env create -f requirements/env_climada.yml --name climada_env

   *Optional*: To include *climada_python-x.y.z* in the environment's path, do the following. In your environments folder, for example */home/user/miniconda3/*::
   
    cd envs/climada_env/lib/python3.6/site-packages
    echo '/your/path/to/climada_python-x.y.z/' > climada_env_path.pth

3. **Test installation**: Activate the environment, execute the installation tests and deactivate the environment when finished using climada::

    source activate climada_env
    python3 tests_install.py
    source deactivate

 If the installation has been successful, an OK will appear at the end (the execution should last less than 2min).

4. **Run tutorials**: Install and launch *jupyter notebook*::

    jupyter notebook --notebook-dir /path/to/climada_python-x.y.z

 A browser window will show up. Open ``climada_python-x.y.z/doc/tutorial/1_main_climada.ipynb``. This is the tutorial which will guide you through all climada's functionalities. Execute each code cell to see the results, you might also edit the code cells before executing. See :doc:`tutorial` for more information.

Windows Operating System
========================

Install environment with Anaconda
---------------------------------

See `Install environment with Anaconda`_.

Note: 

In Step 2's optional instructions, to include the path into climada's virtual environment, set a file with format `.pth` containing the path of `climada_python-x.y.z` (e.g. ``C:\\Users\\USERNAME\\Documents\\climada_python-x.y.z\\``) in::

   */Anaconda3/envs/climada_env/Lib/site-packages

FAQs
====
* ModuleNotFoundError; climada libraries are not found. Try to include *climada_python-x.y.z* path in the environment *climada_env* path as suggested in Section 2 of `Install environment with Anaconda`_. If it does not work you can always include the path manually before executing your code::

    import sys
    sys.path.append('path/to/climada_python-x.y.z')

* ModuleNotFoundError; some python library is not found. It might happen that the pip dependencies of *env_climada.yml* (the ones specified after ``pip:``) have not been installed in the environment *climada_env*. You can then install them manually one by one as follows::

    source activate climada_env
    pip install library_name

  where ``library_name`` is the missing library.
