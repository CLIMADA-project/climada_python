.. _Contributing:

Contributing
============

Contributions are very welcome! Please follow these steps:

0. **Install** `Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_ and `Anaconda <https://www.anaconda.com/>`_ (or `Miniconda <https://conda.io/miniconda.html>`_).
1. **Fork** the project on GitHub::

    git clone https://github.com/CLIMADA-project/climada_python.git

2. **Install the packages** in ``climada_python/requirements/env_climada.yml`` and ``climada_python/requirements/env_developer.yml`` (see :doc:`install`).
3. Make well commented and clean **commits** to the repository. You can make a new **branch** here if you are modifying more than one part or feature.
4. Make unit and integration **tests** on your code, preferably during development.
5. Perform a **static code analysis** of your code using ``pylint`` with CLIMADA's configuration ``.pylintrc``.
6. Add new **data dependencies** used in :doc:`data_dependencies` and write a **tutorial** if a new class has been introduced (see :doc:`tutorial`).
7. Add your name to the **AUTHORS** file.
8. **Push** the branch to GitHub::

    git push origin my-new-feature

9. On GitHub, create a new **pull request** from the feature branch.


Notes
-----

Update CLIMADA's environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remember to regularly update your code as well as climada's environment. You might use the following commands to update the environments::

    cd climada_python
    git pull
    source activate climada_env
    conda env update --file requirements/env_climada.yml
    conda env update --file requirements/env_developer.yml

If any problem occurs during this process, consider reinstalling everything from scratch following the :doc:install instructions. 
You can find more information about virtual environments with conda `here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.
