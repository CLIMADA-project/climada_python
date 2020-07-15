.. _Contributing:

Contributing
============

Contributions are very welcome! Please follow these steps:

0. **Install** `Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
   and `Anaconda <https://www.anaconda.com/>`_ (or `Miniconda <https://conda.io/miniconda.html>`_).

1. **Clone (or fork)** the project on GitHub::

    git clone https://github.com/CLIMADA-project/climada_python.git

   For more information see :doc:`git_flow.rst`

2. **Install the packages** in ``climada_python/requirements/env_climada.yml`` and
   ``climada_python/requirements/env_developer.yml`` (see :doc:`install`). You
   might need to install additional environments contained in ``climada_python/requirements``
   when using specific functionalities.

3. You might make a new **branch**::

    git flow feature start feature_name
    
   Or work on an existing branch::

    git checkout -b branch_name

   See CLIMADA-python's branching policies in :doc:`git_flow.rst`.
 
   General information about Git branches:
   `about branches <https://help.github.com/en/articles/about-branches>`_.

4. Follow the :doc:`coding_conventions`. Write small readable methods, classes and functions.
   Make well commented and clean **commits** to the repository::

    git pull
    git stats         # use it to see your locally modified files
    git add climada/modified_file.py climada/test/test_modified_file.py
    git commit -m "new functionality of .. implemented"

5. Make unit and integration **tests** on your code, preferably during development:

   * Unit tests are located in the ``test`` folder located in same folder as the corresponding
     module. Unit tests should test all methods and functions using fake data if necessary.
     The whole test suit should run in less than 20 sec. They are all executed after each push
     in `Jenkins <http://ied-wcr-jenkins.ethz.ch/job/climada_branches/>`_.

   * Integration tests are located in ``climada/test/``. They test end-to-end methods and
     functions. Their execution time can be of minutes. They are executed once a day in 
     `Jenkins <http://ied-wcr-jenkins.ethz.ch/job/climada_ci_night/>`_.

6. Make sure your changes are not introducing new test failures.

   Run unit and integration tests::
   
    make unit_test
    make integ_test

   Compare the result to the results before the change. Current test failures are visible on 
   `Jenkins <http://ied-wcr-jenkins.ethz.ch/>`_.
   Fix new test failures before you create a pull request or push to the develop branch of 
   CLIMADA-project/climada_python. See `Continuous Integration`_ below.

7. Perform a **static code analysis** of your code using ``pylint`` with CLIMADA's configuration 
   ``.pylintrc``. `Jenkins <http://ied-wcr-jenkins.ethz.ch>`_ executes it after every push.
   To do it locally, you might use the Interface provided by `Spyder`.
   To do so, search first for `static code analysis` in `View` and then `Panes`.

8. Add new **data dependencies** used in :doc:`data_dependencies` and write a **tutorial** if a new
   class has been introduced (see :doc:`tutorial`).

9. Add your name to the **AUTHORS** file.

10. **Push** the code or branch to GitHub. To push to the develop do so::

    git push origin develop

 To push to your branch ``feature_branch_name`` do::

    git push origin feature_branch_name

11. Create a pull request.

   When the branch is ready, create a new **pull request** from the feature branch. `About pull
   requests <https://help.github.com/en/articles/about-pull-requests>`_.


Notes
-----

Update CLIMADA's environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Remember to regularly update your code as well as climada's environment. You might use the
following commands to update the environments::

    cd climada_python
    git pull
    source activate climada_env
    conda env update --file requirements/env_climada.yml
    conda env update --file requirements/env_developer.yml

If any problem occurs during this process, consider reinstalling everything from scratch following
the :doc:install instructions. 
You can find more information about virtual environments with conda 
`here <https://conda.io/docs/user-guide/tasks/manage-environments.html>`_.


Continuous Integration
======================
The results from the Jenkins server are to be taken seriously. 
Please run unit tests localy on the whole project, by calling `make unit_test` and if possible 
remotely on Jenkins in a feature branch.

Before pushing to the develop branch they should run without errors or (novel) failures.
After pushing, check the CI results on Jenkins, if the commit causes an error there, revert it 
immediately.
If the commit merely introduces novel failures, fix them within 3 days, or revert the commit.

Similar rules apply for the Pylint results on the deveolp branch. Novel high priority warnings
are not acceptable on the develop branch.
Novel medium priority warnings should be fixed within 3 days.

Tolerance overview
------------------

======= ===== ======= ==== ====== ===
Branch  Unittest          Linter
------- ------------- ---------------
\       Error Failure High Medium Low
======= ===== ======= ==== ====== ===
Master  x     x       x    \(x\)  \-
Develop x     3 days  x    3 days \-
Feature \(x\) \-      \-   \-     \-
======= ===== ======= ==== ====== ===

x indicates "no tolerance", meaning that any code changes producing such offences should be 
fixed *before* pushing them
to the respective branch.


Issues
======
Issues are the main platform for discussing matters. Use them extensively! Each issue should 
have one categoric label:

- bug
- enhancement
- question
- incident

and optionally others. When closing issues they should get another label for the closing reason:

- fixed
- wontfix
- duplicate
- invalid

(Despite their names, `fixed` and `wontfix` are applicable for questions and enhancements as well.)


Code Review
===========
Ask for a code review before merging a feature branch back into the develop branch.
A list of potential reviewers can be found in the `WIKI <https://github.com/CLIMADA-project/climada_python/wiki/Developer-Board>`_.


Regular Releases
================
Regular releases are planned on a quarterly base. Upcoming releases are listed in the `WIKI <https://github.com/CLIMADA-project/climada_python/wiki/Upcoming-Releases>`_.

