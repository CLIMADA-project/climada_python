.. _Contributing:

Contributing
============

Contributions are very welcome! Please follow these steps:

0. **Install** `Git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_
   and `Anaconda <https://www.anaconda.com/>`_ (or `Miniconda <https://conda.io/miniconda.html>`_).

   Also consider installing Git flow. This is included with `Git for Windows <https://gitforwindows.org/>`_, and has different implementations e.g. `here <https://skoch.github.io/Git-Workflow/>`_ for Windows and Mac

1. **Clone (or fork)** the project on GitHub

   From the location where you want to create the project folder, run in your terminal::

        git clone https://github.com/CLIMADA-project/climada_python.git

   For more information on the Git flow approach to development see :doc:`install`.

2. **Install the packages** in ``climada_python/requirements/env_climada.yml`` and
   ``climada_python/requirements/env_developer.yml`` (see :doc:`install`). You
   might need to install additional environments contained in ``climada_python/requirements``
   when using specific functionalities.

3. Make a new **branch**

   For new features in Git flow::

    git flow feature start feature_name
    
   Which is equivalent to (in vanilla git)::

    git checkout -b feature/feature_name

   Or work on an existing branch::

    git checkout -b branch_name

   See CLIMADA-python's branching policies in :doc:`git_flow`.
 
   `General information about Git branches <https://help.github.com/en/articles/about-branches>`_.

4. Follow the :doc:`coding_conventions`. Write small readable methods, classes and functions.
   Make well commented and clean **commits** to the repository::

    # get the latest data from the remote repository and update your branch
    git pull

    # see your locally modified files
    git status

    # add changes you want to include in the commit
    git add climada/modified_file.py climada/test/test_modified_file.py

    # commit the changes
    git commit -m "new functionality of .. implemented"

   Usually you will want a longer commit message than the one-line message above. In this case ``git commit`` will open your terminal's default text editor for a more detailed description. You can also create your commits interactively through your IDE's version control GUI (Spyder/PyCharm/etc).


5. Make unit and integration **tests** on your code, preferably during development:

   * Unit tests are located in the ``test`` folder located in same folder as the corresponding
     module. Unit tests should test all methods and functions using fake data if necessary.
     The whole test suite should run in less than 20 sec. They are all executed `after each push
     in Jenkins <http://ied-wcr-jenkins.ethz.ch/job/climada_branches/>`_.

   * Integration tests are located in ``climada/test/``. They test end-to-end methods and
     functions. Their execution time can be of minutes. They are executed `once a day in
     Jenkins <http://ied-wcr-jenkins.ethz.ch/job/climada_ci_night/>`_.

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

10. Merge any updates to ``develop`` into your branch.

   There may have been changes to the remote ``develop`` branch since you created your branch. You can deal with potential conflicts by updating and merging ``develop`` into your branch::

    git checkout develop
    git pull
    git checkout feature/feature_name
    git merge develop

   Then `resolve any conflicts <https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts>`_. In the case of more complex conflicts, you may want to speak with others who worked on the same code.

11. **Push** the branch to GitHub.

    To push your branch ``feature_branch_name`` for the first time call::

     git push -u origin feature/feature_branch_name

    or, if you're updating a branch that's already on GitHub::

      git push

    Only push small bugfixes and comments directly to ``develop`` - most new code should be pushed as a feature branch, which can then be reviewed with a pull request. Only emergency hotfixes are pushed to ``master``.

12. Create a pull request.

    When the branch is ready, create a new **pull request** from the feature branch. `About pull
    requests <https://help.github.com/en/articles/about-pull-requests>`_.

    To do this,

    - On the `CLIMADA GitHub page <https://github.com/CLIMADA-project/climada_python>`_, navigate to your feature branch. Above the list of files is a summary of the branch and an icon to the right labelled "Pull request".
    - Choose which branch you want to merge with. This will usually be ``develop``, but may be a feature branch for more complex feature development.
    - Give your pull request an informative title, like a commit message.
    - Write a description of the pull request. This can usually be adapted from your branch's commit messages, and should give a high-level summary of the changes, specific points you want the reviewers' input on, and possibly explanations for decisions you've made.
    - Assign reviewers using the right hand sidebar on the page. Tag anyone who might be interested in reading the code. You should have found someone who is happy to read the whole request and sign it off (this person could also be added to 'Assignees'). A list of potential reviewers can be found in the `WIKI <https://github.com/CLIMADA-project/climada_python/wiki/Developer-Board>`_.
    - Contact reviewers. GitHub's settings mean that they may not be alerted automatically, so send them a message.

13. Review and merge the pull request.

    For big pull requests, stay in touch with reviewers. When everyone has had the chance to make comments and suggestions, and at least one person has read and approved the whole request, it's ready to be merged.

    If ``develop`` has been updated during the review process, it may be necessary to resolve merge conflicts again.

    Merging the pull request is done through the GitHub site. Once it's merged you can delete the feature branch and update your local copy of ``develop`` with ``git pull``.



Update CLIMADA's environment
----------------------------
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
----------------------
The results from the Jenkins server are to be taken seriously. 
Please run unit tests locally on the whole project, by calling `make unit_test` and if possible
remotely on Jenkins in a feature branch.

Before pushing to the develop branch they should run without errors or (novel) failures.
After pushing, check the CI results on Jenkins, if the commit causes an error there, revert it 
immediately.
If the commit merely introduces novel failures, fix them within 3 days, or revert the commit.

Similar rules apply for the Pylint results on the deveolp branch. Novel high priority warnings
are not acceptable on the develop branch.
Novel medium priority warnings should be fixed within 3 days.

Tolerance overview
~~~~~~~~~~~~~~~~~~

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
------
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


Regular Releases
----------------
Regular releases are planned on a quarterly base. Upcoming releases are listed in the `WIKI <https://github.com/CLIMADA-project/climada_python/wiki/Upcoming-Releases>`_.

