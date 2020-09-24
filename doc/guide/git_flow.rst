.. _Git Flow:

Git Flow
========

In general, our policy and naming of branches follow broadly the conventions of 
`git flow <https://jeffkreeftmeijer.com/git-flow/>`_.

Philosophy
----------

We use a *merge only* philosophy in all shared branches. This is safer
when working with people with various levels of git-skills.

We use rather long-lived feature branches, as we are scientist and not
software engineers. The creator of a feature branch is the
owner/responsible for its content. You can use the workflow you prefer
inside of your feature branch. It must be regularly merged into the
develop branch, and at this point the branch owner must organize a code
review. This allows for a smooth development of both the code and the
science, without compromising the code legacy.

Scientific publication should, once accepted, be made into a minimal 
working example and pushed onto the Climada_papers repository.

Do not forget to update the Jenkins test and the CLIMADA tutorial.

Release cycle
-------------

When a new release is made, everything in the develop branch is merged
into the master branch.

Fork or clone?
--------------

Core developers should clone the project.

External developers can fork the project. If you want to become a core
developer, please contact us.

Branches
--------

The branching system is adapted for the scientific setting from

+-------------+--------------------+------------+---------------+
| Branches    | Purpose            | Code       | Longevity     |
+=============+====================+============+===============+
| Main      | Releases           | Stable     | Infinite      |
+-------------+--------------------+------------+---------------+
| Develop     | Between releases   | Reviewed   | Infinite      |
+-------------+--------------------+------------+---------------+
| Feature/    | Scientific Dev.    | Anything   | Max. 1 year   |
+-------------+--------------------+------------+---------------+
| Feature/Fix-| Minor changes      | Anything   | few days      |
+-------------+--------------------+------------+---------------+

-  The ``master`` branch is used for the quarterly releases of stable code.

-  The ``develop`` branch is used to gather working/tested code in between
   releases.

-  Development is done in feature branches.

-  Feature branches are used for any CLIMADA development. Note that a
   feature branch should not live forever and should contribute to the
   ``develop`` branch. Hence, try to merge your working code for each
   release, or every second release at least. Ideally, a branch is
   cleaned after max. one year and then archived.
   
-  Small fixes, bugs, code updates, improvements etc. are done as feature branches and should use a clear prefix such as 'Fix-'. These branches are thought of to be short lived, but still require a review! After the review, these branches are usually deleted.
 
-  Code for a scientific project/paper (i.e. CLIMADA application, not
   development) is not pushed to this repository. A minimal working
   example should be pushed to the climada\_papers repository.

What files to commit?
---------------------

-  Git is for the code, *not* the data. The only exceptions are small
   data samples for the unit tests. Small means (<1mb). A very very
   strong reason must be given to commit larger files. A more systematic
   way to handle data will be developed soon.

.gitignore
----------

See
`here <https://www.atlassian.com/git/tutorials/saving-changes/gitignore>`__\ for
details on how to use the .gitignore file.

-  If your script (not a paper script, but a core CLIMADA script)
   produces files, add these to the gitignore file so that they are not
   committed. Add .gitignore to a commit. Do only add your file.

-  Remember to remove a file from the gitignore if it is not produced by
   the code anymore.

Don'ts
------

-  Do not rebase on the ``develop``/``master`` branches, or on any branch that has already been pushed to GitHub.
-  Do not use fast-forwarding on the remote branches.

Creating feature branches
-------------------------

Before/After starting a new scientific feature branch, present a 2-3 mins plan at the
bi-weekly developers meeting.

Naming convention: feature/my\_feature\_name

For small features (fixes, improvements,...), use a clear prefix such as 'Fix-'.

Naming convention: feature/Fix-my\_feature\_name

How to use GIT for CLIMADA
==========================

How to use Git?
---------------

Please check the `Git
book <https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control>`__
tutorial to get a basic understanding of git.

Recommended reading (to begin with):

-  Chapter 1 Getting started
-  Chapter 2 Git Basics
-  Chapter 3 Git Branching,
-  Chapter 6 GitHub

Also checkout this
`cheatsheet <https://www.atlassian.com/git/tutorials/atlassian-git-cheatsheet>`__
for git commands.

GUI or command line
-------------------

-  The probably most complete way to use git is through the command
   line. However, this is not very visual, in particular at the
   beginning. Consider using a GUI program such as “git desktop” or
   “Gitkraken”. Your python IDE is also likely to have a visual git interface.

-  Consider using an external merging and conflict resolution tool

Commit messages
---------------

Basic syntax guidelines taken from
`here <https://chris.beams.io/posts/git-commit/>`__ (on 17.06.2020)

-  Limit the subject line to 50 characters
-  Capitalize the subject line
-  Do not end the subject line with a period
-  Use the imperative mood in the subject line (e.g. "Add new tests")
-  Wrap the body at 72 characters (most editors will do this automatically)
-  Use the body to explain what and why vs. how
-  Separate the subject from body with a blank line (This is best done with
   a GUI. With the command line you have to use text editor, you cannot
   do it directly with the git command)
-  Put the name of the function/class/module/file that was edited
-  When fixing an issue, add the reference gh-ISSUENUMBER to the commit message 
   e.g. “fixes gh-40.” or “Closes gh-40.” For more infos see `here <https://docs.github.com/en/enterprise/2.16/user/github/managing-your-work-on-github/closing-issues-using-keywords#about-issue-references>`__.

Git commands for CLIMADA
------------------------

Below should be all the commands you need to get started for working on
a feature branch (assuming it already exists). More features are
available in git, and feel free to use them (e.g. stashing or cherry
picking). However, you should follow the dont's (do not rebase *on* the
develop branch, and do not fast-foward on remote branches).

A) Regular / daily commits locally

0. ``git fetch --all`` (make your local git know the changes that
   happened on the repository)
1. ``git checkout feature/feature_name`` (be sure to be on your branch)
2. ``git status``
3. ``git add file1``
4. ``git commit -m “Remove function xyz from feature.py”``
5. ``git status`` (verify that there are no more tracked files to be committed)

B) Push to remote branch (at least once/week, ideally daily)

1. ``git fetch --all``
2. ``git checkout feature/feature_name`` (be sure to be on your branch)
3. Make all commits according to A
4. ``git status`` (check whether your local branch is behind the remote)
5. ``git pull --rebase`` (`resolve all conflicts <https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts>`_ if there are any)
6. ``git push -u origin feature/feature_name`` if this is the first time you're pushing to the remote repository. Or just ``git push`` if the branch already exists there.

C) Merge develop into your branch (regularly/when develop changes)

1. ``git fetch --all``
2. Make all commit according to A
3. ``git status`` (verify that there are no tracked files that are
   uncommitted)
4. ``git checkout develop``
5. ``git pull --rebase``
6. ``git checkout feature/feature_name``
7. ``git merge --no-ff develop``
8. resolve all conflicts if there are any
9. ``git push origin feature/feature_name`` if this is the first time you're pushing to the remote repository. Or just ``git push`` if the branch already exists there.

D) Prepare to merge into develop (ideally before every release)

1.  ``git fetch --all``
2.  ``git checkout feature/feature_name``
3.  ``git status`` (see how many commits the branch is behind the
    remote)
4.  Make all commits according to A
5.  Merge develop into your branch according to C
6.  Push the branch to GitHub. If parts of the feature are incomplete and not everything is ready to go into ``develop``, create a new branch
    ``feature/feature_name-release`` with

    -  ``git checkout feature/feature_name-release``
    -  Clean the code so that only changes to be pushed remain
    -  Check that the code on the new branch passes unit and integration testing.
    -  Commit all changes according to A)
    -  ``git push -u origin feature/feature_name-release``

7.  Make a pull request
8.  Find someone to do a code review on ``feature/feature\_name-release``.
    Implement the code review suggestions (once done, redo steps 4 - 6))

