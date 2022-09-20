# CLIMADA Contribution Guide

Thank you for contributing to CLIMADA!

## Overview

Before you start, please have a look at our [Developer Guide][devguide].

To contribute follow these steps:

1. Install CLIMADA following the [installation instructions for developers](https://climada-python.readthedocs.io/en/stable/guide/Guide_Installation.html#Install-CLIMADA-from-sources-(for-developers)).
2. Additionally, install the packages for developers:

    ```
    $ conda update -n climada_env -f climada_python/requirements/env_developer.yml
    ```
3. In the CLIMADA repository, create a new feature branch from the latest `develop` branch:

    ```
    $ git checkout develop && git pull
    $ git checkout -b feature/my-fancy-branch
    ```
4. Implement your changes and commit them with [meaningful and well formatted](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) commit messages.
5. Add [unit and integration tests][testing] to your code, if applicable.
6. Use [Pylint](https://pypi.org/project/pylint/) for a static code analysis of your code with CLIMADA's configuration `.pylintrc`:

    ```
    $ pylint
    ```
7. Add your name to the [AUTHORS](/AUTHORS) file.
8. Push your updates to the remote repository:

    ```
    $ git push --set-upstream origin feature/my-fancy-branch
    ```

    **NOTE:** Only team members are allowed to push to the original repository.
    Most contributors are/will be team members. To be added to the team list and get permissions please contact one of the [owners](https://github.com/orgs/CLIMADA-project/people).
    Alternatively, you can [fork the CLIMADA repository](https://github.com/CLIMADA-project/climada_python/fork) and add this fork as a new [remote](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) to your local repository.
    You can then push to the fork remote:

    ```
    $ git remote add fork <your-fork-url>
    $ git push --set-upstream fork feature/my-fancy-branch
    ```

9. On the [CLIMADA-project/climada_python](https://github.com/CLIMADA-project/climada_python) GitHub repository, create a new pull request with target branch `develop`.
    This also works if you pushed to a fork instead of the main repository.
    Add a description and explanation of your changes and work through the pull request author checklist provided.
    Feel free to request reviews from specific team members.
10. After approval of the pull request, the branch is merged into `develop` and your changes will become part of the next CLIMADA release.

## Resources

The CLIMADA documentation provides a [Developer Guide][devguide].
Here's a selection of the commonly required information:

* How to use Git and GitHub for CLIMADA development: [Development and Git and CLIMADA](https://climada-python.readthedocs.io/en/latest/guide/Guide_Git_Development.html)
* Coding instructions for CLIMADA: [Python Dos and Don'ts](https://climada-python.readthedocs.io/en/latest/guide/Guide_PythonDos-n-Donts.html), [Performance Tips](https://climada-python.readthedocs.io/en/latest/guide/Guide_Py_Performance.html), [CLIMADA Conventions](https://climada-python.readthedocs.io/en/latest/guide/Guide_Miscellaneous.html)
* How to execute tests in CLIMADA: [Testing and Continuous Integration][testing]

## Pull Requests

After developing a new feature, fixing a bug, or updating the tutorials, you can create a [pull request](https://docs.github.com/en/pull-requests) to have your changes reviewed and then merged into the CLIMADA code base.
To ensure that your pull request can be reviewed quickly and easily, please have a look at the [Resources](#resources) above before opening a pull request.
In particular, please check out the [Pull Request instructions](https://climada-python.readthedocs.io/en/latest/guide/Guide_Git_Development.html#Pull-requests).

We provide a description template for pull requests that helps you provide the essential information for reviewers.
It also contains a checklist for both pull request authors and reviewers to guide the review process.

[devguide]: https://climada-python.readthedocs.io/en/latest/#developer-guide
[testing]: https://climada-python.readthedocs.io/en/latest/guide/Guide_Continuous_Integration_and_Testing.html
