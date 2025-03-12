# CLIMADA Contribution Guide

We welcome any contribution to CLIMADA and want to express our thanks to everybody who contributes.

## What Warrants a Contribution?

Anything!
For orientation, these are some categories of possible contributions we can think of:

* **Technical problems and bugs:** Did you encounter a problem when using CLIMADA? Raise an [issue](https://github.com/CLIMADA-project/climada_python/issues) in our repository, providing a description or ideally a code replicating the error. Did you already find a solution to the problem? Please raise a pull request to help us resolve the issue!
* **Documentation and Tutorial Updates:** Found a typo in the documentation? Is a tutorial lacking some information you find important? Simply fix a line, or add a paragraph. We are happy to incorporate you additions! Please raise a pull request!
* **New Modules and Utility Functions:** Did you create a function or an entire module you find useful for your work? Maybe you are not the only one! Feel free to simply raise a pull request for functions that improve, e.g., plotting or data handling. As an entire module has to be carefully integrated into the framework, it might help if you talk to us first so we can design the module and plan the next steps. You can do that by raising an issue or starting a [discussion](https://github.com/CLIMADA-project/climada_python/discussions) on GitHub.

A good place to start a personal discussion is our monthly CLIMADA developers call.
Please contact the [lead developers](https://wcr.ethz.ch/research/climada.html) if you want to join.

## Why Should You Contribute?

* You will be listed as author of the CLIMADA repository in the [AUTHORS](AUTHORS.md) file.
* You will improve the quality of the CLIMADA software for you and for everybody else using it.
* You will gain insights into scientific software development.

## Minimal Steps to Contribute

Before you start, please have a look at our Developer Guide section in the [CLIMADA Docs][docs].

To contribute follow these steps:

1. Install CLIMADA following the [installation instructions for developers](https://climada-python.readthedocs.io/en/latest/guide/install.html#advanced-instructions).
2. In the CLIMADA repository, create a new feature branch from the latest `develop` branch:

    ```bash
    git checkout develop && git pull
    git checkout -b feature/my-fancy-branch
    ```
3. Implement your changes and commit them with [meaningful and well formatted](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html) commit messages.
4. Add [unit and integration tests][testing] to your code, if applicable.
5. Use [Pylint](https://pypi.org/project/pylint/) for a static code analysis of your code with CLIMADA's configuration `.pylintrc`:

    ```bash
    pylint
    ```
6. Add your name to the [AUTHORS](AUTHORS.md) file.
7. Push your updates to the remote repository:

    ```bash
    git push --set-upstream origin feature/my-fancy-branch
    ```

    **NOTE:** Only team members are allowed to push to the original repository.
    Most contributors are/will be team members. To be added to the team list and get permissions please contact one of the [owners](https://github.com/orgs/CLIMADA-project/people).
    Alternatively, you can [fork the CLIMADA repository](https://github.com/CLIMADA-project/climada_python/fork) and add this fork as a new [remote](https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes) to your local repository.
    You can then push to the fork remote:

    ```bash
    git remote add fork <your-fork-url>
    git push --set-upstream fork feature/my-fancy-branch
    ```

8. On the [CLIMADA-project/climada_python](https://github.com/CLIMADA-project/climada_python) GitHub repository, create a new pull request with target branch `develop`.
    This also works if you pushed to a fork instead of the main repository.
    Add a description and explanation of your changes and work through the pull request author checklist provided.
    Feel free to request reviews from specific team members.
9.  After approval of the pull request, the branch is merged into `develop` and your changes will become part of the next CLIMADA release.

## Resources

The [CLIMADA documentation][docs] provides several Developer Guides.
Here's a selection of the commonly required information:

* How to use Git and GitHub for CLIMADA development: [Development and Git and CLIMADA](https://climada-python.readthedocs.io/en/latest/guide/Guide_Git_Development.html)
* Coding instructions for CLIMADA: [Python Dos and Don'ts](https://climada-python.readthedocs.io/en/latest/guide/Guide_PythonDos-n-Donts.html), [Performance Tips](https://climada-python.readthedocs.io/en/latest/guide/Guide_Py_Performance.html), [CLIMADA Conventions](https://climada-python.readthedocs.io/en/latest/guide/Guide_CLIMADA_conventions.html)
* How to execute tests in CLIMADA: [Testing][testing] and [Continuous Integration](https://climada-python.readthedocs.io/en/latest/guide/Guide_continuous_integration_GitHub_actions.html)

## Pull Requests

After developing a new feature, fixing a bug, or updating the tutorials, you can create a [pull request](https://docs.github.com/en/pull-requests) to have your changes reviewed and then merged into the CLIMADA code base.
To ensure that your pull request can be reviewed quickly and easily, please have a look at the _Resources_ above before opening a pull request.
In particular, please check out the [Pull Request instructions](https://climada-python.readthedocs.io/en/latest/guide/Guide_Git_Development.html#pull-requests).

We provide a description template for pull requests that helps you provide the essential information for reviewers.
It also contains a checklist for both pull request authors and reviewers to guide the review process.

[docs]: https://climada-python.readthedocs.io/en/latest/
[devguide]: https://climada-python.readthedocs.io/en/latest/#developer-guide
[testing]: https://climada-python.readthedocs.io/en/latest/guide/Guide_Testing.html
