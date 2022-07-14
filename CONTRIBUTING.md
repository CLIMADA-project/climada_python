# CLIMADA Contribution Guide

Thank you for contributing to CLIMADA!

## Overview

Before you start, please have a look at our [Developer Guide][devguide].

To contribute follow these steps:

1. Fork the project on GitHub.
2. Create a local clone of the develop branch (`git clone https://github.com/YOUR-USERNAME/climada_python.git -b develop`)
3. Install the packages in `climada_python/requirements/env_climada.yml` and `climada_python/requirements/env_developer.yml`.
4. Make well commented and clean commits to your repository.
5. Make unit and integration tests on your code, preferably during development.
6. Perform a static code analysis of your code with CLIMADA's configuration `.pylintrc`.
7. Add your name to the [AUTHORS](/AUTHORS) file.
8. Push the changes to GitHub (`git push origin develop`).
9. On GitHub, create a new pull request onto the develop branch of CLIMADA-project/climada_python.

## Resources

The CLIMADA documentation provides a [Developer Guide][devguide].
Here's a selection of the commonly required information:

* How to use Git and GitHub for CLIMADA development: [Development and Git and CLIMADA](https://climada-python.readthedocs.io/en/latest/guide/Guide_Git_Development.html)
* Coding instructions for CLIMADA: [Python Dos and Don'ts](https://climada-python.readthedocs.io/en/latest/guide/Guide_PythonDos-n-Donts.html), [Performance Tips](https://climada-python.readthedocs.io/en/latest/guide/Guide_Py_Performance.html), [CLIMADA Conventions](https://climada-python.readthedocs.io/en/latest/guide/Guide_Miscellaneous.html)
* How to execute tests in CLIMADA: [Testing and Continuous Integration](https://climada-python.readthedocs.io/en/latest/guide/Guide_Continuous_Integration_and_Testing.html)

## Pull Requests

After developing a new feature, fixing a bug, or updating the tutorials, you can create a [pull request](https://docs.github.com/en/pull-requests) to have your changes reviewed and then merged into the CLIMADA code base.
To ensure that your pull request can be reviewed quickly and easily, please have a look at the [Resources](#resources) above before opening a pull request.
In particular, please check out the [Pull Request instructions](https://climada-python.readthedocs.io/en/latest/guide/Guide_Git_Development.html#Pull-requests).

We provide a description template for pull requests that helps you provide the essential information for reviewers.
It also contains a checklist for both pull request authors and reviewers to guide the review process.

[devguide]: https://climada-python.readthedocs.io/en/latest/#developer-guide
