# CLIMADA Documentation

The CLIMADA documentation consists of ``.rst`` files and [Jupyter](https://jupyter.org/) Notebooks.
It is built into an HTML webpage by the [Sphinx](https://www.sphinx-doc.org/en/master/index.html) package.

The [online documentation](https://climada-python.readthedocs.io/en/stable/) is automatically built when the `main` or `develop` branch are updated.
Additionally, documentation previews will be built for every [pull request](https://github.com/CLIMADA-project/climada_python/pulls) on GitHub, and will be displayed under "Checks".

Note that the online documentation allows you to switch versions.
By default, you will see the `stable` version, which refers to the latest release.
You can switch to `latest`, which refers to the latest version of the `develop` branch.

## Local Build

You can also build and browse the documentation on your machine.
This can be useful if you want to access the documentation of a particular feature branch or to check your updates to the documentation.

For building the documentation, you need to follow the [advanced installation instructions](https://climada-python.readthedocs.io/en/latest/guide/install.html#advanced-instructions).
Make sure to install the developer requirements as well.

Then, activate the `climada_env` and navigate to the `doc` directory:
```
conda activate climada_env
cd climada_python/doc
```

Next, execute `make` (this might take a while when executed for the first time)
```
make html
```

The documentation will be placed in `doc/_build/html`. Simply open the page `doc/_build/html/index.html` with your browser.

## Updating the Documentation Environment for Readthedocs.org

The online documentation is built by [`readthedocs.org`](https://readthedocs.org/).
Their servers have a limited capacity.
In the past, this capacity was exceeded by Anaconda when it tried to resolve all dependencies for CLIMADA.
We therefore provided a dedicated environment with *fixed* package versions in `requirements/env_docs.yml`.
As of commit `8c66d8e4a4c93225e3a337d8ad69ab09b48278e3`, this environment was removed and the online documentation environment is built using the specs in `requirements/env_climada.yml`.
If this should fail in the future, revert the changes by `8c66d8e4a4c93225e3a337d8ad69ab09b48278e3` and update the environment specs in `requirements/env_docs.yml` with the following instructions.

For re-creating the documentation environment, we provide a Dockerfile.
You can use it to build a new environment and extract the exact versions from it.
This might be necessary when we upgrade to a new version of Python, or when dependencies are updated.
**NOTE:** Your machine must be able to run/virtualize an AMD64 OS.

Follow these instructions:

1. [Install Docker](https://docs.docker.com/get-docker/) on your machine.
2. Enter the top-level directory of the CLIMADA repository with your shell:

    ```
    cd climada_python
    ```
3. Instruct Docker to build an image from the `doc/create_env_doc.dockerfile`:

    ```
    docker build -f doc/create_env_doc.dockerfile -t climada_env_doc ./
    ```
4. Run a container from this image:

    ```
    docker run -it climada_env_doc
    ```
5. You have now entered the container.
   Activate the conda environment and export its specs:

    ```
    conda activate climada_doc
    conda env export
    ```
    Copy and paste the shell output of the last command into the `requirements/env_docs.yml` file in the CLIMADA repository, overwriting all its contents.
