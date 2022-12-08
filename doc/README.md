# CLIMADA Documentation

The CLIMADA documentation consists of ``.rst`` files and [Jupyter](https://jupyter.org/) Notebooks.
It is built into an HTML webpage by the [Sphinx](https://www.sphinx-doc.org/en/master/index.html) package.

The online documentation is automatically built when the `main` or `develop` branch is updated.
It is located here: https://climada-python.readthedocs.io/en/stable/

Note that the online documentation allows you to switch versions.
By default, you will see the `stable` version, which refers to the latest release.
You can switch to `latest`, which refers to the latest version of the `develop` branch.

## Local Build

You can also build and browse the documentation on your machine.
This can be useful if you want to access the documentation of a particular feature branch or to check your updates to the documentation.

For building the documentation, you need to follow the [installation instructions for developers](https://climada-python.readthedocs.io/en/latest/guide/Guide_Installation.html#Install-CLIMADA-from-sources-(for-developers)).
Make sure to install the developer requirements as well.

Then, activate the `climada_env` and navigate to the `doc` directory:
```bash
conda activate climada_env
cd climada_python/doc
```

Next, execute `make` (this might take a while when executed for the first time)
```
make html
```

The documentation will be placed in `doc/_html`. Simply open the page `doc/_html/index.html` with your browser.
