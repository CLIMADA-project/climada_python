"""A setuptools based setup module.
"""

from pathlib import Path

from setuptools import find_namespace_packages, setup

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here.joinpath("README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Requirements for documentation
DEPS_DOC = [
    "ipython",
    "myst-nb",
    "readthedocs-sphinx-ext>=2.2",
    "sphinx",
    "sphinx-book-theme",
    "sphinx-markdown-tables",
    "sphinx-mdinclude",
]

# Requirements for testing
DEPS_TEST = [
    "ipython",
    "mccabe>=0.6",
    "pylint>=3.0",
    "pytest",
    "pytest-cov",
    "pytest-subtests",
]

# Requirements for development
DEPS_DEV = (
    DEPS_DOC
    + DEPS_TEST
    + [
        "pre-commit",
    ]
)

setup(
    name="climada",
    version="6.0.1",
    description="CLIMADA in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CLIMADA-project/climada_python",
    author="ETH",
    author_email="schmide@ethz.ch",
    license="OSI Approved :: GNU Lesser General Public License v3 (GPLv3)",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="climate adaptation",
    python_requires=">=3.10,<3.12",
    install_requires=[
        "bayesian-optimization<2.0",
        "bottleneck",
        "cartopy",
        "cfgrib",
        "contextily",
        "dask",
        "deprecation",
        "geopandas",
        "h5py",
        "haversine",
        "matplotlib",
        "netcdf4",
        "numba",
        "openpyxl",
        "overpy",
        "pandas",
        "pandas-datareader",
        "pathos",
        "peewee",
        "pillow",
        "pint",
        "pycountry",
        "pyproj",
        "rasterio",
        "salib",
        "scikit-learn",
        "seaborn",
        "statsmodels",
        "sparse",
        "tables",
        "tabulate",
        "tqdm",
        "xarray",
        "xlrd",
        "xlsxwriter",
    ],
    extras_require={
        "doc": DEPS_DOC,
        "test": DEPS_TEST,
        "dev": DEPS_DEV,
    },
    packages=find_namespace_packages(include=["climada*"]),
    setup_requires=["setuptools_scm"],
    include_package_data=True,
)
