"""A setuptools based setup module.
"""

from pathlib import Path
from setuptools import setup, find_namespace_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Requirements for documentation
DEPS_DOC = [
    "ipython",
    "myst-nb",
    "readthedocs-sphinx-ext>=2.2",
    "sphinx",
    "sphinx-book-theme",
    "sphinx-markdown-tables",
]

# Requirements for testing
DEPS_TEST = [
    "ipython",
    "mccabe>=0.6",
    "pylint==2.7.1",
    "pytest",
    "pytest-cov",
    "pytest-subtests",
]

setup(
    name='climada',

    version='4.0.1',

    description='CLIMADA in Python',

    long_description=long_description,
    long_description_content_type="text/markdown",

    url='https://github.com/CLIMADA-project/climada_python',

    author='ETH',
    author_email='schmide@ethz.ch',

    license='OSI Approved :: GNU Lesser General Public License v3 (GPLv3)',

    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],

    keywords='climate adaptation',

    python_requires=">=3.9,<3.12",

    install_requires=[
        'bottleneck',
        'cartopy',
        'cfgrib',
        'contextily',
        'dask',
        'deprecation',
        'geopandas',
        'h5py',
        'haversine',
        'matplotlib',
        'netcdf4',
        'numba',
        'openpyxl',
        'overpy',
        'pandas',
        'pandas-datareader',
        'pathos',
        'peewee',
        'pillow',
        'pint',
        'pycountry',
        'rasterio',
        'salib',
        'scikit-learn',
        'statsmodels',
        'sparse',
        'tables',
        'tabulate',
        'tqdm',
        'xarray',
        'xlrd',
        'xlsxwriter',
        'xmlrunner'
    ],

    extras_require={
        "doc": DEPS_DOC,
        "test": DEPS_TEST,
        "dev": DEPS_DOC + DEPS_TEST
    },

    packages=find_namespace_packages(include=['climada*']),

    setup_requires=['setuptools_scm'],
    include_package_data=True,
)
