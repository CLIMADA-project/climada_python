"""A setuptools based setup module.
"""

from pathlib import Path
from setuptools import setup, find_packages

here = Path(__file__).parent.absolute()

# Get the long description from the README file
with open(here.joinpath('README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the data recursively from the data folder
def package_files(directory):
    return [str(path_object)
        for path_object in directory.glob('**/*')
        if path_object.is_file() and path_object.name[0] != '.'
    ]

extra_files = package_files(here / 'data')
# Add configuration files
extra_files.append(str(here / 'climada/conf/climada.conf'))

setup(
    name='climada',

    version='2.1.1',

    description='CLIMADA in Python',

    long_description=long_description,

    url='https://github.com/davidnbresch/climada_python',

    author='ETH',

    license='OSI Approved :: GNU General Public License v3 (GPLv3)',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Topic :: Climate Adaptation',
        'Programming Language :: Python :: 3.8',
    ],

    keywords='climate adaptation',

    packages=find_packages(where='.'),

    install_requires=[
        'bottleneck',
        'cartopy',
        'cfgrib',
        'contextily',
        'dask',
        'geopandas',
        'h5py',
        'haversine',
        'iso3166',
        'matplotlib',
        'netcdf4',
        'numba',
        'overpy',
        'pandas',
        'pandas-datareader',
        'pathos',
        'pillow',
        'pint',
        'pybufrkit',
        'pycountry',
        'rasterio',
        'scikit-learn',
        'statsmodels',
        'tables',
        'tabulate',
        'tqdm',
        'xarray',
        'xlrd',
        'xlsxwriter',
        'xmlrunner'
    ],

    package_data={'': extra_files},

    include_package_data=True
)
