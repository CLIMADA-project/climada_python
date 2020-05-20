"""A setuptools based setup module.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path
import os

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the data recursively from the data folder
def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if filename != '.DS_Store':
                paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files(here + '/data/')
# Add configuration files
extra_files.append(here + '/climada/conf/defaults.conf')

setup(
    name='climada',

    version='1.4.0',

    description='CLIMADA in Python',

    long_description=long_description,

    url='https://github.com/davidnbresch/climada_python',

    author='ETH',

    license='GNU General Public License',

    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Topic :: Climate Adaptation',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='climate adaptation',

    packages=find_packages(where='.'),

    install_requires=[
        'cartopy==0.17.0', # conda!
        'cloudpickle', # install_test
        'contextily==1.0rc2',
        'dask==1.2.2',
        'descartes',
        #'earthengine_api==0.1.210', # ee, conda!
        'elevation==1.0.6',
        'fiona==1.8.4',
        'fsspec>=0.3.6', # < dask
        'gdal==2.3.3', # conda!
        'geopandas==0.4.1',
        'h5py==2.9.0',
        'haversine==2.1.1',
        'iso3166==1.0',
        #'kealib==1.4.7', < fiona
        'matplotlib==3.1', #
        'mercantile',
        #'mpl_toolkits', matplotlib
        'netCDF4==1.4.2', # conda!
        'numba==0.43.1', # conda!
        'numpy==1.16.3', # conda+
        'overpy==0.4',
        'pandas==0.24.2',
        'pandas_datareader==0.7.0',
        'pathos==0.2.3',
        'pillow==6.2.2', # PIL 7.0 has a conflict with libtiff 4.0 which is necessary for - at least - Windows
        'pint==0.9',
        #'pylab', matplotlib
        'pyproj==1.9.6', #
        'pyshp', # shapefile
        'rasterio==1.0.21',
        'requests==2.21.0', #
        'rtree==0.8.3', # < geopandas.overlay
        'scikit-learn==0.20.3', # sklearn
        'scipy==1.2.1', # conda+
        'shapely==1.6.4', #
        'six==1.13.0', #
        'tables', # < pandas (climada.entity.measures.test.test_base.TestApply)
        'tabulate==0.8.3',
        'toolz', # < dask
        'tqdm==4.31.1',
        'xarray==0.12.1',
        'xlrd', # < pandas
        'xlsxwriter==1.1.7',
        'xmlrunner==1.7.7', # ci tests
    ],

    package_data={'': extra_files},

    include_package_data=True
)
