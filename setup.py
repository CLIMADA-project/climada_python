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

extra_files = package_files(here + '/climada/test/data/')
# Add configuration files
extra_files.append(here + '/climada/conf/defaults.conf')
extra_files.append(here + '/climada/conf/logging.conf')

setup(
    name='climada',

    version='0.0.1',

    description='CLIMADA in Python',

    long_description=long_description,
    
    url='https://github.com/davidnbresch/climada_python',

    author='ETH',

    license='GNU General Public License',

    classifiers=[ 
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Topic :: Climate Adaptation',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='climate adaptation',  

    packages=find_packages(where='.'), 

    install_requires=['cartopy==0.16.0',
                      'shapely==1.6.2',
                      'cython',
                      'xarray',
                      'matplotlib',
                      'pandas',
                      'h5py',
                      'scipy',
                      'scikit-learn',
                      'xlrd',
                      'pint',
                      'numpy',
                      'numba',
                      'pathos',
                      'xmlrunner'
                     ], 

    package_data={'': extra_files },  
)
