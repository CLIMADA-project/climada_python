"""A setuptools based setup module.
"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='climada',

    version='0.0.1',

    description='CLIMADA in Python',

    long_description=long_description,
    
    url='https://github.com/davidnbresch/climada_python',

    author='ETH',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[ 
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Topic :: Climate Adaptation',
        'License :: GNU General Public License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='climate adaptation',  

    packages=find_packages(where='.'), 

    install_requires=['numpy',
                      'matplotlib',
                      'pandas',
                      'h5py',
                      'scipy',
                      'scikit-learn',
                      'xlrd',
                      'xmlrunner',
                      'coverage',
                      'pylint',
                      'spyder'
                     ], 

    #package_data={ 'climada': ['data/*'],
    #},

    data_files=[('climada', ['data/*'])],  
)
