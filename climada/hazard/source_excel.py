"""
Define Hazard reader function from an excel file.
"""

import pandas
from scipy import sparse
import numpy as np

from climada.hazard.centroids.base import Centroids
from climada.hazard.tag import Tag as TagHazard

# Define tha name of the sheet that is read.
# Name of the table columns for each of the attributes.
# Define tha names of the columns describing the centroids.
# Used only when the centroids are not provided and have to be read
# from the same file as the hazard
DEF_VAR_NAME = {'sheet_name': {'centroid' : 'centroids',
                               'inten' : 'hazard_intensity',
                               'freq' : 'hazard_frequency'
                              },
                'col_name': {'cen_id' : 'centroid_ID',
                             'even_id' : 'event_ID',
                             'freq' : 'frequency'
                            },
                'col_centroids': {'cen_id' : 'centroid_ID',
                                  'lat' : 'Latitude',
                                  'lon' : 'Longitude'
                                 }
               }

def read(hazard, file_name, haztype, description='', centroids=None, \
         var_names=None):
    """Read excel file and store variables in hazard. """
    # set variable names in source file
    if var_names is None:
        var_names = DEF_VAR_NAME

    # File name and description into the instance class.
    hazard.tag = TagHazard(file_name, haztype, description)

    # Set the centroids if given, otherwise load them from the same file
    read_centroids(hazard, centroids, var_names)

    # number of centroids
    num_cen = len(hazard.centroids.id)

    # Load hazard frequency
    dfr = pandas.read_excel(file_name, var_names['sheet_name']['freq'])
    # number of events
    num_events = dfr.shape[0]
    hazard.frequency = dfr[var_names['col_name']['freq']].values
    hazard.event_id = dfr[var_names['col_name']['even_id']].values

    # Load hazard intensity
    dfr = pandas.read_excel(file_name, var_names['sheet_name']['inten'])
    hazard.event_name = dfr.keys().values[1:].tolist()
    # number of events (ignore centroid_ID column)
    # check the number of events is the same as the one in the frequency
    if dfr.shape[1] - 1 is not num_events:
        raise ValueError('Hazard intensity is given for a number of \
              events different from the number of defined in its \
              frequency: ', dfr.shape[1] - 1, ' != ', num_events)
    # check number of centroids is the same as retrieved before
    if dfr.shape[0] is not num_cen:
        raise ValueError( \
                'Hazard intensity is given for a number of centroids \
                different from the number of centroids defined: %s != %s'\
                % (str(dfr.shape[0]), str(num_cen)))
    # check centroids ids are correct
    if not np.array_equal(dfr[var_names['col_name']['cen_id']].values,
                          hazard.centroids.id[-num_cen:]):
        raise ValueError('Hazard intensity centroids ids do not match \
                         previously defined centroids.')

    hazard.intensity = dfr.values[:, 1:num_events+1].transpose()
    # make the intensity a sparse matrix
    hazard.intensity = sparse.csr_matrix(hazard.intensity)

    # Set fraction matrix to default value of 1
    hazard.fraction = sparse.csr_matrix(np.ones(hazard.intensity.shape, \
                                      dtype=np.float))

def read_centroids(hazard, centroids, var_names):
    """Read centroids file if no centroids provided"""
    if centroids is None:
        hazard.centroids = Centroids()
        hazard.centroids.sheet_name = var_names['sheet_name']['centroid']
        hazard.centroids.col_names = var_names['col_centroids']
        hazard.centroids.read_one(hazard.tag.file_name, hazard.tag.description)
    else:
        hazard.centroids = centroids
