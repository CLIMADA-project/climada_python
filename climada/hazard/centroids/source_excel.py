"""
Define Centroids reader function from an Excel file.
"""

import pandas
import numpy as np

from climada.entity.tag import Tag

# Define tha name of the sheet that is read
SHEET_NAME = 'centroids'
# Define the names of the columns that are read
COL_NAME = {'cen_id' : 'centroid_ID',
            'lat' : 'Latitude',
            'lon' : 'Longitude'
           }

def read(centroids, file_name, description=None):
    """Read excel file and store variables in centroids. """
    dfr = pandas.read_excel(file_name, SHEET_NAME)

    centroids.tag = Tag(file_name, description)
    coord_cols = [COL_NAME['lat'], COL_NAME['lon']]

    centroids.coord = np.array(dfr[coord_cols])
    centroids.id = dfr[COL_NAME['cen_id']].values
