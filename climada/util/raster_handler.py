#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 19:30:51 2020

@author: evelynm
"""

import pyproj
import rasterio

def _get_window_from_coords(path_sourcefile, bbox=[]):
    """
    get row, column, width and height from coordinate values of bounding box
    required for rasterio window function or coords.set_raster_from_pix_bounds
    
    Parameters:
        bbox (list): [north, east, south, west]
        path_sourcefile (str): path of file from which window should be read in
    Returns:
        window_array (array): corner, width & height for Window() function of rasterio
    """
    with rasterio.open(path_sourcefile) as src:
        utm = pyproj.Proj(init='epsg:4326') # Pass CRS of image from rasterio
        lonlat = pyproj.Proj(init='epsg:4326')
        lon, lat = (bbox[3], bbox[0])
        west, north = pyproj.transform(lonlat, utm, lon, lat)

        row, col = src.index(west, north) # spatial --> image coordinates

        lon, lat = (bbox[1], bbox[2])
        east, south = pyproj.transform(lonlat, utm, lon, lat)
        row2, col2 = src.index(east, south)
    width = abs(col2-col)
    height = abs(row2-row)

    window_array = [col, row, width, height]

    return window_array
