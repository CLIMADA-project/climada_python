"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Regroup methods to obtain images from Google Earth Engine API
"""

import logging
import webbrowser

# This module works only if you have a Google Earth Engine account.
# That's why `earthengine-api` is not in the CLIMADA requirements.
# See tutorial: climada_util_earth_engine.ipynb
# pylint: disable=import-error
import ee

LOGGER = logging.getLogger(__name__)
ee.Initialize()


def obtain_image_landsat_composite(landsat_collection, time_range, area):
    """Selection of Landsat cloud-free composites in the Earth Engine library
    See also: https://developers.google.com/earth-engine/landsat

    Parameters
    ----------
    collection :
        name of the collection
    time_range : ['YYYY-MT-DY','YYYY-MT-DY']
        must be inside the available data
    area : ee.geometry.Geometry
        area of interest

    Returns
    -------
    image_composite : ee.image.Image
     """
    collection = ee.ImageCollection(landsat_collection)

    # Filter by time range and location
    collection_time = collection.filterDate(time_range[0], time_range[1])
    image_area = collection_time.filterBounds(area)
    image_composite = ee.Algorithms.Landsat.simpleComposite(image_area, 75, 3)
    return image_composite

def obtain_image_median(collection, time_range, area):
    """Selection of median from a collection of images in the Earth Engine library
    See also: https://developers.google.com/earth-engine/reducers_image_collection

    Parameters
    ----------
    collection :
        name of the collection
    time_range : ['YYYY-MT-DY','YYYY-MT-DY']
        must be inside the available data
    area : ee.geometry.Geometry
        area of interest

    Returns
    -------
    image_median : ee.image.Image
     """
    collection = ee.ImageCollection(collection)

    # Filter by time range and location
    collection_time = collection.filterDate(time_range[0], time_range[1])
    image_area = collection_time.filterBounds(area)
    image_median = image_area.median()
    return image_median

def obtain_image_sentinel(sentinel_collection, time_range, area):
    """Selection of median, cloud-free image from a collection of images in the Sentinel 2 dataset
    See also: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2

    Parameters
    ----------
    collection :
        name of the collection
    time_range : ['YYYY-MT-DY','YYYY-MT-DY']
        must be inside the available data
    area : ee.geometry.Geometry
        area of interest

    Returns
    -------
    sentinel_median : ee.image.Image
     """
# First, method to remove cloud from the image
    def maskclouds(image):
        band_qa = image.select('QA60')
        cloud_mask = ee.Number(2).pow(10).int()
        cirrus_mask = ee.Number(2).pow(11).int()
        mask = band_qa.bitwiseAnd(cloud_mask).eq(0) and (band_qa.bitwiseAnd(cirrus_mask).eq(0))
        return image.updateMask(mask).divide(10000)

    sentinel_filtered = (ee.ImageCollection(sentinel_collection).
                         filterBounds(area).
                         filterDate(time_range[0], time_range[1]).
                         filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).
                         map(maskclouds))

    sentinel_median = sentinel_filtered.median()
    return sentinel_median


def get_region(geom):
    """Get the region of a given geometry, needed for exporting tasks.

    Parameters
    ----------
    geom : ee.Geometry, ee.Feature, ee.Image
        region of interest

    Returns
    -------
    region : list
    """
    if isinstance(geom, ee.Geometry):
        region = geom.getInfo()["coordinates"]
    elif isinstance(geom, ee.Feature, ee.Image):
        region = geom.geometry().getInfo()["coordinates"]
    elif isinstance(geom, list):
        condition = all([isinstance(item) == list for item in geom])
        if condition:
            region = geom
    return region

def get_url(name, image, scale, region):
    """It will open and download automatically a zip folder containing Geotiff data of 'image'.
    If additional parameters are needed, see also:
    https://github.com/google/earthengine-api/blob/master/python/ee/image.py

    Parameters
    ----------
    name : str
        name of the created folder
    image : ee.image.Image
        image to export
    scale : int
        resolution of export in meters (e.g: 30 for Landsat)
    region : list
        region of interest

    Returns
    -------
    path : str
     """
    path = image.getDownloadURL({
        'name': (name),
        'scale': scale,
        'region': (region)
    })

    webbrowser.open_new_tab(path)
    return path
