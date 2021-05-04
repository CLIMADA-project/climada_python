#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  3 16:12:25 2021

@author: eberenzs
"""
import numpy as np
import shapefile
from shapely.geometry import Polygon
import rasterio
from matplotlib import pyplot


import climada.util.coordinates as u_coord
import climada.entity.exposures.nightlight as nl_utils
from climada.entity.exposures import gpw_import

from climada.util.constants import SYSTEM_DIR

import_bm = True
import_pop = True

# Malta
year = 2016
bounds = (14.18, 35.78, 14.58, 36.09) # (14.18, 14.58, 35.78, 36.09) # (min_lon, max_lon, min_lat, max_lat)
# bounds = (-85, -11, 5, 40)
shape_cntry = Polygon([
    (bounds[0], bounds[3]),
    (bounds[2], bounds[3]),
    (bounds[2], bounds[1]),
    (bounds[0], bounds[1])
    ])
# Spain
#shape_cntry = u_coord.get_land_geometry(['ESP', 'POR'])
# shape_cntry = u_coord.get_land_geometry(['RUS'])


if import_pop:
    ver = 11
    year = 2020
    DIRNAME_GPW = f'gpw-v4-population-count-rev{ver}_{year}_30_sec_tif'
    path = SYSTEM_DIR / DIRNAME_GPW / (gpw_import.FILENAME_GPW % (ver, year))
    src = rasterio.open(path)
    pop, out_transform = rasterio.mask.mask(src, [shape_cntry], crop=True,
                                                  nodata=0)
    pyplot.imshow(pop[0,:,:], cmap='pink')
    pyplot.show()
    meta_pop = src.meta
    meta_pop.update({"driver": "GTiff",
                 "height": pop.shape[1],
                 "width": pop.shape[2],
                 "transform": out_transform})
    print(meta_pop)
    pop = pop[0,:,:]
if import_bm:
    nl, meta_nl = nl_utils.load_nasa_nl_shape(shape_cntry, 2016)
    pyplot.imshow(nl, cmap='pink')
    print(meta_nl)
    pyplot.show()


destination = np.zeros(pop.shape, dtype=meta_nl['dtype'])

resampling = rasterio.warp.Resampling.bilinear

rasterio.warp.reproject(
                source=nl,
                destination=destination,
                src_transform=meta_nl['transform'],
                #src_crs=meta_nl['crs'],
                src_crs=meta_pop['crs'], # why?
                dst_transform=meta_pop['transform'],
                dst_crs=meta_pop['crs'],
                resampling=resampling,
                )
pyplot.imshow(destination, cmap='pink')
pyplot.show()

#rasterio.warp.reproject(
#                source=rasterio.band(src, band),
#                destination=data[iband],
#                src_transform=src.transform,
#                src_crs=src.crs,
#                dst_transform=transform,
#                dst_crs=crs,
#                resampling=resampling)


"""for meta in metas:
    if meta is None:
        lon_mins.append(None)
        lat_maxs.append(None)
    else:
        lon_mins.append(meta['transform'][2])
        lat_maxs.append(meta['transform'][5])"""


"""
req_files = nl_utils.check_required_nl_files(bounds)
print(req_files)


# shape = shapefile.Reader(SYSTEM_DIR/ "BlackMarbleTiles" /"BlackMarbleTiles.shp")


# check whether BM file exists
nl_utils.check_nl_local_file_exists(required_files=req_files,
                                    year=year)

data = [None] * int(sum(req_files))
transform = [None] * int(sum(req_files))
for num_files in range(0, np.count_nonzero(nl_utils.BM_FILENAMES)):
    if req_files[num_files] == 0:
        continue

    src = rasterio.open(SYSTEM_DIR / (nl_utils.BM_FILENAMES[num_files] %(year)))
    out_image, out_transform = rasterio.mask.mask(src, [bbox], crop=True)
    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})
    lon=np.arange(out_transform[2], out_transform[2]+(out_image.shape[2]-1)*out_transform[0], out_transform[0])
    lat=np.arange(out_transform[5], out_transform[5]+(out_image.shape[1]-1)*out_transform[4], out_transform[4])
    
    
            lon=np.arange(transform[2], transform[2]+(out_image.shape[2]-1)*
                      transform[0], transform[0])
        lat=np.arange(transform[5], transform[5]+(out_image.shape[1]-1)*
                      transform[4], transform[4])
    
    pyplot.imshow(out_image[0,:,:], cmap='pink')
    
    with rasterio.open(SYSTEM_DIR / "BlackMarble_demo_malta.tif", "w", **out_meta) as dest:
        dest.write(out_image)
    """
