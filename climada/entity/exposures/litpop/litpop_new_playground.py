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
from matplotlib import pyplot as plt


import climada.util.coordinates as u_coord
from climada.entity.exposures.litpop import nightlight as nl_util
# from climada.entity.exposures.litpop import gpw_import
from climada.entity.exposures.litpop import gpw_population as pop_util

from climada.util.constants import SYSTEM_DIR

import_bm = False
import_pop = False
test_litpop = True

cntry = ['USA']
year = 2016

if test_litpop:
    from climada.entity.exposures.litpop import litpop as lp
    exp = lp.LitPop()
    exp.set_countries(cntry, res_arcsec=None,
                    exponents=(1,1), fin_mode='pc', total_values=None,
                    admin1_calc=False, conserve_cntrytotal=True,
                    reference_year=2016, gpw_version=None, data_dir=None,
                    resample_first=True)


# Malta
bounds = (14.18, 35.78, 14.58, 36.09) # (14.18, 14.58, 35.78, 36.09) # (min_lon, max_lon, min_lat, max_lat)
# bounds = (-85, -11, 5, 40)
shape_cntry = Polygon([
    (bounds[0], bounds[3]),
    (bounds[2], bounds[3]),
    (bounds[2], bounds[1]),
    (bounds[0], bounds[1])
    ])
# Spain
shape_cntry = u_coord.get_land_geometry([cntry])
#shape_cntry = u_coord.get_land_geometry(['RUS'])
#shape_cntry = u_coord.get_land_geometry(['CHE'])

if import_pop:
    
    ver = 11
    year = 2020
    """
    DIRNAME_GPW = f'gpw-v4-population-count-rev{ver}_{year}_30_sec_tif'
    path = SYSTEM_DIR / DIRNAME_GPW / (gpw_import.FILENAME_GPW % (ver, year))
    src = rasterio.open(path)
    pop_trafo_glb = src.transform
    pop, out_transform = rasterio.mask.mask(src, [shape_cntry], crop=True,
                                                  nodata=0)
    plt.imshow(pop[0,:,:], cmap='pink')
    plt.show()
    meta_pop = src.meta
    meta_pop.update({"driver": "GTiff",
                 "height": pop.shape[1],
                 "width": pop.shape[2],
                 "transform": out_transform})
    print(meta_pop)
    pop = pop[0,:,:]
    """
    pop, meta_pop, global_transform = pop_util.load_gpw_pop_shape(shape_cntry, year, gpw_version=ver,
                                       data_dir=None, layer=0)
if import_bm:
    nl, meta_nl = nl_util.load_nasa_nl_shape(shape_cntry, 2016)
    plt.imshow(nl, cmap='pink')
    print(meta_nl)
    plt.show()


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
plt.imshow(destination, cmap='pink', filternorm=False)
plt.show()


res_arcsec = 1800
res_degree = res_arcsec / 3600

# calculate new longitude and latitude origins in destination grid
# (for global consistency):
buffer = 0
dst_orig_lon = int(np.floor((180 + meta_pop['transform'][2]) / res_degree))-buffer
dst_orig_lon = -180 + np.max([0, dst_orig_lon]) * res_degree

dst_orig_lat = int(np.floor((90 - meta_pop['transform'][5]) / res_degree))-buffer
dst_orig_lat = 90 - np.max([0, dst_orig_lat]) * res_degree

dst_shape = (int(min([180/res_degree,
                      pop.shape[0] / (res_degree/meta_pop['transform'][0])+1+2*buffer])),
             int(min([360/res_degree,
                      pop.shape[1] / (res_degree/meta_pop['transform'][0])+1+2*buffer])),
             )

dst_transform = rasterio.Affine(res_degree,
                              global_transform[1],
                              dst_orig_lon,
                              global_transform[3],
                              -res_degree, # pop_trafo_glb[4],
                              dst_orig_lat,
                              )

dst_transform = rasterio.Affine(res_degree,
                              0,
                              dst_orig_lon,
                              0,
                              -res_degree, # pop_trafo_glb[4],
                              dst_orig_lat,
                              )

destination_rough = np.zeros(dst_shape, dtype=meta_nl['dtype'])

rasterio.warp.reproject(
                source=nl,
                destination=destination_rough,
                src_transform=meta_nl['transform'],
                # src_crs=meta_nl['crs'],
                src_crs=meta_pop['crs'], # why?
                dst_transform=dst_transform,
                dst_crs=meta_pop['crs'],
                resampling=resampling,
                )
plt.imshow(destination_rough, cmap='pink', filternorm=False)
plt.show()



data_out, meta_out = resample_input_data([pop, nl], [meta_pop, meta_nl],
                        i_ref=0,
                        target_res_arcsec=None,
                        global_origins=(-180.0, 89.99999999999991),
                        target_crs=None,
                        resampling=None)

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
    
    plt.imshow(out_image[0,:,:], cmap='pink')
    
    with rasterio.open(SYSTEM_DIR / "BlackMarble_demo_malta.tif", "w", **out_meta) as dest:
        dest.write(out_image)
    """
