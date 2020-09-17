"""
This file is part of CLIMADA.
Copyright (C) 2019 ETH Zurich, CLIMADA contributors listed in AUTHORS.
CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.
CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import time
import logging
from functools import partial

#matplotlib.use('Qt5Agg', force=True)
import matplotlib.pyplot as plt
import pandas as pd
import fiona
from fiona.crs import from_epsg
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely import geometry
from shapely.ops import unary_union, transform, nearest_points
import pyproj
import overpy

from climada.entity import Exposures
from climada.entity.exposures.litpop import LitPop


def _insistent_osm_api_query(query_clause, read_chunk_size=100000, end_of_patience=127):
    """Runs a single Overpass API query through overpy.Overpass.query.
    In case of failure it tries again after an ever increasing waiting period.
    If the waiting period surpasses a given limit an exception is raised.

    Parameters:
        query_clause (str): the query
        read_chunk_size (int): paramter passed over to overpy.Overpass.query
        end_of_patience (int): upper limit for the next waiting period to proceed.

    Returns:
        result as returned by overpy.Overpass.query
    """
    api = overpy.Overpass(read_chunk_size=read_chunk_size)
    waiting_period = 1
    while True:
        try:
            return api.query(query_clause)
        except overpy.exception.OverpassTooManyRequests:
            if waiting_period < end_of_patience:
                print(' WARNING: too many Overpass API requests - try again in {} seconds'.format(
                    waiting_period))
            else:
                raise Exception("Overpass API is consistently unavailable")
        except Exception as exc:
            if waiting_period < end_of_patience:
                print(' WARNING: !!!!\n {}\n try again in {} seconds'.format(exc, waiting_period))
            else:
                raise Exception("The Overpass API is consistently unavailable")
        time.sleep(waiting_period)
        waiting_period *= 2


def _osm_api_query(item, bbox):
    """format query such that it can be passed to OSM api via overpass api

    Parameters:
        item (str): query feature for OSM
        bbox (array): Bounding box for query

    Returns:
        result_NodesFromWays (overpy result object)
        result_NodesWaysFromRels (overpy result object)
        """
    query_clause_NodesFromWays = "way[%s](%f6, %f6, %f6, %f6);(._;>;);out geom;" \
    % (item, bbox[0], bbox[1], bbox[2], bbox[3])
    result_NodesFromWays = _insistent_osm_api_query(query_clause_NodesFromWays)
    print('Nodes from Ways query for %s: done.' % item)

    query_clause_NodesWaysFromRels = ("rel[%s][type=multipolygon](%f6, %f6, %f6, %f6);"
                                      "(._;>;);out;" % (item, bbox[0], bbox[1], bbox[2], bbox[3]))
    result_NodesWaysFromRels = _insistent_osm_api_query(query_clause_NodesWaysFromRels)
    print('Nodes and Ways from Relations query for %s: done.' % item)

    return result_NodesFromWays, result_NodesWaysFromRels


def _format_shape_osm(bbox, result_NodesFromWays, result_NodesWaysFromRels, item, save_path):
    """format edges, nodes and relations from overpy result objects into shapes
    Parameters:
        bbox
        result_NodesFromWays
        result_NodesWaysFromRels
        item
        save_path

    Returns:
        gdf_all: Geodataframe with Linestrings, Polygons & Multipolygons
    """
    # polygon vs. linestrings in nodes from ways result:

    schema_poly = {'geometry': 'Polygon',
                   'properties': {'Name': 'str:80', 'Natural_Type': 'str:80', 'Item': 'str:80'}}
    schema_line = {'geometry': 'LineString',
                   'properties': {'Name': 'str:80', 'Natural_Type': 'str:80', 'Item': 'str:80'}}
    shapeout_poly = save_path + '/' + str(item) + '_poly_' + str(int(bbox[0])) +\
    '_' + str(int(bbox[1])) + ".shp"
    shapeout_line = save_path + '/' + str(item) + '_line_' + str(int(bbox[0])) +\
    '_' + str(int(bbox[1])) + ".shp"

    way_poly = []
    way_line = []
    for way in result_NodesFromWays.ways:
        if (way.nodes[0].id == way.nodes[-1].id) & (len(way.nodes) > 2):
            way_poly.append(way)
        else:
            way_line.append(way)

    with fiona.open(shapeout_poly, 'w', crs=from_epsg(4326), driver='ESRI Shapefile',
                    schema=schema_poly) as output:
        for way in way_poly:
            geom = mapping(geometry.Polygon([node.lon, node.lat] for node in way.nodes))
            prop = {'Name': way.tags.get("name", "n/a"),
                    'Natural_Type': way.tags.get("natural", "n/a"), 'Item': item}
            output.write({'geometry': geom, 'properties': prop})

    with fiona.open(shapeout_line, 'w', crs=from_epsg(4326), driver='ESRI Shapefile',
                    schema=schema_line) as output2:
        for way in way_line:
            geom2 = {'type': 'LineString',
                     'coordinates': [(node.lon, node.lat) for node in way.nodes]}
            prop2 = {'Name': way.tags.get("name", "n/a"),
                     'Natural_Type': way.tags.get("natural", "n/a"), 'Item': item}
            output2.write({'geometry': geom2, 'properties': prop2})

    gdf_poly = geopandas.read_file(shapeout_poly)
    for ending in ['.shp', ".cpg", ".dbf", ".prj", '.shx']:
        os.remove(save_path + '/' + str(item) + '_poly_' + str(int(bbox[0])) +
                  '_' + str(int(bbox[1])) + ending)
    gdf_line = geopandas.read_file(shapeout_line)
    for ending in ['.shp', ".cpg", ".dbf", ".prj", '.shx']:
        os.remove(save_path + '/' + str(item) + '_line_' + str(int(bbox[0])) +
                  '_' + str(int(bbox[1])) + ending)

    # add buffer to the lines (0.000045° are ~5m)
    for geom in gdf_line.geometry:
        geom = geom.buffer(0.000045)

    gdf_all = gdf_poly.append(gdf_line)

    # detect multipolygons in relations:
    print('Converting results for %s to correct geometry and GeoDataFrame: MultiPolygons' % item)

    MultiPoly = []
    for relation in result_NodesWaysFromRels.relations:
        OuterList = []
        InnerList = []
        PolyList = []
        # get inner and outer parts from overpy results, convert into linestrings
        # to check for closedness later
        for relationway in relation.members:
            if relationway.role == 'outer':
                for way in result_NodesWaysFromRels.ways:
                    if way.id == relationway.ref:
                        OuterList.append(
                            geometry.LineString([node.lon, node.lat] for node in way.nodes))
            else:
                for way in result_NodesWaysFromRels.ways:
                    if way.id == relationway.ref:
                        InnerList.append(
                            geometry.LineString([node.lon, node.lat] for node in way.nodes))

        OuterPoly = []
        # in case outer polygons are not fragmented, add those already in correct geometry
        for outer in OuterList:
            if outer.is_closed:
                OuterPoly.append(Polygon(outer.coords[0:(len(outer.coords) + 1)]))
                OuterList.remove(outer)

        initialLength = len(OuterList)
        i = 0
        OuterCoords = []

        # loop to account for more than one fragmented outer ring
        while (len(OuterList) > 0) & (i <= initialLength):
            OuterCoords.append(OuterList[0].coords[0:(len(OuterList[0].coords) + 1)])
            OuterList.remove(OuterList[0])
            for _ in range(0, len(OuterList)):
                # get all the other outer polygon pieces in the right order
                # (only works if fragments are in correct order, anyways!!
                # so added another loop around it in case not!)
                for outer in OuterList:
                    if outer.coords[0] == OuterCoords[-1][-1]:
                        OuterCoords[-1] = OuterCoords[-1] + outer.coords[0:(len(outer.coords) + 1)]
                        OuterList.remove(outer)

        for entry in OuterCoords:
            if len(entry) > 2:
                OuterPoly.append(Polygon(entry))

        PolyList = OuterPoly
        # get the inner polygons (usually in correct, closed shape - not accounting
        # for the fragmented case as in outer poly)
        for inner in InnerList:
            if inner.is_closed:
                PolyList.append(Polygon(inner))

        MultiPoly.append(MultiPolygon([shape(poly) for poly in PolyList]))

    schema_multi = {'geometry': 'MultiPolygon',
                    'properties': {'Name': 'str:80', 'Type': 'str:80', 'Item': 'str:80'}}

    shapeout_multi = (save_path + '/' + str(item) + '_multi_' + str(int(bbox[0])) + '_'
                      + str(int(bbox[1])) + ".shp")

    with fiona.open(shapeout_multi, 'w', crs=from_epsg(4326),
                    driver='ESRI Shapefile', schema=schema_multi) as output:
        for i in range(0, len(MultiPoly)):
            prop1 = {'Name': relation.tags.get("name", "n/a"),
                     'Type': relation.tags.get("type", "n/a"),
                     'Item': item}
            geom = mapping(MultiPoly[i])
            output.write({'geometry': geom, 'properties': prop1})
    gdf_multi = geopandas.read_file(shapeout_multi)  # save_path + '/' + shapeout_multi)
    for ending in ['.shp', ".cpg", ".dbf", ".prj", '.shx']:
        os.remove(save_path + '/' + str(item) + '_multi_' + str(int(bbox[0])) +
                  '_' + str(int(bbox[1])) + ending)
    gdf_all = gdf_all.append(gdf_multi, sort=True)

    print('Combined all results for %s to one GeoDataFrame: done' % item)

    return gdf_all


def _combine_dfs_osm(types, save_path, bbox):
    """Combine all dataframes from individual features into one GeoDataFrame
    Parameters:
        ..
    Returns:
        (gdf)
    """
    print('Combining all low-value GeoDataFrames into one GeoDataFrame...')
    OSM_features_gdf_combined = \
    GeoDataFrame(pd.DataFrame(columns=['Item', 'Name', 'Type', 'Natural_Type', 'geometry']),
                 crs='epsg:4326', geometry='geometry')
    for item in types:
        print('adding results from %s ...' % item)
        OSM_features_gdf_combined = \
        OSM_features_gdf_combined.append(
            globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))],
            ignore_index=True)
    i = 0
    for geom in OSM_features_gdf_combined.geometry:
        if geom.type == 'LineString':
            OSM_features_gdf_combined.geometry[i] = geom.buffer(0.000045)
        i += 1

    OSM_features_gdf_combined.to_file(save_path + '/OSM_features_' + str(int(bbox[0])) +
                                      '_' + str(int(bbox[1])) + '.shp')

    return OSM_features_gdf_combined

def get_features_OSM(bbox, types, save_path=os.getcwd(), check_plot=1):
    """
    Get shapes from all types of objects that are available on Open Street Map via an API query
    and save them as geodataframe.

    Parameters:
         bbox (array): List of coordinates in format [South, West, North, East]
         types (list): List of features items that should be downloaded from OSM, e.g.
                {'natural','waterway','water', 'landuse=forest','landuse=farmland',
                'landuse=grass','wetland'}
         save_path (str): String with absolute path for saving output. Default is cwd
         check_plot: default is 1 (yes), else 0.

    Returns:
          OSM_features_gdf_combined(gdf): combined GeoDataframe with all features saved as
          "OSM_features_lat_lon".
          Shapefiles with correct geometry (LineStrings,Polygons, MultiPolygons)
           for each of requested OSM feature saved as "item_gdf_all_lat_lon"

    Example 1:
        Houses_47_8 = get_features_OSM([47.16, 8.0, 47.3, 8.0712],\
                                      {'building'}, \
                                      save_path = save_path, check_plot=1)
    Example 2:
        Low_Value_gdf_47_8 = get_features_OSM([47.16, 8.0, 47.3, 8.0712],\
                                      {'natural','water', 'waterway',
                                      'landuse=forest', 'landuse=farmland',
                                      'landuse=grass', 'wetland'}, \
                                      save_path = save_path, check_plot=1)
    """
    for item in types:
        # API Queries for relations, nodes and ways
        print('Querying Relations, Nodes and Ways for %s...' % item)
        result_NodesFromWays, result_NodesWaysFromRels = _osm_api_query(item, bbox)

        # Formatting results for each feature
        # into correct shapes (LineStrings, Polygons, MultiPolygons)
        print('Converting results for %s to correct geometry and GeoDataFrame: Lines and Polygons'
              % item)
        globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' + str(int(bbox[1]))] = \
        _format_shape_osm(bbox, result_NodesFromWays, result_NodesWaysFromRels, item, save_path)

        # Checkplot for each feature (1 dataframe each)
        if check_plot == 1:
            f, ax = plt.subplots(1)
            ax = globals()[str(item) + '_gdf_all_' + str(int(bbox[0])) + '_' +
                           str(int(bbox[1]))].plot(ax=ax)
            f.suptitle(str(item) + '_' + str(int(bbox[0])) + '_' + str(int(bbox[1])))
            plt.show()

    # Combine all dataframes into one, save with converting all to (multi)polygons.
    OSM_features_gdf_combined = _combine_dfs_osm(types, save_path, bbox)

    if check_plot == 1:
        f, ax = plt.subplots(1)
        ax = OSM_features_gdf_combined.plot(ax=ax)
        f.suptitle('Features_' + str(int(bbox[0])) + '_' + str(int(bbox[1])))
        plt.show()
        f.savefig('Features_' + str(int(bbox[0])) + '_' + str(int(bbox[1])) + '.pdf',
                  bbox_inches='tight')

    return OSM_features_gdf_combined


def _makeUnion(gdf):
    """
    Solve issue of invalid geometries in MultiPolygons, which prevents that
    shapes can be combined into one unary union, save the respective Union
    """
    union1 = gdf[gdf.geometry.type == 'Polygon'].unary_union
    union2 = gdf[gdf.geometry.type != 'Polygon'].geometry.buffer(0).unary_union
    Low_Value_Union = unary_union([union1, union2])
    return Low_Value_Union

def get_highValueArea(bbox, save_path=os.getcwd(), Low_Value_gdf=None, check_plot=1):
    """
    In case low-value features were queried with get_features_OSM(),
    calculate the "counter-shape" representig high value area for a given bounding box.

    Parameters:
        bbox (array): List of coordinates in format [South, West, North, East]
        save_path (str): path for results
        Low_Value_gdf (str): absolute path of gdf of low value items which is to be inverted.
          If left empty, searches for OSM_features_gdf_combined_lat_lon.shp in save_path.
        checkplot

    Returns:
        High_Value_Area (gdf): GeoDataFrame of High Value Area as High_Value_Area_lat_lon

    Example:
        High_Value_gdf_47_8 = get_highValueArea([47.16, 8.0, 47.3, 8.0712], save_path = save_path,\
                                    Low_Value_gdf = save_path+'/Low_Value_gdf_combined_47_8.shp')
    important: Use same bbox and save_path as for get_features_OSM().
    """

    Outer_Poly = geometry.Polygon([(bbox[1], bbox[2]), (bbox[1], bbox[0]),
                                   (bbox[3], bbox[0]), (bbox[3], bbox[2])])


    if Low_Value_gdf is None:
        try:
            Low_Value_gdf = geopandas.read_file(
                save_path + '/OSM_features_gdf_combined_' + str(int(bbox[0])) + '_'
                + str(int(bbox[1])) + '.shp')
        except:
            print('No Low-Value-Union found with name %s. \n Please add.'
                  % (save_path + '/OSM_features_gdf_combined_' + str(int(bbox[0])) + '_' +
                     str(int(bbox[1])) + '.shp'))
    else:
        Low_Value_gdf = geopandas.read_file(Low_Value_gdf)

    # Making one Union of individual shapes in gdfs
    Low_Value_Union = _makeUnion(Low_Value_gdf)

    # subtract low-value areas from high-value polygon
    High_Value_Area = Outer_Poly.difference(Low_Value_Union)

    # save high value multipolygon as shapefile and re-read as gdf:
    schema = {'geometry': 'MultiPolygon', 'properties': {'Name': 'str:80'}}
    shapeout = (save_path + '/High_Value_Area_' + str(int(bbox[0]))
                + '_' + str(int(bbox[1])) + ".shp")
    with fiona.open(shapeout, 'w', crs=from_epsg(4326), driver='ESRI Shapefile',
                    schema=schema) as output:
        prop1 = {'Name': 'High Value Area'}
        geom = mapping(High_Value_Area)
        output.write({'geometry': geom, 'properties': prop1})

    High_Value_Area = geopandas.read_file(shapeout)

    # plot
    if check_plot == 1:
        f, ax = plt.subplots(1)
        ax = High_Value_Area.plot(ax=ax)
        f.suptitle('High Value Area ' + str(int(bbox[0])) + ' ' + str(int(bbox[1])))
        plt.show()
        f.savefig('High Value Area ' + str(int(bbox[0])) + '_' + str(int(bbox[1])) +
                  '.pdf', bbox_inches='tight')

    return High_Value_Area

def _get_litpop_bbox(country, highValueArea, **kwargs):
    """get litpop exposure for the bbox area of the queried OSM features
    Parameters:
        country (str)
        highValueArea (gdf)
        bbox (array)
        kwargs (dict): arguments for LitPop set_country method
    Returns:
        exp_sub (exposure)
        High_Value_Area_gdf (gdf)
    """
    # Load LitPop Exposure for whole country, and High Value Area
    exp = LitPop()
    exp.set_country(country, **kwargs)
    exp.set_geometry_points()

    # Crop bbox of High Value Area from Country Exposure
    exp_sub = exp.cx[min(highValueArea.bounds.minx):max(highValueArea.bounds.maxx),
                     min(highValueArea.bounds.miny):max(highValueArea.bounds.maxy)]

    return exp_sub

def _split_exposure_highlow(exp_sub, mode, High_Value_Area_gdf):
    """divide litpop exposure into high-value exposure and low-value exposure
    according to area queried in OSM, re-assign all low values to high-value centroids
    Parameters:
        exp_sub (exposure)
        mode (str)
    Returns:
        exp_sub_high (exposure)
    """

    exp_sub_high = pd.DataFrame(columns=exp_sub.columns)
    exp_sub_low = pd.DataFrame(columns=exp_sub.columns)
    for i, pt in enumerate(exp_sub.geometry):
        if pt.within(High_Value_Area_gdf.loc[0]['geometry']):
            exp_sub_high = exp_sub_high.append(exp_sub.iloc[i])
        else:
            exp_sub_low = exp_sub_low.append(exp_sub.iloc[i])

    exp_sub_high = GeoDataFrame(exp_sub_high, crs=exp_sub.crs, geometry=exp_sub_high.geometry)
    exp_sub_low = GeoDataFrame(exp_sub_low, crs=exp_sub.crs, geometry=exp_sub_low.geometry)

    if mode == "nearest":
        # assign asset values of low-value points to nearest point in high-value df.
        pointsToAssign = exp_sub_high.geometry.unary_union
        exp_sub_high["addedValNN"] = 0
        for i in range(0, len(exp_sub_low)):
            nearest = exp_sub_high.geometry == nearest_points(exp_sub_low.iloc[i].geometry,
                                                              pointsToAssign)[1]  # point
            exp_sub_high.addedValNN.loc[nearest] = exp_sub_low.iloc[i].value
        exp_sub_high["combinedValNN"] = exp_sub_high[['addedValNN', 'value']].sum(axis=1)
        exp_sub_high.rename(columns={'value': 'value_old', 'combinedValNN': 'value'},
                            inplace=True)

    elif mode == "even":
        # assign asset values of low-value points evenly to points in high-value df.
        exp_sub_high['addedValeven'] = sum(exp_sub_low.value) / len(exp_sub_high)
        exp_sub_high["combinedValeven"] = exp_sub_high[['addedValeven', 'value']].sum(axis=1)
        exp_sub_high.rename(columns={'value': 'value_old', 'combinedValeven': 'value'},
                            inplace=True)

    elif mode == "proportional":
        # assign asset values of low-value points proportionally
        # to value of points in high-value df.
        exp_sub_high['addedValprop'] = 0
        for i in range(0, len(exp_sub_high)):
            asset_factor = exp_sub_high.iloc[i].value / sum(exp_sub_high.value)
            exp_sub_high.addedValprop.iloc[i] = asset_factor * sum(exp_sub_low.value)
        exp_sub_high["combinedValprop"] = exp_sub_high[['addedValprop', 'value']].sum(axis=1)
        exp_sub_high.rename(columns={'value': 'value_old', 'combinedValprop': 'value'},
                            inplace=True)

    else:
        print("No proper re-assignment mode set. "
              "Please choose either nearest, even or proportional.")

    return exp_sub_high

def get_osmstencil_litpop(bbox, country, mode, highValueArea=None,
                          save_path=os.getcwd(), check_plot=1, **kwargs):
    """
    Generate climada-compatible exposure by downloading LitPop exposure for a bounding box,
    corrected for centroids which lie inside a certain high-value multipolygon area
    from previous OSM query.

    Parameters:
        bbox (array): List of coordinates in format [South, West, North, East]
        Country (str): ISO3 code or name of country in which bbox is located
        highValueArea (str): path of gdf of high-value area from previous step.
          If empty, searches for cwd/High_Value_Area_lat_lon.shp
        mode (str): mode of re-assigning low-value points to high-value points.
          "nearest", "even", or "proportional"
        kwargs (dict): arguments for LitPop set_country method

    Returns:
        exp_sub_high_exp (Exposure): (CLIMADA-compatible) with re-allocated asset
          values with name exposure_high_lat_lon

    Example:
        exposure_high_47_8 = get_osmstencil_litpop([47.16, 8.0, 47.3, 8.0712],\
                          'CHE',"proportional", highValueArea = \
                          save_path + '/High_Value_Area_47_8.shp' ,\
                          save_path = save_path)
    """
    if highValueArea is None:
        try:
            High_Value_Area_gdf = \
            geopandas.read_file(os.getcwd() + '/High_Value_Area_' + str(int(bbox[0])) + '_' +
                                str(int(bbox[1])) + ".shp")
        except:
            print('No file found of form %s. Please add or specify path.'
                  % (os.getcwd() + 'High_Value_Area_' + str(int(bbox[0])) + '_' +
                     str(int(bbox[1])) + ".shp"))
    else:
        High_Value_Area_gdf = geopandas.read_file(highValueArea)

    exp_sub = _get_litpop_bbox(country, High_Value_Area_gdf, **kwargs)

    exp_sub_high = _split_exposure_highlow(exp_sub, mode, High_Value_Area_gdf)

    ###### how to "spread" centroids with value to e.g. hexagons? ###########
    # put exp_sub_high back into CLIMADA-compatible exposure format and save as hdf5 file:
    exp_sub_high_exp = Exposures(exp_sub_high)
    exp_sub_high_exp.set_lat_lon()
    exp_sub_high_exp.check()
    exp_sub_high_exp.write_hdf5(save_path + '/exposure_high_' + str(int(bbox[0])) +
                                '_' + str(int(bbox[1])) + '.h5')
    # plotting
    if check_plot == 1:
        # normal hexagons
        exp_sub_high_exp.plot_hexbin(pop_name=True)
        # select the OSM background image from the available ctx.sources - doesnt work atm
        #fig, ax = exp_sub_high_exp.plot_basemap(buffer=30000, url=ctx.sources.OSM_C, cmap='brg')

    return exp_sub_high_exp

def _get_midpoints(highValueArea):
    """get midpoints from polygon and multipolygon shapes for current CLIMADA-
    exposure compatibility (centroids / points)

    Parameters:
        highValueArea (gdf)

    Returns:
        High_Value_Area_gdf
    """
    High_Value_Area_gdf = geopandas.read_file(highValueArea)

    # For current exposure structure, simply get centroid
    # and area (in m2) for each building polygon
    High_Value_Area_gdf['projected_area'] = 0
    High_Value_Area_gdf['Midpoint'] = 0

    for index in High_Value_Area_gdf.index:
        High_Value_Area_gdf.loc[index, "Midpoint"] = \
        High_Value_Area_gdf.loc[index, "geometry"].centroid.wkt
        s = shape(High_Value_Area_gdf.loc[index, "geometry"])
        # turn warnings off, otherwise Future and Deprecation warnings are flooding the logs
        logging.captureWarnings(True)
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                       pyproj.Proj(init='epsg:3857'))
        High_Value_Area_gdf.loc[index, "projected_area"] = transform(proj, s).area
        # turn warnings on again
        logging.captureWarnings(False)
        
    # change active geometry from polygons to midpoints
    from shapely.wkt import loads
    High_Value_Area_gdf = High_Value_Area_gdf.rename(columns={'geometry': 'geo_polys',
                                                              'Midpoint': 'geometry'})
    High_Value_Area_gdf['geometry'] = High_Value_Area_gdf['geometry'].apply(lambda x: loads(x))
    High_Value_Area_gdf = High_Value_Area_gdf.set_geometry('geometry')

    return High_Value_Area_gdf

def _assign_values_exposure(High_Value_Area_gdf, mode, country, **kwargs):
    """add value-columns to high-resolution exposure gdf
    according to m2 area of underlying features.

    Parameters:
        High_Value_Area_gdf
        mode
        country
        kwargs (dict): arguments for LitPop set_country method

    Returns:
        exp_sub_high
    """

    if mode == "LitPop":
        # assign LitPop values of this area to houses.
        exp_sub = _get_litpop_bbox(country, High_Value_Area_gdf, **kwargs)
        totalValue = sum(exp_sub.value)
        totalArea = sum(High_Value_Area_gdf['projected_area'])
        High_Value_Area_gdf['value'] = 0
        for index in High_Value_Area_gdf.index:
            High_Value_Area_gdf.loc[index, 'value'] = \
            High_Value_Area_gdf.loc[index, 'projected_area'] / totalArea * totalValue

    elif mode == "default":  # 5400 Chf / m2 base area
        High_Value_Area_gdf['value'] = 0
        for index in High_Value_Area_gdf.index:
            High_Value_Area_gdf.loc[index, 'value'] = \
            High_Value_Area_gdf.loc[index, 'projected_area'] * 5400

    return High_Value_Area_gdf

def make_osmexposure(highValueArea, mode="default", country=None,
                     save_path=os.getcwd(), check_plot=1, **kwargs):
    """
    Generate climada-compatiple entity by assigning values to midpoints of
    individual house shapes from OSM query, according to surface area and country.

    Parameters:
        highValueArea (str): absolute path for gdf of building features queried
          from get_features_OSM()
        mode (str): "LitPop" or "default": Default assigns a value of 5400 Chf to
          each m2 of building, LitPop assigns total LitPop value for the region
          proportionally to houses (by base area of house)
        Country (str): ISO3 code or name of country in which entity is located.
          Only if mode = LitPop
        kwargs (dict): arguments for LitPop set_country method

    Returns:
        exp_building (Exposure): (CLIMADA-compatible) with allocated asset values.
          Saved as exposure_buildings_mode_lat_lon.h5

    Example:
        buildings_47_8 = \
        make_osmexposure(save_path + '/OSM_features_47_8.shp',
                         mode="default", save_path = save_path, check_plot=1)
    """
    High_Value_Area_gdf = _get_midpoints(highValueArea)

    High_Value_Area_gdf = _assign_values_exposure(High_Value_Area_gdf, mode, country, **kwargs)

    # put back into CLIMADA-compatible entity format and save as hdf5 file:
    exp_buildings = Exposures(High_Value_Area_gdf)
    exp_buildings.set_lat_lon()
    exp_buildings.check()
    exp_buildings.write_hdf5(save_path + '/exposure_buildings_' + mode + '_' +
                             str(int(min(High_Value_Area_gdf.bounds.miny))) +
                             '_' + str(int(min(High_Value_Area_gdf.bounds.minx))) + '.h5')

    # plotting
    if check_plot == 1:
        # normal hexagons
        exp_buildings.plot_hexbin(pop_name=True)
        # select the OSM background image from the available ctx.sources
        # - returns connection error, left out for now:
        #fig, ax = exp_buildings.plot_basemap(buffer=30000, url=ctx.sources.OSM_C, cmap='brg')

    return exp_buildings
