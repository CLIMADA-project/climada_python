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
import matplotlib
matplotlib.use('Qt5Agg', force=True)
matplotlib.get_backend()
import matplotlib.pyplot as plt
import pandas as pd
import os
import fiona
from fiona.crs import from_epsg
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely import geometry
from shapely.ops import unary_union, transform, nearest_points
import contextily as ctx
import pyproj
from functools import partial
import overpy

from climada.entity.exposures.base import Exposures
from climada.entity.exposures.litpop import LitPop


def get_features_OSM(bbox, types, save_path=os.getcwd(), check_plot=1):
    """
    Get shapes from all types of objects that are available on Open Street Map via an API query.
    Input:
         bbox: List of coordinates in format [South, West, North, East]
         types: List of low-value items that should be downloaded from OSM, e.g.
                recommended:
                {'natural','waterway','water','landuse' (recommended) OR
                'landuse=forest','landuse=farmland','landuse=grass', ...,
                'wetland'}
         save_path: String with absolute path for saving output. Default is cwd
         check_plot = default is 1 (yes), else 0.

    Output:
         - GeoDataframes, shapefiles and plots with correct geometry
           (LineStrings,Polygons, MultiPolygons)
           for each of requested OSM feature saved as "item_gdf_all_lat_lon"
         - 1 combined GeoDataframe with all features saved as
           "OSM_features_gdf_combined_lat_lon".
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
        ###################
        # Queries for relations, nodes and ways
        ###################

        print('Querying Relations, Nodes and Ways for %s...' %item)

        query_clause_NodesFromWays = "way[%s](%f6, %f6, %f6, %f6);(._;>;);out geom;" \
        % (item, bbox[0], bbox[1], bbox[2], bbox[3])

        query_clause_NodesWaysFromRels = \
        "rel[%s][type=multipolygon](%f6, %f6, %f6, %f6);(._;>;);out;" \
        % (item, bbox[0], bbox[1], bbox[2], bbox[3])

        api = overpy.Overpass(read_chunk_size=100000)
        try:
            result_NodesFromWays = api.query(query_clause_NodesFromWays)
            print('Nodes from Ways query for %s: done.' %item)
        except Exception as e:
            print(' WARNING: !!!! \n %s - try again in a few moments \n !!!!' %e)

        api = overpy.Overpass(read_chunk_size=100000)
        try:
            result_NodesWaysFromRels = api.query(query_clause_NodesWaysFromRels)
            print('Nodes and Ways from Relations query for %s: done.' %item)
        except Exception as e:
            print(' WARNING: !!!! \n %s - try again in a few moments \n !!!!' %e)

        print('Resolving missing nodes and ways for %s: done.' %item)

        ###################
        # Formatting results into correct shapes (LineStrings, Polygons, MultiPolygons)
        ###################
        print('Converting results for %s to correct geometry and GeoDataFrame: Lines and Polygons' %item)

        def format_shape_OSM(bbox, result_NodesFromWays, result_NodesWaysFromRels):
            # polygon vs. linestrings in nodes from ways result:

            schema_poly = {'geometry': 'Polygon', 'properties': {'Name':'str:80', \
                                                                 'Natural_Type':'str:80',\
                                                                 'Item':'str:80'}}
            schema_line = {'geometry': 'LineString', 'properties': {'Name':'str:80', \
                                                                    'Natural_Type':'str:80',\
                                                                    'Item':'str:80'}}
            shapeout_poly = save_path + '/' + str(item)+'_poly_'+ \
            str(int(bbox[0]))+'_'+str(int(bbox[1]))+".shp"
            shapeout_line = save_path + '/' + str(item)+'_line_'+ \
            str(int(bbox[0]))+'_'+str(int(bbox[1]))+".shp"

            way_poly = []
            way_line = []
            for way in result_NodesFromWays.ways:
                if (way.nodes[0].id == way.nodes[-1].id) & (len(way.nodes) > 2):
                    way_poly.append(way)
                else:
                    way_line.append(way)

            with fiona.open(shapeout_poly, 'w', crs=from_epsg(4326), \
                            driver='ESRI Shapefile', schema=schema_poly)\
                            as output:
                for way in way_poly:
                    geom = mapping(geometry.Polygon([node.lon, node.lat] \
                                                    for node in way.nodes))
                    prop = {'Name': way.tags.get("name", "n/a"), \
                            'Natural_Type': way.tags.get("natural", "n/a"),\
                            'Item': item}
                    output.write({'geometry': geom, 'properties': prop})

            with fiona.open(shapeout_line, 'w', crs=from_epsg(4326), \
                            driver='ESRI Shapefile', schema=schema_line)\
                            as output2:
                for way in way_line:
                    geom2 = {'type': 'LineString', \
                             'coordinates':[(node.lon, node.lat) for node in way.nodes]}
                    prop2 = {'Name': way.tags.get("name", "n/a"), \
                             'Natural_Type': way.tags.get("natural", "n/a"), \
                             'Item': item}
                    output2.write({'geometry': geom2, 'properties': prop2})

            gdf_poly = geopandas.read_file(shapeout_poly) #save_path + '/' + shapeout_poly

            gdf_line = geopandas.read_file(shapeout_line) #save_path + '/' + shapeout_line
            # add buffer to the lines (0.000045° are ~5m)
            for geom in gdf_line.geometry:
                geom = geom.buffer(0.000045)

            gdf_all = gdf_poly.append(gdf_line)

            # detect multipolygons in relations:
            print('Converting results for %s to correct geometry and GeoDataFrame: MultiPolygons'\
                  %item)

            MultiPoly = []
            for relation in result_NodesWaysFromRels.relations:
                OuterList = []
                InnerList = []
                PolyList = []
                for relationway in relation.members: # get inner and outer parts from overpy results, convert into linestrings to check for closedness later
                    if relationway.role == 'outer':
                        for way in result_NodesWaysFromRels.ways:
                            if way.id == relationway.ref:
                                OuterList.append(geometry.LineString([node.lon, node.lat] \
                                                                     for node in way.nodes))
                    else:
                        for way in result_NodesWaysFromRels.ways:
                            if way.id == relationway.ref:
                                InnerList.append(geometry.LineString([node.lon, node.lat] \
                                                                     for node in way.nodes))

                OuterPoly = []
                for outer in OuterList: # in case outer polygons are not fragmented, add those already in correct geometry
                    if outer.is_closed:
                        OuterPoly.append(Polygon(outer.coords[0:(len(outer.coords)+1)]))
                        OuterList.remove(outer)

                initialLength = len(OuterList)

                i = 0
                OuterCoords = []

                while (len(OuterList) > 0) & (i <= initialLength): # loop to account for more than one fragmented outer ring
                    OuterCoords.append(OuterList[0].coords[0:(len(OuterList[0].coords)+1)])
                    OuterList.remove(OuterList[0])
                    count = 0
                    for count in range(0, len(OuterList)):
                        for outer in OuterList: # get all the other outer polygon pieces in the right order (only works if fragments are in correct order, anyways!! so added another loop around it in case not!)
                            if outer.coords[0] == OuterCoords[-1][-1]:
                                OuterCoords[-1] = OuterCoords[-1] + \
                                outer.coords[0:(len(outer.coords)+1)]
                                OuterList.remove(outer)

                for entry in OuterCoords:
                    if len(entry) > 2:
                        OuterPoly.append(Polygon(entry))

                PolyList = OuterPoly
                #get the inner polygons (usually in correct, closed shape - not accounting for the fragmented case as in outer poly)
                for inner in InnerList:
                    if inner.is_closed:
                        PolyList.append(Polygon(inner))

                MultiPoly.append(MultiPolygon([shape(poly) for poly in PolyList]))

            schema_multi = {'geometry': 'MultiPolygon',\
                            'properties': {'Name':'str:80', 'Type':'str:80',\
                                           'Item': 'str:80'}}

            shapeout_multi = save_path + '/' + str(item)+'_multi_'+str(int(bbox[0]))+'_'+\
            str(int(bbox[1]))+".shp"

            with fiona.open(shapeout_multi, 'w', crs=from_epsg(4326), \
                            driver='ESRI Shapefile', schema=schema_multi) \
                            as output:
                for i in range(0, len(MultiPoly)):
                    prop1 = {'Name': relation.tags.get("name", "n/a"),
                             'Type': relation.tags.get("type", "n/a"),
                             'Item': item}
                    geom = mapping(MultiPoly[i])
                    output.write({'geometry': geom, 'properties': prop1})
            gdf_multi = geopandas.read_file(shapeout_multi) #save_path + '/' + shapeout_multi)

            gdf_all = gdf_all.append(gdf_multi, sort=True)

            print('Combined all results for %s to one GeoDataFrame: done' %item)

            return gdf_all


        globals()[str(item)+'_gdf_all_'+str(int(bbox[0]))+'_'+str(int(bbox[1]))] = \
        format_shape_OSM(bbox, result_NodesFromWays, result_NodesWaysFromRels)

        if check_plot == 1:
            f, ax = plt.subplots(1)
            ax = globals()[str(item)+'_gdf_all_'+str(int(bbox[0]))+'_'+\
                        str(int(bbox[1]))].plot(ax=ax)
            f.suptitle(str(item)+'_'+str(int(bbox[0]))+'_'+str(int(bbox[1])))
            plt.show()

    ###################
    # Combining all results into one GeoDataFrame
    ###################

    print('Combining all low-value GeoDataFrames into one GeoDataFrame...')
    OSM_features_gdf_combined = \
    GeoDataFrame(pd.DataFrame(columns=['Item', 'Name', 'Type', 'Natural_Type', 'geometry']),
                 crs='epsg:4326', geometry='geometry')
    for item in types:
        print('adding results from %s ...' %item)
        OSM_features_gdf_combined = \
        OSM_features_gdf_combined.append(
            globals()[str(item)+'_gdf_all_'+str(int(bbox[0]))+'_'+str(int(bbox[1]))],
            ignore_index=True)
    i = 0
    for geom in OSM_features_gdf_combined.geometry:
        if geom.type == 'LineString':
            OSM_features_gdf_combined.geometry[i] = geom.buffer(0.000045)
        i += 1

    OSM_features_gdf_combined.to_file(save_path +'/OSM_features_gdf_combined_'+str(int(bbox[0]))+\
                                      '_'+str(int(bbox[1]))+'.shp')


    if check_plot == 1:
        f, ax = plt.subplots(1)
        ax = OSM_features_gdf_combined.plot(ax=ax)
        f.suptitle('Features'+str(int(bbox[0]))+'_'+str(int(bbox[1])))
        plt.show()
        f.savefig('Features'+str(int(bbox[0]))+'_'+str(int(bbox[1]))+'.pdf', bbox_inches='tight')

    return OSM_features_gdf_combined

def get_highValueArea(bbox, save_path=os.getcwd(), Low_Value_gdf=None, check_plot=1):
    """
    In case low-value features were queried with get_features_OSM(),
    calculate the "counter-shape" representig high value area for a given bounding box.

    Input:
        - bbox: List of coordinates in format [South, West, North, East]
        - save_path: path for results
        - Low_Value_gdf: absolute path of gdf of low value items which is to be inverted.
          If left empty, searches for OSM_features_gdf_combined_lat_lon.shp in save_path.
        - checkplot
    Output:
        - Shapefile and GeoDataFrame of High Value Area as High_Value_Area_lat_lon
    Example:
        High_Value_gdf_47_8 = get_highValueArea([47.16, 8.0, 47.3, 8.0712], save_path = save_path,\
                                    Low_Value_gdf = save_path+'/Low_Value_gdf_combined_47_8.shp')
    important: Use same bbox and save_path as for get_features_OSM().
    """

    Outer_Poly = geometry.Polygon([(bbox[1],bbox[2]), \
                                   (bbox[1],bbox[0]), \
                                   (bbox[3],bbox[0]), \
                                   (bbox[3],bbox[2])])

    if Low_Value_gdf == None:
        try:
            Low_Value_gdf = geopandas.read_file(save_path \
                                                +'/OSM_features_gdf_combined_'+str(int(bbox[0]))+\
                                                '_'+str(int(bbox[1]))+'.shp')
        except:
            print('No Low-Value-Union found with name %s. \n Please add.' \
                  % (save_path +'/OSM_features_gdf_combined_'+str(int(bbox[0]))+'_'+\
                     str(int(bbox[1]))+'.shp'))
    else:
        Low_Value_gdf = geopandas.read_file(Low_Value_gdf)

    ###################
    # Making one Union
    ###################

    def makeUnion(gdf):
        """
        Solve issue of invalid geometries in MultiPolygons, which prevents that
        shapes can be combined into one unary union, save the respective Union
        """
        union1 = gdf[gdf.geometry.type == 'Polygon'].unary_union
        union2 = gdf[gdf.geometry.type != 'Polygon'].geometry.buffer(0).unary_union
        Low_Value_Union = unary_union([union1, union2])
        return Low_Value_Union

    Low_Value_Union = makeUnion(Low_Value_gdf)

    # subtract low-value areas from high-value polygon
    High_Value_Area = Outer_Poly.difference(Low_Value_Union)

    # save high value multipolygon as shapefile and re-read as gdf:
    schema = {'geometry': 'MultiPolygon', 'properties': {'Name':'str:80'}}
    shapeout = save_path + '/High_Value_Area_'+str(int(bbox[0]))+'_'+str(int(bbox[1]))+".shp"
    with fiona.open(shapeout, 'w', crs=from_epsg(4326), driver='ESRI Shapefile', \
                    schema=schema) as output:
        prop1 = {'Name':'High Value Area'}
        geom = mapping(High_Value_Area)
        output.write({'geometry': geom, 'properties': prop1})

    High_Value_Area = geopandas.read_file(shapeout)

    # plot
    if check_plot == 1:
        f, ax = plt.subplots(1)
        ax = High_Value_Area.plot(ax=ax)
        f.suptitle('High Value Area '+str(int(bbox[0]))+' '+str(int(bbox[1])))
        plt.show()
        f.savefig('High Value Area '+str(int(bbox[0]))+'_'+str(int(bbox[1]))+\
                  '.pdf', bbox_inches='tight')

    return High_Value_Area

def fill_highValueArea_LitPop(bbox, country, mode, highValueArea=None, \
                              save_path=os.getcwd(), check_plot=1):
    """
    Generate climada-compatiple entity by downloading values for a bounding box
    from LitPop, corrected for centroids only inside a certain high-value
    multipolygon area from OSM query.

    Input:
        - bbox: List of coordinates in format [South, West, North, East]
        - Country: ISO3 code or name of country in which bbox is located
        - highValueArea: path of gdf of high-value area from previous step.
          If empty, searches for cwd/OSM_features_gdf_combined_lat_lon.shp
        - mode: mode of re-assigning low-value points to high-value points.
          "nearest", "even", or "proportional"
    Returns:
        - entity (CLIMADA-compatible) with re-allocated asset values with name
          entity_high_lat_lon
    Example:
        entity_high_47_8 = fill_highValueArea_LitPop([47.16, 8.0, 47.3, 8.0712],\
                          'CHE',"proportional", highValueArea = \
                          save_path + '/High_Value_Area_47_8.shp' ,\
                          save_path = save_path)
    """

    # Load Country Exposure and High Value Area
    ent = LitPop()
    ent.set_country(country)
    ent.set_geometry_points()

    if highValueArea == None:
        try:
            High_Value_Area_gdf = geopandas.read_file(os.getcwd() + \
                                                      '/High_Value_Area_'+ \
                                                      str(int(bbox[0]))+'_'+ \
                                                      str(int(bbox[1]))+".shp")
        except:
            print('No file found of form %s. Please add or specify path.' \
                  %(os.getcwd() + 'High_Value_Area_'+str(int(bbox[0]))+'_'+\
                    str(int(bbox[1]))+".shp"))
    else:
        High_Value_Area_gdf = geopandas.read_file(highValueArea)

    # Crop bbox of High Value Area from Country Exposure
    ent_sub = ent.cx[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    # divide litpop assets points into within high-value area and outside high-value area
    ent_sub_high = pd.DataFrame(columns=ent_sub.columns)
    ent_sub_low = pd.DataFrame(columns=ent_sub.columns)
    for i, pt in enumerate(ent_sub.geometry):
        if pt.within(High_Value_Area_gdf.loc[0]['geometry']):
            ent_sub_high = ent_sub_high.append(ent_sub.iloc[i])
        else:
            ent_sub_low = ent_sub_low.append(ent_sub.iloc[i])

    ent_sub_high = GeoDataFrame(ent_sub_high, crs=ent_sub.crs, geometry=ent_sub_high.geometry)
    ent_sub_low = GeoDataFrame(ent_sub_low, crs=ent_sub.crs, geometry=ent_sub_low.geometry)

    if mode == "nearest":
        # assign asset values of low-value points to nearest point in high-value df.
        pointsToAssign = ent_sub_high.geometry.unary_union
        ent_sub_high["addedValNN"] = 0
        for i in range(0, len(ent_sub_low)):
            nearest = ent_sub_high.geometry == nearest_points(ent_sub_low.iloc[i].geometry, \
                                                              pointsToAssign)[1] #point
            ent_sub_high.addedValNN.loc[nearest] = ent_sub_low.iloc[i].value
        ent_sub_high["combinedValNN"] = ent_sub_high[['addedValNN', 'value']].sum(axis=1)
        ent_sub_high.rename(columns={'value': 'value_old', 'combinedValNN': 'value'},\
                            inplace=True)

    elif mode == "even":
        # assign asset values of low-value points evenly to points in high-value df.
        ent_sub_high['addedValeven'] = sum(ent_sub_low.value)/len(ent_sub_high)
        ent_sub_high["combinedValeven"] = ent_sub_high[['addedValeven', 'value']].sum(axis=1)
        ent_sub_high.rename(columns={'value': 'value_old', 'combinedValeven': 'value'},\
                            inplace=True)

    elif mode == "proportional":
        # assign asset values of low-value points proportionally to value of points in high-value df.
        ent_sub_high['addedValprop'] = 0
        for i in range(0, len(ent_sub_high)):
            asset_factor = ent_sub_high.iloc[i].value/sum(ent_sub_high.value)
            ent_sub_high.addedValprop.iloc[i] = asset_factor*sum(ent_sub_low.value)
        ent_sub_high["combinedValprop"] = ent_sub_high[['addedValprop', 'value']].sum(axis=1)
        ent_sub_high.rename(columns={'value': 'value_old', 'combinedValprop': 'value'},\
                            inplace=True)

    else:
        print("No proper re-assignment mode set. Please choose either nearest, even or proportional.")

    ###### how to "spread" centroids with value to e.g. hexagons? ###########

    # put back into CLIMADA-compatible entity format and save as hdf5 file:
    ent_sub_high_exp = Exposures(ent_sub_high)
    ent_sub_high_exp.set_lat_lon()
    ent_sub_high_exp.check()
    ent_sub_high_exp.write_hdf5(save_path + '/entity_high_'+str(int(bbox[0]))+\
                                '_'+str(int(bbox[1]))+'.h5')

    # plotting
    if check_plot == 1:
        # normal hexagons
        ent_sub_high_exp.plot_hexbin(pop_name=True)
        # select the OSM background image from the available ctx.sources
        fig, ax = ent_sub_high_exp.plot_basemap(buffer=30000, url=ctx.sources.OSM_C, cmap='brg')

    return ent_sub_high_exp

def fill_highValueArea_Houses(highValueArea, mode="default", country=None, \
                              save_path=os.getcwd(), check_plot=1):
    """
    Generate climada-compatiple entity by assigning values to midpoints of
    individual houses from OSM query, according to surface area and country.

    Input:
        - highValueArea: absolute path for gdf of building features queried
          from get_features_OSM()
        - mode: "LitPop" or "default": Default assigns a value of 5400 Chf to
          each m2 of building, LitPop assigns total LitPop value for the region
          proportionally to houses (by base area of house)
        - Country: ISO3 code or name of country in which entity is located.
          Only if mode = LitPop
    Returns:
        - ent_building (CLIMADA-compatible) with allocated asset values.
          Saved as entity_buildings_lat_lon.h5
    Example:
        buildings_47_8 = fill_highValueArea_Houses(save_path+ \
                                                  '/OSM_features_gdf_combined_47_8.shp',\
                                                  mode="default", save_path = \
                                                  save_path, check_plot=1)
    """

    High_Value_Area_gdf = geopandas.read_file(highValueArea)

    # For current entity structure, simply get centroid and area (in m2) for each building polygon
    High_Value_Area_gdf['projected_area'] = 0
    High_Value_Area_gdf['Midpoint'] = 0

    for index, row in High_Value_Area_gdf.iterrows():
        High_Value_Area_gdf.loc[index, "Midpoint"] = \
        High_Value_Area_gdf.loc[index,"geometry"].centroid.wkt
        s = shape(High_Value_Area_gdf.loc[index, "geometry"])
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                       pyproj.Proj(init='epsg:3857'))
        High_Value_Area_gdf.loc[index, "projected_area"] = transform(proj, s).area

    # change active geometry from polygons to midpoints
    from shapely.wkt import loads
    High_Value_Area_gdf = High_Value_Area_gdf.rename(columns={'geometry': 'geo_polys', \
                                                              'Midpoint':'geometry'})
    High_Value_Area_gdf['geometry'] = High_Value_Area_gdf['geometry'].apply(lambda x: loads(x))
    High_Value_Area_gdf = High_Value_Area_gdf.set_geometry('geometry')

    if mode == "LitPop":
        # assign LitPop values of this area to houses.
        ent = LitPop()
        ent.set_country(country)
        ent.set_geometry_points()
        ent_sub = ent.cx[min(High_Value_Area_gdf.bounds.minx):max(High_Value_Area_gdf.bounds.maxx),
                         min(High_Value_Area_gdf.bounds.miny):max(High_Value_Area_gdf.bounds.maxy)]
        totalValue = sum(ent_sub.value)
        totalArea = sum(High_Value_Area_gdf['projected_area'])
        High_Value_Area_gdf['value'] = 0
        for index, row in High_Value_Area_gdf.iterrows():
            High_Value_Area_gdf.loc[index, 'value'] = \
            High_Value_Area_gdf.loc[index, 'projected_area']/totalArea*totalValue

    elif mode == "default": # 5400 Chf / m2 base area
        High_Value_Area_gdf['value'] = 0
        for index, row in High_Value_Area_gdf.iterrows():
            High_Value_Area_gdf.loc[index, 'value'] = \
            High_Value_Area_gdf.loc[index, 'projected_area']*5400

    # put back into CLIMADA-compatible entity format and save as hdf5 file:
    ent_buildings = Exposures(High_Value_Area_gdf)
    ent_buildings.set_lat_lon()
    ent_buildings.check()
    ent_buildings.write_hdf5(save_path + '/entity_buildings_'+ \
                             str(int(min(High_Value_Area_gdf.bounds.miny)))+ \
                             '_'+str(int(min(High_Value_Area_gdf.bounds.minx)))+'.h5')

    # plotting
    if check_plot == 1:
        # normal hexagons
        ent_buildings.plot_hexbin(pop_name=True)
        # select the OSM background image from the available ctx.sources
        #fig, ax = ent_buildings.plot_basemap(buffer=30000, url=ctx.sources.OSM_C, cmap='brg')

    return ent_buildings
