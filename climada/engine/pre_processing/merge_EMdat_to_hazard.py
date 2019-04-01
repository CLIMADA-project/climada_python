"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

functions to merge EMDAT damages to hazard events
"""

import pandas as pd
import numpy as np
import pickle
from iso3166 import countries as iso_cntry
import climada
from datetime import datetime

# inputs
checkset = pd.read_csv('~.csv')
intensity_path = '~.p'
names_path = '~.p'
reg_ID_path = '~.p'
date_path = '~.p'
EMdat_raw = pd.read_excel('~.xlsx')
a = 'yyyy-mm-dd'
b = 'yyyy-mm-dd'

# assign hazard to EMdat event

data = assign_hazard_to_EMdat(certainty_level = 'low',intensity_path_haz = intensity_path,
                             names_path_haz = names_path, reg_ID_path_haz = reg_ID_path,
                             date_path_haz = date_path, EMdat_data = EMdat_raw,
                             start_time = a, end_time = b,keep_checks = True)
check_assigned_track(lookup = data, checkset = checkset)

###############################################################################

def assign_hazard_to_EMdat(certainty_level,intensity_path_haz, names_path_haz,
                           reg_ID_path_haz, date_path_haz, EMdat_data,
                           start_time, end_time, keep_checks = False):
    """assign_hazard_to_EMdat: link EMdat event to hazard

        Parameters:
            input files (paths):
                intensity: sparse matrix with hazards as rows and grid points as cols, values only at location with impacts
                names: identifier for each hazard (i.e. IBtracID) (rows of the matrix)
                reg_ID: ISO country ID of each grid point (cols of the matrix)
                date: start date of each hazard (rows of the matrix)
                EMdat_data: pd.dataframe with EMdat data
                start: start date of events to be assigned 'yyyy-mm-dd'
                end: end date of events to be assigned 'yyyy-mm-dd'
                disaster_subtype: EMdat disaster subtype
                
    Returns:
        pd.dataframe with EMdat entries linked to a hazard
    """
    # check valid certainty level
    certainty_levels = ['high', 'low']
    if certainty_level not in certainty_levels:
        raise ValueError("Invalid certainty level. Expected one of: %s" % certainty_levels)
    
    # prepare hazard set
    print("Start preparing hazard set")
    hit_countries = hit_country_per_hazard(intensity_path_haz, names_path_haz, reg_ID_path_haz, date_path_haz)
    # prepare damage set
    
    #### adjust EMdat_data to the path!!
    print("Start preparing damage set")
    lookup = create_lookup(EMdat_data, start_time, end_time, disaster_subtype = 'Tropical cyclone')
    # calculate possible hits
    print("Calculate possible hits")
    hit5 = EMdat_possible_hit(lookup = lookup, hit_countries = hit_countries, delta_t = 5)
    hit5_match = match_EM_ID(lookup = lookup, poss_hit = hit5)
    print("1/5")
    hit10 = EMdat_possible_hit(lookup = lookup, hit_countries = hit_countries, delta_t = 10)
    hit10_match = match_EM_ID(lookup = lookup, poss_hit = hit10)
    print("2/5")
    hit15 = EMdat_possible_hit(lookup = lookup, hit_countries = hit_countries, delta_t = 15)
    hit15_match = match_EM_ID(lookup = lookup, poss_hit = hit15)
    print("3/5")
    hit25 = EMdat_possible_hit(lookup = lookup, hit_countries = hit_countries, delta_t = 25)
    hit25_match = match_EM_ID(lookup = lookup, poss_hit = hit25)
    print("4/5")
    hit50 = EMdat_possible_hit(lookup = lookup, hit_countries = hit_countries, delta_t = 50)
    hit50_match = match_EM_ID(lookup = lookup, poss_hit = hit50)
    print("5/5")
    
    # assign only tracks with high certainty
    print("Assign tracks")
    if certainty_level == 'high':
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit10_match, possible_tracks_2 = hit50_match, level = 1)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit15_match, possible_tracks_2 = hit50_match, level = 2)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit25_match, possible_tracks_2 = hit50_match, level = 3)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit10_match, possible_tracks_2 = hit25_match, level = 4)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit15_match, possible_tracks_2 = hit25_match, level = 5)
    # assign all tracks
    elif certainty_level == 'low':
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit5_match, possible_tracks_2 = hit50_match, level = 1)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit10_match, possible_tracks_2 = hit50_match, level = 2)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit15_match, possible_tracks_2 = hit50_match, level = 3)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit5_match, possible_tracks_2 = hit25_match, level = 4)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit10_match, possible_tracks_2 = hit25_match, level = 5)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit15_match, possible_tracks_2 = hit25_match, level = 6)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit5_match, possible_tracks_2 = hit15_match, level = 7)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit10_match, possible_tracks_2 = hit15_match, level = 8)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit5_match, possible_tracks_2 = hit10_match, level = 9)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit15_match, possible_tracks_2 = hit15_match, level = 10)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit10_match, possible_tracks_2 = hit10_match, level = 11)
        lookup = assign_track_to_EM(lookup = lookup, possible_tracks_1 = hit5_match, possible_tracks_2 = hit5_match, level = 12)
    
    if keep_checks == False:
        lookup = lookup.drop(['Date_start_EM_ordinal','possible_track','possible_track_all'], axis = 1)
    lookup.groupby('allocation_level').count()
    print('(%d/%s) tracks allocated' %(len(lookup[lookup.allocation_level.notnull()]),len(lookup)))
    return lookup


    """hit_country_per_hazard: create list of hit countries from hazard set

        Parameters:
            input files:
                intensity: sparse matrix with hazards as rows and grid points as cols, values only at location with impacts
                names: identifier for each hazard (i.e. IBtracID) (rows of the matrix)
                reg_ID: ISO country ID of each grid point (cols of the matrix)
                date: start date of each hazard (rows of the matrix)
    Returns:
        pd.dataframe with all hit countries per hazard
    """


def hit_country_per_hazard(intensity_path, names_path, reg_ID_path, date_path):
    """hit_country_per_hazard: create list of hit countries from hazard set

        Parameters:
            input files:
                intensity: sparse matrix with hazards as rows and grid points as cols, values only at location with impacts
                names: identifier for each hazard (i.e. IBtracID) (rows of the matrix)
                reg_ID: ISO country ID of each grid point (cols of the matrix)
                date: start date of each hazard (rows of the matrix)
    Returns:
        pd.dataframe with all hit countries per hazard
    """
    with open(intensity_path, 'rb') as f:
        inten = pickle.load(f)
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    with open(reg_ID_path, 'rb') as f:
        reg_ID = pickle.load(f)
    with open(date_path, 'rb') as f:
        date = pickle.load(f)
        
    # loop over the tracks (over the rows of the intensity matrix)
    all_hits = []
    for track in range(0,len(names)):
        # select track 
        TC = inten[track,]
        # select only indices that are not zero
        hits = TC.nonzero()[1]
        
        # get the country of these indices and remove dublicates
        hits = list(set(reg_ID[hits]))
        
        # append hit countries to list
        all_hits.append(hits)
    
    # create data frame for output
    hit_countries = pd.DataFrame(columns = ['hit_country', 'Date_start', 'ibtracsID'])            
    for track in range(0,len(names)):
        #Check if track has hit any country else go to the next track
        if len(all_hits[track]) > 0:
            # loop over hit_country
            for hit in range(0,len(all_hits[track])):
                # Hit country ISO
                ctry_iso = iso_cntry.get(all_hits[track][hit]).alpha3
                
                # create entry for each country a hazard has hit
                hit_countries = hit_countries.append({'hit_country': ctry_iso,
                                                    'Date_start' : date[track],
                                                    'ibtracsID' : names[track]},
                                                    ignore_index = True) 
    
    # retrun data frame with all hit countries per hazard
    return hit_countries

def create_lookup(EMdat_data, start, end, disaster_subtype = 'Tropical cyclone'):
    """create_lookup: prepare a lookup table of EMdat events to which hazards can be assigned

        Parameters:
                EMdat_data: pd.dataframe with EMdat data
                start: start date of events to be assigned 'yyyy-mm-dd'
                end: end date of events to be assigned 'yyyy-mm-dd'
                disaster_subtype: EMdat disaster subtype
        Returns:
            pd.dataframe lookup
        """
    data = EMdat_data[EMdat_data['Disaster_subtype'] == disaster_subtype]
    lookup = pd.DataFrame(columns = ['hit_country','Date_start_EM','Date_start_EM_ordinal','Disaster_name','EM_ID', 'ibtracsID', 'allocation_level', 'possible_track', 'possible_track_all'])
    lookup.hit_country = data.ISO
    lookup.Date_start_EM = data.Date_start_clean
    lookup.Disaster_name = data.Disaster_name
    lookup.EM_ID = data.Disaster_No
    lookup = lookup.reset_index(drop = True)
    # create ordinals
    for i in range(0, len(data.Date_start_clean.values)): 
        lookup.Date_start_EM_ordinal[i] = datetime.toordinal(datetime.strptime(lookup.Date_start_EM.values[i],'%Y-%m-%d'))
        # ordinals to numeric
    lookup.Date_start_EM_ordinal = pd.to_numeric(lookup.Date_start_EM_ordinal)
    # select time
    EM_start =datetime.toordinal(datetime.strptime(start,'%Y-%m-%d'))
    EM_end = datetime.toordinal(datetime.strptime(end,'%Y-%m-%d'))

    lookup = lookup[lookup.Date_start_EM_ordinal.values > EM_start]
    lookup = lookup[lookup.Date_start_EM_ordinal.values < EM_end]
    
    return lookup


# Function to relate EM disaster to IBtrack using hit countries and time
def EMdat_possible_hit(lookup, hit_countries, delta_t):
    """relate EM disaster to hazard using hit countries and time

        Parameters:
            input files:
                lookup: pd.dataframe to relate EMdatID to hazard
                tracks: pd.dataframe with all hit countries per hazard
                delta_t: max time difference of start of EMdat event and hazard
                hit_countries: 
                start: start date of events to be assigned
                end: end date of events to be assigned
                disaster_subtype: EMdat disaster subtype
        Returns:
            list with possible hits
    """
    # lookup: PD dataframe that relates EMdatID to an IBtracsID
    # tracks: processed IBtracks with info which track hit which country
    # delta_t: time difference of start of EMdat and IBrtacks
    possible_hit_all = []
    for i in range(0,len(lookup.EM_ID.values)):
        possible_hit = []
        country_tracks = hit_countries[hit_countries['hit_country'] == lookup.hit_country.values[i]]
        for j in range(0, len(country_tracks.Date_start.values)):
            if (lookup.Date_start_EM_ordinal.values[i]-country_tracks.Date_start.values[j]) < delta_t and (lookup.Date_start_EM_ordinal.values[i]-country_tracks.Date_start.values[j]) >= 0:
                possible_hit.append(country_tracks.ibtracsID.values[j])
        possible_hit_all.append(possible_hit)                        

    return possible_hit_all


# function to check if EM_ID has been assigned already
def match_EM_ID(lookup, poss_hit):
    """function to check if EM_ID has been assigned already and combine possible hits

        Parameters:
            lookup: pd.dataframe to relate EMdatID to hazard
            poss_hit: list with possible hits

        Returns:
            list with all possible hits per EMdat ID
        """
    possible_hit_all = []
    for i in range(0,len(lookup.EM_ID.values)):
        possible_hit = []
        # lookup without line i
        #lookup_match = lookup.drop(i)
        lookup_match = lookup
        # Loop over check if EM dat ID is the same
        for i_match in range(0,len(lookup_match.EM_ID.values)):
            if lookup.EM_ID.values[i] == lookup_match.EM_ID.values[i_match]:
                possible_hit.append(poss_hit[i])
        possible_hit_all.append(possible_hit)                        
    return possible_hit_all


def assign_track_to_EM(lookup, possible_tracks_1, possible_tracks_2, level):
    """function to assign a hazard to an EMdat event
        to get some confidene into the procedure, hazards get only assigned
        if there is no other hazard occuring at a bigger time interval in that country
        Thus a track of possible_tracks_1 gets only assigned if there are no other
        tracks in possible_tracks_2.
        The confidence can be expressed with a certainty level
        

        Parameters:
            lookup: pd.dataframe to relate EMdatID to hazard
            possible_tracks_1: list of possible hits with smaller time horizon
            possible_tracks_2: list of possible hits with larger time horizon
            level: level of confidence
        Returns:
            pd.dataframe lookup with assigend tracks and possible hits
    """
    
    for i in range(0, len(possible_tracks_1)):
        if np.isnan(lookup.allocation_level.values[i]):
            number_EMdat_id = len(possible_tracks_1[i])
            #print(number_EMdat_id)
            for j in range(0,number_EMdat_id):
                # check that number of possible track stays the same at given time difference and that list is not empty
                if len(possible_tracks_1[i][j]) == len(possible_tracks_2[i][j]) == 1 and possible_tracks_1[i][j] != []:
                    # check that all tracks are the same
                    if all(possible_tracks_1[i][0] == possible_tracks_1[i][k] for k in range(0,len(possible_tracks_1[i]))) == True:
                        #check that track ID has not been assigned to that country already
                        ctry_lookup = lookup[lookup['hit_country'] == lookup.hit_country.values[i]]
                        if possible_tracks_1[i][0][0] not in ctry_lookup.ibtracsID.values:
                            lookup.ibtracsID.values[i] = possible_tracks_1[i][0][0]
                            lookup.allocation_level.values[i] = level
                elif possible_tracks_1[i][j] != []:
                    lookup.possible_track.values[i] = possible_tracks_1[i]
        else: lookup.possible_track_all.values[i] = possible_tracks_1[i]
    return lookup

def check_assigned_track(lookup, checkset):
    """compare lookup with assigned tracks to a set with checked sets
        

        Parameters:
            lookup: pd.dataframe to relate EMdatID to hazard
            checkset: pd.dataframe with already checked hazards
        Returns:
            error scores
    """    
    # merge checkset and lookup
    check = pd.merge(checkset, lookup[['hit_country', 'EM_ID','ibtracsID']], on = ['hit_country', 'EM_ID'])
    check_size = len(check.ibtracsID.values)
    # not assigned values
    not_assigned = check.ibtracsID.isnull().sum(axis = 0)
    # correct assigned values
    correct = sum(check.ibtracsID.values == check.IBtracsID_checked.values)
    # wrongly assigned values
    wrong = len(check.ibtracsID.values)-not_assigned-correct
    print('%.1f%% tracks assigned correctly, %.1f%% wrongly, %.1f%% not assigned' %(correct/check_size*100,wrong/check_size*100,not_assigned/check_size*100))

