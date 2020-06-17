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
import logging
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
from iso3166 import countries as iso_cntry
from cartopy.io import shapereader
import shapefile
# import climada

from climada.util.finance import gdp
from climada.util.constants import DEF_CRS
from climada.engine import Impact
from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHaz

LOGGER = logging.getLogger(__name__)

PERIL_SUBTYPE_MATCH_DICT = dict(TC='Tropical cyclone',
                                T1='Storm',
                                TS='Coastal flood',
                                EQ='Ground movement',
                                E1='Earthquake',
                                RF='Riverine flood',
                                F1='Flood',
                                F2='Flash flood',
                                WS='Extra-tropical storm',
                                W1='Storm',
                                DR='Drought',
                                LS='Landslide',
                                FF='Forest fire',
                                FW='Wildfire',
                                FB='Land fire (Brush, Bush, Pastur')

PERIL_TYPE_MATCH_DICT = dict(DR='Drought',
                             TC='Storm',
                             EQ='Earthquake',
                             FL='Flood',
                             LS='Landslide',
                             WS='Storm',
                             VQ='Volcanic activity',
                             BF='Wildfire',
                             HW='Extreme temperature')

varnames_emdat = dict()
varnames_emdat[2018] = ['Start date', 'End date', 'Country', 'ISO', 'Location', 'Latitude', # 0-5
       'Longitude', 'Magnitude value', 'Magnitude scale', 'Disaster type', # 6-9
       'Disaster subtype', 'Associated disaster', 'Associated disaster2', # 10-12
       'Total deaths', 'Total affected', "Total damage ('000 US$)", # 13-15
       "Insured losses ('000 US$)", 'Disaster name', 'Disaster No.'] # 16-18

varnames_emdat[2020] = ['Dis No', 'Year', 'Seq', 'Disaster Group', 'Disaster Subgroup', # 0-4
       'Disaster Type', 'Disaster Subtype', 'Disaster Subsubtype', # 5-7
       'Event Name', 'Entry Criteria', 'Country', 'ISO', 'Region', 'Continent', # 8-13
       'Location', 'Origin', 'Associated Dis', 'Associated Dis2', # 14-17
       'OFDA Response', 'Appeal', 'Declaration', 'Aid Contribution', # 18-21
       'Dis Mag Value', 'Dis Mag Scale', 'Latitude', 'Longitude', 'Local Time', # 22-26
       'River Basin', 'Start Year', 'Start Month', 'Start Day', 'End Year', # 27-31
       'End Month', 'End Day', 'Total Deaths', 'No Injured', 'No Affected', # 32-36
       'No Homeless', 'Total Affected', "Reconstruction Costs ('000 US$)", # 37-39
       "Insured Damages ('000 US$)", "Total Damages ('000 US$)", 'CPI'] # 40-41

varnames_mapping = dict()
varnames_mapping['2018_2020'] = [18, -18, -99, -99, -99, # 0-4
       9, 10, -99, # 5-7
       17, -99, 2, 3, -99, -99, # 8-13
       4, -99, 11, 12, # 14-17
       -99, -99, -99, -99, # 18-21
       7, 8, 5, 6, -99, # 22-26
       -99, -18, -99, -99, -99, # 27-31
       -99, -99, 13, -99, -99, # 32-36
       -99, 14,-99, # 37-39
       16, 15, -99] # 40-41

VARNAMES = dict()
VARNAMES[2018] = dict()
VARNAMES[2020] = dict()
for idx, var in enumerate(varnames_emdat[2020]):
    if varnames_mapping['2018_2020'][idx]>=0:
        VARNAMES[2018][var] = varnames_emdat[2018][varnames_mapping['2018_2020'][idx]]
    VARNAMES[2020][var] = var

def assign_hazard_to_EMdat(certainty_level, intensity_path_haz, names_path_haz,
                           reg_ID_path_haz, date_path_haz, EMdat_data,
                           start_time, end_time, keep_checks=False):
    """assign_hazard_to_EMdat: link EMdat event to hazard
        Parameters:
            input files (paths):
                intensity: sparse matrix with hazards as rows and grid points as cols,
                values only at location with impacts
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
    hit_countries = hit_country_per_hazard(intensity_path_haz, names_path_haz, \
                                           reg_ID_path_haz, date_path_haz)
    # prepare damage set

    #### adjust EMdat_data to the path!!
    print("Start preparing damage set")
    lookup = create_lookup(EMdat_data, start_time, end_time, disaster_subtype='Tropical cyclone')
    # calculate possible hits
    print("Calculate possible hits")
    hit5 = EMdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=5)
    hit5_match = match_EM_ID(lookup=lookup, poss_hit=hit5)
    print("1/5")
    hit10 = EMdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=10)
    hit10_match = match_EM_ID(lookup=lookup, poss_hit=hit10)
    print("2/5")
    hit15 = EMdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=15)
    hit15_match = match_EM_ID(lookup=lookup, poss_hit=hit15)
    print("3/5")
    hit25 = EMdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=25)
    hit25_match = match_EM_ID(lookup=lookup, poss_hit=hit25)
    print("4/5")
    hit50 = EMdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=50)
    hit50_match = match_EM_ID(lookup=lookup, poss_hit=hit50)
    print("5/5")

    # assign only tracks with high certainty
    print("Assign tracks")
    if certainty_level == 'high':
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit50_match, level=1)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit50_match, level=2)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit25_match,
                                    possible_tracks_2=hit50_match, level=3)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit25_match, level=4)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit25_match, level=5)
    # assign all tracks
    elif certainty_level == 'low':
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit50_match, level=1)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit50_match, level=2)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit50_match, level=3)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit25_match, level=4)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit25_match, level=5)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit25_match, level=6)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit15_match, level=7)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit15_match, level=8)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit10_match, level=9)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit15_match, level=10)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit10_match, level=11)
        lookup = assign_track_to_EM(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit5_match, level=12)
    if keep_checks == False:
        lookup = lookup.drop(['Date_start_EM_ordinal', 'possible_track', \
                              'possible_track_all'], axis=1)
    lookup.groupby('allocation_level').count()
    print('(%d/%s) tracks allocated' %(len(lookup[lookup.allocation_level.notnull()]), len(lookup)))
    return lookup


def hit_country_per_hazard(intensity_path, names_path, reg_ID_path, date_path):
    """hit_country_per_hazard: create list of hit countries from hazard set

        Parameters:
            input files:
                intensity: sparse matrix with hazards as rows and grid points
                as cols, values only at location with impacts
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
    for track in range(0, len(names)):
        # select track
        TC = inten[track,]
        # select only indices that are not zero
        hits = TC.nonzero()[1]
        # get the country of these indices and remove dublicates
        hits = list(set(reg_ID[hits]))
        # append hit countries to list
        all_hits.append(hits)

    # create data frame for output
    hit_countries = pd.DataFrame(columns=['hit_country', 'Date_start', 'ibtracsID'])
    for track in range(0, len(names)):
        #Check if track has hit any country else go to the next track
        if len(all_hits[track]) > 0:
            # loop over hit_country
            for hit in range(0, len(all_hits[track])):
                # Hit country ISO
                ctry_iso = iso_cntry.get(all_hits[track][hit]).alpha3
                # create entry for each country a hazard has hit
                hit_countries = hit_countries.append({'hit_country': ctry_iso,
                                                      'Date_start' : date[track],
                                                      'ibtracsID' : names[track]},
                                                     ignore_index=True)
    # retrun data frame with all hit countries per hazard
    return hit_countries

def create_lookup(EMdat_data, start, end, disaster_subtype='Tropical cyclone'):
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
    lookup = pd.DataFrame(columns=['hit_country', 'Date_start_EM', \
                                     'Date_start_EM_ordinal', 'Disaster_name', \
                                     'EM_ID', 'ibtracsID', 'allocation_level', \
                                     'possible_track', 'possible_track_all'])
    lookup.hit_country = data.ISO
    lookup.Date_start_EM = data.Date_start_clean
    lookup.Disaster_name = data.Disaster_name
    lookup.EM_ID = data.Disaster_No
    lookup = lookup.reset_index(drop=True)
    # create ordinals
    for i in range(0, len(data.Date_start_clean.values)):
        lookup.Date_start_EM_ordinal[i] = datetime.toordinal(datetime.strptime(lookup.Date_start_EM.values[i], '%Y-%m-%d'))
        # ordinals to numeric
    lookup.Date_start_EM_ordinal = pd.to_numeric(lookup.Date_start_EM_ordinal)
    # select time
    EM_start = datetime.toordinal(datetime.strptime(start, '%Y-%m-%d'))
    EM_end = datetime.toordinal(datetime.strptime(end, '%Y-%m-%d'))

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
    for i in range(0, len(lookup.EM_ID.values)):
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
    for i in range(0, len(lookup.EM_ID.values)):
        possible_hit = []
        # lookup without line i
        #lookup_match = lookup.drop(i)
        lookup_match = lookup
        # Loop over check if EM dat ID is the same
        for i_match in range(0, len(lookup_match.EM_ID.values)):
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
            for j in range(0, number_EMdat_id):
                # check that number of possible track stays the same at given
                # time difference and that list is not empty
                if len(possible_tracks_1[i][j]) == len(possible_tracks_2[i][j]) == 1 and possible_tracks_1[i][j] != []:
                    # check that all tracks are the same
                    if all(possible_tracks_1[i][0] == possible_tracks_1[i][k] for k in range(0, len(possible_tracks_1[i]))):
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
    check = pd.merge(checkset, lookup[['hit_country', 'EM_ID', 'ibtracsID']],\
                     on=['hit_country', 'EM_ID'])
    check_size = len(check.ibtracsID.values)
    # not assigned values
    not_assigned = check.ibtracsID.isnull().sum(axis=0)
    # correct assigned values
    correct = sum(check.ibtracsID.values == check.IBtracsID_checked.values)
    # wrongly assigned values
    wrong = len(check.ibtracsID.values)-not_assigned-correct
    print('%.1f%% tracks assigned correctly, %.1f%% wrongly, %.1f%% not assigned' \
          %(correct/check_size*100, wrong/check_size*100, not_assigned/check_size*100))


def _check_emdat_df(df_emdat, target_version=2020, varnames_emdat=varnames_emdat, \
                   varnames_mapping=varnames_mapping):
    """Check EM-DAT dataframe from CSV and update variable names if required.
    
    Parameters:
        Input:
            df_emdat: pandas dataframe loaded from EM-DAT CSV
            target_version (int): target format version. default: 2020
            varnames_emdat: dict with list of variable names in different
                versions of EM-DAT
            varnames_mapping: dict of strings, mapping between varnames_emdat"""

    if not (target_version in varnames_emdat.keys()):
        raise NotImplementedError('EM-DAT Version or column names not found in varnames_emdat!')
    if len(df_emdat.columns)==len(varnames_emdat[target_version]) and \
        min(df_emdat.columns==varnames_emdat[target_version]):
        for var in ['Disaster Subtype', 'Disaster Type', 'Country']:
            df_emdat[VARNAMES[target_version][var]].fillna('None', inplace=True)
        return df_emdat
    else:
        for version in list(varnames_emdat.keys()):
            if version<2019:
                df_emdat['ISO'].replace('', np.nan, inplace=True)
                df_emdat['ISO'].replace(' Belgium"',np.nan, inplace=True)
                df_emdat.dropna(subset=['ISO'], inplace=True)
            if min(df_emdat.columns==varnames_emdat[version]):
                df = pd.DataFrame(index=df_emdat.index.values, columns=varnames_emdat[target_version])
                for idc, col in enumerate(df.columns):
                    if varnames_mapping['%i_%i' % (version, target_version)][idc]>=0:
                        df[col] = df_emdat[varnames_emdat[version]\
                                           [varnames_mapping['%i_%i' % (version, target_version)][idc]]]
                    elif varnames_mapping['%i_%i' % (version, target_version)][idc]==-18 \
                        and version==2018:
                        years_list = list()
                        for _, disaster_no in enumerate(df_emdat['Disaster No.']):
                            if isinstance(disaster_no, str):
                                years_list.append(int(disaster_no[0:4]))
                            else:
                                years_list.append(np.nan)
                        df[col] = years_list
                if version<=2018 and target_version>=2020:
                    date_list = list()
                    year_list = list()
                    month_list = list()
                    day_list = list()
                    for year in list(df['Start Year']):
                        date_list.append(datetime.strptime(str(year), '%Y'))
                    boolean_warning = True
                    for idx, datestr in enumerate(list(df_emdat['Start date'])):
                        try:
                            date_list[idx] = datetime.strptime(datestr[-7:], '%m/%Y')
                        except ValueError:
                            if boolean_warning:
                                LOGGER.warning('EM_DAT CSV contains invalid time formats')
                                boolean_warning = False
                        try:
                            date_list[idx] = datetime.strptime(datestr, '%d/%m/%Y')
                        except ValueError:
                            if boolean_warning:
                                LOGGER.warning('EM_DAT CSV contains invalid time formats')
                                boolean_warning = False
                        day_list.append(date_list[idx].day)
                        month_list.append(date_list[idx].month)
                        year_list.append(date_list[idx].year)
                    df['Start Month'] = np.array(month_list, dtype='int')
                    df['Start Day'] = np.array(day_list, dtype='int')
                    df['Start Year'] = np.array(year_list, dtype='int')
                    for var in ['Disaster Subtype', 'Disaster Type', 'Country']:
                        df[VARNAMES[target_version][var]].fillna('None', inplace=True)
                return df
    raise NotImplementedError('EM-DAT Version or column names not found in varnames_emdat!')
    return None


def emdat_countries_by_hazard(hazard_name, emdat_file_csv, ignore_missing=True, \
                              verbose=True, year_range=None, target_version=2020):
    """return list of all countries exposed to a chosen hazard type
    from EMDAT data as CSV.

    Parameters:
        hazard_name (str): Disaster (sub-)type accordung EMDAT terminology, i.e.:
            Animal accident, Drought, Earthquake, Epidemic, Extreme temperature,
            Flood, Fog, Impact, Insect infestation, Landslide, Mass movement (dry),
            Storm, Volcanic activity, Wildfire;
            Coastal Flooding, Convective Storm, Riverine Flood, Tropical cyclone,
            Tsunami, etc.
        emdat_file_csv (str): Full path to EMDAT-file (CSV), i.e.:
            emdat_file_csv = os.path.join(SYSTEM_DIR, 'emdat_201810.csv')
        ignore_missing (boolean): Ignore countries that that exist in EMDAT but
            are missing in iso_cntry(). Default: True.
        verbose (boolean): silent mode
        year_range (tuple of integers or None): range of years to consider, i.e. (1950, 2000)
            default is None, i.e. consider all years
    Returns:
        exp_iso: List of ISO3-codes of countries impacted by the disaster type
        exp_name: List of names of countries impacted by the disaster type
            """
    if hazard_name in PERIL_SUBTYPE_MATCH_DICT.keys():
        hazard_name = PERIL_SUBTYPE_MATCH_DICT[hazard_name]
    elif hazard_name in PERIL_TYPE_MATCH_DICT.keys():
        hazard_name = PERIL_TYPE_MATCH_DICT[hazard_name]
        LOGGER.debug('Used "Disaster Type" instead of "Disaster Subtype" for matching hazard_name.')

    out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=0)
    counter = 0
    while not ('Country' in out.columns and 'ISO' in out.columns):
        counter += 1
        out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=counter)
        if counter==10: break
    del counter

    out = _check_emdat_df(out, target_version=target_version)

    if not not year_range: # if year range is given, extract years in range
        year_boolean = []
        all_years = np.arange(min(year_range), max(year_range)+1, 1)
        if target_version<=2018:
            for _, disaster_no in enumerate(out[VARNAMES[target_version]['Dis No']]):
                if isinstance(disaster_no, str) and int(disaster_no[0:4]) in all_years:
                    year_boolean.append(True)
                else:
                    year_boolean.append(False)
        else:
            for _, year in enumerate(out[VARNAMES[target_version]['Year']]):
                if int(year) in all_years:
                    year_boolean.append(True)
                else:
                    year_boolean.append(False)
        out = out[year_boolean]

    # List of countries that exist in EMDAT but are missing in iso_cntry():
    #(these countries are ignored)
    list_miss = ['Netherlands Antilles', 'Guadeloupe', 'Martinique', \
                 'Réunion', 'Tokelau', 'Azores Islands', 'Canary Is']
    # list_miss_iso = ['ANT', 'GLP', 'MTQ', 'REU', 'TKL', '', '']
    exp_iso = []
    exp_name = []
    shp_file = shapereader.natural_earth(resolution='10m',
                                         category='cultural',
                                         name='admin_0_countries')
    shp_file = shapereader.Reader(shp_file)

    # countries with TCs:
    if not out[out[VARNAMES[target_version]['Disaster Subtype']] == hazard_name].empty:
        uni_cntry = np.unique(out[out[VARNAMES[target_version]['Disaster Subtype']] == hazard_name][VARNAMES[target_version]['Country']].values)
    elif not out[out[VARNAMES[target_version]['Disaster Type']] == hazard_name].empty:
        uni_cntry = np.unique(out[out[VARNAMES[target_version]['Disaster Type']] == hazard_name][VARNAMES[target_version]['Country']].values)
    else:
        LOGGER.error('Disaster (sub-)type not found.')
    for cntry in uni_cntry:
        if (cntry in list_miss) and not ignore_missing:
            LOGGER.debug(cntry, '... not in iso_cntry')
            exp_iso.append('ZZZ')
            exp_name.append(cntry)
        elif cntry not in list_miss:
            if '(the)' in cntry:
                cntry = cntry.strip('(the)').rstrip()
            cntry = cntry.replace(' (the', ',').replace(')', '')
            cntry = cntry.replace(' (', ', ').replace(')', '')
            if cntry == 'Saint Barth?lemy' or cntry=='Saint BarthÃ©lemy':
                cntry = 'Saint Barthélemy'
            if cntry == 'Saint Martin, French Part':
                cntry = 'Saint Martin (French part)'
            if cntry == 'Sint Maarten, Dutch part':
                cntry = 'Sint Maarten (Dutch part)'
            if cntry == 'Swaziland':
                cntry = 'Eswatini'
            if cntry == 'Virgin Island, British':
                cntry = 'Virgin Islands, British'
            if cntry == 'Virgin Island, U.S.':
                cntry = 'Virgin Islands, U.S.'
            if cntry == 'Côte d\x92Ivoire':
                cntry = "Côte d'Ivoire"
            if cntry == 'Macedonia, former Yugoslav Republic of':
                cntry = 'Macedonia, the former Yugoslav Republic of'
            if cntry == 'RÃ©union':
                cntry = 'Réunion'
            if not verbose:
                LOGGER.debug(cntry, ':', iso_cntry.get(cntry).name)
            exp_iso.append(iso_cntry.get(cntry).alpha3)
            exp_name.append(iso_cntry.get(cntry).name)
    return exp_iso, exp_name

def emdat_df_load(country, hazard_name, emdat_file_csv, year_range=None, \
                  target_version=2020):
    """function to load EM-DAT data by country, hazard type and year range

    Parameters:
        country (list of str): country ISO3-codes or names, i.e. ['JAM'].
            set None or 'all' for all countries"""


    # Mapping of hazard type between EM-DAT and CLIMADA:
    if hazard_name in PERIL_SUBTYPE_MATCH_DICT.keys():
        hazard_name = PERIL_SUBTYPE_MATCH_DICT[hazard_name]
    elif hazard_name in PERIL_TYPE_MATCH_DICT.keys():
        hazard_name = PERIL_TYPE_MATCH_DICT[hazard_name]
        LOGGER.debug('Used "Disaster Type" instead of "Disaster Subtype" for matching hazard_name.')

    # Read CSV file with raw EMDAT data:
    out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=0)
    counter = 0
    while not ('Country' in out.columns and 'ISO' in out.columns):
        counter += 1
        out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=counter)
        if counter==10: break
    del counter
    out = _check_emdat_df(out, target_version=target_version)
    # Clean data frame from footer in original EM-DAT CSV:

    # Reduce data to country and hazard type selected:
    if not country or country == 'all':
        out[VARNAMES[target_version]['Disaster Type']].replace('', np.nan, inplace=True)
        out.dropna(subset=[VARNAMES[target_version]['Disaster Type']], inplace=True)
        out[VARNAMES[target_version]['Disaster Subtype']].replace('', np.nan, inplace=True)
        out.dropna(subset=[VARNAMES[target_version]['Disaster Subtype']], inplace=True)
    else:
        exp_iso, exp_name = emdat_countries_by_hazard(hazard_name, emdat_file_csv, \
                                                      target_version=target_version)
        if isinstance(country, int) | (not isinstance(country, str)):
            country = iso_cntry.get(country).alpha3
        if country in exp_name:
            country = exp_iso[exp_name.index(country)]
        if (country not in exp_iso) or country not in out.ISO.values:
            print('Country ' + country + ' not in EM-DAT for hazard ' + hazard_name)
            return None, None, country
        out = out[out[VARNAMES[target_version]['ISO']].str.contains(country)]
    out_ = out[out[VARNAMES[target_version]['Disaster Subtype']].str.contains(hazard_name)]
    out_ = out_.append(out[out[VARNAMES[target_version]['Disaster Type']].str.contains(hazard_name)])
    del out
    # filter by years and return output:
    year_boolean = []

    if not not year_range: # if year range is given, extract years in range
        year_boolean = []
        all_years = np.arange(min(year_range), max(year_range)+1, 1)
        if target_version<=2018:
            for _, disaster_no in enumerate(out_[VARNAMES[target_version]['Dis No']]):
                if isinstance(disaster_no, str) and int(disaster_no[0:4]) in all_years:
                    year_boolean.append(True)
                else:
                    year_boolean.append(False)
        else:
            for _, year in enumerate(out_[VARNAMES[target_version]['Year']]):
                if int(year) in all_years:
                    year_boolean.append(True)
                else:
                    year_boolean.append(False)
        out_ = out_[year_boolean]

    elif target_version<=2018:
        years = list()
        for _, disaster_no in enumerate(out_[VARNAMES[target_version]['Dis No']]):
            years.append(int(disaster_no[0:4]))
        all_years = np.arange(np.unique(years).min(), np.unique(years).max()+1, 1)
        del years
    else:
        all_years = np.arange(out_.Year.min(), out_.Year.max(), dtype='int')
    out_ = out_[out_[VARNAMES[target_version]['Dis No']].str.contains(str())]
    out_ = out_.reset_index(drop=True)
    return out_, sorted(all_years), country

def emdat_impact_yearlysum(countries, hazard_name, emdat_file_csv, year_range=None, \
                         reference_year=0, imp_str="Total Damages ('000 US$)", \
                         target_version=2020):
    """function to load EM-DAT data and sum impact per year
    Parameters:
        countries (list of str): country ISO3-codes or names, i.e. ['JAM'].
        hazard_name (str): Hazard name according to EMDAT terminology or
            CLIMADA abbreviation
        emdat_file_csv (str): Full path to EMDAT-file (CSV), i.e.:
            emdat_file_csv = os.path.join(SYSTEM_DIR, 'emdat_201810.csv')
        reference_year (int): reference year of exposures. Impact is scaled
            proportional to GDP to the value of the reference year. No scaling
            for 0 (default)
        imp_str (str): Column name of impact metric in EMDAT CSV,
            default = "Total Damages ('000 US$)"

    Returns:
        yearly_impact (dict, mapping years to impact):
            total impact per year, same unit as chosen impact,
            i.e. 1000 current US$ for imp_str="Total Damages ('000 US$)".
        all_years (list of int): list of years
    """
    imp_str = VARNAMES[target_version][imp_str]
    out = pd.DataFrame()
    for country in countries:
        data, all_years, country = emdat_df_load(country, hazard_name, \
                                   emdat_file_csv, year_range, target_version=target_version)
        if data is None:
            continue
        data_out = pd.DataFrame(index=np.arange(0, len(all_years)), \
                                columns=['ISO3', 'region_id', 'year', 'impact', \
                                'reference_year', 'impact_scaled'])
        if reference_year > 0:
            gdp_ref = gdp(country, reference_year)[1]
        for cnt, year in enumerate(all_years):
            data_out.loc[cnt, 'year'] = year
            data_out.loc[cnt, 'reference_year'] = reference_year
            data_out.loc[cnt, 'ISO3'] = country
            data_out.loc[cnt, 'region_id'] = int(iso_cntry.get(country).numeric)
            data_out.loc[cnt, 'impact'] = \
                np.nansum(data.loc[data[VARNAMES[target_version]['Dis No']].str.contains(str(year))]\
                             [imp_str])
            if '000 US' in imp_str: # EM-DAT damages provided in '000 USD
                data_out.loc[cnt, 'impact'] = data_out.loc[cnt, 'impact']*1000
            if reference_year > 0:
                data_out.loc[cnt, 'impact_scaled'] = data_out.loc[cnt, 'impact'] * \
                gdp_ref / gdp(country, year)[1]
        out = out.append(data_out)
    out = out.reset_index(drop=True)
    return out
    # out.loc[out['Year']==1980]['impact'].sum() < sum for year 1980

def emdat_impact_event(countries, hazard_name, emdat_file_csv, year_range, \
                       reference_year=0, imp_str="Total Damages ('000 US$)", \
                    target_version=2020):
    """function to load EM-DAT data return impact per event

    Parameters:
        countries (list of str): country ISO3-codes or names, i.e. ['JAM'].
        hazard_name (str): Hazard name according to EMDAT terminology or
            CLIMADA abbreviation, i.e. 'TC'
        emdat_file_csv (str): Full path to EMDAT-file (CSV), i.e.:
            emdat_file_csv = os.path.join(SYSTEM_DIR, 'emdat_201810.csv')
        reference_year (int): reference year of exposures. Impact is scaled
            proportional to GDP to the value of the reference year. No scaling
            for 0 (default)
        imp_str (str): Column name of impact metric in EMDAT CSV,
            default = "Total Damages ('000 US$)"

    Returns:
        out (pandas DataFrame): EMDAT DataFrame with new columns "year",
            "region_id", and scaled total impact per event with
            same unit as chosen impact,
            i.e. 1000 current US$ for imp_str="Total Damages ('000 US$) scaled".
    """
    imp_str = VARNAMES[target_version][imp_str]
    out = pd.DataFrame()
    for country in countries:
        data, _, country = emdat_df_load(country, hazard_name, \
                           emdat_file_csv, year_range, target_version=target_version)
        if data is None:
            continue
        if reference_year > 0:
            gdp_ref = gdp(country, reference_year)[1]
        else: gdp_ref = 0
        if not 'Year' in data.columns:
            data['Year'] = pd.Series(np.zeros(data.shape[0], dtype='int'), \
                index=data.index)
        data['region_id'] = pd.Series(int(iso_cntry.get(country).numeric) + \
            np.zeros(data.shape[0], dtype='int'), \
            index=data.index)
        data['reference_year'] = pd.Series(reference_year+np.zeros(\
            data.shape[0], dtype='int'), index=data.index)
        data[imp_str + " scaled"] = pd.Series(np.zeros(data.shape[0], dtype='int'), \
            index=data.index)
        for cnt in np.arange(data.shape[0]):
            if target_version<=2018:
                data.loc[cnt, 'Year'] = int(data.loc[cnt, VARNAMES[target_version]['Dis No']][0:4])
            data.loc[cnt, 'reference_year'] = int(reference_year)
            if data.loc[cnt][imp_str] > 0 and gdp_ref > 0:
                data.loc[cnt, imp_str + " scaled"] = \
                    data.loc[cnt, imp_str] * gdp_ref / \
                    gdp(country, int(data.loc[cnt, 'Year']))[1]
        out = out.append(data)
        del data
    out = out.reset_index(drop=True)
    if '000 US' in imp_str and not out.empty: # EM-DAT damages provided in '000 USD
        out[imp_str + " scaled"] = out[imp_str + " scaled"]*1e3
        out[imp_str] = out[imp_str]*1e3
    return out

def emdat_to_impact(emdat_file_csv, year_range=None, countries=None,\
                    hazard_type_emdat=None, hazard_type_climada=None, \
                    reference_year=0, imp_str="Total Damages ('000 US$)", \
                    target_version=2020):
    """function to load EM-DAT data return impact per event

    Parameters:
        emdat_file_csv (str): Full path to EMDAT-file (CSV), i.e.:
            emdat_file_csv = os.path.join(SYSTEM_DIR, 'emdat_201810.csv')

        hazard_type_emdat (str): Hazard (sub-)type according to EMDAT terminology,
            i.e. 'Tropical cyclone' for tropical cyclone
        OR
        hazard_type_climada (str): Hazard type CLIMADA abbreviation,
            i.e. 'TC' for tropical cyclone
    Optional parameters:
        year_range (list with 2 integers): start and end year i.e. [1980, 2017]
            default: None --> take year range from EM-DAT file
        countries (list of str): country ISO3-codes or names, i.e. ['JAM'].
            Set to None or ['all'] for all countries (default)

        reference_year (int): reference year of exposures. Impact is scaled
            proportional to GDP to the value of the reference year. No scaling
            for reference_year=0 (default)
        imp_str (str): Column name of impact metric in EMDAT CSV,
            default = "Total Damages ('000 US$)"

    Returns:
        impact_instance (instance of climada.engine.Impact):
            impact object of same format as output from CLIMADA
            impact computation
            scaled with GDP to reference_year if reference_year noit equal 0
            i.e. 1000 current US$ for imp_str="Total Damages ('000 US$) scaled".
            impact_instance.eai_exp holds expected annual impact for each country.
            impact_instance.coord_exp holds rough central coordinates for each country.
        countries (list): ISO3-codes of countries imn same order as in impact_instance.eai_exp
    """
    imp_str = VARNAMES[target_version][imp_str]
    # Mapping of hazard type between EM-DAT and CLIMADA:
    if not hazard_type_climada:
        if not hazard_type_emdat:
            LOGGER.error('Either hazard_type_climada or hazard_type_emdat need to be defined.')
            return None
        if hazard_type_emdat in PERIL_SUBTYPE_MATCH_DICT.values():
            hazard_type_climada = list(PERIL_SUBTYPE_MATCH_DICT.keys())[list(PERIL_TYPE_MATCH_DICT.values()).index(hazard_type_emdat)]
        elif hazard_type_emdat in PERIL_TYPE_MATCH_DICT.values():
            hazard_type_climada = list(PERIL_TYPE_MATCH_DICT.keys())[list(PERIL_TYPE_MATCH_DICT.values()).index(hazard_type_emdat)]
    elif not hazard_type_emdat:
        if hazard_type_climada in PERIL_SUBTYPE_MATCH_DICT.keys():
            hazard_type_emdat = PERIL_SUBTYPE_MATCH_DICT[hazard_type_climada]
        elif hazard_type_climada in PERIL_TYPE_MATCH_DICT.keys():
            hazard_type_emdat = PERIL_TYPE_MATCH_DICT[hazard_type_climada]

    # Inititate Impact-instance:
    impact_instance = Impact()

    impact_instance.tag = dict()
    impact_instance.tag['haz'] = TagHaz(haz_type=hazard_type_climada, \
                       file_name=emdat_file_csv, description='EM-DAT impact, direct import')
    impact_instance.tag['exp'] = Tag(file_name=emdat_file_csv, \
                       description='EM-DAT impact, direct import')
    impact_instance.tag['if_set'] = Tag(file_name=None, description=None)

    if not countries or countries == ['all']:
        countries = emdat_countries_by_hazard(hazard_type_emdat, emdat_file_csv, \
                                    ignore_missing=True, verbose=True, \
                                    target_version=target_version)[0]
    else:
        if isinstance(countries, str):
            countries = [countries]
    # Load EM-DAT impact data by event:
    em_data = emdat_impact_event(countries, hazard_type_emdat, emdat_file_csv, \
                                 year_range, reference_year=reference_year, target_version=target_version)
    if em_data.empty:
        return impact_instance, countries
    impact_instance.event_id = np.array(em_data.index, int)
    impact_instance.event_name = list(em_data[VARNAMES[target_version]['Dis No']])

    date_list = list()
    for year in list(em_data['Year']):
        date_list.append(datetime.toordinal(datetime.strptime(str(year), '%Y')))
    boolean_warning = True
    if target_version<=2018:
        for idx, datestr in enumerate(list(em_data['Start date'])):
            try:
                date_list[idx] = datetime.toordinal(datetime.strptime(datestr[-7:], '%m/%Y'))
            except ValueError:
                if boolean_warning:
                    LOGGER.warning('EM_DAT CSV contains invalid time formats')
                    boolean_warning = False
            try:
                date_list[idx] = datetime.toordinal(datetime.strptime(datestr, '%d/%m/%Y'))
            except ValueError:
                if boolean_warning:
                    LOGGER.warning('EM_DAT CSV contains invalid time formats')
                    boolean_warning = False
    else:
        idx = 0
        for year, month, day in zip(em_data['Start Year'], em_data['Start Month'], em_data['Start Day']):
            if np.isnan(year): year=-9999
            if np.isnan(month): month=1
            if np.isnan(day): day=1
            date_list[idx] = datetime.toordinal(datetime.strptime('%02i/%02i/%04i' %(day, month, year), '%d/%m/%Y'))
            idx += 1

    impact_instance.date = np.array(date_list, int)

    impact_instance.crs = DEF_CRS

    if reference_year == 0:
        impact_instance.at_event = np.array(em_data[imp_str])
    else:
        impact_instance.at_event = np.array(em_data[imp_str + " scaled"])
    impact_instance.at_event[np.isnan(impact_instance.at_event)]=0
    if not year_range:
        year_range = [em_data['Year'].min(), em_data['Year'].max()]
    impact_instance.frequency = np.ones(em_data.shape[0])/(1+np.diff(year_range))
    impact_instance.tot_value = 0
    impact_instance.aai_agg = np.nansum(impact_instance.at_event * impact_instance.frequency)
    impact_instance.unit = 'USD'
    impact_instance.imp_mat = []

    # init rough exposure with central point per country
    shp = shapereader.natural_earth(resolution='110m',
                                    category='cultural',
                                    name='admin_0_countries')
    shp = shapefile.Reader(shp)
    countries_reg_id = list()
    countries_lat = list()
    countries_lon = list()
    impact_instance.eai_exp = np.zeros(len(countries)) # empty: damage at exposure
    for idx, cntry in enumerate(countries):
        try:
            cntry = iso_cntry.get(cntry).alpha3
        except KeyError:
            LOGGER.error('Country not found in iso_country: ' + cntry)
        cntry_boolean = False
        for rec_i, rec in enumerate(shp.records()):
            if rec[9].casefold() == cntry.casefold():
                bbox = shp.shapes()[rec_i].bbox
                cntry_boolean = True
                break
        if cntry_boolean:
            countries_lat.append(np.mean([bbox[1], bbox[3]]))
            countries_lon.append(np.mean([bbox[0], bbox[2]]))
        else:
            countries_lat.append(np.nan)
            countries_lon.append(np.nan)
        try:
            countries_reg_id.append(int(iso_cntry.get(cntry).numeric))
        except KeyError:
            countries_reg_id.append(0)
        df_tmp = em_data[em_data[VARNAMES[target_version]['ISO']].str.contains(cntry)]
        if reference_year == 0:
            impact_instance.eai_exp[idx] = sum(np.array(df_tmp[imp_str])*\
                                   impact_instance.frequency[0])
        else:
            impact_instance.eai_exp[idx] = sum(np.array(df_tmp[imp_str + " scaled"])*\
                                   impact_instance.frequency[0])
 
    impact_instance.coord_exp = np.stack([countries_lat, countries_lon], axis=1)
    #impact_instance.plot_raster_eai_exposure()

    return impact_instance, countries