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
# import climada

from climada.util.finance import gdp

LOGGER = logging.getLogger(__name__)

PERIL_SUBTYPE_MATCH_DICT = dict(TC='Tropical cyclone',
                                T1='Storm',
                                TS='Coastal flood',
                                EQ='Ground movement',
                                E1='Earthquake',
                                FL='Riverine flood',
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

if False:
    # inputs
    checkset = pd.read_csv('~.csv')
    intensity_path = '~.p'
    names_path = '~.p'
    reg_ID_path = '~.p'
    date_path = '~.p'
    EMdat_raw = pd.read_excel('~.xlsx')
    start = 'yyyy-mm-dd'
    end = 'yyyy-mm-dd'

# assign hazard to EMdat event
    data = assign_hazard_to_EMdat(certainty_level='low', intensity_path_haz=intensity_path,
                                  names_path_haz=names_path, reg_ID_path_haz=reg_ID_path,
                                  date_path_haz=date_path, EMdat_data=EMdat_raw,
                                  start_time=start, end_time=end, keep_checks=True)
    check_assigned_track(lookup=data, checkset=checkset)

###############################################################################

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
    lookup = pd.DataFrame(columns = ['hit_country', 'Date_start_EM', \
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

def emdat_countries_by_hazard(hazard_name, emdat_file_csv, ignore_missing=True, \
                              verbose=True):
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
    Returns:
        exp_iso: List of ISO3-codes of countries impacted by the disaster type
        exp_name: List of names of countries impacted by the disaster type
            """



    if hazard_name in PERIL_SUBTYPE_MATCH_DICT.keys():
        hazard_name = PERIL_SUBTYPE_MATCH_DICT[hazard_name]
    elif hazard_name in PERIL_TYPE_MATCH_DICT.keys():
        hazard_name = PERIL_TYPE_MATCH_DICT[hazard_name]
        LOGGER.debug('Used "Disaster type" instead of "Disaster subtype" for matching hazard_name.')


    out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=1)
    if not 'Disaster type' in out.columns:
        out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=0)

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
    if not out[out['Disaster subtype'] == hazard_name].empty:
        uni_cntry = np.unique(out[out['Disaster subtype'] == hazard_name]['Country'].values)
    elif not out[out['Disaster type'] == hazard_name].empty:
        uni_cntry = np.unique(out[out['Disaster type'] == hazard_name]['Country'].values)
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
            if cntry == 'Saint Barth?lemy':
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
            if not verbose:
                LOGGER.debug(cntry, ':', iso_cntry.get(cntry).name)
            exp_iso.append(iso_cntry.get(cntry).alpha3)
            exp_name.append(iso_cntry.get(cntry).name)   
    return exp_iso, exp_name

def emdat_df_load(country, hazard_name, emdat_file_csv, year_range):
    """function to load EM-DAT data by country, hazard type and year range"""
    if hazard_name == 'TC':
        hazard_name = 'Tropical cyclone'
    elif hazard_name == 'DR':
        hazard_name = 'Drought'

    exp_iso, exp_name = emdat_countries_by_hazard(hazard_name, emdat_file_csv)
    if isinstance(country, int) | (not isinstance(country,str)):
        country = iso_cntry.get(country).alpha3
    if country in exp_name:
        country = exp_iso[exp_name.index(country)]
    if country not in exp_iso:
        raise NameError

    all_years = np.arange(min(year_range), max(year_range)+1, 1)
    out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=1)
    if not 'Disaster type' in out.columns:
        out = pd.read_csv(emdat_file_csv, encoding="ISO-8859-1", header=0)
    out = out[out['ISO'].str.contains(country) == True]
    out_ = out[out['Disaster subtype'].str.contains(hazard_name)]
    out_ = out_.append(out[out['Disaster type'].str.contains(hazard_name)])
    del out
    year_boolean = []
    for _, disaster_no in enumerate(out_['Disaster No.']):
        if isinstance(disaster_no, str) and int(disaster_no[0:4]) in all_years:
            year_boolean.append(True)
        else:
            year_boolean.append(False)
    out_ = out_[year_boolean]
    out_ = out_[out_['Disaster No.'].str.contains(str())]
    out_ = out_.reset_index(drop=True)
    return out_, sorted(all_years), country

def emdat_impact_yearlysum(countries, hazard_name, emdat_file_csv, year_range, \
                         reference_year=0, imp_str="Total damage ('000 US$)"):
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
            default = "Total damage ('000 US$)"

    Returns:
        yearly_impact (dict, mapping years to impact):
            total impact per year, same unit as chosen impact,
            i.e. 1000 current US$ for imp_str="Total damage ('000 US$)".
        all_years (list of int): list of years
    """

    out = pd.DataFrame()
    for country in countries:
        data, all_years, country = emdat_df_load(country, hazard_name, \
                                            emdat_file_csv, year_range)
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
                sum(data.loc[data['Disaster No.'].str.contains(str(year))]\
                             [imp_str])
            if '000 US' in imp_str: # EM-DAT damages provided in '000 USD
                data_out.loc[cnt, 'impact'] = data_out.loc[cnt, 'impact']*1000
            if reference_year > 0:
                data_out.loc[cnt, 'impact_scaled'] = data_out.loc[cnt, 'impact'] * \
                gdp_ref / gdp(country, year)[1]
        out = out.append(data_out)
    out = out.reset_index(drop=True)
    return out
    # out.loc[out['year']==1980]['impact'].sum() < sum for year 1980

def emdat_impact_event(countries, hazard_name, emdat_file_csv, year_range, \
                       reference_year=0, imp_str="Total damage ('000 US$)"):
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
            default = "Total damage ('000 US$)"

    Returns:
        out (pandas DataFrame): EMDAT DataFrame with new columns "year",
            "region_id", and scaled total impact per event with
            same unit as chosen impact,
            i.e. 1000 current US$ for imp_str="Total damage ('000 US$) scaled".
    """
    out = pd.DataFrame()
    for country in countries:
        data, all_years, country = emdat_df_load(country, hazard_name, \
                                                 emdat_file_csv, year_range)
        if reference_year > 0:
            gdp_ref = gdp(country, reference_year)[1]
        else: gdp_ref = 0
        data['year'] = pd.Series(np.zeros(data.shape[0], dtype='int'), \
            index=data.index)
        data['region_id'] = pd.Series(int(iso_cntry.get(country).numeric) + \
            np.zeros(data.shape[0], dtype='int'), \
            index=data.index)
        data['reference_year'] = pd.Series(reference_year+np.zeros(\
            data.shape[0], dtype='int'), index=data.index)
        data[imp_str + " scaled"] = pd.Series(np.zeros(data.shape[0], dtype='int'), \
            index=data.index)
        for cnt in np.arange(data.shape[0]):
            data.loc[cnt, 'year'] = int(data.loc[cnt, 'Disaster No.'][0:4])
            data.loc[cnt, 'reference_year'] = int(reference_year)
            if data.loc[cnt][imp_str] > 0 and gdp_ref > 0:
                data.loc[cnt, imp_str + " scaled"] = \
                    data.loc[cnt, imp_str] * gdp_ref / \
                    gdp(country, int(data.loc[cnt, 'year']))[1]
        out = out.append(data)
        del data
    out = out.reset_index(drop=True)
    if '000 US' in imp_str: # EM-DAT damages provided in '000 USD
        out[imp_str + " scaled"] = out[imp_str + " scaled"]*1e3
        out[imp_str] = out[imp_str]*1e3
    return out

"""
function emdata = emdat_load_yearlysum(country_emdat,peril_ID,exposure_growth,years_range)
% function to load EM-DAT data and sum damages per year
all_years=years_range(1):years_range(2);
emdata.year=all_years;
emdata.values=zeros([length(all_years) 1]);
em_data_i=emdat_read('',country_emdat,peril_ID,exposure_growth,0);
% if EM-DAT data available for this country, use, if not, assume zeros
if ~isempty(em_data_i)
    for iy=1:length(all_years)
        ii=find(all_years(iy) == em_data_i.year);
        if ~isempty(ii)
            emdata.values(iy,1) = sum(em_data_i.damage(ii));
        end
    end
end
end
"""
