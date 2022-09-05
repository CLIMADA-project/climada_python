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

Functions to merge EMDAT damages to hazard events.
"""
import logging
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from cartopy.io import shapereader

from climada.util.finance import gdp
from climada.util.constants import DEF_CRS
import climada.util.coordinates as u_coord
from climada.engine import Impact
from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHaz

LOGGER = logging.getLogger(__name__)

PERIL_SUBTYPE_MATCH_DICT = dict(TC=['Tropical cyclone'],
                                FL=['Coastal flood'],
                                EQ=['Ground movement', 'Earthquake'],
                                RF=['Riverine flood', 'Flood'],
                                WS=['Extra-tropical storm', 'Storm'],
                                DR=['Drought'],
                                LS=['Landslide'],
                                BF=['Forest fire', 'Wildfire', 'Land fire (Brush, Bush, Pastur']
                                )

PERIL_TYPE_MATCH_DICT = dict(DR=['Drought'],
                             EQ=['Earthquake'],
                             FL=['Flood'],
                             LS=['Landslide'],
                             VQ=['Volcanic activity'],
                             BF=['Wildfire'],
                             HW=['Extreme temperature']
                             )

VARNAMES_EMDAT = \
    {2018: {'Dis No': 'Disaster No.',
            'Disaster Type': 'Disaster type',
            'Disaster Subtype': 'Disaster subtype',
            'Event Name': 'Disaster name',
            'Country': 'Country',
            'ISO': 'ISO',
            'Location': 'Location',
            'Associated Dis': 'Associated disaster',
            'Associated Dis2': 'Associated disaster2',
            'Dis Mag Value': 'Magnitude value',
            'Dis Mag Scale': 'Magnitude scale',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude',
            'Total Deaths': 'Total deaths',
            'Total Affected': 'Total affected',
            "Insured Damages ('000 US$)": "Insured losses ('000 US$)",
            "Total Damages ('000 US$)": "Total damage ('000 US$)"},
     2020: {'Dis No': 'Dis No',
            'Year': 'Year',
            'Seq': 'Seq',
            'Disaster Group': 'Disaster Group',
            'Disaster Subgroup': 'Disaster Subgroup',
            'Disaster Type': 'Disaster Type',
            'Disaster Subtype': 'Disaster Subtype',
            'Disaster Subsubtype': 'Disaster Subsubtype',
            'Event Name': 'Event Name',
            'Entry Criteria': 'Entry Criteria',
            'Country': 'Country',
            'ISO': 'ISO',
            'Region': 'Region',
            'Continent': 'Continent',
            'Location': 'Location',
            'Origin': 'Origin',
            'Associated Dis': 'Associated Dis',
            'Associated Dis2': 'Associated Dis2',
            'OFDA Response': 'OFDA Response',
            'Appeal': 'Appeal',
            'Declaration': 'Declaration',
            'Aid Contribution': 'Aid Contribution',
            'Dis Mag Value': 'Dis Mag Value',
            'Dis Mag Scale': 'Dis Mag Scale',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude',
            'Local Time': 'Local Time',
            'River Basin': 'River Basin',
            'Start Year': 'Start Year',
            'Start Month': 'Start Month',
            'Start Day': 'Start Day',
            'End Year': 'End Year',
            'End Month': 'End Month',
            'End Day': 'End Day',
            'Total Deaths': 'Total Deaths',
            'No Injured': 'No Injured',
            'No Affected': 'No Affected',
            'No Homeless': 'No Homeless',
            'Total Affected': 'Total Affected',
            "Reconstruction Costs ('000 US$)": "Reconstruction Costs ('000 US$)",
            "Insured Damages ('000 US$)": "Insured Damages ('000 US$)",
            "Total Damages ('000 US$)": "Total Damages ('000 US$)",
            'CPI': 'CPI'}}


def assign_hazard_to_emdat(certainty_level, intensity_path_haz, names_path_haz,
                           reg_id_path_haz, date_path_haz, emdat_data,
                           start_time, end_time, keep_checks=False):
    """assign_hazard_to_emdat: link EMdat event to hazard

    Parameters
    ----------
    certainty_level : str
        'high' or 'low'
    intensity_path_haz : sparse matrix
        with hazards as rows and grid points as cols,
        values only at location with impacts
    names_path_haz : str
        identifier for each hazard (i.e. IBtracID) (rows of the matrix)
    reg_id_path_haz : str
        ISO country ID of each grid point (cols of the matrix)
    date_path_haz : str
        start date of each hazard (rows of the matrix)
    emdat_data: pd.DataFrame
        dataframe with EMdat data
    start_time : str
        start date of events to be assigned 'yyyy-mm-dd'
    end_time : str
        end date of events to be assigned 'yyyy-mm-dd'
    keep_checks : bool, optional

    Returns
    -------
    pd.dataframe with EMdat entries linked to a hazard
    """
    # check valid certainty level
    certainty_levels = ['high', 'low']
    if certainty_level not in certainty_levels:
        raise ValueError("Invalid certainty level. Expected one of: %s" % certainty_levels)

    # prepare hazard set
    print("Start preparing hazard set")
    hit_countries = hit_country_per_hazard(intensity_path_haz, names_path_haz,
                                           reg_id_path_haz, date_path_haz)
    # prepare damage set

    # adjust emdat_data to the path!!
    print("Start preparing damage set")
    lookup = create_lookup(emdat_data, start_time, end_time, disaster_subtype='Tropical cyclone')
    # calculate possible hits
    print("Calculate possible hits")
    hit5 = emdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=5)
    hit5_match = match_em_id(lookup=lookup, poss_hit=hit5)
    print("1/5")
    hit10 = emdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=10)
    hit10_match = match_em_id(lookup=lookup, poss_hit=hit10)
    print("2/5")
    hit15 = emdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=15)
    hit15_match = match_em_id(lookup=lookup, poss_hit=hit15)
    print("3/5")
    hit25 = emdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=25)
    hit25_match = match_em_id(lookup=lookup, poss_hit=hit25)
    print("4/5")
    hit50 = emdat_possible_hit(lookup=lookup, hit_countries=hit_countries, delta_t=50)
    hit50_match = match_em_id(lookup=lookup, poss_hit=hit50)
    print("5/5")

    # assign only tracks with high certainty
    print("Assign tracks")
    if certainty_level == 'high':
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit50_match, level=1)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit50_match, level=2)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit25_match,
                                    possible_tracks_2=hit50_match, level=3)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit25_match, level=4)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit25_match, level=5)
    # assign all tracks
    elif certainty_level == 'low':
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit50_match, level=1)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit50_match, level=2)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit50_match, level=3)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit25_match, level=4)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit25_match, level=5)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit25_match, level=6)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit15_match, level=7)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit15_match, level=8)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit10_match, level=9)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit15_match,
                                    possible_tracks_2=hit15_match, level=10)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit10_match,
                                    possible_tracks_2=hit10_match, level=11)
        lookup = assign_track_to_em(lookup=lookup, possible_tracks_1=hit5_match,
                                    possible_tracks_2=hit5_match, level=12)
    if not keep_checks:
        lookup = lookup.drop(['Date_start_EM_ordinal', 'possible_track',
                              'possible_track_all'], axis=1)
    lookup.groupby('allocation_level').count()
    print('(%d/%s) tracks allocated' % (
        len(lookup[lookup.allocation_level.notnull()]), len(lookup)))
    return lookup


def hit_country_per_hazard(intensity_path, names_path, reg_id_path, date_path):
    """hit_country_per_hazard: create list of hit countries from hazard set

    Parameters
    ----------
    intensity_path : str
        Path to file containing sparse matrix with hazards as rows and grid points
        as cols, values only at location with impacts
    names_path : str
        Path to file with identifier for each hazard (i.e. IBtracID) (rows of the matrix)
    reg_id_path : str
        Path to file with ISO country ID of each grid point (cols of the matrix)
    date_path : str
        Path to file with start date of each hazard (rows of the matrix)

    Returns
    -------
        pd.DataFrame with all hit countries per hazard
    """
    with open(intensity_path, 'rb') as filef:
        inten = pickle.load(filef)
    with open(names_path, 'rb') as filef:
        names = pickle.load(filef)
    with open(reg_id_path, 'rb') as filef:
        reg_id = pickle.load(filef)
    with open(date_path, 'rb') as filef:
        date = pickle.load(filef)
    # loop over the tracks (over the rows of the intensity matrix)
    all_hits = []
    for track in range(0, len(names)):
        # select track
        tc_track = inten[track, ]
        # select only indices that are not zero
        hits = tc_track.nonzero()[1]
        # get the country of these indices and remove dublicates
        hits = list(set(reg_id[hits]))
        # append hit countries to list
        all_hits.append(hits)

    # create data frame for output
    hit_countries = pd.DataFrame(columns=['hit_country', 'Date_start', 'ibtracsID'])
    for track, _ in enumerate(names):
        # Check if track has hit any country else go to the next track
        if len(all_hits[track]) > 0:
            # loop over hit_country
            for hit in range(0, len(all_hits[track])):
                # Hit country ISO
                ctry_iso = u_coord.country_to_iso(all_hits[track][hit], "alpha3")
                # create entry for each country a hazard has hit
                hit_countries = hit_countries.append({'hit_country': ctry_iso,
                                                      'Date_start': date[track],
                                                      'ibtracsID': names[track]},
                                                     ignore_index=True)
    # retrun data frame with all hit countries per hazard
    return hit_countries


def create_lookup(emdat_data, start, end, disaster_subtype='Tropical cyclone'):
    """create_lookup: prepare a lookup table of EMdat events to which hazards can be assigned

        Parameters
        ----------
        emdat_data: pd.DataFrame
            with EMdat data
        start : str
            start date of events to be assigned 'yyyy-mm-dd'
        end : str
            end date of events to be assigned 'yyyy-mm-dd'
        disaster_subtype : str
            EMdat disaster subtype

        Returns
        -------
        pd.DataFrame
        """
    data = emdat_data[emdat_data['Disaster_subtype'] == disaster_subtype]
    lookup = pd.DataFrame(columns=['hit_country', 'Date_start_EM',
                                   'Date_start_EM_ordinal', 'Disaster_name',
                                   'EM_ID', 'ibtracsID', 'allocation_level',
                                   'possible_track', 'possible_track_all'])
    lookup.hit_country = data.ISO
    lookup.Date_start_EM = data.Date_start_clean
    lookup.Disaster_name = data.Disaster_name
    lookup.EM_ID = data.Disaster_No
    lookup = lookup.reset_index(drop=True)
    # create ordinals
    for i in range(0, len(data.Date_start_clean.values)):
        lookup.Date_start_EM_ordinal[i] = datetime.toordinal(
            datetime.strptime(lookup.Date_start_EM.values[i], '%Y-%m-%d'))
        # ordinals to numeric
    lookup.Date_start_EM_ordinal = pd.to_numeric(lookup.Date_start_EM_ordinal)
    # select time
    emdat_start = datetime.toordinal(datetime.strptime(start, '%Y-%m-%d'))
    emdat_end = datetime.toordinal(datetime.strptime(end, '%Y-%m-%d'))

    lookup = lookup[lookup.Date_start_EM_ordinal.values > emdat_start]
    lookup = lookup[lookup.Date_start_EM_ordinal.values < emdat_end]

    return lookup


# Function to relate EM disaster to IBtrack using hit countries and time
def emdat_possible_hit(lookup, hit_countries, delta_t):
    """relate EM disaster to hazard using hit countries and time

    Parameters
    ----------
    lookup : pd.DataFrame
        to relate EMdatID to hazard
    delta_t :
        max time difference of start of EMdat event and hazard
    hit_countries:


    Returns
    -------
    list with possible hits
    """
    # lookup: PD dataframe that relates EMdatID to an IBtracsID
    # tracks: processed IBtracks with info which track hit which country
    # delta_t: time difference of start of EMdat and IBrtacks
    possible_hit_all = []
    for i in range(0, len(lookup.EM_ID.values)):
        possible_hit = []
        country_tracks = hit_countries[
            hit_countries['hit_country'] == lookup.hit_country.values[i]]
        for j in range(0, len(country_tracks.Date_start.values)):
            if (lookup.Date_start_EM_ordinal.values[i] - country_tracks.Date_start.values[j]) < \
                delta_t and (lookup.Date_start_EM_ordinal.values[i] -
                             country_tracks.Date_start.values[j]) >= 0:
                possible_hit.append(country_tracks.ibtracsID.values[j])
        possible_hit_all.append(possible_hit)

    return possible_hit_all


# function to check if EM_ID has been assigned already
def match_em_id(lookup, poss_hit):
    """function to check if EM_ID has been assigned already and combine possible hits

        Parameters
        ----------
        lookup : pd.dataframe
            to relate EMdatID to hazard
        poss_hit : list
            with possible hits

        Returns
        -------
        list
            with all possible hits per EMdat ID
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


def assign_track_to_em(lookup, possible_tracks_1, possible_tracks_2, level):
    """function to assign a hazard to an EMdat event
        to get some confidene into the procedure, hazards get only assigned
        if there is no other hazard occuring at a bigger time interval in that country
        Thus a track of possible_tracks_1 gets only assigned if there are no other
        tracks in possible_tracks_2.
        The confidence can be expressed with a certainty level

        Parameters
        ----------
        lookup : pd.DataFrame
            to relate EMdatID to hazard
        possible_tracks_1 : list
            list of possible hits with smaller time horizon
        possible_tracks_2 : list
            list of possible hits with larger time horizon
        level : int
            level of confidence

        Returns
        -------
        pd.DataFrame
            lookup with assigend tracks and possible hits
    """

    for i, _ in enumerate(possible_tracks_1):
        if np.isnan(lookup.allocation_level.values[i]):
            number_emdat_id = len(possible_tracks_1[i])
            # print(number_emdat_id)
            for j in range(0, number_emdat_id):
                # check that number of possible track stays the same at given
                # time difference and that list is not empty
                if len(possible_tracks_1[i][j]) == len(possible_tracks_2[i][j]) == 1 \
                        and possible_tracks_1[i][j] != []:
                    # check that all tracks are the same
                    if all(possible_tracks_1[i][0] == possible_tracks_1[i][k]
                           for k in range(0, len(possible_tracks_1[i]))):
                        # check that track ID has not been assigned to that country already
                        ctry_lookup = lookup[lookup['hit_country'] == lookup.hit_country.values[i]]
                        if possible_tracks_1[i][0][0] not in ctry_lookup.ibtracsID.values:
                            lookup.ibtracsID.values[i] = possible_tracks_1[i][0][0]
                            lookup.allocation_level.values[i] = level
                elif possible_tracks_1[i][j] != []:
                    lookup.possible_track.values[i] = possible_tracks_1[i]
        else:
            lookup.possible_track_all.values[i] = possible_tracks_1[i]
    return lookup


def check_assigned_track(lookup, checkset):
    """compare lookup with assigned tracks to a set with checked sets

        Parameters
        ----------
        lookup: pd.DataFrame
            dataframe to relate EMdatID to hazard
        checkset: pd.DataFrame
            dataframe with already checked hazards

        Returns
        -------
        error scores
    """
    # merge checkset and lookup
    check = pd.merge(checkset, lookup[['hit_country', 'EM_ID', 'ibtracsID']],
                     on=['hit_country', 'EM_ID'])
    check_size = len(check.ibtracsID.values)
    # not assigned values
    not_assigned = check.ibtracsID.isnull().sum(axis=0)
    # correct assigned values
    correct = sum(check.ibtracsID.values == check.IBtracsID_checked.values)
    # wrongly assigned values
    wrong = len(check.ibtracsID.values) - not_assigned - correct
    print('%.1f%% tracks assigned correctly, %.1f%% wrongly, %.1f%% not assigned'
          % (correct / check_size * 100,
             wrong / check_size * 100,
             not_assigned / check_size * 100))


def clean_emdat_df(emdat_file, countries=None, hazard=None, year_range=None,
                   target_version=2020):
    """
    Get a clean and standardized DataFrame from EM-DAT-CSV-file
    (1) load EM-DAT data from CSV to DataFrame and remove header/footer,
    (2) handle version, clean up, and add columns, and
    (3) filter by country, hazard type and year range (if any given)

    Parameters
    ----------
    emdat_file : str, Path, or DataFrame
        Either string with full path to CSV-file or
        pandas.DataFrame loaded from EM-DAT CSV
    countries : list of str
        country ISO3-codes or names, e.g. ['JAM', 'CUB'].
        countries=None for all countries (default)
    hazard : list or str
        List of Disaster (sub-)type accordung EMDAT terminology, i.e.:
        Animal accident, Drought, Earthquake, Epidemic, Extreme temperature,
        Flood, Fog, Impact, Insect infestation, Landslide, Mass movement (dry),
        Storm, Volcanic activity, Wildfire;
        Coastal Flooding, Convective Storm, Riverine Flood, Tropical cyclone,
        Tsunami, etc.;
        OR CLIMADA hazard type abbreviations, e.g. TC, BF, etc.
    year_range : list or tuple
        Year range to be extracted, e.g. (2000, 2015);
        (only min and max are considered)
    target_version : int
        required EM-DAT data format version (i.e. year of download),
        changes naming of columns/variables (default: 2020)

    Returns
    -------
    df_data : pd.DataFrame
        DataFrame containing cleaned and filtered EM-DAT impact data
    """
    # (1) load EM-DAT data from CSV to DataFrame, skipping the header:
    if isinstance(emdat_file, (str, Path)):
        df_emdat = pd.read_csv(emdat_file, encoding="ISO-8859-1", header=0)
        counter = 0
        while not ('Country' in df_emdat.columns and 'ISO' in df_emdat.columns):
            counter += 1
            df_emdat = pd.read_csv(emdat_file, encoding="ISO-8859-1", header=counter)
            if counter == 10:
                break
        del counter
    elif isinstance(emdat_file, pd.DataFrame):
        df_emdat = emdat_file
    else:
        raise TypeError('emdat_file needs to be str or DataFrame')
    # drop rows with 9 or more NaN values (e.g. footer):
    df_emdat = df_emdat.dropna(thresh=9)

    # (2)  handle version, clean up, and add columns:
    # (2.1) identify underlying EMDAT version of csv:
    version = 2020
    for vers in list(VARNAMES_EMDAT.keys()):
        if len(df_emdat.columns) >= len(VARNAMES_EMDAT[vers]) and \
           all(item in list(df_emdat.columns) for item in VARNAMES_EMDAT[vers].values()):
            version = vers
    # (2.2) create new DataFrame df_data with column names as target version
    df_data = pd.DataFrame(index=df_emdat.index.values,
                           columns=VARNAMES_EMDAT[target_version].values())
    if 'Year' not in df_data.columns:  # make sure column "Year" exists
        df_data['Year'] = np.nan
    for _, col in enumerate(df_data.columns):  # loop over columns
        if col in VARNAMES_EMDAT[version]:
            df_data[col] = df_emdat[VARNAMES_EMDAT[version][col]]
        elif col in df_emdat.columns:
            df_data[col] = df_emdat[col]
        elif col == 'Year' and version <= 2018:
            years_list = list()
            for _, disaster_no in enumerate(df_emdat[VARNAMES_EMDAT[version]['Dis No']]):
                if isinstance(disaster_no, str):
                    years_list.append(int(disaster_no[0:4]))
                else:
                    years_list.append(np.nan)
            df_data[col] = years_list
    if version <= 2018 and target_version >= 2020:
        date_list = list()
        year_list = list()
        month_list = list()
        day_list = list()
        for year in list(df_data['Year']):
            if not np.isnan(year):
                date_list.append(datetime.strptime(str(year), '%Y'))
            else:
                date_list.append(datetime.strptime(str('0001'), '%Y'))
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
        df_data['Start Month'] = np.array(month_list, dtype='int')
        df_data['Start Day'] = np.array(day_list, dtype='int')
        df_data['Start Year'] = np.array(year_list, dtype='int')
        for var in ['Disaster Subtype', 'Disaster Type', 'Country']:
            df_data[VARNAMES_EMDAT[target_version][var]].fillna('None', inplace=True)

    # (3) Filter by countries, year range, and disaster type
    # (3.1) Countries:
    if countries and isinstance(countries, str):
        countries = [countries]
    if countries and isinstance(countries, list):
        for idx, country in enumerate(countries):
            # convert countries to iso3 alpha code:
            countries[idx] = u_coord.country_to_iso(country, "alpha3")
        df_data = df_data[df_data['ISO'].isin(countries)].reset_index(drop=True)
    # (3.2) Year range:
    if year_range:
        for idx in df_data.index:
            if np.isnan(df_data.loc[0, 'Year']):
                df_data.loc[0, 'Year'] = \
                    df_data.loc[0, VARNAMES_EMDAT[target_version]['Start Year']]
        df_data = df_data[(df_data['Year'] >= min(year_range)) &
                          (df_data['Year'] <= max(year_range))]

    # (3.3) Disaster type:
    if hazard and isinstance(hazard, str):
        hazard = [hazard]
    if hazard and isinstance(hazard, list):
        disaster_types = list()
        disaster_subtypes = list()
        for idx, haz in enumerate(hazard):
            if haz in df_data[VARNAMES_EMDAT[target_version]['Disaster Type']].unique():
                disaster_types.append(haz)
            if haz in df_data[VARNAMES_EMDAT[target_version]['Disaster Subtype']].unique():
                disaster_subtypes.append(haz)
            if haz in PERIL_TYPE_MATCH_DICT.keys():
                disaster_types += PERIL_TYPE_MATCH_DICT[haz]
            if haz in PERIL_SUBTYPE_MATCH_DICT.keys():
                disaster_subtypes += PERIL_SUBTYPE_MATCH_DICT[haz]
        df_data = df_data[
            (df_data[VARNAMES_EMDAT[target_version]['Disaster Type']].isin(disaster_types)) |
            (df_data[VARNAMES_EMDAT[target_version]['Disaster Subtype']].isin(disaster_subtypes))]
    return df_data.reset_index(drop=True)


def emdat_countries_by_hazard(emdat_file_csv, hazard=None, year_range=None):
    """return list of all countries exposed to a chosen hazard type
    from EMDAT data as CSV.

    Parameters
    ----------
    emdat_file : str, Path, or DataFrame
        Either string with full path to CSV-file or
        pandas.DataFrame loaded from EM-DAT CSV
    hazard : list or str
        List of Disaster (sub-)type accordung EMDAT terminology, i.e.:
        Animal accident, Drought, Earthquake, Epidemic, Extreme temperature,
        Flood, Fog, Impact, Insect infestation, Landslide, Mass movement (dry),
        Storm, Volcanic activity, Wildfire;
        Coastal Flooding, Convective Storm, Riverine Flood, Tropical cyclone,
        Tsunami, etc.;
        OR CLIMADA hazard type abbreviations, e.g. TC, BF, etc.
    year_range : list or tuple
        Year range to be extracted, e.g. (2000, 2015);
        (only min and max are considered)

    Returns
    -------
    countries_iso3a : list
        List of ISO3-codes of countries impacted by the disaster (sub-)types
    countries_names : list
        List of names of countries impacted by the disaster (sub-)types
    """
    df_data = clean_emdat_df(emdat_file_csv, hazard=hazard, year_range=year_range)
    countries_iso3a = list(df_data.ISO.unique())
    countries_names = list()
    for iso3a in countries_iso3a:
        try:
            countries_names.append(u_coord.country_to_iso(iso3a, "name"))
        except LookupError:
            countries_names.append('NA')
    return countries_iso3a, countries_names


def scale_impact2refyear(impact_values, year_values, iso3a_values, reference_year=None):
    """Scale give impact values proportional to GDP to the according value in a reference year
    (for normalization of monetary values)

    Parameters
    ----------
    impact_values : list or array
        Impact values to be scaled.
    year_values : list or array
        Year of each impact (same length as impact_values)
    iso3a_values : list or array
        ISO3alpha code of country for each impact (same length as impact_values)
    reference_year : int, optional
        Impact is scaled proportional to GDP to the value of the reference year.
        No scaling for reference_year=None (default)
    """
    impact_values = np.array(impact_values)
    year_values = np.array(year_values)
    iso3a_values = np.array(iso3a_values)
    if reference_year and isinstance(reference_year, (int, float)):
        reference_year = int(reference_year)
        gdp_ref = dict()
        gdp_years = dict()
        for country in np.unique(iso3a_values):
            # get reference GDP value for each country:
            gdp_ref[country] = gdp(country, reference_year)[1]
            # get GDP value for each country and year:
            gdp_years[country] = dict()
            years_country = np.unique(year_values[iso3a_values == country])
            print(years_country)
            for year in years_country:
                gdp_years[country][year] = gdp(country, year)[1]
        # loop through each value and apply scaling:
        for idx, val in enumerate(impact_values):
            impact_values[idx] = val * gdp_ref[iso3a_values[idx]] / \
                gdp_years[iso3a_values[idx]][year_values[idx]]
        return list(impact_values)
    if not reference_year:
        return impact_values
    raise ValueError('Invalid reference_year')


def emdat_impact_yearlysum(emdat_file_csv, countries=None, hazard=None, year_range=None,
                           reference_year=None, imp_str="Total Damages ('000 US$)",
                           version=2020):
    """function to load EM-DAT data and sum impact per year

    Parameters
    ----------
     emdat_file_csv : str or DataFrame
        Either string with full path to CSV-file or
        pandas.DataFrame loaded from EM-DAT CSV
    countries : list of str
        country ISO3-codes or names, e.g. ['JAM', 'CUB'].
        countries=None for all countries (default)
    hazard : list or str
        List of Disaster (sub-)type accordung EMDAT terminology, i.e.:
        Animal accident, Drought, Earthquake, Epidemic, Extreme temperature,
        Flood, Fog, Impact, Insect infestation, Landslide, Mass movement (dry),
        Storm, Volcanic activity, Wildfire;
        Coastal Flooding, Convective Storm, Riverine Flood, Tropical cyclone,
        Tsunami, etc.;
        OR CLIMADA hazard type abbreviations, e.g. TC, BF, etc.
    year_range : list or tuple
        Year range to be extracted, e.g. (2000, 2015);
        (only min and max are considered)
    version : int
        required EM-DAT data format version (i.e. year of download),
        changes naming of columns/variables (default: 2020)

    Returns
    -------
    out : pd.DataFrame
        DataFrame with summed impact and scaled impact per
        year and country.
    """
    imp_str = VARNAMES_EMDAT[version][imp_str]
    df_data = clean_emdat_df(emdat_file_csv, countries=countries, hazard=hazard,
                             year_range=year_range, target_version=version)

    df_data[imp_str + " scaled"] = scale_impact2refyear(df_data[imp_str].values,
                                                        df_data.Year.values, df_data.ISO.values,
                                                        reference_year=reference_year)
    out = pd.DataFrame(columns=['ISO', 'region_id', 'year', 'impact',
                                'impact_scaled', 'reference_year'])
    for country in df_data.ISO.unique():
        country = u_coord.country_to_iso(country, "alpha3")
        if not df_data.loc[df_data.ISO == country].size:
            continue
        all_years = np.arange(min(df_data.Year), max(df_data.Year) + 1)
        data_out = pd.DataFrame(index=np.arange(0, len(all_years)),
                                columns=out.columns)
        df_country = df_data.loc[df_data.ISO == country]
        for cnt, year in enumerate(all_years):
            data_out.loc[cnt, 'year'] = year
            data_out.loc[cnt, 'reference_year'] = reference_year
            data_out.loc[cnt, 'ISO'] = country
            data_out.loc[cnt, 'region_id'] = u_coord.country_to_iso(country, "numeric")
            data_out.loc[cnt, 'impact'] = \
                np.nansum(df_country[df_country.Year.isin([year])][imp_str])
            data_out.loc[cnt, 'impact_scaled'] = \
                np.nansum(df_country[df_country.Year.isin([year])][imp_str + " scaled"])
            if '000 US' in imp_str:  # EM-DAT damages provided in '000 USD
                data_out.loc[cnt, 'impact'] = data_out.loc[cnt, 'impact'] * 1e3
                data_out.loc[cnt, 'impact_scaled'] = data_out.loc[cnt, 'impact_scaled'] * 1e3
        out = pd.concat([out, data_out])
    out = out.reset_index(drop=True)
    return out


def emdat_impact_event(emdat_file_csv, countries=None, hazard=None, year_range=None,
                       reference_year=None, imp_str="Total Damages ('000 US$)",
                       version=2020):
    """function to load EM-DAT data return impact per event

    Parameters
    ----------
     emdat_file_csv : str or DataFrame
        Either string with full path to CSV-file or
        pandas.DataFrame loaded from EM-DAT CSV
    countries : list of str
        country ISO3-codes or names, e.g. ['JAM', 'CUB'].
        default: countries=None for all countries
    hazard : list or str
        List of Disaster (sub-)type accordung EMDAT terminology, i.e.:
        Animal accident, Drought, Earthquake, Epidemic, Extreme temperature,
        Flood, Fog, Impact, Insect infestation, Landslide, Mass movement (dry),
        Storm, Volcanic activity, Wildfire;
        Coastal Flooding, Convective Storm, Riverine Flood, Tropical cyclone,
        Tsunami, etc.;
        OR CLIMADA hazard type abbreviations, e.g. TC, BF, etc.
    year_range : list or tuple
        Year range to be extracted, e.g. (2000, 2015);
        (only min and max are considered)
    reference_year : int reference year of exposures. Impact is scaled
        proportional to GDP to the value of the reference year. Default: No scaling
        for 0
    imp_str : str
        Column name of impact metric in EMDAT CSV,
        default = "Total Damages ('000 US$)"
    version : int
        EM-DAT version to take variable/column names from (defaul: 2020)

    Returns
    -------
    out : pd.DataFrame
        EMDAT DataFrame with new columns "year",
        "region_id", and "impact" and +impact_scaled" total impact per event with
        same unit as chosen impact, but multiplied by 1000 if impact is given
        as 1000 US$ (e.g. imp_str="Total Damages ('000 US$) scaled").
    """
    imp_str = VARNAMES_EMDAT[version][imp_str]
    df_data = clean_emdat_df(emdat_file_csv, hazard=hazard, year_range=year_range,
                             countries=countries, target_version=version)
    df_data['year'] = df_data['Year']
    df_data['reference_year'] = reference_year
    df_data['impact'] = df_data[imp_str]
    df_data['impact_scaled'] = scale_impact2refyear(df_data[imp_str].values, df_data.Year.values,
                                                    df_data.ISO.values,
                                                    reference_year=reference_year)
    df_data['region_id'] = np.nan
    for country in df_data.ISO.unique():
        try:
            df_data.loc[df_data.ISO == country, 'region_id'] = \
                u_coord.country_to_iso(country, "numeric")
        except LookupError:
            LOGGER.warning('ISO3alpha code not found in iso_country: %s', country)
    if '000 US' in imp_str:
        df_data['impact'] *= 1e3
        df_data['impact_scaled'] *= 1e3
    return df_data.reset_index(drop=True)


def emdat_to_impact(emdat_file_csv, hazard_type_climada, year_range=None, countries=None,
                    hazard_type_emdat=None,
                    reference_year=None, imp_str="Total Damages"):
    """function to load EM-DAT data return impact per event

    Parameters
    ----------
     emdat_file_csv : str or pd.DataFrame
        Either string with full path to CSV-file or
        pandas.DataFrame loaded from EM-DAT CSV
    countries : list of str
        country ISO3-codes or names, e.g. ['JAM', 'CUB'].
        default: countries=None for all countries
    hazard_type_climada : list or str
        List of Disaster (sub-)type accordung EMDAT terminology, i.e.:
        Animal accident, Drought, Earthquake, Epidemic, Extreme temperature,
        Flood, Fog, Impact, Insect infestation, Landslide, Mass movement (dry),
        Storm, Volcanic activity, Wildfire;
        Coastal Flooding, Convective Storm, Riverine Flood, Tropical cyclone,
        Tsunami, etc.;
        OR CLIMADA hazard type abbreviations, e.g. TC, BF, etc.
    year_range : list or tuple
        Year range to be extracted, e.g. (2000, 2015);
        (only min and max are considered)
    reference_year : int reference year of exposures. Impact is scaled
        proportional to GDP to the value of the reference year. Default: No scaling
        for 0
    imp_str : str
        Column name of impact metric in EMDAT CSV,
        default = "Total Damages ('000 US$)"

    Returns
    -------
    impact_instance : climada.engine.Impact
        impact object of same format as output from CLIMADA
        impact computation.
        Values scaled with GDP to reference_year if reference_year is given.
        i.e. current US$ for imp_str="Total Damages ('000 US$) scaled" (factor 1000 is applied)
        impact_instance.eai_exp holds expected impact for each country (within 1/frequency_unit).
        impact_instance.coord_exp holds rough central coordinates for each country.
    countries : list of str
        ISO3-codes of countries in same order as in impact_instance.eai_exp
    """
    if "Total Damages" in imp_str:
        imp_str = "Total Damages ('000 US$)"
    elif "Insured Damages" in imp_str:
        imp_str = "Insured Damages ('000 US$)"
    elif "Reconstruction Costs" in imp_str:
        imp_str = "Reconstruction Costs ('000 US$)"
    imp_str = VARNAMES_EMDAT[max(VARNAMES_EMDAT.keys())][imp_str]
    if not hazard_type_emdat:
        hazard_type_emdat = [hazard_type_climada]
    if reference_year == 0:
        reference_year = None
    # Inititate Impact-instance:
    impact_instance = Impact()

    impact_instance.tag = dict()
    impact_instance.tag['haz'] = TagHaz(haz_type=hazard_type_climada,
                                        file_name=emdat_file_csv,
                                        description='EM-DAT impact, direct import')
    impact_instance.tag['exp'] = Tag(file_name=emdat_file_csv,
                                     description='EM-DAT impact, direct import')
    impact_instance.tag['impf_set'] = Tag(file_name=None, description=None)


    # Load EM-DAT impact data by event:
    em_data = emdat_impact_event(emdat_file_csv, countries=countries, hazard=hazard_type_emdat,
                                 year_range=year_range, reference_year=reference_year,
                                 imp_str=imp_str, version=max(VARNAMES_EMDAT.keys()))

    if isinstance(countries, str):
        countries = [countries]
    elif not countries:
        countries = emdat_countries_by_hazard(emdat_file_csv, year_range=year_range,
                                              hazard=hazard_type_emdat)[0]

    if em_data.empty:
        return impact_instance, countries
    impact_instance.event_id = np.array(em_data.index, int)
    impact_instance.event_name = list(
        em_data[VARNAMES_EMDAT[max(VARNAMES_EMDAT.keys())]['Dis No']])

    date_list = list()
    for year in list(em_data['Year']):
        date_list.append(datetime.toordinal(datetime.strptime(str(year), '%Y')))
    if 'Start Year' in em_data.columns and 'Start Month' in em_data.columns \
            and 'Start Day' in em_data.columns:
        idx = 0
        for year, month, day in zip(em_data['Start Year'], em_data['Start Month'],
                                    em_data['Start Day']):
            if np.isnan(year):
                idx += 1
                continue
            if np.isnan(month):
                month = 1
            if np.isnan(day):
                day = 1
            date_list[idx] = datetime.toordinal(datetime.strptime(
                '%02i/%02i/%04i' % (day, month, year), '%d/%m/%Y'))
            idx += 1
    impact_instance.date = np.array(date_list, int)
    impact_instance.crs = DEF_CRS

    if not reference_year:
        impact_instance.at_event = np.array(em_data["impact"])
    else:
        impact_instance.at_event = np.array(em_data["impact_scaled"])
    impact_instance.at_event[np.isnan(impact_instance.at_event)] = 0
    if not year_range:
        year_range = [em_data['Year'].min(), em_data['Year'].max()]
    impact_instance.frequency = np.ones(em_data.shape[0]) / (1 + np.diff(year_range))
    impact_instance.frequency_unit = '1/year'
    impact_instance.tot_value = 0
    impact_instance.aai_agg = np.nansum(impact_instance.at_event * impact_instance.frequency)
    impact_instance.unit = 'USD'
    impact_instance.imp_mat = []

    # init rough exposure with central point per country
    shp = shapereader.natural_earth(resolution='110m',
                                    category='cultural',
                                    name='admin_0_countries')
    shp = shapereader.Reader(shp)
    countries_reg_id = list()
    countries_lat = list()
    countries_lon = list()
    impact_instance.eai_exp = np.zeros(len(countries))  # empty: damage at exposure
    for idx, cntry in enumerate(countries):
        try:
            cntry = u_coord.country_to_iso(cntry, "alpha3")
        except LookupError:
            LOGGER.warning('Country not found in iso_country: %s', cntry)
        cntry_boolean = False
        for rec in shp.records():
            if rec.attributes['ADM0_A3'].casefold() == cntry.casefold():
                bbox = rec.geometry.bounds
                cntry_boolean = True
                break
        if cntry_boolean:
            countries_lat.append(np.mean([bbox[1], bbox[3]]))
            countries_lon.append(np.mean([bbox[0], bbox[2]]))
        else:
            countries_lat.append(np.nan)
            countries_lon.append(np.nan)
        try:
            countries_reg_id.append(u_coord.country_to_iso(cntry, "numeric"))
        except LookupError:
            countries_reg_id.append(0)
        df_tmp = em_data[em_data[VARNAMES_EMDAT[
            max(VARNAMES_EMDAT.keys())]['ISO']].str.contains(cntry)]
        if not reference_year:
            impact_instance.eai_exp[idx] = sum(np.array(df_tmp["impact"]) *
                                               impact_instance.frequency[0])
        else:
            impact_instance.eai_exp[idx] = sum(np.array(df_tmp["impact_scaled"]) *
                                               impact_instance.frequency[0])

    impact_instance.coord_exp = np.stack([countries_lat, countries_lon], axis=1)
    return impact_instance, countries
