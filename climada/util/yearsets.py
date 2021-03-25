"""
This file is part of CLIMADA.
Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.
CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.
CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.
---
Define functions to handle impact year_sets
"""

import copy
import logging
import numpy as np
from numpy.random import default_rng

import climada.util.dates_times as u_dt

LOGGER = logging.getLogger(__name__)

def impact_yearset(eis, sampled_years=None, sampling_vect=None):

    """PURPOSE:
      Create an annual impact set (ais) by sampling events for each year from an existing
      probabilistic event impact set (eis).

    INPUTS:
      eis (impact class object): an event impact set (eis)
    OPTIONAL INPUT:
        sampled_years (list): list of years for the resulting annual impact set
            (by default a 1000 year long list starting on the 01-01-0001 is generated)
        sampling_vect (dict): the sampling vector. Needs to be obtained in a first
          call, i.e. [ais, sampling_vect] = climada_eis2ais(...) and then
          provided in subsequent calls(s) to obtain the exact same sampling
          structure of yearset, i.e ais = climada_eis2ais(..., sampling_vect)

    OUTPUTS:
      ais: the year impact set (ais), a struct with same fields as eis (such
          as Value, ed, ...). All fields same content as in eis, except:
          date(i): the year i
          impact(i): the sum of impact for year(i).
          frequency(i): the annual frequency, =1
      sampling_vect: the sampling vector (can be used to re-create the exact same yearset)
      """


    if not sampled_years:
        sampled_years = np.arange(1, 1000+1).tolist()
    
    year_list = [str(date) + '-01-01' for date in sampled_years]
    n_resampled_years = len(year_list)

    if not np.all(eis.frequency == eis.frequency[0]):
        LOGGER.warning("The frequencies of the single events differ among each other."
                       "Please beware that this might influence the results.")


    # INITIALISATION
    # to do: add a flag to distinguish ais from other impact class objects
    # artificially generate a generic  annual impact set (by sampling the event impact set)
    ais = copy.deepcopy(eis)
    ais.event_id = [] # to indicate not by event any more
    ais.event_id = np.arange(1, n_resampled_years+1)
    ais.frequency = [] # init, see below
    ais.at_event = np.zeros([1, n_resampled_years+1])
    ais.date = []


    impact_per_year, sampling_vect = sample_annual_impacts(eis, n_resampled_years,
                                                          sampling_vect)


    #adjust for sampling error
    correction_factor = calculate_correction_fac(impact_per_year, n_resampled_years, eis)
    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")


    ais.date = u_dt.str_to_date(year_list)
    ais.at_event = impact_per_year / correction_factor
    ais.frequency = np.ones(n_resampled_years)*sum(sampling_vect['events_per_year'])/n_resampled_years


    return ais, sampling_vect


def sample_annual_impacts(eis, n_resampled_years, sampling_vect=None):
    """Sample multiple events per year

    INPUTS:
        eis (impact class object): event impact set
        n_resampled_years (int): the target number of years the impact yearset shall
          contain.
        sampling_vect (dict): the sampling vector containing two arrays:
            selected_events (array): sampled events (len: total amount of sampled events)
            events_per_year (array): events per resampled year

    OUTPUTS:
        impact_per_year (array): resampled impact per year (length = n_resampled_years)
        sampling_vect (dict): the sampling vector containing two arrays:
            selected_events (array): sampled events (len: total amount of sampled events)
            events_per_year (array): events per resampled year
      """

    n_annual_events = np.sum(eis.frequency)

    if not sampling_vect:
        n_input_events = len(eis.event_id)
        sampling_vect = create_sampling_vector(n_resampled_years, n_annual_events, 
                                               n_input_events)
    
    impact_per_event = np.zeros(np.sum(sampling_vect['events_per_year']))
    impact_per_year = np.zeros(n_resampled_years)

    for idx_event, event in enumerate(sampling_vect):
        impact_per_event[idx_event] = eis.at_event[sampling_vect['selected_events'][event]]

    idx = 0
    for year in range(n_resampled_years):
        impact_per_year[year] = np.sum(impact_per_event[idx:(idx+sampling_vect['events_per_year'][year])])
        idx += sampling_vect['events_per_year'][year]

    return impact_per_year, sampling_vect


def sample_events(tot_n_events, n_input_events):
    """Sample events (length = tot_n_events) uniformely from an array (n_input_events)
    without replacement (if tot_n_events > n_input_events the input events are repeated 
                         (tot_n_events/n_input_events-1) times). 

    INPUT:
        tot_n_events (int): number of events to be sampled
        n_input_events (int): number of events contained in given event impact set (eis)

    OUTPUT:
        selected_events (array): uniformaly sampled events (length: len(tot_n_events))
      """

    repetitions = np.ceil(tot_n_events/(n_input_events-1)).astype('int')

    rng = default_rng()
    if repetitions >= 2:
        selected_events = np.round(rng.choice((n_input_events-1)*repetitions,
                                            size=tot_n_events, replace=False)/repetitions
                                 ).astype('int')
    else:
        selected_events = rng.choice((n_input_events-1), size=tot_n_events,
                                   replace=False).astype('int')

    
    return selected_events

def create_sampling_vector(n_resampled_years, n_annual_events, n_input_events):
    """Sample amount of events per year following a Poisson distribution

    INPUT:
        n_resampled_years (int): the target number of years the impact yearset shall
            contain.
        n_annual_events (int): number of events per year in given event impact set
        n_input_events (int): number of events contained in given event impact set (eis)

    OUTPUT:
        sampling_vect (array): sampling vector
        n_events_per_year (array): number of events per resampled year
    """
    
    if n_annual_events!=1:
        events_per_year = np.round(np.random.poisson(lam=n_annual_events,
                                                     size=n_resampled_years)).astype('int')
    else:
        events_per_year = np.ones(len(n_resampled_years))

    tot_n_events = sum(events_per_year)

    selected_events = sample_events(tot_n_events, n_input_events)
    
    sampling_vect = dict()
    sampling_vect = {'selected_events': selected_events, 'events_per_year': events_per_year}
    
    return sampling_vect

def calculate_correction_fac(impact_per_year, n_resampled_years, eis):
    """Apply a correction factor to ensure the expected annual impact (eai) of the annual
    impact set(ais) amounts to the eai of the event impact set (eis)

    INPUT:
        impact_per_year (array): resampled annual impact set before applying the correction factor
        n_resampled_years (int): the target number of years the annual impact set contains
        eis (impact class object): event impact set

    OUTPUT:
        correction_factor (int): the correction factor is calculated as eis_eai/ais_eai
    """

    ais_eai = np.sum(impact_per_year)/n_resampled_years
    eis_eai = np.sum(eis.frequency*eis.at_event)
    correction_factor = eis_eai/ais_eai
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)

    return correction_factor

def wrapper_multi_impact(list_impacts, n_resampled_years):
    """Compute the total impact of several event impact sets in one annual impact set

    INPUT:
        list_impacts (list): list of impact class objects
        n_resampled_years (int): the target number of years the impact yearset shall
            contain.
    OUTPUT:
        ais_total (impact class object): combined annual impact set for all given event impact sets
    """

    ais_total = np.zeros(n_resampled_years)
    for impact in list_impacts:
        ais_impact = impact_yearset(impact, n_resampled_years)
        ais_total += ais_impact

    return ais_total
