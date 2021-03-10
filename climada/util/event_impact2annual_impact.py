#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:59:03 2020

@author: carmensteinmann
"""

import copy
import logging
import numpy as np
from numpy.random import default_rng

from climada.util.dates_times import str_to_date

LOGGER = logging.getLogger(__name__)

def eis2ais(eis, number_of_years=None, amount_events=None,
            distribution=None, sampling_vect=None):

    """PURPOSE:
      convert an event (per occurrence) impact set (eis) into an annual impact
      set (ais), making reference to hazard.orig_yearset (if exists). If
      there is no yearset in hazard, generate an artificial year impact set
      by sampling the eis into n_years, where n_years is determined based on
      eis.frequency (n_years=1/min(eis.frequency), such that the yearset
      'spans' the period the impact is 'representative' for).

      The code does perform some consistency checks, but the user is
      ultimately repsonsible for results ;-)

      Note that for TS and TR, the TC hazard event set contains
      hazard.orig_yearset, not each sub-peril hazard event set might contain
      its own yearset. See climada_tc_hazard_set for a good example of how
      such an hazard.orig_yearset is constructed.
    CALLING SEQUENCE:
      entity=climada_entity_read('demo_today');
      hazard=climada_hazard_load('TCNA_today_small');
      eis=climada_eis_calc(entity,hazard);
      ais=climada_eis2ais(eis,hazard,number_of_years)
    EXAMPLE:
      ais=climada_eis2ais(climada_eis_calc('',hazard),hazard)
    INPUTS:
      eis: an event impact set (eis), as produced by climada_eis_calc
          (see there)
    OPTIONAL INPUT PARAMETERS:
      hazard: a hazard event set (either a struct or a full filename with
          path) which contains a yearset in hazard.orig_yearset
          Note that for TS and TR, the TC hazard event set contains
          hazard.orig_yearset, not each sub-peril hazard event set
          If empty, the hazard event set is inferred from
          eis.annotation_name, which often contains the filename (without
          path) of the hazard event set. If this is the case, this hazard is
          used, if not, the function prompts for the hazard event set to use.
          If hazard does neither contain a valid hazard struct, nor an
          existing hazard set name, one event per year is assumed and the eis
          is just replicated enough times to result in number_of_years (or,
          if number_of_years is not provided, ais=eis)
      number_of_years: the target number of years the impact yearset shall
          contain. If shorter than the yearset in the hazard set, just cut,
          otherwise replicate until target length is reached. No advanced
          technique, such as re-sampling, is performed (yet).
      sampling_vect: the sampling vector, techincal, see code (can be used to
          re-create the exact same yearset). Neeis to be obtained in a first
          call, i.e. [ais,sampling_vect]=climada_eis2ais(...) and then
          provided in subsequent calls(s) to obtain the exact same sampling
          structure of yearset, i.e ais=climada_eis2ais(...,sampling_vect)
      silent_mode: if =1, no stdout
    OUTPUTS:
      ais: the year impact set (ais), a struct with same fields as eis (such
          as Value, ed, ...) plus yyyy and orig_year_flag. All fields same
          content as in eis, except:
          yyyy(i): the year i
          impact(i): the sum of impact for year(i). Note that for a
              probabilitic hazard event set, there are ens_size+1 same years,
              the first instance being the original year.
          frequency(i): the annual frequency, =1
          orig_year_flag(i): =1 if year i is an original year, =0 else
          Hint: if you want to staore a ais back into an eis, note that there
          are two more fields in ais than eis: yyyy and orig_year_flag
      sampling_vect: the sampling vector, techincal, see code (can be used to
                     re-create the exact same yearset)
      """

    # OLD CODE - STILL NEEDED?
    #nr_years als input
    # if years are known:
    # events = list(EIS.calc_impact_year_set().keys())

    # event_date = date_to_str(eis.date)
    # event_year = [i.split('-', 1)[0] for i in event_date]

    #n_years_his = np.ceil(1/min(eis.frequency)).astype('int')
    # n_events = len(eis.event_id)


    #NUMBER OF EVENTS
    nonzero_pos = np.where(eis.at_event >= (10*np.finfo(float).eps))
    nonzero_impact = eis.at_event[nonzero_pos]
    sorted_impact = np.sort(nonzero_impact)
    n_annual_events = np.sum(eis.frequency[nonzero_pos])
    year_list = [str(date) + '-01-01' for date in np.arange(1, number_of_years+1).tolist()]


    if len(nonzero_pos) == 0:
        LOGGER.warning("No impact causing events.")

    if not np.all(eis.frequency == eis.frequency[0]):
        LOGGER.warning("The frequencies of the single events differ among each other."
                       "Please beware that this might influence the results.")


    # INITIALISATION
    #to do: add a flag to distinguish ais from other impact class objects
    # artificially generate a generic  annual impact set (by sampling the event impact set)
    # essentially a copy, reset some fields:
    ais = copy.deepcopy(eis)
    ais.event_id = [] # to indicate not by event any more
    ais.event_id = np.arange(1, number_of_years+1)
    ais.frequency = [] # init, see below
    ais.at_event = np.zeros([1, number_of_years+1])
    ais.date = []


    impact_per_year = np.zeros(number_of_years)

    # sample from the sorted impact

    if distribution is None: #for hazards such as RC where there's an event every year
        if not sampling_vect:
            sampling_vector = sampling_uniform(number_of_years, nonzero_pos)

        amount_events = np.ones(number_of_years)

        for idx_event, event in enumerate(sampling_vector):
            impact_per_year[idx_event] = sorted_impact[sampling_vector[event]]

        event_names = year_list
    elif distribution == 'Poisson':
        if not sampling_vect:
            sampling_vector, amount_events = sampling_poisson(number_of_years,
                                                              n_annual_events, nonzero_pos)


        impact_per_event = np.zeros(np.sum(amount_events))

        for idx_event, event in enumerate(sampling_vector):
            impact_per_event[idx_event] = sorted_impact[sampling_vector[event]]
            #event_names.append(year_list[year]*amount_events[year])

        idx = 0
        event_names = []
        for year in range(number_of_years):
            impact_per_year[year] = np.sum(impact_per_event[idx:(idx+amount_events[year])])
            idx += amount_events[year]


    #adjust for sampling error
    ais_ed = np.sum(impact_per_year)/number_of_years
    eis_ed = np.sum(eis.frequency*eis.at_event)
    correction_factor = eis_ed/ais_ed
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)
    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")


    ais.date = str_to_date(event_names)
    ais.at_event = impact_per_year / correction_factor
    ais.frequency = np.ones(number_of_years)*sum(amount_events)/number_of_years


    return ais, sampling_vector, amount_events

def sampling_uniform(number_events, nonzero_pos):
    """Sample uniformely from an array (nonzero_pos) for a given amount of
    events (number events)
      """

    repetitions = np.ceil(number_events/(len(nonzero_pos)-1)).astype('int')

    rng = default_rng()
    if repetitions >= 2:
        sampling_vector = np.round(rng.choice((len(nonzero_pos)-1)*repetitions,
                                              size=number_events, replace=False)/repetitions)
    else:
        sampling_vector = rng.choice((len(nonzero_pos)-1), size=number_events, replace=False)

    return sampling_vector

def sampling_poisson(number_of_years, n_annual_events, nonzero_pos):
    """Sample amount of events per year following a Poisson distribution
      """

    amount_events_per_year = np.round(np.random.poisson(lam=n_annual_events,
                                                        size=number_of_years)).astype('int')

    #non_zero_years =  np.where(amount_events_per_year != 0)
    number_events = sum(amount_events_per_year)

    sampling_vector = sampling_uniform(number_events, nonzero_pos)

    return sampling_vector, amount_events_per_year

def wrapper_multi_impact(list_impacts, number_of_years):
    """Compute the total impact of several event impact sets in one annual impact set
      """

    ais_total = np.zeros(number_of_years)
    for impact in list_impacts:
        ais_impact = eis2ais(impact, number_of_years)
        ais_total += ais_impact

    return ais_total
