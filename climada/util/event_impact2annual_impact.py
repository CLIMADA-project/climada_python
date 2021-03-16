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

def eis2ais(eis, number_of_years=None, distribution=None, 
    sampling_vect=None):

    """PURPOSE:
      convert an event (per occurrence) impact set (eis) into an annual impact
      set (ais). 

    INPUTS:
      eis: an event impact set (eis)
    OPTIONAL INPUT PARAMETERS:
      number_of_years(int): the target number of years the impact yearset shall
          contain.
      sampling_vect: the sampling vector, technical, see code (can be used to
          re-create the exact same yearset). Needs to be obtained in a first
          call, i.e. [ais,sampling_vect]=climada_eis2ais(...) and then
          provided in subsequent calls(s) to obtain the exact same sampling
          structure of yearset, i.e ais=climada_eis2ais(...,sampling_vect)
    OUTPUTS:
      ais: the year impact set (ais), a struct with same fields as eis (such
          as Value, ed, ...). All fields same content as in eis, except:
          date(i): the year i
          impact(i): the sum of impact for year(i).
          frequency(i): the annual frequency, =1
      sampling_vect: the sampling vector, technical, see code (can be used to
                     re-create the exact same yearset)
      """

    # OLD CODE - STILL NEEDED?
    # if years are known:
    # events = list(EIS.calc_impact_year_set().keys())

    # event_date = date_to_str(eis.date)
    # event_year = [i.split('-', 1)[0] for i in event_date]

    #n_years_his = np.ceil(1/min(eis.frequency)).astype('int')
    # n_events = len(eis.event_id)


    #NUMBER OF EVENTS
    nr_input_events = len(eis.event_id)
    n_annual_events = np.sum(eis.frequency)
    year_list = [str(date) + '-01-01' for date in np.arange(1, number_of_years+1).tolist()]


    if nr_input_events == 0:
        LOGGER.warning("The event impact does not contain events.")

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


    # sample from the given event impact set
    if distribution == 'Poisson':
        [impact_per_year, amount_events,
         sampling_vect] = resample_multiple_annual_events(eis, number_of_years, nr_input_events,
                                                            n_annual_events, sampling_vect=None)
    else: #for hazards such as RC where there is exactly one event every year
        [impact_per_year, amount_events,
         sampling_vect] = resample_single_annual_event(eis, number_of_years,
                                                       nr_input_events, sampling_vect)
        


    #adjust for sampling error
    ais_ed = np.sum(impact_per_year)/number_of_years
    eis_ed = np.sum(eis.frequency*eis.at_event)
    correction_factor = eis_ed/ais_ed
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)
    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")


    ais.date = str_to_date(year_list)
    ais.at_event = impact_per_year / correction_factor
    ais.frequency = np.ones(number_of_years)*sum(amount_events)/number_of_years


    return ais, sampling_vect, amount_events

def resample_single_annual_event(eis, number_of_years, nr_input_events, sampling_vect=None):
    """Sample one single annual event
    
    INPUTS:
        eis (impact class): event impact set
        number_of_years (int): the target number of years the impact yearset shall
          contain.
        nr_input_events (int): number of events in event impact set
        sampling_vect (array): the sampling vector
        
    OUTPUTS:
        impact_per_year (array): resampled impact per year (length = number_of_years)
        amount_events (array): amount of resampled events per year (length = number_of_years)
        sampling_vect (array): the sampling vector
      """
    impact_per_year = np.zeros(number_of_years)
    if not sampling_vect:
        sampling_vect = sampling_uniform(number_of_years, nr_input_events)

    for idx_event, event in enumerate(sampling_vect):
        impact_per_year[idx_event] = eis.at_event[sampling_vect[event]]

    amount_events = np.ones(number_of_years)

    return impact_per_year, amount_events, sampling_vect

def resample_multiple_annual_events(eis, number_of_years, nr_input_events,
                                    n_annual_events, sampling_vect=None):
    """Sample multiple events per year
    
    INPUTS:
        eis (impact class): event impact set
        number_of_years (int): the target number of years the impact yearset shall
          contain.
        nr_input_events (int): number of events in event impact set
        n_annual_events (int): number of events per year in given event impact set
        sampling_vect (array): the sampling vector
        
    OUTPUTS:
        impact_per_year (array): resampled impact per year (length = number_of_years)
        amount_events (array): amount of resampled events per year (length = number_of_years)
        sampling_vect (array): the sampling vector
      """
    if not sampling_vect:
        sampling_vect, amount_events = sampling_poisson(number_of_years,
                                                          n_annual_events, nr_input_events)

    impact_per_event = np.zeros(np.sum(amount_events))
    impact_per_year = np.zeros(number_of_years)

    for idx_event, event in enumerate(sampling_vect):
        impact_per_event[idx_event] = eis.at_event[sampling_vect[event]]

    idx = 0
    for year in range(number_of_years):
        impact_per_year[year] = np.sum(impact_per_event[idx:(idx+amount_events[year])])
        idx += amount_events[year]

    return impact_per_year, amount_events, sampling_vect


def sampling_uniform(number_events, nr_input_events):
    """Sample uniformely from an array (nr_input_events) for a given amount of
    events (number events)
    
    INPUT:
        number_events (int): number of events to be resampled
        nr_input_events (int): number of events contained in given event impact set (eis)
        
    OUTPUT:
        sampling_vect (array): sampling vector
      """

    repetitions = np.ceil(number_events/(nr_input_events-1)).astype('int')

    rng = default_rng()
    if repetitions >= 2:
        sampling_vect = np.round(rng.choice((nr_input_events-1)*repetitions,
                                              size=number_events, replace=False)/repetitions
                                   ).astype('int')
    else:
        sampling_vect = rng.choice((nr_input_events-1), size=number_events,
                                     replace=False).astype('int')

    return sampling_vect

def sampling_poisson(number_of_years, n_annual_events, nr_input_events):
    """Sample amount of events per year following a Poisson distribution
      
    INPUT:
        number_of_years (int): the target number of years the impact yearset shall
            contain.
        n_annual_events (int): number of events per year in given event impact set
        nr_input_events (int): number of events contained in given event impact set (eis)
        
    OUTPUT:
        sampling_vect (array): sampling vector
        amount_events_per_year (array): number of events per resampled year
    """

    amount_events_per_year = np.round(np.random.poisson(lam=n_annual_events,
                                                        size=number_of_years)).astype('int')

    #non_zero_years =  np.where(amount_events_per_year != 0)
    number_events = sum(amount_events_per_year)

    sampling_vect = sampling_uniform(number_events, nr_input_events)

    return sampling_vect, amount_events_per_year

def wrapper_multi_impact(list_impacts, number_of_years):
    """Compute the total impact of several event impact sets in one annual impact set
    
    INPUT:
        list_impacts (list): list of impact class objects
        number_of_years (int): the target number of years the impact yearset shall
            contain.
    OUTPUT:
        ais_total (impact class): combined annual impact set for all given event impact sets
    """

    ais_total = np.zeros(number_of_years)
    for impact in list_impacts:
        ais_impact = eis2ais(impact, number_of_years)
        ais_total += ais_impact

    return ais_total
