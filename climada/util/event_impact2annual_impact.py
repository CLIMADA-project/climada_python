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

def eis2ais(eis, nr_resampled_years=None, year_list=None, distribution=None,
            sampling_vect=None, nr_events_per_year=None):

    """PURPOSE:
      convert an event (per occurrence) impact set (eis) into an annual impact
      set (ais).

    INPUTS:
      eis: an event impact set (eis)
    OPTIONAL INPUT PARAMETERS:
      nr_resampled_years(int): the target number of years the impact yearset shall
          contain.
      year_list (list): list of years for the resulting annual impact set
          (by default a list starting on the 01-01-0001 is generated)
      sampling_vect: the sampling vector, technical, see code (can be used to
          re-create the exact same yearset). Needs to be obtained in a first
          call, i.e. [ais,sampling_vect]=climada_eis2ais(...) and then
          provided in subsequent calls(s) to obtain the exact same sampling
          structure of yearset, i.e ais=climada_eis2ais(...,sampling_vect)
     nr_events_per_year (array): amount of resampled events per year (length = nr_resampled_years),
         can be reused similar to the sampling_vect (to do: combine these two in one
                                                     variable?)
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


    if not year_list:
        year_list = [str(date) + '-01-01' for date in np.arange(1, nr_resampled_years+1).tolist()]


    if not np.all(eis.frequency == eis.frequency[0]):
        LOGGER.warning("The frequencies of the single events differ among each other."
                       "Please beware that this might influence the results.")


    # INITIALISATION
    # to do: add a flag to distinguish ais from other impact class objects
    # artificially generate a generic  annual impact set (by sampling the event impact set)
    ais = copy.deepcopy(eis)
    ais.event_id = [] # to indicate not by event any more
    ais.event_id = np.arange(1, nr_resampled_years+1)
    ais.frequency = [] # init, see below
    ais.at_event = np.zeros([1, nr_resampled_years+1])
    ais.date = []


    # sample from the given event impact set
    if distribution == 'Poisson':
        [impact_per_year, nr_events,
         sampling_vect] = resample_multiple_annual_events(eis, nr_resampled_years,
                                                          nr_events_per_year, sampling_vect=None)
    else: #for hazards such as RC where there is exactly one event every year
        [impact_per_year, nr_events,
         sampling_vect] = resample_single_annual_event(eis, nr_resampled_years,
                                                       sampling_vect)


    #adjust for sampling error
    ais_ed = np.sum(impact_per_year)/nr_resampled_years
    eis_ed = np.sum(eis.frequency*eis.at_event)
    correction_factor = eis_ed/ais_ed
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)
    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")


    ais.date = str_to_date(year_list)
    ais.at_event = impact_per_year / correction_factor
    ais.frequency = np.ones(nr_resampled_years)*sum(nr_events)/nr_resampled_years


    return ais, sampling_vect, nr_events

def resample_single_annual_event(eis, nr_resampled_years, sampling_vect=None):
    """Sample one single annual event

    INPUTS:
        eis (impact class): event impact set
        nr_resampled_years (int): the target number of years the impact yearset shall
          contain.
        sampling_vect (array): the sampling vector

    OUTPUTS:
        impact_per_year (array): resampled impact per year (length = nr_resampled_years)
        nr_events (array): amount of resampled events per year (length = nr_resampled_years)
        sampling_vect (array): the sampling vector
      """


    impact_per_year = np.zeros(nr_resampled_years)
    if not sampling_vect:
        nr_input_events = len(eis.event_id)
        sampling_vect = sampling_uniform(nr_resampled_years, nr_input_events)

    for idx_event, event in enumerate(sampling_vect):
        impact_per_year[idx_event] = eis.at_event[sampling_vect[event]]

    nr_events = np.ones(nr_resampled_years)

    return impact_per_year, nr_events, sampling_vect

def resample_multiple_annual_events(eis, nr_resampled_years, nr_events=None, sampling_vect=None):
    """Sample multiple events per year

    INPUTS:
        eis (impact class): event impact set
        nr_resampled_years (int): the target number of years the impact yearset shall
          contain.
        nr_annual_events (int): number of events per year in given event impact set
        sampling_vect (array): the sampling vector

    OUTPUTS:
        impact_per_year (array): resampled impact per year (length = nr_resampled_years)
        nr_events (array): amount of resampled events per year (length = nr_resampled_years)
        sampling_vect (array): the sampling vector
      """

    nr_annual_events = np.sum(eis.frequency)

    if not sampling_vect:
        nr_input_events = len(eis.event_id)
        sampling_vect, nr_events = sampling_poisson(nr_resampled_years,
                                                    nr_annual_events, nr_input_events)

    impact_per_event = np.zeros(np.sum(nr_events))
    impact_per_year = np.zeros(nr_resampled_years)

    for idx_event, event in enumerate(sampling_vect):
        impact_per_event[idx_event] = eis.at_event[sampling_vect[event]]

    idx = 0
    for year in range(nr_resampled_years):
        impact_per_year[year] = np.sum(impact_per_event[idx:(idx+nr_events[year])])
        idx += nr_events[year]

    return impact_per_year, nr_events, sampling_vect


def sampling_uniform(tot_nr_events, nr_input_events):
    """Sample uniformely from an array (nr_input_events) for a given amount of
    events (number events)

    INPUT:
        tot_nr_events (int): number of events to be resampled
        nr_input_events (int): number of events contained in given event impact set (eis)

    OUTPUT:
        sampling_vect (array): sampling vector
      """

    repetitions = np.ceil(tot_nr_events/(nr_input_events-1)).astype('int')

    rng = default_rng()
    if repetitions >= 2:
        sampling_vect = np.round(rng.choice((nr_input_events-1)*repetitions,
                                            size=tot_nr_events, replace=False)/repetitions
                                 ).astype('int')
    else:
        sampling_vect = rng.choice((nr_input_events-1), size=tot_nr_events,
                                   replace=False).astype('int')

    return sampling_vect

def sampling_poisson(nr_resampled_years, nr_annual_events, nr_input_events):
    """Sample amount of events per year following a Poisson distribution

    INPUT:
        nr_resampled_years (int): the target number of years the impact yearset shall
            contain.
        nr_annual_events (int): number of events per year in given event impact set
        nr_input_events (int): number of events contained in given event impact set (eis)

    OUTPUT:
        sampling_vect (array): sampling vector
        nr_events_per_year (array): number of events per resampled year
    """

    nr_events_per_year = np.round(np.random.poisson(lam=nr_annual_events,
                                                    size=nr_resampled_years)).astype('int')

    #non_zero_years =  np.where(nr_events_per_year != 0)
    tot_nr_events = sum(nr_events_per_year)

    sampling_vect = sampling_uniform(tot_nr_events, nr_input_events)

    return sampling_vect, nr_events_per_year

def wrapper_multi_impact(list_impacts, nr_resampled_years):
    """Compute the total impact of several event impact sets in one annual impact set

    INPUT:
        list_impacts (list): list of impact class objects
        nr_resampled_years (int): the target number of years the impact yearset shall
            contain.
    OUTPUT:
        ais_total (impact class): combined annual impact set for all given event impact sets
    """

    ais_total = np.zeros(nr_resampled_years)
    for impact in list_impacts:
        ais_impact = eis2ais(impact, nr_resampled_years)
        ais_total += ais_impact

    return ais_total
