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
Define functions to handle impact_yearsets
"""

import copy
import logging
import numpy as np
from numpy.random import default_rng

import climada.util.dates_times as u_dt

LOGGER = logging.getLogger(__name__)

def impact_yearset(event_impacts, sampled_years=None, sampling_vect=None, correction_fac=True):

    """PURPOSE:
      Create an annual_impacts object containing a probabilistic impact for each year
      in the sampled_years list (or a list generated with the length of given sampled_years)
      by sampling events from the existing input event_impacts with a Poisson distribution
      centered around n_events per year ( n_events = sum(event_impacts.frequency)).

    INPUTS:
      event_impacts (impact object): impact per event
    OPTIONAL INPUT:
        sampled_years (int or list): either an integer specifying the amount of years to
            be sampled or a list of years that shall be covered by the resulting annual_impacts
            default: a 1000 year-long list starting in the year 0001
        sampling_vect (dict): the sampling vector specifying which events contained in the
            events_impacts are selected in the newly created annual_impacts and the number
            of events per year in the annual_impacts, therefore the sampling_vect contains
            two arrays:
                selected_events (array): sampled events (len: total amount of sampled events)
                events_per_year (array): events per sampled year
            The sampling_vector needs to be obtained in a first call,
            i.e. [annual_impacts, sampling_vect] = climada_yearsets.impact_yearset(...)
            and can then be provided in subsequent calls(s) to obtain the exact same sampling
            (also for a different event_impacts object)
        correction_fac (boolean): if True the resulting annual_impacts are scaled in such a way
            that the expected annual impact (eai) of the annual_impacts = eai of events_impacts

    OUTPUTS:
      annual_impacts(impact object): annual impacts for all sampled_years
      sampling_vect (dict): the sampling vector containing two arrays:
          selected_events (array): sampled events (len: total amount of sampled events)
          events_per_year (array): events per sampled year
          Can be used to re-create the exact same annual_impacts yearset
      """


    if not sampled_years and not sampling_vect:
        sampled_years = np.arange(1, 1000+1).tolist()
    elif not sampled_years:
        sampled_years = np.arange(1, len(sampling_vect['selected_events'])+1).tolist()
    # elif isinstance(sampled_years, int) and sampling_vect:
        #test sampled_years = len(sampling_vect['selected_events'])
        #if not fulfilled: logger warning!
    # elif isinstance(sampled_years, list) and sampling_vect:
        #test len(sampled_years) = sampling_vect
        #if not fulfilled: logger warning!
    elif isinstance(sampled_years, int):
        sampled_years = np.arange(1, sampled_years+1).tolist() #problem to change the input var?

    year_list = [str(date) + '-01-01' for date in sampled_years]
    n_sampled_years = len(year_list)

    if not np.all(event_impacts.frequency == event_impacts.frequency[0]):
        LOGGER.warning("The frequencies of the single events in the given event_impacts"
                       "differ among each other. Please beware that this will influence"
                       "the resulting annual_impacts as the events are sampled uniformaly"
                       "and different frequencies are (not yet) taken into account.")


    # INITIALISATION
    # to do: add a flag to distinguish annual_impacts from other impact class objects
    # artificially generate a generic  annual impact set (by sampling the event impact set)
    annual_impacts = copy.deepcopy(event_impacts)
    annual_impacts.event_id = [] # to indicate not by event any more
    annual_impacts.event_id = np.arange(1, n_sampled_years+1)
    annual_impacts.frequency = [] # init, see below
    annual_impacts.at_event = np.zeros([1, n_sampled_years+1])
    annual_impacts.date = []

    #create sampling vector if not given as input
    if not sampling_vect:
        n_annual_events = np.sum(event_impacts.frequency)
        n_input_events = len(event_impacts.event_id)
        sampling_vect = create_sampling_vector(n_sampled_years, n_annual_events,
                                               n_input_events)

    #compute annual_impacts
    impact_per_year = compute_annual_impacts(event_impacts, n_sampled_years,
                                             sampling_vect)


    if correction_fac: #adjust for sampling error
        correction_factor = calculate_correction_fac(impact_per_year, event_impacts)
        annual_impacts.at_event = impact_per_year / correction_factor
    else:
        annual_impacts.at_event = impact_per_year

    annual_impacts.date = u_dt.str_to_date(year_list)
    annual_impacts.frequency = np.ones(n_sampled_years)*sum(sampling_vect['events_per_year']
                                                            )/n_sampled_years


    return annual_impacts, sampling_vect

def create_sampling_vector(n_sampled_years, n_annual_events, n_input_events):
    """Sample amount of events per year following a Poisson distribution

    INPUT:
        n_sampled_years (int): the target number of years the impact yearset shall
            contain.
        n_annual_events (int): number of events per year in given event_impacts object
        n_input_events (int): number of events contained in given event_impacts object

    OUTPUT:
        sampling_vect (dict): the sampling vector containing two arrays:
            selected_events (array): sampled events (len: total amount of sampled events)
            events_per_year (array): events per sampled year
        n_events_per_year (array): number of events per sampled year
    """

    if n_annual_events != 1:
        events_per_year = np.round(np.random.poisson(lam=n_annual_events,
                                                     size=n_sampled_years)).astype('int')
    else:
        events_per_year = np.ones(len(n_sampled_years))

    tot_n_events = sum(events_per_year)

    selected_events = sample_events(tot_n_events, n_input_events)

    sampling_vect = dict()
    sampling_vect = {'selected_events': selected_events, 'events_per_year': events_per_year}

    return sampling_vect

def sample_events(tot_n_events, n_input_events):
    """Sample events (length = tot_n_events) uniformely from an array (n_input_events)
    without replacement (if tot_n_events > n_input_events the input events are repeated
                         (tot_n_events/n_input_events-1) times).

    INPUT:
        tot_n_events (int): number of events to be sampled
        n_input_events (int): number of events contained in given event impact set (event_impacts)

    OUTPUT:
        selected_events (array): uniformaly sampled events (length: len(tot_n_events))
      """

    repetitions = np.ceil(tot_n_events/n_input_events).astype('int')
    indices = np.tile(np.arange(n_input_events), repetitions)

    rng = default_rng()
    selected_events = rng.choice(indices, size=tot_n_events,
                                 replace=False).astype('int')


    return selected_events

def compute_annual_impacts(event_impacts, n_sampled_years, sampling_vect):
    """Sample annual impacts from the given event_impacts

    INPUTS:
        event_impacts (impact object): event impact
        n_sampled_years (int): the target number of years the impact yearset shall
          contain.
        sampling_vect (dict): the sampling vector containing two arrays:
            selected_events (array): sampled events (len: total amount of sampled events)
            events_per_year (array): events per sampled year

    OUTPUTS:
        impact_per_year (array): sampled impact per year (length = n_sampled_years)
      """

    impact_per_event = np.zeros(np.sum(sampling_vect['events_per_year']))
    impact_per_year = np.zeros(n_sampled_years)

    for idx_event, event in enumerate(sampling_vect['selected_events']):
        impact_per_event[idx_event] = event_impacts.at_event[sampling_vect[
            'selected_events'][event]]

    idx = 0
    for year in range(n_sampled_years):
        impact_per_year[year] = np.sum(impact_per_event[idx:(idx+sampling_vect[
            'events_per_year'][year])])
        idx += sampling_vect['events_per_year'][year]

    return impact_per_year


def calculate_correction_fac(impact_per_year, event_impacts):
    """Calculate a correction factor that can be used to scale the annual_impacts in such
    a way that the expected annual impact (eai) of the annual_impacts amounts to the eai
    of the input event_impacts

    INPUT:
        impact_per_year (array): sampled annual_impacts
        event_impacts (impact object): event impact

    OUTPUT:
        correction_factor (int): the correction factor is calculated as
            event_impacts_eai/annual_impacts_eai
    """

    annual_impacts_eai = np.sum(impact_per_year)/len(impact_per_year)
    event_impacts_eai = np.sum(event_impacts.frequency*event_impacts.at_event)
    correction_factor = event_impacts_eai/annual_impacts_eai
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)
    
    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")
    
    return correction_factor
