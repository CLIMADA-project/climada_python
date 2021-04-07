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

def impact_yearset(event_impacts, sampled_years=None, sampling_dict=None, correction_fac=True):

    """Create an annual_impacts object containing a probabilistic impact for each year
      in the sampled_years list (or for a list of sampled_years generated with the length
                                 of given sampled_years)
      by sampling events from the existing input event_impacts with a Poisson distribution
      centered around n_events per year (n_events = sum(event_impacts.frequency)).
      In contrast to the expected annual impact (eai) annual_impacts contains impact values that
      differ among years (the correction factor can however be used to scale the annual_impacts
                          to fit the eai of the events_impacts object that is used to generated it)

    Parameters:
      event_impacts : climada.engine.Impact()
          impact object containing impacts per event
    Optional parameters:
        sampled_years : int or list
            Either an integer specifying the number of years to
            be sampled (labelled [0001,...,sampled_years]) or a list
            of years that shall be covered by the resulting annual_impacts.
            The default is a 1000 year-long list starting in the year 0001.
        sampling_dict : dict
            The sampling dictionary specifying how to sample the annual_impacts
            It consists of two arrays:
                selected_events: array
                    indices of sampled events in event_impacts.at_event()
                events_per_year: array
                    number of events per sampled year
            The sampling_dict needs to be obtained in a first call,
            i.e. [annual_impacts, sampling_dict] = climada_yearsets.impact_yearset(...)
            and can then be provided in subsequent calls(s) to obtain the exact same sampling
            (also for a different event_impacts object)
        correction_fac : boolean
            If True a correction factor is applied to the resulting annual_impacts. They are
            scaled in such a way that the expected annual impact (eai) of the annual_impacts
            equals the eai of the events_impacts

    Returns:
      annual_impacts : climada.engine.Impact()
          annual impacts for all sampled_years
      sampling_dict : dict
          the sampling dictionary containing two arrays:
              selected_events (array) : sampled events (len: total amount of sampled events)
              events_per_year (array) : events per sampled year
          Can be used to re-create the exact same annual_impacts yearset
      """


    if not sampled_years and not sampling_dict:
        sampled_years = list(range(1, 1001))
    elif isinstance(sampled_years, int):
        sampled_years = list(range(1, sampled_years+1))
    elif not sampled_years:
        sampled_years = list(range(1, len(sampling_dict['selected_events'])+1))
    elif sampling_dict and (len(sampled_years) != len(sampling_dict['events_per_year'])):
        LOGGER.info("The number of sampled_years and the length of the list of events_per_year "
                    "in the sampling_dict differ. The number of years contained in the "
                    "sampling_dict are used as number of sampled_years.")
        sampled_years = list(range(1, len(sampling_dict['selected_events'])+1))

    if sampling_dict and (
            np.sum(sampling_dict['events_per_year']) != len(sampling_dict['selected_events'])):
        raise ValueError("The sampling dictionary is faulty: the sum of selected events "
                         "does not correspond to the number of selected events.")

    n_sampled_years = len(sampled_years)

    if len(np.unique(event_impacts.frequency)) > 1:
        LOGGER.warning("The frequencies of the single events in the given event_impacts "
                       "differ among each other. Please beware that this will influence "
                       "the resulting annual_impacts as the events are sampled uniformaly "
                       "and different frequencies are (not yet) taken into account.")



    #create sampling dictionary if not given as input
    if not sampling_dict:
        sampling_dict = create_sampling_dict(n_sampled_years, event_impacts)

    #compute annual_impacts
    impact_per_year = compute_annual_impacts(event_impacts, sampling_dict)

    #copy event_impacts object as basis for the annual_impacts object
    annual_impacts = copy.deepcopy(event_impacts)

    #save impact_per_year in annual_impacts
    if correction_fac: #adjust for sampling error
        correction_factor = calculate_correction_fac(impact_per_year, event_impacts)
        annual_impacts.at_event = impact_per_year / correction_factor
    else:
        annual_impacts.at_event = impact_per_year

    annual_impacts.event_id = np.arange(1, n_sampled_years+1)
    annual_impacts.tag['annual_impacts object'] = True
    annual_impacts.date = u_dt.str_to_date([str(date) + '-01-01' for date in sampled_years])
    annual_impacts.frequency = np.ones(n_sampled_years)*np.sum(sampling_dict['events_per_year']
                                                            )/n_sampled_years


    return annual_impacts, sampling_dict

def create_sampling_dict(n_sampled_years, event_impacts):
    """Create a sampling dictionary consisting of the amount of events per sample year and the
    index of the sampled events

    Parameters:
        n_sampled_years : int
            The target number of years the impact yearset shall contain.
        event_impacts : climada.engine.Impact()
            impact object containing impacts per event

    Returns:
        sampling_dict : dict
            The sampling dictionary containing two arrays:
                selected_events (array): sampled events (len: total amount of sampled events)
                events_per_year (array): events per sampled year
    """
    n_annual_events = np.sum(event_impacts.frequency)
    n_input_events = len(event_impacts.frequency)
    #sample number of events per year
    if n_annual_events != 1:
        events_per_year = np.round(np.random.poisson(lam=n_annual_events,
                                                     size=n_sampled_years)).astype('int')
    else:
        events_per_year = np.ones(len(n_sampled_years))

    tot_n_events = np.sum(events_per_year)

    selected_events = sample_events(tot_n_events, n_input_events, event_impacts.frequency)

    sampling_dict = dict()
    sampling_dict = {'selected_events': selected_events, 'events_per_year': events_per_year}

    return sampling_dict

def sample_events(tot_n_events, n_input_events, freqs):
    """Sample events (length = tot_n_events) uniformely from an array (n_input_events)
    without replacement (if tot_n_events > n_input_events the input events are repeated
                         (tot_n_events/n_input_events) times).

    Parameters:
        tot_n_events : int
            Number of events to be sampled
        n_input_events : int
            Number of events contained in given event_impacts object
        freqs : array
            Frequency of each event (length: n_input_events)

    Returns:
        selected_events : array
            Uniformaly sampled events (length: tot_n_events)
      """

    repetitions = np.ceil(tot_n_events/n_input_events).astype('int')
    indices = np.tile(np.arange(n_input_events), repetitions)
    probab_dis = np.tile(freqs, repetitions)/np.sum(np.tile(freqs, repetitions))

    rng = default_rng()
    selected_events = rng.choice(indices, size=tot_n_events, replace=False,
                                 p=probab_dis).astype('int')


    return selected_events

def compute_annual_impacts(event_impacts, sampling_dict):
    """Sample annual impacts from the given event_impacts according to the sampling dictionary

    Parameters:
        event_impacts : climada.engine.Impact()
            impact object containing impacts per event
        sampling_dict : dict
            The sampling dictionary containing two arrays:
                selected_events (array) : sampled events (len: total amount of sampled events)
                events_per_year (array) : events per sampled year

    Returns:
        impact_per_year: array
            Sampled impact per year (length = n_sampled_years)
      """

    impact_per_event = np.zeros(np.sum(sampling_dict['events_per_year']))
    impact_per_year = np.zeros(len(sampling_dict['events_per_year']))

    for idx_event, event in enumerate(sampling_dict['selected_events']):
        impact_per_event[idx_event] = event_impacts.at_event[event]

    idx = 0
    for year in range(len(sampling_dict['events_per_year'])):
        impact_per_year[year] = np.sum(impact_per_event[idx:(idx+sampling_dict[
            'events_per_year'][year])])
        idx += sampling_dict['events_per_year'][year]

    return impact_per_year


def calculate_correction_fac(impact_per_year, event_impacts):
    """Calculate a correction factor that can be used to scale the annual_impacts in such
    a way that the expected annual impact (eai) of the annual_impacts amounts to the eai
    of the input event_impacts

    Parameters:
        impact_per_year : array
            sampled annual_impacts
        event_impacts : climada.engine.Impact()
            impact object containing impacts per event

    Returns:
        correction_factor: int
            The correction factor is calculated as event_impacts_eai/annual_impacts_eai
    """

    annual_impacts_eai = np.sum(impact_per_year)/len(impact_per_year)
    event_impacts_eai = np.sum(event_impacts.frequency*event_impacts.at_event)
    correction_factor = event_impacts_eai/annual_impacts_eai
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)

    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")

    return correction_factor
