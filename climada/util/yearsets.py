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

def impact_yearset(imp, sampled_years=None, sampling_vect=None, correction_fac=True):

    """Create an yearset of impacts (yimp) containing a probabilistic impact for each year
      in the sampled_years list (or for a list of sampled_years generated with the length
                                 of given sampled_years)
      by sampling events from the impact received as input with a Poisson distribution
      centered around n_events per year (n_events = sum(imp.frequency)).
      In contrast to the expected annual impact (eai) yimp contains impact values that
      differ among years (the correction factor can however be used to scale the yimp
                          to fit the eai of the impact that is used to generated it)

    Parameters:
      imp : climada.engine.Impact()
          impact object containing impacts per event
    Optional parameters:
        sampled_years : int or list
            Either an integer specifying the number of years to
            be sampled (labelled [0001,...,sampled_years]) or a list
            of years that shall be covered by the resulting yimp.
            The default is a 1000 year-long list starting in the year 0001.
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
            It needs to be obtained in a first call,
            i.e. [yimp, sampling_vect] = climada_yearsets.impact_yearset(...)
            and can then be provided in subsequent calls(s) to obtain the exact same sampling
            (also for a different imp object)
        correction_fac : boolean
            If True a correction factor is applied to the resulting yimp. It is
            scaled in such a way that the expected annual impact (eai) of the yimp
            equals the eai of the input impact

    Returns:
        yimp : climada.engine.Impact()
             yearset of impacts containing annual impacts for all sampled_years
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
            Can be used to re-create the exact same yimp
      """

    if not sampled_years and not sampling_vect:
        sampled_years = list(range(1, 1001))
    elif isinstance(sampled_years, int):
        sampled_years = list(range(1, sampled_years+1))
    elif not sampled_years:
        sampled_years = list(range(1, len(sampling_vect)+1))
    elif sampling_vect and (len(sampled_years) != len(sampling_vect)):
        LOGGER.info("The number of sampled_years and the length of the list of events_per_year "
                    "in the sampling_vect differ. The number of years contained in the "
                    "sampling_vect are used as number of sampled_years.")
        sampled_years = list(range(1, len(sampling_vect)+1))

    n_sampled_years = len(sampled_years)

    #create sampling vector if not given as input
    if not sampling_vect:
        sampling_vect = create_sampling_vect(n_sampled_years, imp)

    #compute impact per sampled_year
    impact_per_year = np.zeros(len(sampling_vect))

    for year, sampled_events in enumerate(sampling_vect):
        impact_per_year[year] = np.sum(imp.at_event[sampled_events])

    #copy imp object as basis for the yimp object
    yimp = copy.deepcopy(imp)

    #save impact_per_year in yimp
    if correction_fac: #adjust for sampling error
        correction_factor = calculate_correction_fac(impact_per_year, imp)
        yimp.at_event = impact_per_year / correction_factor
    else:
        yimp.at_event = impact_per_year

    yimp.event_id = np.arange(1, n_sampled_years+1)
    yimp.tag['yimp object'] = True
    yimp.date = u_dt.str_to_date([str(date) + '-01-01' for date in sampled_years])
    yimp.frequency = np.ones(n_sampled_years)*sum(len(row) for row in sampling_vect
                                                            )/n_sampled_years

    return yimp, sampling_vect

def create_sampling_vect(n_sampled_years, imp, lam=None):
    """Create a sampling vector consisting of a subarray for each sampled_year that
    contains the index of the sampled events for that year

    Parameters:
        n_sampled_years : int
            The target number of years the impact yearset shall contain.
        imp : climada.engine.Impact()
            impact object containing impacts per event
        lam: int
            the applied Poisson distribution is centered around lam events per year

    Returns:
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
    """
    if not lam:
        n_annual_events = np.sum(imp.frequency)

    #sample number of events per year
    if n_annual_events != 1:
        events_per_year = np.round(np.random.poisson(lam=n_annual_events,
                                                     size=n_sampled_years)).astype('int')
    else:
        events_per_year = np.ones(len(n_sampled_years))

    #create a sampling vector and check if an event occurs several times in one year
    #if this is the case the sampling vector is created again
    run = 0
    sampling_vect = []
    y_unique_events = []
    while run<1 or (
            sum(len(row) for row in y_unique_events) != sum(len(row) for row in sampling_vect)):

        selected_events = sample_events(np.sum(events_per_year), imp.frequency)

        idx = 0
        for year in range(n_sampled_years):
            sampling_vect.append(selected_events[idx:(idx+events_per_year[year])])
            idx += idx+events_per_year[year]

        y_unique_events = list(np.unique(row) for row in sampling_vect)
        run += run +1

    return sampling_vect

def sample_events(tot_n_events, freqs):
    """Sample events (length = tot_n_events) uniformely from an array (n_input_events)
    without replacement (if tot_n_events > n_input_events the input events are repeated
                         (tot_n_events/n_input_events) times).

    Parameters:
        tot_n_events : int
            Number of events to be sampled
        freqs : array
            Frequency of each input event (length: n_input_events)

    Returns:
        selected_events : array
            Uniformaly sampled events (length: tot_n_events)
      """

    n_input_events = len(freqs)
    repetitions = np.ceil(tot_n_events/n_input_events).astype('int')
    indices = np.tile(np.arange(n_input_events), repetitions)
    probab_dis = np.tile(freqs, repetitions)/np.sum(np.tile(freqs, repetitions))

    rng = default_rng()
    selected_events = rng.choice(indices, size=tot_n_events, replace=False,
                                 p=probab_dis).astype('int')

    return selected_events

def calculate_correction_fac(impact_per_year, imp):
    """Calculate a correction factor that can be used to scale the yimp in such
    a way that the expected annual impact (eai) of the yimp amounts to the eai
    of the input imp

    Parameters:
        impact_per_year : array
            sampled yimp
        imp : climada.engine.Impact()
            impact object containing impacts per event

    Returns:
        correction_factor: int
            The correction factor is calculated as imp_eai/yimp_eai
    """

    yimp_eai = np.sum(impact_per_year)/len(impact_per_year)
    imp_eai = np.sum(imp.frequency*imp.at_event)
    correction_factor = imp_eai/yimp_eai
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)

    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")

    return correction_factor
