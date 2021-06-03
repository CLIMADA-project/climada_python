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

def impact_yearset(imp, sampled_years=None, sampling_vect=None, lam=None,
                   correction_fac=True):

    """Create a yearset of impacts (yimp) containing a probabilistic impact for each year
      in the sampled_years list (or for a list of sampled_years generated with the length
                                 of given sampled_years)
      by sampling events from the impact received as input with a Poisson distribution
      centered around n_events per year (n_events = sum(imp.frequency)).
      In contrast to the expected annual impact (eai) yimp contains impact values that
      differ among years. When correction factor is true, the yimp are scaled such
      that the average over all years is equal to the eai.

    Parameters
    -----------
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
        lam: int
            the applied Poisson distribution is centered around lam events per year
        correction_fac : boolean
            If True a correction factor is applied to the resulting yimp. It is
            scaled in such a way that the expected annual impact (eai) of the yimp
            equals the eai of the input impact

    Returns
    -------
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
        events_per_year = sample_n_events(n_sampled_years, imp, lam)
        sampling_vect = sample_events(events_per_year, imp.frequency)

    #compute impact per sampled_year
    imp_per_year = compute_imp_per_year(imp, sampling_vect)

    #copy imp object as basis for the yimp object
    yimp = copy.deepcopy(imp)

    #save imp_per_year in yimp
    if correction_fac: #adjust for sampling error
        correction_factor = calculate_correction_fac(imp_per_year, imp)
        yimp.at_event = imp_per_year / correction_factor
    else:
        yimp.at_event = imp_per_year

    #save calculations in yimp
    yimp.event_id = np.arange(1, n_sampled_years+1)
    yimp.tag['yimp object'] = True
    yimp.date = u_dt.str_to_date([str(date) + '-01-01' for date in sampled_years])
    yimp.frequency = np.ones(n_sampled_years)*sum(len(row) for row in sampling_vect
                                                            )/n_sampled_years

    return yimp, sampling_vect

def sample_n_events(n_sampled_years, imp, lam=None):
    """Create a sampling vector consisting of a subarray for each sampled_year that
    contains the index of the sampled events for that year

    Parameters
    -----------
        n_sampled_years : int
            The target number of years the impact yearset shall contain.
        imp : climada.engine.Impact()
            impact object containing impacts per event
        lam: int
            the applied Poisson distribution is centered around lam events per year

    Returns
    -------
        events_per_year : array
            Number of events per sampled year
    """
    if not lam:
        lam = np.sum(imp.frequency)

    #sample number of events per year
    if lam != 1:
        events_per_year = np.round(np.random.poisson(lam=lam,
                                                     size=n_sampled_years)).astype('int')
    else:
        events_per_year = np.ones(len(n_sampled_years))


    return events_per_year

def sample_events(events_per_year, freqs_orig):
    """Sample events uniformely from an array (indices_orig) without replacement
    (if sum(events_per_year) > n_input_events the input events are repeated
     (tot_n_events/n_input_events) times, by ensuring that the same events doens't
     occur more than once per sampled year).

    Parameters
    -----------
        events_per_year : array
            Number of events per sampled year
        freqs_orig : array
            Frequency of each input event

    Returns
    -------
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
      """

    sampling_vect = []

    indices_orig = np.arange(len(freqs_orig))

    freqs = freqs_orig
    indices = indices_orig

    #if check_doubling true: check that every event doesn't occur more than once per sampled year
    check_doubling = False

    #sample events for each sampled year
    for idx_year, amount_events in enumerate(events_per_year):

        probab_dis = freqs/np.sum(freqs)

        rng = default_rng()
        selected_events = rng.choice(indices, size=amount_events, replace=False,
                                     p=probab_dis).astype('int')

        if check_doubling: #check if an event occurs more than once in a year
            unique_events = np.unique(selected_events)
            #resample until each event occurs max. once per year
            while len(unique_events) != len(selected_events):
                rng = default_rng()
                selected_events = rng.choice(indices, size=amount_events, replace=False,
                                             p=probab_dis).astype('int')
            check_doubling = False

        idx_to_remove = [] #determine used events to remove them from sampling pool
        for event in selected_events:
            idx_to_remove.append(np.where(indices == event)[0][0])
        indices = np.delete(indices, idx_to_remove)
        freqs = np.delete(freqs, idx_to_remove)

        #add the original indices and frequencies to the pool if there are less events
        #in the pool than needed to fill the next sampled year
        if (idx_year < (len(events_per_year)-1)) and (
                len(indices) < events_per_year[idx_year+1]):
            indices = np.append(indices, indices_orig)
            freqs = np.append(freqs, freqs_orig)
            check_doubling = True

        sampling_vect.append(selected_events)

    return sampling_vect

def compute_imp_per_year(imp, sampling_vect):
    """Sample annual impacts from the given event_impacts according to the sampling dictionary

    Parameters
    -----------
        imp : climada.engine.Impact()
            impact object containing impacts per event
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.

    Returns
    -------
        imp_per_year: array
            Sampled impact per year (length = sampled_years)
      """

    imp_per_year = np.zeros(len(sampling_vect))
    for year, sampled_events in enumerate(sampling_vect):
        if sampled_events.size > 0:
            imp_per_year[year] = np.sum(imp.at_event[sampled_events])

    return imp_per_year

def calculate_correction_fac(imp_per_year, imp):
    """Calculate a correction factor that can be used to scale the yimp in such
    a way that the expected annual impact (eai) of the yimp amounts to the eai
    of the input imp

    Parameters
    -----------
        imp_per_year : array
            sampled yimp
        imp : climada.engine.Impact()
            impact object containing impacts per event

    Returns
    -------
        correction_factor: int
            The correction factor is calculated as imp_eai/yimp_eai
    """

    yimp_eai = np.sum(imp_per_year)/len(imp_per_year)
    imp_eai = np.sum(imp.frequency*imp.at_event)
    correction_factor = imp_eai/yimp_eai
    LOGGER.info("The correction factor amounts to %s", (correction_factor-1)*100)

    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")

    return correction_factor
