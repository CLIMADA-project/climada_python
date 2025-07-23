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
Define functions to handle impact_yearsets
"""

import logging

import numpy as np
from numpy.random import default_rng

import climada.util.dates_times as u_dt
from climada.engine import Impact

LOGGER = logging.getLogger(__name__)


def impact_yearset(
    imp, sampled_years, lam=None, correction_fac=False, with_replacement=True, seed=None
):
    """Create a yearset of impacts (yimp) containing a probabilistic impact for each year
    in the sampled_years list by sampling events from the impact received as input with a
    Poisson distribution centered around lam per year (lam = sum(imp.frequency)).
    In contrast to the expected annual impact (eai) yimp contains impact values that
    differ among years. When correction factor is true, the yimp are scaled such
    that the average over all years is equal to the eai.

    Parameters
    -----------
        imp : climada.engine.Impact()
            impact object containing impacts per event
        sampled_years : list
            A list of years that shall be covered by the resulting yimp.
        with_replacement : bool, optional
            If True, impact events are sampled with replacement. If False, events are sampled
            without replacement. Sampling without replacement can yield distorted samples if
            frequencies of different events are unqual. Defaults to True.
        seed : Any, optional
            seed for the default bit generator
            default: None

    Optional parameters
        lam: int
            The applied Poisson distribution is centered around lam events per year.
            If no lambda value is given, the default lam = sum(imp.frequency) is used.
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
            Can be used to re-create the exact same yimp.
    """

    n_sampled_years = len(sampled_years)

    # create sampling vector
    if not lam:
        lam = np.sum(imp.frequency)
    events_per_year = sample_from_poisson(n_sampled_years, lam, seed=seed)
    sampling_vect = sample_events(
        events_per_year, imp.frequency, with_replacement=with_replacement, seed=seed
    )

    # compute impact per sampled_year
    yimp = impact_yearset_from_sampling_vect(
        imp, sampled_years, sampling_vect, correction_fac=correction_fac
    )

    return yimp, sampling_vect


def impact_yearset_from_sampling_vect(
    imp, sampled_years, sampling_vect, correction_fac=False
):
    """Create a yearset of impacts (yimp) containing a probabilistic impact for each year
    in the sampled_years list by sampling events from the impact received as input following
    the sampling vector provided.
    In contrast to the expected annual impact (eai) yimp contains impact values that
    differ among years. When correction factor is true, the yimp are scaled such
    that the average over all years is equal to the eai.

    Parameters
    -----------
        imp : climada.engine.Impact()
            impact object containing impacts per event
        sampled_years : list
            A list of years that shall be covered by the resulting yimp.
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
            It needs to be obtained in a first call,
            i.e. [yimp, sampling_vect] = climada_yearsets.impact_yearset(...)
            and can then be provided in this function to obtain the exact same sampling
            (also for a different imp object)

    Optional parameter
        correction_fac : boolean
            If True a correction factor is applied to the resulting yimp. It is
            scaled in such a way that the expected annual impact (eai) of the yimp
            equals the eai of the input impact

    Returns
    -------
        yimp : climada.engine.Impact()
             yearset of impacts containing annual impacts for all sampled_years

    """

    # compute impact per sampled_year
    imp_per_year = compute_imp_per_year(imp, sampling_vect)

    if correction_fac:  # adjust for sampling error
        correction_factor = calculate_correction_fac(imp_per_year, imp)
        imp_per_year = imp_per_year.astype(float) * correction_factor
        LOGGER.info("The correction factor is %s.", correction_factor)

    # create yearly impact object
    yimp = Impact(
        at_event=imp_per_year,
        date=u_dt.str_to_date([str(date) + "-01-01" for date in sampled_years]),
        event_name=np.arange(1, len(sampled_years) + 1),
        event_id=np.arange(1, len(sampled_years) + 1),
        unit=imp.unit,
        haz_type=imp.haz_type,
        frequency=np.ones(len(sampled_years)) / len(sampled_years),
    )

    yimp.aai_agg = np.sum(yimp.frequency * yimp.at_event)

    return yimp


def sample_from_poisson(n_sampled_years, lam, seed=None):
    """Sample the number of events for n_sampled_years

    Parameters
    -----------
        n_sampled_years : int
            The target number of years the impact yearset shall contain.
        lam: float
            the applied Poisson distribution is centered around lambda events per year
        seed : int, optional
            seed for numpy.random, will be set if not None
            default: None

    Returns
    -------
        events_per_year : np.ndarray
            Number of events per sampled year
    """
    if seed is not None:
        np.random.seed(seed)
    return np.round(np.random.poisson(lam=lam, size=n_sampled_years)).astype("int")


def sample_events(events_per_year, freqs_orig, with_replacement=True, seed=None):
    """Sample events uniformely from an array (indices_orig) without replacement
    (if sum(events_per_year) > n_input_events the input events are repeated
    (tot_n_events/n_input_events) times, by ensuring that the same events doens't
    occur more than once per sampled year).

    Parameters
    -----------
        events_per_year : np.ndarray
            Number of events per sampled year
        freqs_orig : np.ndarray
            Frequency of each input event
        with_replacement : bool, optional
            If True, impact events are sampled with replacement. If False, events are sampled
            without replacement. Sampling without replacement can yield distorted samples if
            frequencies of different events are unqual. Defaults to True.
        seed : Any, optional
            seed for the default bit generator.
            Default: None

    Returns
    -------
        sampling_vect : 2D array
            The sampling vector specifies how to sample the yimp, it consists of one
            sub-array per sampled_year, which contains the event_ids of the events used to
            calculate the annual impacts.
    """

    sampling_vect = []
    indices_orig = np.arange(len(freqs_orig))
    rng = default_rng(seed)

    # sample without replacement, works well if event frequencies are equal
    if with_replacement is False:
        # warn if frequencies of different events are not equal
        if np.unique(freqs_orig).size != 1:
            LOGGER.warning(
                "The frequencies of the different events are not equal. This can lead to "
                "distorted sampling if the frequencies vary significantly. To avoid this, "
                "please set with_replacement=True to sample with replacement instead."
            )

        indices = indices_orig
        freqs = freqs_orig

        # sample events for each sampled year
        for amount_events in events_per_year:
            # if there are not enough input events, choice with no replace will fail
            if amount_events > len(freqs_orig):
                raise ValueError(
                    f"cannot sample {amount_events} distinct events for a single year"
                    f" when there are only {len(freqs_orig)} input events. Set "
                    "with_replacement=True if you want to sample with replacmeent."
                )

            # if not enough events remaining, append original events
            if len(indices) < amount_events or len(indices) == 0:
                indices = np.append(indices, indices_orig)
                freqs = np.append(freqs, freqs_orig)

            # ensure that each event only occurs once per sampled year
            unique_events = np.unique(indices, return_index=True)[0]
            probab_dis = freqs[np.unique(indices, return_index=True)[1]] / (
                np.sum(freqs[np.unique(indices, return_index=True)[1]])
            )

            # sample events
            selected_events = rng.choice(
                unique_events, size=amount_events, replace=False, p=probab_dis
            ).astype("int")

            # determine used events to remove them from sampling pool
            idx_to_remove = [
                np.where(indices == event)[0][0] for event in selected_events
            ]
            indices = np.delete(indices, idx_to_remove)
            freqs = np.delete(freqs, idx_to_remove)

            # save sampled events in sampling vector
            sampling_vect.append(selected_events)

    else:
        # easier method if we allow for replacement sample with replacement
        probab_dis = freqs_orig / sum(freqs_orig)

        # sample events for each sampled year
        selected_events = rng.choice(
            indices_orig, size=sum(events_per_year), replace=True, p=probab_dis
        ).astype("int")

        # group to list of lists
        index = 0
        for amount_events in events_per_year:
            sampling_vect.append(selected_events[index : index + amount_events])
            index += amount_events

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
        imp_per_year: np.ndarray
            Sampled impact per year (length = sampled_years)
    """

    imp_per_year = [
        np.sum(imp.at_event[list(sampled_events)]) for sampled_events in sampling_vect
    ]

    return np.array(imp_per_year)


def calculate_correction_fac(imp_per_year, imp):
    """Calculate a correction factor that can be used to scale the yimp in such
    a way that the expected annual impact (eai) of the yimp amounts to the eai
    of the input imp

    Parameters
    -----------
        imp_per_year : np.ndarray
            sampled yimp
        imp : climada.engine.Impact()
            impact object containing impacts per event

    Returns
    -------
        correction_factor: int
            The correction factor is calculated as imp_eai/yimp_eai
    """

    yimp_eai = np.mean(imp_per_year)
    imp_eai = np.sum(imp.frequency * imp.at_event)
    correction_factor = imp_eai / yimp_eai

    # if correction_factor > 0.1:
    #     tex = raw_input("Do you want to exclude small events?")

    return correction_factor
