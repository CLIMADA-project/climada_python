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

Randomized sampling tools for the hazard event emulator.
"""

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)

random_state = np.random.RandomState(123456789)
# random_state = np.random.default_rng(123456789)

def estimate_drop(events, time_col, val_col, norm_period, norm_fact=None, norm_mean=None):
    """Determine fraction of outlying events to be dropped

    If the mean intensity of events in the given time period `norm_period`
    is far from the desired mean `norm_mean`, sampling from `events` will
    usually yield draws whose mean is far from the desired mean, so that many
    resamplings will be necessary in order to get an acceptable draw.

    Dropping events off the desired mean before sampling can reduce the
    necessary number of samplings.

    This function estimates which portion of the events should be dropped.

    Parameters
    ----------
    events : DataFrame
        Each row describes one event.
        The dataset should contain at least the columns `time_col` and `val_col`.
    time_col : str
        Name of time column in `events`.
    val_col : str
        Name of value column in `events`.
    norm_period : pair of timestamps (e.g. floats or ints)
        Normalization period for which a specific mean intensity is expected.
    norm_mean : float
        Desired mean intensity of events in the given time period.
    norm_fact : float
        Instead of `norm_mean`, the ratio between desired and observed
        intensity in the given time period can be given.

    Returns
    -------
    drop : pair [expr, frac]
        Only events satisfying the pandas query expression `expr` should be
        eligible for dropping. `frac` specifies the fraction of these events
        that are to be dropped.
    """
    all_idx = events.index[(events[time_col] >= norm_period[0])
                           & (events[time_col] <= norm_period[1])]
    all_mean = events.loc[all_idx, val_col].mean()
    all_std = events.loc[all_idx, val_col].std()

    if norm_mean is None:
        assert norm_fact is not None
        norm_mean = all_mean * norm_fact
    else:
        norm_fact = norm_mean / all_mean

    drop_expr = f"{val_col} {'<' if norm_fact > 1 else '>'} {all_mean}"
    drop_frac = 0.0
    if 0.98 < norm_fact < 1.02:
        return drop_expr, drop_frac

    step_size = 0.5 * np.abs(norm_mean - all_mean) / all_std
    drop_frac += step_size
    diff = 0.1
    sub_mean = 0
    while drop_frac < 1.0 and np.abs(diff) > 0.025 and diff > 0:
        drop_idx = events.query(drop_expr) \
                         .sample(frac=drop_frac, random_state=random_state) \
                         .index
        events_sub = events.drop(drop_idx).reset_index(drop=True)
        sub_idx = events_sub.index[(events_sub[time_col] >= norm_period[0])
                                   & (events_sub[time_col] <= norm_period[1])]
        sub_mean = events_sub.loc[sub_idx, val_col].mean()

        diff = (norm_mean - sub_mean) / np.abs(norm_mean)
        diff = diff if norm_fact > 1.0 else -diff
        if np.abs(diff) > 0.025:
            drop_frac += step_size
    drop_frac = min(1.0, drop_frac)
    LOGGER.info("Results of intensity normalization by subsampling:")
    LOGGER.info("- drop %d%% of entries satisfying '%s'", int(100 * drop_frac), drop_expr)
    LOGGER.info("- mean intensity of simulated events before dropping is %.4f", all_mean)
    LOGGER.info("- mean intensity of simulated events after dropping is %.4f", sub_mean)
    LOGGER.info("- mean intensity of observed events is %.4f", norm_mean)

    return drop_expr, drop_frac


def draw_poisson_events(poisson, events, val_col, val_accept, drop=None):
    """Draw poisson distributed events with acceptable value statistics

    The size of the draw is poisson distributed. Redraws are made until the
    draw mean is within the range specified by `val_accept`.

    If `drop` is specified, a random choice of entries is dropped from `events`
    before the actual drawing is done in order to speed up the process in case
    of data sets where the acceptable mean is far from the input data mean.

    Parameters
    ----------
    poisson : float
        Poisson parameter.
    events : DataFrame
        Each row describes one event.
        The dataset should contain at least the column `val_col`.
    val_col : str
        Name of value column in `events`.
    val_accept : pair of floats
        Acceptable range of draw means.
    drop : pair [expr, frac] or None
        If given, only events satisfying the pandas query expression `expr` are dropped.
        `frac` specifies the fraction of these events that is dropped.

    Returns
    -------
    draw_idx : Series or None
        Indices into `events`.
        If no acceptable draw was among the first 10,000 attempts, the return value is None.
    """
    fail_counts = 0
    if drop is not None:
        drop_query, drop_frac = drop
    while True:
        events_sub = events
        if drop is not None:
            drop_idx = events.query(drop_query) \
                             .sample(frac=drop_frac, random_state=random_state) \
                             .index
            events_sub = events.drop(drop_idx)

        draw_size = max(1, random_state.poisson(poisson, 1)[0])
        draw_inds = random_state.choice(events_sub.shape[0], draw_size,
                                        replace=(events_sub.shape[0] < 1.2 * draw_size))
        draw_mean = events_sub[val_col].iloc[draw_inds].mean()


        if val_accept[0] <= draw_mean <= val_accept[1]:
            return events_sub.index[draw_inds]

        fail_counts += 1
        if fail_counts >= 10000:
            return None
