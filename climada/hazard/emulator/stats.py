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

Statistical tools for the hazard event emulator.
"""

import datetime
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

LOGGER = logging.getLogger(__name__)

def seasonal_average(data, season):
    """Compute seasonal average from monthly-time series.

    For seasons that are across newyear, the months after June are attributed to the following
    year's season. For example: The 6-month season from November 1980 till April 1981 is attributed
    to the year 1981.

    The two seasons that are truncated at the beginning/end of the dataset's time period are
    discarded. When the input data is 1980-2010, the output data will be 1981-2010, where 2010
    corresponds to the 2009/2010 season and 1981 corresponds to the 1980/1981 season.

    Parameters
    ----------
    data : DataFrame { year, month, ... }
        All further columns will be averaged over.
    season : pair of ints
        Start/end month of season.

    Returns
    -------
    averaged_data : DataFrame { year, ... }
        Same format as input, but with month column removed.
    """
    start, end = season

    if data['year'].unique().size == data.shape[0] and "month" in data.columns:
        data = data.drop(labels=['month'], axis=1)

    if "month" not in data.columns:
        return data.iloc[1:] if start > end else data

    if start > end:
        msk = (data['month'] >= start) | (data['month'] <= end)
    else:
        msk = (data['month'] >= start) & (data['month'] <= end)
    data = data[msk]
    if start > end:
        year_min, year_max = data['year'].min(), data['year'].max()
        data['year'][data['month'] > 6] += 1
        data = data[(data['year'] > year_min) & (data['year'] <= year_max)]
    data = data.reset_index(drop=True)
    data = data.groupby('year').mean().reset_index()
    return data.drop('month', 1)


def seasonal_statistics(events, season):
    """Compute seasonal statistics from given hazard event data

    Parameters
    ----------
    events : DataFrame { year, month, intensity, ... }
        Events outside of the given season are ignored.
    season : pair of ints
        Start/end month of season.

    Returns
    -------
    haz_stats : DataFrame { year, events, intensity_mean, intensity_std, intensity_max }
        For seasons that are across newyear, this might cover one year less than the input data
        since truncated seasons are discarded.
    """
    events = events.reindex(columns=['year', 'month', 'eventcount', 'intensity'])
    events['eventcount'] = 1
    year_min, year_max = events['year'].min(), events['year'].max()
    sea_start, sea_end = season
    if sea_start > sea_end:
        events['year'][events['month'] > 6] += 1
        msk = (events['month'] >= sea_start) | (events['month'] <= sea_end)
    else:
        msk = (events['month'] >= sea_start) & (events['month'] <= sea_end)
    events = events[msk].drop(labels=['month'], axis=1)

    def collapse(group):
        new_cols = ['eventcount', 'intensity_mean', 'intensity_std', 'intensity_max']
        new_vals = [group['eventcount'].sum(),
                    group['intensity'].mean(),
                    group['intensity'].std(ddof=0),
                    group['intensity'].max()]
        return pd.Series(new_vals, index=new_cols)
    haz_stats = events.groupby(['year']).apply(collapse).reset_index()

    if sea_end < sea_start:
        # drop first and last years as they are incomplete
        haz_stats = haz_stats[(haz_stats['year'] > year_min)
                              & (haz_stats['year'] <= year_max)].reset_index(drop=True)

    return haz_stats


def haz_max_events(hazard, min_thresh=0):
    """Table of max intensity events for given hazard

    Parameters
    ----------
    hazard : climada.hazard.Hazard object
    min_thresh : float
        Minimum intensity for event to be registered.

    Returns
    -------
    events : DataFrame { id, name, year, month, day, lat, lon, intensity }
        The integer value in column `id` refers to the internal order of events in the given
        `hazard` object. `lat`, `lon` and `intensity` specify location and intensity of the maximum
        intensity registered.
    """
    inten = hazard.intensity
    if min_thresh == 0:
        # this might require considerable amounts of memory
        exp_hazards = inten.todense() >= min_thresh
    else:
        exp_hazards = (inten >= min_thresh).todense()
    exp_hazards = np.where(np.any(exp_hazards, axis=1))[0]
    LOGGER.info("Condensing %d hazards to %d max events ...", inten.shape[0], exp_hazards.size)
    inten = inten[exp_hazards]
    inten_max_ids = np.asarray(inten.argmax(axis=1)).ravel()
    inten_max = inten[range(inten.shape[0]), inten_max_ids]
    dates = hazard.date[exp_hazards]
    dates = [datetime.date.fromordinal(d) for d in dates]
    return pd.DataFrame({
        'id': exp_hazards,
        'name': [hazard.event_name[s] for s in exp_hazards],
        'year': np.int64([d.year for d in dates]),
        'month': np.int64([d.month for d in dates]),
        'day': np.int64([d.day for d in dates]),
        'lat': hazard.centroids.lat[inten_max_ids],
        'lon': hazard.centroids.lon[inten_max_ids],
        'intensity': np.asarray(inten_max).ravel(),
    })


def normalize_seasonal_statistics(haz_stats, haz_stats_obs, freq_norm):
    """Bias-corrected annual hazard statistics

    Parameters
    ----------
    haz_stats : DataFrame { ... }
        Output of `seasonal_statistics`.
    haz_stats_obs : DataFrame { ... }
        Output of `seasonal_statistics`.
    freq_norm : DataFrame { year, freq }
        Information about the relative surplus of hazard events per year, i.e.,
        if `freq_norm` specifies the value 0.2 in some year, then it is
        assumed that the number of events given for that year is 5 times as
        large as it is predicted to be.

    Returns
    -------
    statistics : DataFrame { year, intensity_max, intensity_mean, eventcount,
                             intensity_max_obs, intensity_mean_obs, eventcount_obs }
        Normalized and observed hazard statistics.
    """
    norm_period = [haz_stats_obs['year'].min(), haz_stats_obs['year'].max()]

    # Merge observed into modelled statistics for comparison
    haz_stats = pd.merge(haz_stats, haz_stats_obs, suffixes=('', '_obs'),
                         on="year", how="left", sort=True)

    # Normalize `eventcount` according to simulated frequency.
    # In case of season across newyear, this normalizes by the year with most of
    # hazard season, ignoring the fractional contribution from the year before.
    haz_stats = pd.merge(haz_stats, freq_norm, on='year', how='left', sort=True)
    haz_stats['eventcount'] *= haz_stats['freq']
    haz_stats = haz_stats.drop(labels=['freq'], axis=1)

    # Bias-correct intensity and frequency to observations in norm period
    for col in ['eventcount', 'intensity_mean', 'intensity_std', 'intensity_max']:
        idx = haz_stats.index[(haz_stats['year'] >= norm_period[0]) \
                            & (haz_stats['year'] <= norm_period[1])]
        col_data = haz_stats.loc[idx, col]
        col_data_obs = haz_stats.loc[idx, f"{col}_obs"].dropna()
        if col == 'eventcount':
            fact = col_data_obs.sum() / col_data.sum()
        else:
            fact = col_data_obs.mean() / col_data.mean()
        haz_stats[col] *= fact
    return haz_stats


def fit_data(data, explained, explanatory, poisson=False):
    """Fit a response variable (e.g. intensity) to a list of explanatory variables

    The fitting is run twice, restricting to the significant explanatory
    variables in the second run.

    Parameters
    ----------
    data : DataFrame { year, `explained`, `explanatory`, ... }
        An intercept column is added automatically.
    explained : str
        Name of explained variable, e.g. 'intensity'.
    explanatory : list of str
        Names of explanatory variables, e.g. ['gmt','esoi'].
    poisson : boolean
        Optionally, use Poisson regression for fitting.
        If False (default), uses ordinary least squares (OLS) regression.

    Returns
    -------
    sm_results : pair of statsmodels Results object
        Results for first and second run.
    """
    d_explained = data[explained]
    d_explanatory = data[explanatory]

    # for the first run, assume that all variables are significant
    significant = explanatory
    sm_results = []
    for _ in range(2):
        # restrict to variables with significant relationship
        d_explanatory = d_explanatory[significant]

        # add column for intercept
        d_explanatory['const'] = 1.0

        if poisson:
            mod = smd.Poisson(d_explained, d_explanatory)
            res = mod.fit(maxiter=100, disp=0, cov_type='HC1')
        else:
            mod = sm.OLS(d_explained, d_explanatory)
            res = mod.fit(maxiter=100, disp=0, cov_type='HC1', use_t=True)
        significant = fit_significant(res)
        sm_results.append(res)

    return sm_results


def fit_significant(sm_results):
    """List significant variables in `sm_results`

    Note: The last variable (usually intercept) is omitted!
    """
    significant = []
    cols = sm_results.params.index.tolist()
    for i, pval in enumerate(sm_results.pvalues[:-1]):
        if pval <= 0.1:
            significant.append(cols[i])
    return significant


def fit_significance(sm_results):
    """Extract and visualize significance of model parameters"""
    significance = ['***' if el <= 0.01 else \
                    '**' if el <= 0.05 else \
                    '*' if el <= 0.1 else \
                    '-' for el in sm_results.pvalues[:-1]]
    significance = dict(zip(fit_significant(sm_results), significance))
    return significance
