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

Hazard event emulator.
"""

import logging
import sys

import numpy as np
import pandas as pd

import climada.hazard.emulator.const as const
import climada.hazard.emulator.stats as stats
import climada.hazard.emulator.random as random

LOGGER = logging.getLogger(__name__)

class HazardEmulator():
    """Draw samples for a time period driven by climate forcing

    Draw samples from the given pool of hazard events while making sure that the frequency and
    intensity are as predicted according to given climate indices.
    """
    explaineds = ['intensity_mean', 'eventcount']

    def __init__(self, haz_events, haz_events_obs, region, freq_norm, pool=None):
        """Initialize HazardEmulator

        Parameters
        ----------
        haz_events : DataFrame
            Output of `stats.haz_max_events`.
        haz_events_obs : DataFrame
            Observed events for normalization. Output of `stats.haz_max_events`.
        region : HazRegion object
            The geographical region for which to run emulations.
        freq_norm : DataFrame { year, freq }
            Information about the relative surplus of events in `tracks`, i.e., if `freq_norm`
            specifies the value 0.2 in some year, then it is assumed that the number of events
            given for that year is 5 times as large as it is predicted to be. Usually, the value
            will be smaller than 1 because the event set should be a good representation of TC
            distribution, but this is not necessary.
        pool : EventPool object, optional
            If omitted, draws are made from the events that are used to calibrate the emulator.
        """
        self.pool = EventPool(haz_events) if pool is None else pool
        self.region = region

        haz_stats = stats.seasonal_statistics(haz_events, region.season)
        haz_stats_obs = stats.seasonal_statistics(haz_events_obs, region.season)
        self.stats = stats.normalize_seasonal_statistics(haz_stats, haz_stats_obs, freq_norm)

        self.stats_pred = None
        self.fit_info = None
        self.ci_cols = []

        norm_period = [haz_events_obs['year'].min(), haz_events_obs['year'].max()]
        idx = self.stats.index[(self.stats['year'] >= norm_period[0]) \
                               & (self.stats['year'] <= norm_period[1])]
        norm_mean = self.stats.loc[idx, "intensity_mean_obs"].mean()
        self.pool.init_drop(norm_period, norm_mean)


    def calibrate_statistics(self, climate_indices):
        """Statistically fit hazard data to given climate indices

        The internal statistics are truncated to fit the temporal range of the climate indices.

        Parameters
        ----------
        climate_indices : list of DataFrames { year, month, ... }
            Yearly or monthly time series of GMT, ESOI etc.
        """
        if len(self.ci_cols) > 0:
            self.stats = self.stats.drop(labels=self.ci_cols, axis=1)

        self.ci_cols = []
        for cidx in climate_indices:
            ci_name = cidx.columns.values.tolist()
            ci_name.remove("year")
            ci_name.remove("month")
            self.ci_cols += ci_name
            avg_season = const.PDO_SEASON if "pdo" in ci_name else self.region.season
            avg = stats.seasonal_average(cidx, avg_season)
            self.stats = pd.merge(self.stats, avg, on="year", how="inner", sort=True)
        self.stats = self.stats.dropna(axis=0, how="any", subset=self.explaineds + self.ci_cols)

        self.fit_info = {}
        for explained in self.explaineds:
            self.fit_info[explained] = stats.fit_data(
                self.stats, explained, self.ci_cols, poisson=(explained == 'eventcount'))


    def predict_statistics(self, climate_indices=None):
        """Predict hypothetical hazard statistics according to climate indices

        The statistical fit from `calibrate_statistics` is used to predict the frequency and
        intensity of hazard events. The standard deviation of yearly residuals is used to define
        the yearly acceptable deviation of sample intensity.

        Without calibration, the prediction is done according to the (bias-corrected) within-year
        statistics of the event pool. In this case, the within-year standard deviation of intensity
        is taken as the acceptable deviation of samples for that year.

        Parameters
        ----------
        climate_indices : list of DataFrames { year, month, ... }
            Yearly or monthly time series of GMT, ESOI etc. including at least
            those passed to `calibrate_statistics`.
            If omitted, and if `calibrate_statistics` has been called before,
            the climate indices from calibration are reused for prediction.
            Otherwise, the internal (within-year) statistics of the data set
            are used to predict frequency and intensity.
        """
        reuse_indices = False
        if not climate_indices:
            reuse_indices = True
        elif len(climate_indices) > 0 and len(self.ci_cols) == 0:
            self.calibrate_statistics(climate_indices)
            reuse_indices = True

        if len(self.ci_cols) == 0:
            LOGGER.info("Predicting statistics without climate index predictor...")
            self.stats_pred = self.stats[['year', 'intensity_mean', 'eventcount']]
            self.stats_pred["intensity_mean_residuals"] = self.stats["intensity_std"]
            self.stats_pred["events_rediduals"] = 0
        elif reuse_indices:
            LOGGER.info("Predicting statistics with climate indices from calibration...")
            self.stats_pred = self.stats[['year'] + self.ci_cols]
            for explained in self.explaineds:
                sm_results = self.fit_info[explained][-1]
                self.stats_pred[explained] = sm_results.fittedvalues
                self.stats_pred[f"{explained}_residuals"] = sm_results.resid
        else:
            LOGGER.info("Predicting statistics with new climate index time series...")
            ci_avg = None
            for cidx in climate_indices:
                ci_name = cidx.columns.values.tolist()
                ci_name.remove("year")
                ci_name.remove("month")
                avg_season = const.PDO_SEASON if "pdo" in ci_name else self.region.season
                avg = stats.seasonal_average(cidx, avg_season)
                if ci_avg is None:
                    ci_avg = avg
                else:
                    ci_avg = pd.merge(ci_avg, avg, on="year", how="inner")
            self.stats_pred = ci_avg[["year"] + self.ci_cols]
            ci_data = self.stats_pred[self.ci_cols]
            ci_data['const'] = 1.0
            for explained in self.explaineds:
                sm_results = self.fit_info[explained][-1]
                explanatory = sm_results.params.index.tolist()
                haz_stats_pred = sm_results.predict(ci_data[explanatory])
                self.stats_pred[explained] = haz_stats_pred
                # use standard deviation of calibration residuals as "residuals":
                self.stats_pred[f"{explained}_residuals"] = float(sm_results.resid.std())


    def draw_realizations(self, nrealizations, period):
        """Draw samples for given time period according to calibration

        Draws for a specific year in the given period are not necessarily restricted to events in
        the pool that are explicitly assigned to that year because the pool might be too small to
        allow for draws of the expected sample size and mean intensity.

        Parameters
        ----------
        nrealizations : int
            Number of samples to draw.
        period : pair of ints [minyear, maxyear]
            Period for which to make draws.

        Returns
        -------
        draws : list of DataFrames, length `nrealizations`
            Each entry is a sample for the whole period, given as a DataFrame
            with columns as in `self.pool.events`. The `year` column is set to
            the respective year and columns for the driving climate indices are
            added for reference.
        """
        if self.stats_pred is None:
            raise Exception("Run `predict_statistics` before making draws!")

        LOGGER.info("Drawing %d realizations for period  (%d, %d)",
                    nrealizations, period[0], period[1])
        year_draws = []
        for year in range(period[0], period[1] + 1):
            sys.stdout.write(f"\r{period[0]} ... {year} ... {period[1]}")
            sys.stdout.flush()

            year_idx = self.stats_pred.index[self.stats_pred['year'] == year]
            freq_poisson = self.stats_pred.loc[year_idx, 'eventcount'].values[0]
            intensity_mean = self.stats_pred.loc[year_idx, 'intensity_mean'].values[0]
            intensity_std = self.stats_pred.loc[year_idx, 'intensity_mean_residuals'].values[0]
            intensity_std = np.clip(np.abs(intensity_std), 0.5, 10)
            draws = self.pool.draw_realizations(nrealizations, freq_poisson,
                                                intensity_mean, intensity_std)

            for real_id, draw in enumerate(draws):
                draw['year'] = year
                draw['real_id'] = real_id
                draws[real_id] = draw[['id', 'name', 'year', 'real_id']]
            year_draws += draws
        sys.stdout.write(f"\r{period[0]} ... {period[1]} ... {period[1]}\n")
        return pd.concat(year_draws, ignore_index=True)


class EventPool():
    """Make draws from a hazard event pool according to given statistics

    The event pool might cover an arbitrary number of years and an arbitrary geographical region
    since the time and geo information fields are ignored when making draws.

    No assumptions are made about where the statistics come from that are used in making the draw.

    Example
    -------
    Let `haz_events` be a given dataset of all TC events making landfall in Belize between 1980 and
    2050, together with their respective maximum wind speeds on land. Assume that we expect (from
    some other statistical model) 5 events of annual mean maximum wind speed 30 ± 10 m/s in the
    year 2025. Then, we can draw 100 realizations of hypothetical 2025 TC event sets hitting
    Belize with the following commands:

    >>> pool = EventPool(haz_events)
    >>> draws = pool.draw_realizations(100, 5, 30, 10)

    The realization `draw[i]` might contain events from any year between 1980 and 2050, but the
    size of the realization and the mean maximum wind speed will be according to the given
    statistics.
    """

    def __init__(self, haz_events):
        """Initialize instance of EventPool

        Parameters
        ----------
        haz_events : DataFrame
            Output of `stats.haz_max_events`.
        """
        self.events = haz_events
        self.drop = None


    def init_drop(self, norm_period, norm_mean):
        """Use a drop rule when making draws

        With the drop rule, a random choice of entries is dropped from events before the actual
        drawing is done in order to speed up the process in case of data sets where the acceptable
        mean is far from the input data mean.

        Parameters
        ----------
        norm_period : pair of ints [minyear, maxyear]
            Normalization period for which a specific mean intensity is expected.
        norm_mean : float
            Desired mean intensity of events in the given time period.
        """
        self.drop = random.estimate_drop(self.events, 'year', 'intensity',
                                         norm_period, norm_mean=norm_mean)


    def draw_realizations(self, nrealizations, freq_poisson, intensity_mean, intensity_std):
        """Draw samples from the event pool according to given statistics

        If `EventPool.init_drop` has been called before, the drop rule is applied.

        Parameters
        ----------
        nrealizations : int
            Number of samples to draw
        freq_poisson : float
            Expected sample size ("frequency", Poisson distributed).
        intensity_mean : float
            Expected sample mean intensity.
        intensity_std : float
            Acceptable deviation from `intensity_mean`.

        Returns
        -------
        draws : list of DataFrames, length `nrealizations`
            Each entry is a sample, given as a DataFrame with columns as in
            `self.events`.
        """
        draws = []
        while len(draws) < nrealizations:
            intensity_accept = [intensity_mean - intensity_std, intensity_mean + intensity_std]
            drawn = None
            while drawn is None:
                drawn = random.draw_poisson_events(
                    freq_poisson, self.events, 'intensity', intensity_accept, drop=self.drop)
                intensity_accept[0] -= 1
                intensity_accept[1] += 1
            draws.append(self.events.loc[drawn])
        return draws
