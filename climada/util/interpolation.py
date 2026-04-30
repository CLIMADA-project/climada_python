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

Define interpolation and extrapolation functions for calculating
(local) exceedance frequencies and return periods
"""

import logging

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from scipy.stats import genextreme

LOGGER = logging.getLogger(__name__)


def preprocess_and_interpolate_ev(
    exceedance_frequency,
    test_values,
    frequency,
    values,
    log_frequency=False,
    log_values=False,
    value_threshold=None,
    method="interpolate",
    y_asymptotic=np.nan,
    bin_decimals=None,
):
    """Function to first preprocess (frequency, values) data (if extrapolating, one can bin the
    data according to their value, see Notes), compute the cumulative frequencies, and then
    inter- and extrapolate either to test frequencies or to test values.

    Parameters
    ----------
    exceedance_frequency : array_like
        1-D array of test exceedance frequencies for which values (e.g., intensities or impacts) should be
        assigned. If given, test_values must be None.
    test_values : array_like
        1-D array of test values (e.g., intensities or impacts) for which exceedance frequencies should be
        assigned. If given, exceedance_frequency must be None.
    frequency : array_like
        1-D array of frequencies to be interpolated.
    values : array_like
        1-D array of values (e.g., intensities or impacts) to be interpolated.
    log_frequency : bool, optional
        If set to True, frequencies are interpolated on log scale. Defaults to False.
    log_values : bool, optional
        If set to True, values (e.g., intensities) are interpolated on log scale.
        Defaults to False.
    value_threshold : float, optional
        Lower threshold to filter values (e.g., intensities or impacts). Defaults to None.
    method : str, optional
        Method to interpolate to test x values. Currently available are
        "interpolate", "extrapolate", "extrapolate_constant" and "stepfunction". If set to
        "interpolate", test x values outside the range of the given x values will be assigned NaN.
        If set to "extrapolate_constant" or "stepfunction", test x values larger than given
        x values will be assigned largest given y value, and test x values smaller than the given
        x values will be assigned y_asymtotic. If set to "extrapolate", values will be extrapolated
        (and interpolated). The extrapolation to test frequencies or test values outside of the
        data range extends the two interpolations at the edges of the data to outside of
        the data. Defaults to "interpolate".
    y_asymptotic : float, optional
        Has no effect if method is "interpolate". Else, provides return value and if
        for test x values larger than given x values, if size < 2 or if method is set
        to "extrapolate_constant" or "stepfunction". Defaults to np.nan.
    bin_decimals : int, optional
        Number of decimals to group and bin the values. Binning results in smoother (and coarser)
        interpolation and more stable extrapolation. For more details and sensible values for
        bin_decimals, see Notes. If None, values are not binned. Defaults to None.

    Returns
    -------
    np.array
        interpolated (and extrapolated) values or frequencies for given test frequencies or test
        values, respectively.

    Raises
    ------
    ValueError
        If both test frequencies and test values are given or none of them.

    Notes
    -----
    If an integer bin_decimals is given, the values are binned according to their
    bin_decimals decimals, and their corresponding frequencies are summed. This binning leads to
    a smoother (and coarser) interpolation, and a more stable extrapolation. For instance, if
    bin_decimals=1, the two values 12.01 and 11.97 with corresponding frequencies 0.1 and 0.2 are
    combined to a value 12.0 with frequency 0.3. The default bin_decimals=None results in not
    binning the values.
    E.g., if your values range from 1 to 100, you could use bin_decimals=1, if your values range
    from 1e6 to 1e9, you could use bin_decimals=-5, if your values range from 0.0001 to .01, you
    could use bin_decimals=5.
    """

    # check method
    if method not in [
        "interpolate",
        "extrapolate",
        "extrapolate_constant",
        "stepfunction",
    ]:
        raise ValueError(f"Unknown method: {method}")

    # check that only test frequencies or only test values are given
    if exceedance_frequency is not None and test_values is not None:
        raise ValueError(
            "Both test frequencies and test values are given. This method only handles one of "
            "the two. To use this method, please only use one of them."
        )
    if exceedance_frequency is None and test_values is None:
        raise ValueError("No test values or test frequencies are given.")

    # sort values and frequencies
    sorted_idxs = np.argsort(values)
    values = np.squeeze(values[sorted_idxs])
    frequency = frequency[sorted_idxs]

    # group similar values together
    if isinstance(bin_decimals, int):
        frequency, values = _group_frequency(frequency, values, bin_decimals)

    # transform frequencies to cummulative frequencies
    frequency = np.cumsum(frequency[::-1])[::-1]

    # if test frequencies are provided
    if exceedance_frequency is not None:
        if method == "stepfunction":
            return _stepfunction_ev(
                exceedance_frequency,
                frequency[::-1],
                values[::-1],
                y_threshold=value_threshold,
                y_asymptotic=y_asymptotic,
            )
        extrapolation = None if method == "interpolate" else method
        return _interpolate_ev(
            exceedance_frequency,
            frequency[::-1],
            values[::-1],
            logx=log_frequency,
            logy=log_values,
            y_threshold=value_threshold,
            extrapolation=extrapolation,
            y_asymptotic=y_asymptotic,
        )

    # if test values are provided
    if method == "stepfunction":
        return _stepfunction_ev(
            test_values,
            values,
            frequency,
            x_threshold=value_threshold,
            y_asymptotic=y_asymptotic,
        )
    extrapolation = None if method == "interpolate" else method
    return _interpolate_ev(
        test_values,
        values,
        frequency,
        logx=log_values,
        logy=log_frequency,
        x_threshold=value_threshold,
        extrapolation=extrapolation,
    )


def _interpolate_ev(
    x_test,
    x_train,
    y_train,
    logx=False,
    logy=False,
    x_threshold=None,
    y_threshold=None,
    extrapolation=None,
    y_asymptotic=np.nan,
):
    """
    Util function to interpolate (and extrapolate) training data (x_train, y_train)
    to new points x_test with several options (log scale, thresholds)

    Parameters
    ----------
        x_test : array_like
            1-D array of x-values for which training data should be interpolated
        x_train : array_like
            1-D array of x-values of training data sorted in ascending order
        y_train : array_like
            1-D array of y-values of training data
        logx : bool, optional
            If set to True, x_values are converted to log scale. Defaults to False.
        logy : bool, optional
            If set to True, y_values are converted to log scale. Defaults to False.
        x_threshold : float, optional
            Lower threshold to filter x_train. Defaults to None.
        y_threshold : float, optional
            Lower threshold to filter y_train. Defaults to None.
        extrapolation : str, optional
            If set to 'extrapolate', values will be extrapolated. If set to 'extrapolate_constant',
            x_test values smaller than x_train will be assigned y_train[0] (x_train must be sorted
            in ascending order), and x_test values larger than x_train will be assigned
            y_asymptotic. If set to None, x_test values outside of the range of x_train will be
            assigned np.nan. Defaults to None.
        y_asymptotic : float, optional
            Has no effect if extrapolation is None. Else, provides return value and if
            for x_test values larger than x_train, for x_train.size < 2 or if extrapolation is set
            to 'extrapolate_constant'. Defaults to np.nan.

    Returns
    -------
    np.array
        interpolated values y_test for the test points x_test
    """

    # preprocess interpolation data
    x_test, x_train, y_train = _preprocess_interpolation_data(
        x_test, x_train, y_train, logx, logy, x_threshold, y_threshold
    )

    # handle case of small training data sizes
    if x_train.size < 2:
        if not extrapolation:
            return np.full_like(x_test, np.nan)
        return _interpolate_small_input(x_test, x_train, y_train, logy, y_asymptotic)

    # calculate fill values
    if extrapolation == "extrapolate":
        fill_value = "extrapolate"
    elif extrapolation == "extrapolate_constant":
        fill_value = (y_train[0], np.log10(y_asymptotic) if logy else y_asymptotic)
    else:
        fill_value = np.nan

    interpolation = interpolate.interp1d(
        x_train, y_train, fill_value=fill_value, bounds_error=False
    )
    y_test = interpolation(x_test)

    # adapt output scale
    if logy:
        y_test = np.power(10.0, y_test)
    return y_test


def _stepfunction_ev(
    x_test, x_train, y_train, x_threshold=None, y_threshold=None, y_asymptotic=np.nan
):
    """
    Util function to interpolate and extrapolate training data (x_train, y_train)
    to new points x_test using a step function

    Parameters
    ----------
        x_test : array_like
            1-D array of x-values for which training data should be interpolated
        x_train : array_like
            1-D array of x-values of training data sorted in ascending order
        y_train : array_like
            1-D array of y-values of training data
        x_threshold : float, optional
            Lower threshold to filter x_train. Defaults to None.
        y_threshold : float, optional
            Lower threshold to filter y_train. Defaults to None.
        y_asymptotic : float, optional
            Return value if x_test > x_train. Defaults to np.nan.

    Returns
    -------
    np.array
        interpolated values y_test for the test points x_test
    """

    # preprocess interpolation data
    x_test, x_train, y_train = _preprocess_interpolation_data(
        x_test, x_train, y_train, None, None, x_threshold, y_threshold
    )

    # handle case of small training data sizes
    if x_train.size < 2:
        return _interpolate_small_input(x_test, x_train, y_train, None, y_asymptotic)

    # find indices of x_test if sorted into x_train
    indx = np.searchsorted(x_train, x_test)
    y_test = y_train[indx.clip(max=len(x_train) - 1)]
    y_test[indx == len(x_train)] = y_asymptotic

    return y_test


def _preprocess_interpolation_data(
    x_test, x_train, y_train, logx, logy, x_threshold, y_threshold
):
    """
    helper function to preprocess interpolation training and test data by filtering data below
    thresholds and converting to log scale if required
    """

    if x_train.shape != y_train.shape:
        raise ValueError(
            f"Incompatible shapes of input data, x_train {x_train.shape} "
            f"and y_train {y_train.shape}. Should be the same"
        )

    # transform input to float arrays
    x_test, x_train, y_train = (
        np.array(x_test).astype(float),
        np.array(x_train).astype(float),
        np.array(y_train).astype(float),
    )

    # cut x and y above threshold
    if x_threshold or x_threshold == 0:
        x_th = np.asarray(x_train > x_threshold).squeeze()
        x_train = x_train[x_th]
        y_train = y_train[x_th]

    if y_threshold or y_threshold == 0:
        y_th = np.asarray(y_train > y_threshold).squeeze()
        x_train = x_train[y_th]
        y_train = y_train[y_th]

    # convert to log scale
    if logx:
        x_train, x_test = np.log10(x_train), np.log10(x_test)
    if logy:
        y_train = np.log10(y_train)

    return (x_test, x_train, y_train)


def _interpolate_small_input(x_test, x_train, y_train, logy, y_asymptotic):
    """
    helper function to handle if interpolation data is small (empty or one point)
    """
    # return y_asymptotic if x_train and y_train empty
    if x_train.size == 0:
        return np.full_like(x_test, y_asymptotic)

    # reconvert logarithmic y_train to original y_train
    if logy:
        y_train = np.power(10.0, y_train)

    # if only one (x_train, y_train), return stepfunction with
    # y_train if x_test < x_train and y_asymtotic if x_test > x_train
    y_test = np.full_like(x_test, y_train[0])
    y_test[np.squeeze(x_test) > np.squeeze(x_train)] = y_asymptotic
    return y_test


def _group_frequency(frequency, value, bin_decimals):
    """
    Util function to aggregate (add) frequencies for equal values

    Parameters
    ----------
        frequency : array_like
            Frequency array
        value : array_like
            Value array in ascending order
        bin_decimals : int
            decimals according to which values are binned and their corresponding frequency are
            grouped.

    Returns
    -------
        tuple
            (frequency array after aggregation,
            unique value array in ascending order)
    """
    frequency, value = np.array(frequency), np.array(value)
    if frequency.size == 0 and value.size == 0:
        return ([], [])

    # round values and group them
    value = np.around(value, decimals=bin_decimals)
    value_unique, start_indices = np.unique(value, return_index=True)
    if value_unique.size != frequency.size:
        if not all(sorted(start_indices) == start_indices):
            LOGGER.warning(
                "After grouping values using to their decimals, the value array is not sorted."
                "The values are not binned. This might be due to floating point error while "
                "binning. Please choose a larger value of bin_decimals=%s.",
                bin_decimals,
            )
            return frequency, value

        # add frequency for equal value
        start_indices = np.insert(start_indices, value_unique.size, frequency.size)
        frequency = np.add.reduceat(frequency, start_indices[:-1])
        return frequency, value_unique

    return frequency, value


def _gpd_distribution(values, xi, beta, lambda_u, threshold):
    """
    Survival function (1-CDF) of generalized Pareto distribution including probability
    of threhsold exceedance (lambda_u).
    See https://en.wikipedia.org/wiki/Generalized_Pareto_distribution
    """
    values = np.asarray(values)
    return lambda_u * (1 + xi * (values - threshold) / beta) ** (-1 / xi)


def _gpd_inverse_distribution(lambdas, xi, beta, lambda_u, threshold):
    """
    Inverse survival function of generalized Pareto distribution including probability
    of threhsold exceedance (lambda_u).
    See https://en.wikipedia.org/wiki/Generalized_Pareto_distribution
    """
    lambdas = np.asarray(lambdas)
    return threshold + (beta / xi) * ((lambdas / lambda_u) ** (-xi) - 1)


def _gev_distribution(values, xi, mu, sigma, lambda_u, threshold):
    """
    Survival function (1-CDF) of generalized extreme value distribution including probability
    of threhsold exceedance (lambda_u).
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
    """
    values = np.asarray(values)
    cdf_vals = genextreme.cdf(values, c=-xi, loc=mu, scale=sigma)
    return lambda_u * (1 - cdf_vals)


def _gev_inverse_distribution(lambdas, xi, mu, sigma, lambda_u, threshold):
    """
    Inverse survival function of generalized extreme value distribution including probability
    of threhsold exceedance (lambda_u).
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.genextreme.html
    """
    lambdas = np.asarray(lambdas)
    ppf_vals = genextreme.ppf(1 - lambdas / lambda_u, c=-xi, loc=mu, scale=sigma)
    return ppf_vals


TAIL_MODELS = {
    "GPD": {
        "init": lambda x_tail, threshold: [0.1, np.std(x_tail - threshold)],
        "bounds": [(-1, None), (1e-6, None)],
        "distribution": _gpd_distribution,
        "inverse": _gpd_inverse_distribution,
    },
    "GEV": {
        "init": lambda x_tail, threshold: [0.1, threshold, np.std(x_tail - threshold)],
        # "init": lambda x_tail, threshold: [0.1, np.mean(x_tail), np.std(x_tail)],
        "bounds": [(-1, None), (None, None), (1e-6, None)],
        "distribution": _gev_distribution,
        "inverse": _gev_inverse_distribution,
    },
}


def fit_tail_distribution(
    test_frequency=None,
    test_values=None,
    frequency=None,
    values=None,
    dist="GPD",
    value_threshold=0,
    threshold_percentile=90,
    min_sample_size=30,
):
    """
    Fit a tail distribution (GPD or GEV) to exceedance data and extrapolate to test points.

    Extreme value theory provides two principal approaches for modeling extremes: the block
    maxima method leading to the GEV distribution and the peaks-over-threshold method
    leading to the GPD. These approaches are complementary and involve different practical
    trade-offs, such as efficiency of data usage and the need for threshold selection,
    see e.g. (Coles, 2001, Chapters 4–5, https://doi.org/10.1007/978-1-4471-3675-0).

    Parameters
    ----------
    test_frequency : array_like, optional
        Exceedance frequencies for which to compute values. If given, test_values must be None.
    test_values : array_like, optional
        Values for which to compute exceedance frequencies. If given, test_frequency must be None.
    frequency : array_like
        Frequencies of the observed values.
    values : array_like
        Observed values.
    dist : str, optional
        Distribution to fit: "GPD" or "GEV". Defaults to "GPD".
    value_threshold : float, optional
        Lower threshold to filter values. Defaults to None (no filtering).
    threshold_percentile : float, optional
        Percentile for the tail threshold. Defaults to 90.
    min_sample_size : int, optional
        Minimum number of points above the threshold. Defaults to 30.

    Returns
    -------
    tuple
        (test_frequency, values) if test_frequency is given, else (exceedance_frequencies, test_values)
    """

    # Validate inputs
    if (test_frequency is not None and test_values is not None) or (
        test_frequency is None and test_values is None
    ):
        raise ValueError("Provide exactly one of test_frequency or test_values")

    if dist not in TAIL_MODELS:
        raise ValueError(
            f"Unknown distribution: {dist}. Implemented distributions are {TAIL_MODELS.keys()}"
        )

    # Filter by value_threshold before computing threshold with percentile
    mask = values > value_threshold
    frequency = frequency[mask]
    values = values[mask]

    # Sort values and frequencies
    sorted_idxs = np.argsort(values)
    values = np.squeeze(values[sorted_idxs])
    frequency = frequency[sorted_idxs]
    ex_freq = np.cumsum(frequency[::-1])[::-1]

    threshold = np.percentile(values, threshold_percentile)
    mask = values > threshold
    if sum(mask) < min_sample_size:
        raise ValueError(
            f"Not enough data points above the threshold for fitting the {dist}. You can try to "
            f"choose a smaller threshold_percentile={threshold_percentile} or a smaller "
            f"min_sample_size={min_sample_size}."
        )
    x_tail = values[mask]
    lambda_tail = ex_freq[mask]
    lambda_u = lambda_tail[0]  # estimated frequency for exceeding the threshold

    model_config = TAIL_MODELS[dist]
    init = model_config["init"](x_tail, threshold)

    def exceedance_negerror(params):
        if dist == "GPD":
            xi, beta = params
            if beta <= 0:
                return np.inf
        elif dist == "GEV":
            xi, mu, sigma = params
            if sigma <= 0:
                return np.inf
        model = model_config["distribution"](x_tail, *params, lambda_u, threshold)
        return np.sum((np.log(lambda_tail) - np.log(model)) ** 2)

    res = minimize(exceedance_negerror, init, bounds=model_config["bounds"])
    params_hat = res.x

    # Compute some goodness-of-fit metrics
    # Recompute model on tail with fitted params
    model_tail = model_config["distribution"](x_tail, *params_hat, lambda_u, threshold)
    # Avoid log(0) issues
    eps = 1e-12
    lambda_tail_safe = np.maximum(lambda_tail, eps)
    model_tail_safe = np.maximum(model_tail, eps)

    # Residuals in log-survival space
    residuals = np.log(lambda_tail_safe) - np.log(model_tail_safe)
    # Root mean-squared error (log-survival space)
    rmse_log = np.sqrt(np.mean(residuals**2))

    # Kolmogorov–Smirnov distance on normalized survival function
    empirical_prob = lambda_tail_safe / lambda_u
    model_prob = model_tail_safe / lambda_u
    ks_distance = np.max(np.abs(empirical_prob - model_prob))

    if dist == "GPD":
        xi_hat, beta_hat = params_hat
        fit_result = {"xi": xi_hat, "beta": beta_hat}
        LOGGER.info(
            "Fitted GPD parameters using %.3g points: xi=%.3g, beta=%.3g.",
            sum(mask),
            xi_hat,
            beta_hat,
        )
    elif dist == "GEV":
        xi_hat, mu_hat, sigma_hat = params_hat
        fit_result = {"xi": xi_hat, "mu": mu_hat, "sigma": sigma_hat}
        LOGGER.info(
            "Fitted GEV parameters using %.3g points:: xi=%.3g, mu=%.3g, sigma=%.3g.",
            sum(mask),
            xi_hat,
            mu_hat,
            sigma_hat,
        )
    fit_result.update({"rmse_log": rmse_log, "ks_distance": ks_distance})
    LOGGER.info(
        "GOF metrics: RMSE_log=%.3g, KS=%.3g",
        rmse_log,
        ks_distance,
    )

    if test_values is not None:
        mask_tail = test_values > threshold
        lambda_dist = model_config["distribution"](
            test_values[mask_tail], *params_hat, lambda_u, threshold
        )
        return (
            (
                np.where(mask_tail, lambda_dist, np.nan)
                if lambda_dist.size > 0
                else np.full(mask_tail.shape, np.nan)
            ),
            test_values,
            fit_result,
        )
    else:
        vals = model_config["inverse"](test_frequency, *params_hat, lambda_u, threshold)
        mask_tail = vals > threshold
        return (
            test_frequency,
            (
                np.where(mask_tail, vals, np.nan)
                if vals.size > 0
                else np.full(mask_tail.shape, np.nan)
            ),
            fit_result,
        )
