"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

Most of this module are modifications of work originally published under the following license:

MIT License

Copyright (c) 2021 Lynne Jewson, Stephen Jewson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

Define scaling factors to model the impact of climate change on tropical cyclones.
"""

import logging
from math import log

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

MAP_BASINS_NAMES = {"NA": 0, "WP": 1, "EP": 2, "NI": 3, "SI": 4, "SP": 5}

MAP_VARS_NAMES = {"cat05": 0, "cat45": 1, "intensity": 2}

MAP_PERC_NAMES = {"5/10": 0, "25": 1, "50": 2, "75": 3, "90/95": 4}

# it defines the first and last projection years as well as the largest smoothing window
YEAR_WINDOWS_PROPS = {"start": 2000, "end": 2100, "smoothing": 5}


def get_knutson_scaling_factor(
    variable: str = "cat05",
    percentile: str = "50",
    basin: str = "NA",
    baseline: tuple = (1982, 2022),
    yearly_steps: int = 5,
):
    """
    This code combines data in Knutson et al. (2020) and global mean surface
    temperature (GMST) data (historical and CMIP5 simulated) to produce TC
    projections for 4 RCPs and any historical baseline. The code uses GMST data
    implements to interpolate and extrapolate the Knutson data
    relative to the specified baseline time period for various RCPs
    with a log-linear model. The methodology was developed and explained
    in Jewson et al., (2021).

    Related publications:

    - Knutson et al., (2020): Tropical cyclones and climate
      change assessment. Part II: Projected response to anthropogenic warming.
      Bull. Amer. Meteor. Soc., 101 (3), E303–E322,
      https://doi.org/10.1175/BAMS-D-18-0194.1.

    - Jewson (2021): Conversion of the Knutson et al. (2020) Tropical Cyclone
      Climate Change Projections to Risk Model Baselines,
      https://doi.org/10.1175/JAMC-D-21-0102.1

    Parameters
    ----------
    variable: int
        variable of interest, possible choices are 'cat05' (frequencies of all
        tropical cyclones), 'cat45' (frequencies of category 4 and 5 tropical cyclones)
        and 'intensity' (mean intensity of all tropical cyclones)
    percentile: str
        percentiles of Knutson et al. 2020 estimates, representing the model uncertainty
        in future changes in TC activity. These estimates come from a review of state-of-the-art
        literature and models. For the 'cat05' variable (i.e. frequency of all tropical cyclones)
        the 5th, 25th, 50th, 75th and 95th percentiles are provided. For 'cat45' and 'intensity',
        the provided percentiles are the 10th, 25th, 50th, 75th and 90th. Please refer to the
        mentioned publications for more details.
        possible percentiles:

        * '5/10' either the 5th or 10th percentile depending on variable (see text above)
        * '25' for the 25th percentile
        * '50' for the 50th percentile
        * '75' for the 75th percentile
        * '90/95' either the 90th or 95th percentile depending on variable  (see text above)

        Default: '50'
    basin : str
        region of interest, possible choices are:
        'NA', 'WP', 'EP', 'NI', 'SI', 'SP'
    baseline : tuple of int
        the starting and ending years that define the historical
        baseline. The historical baseline period must fall within
        the GSMT data period, i.e., 1880-2100. Default is 1982-2022.
    yearly_steps : int
        yearly resolution at which projections are provided. Default is 5 years.

    Returns
    -------
    future_change_variable : pd.DataFrame
        data frame with future projections of the selected variables at different
        times (indexes) and for RCPs 2.6, 4.5, 6.0 and 8.5 (columns).
    """

    base_start_year, base_end_year = baseline
    gmst_info = get_gmst_info()

    knutson_data = get_knutson_data()

    num_of_rcps, gmst_years = gmst_info["gmst_data"].shape

    if (
        (base_start_year <= gmst_info["gmst_start_year"])
        or (base_start_year >= gmst_info["gmst_end_year"])
        or (base_end_year <= gmst_info["gmst_start_year"])
        or (base_end_year >= gmst_info["gmst_end_year"])
    ):

        raise ValueError(
            "The selected historical baseline falls outside"
            f"the GMST data period {gmst_info['gmst_start_year']}"
            f"-{gmst_info['gmst_end_year']}"
        )

    var_id = MAP_VARS_NAMES[variable]
    perc_id = MAP_PERC_NAMES[percentile]

    # Steps:
    # 1. transform annual GMST values using e^βT
    # 2. calculate the average of these transformed values over the two time periods
    # 3. calculate the fractional change in the averages
    # please refer to section 4. Methods of Jewson (2021) for more details.

    mid_years = np.arange(
        YEAR_WINDOWS_PROPS["start"], YEAR_WINDOWS_PROPS["end"] + 1, yearly_steps
    )
    predicted_change = np.ones((mid_years.shape[0], num_of_rcps))

    try:
        basin_id = MAP_BASINS_NAMES[basin]
        knutson_value = knutson_data[var_id, basin_id, perc_id]

    except KeyError:
        LOGGER.warning(
            f"No scaling factors are defined for basin {basin} therefore"
            "no change will be projected for tracks in this basin"
        )
        return pd.DataFrame(
            predicted_change, index=mid_years, columns=gmst_info["rcps"]
        )

    base_start_pos = base_start_year - gmst_info["gmst_start_year"]
    base_end_pos = base_end_year - gmst_info["gmst_start_year"]

    # Step 1.
    beta = 0.5 * log(0.01 * knutson_value + 1)  # equation 6 in Jewson (2021)
    tc_properties = np.exp(beta * gmst_info["gmst_data"])  # equation 3 in Jewson (2021)

    # Step 2.
    baseline = np.mean(tc_properties[:, base_start_pos : base_end_pos + 1], 1)

    # Step 3.
    for i, mid_year in enumerate(mid_years):
        mid_year_in_gmst_ind = mid_year - gmst_info["gmst_start_year"]
        actual_smoothing = min(
            YEAR_WINDOWS_PROPS["smoothing"],
            gmst_years - mid_year_in_gmst_ind - 1,
            mid_year_in_gmst_ind,
        )
        fut_start_pos = mid_year_in_gmst_ind - actual_smoothing
        fut_end_pos = mid_year_in_gmst_ind + actual_smoothing + 1

        prediction = np.mean(tc_properties[:, fut_start_pos:fut_end_pos], 1)

        # assess fractional changes
        predicted_change[i] = ((prediction - baseline) / baseline) * 100

    return pd.DataFrame(predicted_change, index=mid_years, columns=gmst_info["rcps"])


def get_gmst_info():
    """
    Get Global Mean Surface Temperature (GMST) data from 1880 to 2100 for
    RCPs 2.6, 4.5, 6.0 and 8.5. Data are provided in:

    Jewson (2021): Conversion of the Knutson et al. (2020) Tropical Cyclone
    Climate Change Projections to Risk Model Baselines,
    https://doi.org/10.1175/JAMC-D-21-0102.1

    and in supporting documentation and code.

    Returns
    -------
    gmst_info : dict
        dictionary with four keys, which are:

        * rcps: list of strings referring to RCPs 2.6, 4.5, 6.0 and 8.5
        * gmst_start_year: integer with the GMST data starting year, 1880
        * gmst_end_year: integer with the GMST data ending year, 2100
        * gmst_data: array with GMST data across RCPs (first dim) and years (second dim)
    """

    gmst_data = np.array(
        [
            [
                -0.16,
                -0.08,
                -0.1,
                -0.16,
                -0.28,
                -0.32,
                -0.3,
                -0.35,
                -0.16,
                -0.1,
                -0.35,
                -0.22,
                -0.27,
                -0.31,
                -0.3,
                -0.22,
                -0.11,
                -0.11,
                -0.26,
                -0.17,
                -0.08,
                -0.15,
                -0.28,
                -0.37,
                -0.47,
                -0.26,
                -0.22,
                -0.39,
                -0.43,
                -0.48,
                -0.43,
                -0.44,
                -0.36,
                -0.34,
                -0.15,
                -0.14,
                -0.36,
                -0.46,
                -0.29,
                -0.27,
                -0.27,
                -0.19,
                -0.28,
                -0.26,
                -0.27,
                -0.22,
                -0.1,
                -0.22,
                -0.2,
                -0.36,
                -0.16,
                -0.1,
                -0.16,
                -0.29,
                -0.13,
                -0.2,
                -0.15,
                -0.03,
                -0.01,
                -0.02,
                0.13,
                0.19,
                0.07,
                0.09,
                0.2,
                0.09,
                -0.07,
                -0.03,
                -0.11,
                -0.11,
                -0.17,
                -0.07,
                0.01,
                0.08,
                -0.13,
                -0.14,
                -0.19,
                0.05,
                0.06,
                0.03,
                -0.02,
                0.06,
                0.04,
                0.05,
                -0.2,
                -0.11,
                -0.06,
                -0.02,
                -0.08,
                0.05,
                0.02,
                -0.08,
                0.01,
                0.16,
                -0.07,
                -0.01,
                -0.1,
                0.18,
                0.07,
                0.16,
                0.26,
                0.32,
                0.14,
                0.31,
                0.15,
                0.11,
                0.18,
                0.32,
                0.38,
                0.27,
                0.45,
                0.4,
                0.22,
                0.23,
                0.32,
                0.45,
                0.33,
                0.47,
                0.61,
                0.39,
                0.39,
                0.54,
                0.63,
                0.62,
                0.54,
                0.68,
                0.64,
                0.66,
                0.54,
                0.66,
                0.72,
                0.61,
                0.64,
                0.68,
                0.75,
                0.9,
                1.02,
                0.92,
                0.85,
                0.98,
                0.909014286,
                0.938814286,
                0.999714286,
                1.034314286,
                1.009714286,
                1.020014286,
                1.040914286,
                1.068614286,
                1.072114286,
                1.095114286,
                1.100414286,
                1.099014286,
                1.118514286,
                1.133414286,
                1.135314286,
                1.168814286,
                1.200414286,
                1.205414286,
                1.227214286,
                1.212614286,
                1.243014286,
                1.270114286,
                1.250114286,
                1.254514286,
                1.265814286,
                1.263314286,
                1.294714286,
                1.289814286,
                1.314214286,
                1.322514286,
                1.315614286,
                1.276314286,
                1.302414286,
                1.318414286,
                1.312014286,
                1.317914286,
                1.341214286,
                1.297414286,
                1.308514286,
                1.314614286,
                1.327814286,
                1.335814286,
                1.331214286,
                1.318014286,
                1.289714286,
                1.334414286,
                1.323914286,
                1.316614286,
                1.300214286,
                1.302414286,
                1.303114286,
                1.311014286,
                1.283914286,
                1.293814286,
                1.296914286,
                1.316614286,
                1.306314286,
                1.290614286,
                1.288814286,
                1.272114286,
                1.264614286,
                1.262514286,
                1.290514286,
                1.285114286,
                1.267214286,
                1.267414286,
                1.294314286,
                1.315614286,
                1.310314286,
                1.283914286,
                1.296614286,
                1.281214286,
                1.301014286,
                1.300114286,
                1.303114286,
                1.286714286,
                1.297514286,
                1.312114286,
                1.276714286,
                1.281414286,
                1.276414286,
            ],
            [
                -0.16,
                -0.08,
                -0.1,
                -0.16,
                -0.28,
                -0.32,
                -0.3,
                -0.35,
                -0.16,
                -0.1,
                -0.35,
                -0.22,
                -0.27,
                -0.31,
                -0.3,
                -0.22,
                -0.11,
                -0.11,
                -0.26,
                -0.17,
                -0.08,
                -0.15,
                -0.28,
                -0.37,
                -0.47,
                -0.26,
                -0.22,
                -0.39,
                -0.43,
                -0.48,
                -0.43,
                -0.44,
                -0.36,
                -0.34,
                -0.15,
                -0.14,
                -0.36,
                -0.46,
                -0.29,
                -0.27,
                -0.27,
                -0.19,
                -0.28,
                -0.26,
                -0.27,
                -0.22,
                -0.1,
                -0.22,
                -0.2,
                -0.36,
                -0.16,
                -0.1,
                -0.16,
                -0.29,
                -0.13,
                -0.2,
                -0.15,
                -0.03,
                -0.01,
                -0.02,
                0.13,
                0.19,
                0.07,
                0.09,
                0.2,
                0.09,
                -0.07,
                -0.03,
                -0.11,
                -0.11,
                -0.17,
                -0.07,
                0.01,
                0.08,
                -0.13,
                -0.14,
                -0.19,
                0.05,
                0.06,
                0.03,
                -0.02,
                0.06,
                0.04,
                0.05,
                -0.2,
                -0.11,
                -0.06,
                -0.02,
                -0.08,
                0.05,
                0.02,
                -0.08,
                0.01,
                0.16,
                -0.07,
                -0.01,
                -0.1,
                0.18,
                0.07,
                0.16,
                0.26,
                0.32,
                0.14,
                0.31,
                0.15,
                0.11,
                0.18,
                0.32,
                0.38,
                0.27,
                0.45,
                0.4,
                0.22,
                0.23,
                0.32,
                0.45,
                0.33,
                0.47,
                0.61,
                0.39,
                0.39,
                0.54,
                0.63,
                0.62,
                0.54,
                0.68,
                0.64,
                0.66,
                0.54,
                0.66,
                0.72,
                0.61,
                0.64,
                0.68,
                0.75,
                0.9,
                1.02,
                0.92,
                0.85,
                0.98,
                0.903592857,
                0.949092857,
                0.955792857,
                0.997892857,
                1.048392857,
                1.068092857,
                1.104792857,
                1.122192857,
                1.125792857,
                1.156292857,
                1.160992857,
                1.201692857,
                1.234692857,
                1.255392857,
                1.274392857,
                1.283792857,
                1.319992857,
                1.369992857,
                1.385592857,
                1.380892857,
                1.415092857,
                1.439892857,
                1.457092857,
                1.493592857,
                1.520292857,
                1.517692857,
                1.538092857,
                1.577192857,
                1.575492857,
                1.620392857,
                1.657092857,
                1.673492857,
                1.669992857,
                1.706292857,
                1.707892857,
                1.758592857,
                1.739492857,
                1.740192857,
                1.797792857,
                1.839292857,
                1.865392857,
                1.857692857,
                1.864092857,
                1.881192857,
                1.907592857,
                1.918492857,
                1.933992857,
                1.929392857,
                1.931192857,
                1.942492857,
                1.985592857,
                1.997392857,
                2.000992857,
                2.028692857,
                2.016192857,
                2.020792857,
                2.032892857,
                2.057492857,
                2.092092857,
                2.106292857,
                2.117492857,
                2.123492857,
                2.121092857,
                2.096892857,
                2.126892857,
                2.131292857,
                2.144892857,
                2.124092857,
                2.134492857,
                2.171392857,
                2.163692857,
                2.144092857,
                2.145092857,
                2.128992857,
                2.129992857,
                2.169192857,
                2.186492857,
                2.181092857,
                2.217592857,
                2.210492857,
                2.223692857,
            ],
            [
                -0.16,
                -0.08,
                -0.1,
                -0.16,
                -0.28,
                -0.32,
                -0.3,
                -0.35,
                -0.16,
                -0.1,
                -0.35,
                -0.22,
                -0.27,
                -0.31,
                -0.3,
                -0.22,
                -0.11,
                -0.11,
                -0.26,
                -0.17,
                -0.08,
                -0.15,
                -0.28,
                -0.37,
                -0.47,
                -0.26,
                -0.22,
                -0.39,
                -0.43,
                -0.48,
                -0.43,
                -0.44,
                -0.36,
                -0.34,
                -0.15,
                -0.14,
                -0.36,
                -0.46,
                -0.29,
                -0.27,
                -0.27,
                -0.19,
                -0.28,
                -0.26,
                -0.27,
                -0.22,
                -0.1,
                -0.22,
                -0.2,
                -0.36,
                -0.16,
                -0.1,
                -0.16,
                -0.29,
                -0.13,
                -0.2,
                -0.15,
                -0.03,
                -0.01,
                -0.02,
                0.13,
                0.19,
                0.07,
                0.09,
                0.2,
                0.09,
                -0.07,
                -0.03,
                -0.11,
                -0.11,
                -0.17,
                -0.07,
                0.01,
                0.08,
                -0.13,
                -0.14,
                -0.19,
                0.05,
                0.06,
                0.03,
                -0.02,
                0.06,
                0.04,
                0.05,
                -0.2,
                -0.11,
                -0.06,
                -0.02,
                -0.08,
                0.05,
                0.02,
                -0.08,
                0.01,
                0.16,
                -0.07,
                -0.01,
                -0.1,
                0.18,
                0.07,
                0.16,
                0.26,
                0.32,
                0.14,
                0.31,
                0.15,
                0.11,
                0.18,
                0.32,
                0.38,
                0.27,
                0.45,
                0.4,
                0.22,
                0.23,
                0.32,
                0.45,
                0.33,
                0.47,
                0.61,
                0.39,
                0.39,
                0.54,
                0.63,
                0.62,
                0.54,
                0.68,
                0.64,
                0.66,
                0.54,
                0.66,
                0.72,
                0.61,
                0.64,
                0.68,
                0.75,
                0.9,
                1.02,
                0.92,
                0.85,
                0.98,
                0.885114286,
                0.899814286,
                0.919314286,
                0.942414286,
                0.957814286,
                1.000414286,
                1.023114286,
                1.053414286,
                1.090814286,
                1.073014286,
                1.058114286,
                1.117514286,
                1.123714286,
                1.123814286,
                1.177514286,
                1.190814286,
                1.187514286,
                1.223514286,
                1.261714286,
                1.289014286,
                1.276414286,
                1.339114286,
                1.365714286,
                1.375314286,
                1.402214286,
                1.399914286,
                1.437314286,
                1.464914286,
                1.479114286,
                1.505514286,
                1.509614286,
                1.539814286,
                1.558214286,
                1.595014286,
                1.637114286,
                1.653414286,
                1.636714286,
                1.652214286,
                1.701014286,
                1.731114286,
                1.759214286,
                1.782114286,
                1.811014286,
                1.801714286,
                1.823014286,
                1.842914286,
                1.913014286,
                1.943114286,
                1.977514286,
                1.982014286,
                2.007114286,
                2.066314286,
                2.079214286,
                2.126014286,
                2.147314286,
                2.174914286,
                2.184414286,
                2.218514286,
                2.261514286,
                2.309614286,
                2.328014286,
                2.347014286,
                2.369414286,
                2.396614286,
                2.452014286,
                2.473314286,
                2.486514286,
                2.497914286,
                2.518014286,
                2.561814286,
                2.613014286,
                2.626814286,
                2.585914286,
                2.614614286,
                2.644714286,
                2.688414286,
                2.688514286,
                2.685314286,
                2.724614286,
                2.746214286,
                2.773814286,
            ],
            [
                -0.16,
                -0.08,
                -0.1,
                -0.16,
                -0.28,
                -0.32,
                -0.3,
                -0.35,
                -0.16,
                -0.1,
                -0.35,
                -0.22,
                -0.27,
                -0.31,
                -0.3,
                -0.22,
                -0.11,
                -0.11,
                -0.26,
                -0.17,
                -0.08,
                -0.15,
                -0.28,
                -0.37,
                -0.47,
                -0.26,
                -0.22,
                -0.39,
                -0.43,
                -0.48,
                -0.43,
                -0.44,
                -0.36,
                -0.34,
                -0.15,
                -0.14,
                -0.36,
                -0.46,
                -0.29,
                -0.27,
                -0.27,
                -0.19,
                -0.28,
                -0.26,
                -0.27,
                -0.22,
                -0.1,
                -0.22,
                -0.2,
                -0.36,
                -0.16,
                -0.1,
                -0.16,
                -0.29,
                -0.13,
                -0.2,
                -0.15,
                -0.03,
                -0.01,
                -0.02,
                0.13,
                0.19,
                0.07,
                0.09,
                0.2,
                0.09,
                -0.07,
                -0.03,
                -0.11,
                -0.11,
                -0.17,
                -0.07,
                0.01,
                0.08,
                -0.13,
                -0.14,
                -0.19,
                0.05,
                0.06,
                0.03,
                -0.02,
                0.06,
                0.04,
                0.05,
                -0.2,
                -0.11,
                -0.06,
                -0.02,
                -0.08,
                0.05,
                0.02,
                -0.08,
                0.01,
                0.16,
                -0.07,
                -0.01,
                -0.1,
                0.18,
                0.07,
                0.16,
                0.26,
                0.32,
                0.14,
                0.31,
                0.15,
                0.11,
                0.18,
                0.32,
                0.38,
                0.27,
                0.45,
                0.4,
                0.22,
                0.23,
                0.32,
                0.45,
                0.33,
                0.47,
                0.61,
                0.39,
                0.39,
                0.54,
                0.63,
                0.62,
                0.54,
                0.68,
                0.64,
                0.66,
                0.54,
                0.66,
                0.72,
                0.61,
                0.64,
                0.68,
                0.75,
                0.9,
                1.02,
                0.92,
                0.85,
                0.98,
                0.945764286,
                1.011064286,
                1.048564286,
                1.049564286,
                1.070264286,
                1.126564286,
                1.195464286,
                1.215064286,
                1.246964286,
                1.272564286,
                1.262464286,
                1.293464286,
                1.340864286,
                1.391164286,
                1.428764286,
                1.452564286,
                1.494164286,
                1.520664286,
                1.557164286,
                1.633664286,
                1.654264286,
                1.693264286,
                1.730264286,
                1.795264286,
                1.824264286,
                1.823864286,
                1.880664286,
                1.952864286,
                1.991764286,
                1.994764286,
                2.085764286,
                2.105764286,
                2.155064286,
                2.227464286,
                2.249964286,
                2.313664286,
                2.341464286,
                2.394064286,
                2.457364286,
                2.484664286,
                2.549564286,
                2.605964286,
                2.656864286,
                2.707364286,
                2.742964286,
                2.789764286,
                2.847664286,
                2.903564286,
                2.925064286,
                2.962864286,
                3.002664286,
                3.069264286,
                3.133364286,
                3.174764286,
                3.217764286,
                3.256564286,
                3.306864286,
                3.375464286,
                3.420264286,
                3.476464286,
                3.493864286,
                3.552964286,
                3.592364286,
                3.630664286,
                3.672464286,
                3.734364286,
                3.789764286,
                3.838164286,
                3.882264286,
                3.936064286,
                3.984064286,
                4.055764286,
                4.098964286,
                4.122364286,
                4.172064286,
                4.225264286,
                4.275064286,
                4.339064286,
                4.375864286,
                4.408064286,
                4.477764286,
            ],
        ]
    )

    gmst_info = {
        "rcps": ["2.6", "4.5", "6.0", "8.5"],
        "gmst_start_year": 1880,
        "gmst_end_year": 2100,
        "gmst_data": gmst_data,
    }

    return gmst_info


def get_knutson_data():
    """
    Retrieve projections data in Knutson et al., (2020):

    Tropical cyclones and climate change assessment. Part II: Projected
        response to anthropogenic warming. Bull. Amer. Meteor. Soc., 101 (3), E303–E322,
        https://doi.org/10.1175/BAMS-D-18-0194.1.

    for 4 variables (i.e., cat05 frequency, cat45 frequency, intensity, precipitation rate),
    6 regions, i.e., N. Atl (NA), NW Pac. (WP), NE Pac (EP)., N. Ind (NI), S. Ind. (SI),
    SW Pac. (SP), and 5 percentiles, i.e., 5% or 10%, 25%, 50%, 75%, 95% or 90%.

    The data are available at:
        S Jewson, T Knutson, S Camargo, J Chan, K Emanuel, C Ho, J Kossin, M Mohapatra, M Satoh,
        M Sugi, K Walsh, & L Wu. (2021). Knutson et al 2020 Tropical Cyclone Projections Data (v0.2)
        [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4757343

    Returns
    -------
    knutson_data : np.array of dimension (4x6x5)
        array contaning data used by Knutson et al. (2020) to project changes in cat05 frequency,
        cat45 frequency, intensity and precipitation rate (first array's dimension), for the
        N. Atl (NA), NW Pac. (WP), NE Pac (EP)., N. Ind (NI), S. Ind. (SI), SW Pac. (SP) regions
        (second array's dimension) for the 5%/10%, 25%, 50%, 75%, 95%/90% percentiles
        (thirs array's dimension).
    """

    # The knutson_data array has dimension:
    # 4 (tropical cyclones variables) x 6 (tropical cyclone regions) x 5 (percentiles)
    knutson_data = np.array(
        [
            [
                [-34.49, -24.875, -14.444, 3.019, 28.737],
                [-30.444, -20, -10.27, 0.377, 17.252],
                [-32.075, -18.491, -3.774, 11.606, 36.682],
                [-35.094, -15.115, -4.465, 5.785, 29.405],
                [-32.778, -22.522, -17.297, -8.995, 7.241],
                [-40.417, -26.321, -18.113, -8.21, 4.689],
            ],
            [
                [-38.038, -22.264, 11.321, 38.302, 81.874],
                [-25.811, -14.34, -4.75, 16.146, 41.979],
                [-24.83, -6.792, 22.642, 57.297, 104.315],
                [-30.566, -16.415, 5.283, 38.491, 79.119],
                [-23.229, -13.611, 4.528, 26.645, 63.514],
                [-42.453, -29.434, -14.467, -0.541, 19.061],
            ],
            [
                [0.543, 1.547, 2.943, 4.734, 6.821],
                [1.939, 3.205, 5.328, 6.549, 9.306],
                [-2.217, 0.602, 5.472, 9.191, 10.368],
                [-0.973, 1.944, 4.324, 6.15, 7.808],
                [1.605, 3.455, 5.405, 7.69, 10.884],
                [-6.318, -0.783, 0.938, 5.314, 12.213],
            ],
            [
                [5.848, 9.122, 15.869, 20.352, 22.803],
                [6.273, 12.121, 16.486, 18.323, 23.784],
                [6.014, 8.108, 21.081, 29.324, 31.838],
                [12.703, 14.347, 17.649, 19.182, 20.77],
                [2.2, 11.919, 19.73, 23.115, 26.243],
                [-1.299, 5.137, 7.297, 11.091, 15.419],
            ],
        ]
    )

    return knutson_data
