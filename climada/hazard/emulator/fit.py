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

Data fitting routines for the hazard event emulator.
"""

import logging
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

LOGGER = logging.getLogger(__name__)

def fit_data(data, explained, explanatory, poisson=False):
    """Fit a response variable (e.g. intensity) to a list of explanatory variables

    The fitting is run twice, restricting to the significant explanatory
    variables in the second run.

    Parameters
    ----------
    data : DataFrame { year, `resp_var`, ... }
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
    for run in range(2):
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
    significance = ['***' if el <= 0.01 else \
                    '**' if el <= 0.05 else \
                    '*' if el <= 0.1 else \
                    '-' for el in sm_results.pvalues[:-1]]
    significance = dict(zip(fit_significant(sm_results), significance))
    return significance
