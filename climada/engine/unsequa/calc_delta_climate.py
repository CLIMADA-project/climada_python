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

Define Uncertainty class for Impact differences between two climates.
"""

__all__ = ["CalcDeltaImpact"]

import itertools
import logging
import time
from typing import Union

import numpy as np
import pandas as pd
import pathos.multiprocessing as mp

from climada.engine import ImpactCalc
from climada.engine.unsequa.calc_base import (
    Calc,
    _multiprocess_chunksize,
    _sample_parallel_iterator,
    _transpose_chunked_data,
)
from climada.engine.unsequa.input_var import InputVar
from climada.engine.unsequa.unc_output import UncImpactOutput
from climada.entity import Exposures, ImpactFuncSet
from climada.hazard import Hazard
from climada.util import log_level
from climada.util.value_representation import safe_divide

# use pathos.multiprocess fork of multiprocessing for compatibility
# wiht notebooks and other environments https://stackoverflow.com/a/65001152/12454103


LOGGER = logging.getLogger(__name__)


class CalcDeltaImpact(Calc):
    """
    Delta Impact uncertainty caclulation class.

    This is the class to perform uncertainty analysis on the outputs of a relative
    change in impact from a (final impact - initial impact) / initial impact.
    Impact objects are regular climada.engine.impact.Impact() objects. The resulting
    Delta Impact is a relative change (fraction of the inital impact). The relative
    change is intuitive to understand in contrast to absolute changes which are
    hard to understand without knowledge of the absolute initial (baseline) state.

    Attributes
    ----------
    rp : list(int)
        List of the chosen return periods.
    calc_eai_exp : bool
        Compute eai_exp or not
    calc_at_event : bool
        Compute eai_exp or not
    value_unit : str
        Unit of the exposures value
    exp_input_var : InputVar or Exposures
        Exposure uncertainty variable
    impf_input_var : InputVar if ImpactFuncSet
        Impact function set uncertainty variable
    haz_input_var: InputVar or Hazard
        Hazard uncertainty variable
    _input_var_names : tuple(str)
        Names of the required uncertainty input variables
        ('exp_initial_input_var', 'impf_initial_input_var', 'haz_initial_input_var',
        'exp_final_input_var', 'impf_final_input_var', 'haz_final_input_var'')
    _metric_names : tuple(str)
        Names of the impact output metrics
        ('aai_agg', 'freq_curve', 'at_event', 'eai_exp')
    """

    _input_var_names = (
        "exp_initial_input_var",
        "impf_initial_input_var",
        "haz_initial_input_var",
        "exp_final_input_var",
        "impf_final_input_var",
        "haz_final_input_var",
    )
    """Names of the required uncertainty variables"""

    _metric_names = ("aai_agg", "freq_curve", "at_event", "eai_exp")
    """Names of the cost benefit output metrics"""

    def __init__(
        self,
        exp_initial_input_var: Union[InputVar, Exposures],
        impf_initial_input_var: Union[InputVar, ImpactFuncSet],
        haz_initial_input_var: Union[InputVar, Hazard],
        exp_final_input_var: Union[InputVar, Exposures],
        impf_final_input_var: Union[InputVar, ImpactFuncSet],
        haz_final_input_var: Union[InputVar, Hazard],
    ):
        """Initialize UncCalcImpact

        Sets the uncertainty input variables, the impact metric_names, and the
        units.

        Parameters
        ----------
        exp_initial_input_var : climada.engine.uncertainty.input_var.InputVar
            or climada.entity.Exposure
            Exposure uncertainty variable or Exposure of initial state
        impf_initital_input_var : climada.engine.uncertainty.input_var.InputVar
            or climada.entity.ImpactFuncSet
            Impact function set uncertainty variable or Impact function set of
            initial state
        haz_initial_input_var : climada.engine.uncertainty.input_var.InputVar
            or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard of initial state
        exp_final_input_var : climada.engine.uncertainty.input_var.InputVar or
            climada.entity.Exposure
            Exposure uncertainty variable or Exposure of final state
        impf_final_input_var : climada.engine.uncertainty.input_var.InputVar or
            climada.entity.ImpactFuncSet
            Impact function set uncertainty variable or Impact function set of
            final state
        haz_final_input_var : climada.engine.uncertainty.input_var.InputVar or
            climada.hazard.Hazard
            Hazard uncertainty variable or Hazard of final state

        """

        Calc.__init__(self)
        self.exp_initial_input_var = InputVar.var_to_inputvar(exp_initial_input_var)
        self.impf_initial_input_var = InputVar.var_to_inputvar(impf_initial_input_var)
        self.haz_initial_input_var = InputVar.var_to_inputvar(haz_initial_input_var)
        self.exp_final_input_var = InputVar.var_to_inputvar(exp_final_input_var)
        self.impf_final_input_var = InputVar.var_to_inputvar(impf_final_input_var)
        self.haz_final_input_var = InputVar.var_to_inputvar(haz_final_input_var)

        self.value_unit = self.exp_initial_input_var.evaluate().value_unit
        self.check_distr()

    def uncertainty(
        self,
        unc_sample,
        rp=None,
        calc_eai_exp=False,
        calc_at_event=False,
        relative_delta=True,
        processes=1,
        chunksize=None,
    ):
        """
        Computes the differential impact between the reference (initial) and
        future (final) for each sample in unc_data.sample_df.

        By default, the aggregated average impact within a period of 1/frequency_unit
        (impact.aai_agg) and the excess impact at return periods rp
        (imppact.calc_freq_curve(self.rp).impact) is computed.
        Optionally, eai_exp and at_event is computed (this may require
        a larger amount of memory if the number of samples and/or the number
        of centroids and/or exposures points is large).
        For all metrics, the impacts are caculated first and the the difference
        thereof is computed.
        For example:
        (impact_final.aai_agg - impact_inital.aai_agg / impact_inital.aai_agg)

        This sets the attributes self.rp, self.calc_eai_exp,
        self.calc_at_event, self.metrics.

        This sets the attributes:
        unc_output.aai_agg_unc_df,
        unc_output.freq_curve_unc_df
        unc_output.eai_exp_unc_df
        unc_output.at_event_unc_df
        unc_output.unit

        Parameters
        ----------
        unc_sample : climada.engine.uncertainty.unc_output.UncOutput
            Uncertainty data object with the input parameters samples
        rp : list(int), optional
            Return periods in years to be computed.
            The default is [5, 10, 20, 50, 100, 250].
        calc_eai_exp : boolean, optional
            Toggle computation of the impact at each centroid location.
            The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.
        relative_delta : bool, optional
            Normalize delta impacts by past impacts or not.
            The default is True.
        processes : int, optional
            Number of CPUs to use for parralel computations.
            The default is 1 (not parallel)
        chunksize: int, optional
            Size of the sample chunks for parallel processing.
            Default is equal to the number of samples divided by the
            number of processes.

        Returns
        -------
        unc_output : climada.engine.uncertainty.unc_output.UncImpactOutput
            Uncertainty data object with the delta impact outputs for each sample
            and all the sample data copied over from unc_sample.

        Raises
        ------
        ValueError:
            If no sampling parameters defined, the distribution cannot
            be computed.

        Notes
        -----
        Parallelization logic is described in the base class
        here :py:class:`~climada.engine.unsequa.calc_base.Calc`

        See Also
        --------
        climada.engine.impact:
            compute impact and risk.

        """

        if unc_sample.samples_df.empty:
            raise ValueError(
                "No sample was found. Please create one first"
                "using UncImpact.make_sample(N)"
            )

        # copy may not be needed, but is kept to prevent potential
        # data corruption issues. The computational cost should be
        # minimal as only a list of floats is copied.'''
        samples_df = unc_sample.samples_df.copy(deep=True)

        if chunksize is None:
            chunksize = _multiprocess_chunksize(samples_df, processes)
        unit = self.value_unit

        if rp is None:
            rp = [5, 10, 20, 50, 100, 250]

        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event
        self.relative_delta = relative_delta

        one_sample = samples_df.iloc[0:1]
        start = time.time()
        self._compute_imp_metrics(one_sample, chunksize=1, processes=1)
        elapsed_time = time.time() - start
        self.est_comp_time(unc_sample.n_samples, elapsed_time, processes)

        [
            aai_agg_list,
            freq_curve_list,
            eai_exp_list,
            at_event_list,
        ] = self._compute_imp_metrics(
            samples_df, chunksize=chunksize, processes=processes
        )

        # Assign computed impact distribution data to self
        aai_agg_unc_df = pd.DataFrame(aai_agg_list, columns=["aai_agg"])
        freq_curve_unc_df = pd.DataFrame(
            freq_curve_list, columns=["rp" + str(n) for n in rp]
        )
        eai_exp_unc_df = pd.DataFrame(eai_exp_list)
        at_event_unc_df = pd.DataFrame(at_event_list)

        if calc_eai_exp:
            exp = self.exp_input_var.evaluate()
            coord_df = exp.gdf[["latitude", "longitude"]]
        else:
            coord_df = pd.DataFrame([])

        return UncImpactOutput(
            samples_df=samples_df,
            unit=unit,
            aai_agg_unc_df=aai_agg_unc_df,
            freq_curve_unc_df=freq_curve_unc_df,
            eai_exp_unc_df=eai_exp_unc_df,
            at_event_unc_df=at_event_unc_df,
            coord_df=coord_df,
        )

    def _compute_imp_metrics(self, samples_df, chunksize, processes):
        """Compute the uncertainty metrics

        Parameters
        ----------
        samples_df : pd.DataFrame
            dataframe of input parameter samples
        chunksize : int
            size of the samples chunks
        processes : int
            number of processes to use

        Returns
        -------
        list
            values of impact metrics per sample
        """
        # Compute impact distributions
        with log_level(level="ERROR", name_prefix="climada"):
            p_iterator = _sample_parallel_iterator(
                samples=samples_df,
                chunksize=chunksize,
                exp_initial_input_var=self.exp_initial_input_var,
                impf_initial_input_var=self.impf_initial_input_var,
                haz_initial_input_var=self.haz_initial_input_var,
                exp_final_input_var=self.exp_final_input_var,
                impf_final_input_var=self.impf_final_input_var,
                haz_final_input_var=self.haz_final_input_var,
                rp=self.rp,
                calc_eai_exp=self.calc_eai_exp,
                calc_at_event=self.calc_at_event,
                relative_delta=self.relative_delta,
            )
            if processes > 1:
                with mp.Pool(processes=processes) as pool:
                    LOGGER.info("Using %s CPUs.", processes)
                    imp_metrics = pool.starmap(_map_impact_calc, p_iterator)
            else:
                imp_metrics = itertools.starmap(_map_impact_calc, p_iterator)

        # Perform the actual computation
        with log_level(level="ERROR", name_prefix="climada"):
            return _transpose_chunked_data(imp_metrics)


def _map_impact_calc(
    sample_chunks,
    exp_initial_input_var,
    impf_initial_input_var,
    haz_initial_input_var,
    exp_final_input_var,
    impf_final_input_var,
    haz_final_input_var,
    rp,
    calc_eai_exp,
    calc_at_event,
    relative_delta,
):
    """
    Map to compute impact for all parameter samples in parallel

    Parameters
    ----------
    sample_chunks : pd.DataFrame
        Dataframe of the parameter samples
    exp_input_var : InputVar or Exposures
        Exposure uncertainty variable
    impf_input_var : InputVar if ImpactFuncSet
        Impact function set uncertainty variable
    haz_input_var: InputVar or Hazard
        Hazard uncertainty variable
    rp : list(int)
        List of the chosen return periods.
    calc_eai_exp : bool
        Compute eai_exp or not
    calc_at_event : bool
        Compute at_event or not
    relative_delta : bool
        Normalize delta impacts by past impacts or not

    Returns
    -------
        : list
        impact metrics list for all samples containing aai_agg, rp_curve,
        eai_exp (np.array([]) if self.calc_eai_exp=False) and at_event
        (np.array([]) if self.calc_at_event=False).

    """
    uncertainty_values = []
    for _, sample in sample_chunks.iterrows():
        exp_initial_samples = sample[exp_initial_input_var.labels].to_dict()
        impf_initial_samples = sample[impf_initial_input_var.labels].to_dict()
        haz_initial_samples = sample[haz_initial_input_var.labels].to_dict()
        exp_final_samples = sample[exp_final_input_var.labels].to_dict()
        impf_final_samples = sample[impf_final_input_var.labels].to_dict()
        haz_final_samples = sample[haz_final_input_var.labels].to_dict()

        exp_initial = exp_initial_input_var.evaluate(**exp_initial_samples)
        impf_initial = impf_initial_input_var.evaluate(**impf_initial_samples)
        haz_initial = haz_initial_input_var.evaluate(**haz_initial_samples)
        exp_final = exp_final_input_var.evaluate(**exp_final_samples)
        impf_final = impf_final_input_var.evaluate(**impf_final_samples)
        haz_final = haz_final_input_var.evaluate(**haz_final_samples)

        exp_initial.assign_centroids(haz_initial, overwrite=False)
        exp_final.assign_centroids(haz_final, overwrite=False)

        imp_initial = ImpactCalc(
            exposures=exp_initial, impfset=impf_initial, hazard=haz_initial
        ).impact(assign_centroids=False, save_mat=False)
        imp_final = ImpactCalc(
            exposures=exp_final, impfset=impf_final, hazard=haz_final
        ).impact(assign_centroids=False, save_mat=False)

        # Extract from climada.impact the chosen metrics
        freq_curve_initial = imp_initial.calc_freq_curve(rp).impact
        freq_curve_final = imp_final.calc_freq_curve(rp).impact

        if calc_eai_exp:
            eai_exp_initial = imp_initial.eai_exp
            eai_exp_final = imp_final.eai_exp
        else:
            eai_exp_initial = np.array([])
            eai_exp_final = np.array([])

        if calc_at_event:
            at_event_initial = imp_initial.at_event
            at_event_final = imp_final.at_event
        else:
            at_event_initial = np.array([])
            at_event_final = np.array([])

        if relative_delta:
            delta_func = lambda x, y: safe_divide(x - y, y)
        else:
            delta_func = lambda x, y: x - y

        delta_aai_agg = delta_func(imp_final.aai_agg, imp_initial.aai_agg)

        delta_freq_curve = delta_func(freq_curve_final, freq_curve_initial)

        delta_eai_exp = (
            delta_func(eai_exp_final, eai_exp_initial) if calc_eai_exp else np.array([])
        )

        delta_at_event = (
            delta_func(at_event_final, at_event_initial)
            if calc_at_event
            else np.array([])
        )

        uncertainty_values.append(
            [delta_aai_agg, delta_freq_curve, delta_eai_exp, delta_at_event]
        )

    return list(zip(*uncertainty_values))
