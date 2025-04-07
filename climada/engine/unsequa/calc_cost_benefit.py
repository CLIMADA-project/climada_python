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

Define Uncertainty Cost Benefit class
"""

__all__ = ["CalcCostBenefit"]

import itertools
import logging
import time
from typing import Optional, Union

import pandas as pd
import pathos.multiprocessing as mp

from climada.engine.cost_benefit import CostBenefit
from climada.engine.unsequa.calc_base import (
    Calc,
    _multiprocess_chunksize,
    _sample_parallel_iterator,
    _transpose_chunked_data,
)
from climada.engine.unsequa.input_var import InputVar
from climada.engine.unsequa.unc_output import UncCostBenefitOutput
from climada.entity import Entity
from climada.hazard import Hazard
from climada.util import log_level

# use pathos.multiprocess fork of multiprocessing for compatibility
# wiht notebooks and other environments https://stackoverflow.com/a/65001152/12454103


LOGGER = logging.getLogger(__name__)

# Future planed features:
# - Add 'efc' (frequency curve) to UncCostBenenfit


class CalcCostBenefit(Calc):
    """
    Cost Benefit uncertainty analysis class

    This is the base class to perform uncertainty analysis on the outputs of
    climada.engine.costbenefit.CostBenefit().

    Attributes
    ----------
    value_unit : str
        Unit of the exposures value
    haz_input_var : InputVar or Hazard
        Present Hazard uncertainty variable
    ent_input_var : InputVar or Entity
        Present Entity uncertainty variable
    haz_unc_fut_Var: InputVar or Hazard
        Future Hazard uncertainty variable
    ent_fut_input_var : InputVar or Entity
        Future Entity uncertainty variable
    _input_var_names : tuple(str)
        Names of the required uncertainty variables
        ('haz_input_var', 'ent_input_var', 'haz_fut_input_var', 'ent_fut_input_var')
    _metric_names : tuple(str)
        Names of the cost benefit output metrics
        ('tot_climate_risk', 'benefit', 'cost_ben_ratio',
        'imp_meas_present', 'imp_meas_future')
    """

    _input_var_names = (
        "haz_input_var",
        "ent_input_var",
        "haz_fut_input_var",
        "ent_fut_input_var",
    )
    """Names of the required uncertainty variables"""

    _metric_names = (
        "tot_climate_risk",
        "benefit",
        "cost_ben_ratio",
        "imp_meas_present",
        "imp_meas_future",
    )
    """Names of the cost benefit output metrics"""

    def __init__(
        self,
        haz_input_var: Union[InputVar, Hazard],
        ent_input_var: Union[InputVar, Entity],
        haz_fut_input_var: Optional[Union[InputVar, Hazard]] = None,
        ent_fut_input_var: Optional[Union[InputVar, Entity]] = None,
    ):
        """Initialize UncCalcCostBenefit

        Sets the uncertainty input variables, the cost benefit metric_names,
        and the units.

        Parameters
        ----------
        haz_input_var : climada.engine.uncertainty.input_var.InputVar
                        or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard for the present Hazard
            in climada.engine.CostBenefit.calc
        ent_input_var : climada.engine.uncertainty.input_var.InputVar
                        or climada.entity.Entity
            Entity uncertainty variable or Entity for the present Entity
            in climada.engine.CostBenefit.calc
        haz_fut_input_var: climada.engine.uncertainty.input_var.InputVar
                           or climada.hazard.Hazard, optional
            Hazard uncertainty variable or Hazard for the future Hazard
            The Default is None.
        ent_fut_input_var : climada.engine.uncertainty.input_var.InputVar
                            or climada.entity.Entity, optional
            Entity uncertainty variable or Entity for the future Entity
            in climada.engine.CostBenefit.calc

        """

        Calc.__init__(self)
        self.haz_input_var = InputVar.var_to_inputvar(haz_input_var)
        self.ent_input_var = InputVar.var_to_inputvar(ent_input_var)
        self.haz_fut_input_var = InputVar.var_to_inputvar(haz_fut_input_var)
        self.ent_fut_input_var = InputVar.var_to_inputvar(ent_fut_input_var)

        self.value_unit = self.ent_input_var.evaluate().exposures.value_unit
        self.check_distr()

    def uncertainty(
        self, unc_sample, processes=1, chunksize=None, **cost_benefit_kwargs
    ):
        """
        Computes the cost benefit for each sample in unc_output.sample_df.

        By default, imp_meas_present, imp_meas_future, tot_climate_risk,
        benefit, cost_ben_ratio are computed.

        This sets the attributes:
        unc_output.imp_meas_present_unc_df,
        unc_output.imp_meas_future_unc_df
        unc_output.tot_climate_risk_unc_df
        unc_output.benefit_unc_df
        unc_output.cost_ben_ratio_unc_df
        unc_output.unit
        unc_output.cost_benefit_kwargs

        Parameters
        ----------
        unc_sample : climada.engine.uncertainty.unc_output.UncOutput
            Uncertainty data object with the input parameters samples
        processes : int, optional
            Number of CPUs to use for parralel computations.
            The default is 1 (not parallel)
        cost_benefit_kwargs : keyword arguments
            Keyword arguments passed on to climada.engine.CostBenefit.calc()
        chunksize: int, optional
            Size of the sample chunks for parallel processing.
            Default is equal to the number of samples divided by the
            number of processes.

        Returns
        -------
        unc_output : climada.engine.uncertainty.unc_output.UncCostBenefitOutput
            Uncertainty data object in with the cost benefit outputs for each
            sample and all the sample data copied over from unc_sample.

        Raises
        ------
        ValueError:
            If no sampling parameters defined, the uncertainty distribution
            cannot be computed.

        Notes
        -----
        Parallelization logic is described in the base class
        here :py:class:`~climada.engine.unsequa.calc_base.Calc`

        See Also
        --------
        climada.engine.cost_benefit:
            compute risk and adptation option cost benefits.

        """

        if unc_sample.samples_df.empty:
            raise ValueError(
                "No sample was found. Please create one first"
                + "using UncImpact.make_sample(N)"
            )

        # copy may not be needed, but is kept to prevent potential
        # data corruption issues. The computational cost should be
        # minimal as only a list of floats is copied.
        samples_df = unc_sample.samples_df.copy(deep=True)

        if chunksize is None:
            chunksize = _multiprocess_chunksize(samples_df, processes)
        unit = self.value_unit

        LOGGER.info(
            "The freq_curve is not saved. Please "
            "change the risk_func (see climada.engine.cost_benefit) "
            "if return period information is needed"
        )

        one_sample = samples_df.iloc[0:1]
        start = time.time()
        self._compute_cb_metrics(
            one_sample, cost_benefit_kwargs, chunksize=1, processes=1
        )
        elapsed_time = time.time() - start
        self.est_comp_time(unc_sample.n_samples, elapsed_time, processes)

        # Compute impact distributions
        [
            imp_meas_present,
            imp_meas_future,
            tot_climate_risk,
            benefit,
            cost_ben_ratio,
        ] = self._compute_cb_metrics(
            samples_df, cost_benefit_kwargs, chunksize, processes
        )

        # Assign computed impact distribution data to self
        tot_climate_risk_unc_df = pd.DataFrame(
            tot_climate_risk, columns=["tot_climate_risk"]
        )

        benefit_unc_df = pd.DataFrame(benefit)
        benefit_unc_df.columns = [
            column + " Benef" for column in benefit_unc_df.columns
        ]
        cost_ben_ratio_unc_df = pd.DataFrame(cost_ben_ratio)
        cost_ben_ratio_unc_df.columns = [
            column + " CostBen" for column in cost_ben_ratio_unc_df.columns
        ]

        imp_metric_names = ["risk", "risk_transf", "cost_meas", "cost_ins"]

        im_periods = dict()
        for imp_meas, period in zip(
            [imp_meas_present, imp_meas_future], ["present", "future"]
        ):
            df_imp_meas = pd.DataFrame()
            name = "imp_meas_" + period
            if imp_meas[0]:
                for imp in imp_meas:
                    met_dic = {}
                    for meas, imp_dic in imp.items():
                        metrics = [
                            imp_dic["risk"],
                            imp_dic["risk_transf"],
                            *imp_dic["cost"],
                        ]
                        dic_tmp = {
                            meas + " - " + m_name + " - " + period: [m_value]
                            for m_name, m_value in zip(imp_metric_names, metrics)
                        }
                        met_dic.update(dic_tmp)
                    df_imp_meas = pd.concat(
                        [df_imp_meas, pd.DataFrame(met_dic)],
                        ignore_index=True,
                        sort=False,
                    )
            im_periods[name + "_unc_df"] = df_imp_meas
        cost_benefit_kwargs = {
            key: str(val) for key, val in cost_benefit_kwargs.items()
        }
        cost_benefit_kwargs = tuple(cost_benefit_kwargs.items())

        return UncCostBenefitOutput(
            samples_df=samples_df,
            imp_meas_present_unc_df=im_periods["imp_meas_present_unc_df"],
            imp_meas_future_unc_df=im_periods["imp_meas_future_unc_df"],
            tot_climate_risk_unc_df=tot_climate_risk_unc_df,
            cost_ben_ratio_unc_df=cost_ben_ratio_unc_df,
            benefit_unc_df=benefit_unc_df,
            unit=unit,
            cost_benefit_kwargs=cost_benefit_kwargs,
        )

    def _compute_cb_metrics(
        self, samples_df, cost_benefit_kwargs, chunksize, processes
    ):
        """Compute the uncertainty metrics

        Parameters
        ----------
        samples_df : pd.DataFrame
            dataframe of input parameter samples
        cost_benefit_kwargs: kwargs
            arguments to be passed to the cost_benefit.calc method
        chunksize : int
            size of the samples chunks
        processes : int
            number of processes to use

        Returns
        -------
        list
            values of impact metrics per sample
        """
        with log_level(level="ERROR", name_prefix="climada"):
            p_iterator = _sample_parallel_iterator(
                samples=samples_df,
                chunksize=chunksize,
                ent_input_var=self.ent_input_var,
                haz_input_var=self.haz_input_var,
                ent_fut_input_var=self.ent_fut_input_var,
                haz_fut_input_var=self.haz_fut_input_var,
                cost_benefit_kwargs=cost_benefit_kwargs,
            )
            if processes > 1:
                with mp.Pool(processes=processes) as pool:
                    LOGGER.info("Using %s CPUs.", processes)
                    cb_metrics = pool.starmap(_map_costben_calc, p_iterator)
            else:
                cb_metrics = itertools.starmap(_map_costben_calc, p_iterator)

        # Perform the actual computation
        with log_level(level="ERROR", name_prefix="climada"):
            return _transpose_chunked_data(cb_metrics)


def _map_costben_calc(
    sample_chunks,
    ent_input_var,
    haz_input_var,
    ent_fut_input_var,
    haz_fut_input_var,
    cost_benefit_kwargs,
):
    """
     Map to compute cost benefit for all parameter samples in parallel

     Parameters
     ----------
     sample_chunks : pd.DataFrame
         Dataframe of the parameter samples
     haz_input_var : InputVar
         Hazard uncertainty variable or Hazard for the present Hazard
         in climada.engine.CostBenefit.calc
     ent_input_var : InputVar
         Entity uncertainty variable or Entity for the present Entity
         in climada.engine.CostBenefit.calc
     haz_fut_input_var: InputVar
         Hazard uncertainty variable or Hazard for the future Hazard
     ent_fut_input_var : InputVar
         Entity uncertainty variable or Entity for the future Entity
         in climada.engine.CostBenefit.calc
    cost_benefit_kwargs :
         Keyword arguments passed on to climada.engine.CostBenefit.calc()

     Returns
     -------
     list
         icost benefit metrics list for all samples containing
         imp_meas_present, imp_meas_future, tot_climate_risk,
         benefit, cost_ben_ratio

    """

    uncertainty_values = []
    for _, sample in sample_chunks.iterrows():
        haz_samples = sample[haz_input_var.labels].to_dict()
        ent_samples = sample[ent_input_var.labels].to_dict()
        haz_fut_samples = sample[haz_fut_input_var.labels].to_dict()
        ent_fut_samples = sample[ent_fut_input_var.labels].to_dict()

        haz = haz_input_var.evaluate(**haz_samples)
        ent = ent_input_var.evaluate(**ent_samples)
        haz_fut = haz_fut_input_var.evaluate(**haz_fut_samples)
        ent_fut = ent_fut_input_var.evaluate(**ent_fut_samples)

        cb = CostBenefit()
        ent.exposures.assign_centroids(haz, overwrite=False)
        if ent_fut:
            ent_fut.exposures.assign_centroids(
                haz_fut if haz_fut else haz, overwrite=False
            )
        cb.calc(
            hazard=haz,
            entity=ent,
            haz_future=haz_fut,
            ent_future=ent_fut,
            save_imp=False,
            assign_centroids=False,
            **cost_benefit_kwargs
        )
        # Extract from climada.impact the chosen metrics
        uncertainty_values.append(
            [
                cb.imp_meas_present,
                cb.imp_meas_future,
                cb.tot_climate_risk,
                cb.benefit,
                cb.cost_ben_ratio,
            ]
        )

    # Transpose list
    return list(zip(*uncertainty_values))
