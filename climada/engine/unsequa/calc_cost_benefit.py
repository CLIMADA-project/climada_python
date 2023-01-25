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

__all__ = ['CalcCostBenefit']

import logging
import time
from functools import partial
from typing import Optional, Union

import pandas as pd

from climada.engine.cost_benefit import CostBenefit
from climada.engine.unsequa import Calc, InputVar, UncCostBenefitOutput
from climada.util import log_level
from climada.hazard import Hazard
from climada.entity import Entity

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
    haz_input_var : climada.engine.uncertainty.input_var.InputVar
        Present Hazard uncertainty variable
    ent_input_var : climada.engine.uncertainty.input_var.InputVar
        Present Entity uncertainty variable
    haz_unc_fut_Var: climada.engine.uncertainty.input_var.InputVar
        Future Hazard uncertainty variable
    ent_fut_input_var : climada.engine.uncertainty.input_var.InputVar
        Future Entity uncertainty variable
    _input_var_names : tuple(str)
        Names of the required uncertainty variables
        ('haz_input_var', 'ent_input_var', 'haz_fut_input_var', 'ent_fut_input_var')
    _metric_names : tuple(str)
        Names of the cost benefit output metrics
        ('tot_climate_risk', 'benefit', 'cost_ben_ratio', 'imp_meas_present', 'imp_meas_future')
    """

    _input_var_names = (
        'haz_input_var',
        'ent_input_var',
        'haz_fut_input_var',
        'ent_fut_input_var',
    )
    """Names of the required uncertainty variables"""

    _metric_names = (
        'tot_climate_risk',
        'benefit',
        'cost_ben_ratio',
        'imp_meas_present',
        'imp_meas_future',
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



    def uncertainty(self, unc_data, pool=None, **cost_benefit_kwargs):
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
        unc_data : climada.engine.uncertainty.unc_output.UncOutput
            Uncertainty data object with the input parameters samples
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations. Default is None.
            The default is None.
        cost_benefit_kwargs : keyword arguments
            Keyword arguments passed on to climada.engine.CostBenefit.calc()

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

        See Also
        --------
        climada.engine.cost_benefit:
            Compute risk and adptation option cost benefits.

        """

        if unc_data.samples_df.empty:
            raise ValueError("No sample was found. Please create one first" +
                        "using UncImpact.make_sample(N)")

        samples_df = unc_data.samples_df.copy(deep=True)
        unit = self.value_unit

        LOGGER.info("The freq_curve is not saved. Please "
                    "change the risk_func (see climada.engine.cost_benefit) "
                    "if return period information is needed")

        start = time.time()
        one_sample = samples_df.iloc[0:1].iterrows()
        cb_metrics = map(self._map_costben_calc, one_sample)
        [imp_meas_present,
         imp_meas_future,
         tot_climate_risk,
         benefit,
         cost_ben_ratio] = list(zip(*cb_metrics))
        elapsed_time = (time.time() - start)
        self.est_comp_time(unc_data.n_samples, elapsed_time, pool)

        #Compute impact distributions
        with log_level(level='ERROR', name_prefix='climada'):
            if pool:
                LOGGER.info('Using %s CPUs.', pool.ncpus)
                chunksize = min(unc_data.n_samples // pool.ncpus, 100)
                cb_metrics = pool.map(partial(self._map_costben_calc, **cost_benefit_kwargs),
                                               samples_df.iterrows(),
                                               chunsize = chunksize)

            else:
                cb_metrics = map(partial(self._map_costben_calc, **cost_benefit_kwargs),
                                 samples_df.iterrows())

        #Perform the actual computation
        with log_level(level='ERROR', name_prefix='climada'):
            [imp_meas_present,
             imp_meas_future,
             tot_climate_risk,
             benefit,
             cost_ben_ratio] = list(zip(*cb_metrics)) #Transpose list of list

        # Assign computed impact distribution data to self
        tot_climate_risk_unc_df = \
            pd.DataFrame(tot_climate_risk, columns = ['tot_climate_risk'])

        benefit_unc_df = pd.DataFrame(benefit)
        benefit_unc_df.columns = [
            column + ' Benef'
            for column in benefit_unc_df.columns]
        cost_ben_ratio_unc_df = pd.DataFrame(cost_ben_ratio)
        cost_ben_ratio_unc_df.columns = [
            column + ' CostBen'
            for column in cost_ben_ratio_unc_df.columns]

        imp_metric_names = ['risk', 'risk_transf', 'cost_meas',
                            'cost_ins']

        im_periods = dict()
        for imp_meas, period in zip([imp_meas_present, imp_meas_future],
                                  ['present', 'future']):
            df_imp_meas = pd.DataFrame()
            name = 'imp_meas_' + period
            if imp_meas[0]:
                for imp in imp_meas:
                    met_dic = {}
                    for meas, imp_dic in imp.items():
                        metrics = [imp_dic['risk'],
                                   imp_dic['risk_transf'],
                                   *imp_dic['cost']]
                        dic_tmp = {meas + ' - ' + m_name + ' - ' + period: [m_value]
                                   for m_name, m_value
                                   in zip(imp_metric_names, metrics)
                                    }
                        met_dic.update(dic_tmp)
                    df_imp_meas = df_imp_meas.append(
                        pd.DataFrame(met_dic), ignore_index=True
                        )
            im_periods[name + '_unc_df'] = df_imp_meas
        cost_benefit_kwargs = {
            key: str(val)
            for key, val in cost_benefit_kwargs.items()}
        cost_benefit_kwargs = tuple(cost_benefit_kwargs.items())

        return UncCostBenefitOutput(samples_df=samples_df,
                                    imp_meas_present_unc_df=im_periods['imp_meas_present_unc_df'],
                                    imp_meas_future_unc_df=im_periods['imp_meas_future_unc_df'],
                                    tot_climate_risk_unc_df=tot_climate_risk_unc_df,
                                    cost_ben_ratio_unc_df=cost_ben_ratio_unc_df,
                                    benefit_unc_df=benefit_unc_df,
                                    unit=unit,
                                    cost_benefit_kwargs=cost_benefit_kwargs)

    def _map_costben_calc(self, param_sample, **kwargs):
        """
        Map to compute cost benefit for all parameter samples in parallel

        Parameters
        ----------
        param_sample : pd.DataFrame.iterrows()
            Generator of the parameter samples
        kwargs :
            Keyword arguments passed on to climada.engine.CostBenefit.calc()

        Returns
        -------
        list
            icost benefit metrics list for all samples containing
            imp_meas_present, imp_meas_future, tot_climate_risk,
            benefit, cost_ben_ratio

        """

        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        haz_samples = param_sample[1][self.haz_input_var.labels].to_dict()
        ent_samples = param_sample[1][self.ent_input_var.labels].to_dict()
        haz_fut_samples = param_sample[1][self.haz_fut_input_var.labels].to_dict()
        ent_fut_samples = param_sample[1][self.ent_fut_input_var.labels].to_dict()

        haz = self.haz_input_var.evaluate(**haz_samples)
        ent = self.ent_input_var.evaluate(**ent_samples)
        haz_fut = self.haz_fut_input_var.evaluate(**haz_fut_samples)
        ent_fut = self.ent_fut_input_var.evaluate(**ent_fut_samples)

        cb = CostBenefit()
        ent.exposures.assign_centroids(haz, overwrite=False)
        if ent_fut:
            ent_fut.exposures.assign_centroids(haz_fut if haz_fut else haz, overwrite=False)
        cb.calc(hazard=haz, entity=ent, haz_future=haz_fut, ent_future=ent_fut,
                save_imp=False, assign_centroids=False, **kwargs)

        # Extract from climada.impact the chosen metrics
        return  [cb.imp_meas_present,
                 cb.imp_meas_future,
                 cb.tot_climate_risk,
                 cb.benefit,
                 cb.cost_ben_ratio
                 ]
