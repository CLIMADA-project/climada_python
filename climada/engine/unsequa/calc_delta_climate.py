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

Define Uncertainty Delta Impact class
"""

__all__ = ['CalcDeltaImpact']

import logging
import time
from typing import Union

import pandas as pd
import numpy as np

from climada.engine import ImpactCalc
from climada.engine.unsequa import Calc, InputVar, UncImpactOutput
from climada.entity import Exposures, ImpactFuncSet
from climada.hazard import Hazard
from climada.util import log_level

LOGGER = logging.getLogger(__name__)


class CalcDeltaImpact(Calc):
    """
    Impact uncertainty caclulation class.

    This is the class to perform uncertainty analysis on the outputs of a
    climada.engine.impact.Impact() object.

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
    exp_input_var : climada.engine.uncertainty.input_var.InputVar
        Exposure uncertainty variable
    impf_input_var : climada.engine.uncertainty.input_var.InputVar
        Impact function set uncertainty variable
    haz_input_var: climada.engine.uncertainty.input_var.InputVar
        Hazard uncertainty variable
    _input_var_names : tuple(str)
        Names of the required uncertainty input variables
        ('exp_input_var', 'impf_input_var', 'haz_input_var')
    _metric_names : tuple(str)
        Names of the impact output metrics
        ('aai_agg', 'freq_curve', 'at_event', 'eai_exp', 'tot_value')
    """

    _input_var_names = (
        'exp_initial_input_var',
        'impf_initial_input_var',
        'haz_initial_input_var',
        'exp_final_input_var',
        'impf_final_input_var',
        'haz_final_input_var',
    )
    """Names of the required uncertainty variables"""

    _metric_names = (
        'aai_agg',
        'freq_curve',
        'at_event',
        'eai_exp',
        'tot_value',
    )
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
        exp_input_var : climada.engine.uncertainty.input_var.InputVar or climada.entity.Exposure
            Exposure uncertainty variable or Exposure
        impf_input_var : climada.engine.uncertainty.input_var.InputVar or climada.entity.ImpactFuncSet
            Impact function set uncertainty variable or Impact function set
        haz_input_var : climada.engine.uncertainty.input_var.InputVar or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard

        """

        Calc.__init__(self)
        self.exp_initial_input_var =  InputVar.var_to_inputvar(exp_initial_input_var)
        self.impf_initial_input_var =  InputVar.var_to_inputvar(impf_initial_input_var)
        self.haz_initial_input_var =  InputVar.var_to_inputvar(haz_initial_input_var)
        self.exp_final_input_var =  InputVar.var_to_inputvar(exp_final_input_var)
        self.impf_final_input_var =  InputVar.var_to_inputvar(impf_final_input_var)
        self.haz_final_input_var =  InputVar.var_to_inputvar(haz_final_input_var)

        self.value_unit = self.exp_initial_input_var.evaluate().value_unit
        self.check_distr()


    def uncertainty(self,
                    unc_sample,
                    rp=None,
                    calc_eai_exp=False,
                    calc_at_event=False,
                    pool=None
                    ):
        """
        Computes the impact for each sample in unc_data.sample_df.

        By default, the aggregated average impact within a period of 1/frequency_unit
        (impact.aai_agg) and the excees impact at return periods rp
        (imppact.calc_freq_curve(self.rp).impact) is computed.
        Optionally, eai_exp and at_event is computed (this may require
        a larger amount of memory if the number of samples and/or the number
        of centroids and/or exposures points is large).

        This sets the attributes self.rp, self.calc_eai_exp,
        self.calc_at_event, self.metrics.

        This sets the attributes:
        unc_output.aai_agg_unc_df,
        unc_output.freq_curve_unc_df
        unc_output.eai_exp_unc_df
        unc_output.at_event_unc_df
        unc_output.tot_value_unc_df
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
        pool : pathos.pools.ProcessPool, optional
            Pool of CPUs for parralel computations.
            The default is None.

        Returns
        -------
        unc_output : climada.engine.uncertainty.unc_output.UncImpactOutput
            Uncertainty data object with the impact outputs for each sample
            and all the sample data copied over from unc_sample.

        Raises
        ------
        ValueError:
            If no sampling parameters defined, the distribution cannot
            be computed.

        See Also
        --------
        climada.engine.impact: Compute risk.

        """

        if unc_sample.samples_df.empty:
            raise ValueError("No sample was found. Please create one first"
                             "using UncImpact.make_sample(N)")

        samples_df = unc_sample.samples_df.copy(deep=True)

        unit = self.value_unit

        if rp is None:
            rp=[5, 10, 20, 50, 100, 250]

        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event

        start = time.time()
        one_sample = samples_df.iloc[0:1].iterrows()
        delta_metrics = map(self._map_delta_calc, one_sample)
        [aai_agg_list, freq_curve_list,
         eai_exp_list, at_event_list, tot_value_list] = list(zip(*delta_metrics))
        elapsed_time = (time.time() - start)
        self.est_comp_time(unc_sample.n_samples, elapsed_time, pool)

        #Compute impact distributions
        with log_level(level='ERROR', name_prefix='climada'):
            if pool:
                LOGGER.info('Using %s CPUs.', pool.ncpus)
                chunksize = min(unc_sample.n_samples // pool.ncpus, 100)
                delta_metrics = pool.map(self._map_delta_calc,
                                        samples_df.iterrows(),
                                        chunsize = chunksize)

            else:
                delta_metrics = map(self._map_delta_calc,
                                  samples_df.iterrows())

        #Perform the actual computation
        with log_level(level='ERROR', name_prefix='climada'):
            [aai_agg_list, freq_curve_list,
             eai_exp_list, at_event_list,
             tot_value_list] = list(zip(*delta_metrics))

        # Assign computed impact distribution data to self
        aai_agg_unc_df  = pd.DataFrame(aai_agg_list,
                                                columns = ['aai_agg'])
        freq_curve_unc_df = pd.DataFrame(freq_curve_list,
                                    columns=['rp' + str(n) for n in rp])
        eai_exp_unc_df =  pd.DataFrame(eai_exp_list)
        # Setting to sparse dataframes is not compatible with .to_hdf5
        # if np.count_nonzero(df_eai_exp.to_numpy()) / df_eai_exp.size < 0.5:
        #     df_eai_exp = df_eai_exp.astype(pd.SparseDtype("float", 0.0))
        #eai_exp_unc_df = df_eai_exp
        at_event_unc_df = pd.DataFrame(at_event_list)
        # Setting to sparse dataframes is not compatible with .to_hdf5
        # if np.count_nonzero(df_at_event.to_numpy()) / df_at_event.size < 0.5:
        #     df_at_event = df_at_event.astype(pd.SparseDtype("float", 0.0))
        #at_event_unc_df = df_at_event
        tot_value_unc_df = pd.DataFrame(tot_value_list,
                                                 columns = ['tot_value'])

        if calc_eai_exp:
            exp = self.exp_input_var.evaluate()
            coord_df = exp.gdf[['latitude', 'longitude']]
        else:
            coord_df = pd.DataFrame([])

        return UncImpactOutput(samples_df=samples_df,
                               unit=unit,
                               aai_agg_unc_df=aai_agg_unc_df,
                               freq_curve_unc_df=freq_curve_unc_df,
                               eai_exp_unc_df=eai_exp_unc_df,
                               at_event_unc_df=at_event_unc_df,
                               tot_value_unc_df=tot_value_unc_df,
                               coord_df=coord_df
                               )


    def _map_delta_calc(self, sample_iterrows):
        """
        Map to compute impact for all parameter samples in parrallel

        Parameters
        ----------
        sample_iterrows : pd.DataFrame.iterrows()
            Generator of the parameter samples

        Returns
        -------
         : list
            impact metrics list for all samples containing aai_agg, rp_curve,
            eai_exp (np.array([]) if self.calc_eai_exp=False) and at_event
            (np.array([]) if self.calc_at_event=False).

        """

        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        exp_initial_samples = sample_iterrows[1][self.exp_initial_input_var.labels].to_dict()
        impf_initial_samples = sample_iterrows[1][self.impf_initial_input_var.labels].to_dict()
        haz_initial_samples = sample_iterrows[1][self.haz_initial_input_var.labels].to_dict()
        exp_final_samples = sample_iterrows[1][self.exp_final_input_var.labels].to_dict()
        impf_final_samples = sample_iterrows[1][self.impf_final_input_var.labels].to_dict()
        haz_final_samples = sample_iterrows[1][self.haz_final_input_var.labels].to_dict()

        exp_initial = self.exp_initial_input_var.evaluate(**exp_initial_samples)
        impf_initial = self.impf_initial_input_var.evaluate(**impf_initial_samples)
        haz_initial = self.haz_initial_input_var.evaluate(**haz_initial_samples)
        exp_final = self.exp_final_input_var.evaluate(**exp_final_samples)
        impf_final = self.impf_final_input_var.evaluate(**impf_final_samples)
        haz_final = self.haz_final_input_var.evaluate(**haz_final_samples)

        exp_initial.assign_centroids(haz_initial, overwrite=False)
        exp_final.assign_centroids(haz_final, overwrite=False)

        imp_initial = ImpactCalc(exposures=exp_initial, impfset=impf_initial, hazard=haz_initial)\
              .impact(assign_centroids=False)
        imp_final = ImpactCalc(exposures=exp_final, impfset=impf_final, hazard=haz_final)\
              .impact(assign_centroids=False)

        # Extract from climada.impact the chosen metrics
        freq_curve_initial = imp_initial.calc_freq_curve(self.rp).impact
        freq_curve_final = imp_final.calc_freq_curve(self.rp).impact


        if self.calc_eai_exp:
            eai_exp_initial = imp_initial.eai_exp
            eai_exp_final = imp_final.eai_exp
        else:
            eai_exp_initial = np.array([])
            eai_exp_final = np.array([])

        if self.calc_at_event:
            at_event_initial = imp_initial.at_event
            at_event_final = imp_final.at_event
        else:
            at_event_initial = np.array([])
            at_event_final = np.array([])


        return [
            imp_final.aai_agg - imp_initial.aai_agg,
            freq_curve_final - freq_curve_initial,
            eai_exp_final - eai_exp_initial,
            at_event_final - at_event_initial,
            imp_final.tot_value - imp_initial.tot_value]
