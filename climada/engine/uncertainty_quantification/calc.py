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

Define UncCalc (uncertainty calculate) class.
"""

import logging

import pandas as pd
import datetime as dt
import numpy as np

from climada.util.value_representation import sig_dig as u_sig_dig
from climada.util.config import setup_logging as u_setup_logging

LOGGER = logging.getLogger(__name__)
u_setup_logging()


class UncCalc():

    def __init__(self):
        """
        Empty constructor to be overwritten by subclasses
        """
        self.unc_var_names = ()
        self.metrics_names = ()

    @property
    def unc_vars(self):
        """
        Uncertainty variables

        Returns
        -------
        tuple(UncVar)
            All uncertainty variables associated with the calculation

        """
        return (getattr(self, var) for var in self.unc_var_names)

    @property
    def distr_dict(self):
        """
        Dictionary of the input variable distribution

        Probabilitiy density distribution of all the parameters of all the
        uncertainty variables listed in self.unc_vars

        Returns
        -------
        distr_dict : dict( sp.stats objects )
            Dictionary of all probability density distributions.

        """

        distr_dict = dict()
        for unc_var in self.unc_vars:
            distr_dict.update(unc_var.distr_dict)
        return distr_dict


    def est_comp_time(self, n_samples, time_one_run, pool=None):
        """
        Estimate the computation time

        Parameters
        ----------
        n_samples : int/float
            The total number of samples
        time_one_run : int/float
            Estimated computation time for one parameter set in seconds
        pool : pathos.pool, optional
            pool that would be used for parallel computation.
            The default is None.

        Returns
        -------
        Estimated computation time in secs.

        """
        time_one_run = u_sig_dig(time_one_run, n_sig_dig=3)
        if time_one_run > 5:
            LOGGER.warning("Computation time for one set of parameters is "
                "%.2fs. This is rather long."
                "Potential reasons: unc_vars are loading data, centroids have "
                "been assigned to exp before defining unc_var, ..."
                "\n If computation cannot be reduced, consider using"
                " a surrogate model https://www.uqlab.com/", time_one_run)

        ncpus = pool.ncpus if pool else 1
        total_time = n_samples * time_one_run / ncpus
        LOGGER.info("\n\nEstimated computaion time: %s\n",
                    dt.timedelta(seconds=total_time))

        return total_time

    def make_sample(self, unc_data, N, sampling_method='saltelli',
                    sampling_kwargs = None):
        """
        Make samples of input variables

        For all input parameters, sample from their respective
        distributions using the chosen sampling_method from SALib.
        https://salib.readthedocs.io/en/latest/api.html

        This sets the attribute unc_data.samples_df, unc_data.sampling_method
        and unc_data.sampling_kwargs.

        Parameters
        ----------
        unc_data : climada.engine.uncertainty.unc_data.UncData()
            Uncertainty data object in which to store the samples
        N : int
            Number of samples as used in the sampling method from SALib
        sampling_method : str, optional
            The sampling method as defined in SALib. Possible choices:
            'saltelli', 'fast_sampler', 'latin', 'morris', 'dgsm', 'ff'
            https://salib.readthedocs.io/en/latest/api.html
            The default is 'saltelli'.
        sampling_kwargs : kwargs, optional
            Optional keyword arguments passed on to the SALib sampling_method.
            The default is None.

        Raises
        ------
        ValueError
            Error if trying to override existing sample in unc_data

        Returns
        -------
        None.


        See Also
        --------
        SALib.sample: sampling methods from SALib SALib.sample
            https://salib.readthedocs.io/en/latest/api.html

        """
        if not unc_data.samples_df.empty:
            raise ValueError("Samples already present. Please delete the "
                             "content of unc_data.samples_df before making "
                             "new samples")
        if sampling_kwargs is None:
            sampling_kwargs = {}

        distr_dict = dict()
        for var in self.unc_vars:
            distr_dict.update(var.distr_dict)

        param_labels = list(distr_dict.keys())
        problem_sa = {
            'num_vars' : len(param_labels),
            'names' : param_labels,
            'bounds' : [[0, 1]]*len(param_labels)
            }

        uniform_base_sample = self._make_uniform_base_sample(N, problem_sa,
                                                             sampling_method,
                                                             sampling_kwargs)
        df_samples = pd.DataFrame(uniform_base_sample, columns=param_labels)
        for param in list(df_samples):
            df_samples[param] = df_samples[param].apply(
                self.distr_dict[param].ppf
                )

        sampling_kwargs = {
            key: str(val)
            for key, val in sampling_kwargs.items()
            }
        df_samples.attrs['sampling_method'] = sampling_method
        df_samples.attrs['sampling_kwargs'] = tuple(sampling_kwargs.items())
        unc_data.samples_df = df_samples
        LOGGER.info("Effective number of made samples: %d", unc_data.n_samples)

    def _make_uniform_base_sample(self, N, problem_sa, sampling_method,
                                  sampling_kwargs):
        """
        Make a uniform distributed [0,1] sample for the defined
        uncertainty parameters (self.param_labels) with the chosen
        method from SALib (self.sampling_method)
        https://salib.readthedocs.io/en/latest/api.html

        Parameters
        ----------
        N: int
            Number of samples as defined for the SALib sample method.
            Note that the effective number of created samples might be
            larger (c.f. SALib)
        problem_sa: dict()
            Description of input variables. Is used as argument for the
            SALib sampling method.
        sampling_method: string
            The sampling method as defined in SALib. Possible choices:
            'saltelli', 'fast_sampler', 'latin', 'morris', 'dgsm', 'ff'
            https://salib.readthedocs.io/en/latest/api.html
        sampling_kwargs: dict()
            Optional keyword arguments passed on to the SALib sampling method.

        Returns
        -------
        sample_uniform : np.matrix
            Returns a NumPy matrix containing the sampled uncertainty
            parameters using the defined sampling method

        """

        if sampling_kwargs is None:
            sampling_kwargs = {}

        #Import the named submodule from the SALib sample module
        #From the workings of __import__ the use of 'from_list' is necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        salib_sampling_method = getattr(
            __import__('SALib.sample',
                       fromlist=[sampling_method]
                       ),
            sampling_method
            )
        sample_uniform = salib_sampling_method.sample(
            problem = problem_sa, N = N, **sampling_kwargs)
        return sample_uniform

    def calc_sensitivity(self, unc_data, sensitivity_method = 'sobol',
                         sensitivity_kwargs = None):
        """
        Compute the sensitivity indices using SALib.

        Prior to doing the sensitivity analysis, one must compute the
        uncertainty (distribution) of the output values
        (i.e. self.uncertainty_metricsis defined) for all the samples
        (rows of self.samples_df).

        According to Wikipedia, sensitivity analysis is “the study of how the
        uncertainty in the output of a mathematical model or system (numerical
        or otherwise) can be apportioned to different sources of uncertainty
        in its inputs.” The sensitivity of each input is often represented by
        a numeric value, called the sensitivity index. Sensitivity indices
        come in several forms.

        This sets the attribute unc_data.sensistivity_method and
        unc_data.sensitivity_kwargs. For each climada
        metric xxx, an attribute unc_data.xxx_sens_df is set.
        Metrics:
            impact: aai_agg, freq_curve, at_event, eai_exp, tot_value)
            cost benefit: tot_climate_risk, benefit, cost_ben_ratio,
                imp_meas_present, imp_meas_future

        Parameters
        ----------
        unc_data : climada.engine.uncertainty.unc_data.UncData()
            Uncertainty data object in which to store the sensitivity indices
        sensitivity_method : str
            sensitivity analysis method from SALib.analyse
            Possible choices:
                'fast', 'rbd_fact', 'morris', 'sobol', 'delta', 'ff'
            The default is 'sobol'.
            Note that in Salib, sampling methods and sensitivity analysis
            methods should be used in specific pairs.
            https://salib.readthedocs.io/en/latest/api.html
        sensitivity_kwargs: dict(), optional
            Keyword arguments of the chosen SALib analyse method.
            The default is to use SALib's default arguments.
        Returns
        -------
        sensitivity_dict : dict
            dictionary of the sensitivity indices. Keys are the
            metrics names, values the sensitivity indices dictionary
            as returned by SALib.

        """

        if sensitivity_kwargs is None:
            sensitivity_kwargs = {}

        #Check compatibility of sampling and sensitivity methods
        unc_data.check_salib(sensitivity_method)

        #Import the named submodule from the SALib analyse module
        #From the workings of __import__ the use of 'from_list' is necessary
        #c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        method = getattr(
            __import__('SALib.analyze',
                       fromlist=[sensitivity_method]
                       ),
            sensitivity_method
            )

        #Certaint Salib method required model input (X) and output (Y), others
        #need only ouput (Y)
        salib_kwargs = method.analyze.__code__.co_varnames #obtain all kwargs of the salib method
        X = unc_data.samples_df.to_numpy() if 'X' in salib_kwargs else None

        for metric_name in self.metric_names:
            sens_first_order_dict = {}
            sens_second_order_dict = {}
            for (submetric_name, metric_unc) in getattr(unc_data, metric_name + '_unc_df').iteritems():
                Y = metric_unc.to_numpy()
                if X is not None:
                    sens_indices = method.analyze(unc_data.problem_sa, X, Y,
                                                            **sensitivity_kwargs)
                else:
                    sens_indices = method.analyze(unc_data.problem_sa, Y,
                                                            **sensitivity_kwargs)
                sens_first_order = np.array([
                    np.array(si_val_array)
                    for si, si_val_array in sens_indices.items()
                    if (np.array(si_val_array).ndim == 1 and si!='names') #dirty trick due to Salib incoherent output
                    ]).ravel()
                sens_first_order_dict[submetric_name] = sens_first_order

                sens_second_order = np.array([
                    np.array(si_val_array)
                    for si_val_array in sens_indices.values()
                    if np.array(si_val_array).ndim == 2
                    ]).ravel()
                sens_second_order_dict[submetric_name] = sens_second_order

            n_params  = len(unc_data.param_labels)

            si_name_first_order_list = [
                key
                for key, array in sens_indices.items()
                if (np.array(array).ndim == 1 and key!='names') #dirty trick due to Salib incoherent output
                ]
            si_names_first_order = [si for si in si_name_first_order_list for _ in range(n_params)]
            param_names_first_order = unc_data.param_labels * len(si_name_first_order_list)

            si_name_second_order_list = [
                key
                for key, array in sens_indices.items()
                if np.array(array).ndim == 2
                ]
            si_names_second_order = [si for si in si_name_second_order_list for _ in range(n_params**2)]
            param_names_second_order_2 = unc_data.param_labels * len(si_name_second_order_list) * n_params
            param_names_second_order = [
                param
                for param in unc_data.param_labels
                for _ in range(n_params)
                ] * len(si_name_second_order_list)

            sens_first_order_df = pd.DataFrame(sens_first_order_dict, dtype=np.number)
            sens_first_order_df.insert(0, 'si', si_names_first_order)
            sens_first_order_df.insert(1, 'param', param_names_first_order)
            sens_first_order_df.insert(2, 'param2', None)


            sens_second_order_df = pd.DataFrame(sens_second_order_dict)
            sens_second_order_df.insert(0, 'si', si_names_second_order,)
            sens_second_order_df.insert(1, 'param', param_names_second_order)
            sens_second_order_df.insert(2, 'param2', param_names_second_order_2)

            sens_df = pd.concat([sens_first_order_df, sens_second_order_df]).reset_index(drop=True)

            setattr(unc_data, metric_name + '_sens_df', sens_df)
        sensitivity_kwargs = {
            key: str(val)
            for key, val in sensitivity_kwargs.items()}
        unc_data.sensitivity_method = sensitivity_method
        unc_data.sensitivity_kwargs = tuple(sensitivity_kwargs.items())