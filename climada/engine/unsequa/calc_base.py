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

Define Calc (uncertainty calculate) class.
"""

import copy
import datetime as dt
import itertools
import logging

import numpy as np
import pandas as pd

from climada.engine.unsequa.unc_output import UncOutput
from climada.util.value_representation import sig_dig as u_sig_dig

LOGGER = logging.getLogger(__name__)


class Calc:
    """
    Base class for uncertainty quantification

    Contains the generic sampling and sensitivity methods. For computing
    the uncertainty distribution for specific CLIMADA outputs see
    the subclass CalcImpact and CalcCostBenefit.

    Attributes
    ----------
    _input_var_names : tuple(str)
        Names of the required uncertainty variables.
    _metric_names : tuple(str)
        Names of the output metrics.

    Notes
    -----
    Parallelization logics: for computation of the uncertainty users may
    specify a number N of processes on which to perform the computations in
    parallel. Since the computation for each individual sample of the
    input parameters is independent of one another, we implemented a simple
    distribution on the processes.

    1. The samples are divided in N equal sub-sample chunks
    2. Each chunk of samples is sent as one to a node for processing

    Hence, this is equivalent to the user running the computation N times,
    once for each sub-sample.
    Note that for each process, all the input variables must be copied once,
    and hence each parallel process requires roughly the same amount of memory
    as if a single process would be used.

    This approach differs from the usual parallelization strategy (where individual
    samples are distributed), because each sample requires the entire input data.
    With this method, copying data between processes is reduced to a minimum.

    Parallelization is currently not available for the sensitivity computation,
    as this requires all samples simoultenaously in the current implementation
    of the SaLib library.
    """

    _input_var_names = ()
    """Names of the required uncertainty variables"""

    _metric_names = ()
    """Names of the output metrics"""

    def __init__(self):
        """
        Empty constructor to be overwritten by subclasses
        """

    def check_distr(self):
        """
        Log warning if input parameters repeated among input variables

        Returns
        -------
        True.

        """

        distr_dict = dict()
        for input_var in self.input_vars:
            for input_param_name, input_param_func in input_var.distr_dict.items():
                if input_param_name in distr_dict:
                    func = distr_dict[input_param_name]
                    x = np.linspace(func.ppf(0.01), func.ppf(0.99), 100)
                    if not np.all(func.cdf(x) == input_param_func.cdf(x)):
                        raise ValueError(
                            f"The input parameter {input_param_name}"
                            " is shared among two input variables with"
                            " different distributions."
                        )
                    LOGGER.warning(
                        "\n\nThe input parameter %s is shared "
                        "among at least 2 input variables. Their uncertainty is "
                        "thus computed with the same samples for this "
                        "input paramter.\n\n",
                        input_param_name,
                    )
                distr_dict[input_param_name] = input_param_func
        return True

    @property
    def input_vars(self):
        """
        Uncertainty variables

        Returns
        -------
        tuple(UncVar)
            All uncertainty variables associated with the calculation

        """
        return tuple(getattr(self, var) for var in self._input_var_names)

    @property
    def distr_dict(self):
        """
        Dictionary of the input variable distribution

        Probabilitiy density distribution of all the parameters of all the
        uncertainty variables listed in self.InputVars

        Returns
        -------
        distr_dict : dict( sp.stats objects )
            Dictionary of all probability density distributions.

        """

        distr_dict = dict()
        for input_var in self.input_vars:
            distr_dict.update(input_var.distr_dict)
        return distr_dict

    def est_comp_time(self, n_samples, time_one_run, processes=None):
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
            LOGGER.warning(
                "Computation time for one set of parameters is "
                "%.2fs. This is rather long."
                "Potential reasons: InputVars are loading data, centroids have "
                "been assigned to exp before defining input_var, ..."
                "\n If computation cannot be reduced, consider using"
                " a surrogate model https://www.uqlab.com/",
                time_one_run,
            )
        total_time = n_samples * time_one_run / processes
        LOGGER.info(
            "\n\nEstimated computaion time: %s\n", dt.timedelta(seconds=total_time)
        )

        return total_time

    def make_sample(self, N, sampling_method="saltelli", sampling_kwargs=None):
        """
        Make samples of the input variables

        For all input parameters, sample from their respective
        distributions using the chosen sampling_method from SALib.
        https://salib.readthedocs.io/en/latest/api.html

        This sets the attributes
        unc_output.samples_df,
        unc_output.sampling_method,
        unc_output.sampling_kwargs.

        Parameters
        ----------
        N : int
            Number of samples as used in the sampling method from SALib
        sampling_method : str, optional
            The sampling method as defined in SALib. Possible choices:
            'saltelli', 'latin', 'morris', 'dgsm', 'fast_sampler', 'ff', 'finite_diff',
            https://salib.readthedocs.io/en/latest/api.html
            The default is 'saltelli'.
        sampling_kwargs : kwargs, optional
            Optional keyword arguments passed on to the SALib sampling_method.
            The default is None.

        Returns
        -------
        unc_output : climada.engine.uncertainty.unc_output.UncOutput()
            Uncertainty data object with the samples

        Notes
        -----
        The 'ff' sampling method does not require a value for the N parameter.
        The inputed N value is hence ignored in the sampling process in the case
        of this method.
        The 'ff' sampling method requires a number of uncerainty parameters to be
        a power of 2. The users can generate dummy variables to achieve this
        requirement. Please refer to https://salib.readthedocs.io/en/latest/api.html
        for more details.

        See Also
        --------
        SALib.sample: sampling methods from SALib SALib.sample
            https://salib.readthedocs.io/en/latest/api.html

        """

        if sampling_kwargs is None:
            sampling_kwargs = {}

        param_labels = list(self.distr_dict.keys())
        problem_sa = {
            "num_vars": len(param_labels),
            "names": param_labels,
            "bounds": [[0, 1]] * len(param_labels),
        }
        # for the ff sampler, no value of N is needed. For API consistency the user
        # must input a value that is ignored and a warning is given.
        if sampling_method == "ff":
            LOGGER.warning(
                "You are using the 'ff' sampler which does not require "
                "a value for N. The entered N value will be ignored"
                "in the sampling process."
            )
        uniform_base_sample = self._make_uniform_base_sample(
            N, problem_sa, sampling_method, sampling_kwargs
        )
        df_samples = pd.DataFrame(uniform_base_sample, columns=param_labels)

        for param in list(df_samples):
            df_samples[param] = df_samples[param].apply(self.distr_dict[param].ppf)

        sampling_kwargs = {key: str(val) for key, val in sampling_kwargs.items()}
        df_samples.attrs["sampling_method"] = sampling_method
        df_samples.attrs["sampling_kwargs"] = tuple(sampling_kwargs.items())

        unc_output = UncOutput(df_samples)
        LOGGER.info("Effective number of made samples: %d", unc_output.n_samples)
        return unc_output

    def _make_uniform_base_sample(
        self, N, problem_sa, sampling_method, sampling_kwargs
    ):
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
            'saltelli', 'latin', 'morris', 'dgsm', 'fast_sampler', 'ff', 'finite_diff',
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

        # Import the named submodule from the SALib sample module
        # From the workings of __import__ the use of 'from_list' is necessary
        # c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        import importlib  # pylint: disable=import-outside-toplevel

        salib_sampling_method = importlib.import_module(
            f"SALib.sample.{sampling_method}"
        )

        if sampling_method == "ff":  # the ff sampling has a fixed sample size and
            # does not require the N parameter
            if problem_sa["num_vars"] & (problem_sa["num_vars"] - 1) != 0:
                raise ValueError(
                    "The number of parameters must be a power of 2. "
                    "To use the ff sampling method, you can generate "
                    "dummy parameters to overcome this limitation."
                    " See https://salib.readthedocs.io/en/latest/api.html"
                )

            sample_uniform = salib_sampling_method.sample(
                problem=problem_sa, **sampling_kwargs
            )
        else:
            sample_uniform = salib_sampling_method.sample(
                problem=problem_sa, N=N, **sampling_kwargs
            )
        return sample_uniform

    def sensitivity(
        self, unc_output, sensitivity_method="sobol", sensitivity_kwargs=None
    ):
        """
        Compute the sensitivity indices using SALib.

        Prior to doing the sensitivity analysis, one must compute the
        uncertainty (distribution) of the output values
        (with self.uncertainty()) for all the samples
        (rows of self.samples_df).

        According to Wikipedia, sensitivity analysis is “the study of how the
        uncertainty in the output of a mathematical model or system (numerical
        or otherwise) can be apportioned to different sources of uncertainty
        in its inputs.” The sensitivity of each input is often represented by
        a numeric value, called the sensitivity index. Sensitivity indices
        come in several forms.

        This sets the attributes:
        sens_output.sensistivity_method
        sens_output.sensitivity_kwargs
        sens_output.xxx_sens_df for each metric unc_output.xxx_unc_df

        Parameters
        ----------
        unc_output : climada.engine.unsequa.UncOutput
            Uncertainty data object in which to store the sensitivity indices
        sensitivity_method : str, optional
            Sensitivity analysis method from SALib.analyse. Possible choices: 'sobol', 'fast',
            'rbd_fast', 'morris', 'dgsm', 'ff', 'pawn', 'rhdm', 'rsa', 'discrepancy', 'hdmr'.
            Note that in Salib, sampling methods and sensitivity
            analysis methods should be used in specific pairs:
            https://salib.readthedocs.io/en/latest/api.html
        sensitivity_kwargs: dict, optional
            Keyword arguments of the chosen SALib analyse method.
            The default is to use SALib's default arguments.

        Notes
        -----
        The variables 'Em','Term','X','Y' are removed from the output of the
        'hdmr' method to ensure compatibility with unsequa.
        The 'Delta' method is currently not supported.

        Returns
        -------
        sens_output : climada.engine.unsequa.UncOutput
            Uncertainty data object with all the sensitivity indices,
            and all the uncertainty data copied over from unc_output.

        """

        if sensitivity_kwargs is None:
            sensitivity_kwargs = {}

        # Check compatibility of sampling and sensitivity methods
        unc_output.check_salib(sensitivity_method)

        # Import the named submodule from the SALib analyse module
        # From the workings of __import__ the use of 'from_list' is necessary
        # c.f. https://stackoverflow.com/questions/2724260/why-does-pythons-import-require-fromlist
        method = getattr(
            __import__("SALib.analyze", fromlist=[sensitivity_method]),
            sensitivity_method,
        )

        sens_output = copy.deepcopy(unc_output)

        # Certain Salib method required model input (X) and output (Y), others
        # need only ouput (Y)
        salib_kwargs = (
            method.analyze.__code__.co_varnames
        )  # obtain all kwargs of the salib method
        X = unc_output.samples_df.to_numpy() if "X" in salib_kwargs else None

        for metric_name in self._metric_names:
            unc_df = unc_output.get_unc_df(metric_name)
            sens_df = _calc_sens_df(
                method,
                unc_output.problem_sa,
                sensitivity_kwargs,
                unc_output.param_labels,
                X,
                unc_df,
            )
            sens_output.set_sens_df(metric_name, sens_df)
        sensitivity_kwargs = {key: str(val) for key, val in sensitivity_kwargs.items()}
        sens_output.sensitivity_method = sensitivity_method
        sens_output.sensitivity_kwargs = tuple(sensitivity_kwargs.items())

        return sens_output


def _multiprocess_chunksize(samples_df, processes):
    """Divides the samples into chunks for multiprocesses computing

    The goal is to send to each processing node an equal number
    of samples to process. This make the parallel processing anologous
    to running the uncertainty assessment independently on each nodes
    for a subset of the samples, instead of distributing individual samples
    on the nodes dynamically. Hence, all the heavy input variables
    are copied/sent once to each node only.

    Parameters
    ----------
    samples_df : pd.DataFrame
        samples dataframe
    processes : int
        number of processes

    Returns
    -------
    int
        the number of samples in each chunk
    """
    return np.ceil(samples_df.shape[0] / processes).astype(int)


def _transpose_chunked_data(metrics):
    """Transposes the output metrics lists from one list per
    chunk of samples to one list per output metric

    [ [x1, [y1, z1]], [x2, [y2, z2]] ] ->
    [ [x1, x2], [[y1, z1], [y2, z2]] ]

    Parameters
    ----------
    metrics : list
        list of list as returned by the uncertainty mapings

    Returns
    -------
    list
        list of climada output uncertainty

    See Also
    --------
    calc_impact._map_impact_calc
        map for impact uncertainty
    calc_cost_benefits._map_costben_calc
        map for cost benefit uncertainty
    """
    return [list(itertools.chain.from_iterable(x)) for x in zip(*metrics)]


def _sample_parallel_iterator(samples, chunksize, **kwargs):
    """
    Make iterator over chunks of samples
    with repeated kwargs for each chunk.

    Parameters
    ----------
    samples : pd.DataFrame
        Dataframe of samples
    **kwargs : arguments to repeat
        Arguments to repeat for parallel computations

    Returns
    -------
    iterator
        suitable for methods _map_impact_calc and _map_costben_calc

    """

    def _chunker(df, size):
        """
        Divide the dataframe into chunks of size number of lines
        """
        for pos in range(0, len(df), size):
            yield df.iloc[pos : pos + size]

    return zip(
        _chunker(samples, chunksize),
        *(itertools.repeat(item) for item in kwargs.values()),
    )


def _calc_sens_df(method, problem_sa, sensitivity_kwargs, param_labels, X, unc_df):
    """Compute the sensitifity indices

    Parameters
    ----------
    method : str
        SALib sensitivity method name
    problem_sa :dict
        dictionnary for sensitivty method for SALib
    sensitivity_kwargs : kwargs
        passed on to SALib.method.analyse
    param_labels : list(str)
        list of name of uncertainty input parameters
    X : numpy.ndarray
        array of input parameter samples
    unc_df : DataFrame
        Dataframe containing the uncertainty values

    Returns
    -------
    DataFrame
        Values of the sensitivity indices
    """
    sens_first_order_dict = {}
    sens_second_order_dict = {}
    for submetric_name, metric_unc in unc_df.items():
        Y = metric_unc.to_numpy()
        if X is not None:
            sens_indices = method.analyze(problem_sa, X, Y, **sensitivity_kwargs)
        else:
            sens_indices = method.analyze(problem_sa, Y, **sensitivity_kwargs)
        # refactor incoherent SALib output
        nparams = len(param_labels)
        if method.__name__[-3:] == ".ff":  # ff method
            if sensitivity_kwargs["second_order"]:
                # parse interaction terms of sens_indices to a square matrix
                # to ensure consistency with unsequa
                interaction_names = sens_indices.pop("interaction_names")
                interactions = np.full((nparams, nparams), np.nan)
                # loop over interaction names and extract each param pair,
                # then match to the corresponding param from param_labels
                for i, interaction_name in enumerate(interaction_names):
                    interactions[
                        param_labels.index(interaction_name[0]),
                        param_labels.index(interaction_name[1]),
                    ] = sens_indices["IE"][i]
                sens_indices["IE"] = interactions

        if method.__name__[-5:] == ".hdmr":  # hdmr method
            # first, remove variables that are incompatible with unsequa output
            keys_to_remove = ["Em", "Term", "select", "RT", "Y_em", "idx", "X", "Y"]
            sens_indices = {
                k: v for k, v in sens_indices.items() if k not in keys_to_remove
            }
            names = sens_indices.pop("names")  # names of terms

            # second, refactor to 2D
            for si, si_val_array in sens_indices.items():
                if (
                    np.array(si_val_array).ndim
                    == 1  # for everything that is 1d and has
                    and np.array(si_val_array).size > nparams
                ):  # lentgh > n params, refactor to 2D
                    si_new_array = np.full((nparams, nparams), np.nan)
                    np.fill_diagonal(
                        si_new_array, si_val_array[0:nparams]
                    )  # simple terms go on diag
                    for i, interaction_name in enumerate(names[nparams:]):
                        t1, t2 = interaction_name.split("/")  # interaction terms
                        si_new_array[param_labels.index(t1), param_labels.index(t2)] = (
                            si_val_array[nparams + i]
                        )
                    sens_indices[si] = si_new_array

        sens_first_order = np.array(
            [
                np.array(si_val_array)
                for si, si_val_array in sens_indices.items()
                if (
                    np.array(si_val_array).ndim
                    == 1  # dirty trick due to Salib incoherent output
                    and si != "names"
                    and np.array(si_val_array).size == len(param_labels)
                )
            ]
        ).ravel()
        sens_first_order_dict[submetric_name] = sens_first_order

        sens_second_order = np.array(
            [
                np.array(si_val_array)
                for si_val_array in sens_indices.values()
                if np.array(si_val_array).ndim == 2
            ]
        ).ravel()
        sens_second_order_dict[submetric_name] = sens_second_order

    sens_first_order_df = pd.DataFrame(sens_first_order_dict, dtype=np.number)

    if not sens_first_order_df.empty:
        si_names_first_order, param_names_first_order = _si_param_first(
            param_labels, sens_indices
        )
        sens_first_order_df.insert(0, "si", si_names_first_order)
        sens_first_order_df.insert(1, "param", param_names_first_order)
        sens_first_order_df.insert(2, "param2", None)

    sens_second_order_df = pd.DataFrame(sens_second_order_dict)
    if not sens_second_order_df.empty:
        si_names_second_order, param_names_second_order, param_names_second_order_2 = (
            _si_param_second(param_labels, sens_indices)
        )
        sens_second_order_df.insert(
            0,
            "si",
            si_names_second_order,
        )
        sens_second_order_df.insert(1, "param", param_names_second_order)
        sens_second_order_df.insert(2, "param2", param_names_second_order_2)

    sens_df = pd.concat([sens_first_order_df, sens_second_order_df]).reset_index(
        drop=True
    )

    return sens_df


def _si_param_first(param_labels, sens_indices):
    """Extract the first order sensivity indices from SALib ouput

    Parameters
    ----------
    param_labels : list(str)
        name of the unceratinty input parameters
    sens_indices : dict
       sensitivity indidices dictionnary as produced by SALib

    Returns
    -------
    si_names_first_order, param_names_first_order: list, list
        Names of the sensivity indices of first order for all input parameters
        and Parameter names for each sentivity index
    """
    n_params = len(param_labels)

    si_name_first_order_list = [
        key
        for key, array in sens_indices.items()
        if (
            np.array(array).ndim == 1 and key != "names"
        )  # dirty trick due to Salib incoherent output
    ]
    si_names_first_order = [
        si for si in si_name_first_order_list for _ in range(n_params)
    ]
    param_names_first_order = param_labels * len(si_name_first_order_list)
    return si_names_first_order, param_names_first_order


def _si_param_second(param_labels, sens_indices):
    """Extract second order sensitivity indices

    Parameters
    ----------
    param_labels : list(str)
        name of the unceratinty input parameters
    sens_indices : dict
       sensitivity indidices dictionnary as produced by SALib

    Returns
    -------
    si_names_second_order, param_names_second_order, param_names_second_order_2: list, list, list
        Names of the sensivity indices of second order for all input parameters
        and Pairs of parameter names for each 2nd order sentivity index
    """
    n_params = len(param_labels)
    si_name_second_order_list = [
        key for key, array in sens_indices.items() if np.array(array).ndim == 2
    ]
    si_names_second_order = [
        si for si in si_name_second_order_list for _ in range(n_params**2)
    ]
    param_names_second_order_2 = (
        param_labels * len(si_name_second_order_list) * n_params
    )
    param_names_second_order = [
        param for param in param_labels for _ in range(n_params)
    ] * len(si_name_second_order_list)
    return si_names_second_order, param_names_second_order, param_names_second_order_2
