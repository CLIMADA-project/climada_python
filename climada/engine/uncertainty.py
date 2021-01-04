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

Define Uncertainty class.
"""

from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import logging
from pathos import pools

from climada.engine import Impact
from climada.entity import ImpactFuncSet
from climada.entity import Exposures
from climada.hazard import Hazard

LOGGER = logging.getLogger(__name__)


class UncVar():
    """
    Uncertainty variable 
    
    An uncertainty variable requires a single or multi-parameter function. 
    The parameters must follow a given distribution.
    
    Examples
    --------
    
    Categorical variable function: LitPop exposures with m,n exponents in [0,5]
        def unc_var_cat(m, n):
            exp = Litpop()
            exp.set_country('CHE', exponent=[m, n])
            return exp
        distr_dict = {
            m: sp.stats.randint(low=0, high=5),
            n: sp.stats.randint(low=0, high=5)
            }
        
    Continuous variable function: Impact function for TC
        def imp_fun_tc(G, v_half, vmin, k, _id=1):
            imp_fun = ImpactFunc()
            imp_fun.haz_type = 'TC'
            imp_fun.id = _id
            imp_fun.intensity_unit = 'm/s'
            imp_fun.intensity = np.linspace(0, 150, num=100)
            imp_fun.mdd = np.repeat(1, len(imp_fun.intensity))
            imp_fun.paa = np.array([sigmoid_function(v, G, v_half, vmin, k)
                                    for v in imp_fun.intensity])
            imp_fun.check()
            impf_set = ImpactFuncSet()
            impf_set.append(imp_fun)
            return impf_set
        distr_dict = {"G": sp.stats.uniform(0.8, 1),
              "v_half": sp.stats.uniform(50, 100),
              "vmin": sp.stats.norm(loc=15, scale=30),
              "k": sp.stats.randint(low=1, high=9)
              }
    
    """

    def __init__(self, unc_var, distr_dict):
        """
        Initialize UncVar
        
        Parameters
        ----------
        unc_var : function
            Variable defined as a function of the uncertainty parameters
        distr_dict : dict
            Dictionnary of the probability density distributions of the 
            uncertainty parameters, with keys the matching the keyword 
            arguments (i.e. uncertainty parameters) of the unc_var function.
            The distribution must be of type scipy.stats
            https://docs.scipy.org/doc/scipy/reference/stats.html
            
        Returns
        -------
        None.

        """
        self.labels = list(distr_dict.keys())
        self.distr_dict = distr_dict
        self.unc_var = unc_var

    def plot_distr(self):
        """
        Plot the distributions of the parameters of the uncertainty variable.

        Returns
        -------
        None.

        """
        raise NotImplementedError()
        pass

    def eval_unc_var(self, kwargs):
        """
        Evaluate the uncertainty variable.

        Parameters
        ----------
        kwargs : 
            These parameters will be passed to self.unc_var.
            They must be the input parameters of the uncertainty variable .

        Returns
        -------
        
            Evaluated uncertainty variable

        """
        return self.unc_var(**kwargs)


class UncSensitivity():
    """
    Sensitivity analysis class
    
    This class performs sensitivity analysis on the outputs of a 
    climada.engine.impact.Impact() or climada.engine.costbenefit.CostBenefit()
    object.
    
    """
    
    def __init__(self, exp_unc, impf_unc, haz_unc, pool=None):
        """Initialize UncSensitivite

        Parameters
        ----------
        exp_unc : climada.engine.uncertainty.UncVar or climada.entity.Exposure
            Exposure uncertainty variable or Exposure
        impf_unc : climada.engine.uncertainty.UncVar or climada.entity.ImpactFuncSet
            Impactfunction uncertainty variable or Impact function
        haz_unc : climada.engine.uncertainty.UncVar or climada.hazard.Hazard
            Hazard uncertainty variable or Hazard
        pool : pathos.pools.ProcessPool
            Pool of CPUs for parralel computations. Default is None.

        Returns
        -------
        None.

        """
        
        if pool:
            self.pool = pool
            LOGGER.info('Using %s CPUs.', self.pool.ncpus)
        else:
            self.pool = None

        if isinstance(exp_unc, Exposures):
            self.exp = UncVar(unc_var=lambda: exp_unc, distr_dict={})
        else:
            self.exp = exp_unc

        if isinstance(impf_unc, ImpactFuncSet):
            self.impf = UncVar(unc_var=lambda: impf_unc, distr_dict={})
        else:
            self.impf = impf_unc

        if isinstance(haz_unc, Hazard):
            self.haz = UncVar(unc_var=lambda: haz_unc, distr_dict={})
        else:
            self.haz = haz_unc

        self.params = None
        self.problem = None
        self.aai_freq = None
        self.eai_exp = None
        self.at_event = None
        
    @property
    def n_runs(self):
        if self.params:
            return len(self.params)
        else:
            return 0

    @property
    def param_labels(self):
        return self.exp.labels + self.haz.labels + self.impf.labels

    @property
    def distr_dict(self):
        distr_dict = dict(self.exp.distr_dict)
        distr_dict.update(self.haz.distr_dict)
        distr_dict.update(self.impf.distr_dict)
        return distr_dict

    def make_sobol_sample(self, N, calc_second_order):
        """
        Make a sobol sample for all parameters with their respective 
        distributions.

        Parameters
        ----------
        N : int
            Number of samples as defined in SALib.sample.saltelli.sample().
        calc_second_order : boolean
            if True, calculate second-order sensitivities.

        Returns
        -------
        None.

        """
        
        self.n_samples = N
        sobol_uniform_sample = self._make_uniform_sobol_sample(
            calc_second_order=calc_second_order)
        df_params = pd.DataFrame(sobol_uniform_sample, columns=self.param_labels)
        for param in list(df_params):
            df_params[param] = df_params[param].apply(
                self.distr_dict[param].ppf
                )
        self.params = df_params
        return None


    def _make_uniform_sobol_sample(self, calc_second_order):
        """
        Make a uniform sobol sample for the defined model uncertainty parameters 
        (self.param_labels)
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
        
        Parameters
        ----------
        calc_second_order : boolean
            if True, calculate second-order sensitivities.
            
        Returns
        -------
        sobol_params : np.matrix
            Returns a NumPy matrix containing the sampled uncertainty parameters using 
            Saltelli’s sampling scheme.

        """
        problem = {
            'num_vars' : len(self.param_labels),
            'names' : self.param_labels,
            'bounds' : [[0, 1]]*len(self.param_labels)
            }
        self.problem = problem
        sobol_params = saltelli.sample(problem, self.n_samples,
                                       calc_second_order=calc_second_order)
        return sobol_params


    def calc_impact_sobol_sensitivity(self,
                               N,
                               rp=None,
                               calc_eai_exp=False,
                               calc_at_event=False,
                               calc_second_order=True,
                               **kwargs):
        """
        Compute the sobol sensitivity indices using the SALib library:
        https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
        
        Simple introduction:
        https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis
        
        Parameters
        ----------
        N : int
            Number of samples as defined in SALib.sample.saltelli.sample()
        rp : list(int), optional
            Return period in years for which sensitivity indices are computed.
            The default is [5, 10, 20, 50, 100, 250.
        calc_eai_exp : boolean, optional
            Toggle computation of the sensitivity for the impact at each 
            centroid location. The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.
        calc_second_order : boolean, optional
            if True, calculate second-order sensitivities. The default is True.
        **kwargs : 
            These parameters will be passed to SALib.analyze.sobol.analyze()
            The default is num_resamples=100, conf_level=0.95,
            print_to_console=False, parallel=False, n_processors=None,
            seed=None.

        Returns
        -------
        sobol_analysis : dict
            Dictionnary with keys the uncertainty parameter labels. 
            For each uncertainty parameter, the item is another dictionnary
            with keys the sobol sensitivity indices ‘S1’, ‘S1_conf’, ‘ST’, 
            and ‘ST_conf’. If calc_second_order is True, the dictionary also
            contains keys ‘S2’ and ‘S2_conf’.

        """
        
        if calc_eai_exp:
            raise NotImplementedError()
            
        if calc_at_event:
            raise NotImplementedError()

        if rp is None:
            rp =[5, 10, 20, 50, 100, 250]
            
        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event

        if self.params is None:
            self.make_sobol_sample(N, calc_second_order=calc_second_order)
        if self.aai_freq is None:
            self.calc_impact_distribution(rp=rp,
                                          calc_eai_exp=calc_eai_exp,
                                          calc_at_event=calc_at_event
                                          )

        sobol_analysis = {}
        for imp_out in self.aai_freq:
            Y = self.aai_freq[imp_out].to_numpy()
            si_sobol = sobol.analyze(self.problem, Y, **kwargs)
            sobol_analysis.update({imp_out: si_sobol})
            for si_list in si_sobol.values():
                if np.any(np.array(si_list)<0):
                    LOGGER.warning("There is at least one negative sobol " +
                        "index. Consider using more samples or using another "+
                        "sensitivity analysis method." +
                        "See https://github.com/SALib/SALib/issues/109")
                    continue
            
        self.sensitivity = sobol_analysis
        
        return sobol_analysis
    
    
    def calc_cost_benefit_sobol_sensitivity():
        pass


    def calc_impact_distribution(self,
                                 rp=None,
                                 calc_eai_exp=False,
                                 calc_at_event=False
                                 ):
        """
        Computes the impact for each of the parameters set defined in
        uncertainty.params. 
        
        By default, the aggregated average annual impact 
        (impact.aai_agg) and the excees impact at return periods (rp) is
        computed and stored in self.aai_freq. Optionally, the impact at 
        each centroid location is computed (this may require a larger
        amount of memory if the number of centroids is large).

        Parameters
        ----------
        rp : list(int), optional
            Return period in years to be computed. 
            The default is [5, 10, 20, 50, 100, 250].
        calc_eai_exp : boolean, optional
            Toggle computation of the impact at each centroid location.
            The default is False.
        calc_at_event : boolean, optional
            Toggle computation of the impact for each event.
            The default is False.

        Returns
        -------
        None.

        """
        

        if rp is None:
            rp=[5, 10, 20, 50, 100, 250]

        aai_agg_list = []
        freq_curve_list = []
        if calc_eai_exp:
            eai_exp_list = []
        if calc_at_event:
            at_event_list = []
            
        self.rp = rp
        self.calc_eai_exp = calc_eai_exp
        self.calc_at_event = calc_at_event
            
        if self.pool:
            chunksize = min(self.n_runs // self.pool.ncpus, 10)
            impact_metrics = self.pool.map(self._map_impact_eval,
                                           self.params.iterrows(),
                                           chunsize = chunksize)
            
            [aai_agg_list,
             freq_curve_list,
             eai_exp_list,
             at_event_list] = list(zip(*impact_metrics))
            
        else:
            
            impact_metrics = map(self._map_impact_eval, self.params.iterrows())
            
            [aai_agg_list,
             freq_curve_list,
             eai_exp_list,
             at_event_list] = list(zip(impact_metrics))

        df_aai_freq = pd.DataFrame(freq_curve_list,
                                   columns=['rp' + str(n) for n in rp])
        df_aai_freq['aai_agg'] = aai_agg_list
        self.aai_freq = df_aai_freq

        if calc_eai_exp:
            df_eai_exp = pd.DataFrame(eai_exp_list)
            self.eai_exp = df_eai_exp
            
        if calc_at_event:
            df_at_event = pd.DataFrame(at_event_list)
            self.at_event = df_at_event

        return None
    

    def _map_impact_eval(self, param_row):
        
        # [1] only the rows of the dataframe passed by pd.DataFrame.iterrows()
        exp_params = param_row[1][self.exp.labels].to_dict()
        haz_params = param_row[1][self.haz.labels].to_dict()
        impf_params = param_row[1][self.impf.labels].to_dict()

        exp = self.exp.eval_unc_var(exp_params)
        haz = self.haz.eval_unc_var(haz_params)
        impf = self.impf.eval_unc_var(impf_params)

        imp = Impact()
        imp.calc(exposures=exp, impact_funcs=impf, hazard=haz)
        
        rp_curve = imp.calc_freq_curve(self.rp).impact

        if self.calc_eai_exp:
            eai_exp = imp.eai_exp
        else:
            eai_exp = None
            
        if self.calc_at_event:
            at_event= imp.at_event
        else:
            at_event = None
            
        return [imp.aai_agg, rp_curve, eai_exp, at_event]


class UncRobustness():
    """
    Compute variance from multiplicative Gaussian noise
    """

    pass



# class UncVarCat(UncVar):

#     def __init__(self, label, cat_var):
#         self.labels = [label]
#         self.unc_var = cat_var

#     @property
#     def distrs(self):
#         [label] = self.labels
#         return {
#         label : sp.stats.randint(low=0, high=len(self.cat_var))
#             }

#     def get_unc_var(self, idx):
#         return self.cat_var[idx]
#     pass


# class UncVarCont(UncVar):

#     def __init__(self, labels, cont_var, bound_dict=None, distr_dict=None):
#         self.unc_var = cont_var
#         self.labels = list(distr_dict.keys())

#         if bound_dict is not None:
#             if distr_dict() is None:
#                 self.distrs = {
#                     label: sp.stats.uniform(
#                         bound_dict[label][0],
#                         bound_dict[label][1]-bound_dict[label][0]
#                         )
#                     for label in labels
#                     }

#             else:
#                 raise ValueError("Specify either bounds or distr_dict, not both")

#         if distr_dict() is not None:
#             if bound_dict is None:
#                 self.distrs = distr_dict
#             else:
#                 raise ValueError("Specify either bounds or distr_dict, not both")


#     def get_unc_var(self, *params):
#         return self.unc_var(params)

#     pass


# class UncVarCatCont(UncVar):
#     pass



# impacts = []
# dt_aai_freq = pd.DataFrame(columns=['aai_agg', 'freq_curve'])
# dt_imp_map = pd.DataFrame(columns=['imp_map'])
# print('----------  %s  started -------------' %haz_name)

# for cnt, [h, x, G, v_half, vmin, k, m, n] in enumerate(param_values):

#     #Set exposures, hazard, impact functions with parameters

#     imp_fun_set = ImpactFuncSet()
#     imp_fun_set.append(imp_fun_tc(G=G, v_half=v_half, vmin=vmin, k=k))
#     exp = exp_dict[(int(m), int(n))].copy(deep=True)
#     exp.value *= x
#     haz.intensity.multiply(h)
#     exp.assign_centroids(hazard=haz)

#     #Compute and save impact
#     imp = Impact()
#     imp.calc(exp, imp_fun_set, haz, save_mat=False)
#     impacts.append(imp)

#     if cnt % 10 == 0:
#         percent_done = cnt/len(param_values)*100
#         print("\n\n\n %.2f done \n\n\n" %percent_done )

# n_sig_dig = 16
# rp = [1, 5, 10, 50, 100, 250]
# dt_aai_freq.aai_agg = np.array([sig_dig(imp.aai_agg, n_sig_dig = n_sig_dig) for imp in impacts])
# dt_aai_freq.freq_curve = [sig_dig_list(imp.calc_freq_curve(rp).impact, n_sig_dig = n_sig_dig) for imp in impacts]
# dt_imp_map.imp_map = [sig_dig_list(imp.eai_exp, n_sig_dig = n_sig_dig) for imp in impacts]
# dt_imp_map = dt_imp_map['imp_map'].apply(pd.Series)

# filename= haz_name + "_aai_freq_1"
# output_dir = os.path.join(DATA_DIR, expe, 'sampling')
# abs_path = os.path.join(output_dir, filename)
# dt_aai_freq.to_csv(abs_path + '.csv', mode='x')

# filename= haz_name + "_imp_map_1"
# output_dir = os.path.join(DATA_DIR, expe, 'sampling')
# abs_path = os.path.join(output_dir, filename)
# dt_imp_map.to_csv(abs_path + '.csv', mode='x')

