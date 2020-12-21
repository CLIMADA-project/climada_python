#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:17:00 2020

@author: ckropf
"""

from SALib.sample import saltelli
import pandas as pd
from climada.engine import Impact
from climada.entity import ImpactFuncSet
from climada.entity import Exposures
from climada.hazard import Hazard


class UncVar():
    
    def __init__(self, unc_var, distr_dict):
        self.labels = list(distr_dict.keys())
        self.distr_dict = distr_dict
        self.unc_var = unc_var
    
    def plot_distr(self):
        pass    
    
    def get_unc_var(self, kwargs):
        return self.unc_var(**kwargs) 


class UncSensitivity():
    
    def __init__(self, exp_unc, impf_unc, haz_unc):
        
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
            
        
    def calc_impact_distribution_sobol(self, N, save_impacts = False):
        
        self.n_samples = N
        df_params = self._make_sobol_sample()
        
        impacts = []
        for j, row in df_params.iterrows():
            
            exp_params = row[self.exp.labels].to_dict()
            haz_params = row[self.haz.labels].to_dict()
            impf_params = row[self.impf.labels].to_dict()
            
            exp = self.exp.get_unc_var(exp_params)
            haz = self.haz.get_unc_var(haz_params)
            impf = self.impf.get_unc_var(impf_params)
            
            imp = Impact()
            imp.calc(exposures=exp, impact_funcs=impf, hazard=haz)
            
            impacts.append(imp)
            
        return df_params, impacts
        
        

    @property
    def params(self):
        return self.exp.labels + self.haz.labels + self.impf.labels
    
    @property
    def distr_dict(self):
        distr_dict = dict(self.exp.distr_dict)
        distr_dict.update(self.haz.distr_dict)
        distr_dict.update(self.impf.distr_dict)
        return distr_dict

    
    def _make_sobol_sample(self):
        sobol_uniform_sample = self._make_uniform_sobol_sample()
        df_params = pd.DataFrame(sobol_uniform_sample, columns=self.params)
        for param in list(df_params):
            df_params[param] = df_params[param].apply(
                self.distr_dict[param].ppf
                )
        return df_params
    
    
    def _make_uniform_sobol_sample(self):
        problem = {
            'num_vars' : len(self.params),
            'names' : self.params,
            'bounds' : [[0, 1]]*len(self.params)
            }
        self.problem = problem
        sobol_params = saltelli.sample(problem, self.n_samples)
        return sobol_params
    
    
    
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

