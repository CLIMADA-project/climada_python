#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:17:00 2020

@author: ckropf
"""

from SALib.sample import saltelli
import scipy as sp
import pandas as pd
from climada.engine import Impact
from climada.entity import ImpactFuncSet
from climada.entity import Exposures
from climada.hazard import Hazard


class UncVar():
    
    def __init__(self, labels, var, distr_dict):
        self.labels = labels
        self.distrs = {label: distr for label, distr in zip(labels, distr_dict)}
    
    def plot_distr(self):
        pass    
    
    pass


class UncVarCat(UncVar):
    
    def __init__(self, label, cat_var):
        self.labels = [label]
        self.unc_var = cat_var
    
    @property
    def distrs(self):
        [label] = self.labels
        return {
        label : sp.stats.randint(low=0, high=len(self.cat_var))
            }
    
    def get_unc_var(self, idx):
        return self.cat_var[idx]
    pass
   
 
class UncVarCont(UncVar):
    
    def __init__(self, labels, cont_var, bound_dict=None, distr_dict=None):
        self.unc_var = cont_var
        self.labels = list(distr_dict.keys())
        
        if bound_dict is not None:
            if distr_dict() is None: 
                self.distrs = {
                    label: sp.stats.uniform(
                        bound_dict[label][0],
                        bound_dict[label][1]-bound_dict[label][0]
                        )
                    for label in labels
                    }
                
            else:
                raise ValueError("Specify either bounds or distr_dict, not both")
        
        if distr_dict() is not None:
            if bound_dict is None:
                self.distrs = distr_dict
            else:
                raise ValueError("Specify either bounds or distr_dict, not both")
                
            
    def get_unc_var(self, *params):
        return self.unc_var(params)
    
    pass
    

class UncVarCatCont(UncVar):
    pass


class UncSensitivity():
    
    def __init__(self, exp_unc, impf_unc, haz_unc):
        
        if isinstance(exp_unc, Exposures):
            self.exp = UncVar(labels=[], unc_var=exp_unc, distr_dict={})
        else:
            self.exp = exp_unc
            
        if isinstance(impf_unc, ImpactFuncSet):
            self.impf = UncVar(labels=[], unc_var=impf_unc, distr_dict={})
        else:
            self.impf = impf_unc
            
        if isinstance(haz_unc, Hazard):
            self.haz = UncVar(labels=[], unc_var=haz_unc, distr_dict={})
        else:
            self.haz = haz_unc
        
    def calc(self, N, save_impacts = False):
        self.n_samples = N
        df_params = self._make_sobol_sample()
        
        for j, row in df_params.iterrows():
            imp = Impact
            exp_params = row[self.exp.labels].to_list()
            haz_params = row[self.haz.labels].to_list()
            impf_params = row[self.impf.labels].to_list()
            
            exp = self.exp.get_unc_var(*exp_params)
            haz = self.haz.get_unc_var(*haz_params)
            impf = self.impf.get_unc_var(*impf_params)
            
            impact = imp.calc(exp, impf, haz)
            
            print(impact.aai_agg)
            

    @property
    def labels(self):
        return self.exp.labels + self.haz.labels + self.impf.labels
    
    @property
    def distrs(self):
        distrs = dict(self.exp.distrs)
        distrs.update(self.haz.distrs)
        distrs.update(self.impf.distrs)
        return distrs
    
    def _make_sobol_sample(self):
        sobol_uniform_sample = self._make_uniform_sobol_sample()
        df_params = pd.DataFrame(sobol_uniform_sample, columns=self.labels)
        for param in list(df_params):
            df_params[param] = df_params[param].apply(
                self.distrs[param].ppf
                )
        return df_params
    
    
    def _make_uniform_sobol_sample(self):
        problem = {
            'num_vars' : len(self.params),
            'names' : self.params,
            'bounds' : [[0, 1]]*len(self.params)
            }
        params = saltelli.sample(problem, self.n_samples)
        return params
    
    
    
class UncRobustness():
    """
    Compute variance from multiplicative Gaussian noise
    """
    pass
    
    
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

