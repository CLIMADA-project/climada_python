#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import argparse
from climada.entity.exposures.litpop import LitPop
from climada.entity.impact_funcs.flood import IFRiverFlood,flood_imp_func_set, assign_if_simple
from climada.hazard.flood import RiverFlood
from climada.hazard.centroids import Centroids
from climada.entity import ImpactFuncSet
from climada.util.constants import NAT_REG_ID
import copy

from climada.engine import Impact

parser = argparse.ArgumentParser(
    description='run climada for different climate and runoff models')
parser.add_argument(
    '--RF_model', type=str, default='H08',
    help='runoff model')
parser.add_argument(
    '--CL_model', type=str, default='princeton',
    help='Climate model')
args = parser.parse_args()

#Todo for cluster application
# set cluster true
# set output path
# set all countries
# set output dir


PROT_STD = ['flopros']
#for LPJ longrun

#flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
#flood_dir = '/p/projects/ebm/data/hazard/floods/benoit_input_data/'
gdp_path = '/p/projects/ebm/data/exposure/gdp/processed_data/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'
RF_PATH_FRC = '/p/projects/ebm/tobias_backup/floods/climada/isimip2a/flood_maps/fldfrc24_2.nc'
output = currentdir
#For lpj longrun
if args.RF_model == 'lpjml':
    flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
    if args.CL_model == 'watch':
        years = np.arange(1901, 2002)
    else:
        years = np.arange(1901, 2011)
else:
    flood_dir = '/p/projects/ebm/data/hazard/floods/benoit_input_data/'
    if args.CL_model == 'watch':
        years = np.arange(1980, 2002)
    else:
        years = np.arange(1980, 2011)

#years = np.arange(1971, 2011)
country_info = pd.read_csv(NAT_REG_ID)
isos = country_info['ISO'].tolist()
regs = country_info['Reg_name'].tolist()
conts = country_info['if_RF'].tolist()
l = len(years) * len(isos)
continent_names = ['Africa', 'Asia', 'Europe', 'NorthAmerica', 'Oceania', 'SouthAmerica']

dataDF = pd.DataFrame(data={'Year': np.full(l, np.nan, dtype=int),
                            'Country': np.full(l, "", dtype=str),
                            'Region': np.full(l, "", dtype=str),
                            'Continent': np.full(l, "", dtype=str),
                            'TotalAssetValue': np.full(l, np.nan, dtype=float),
                            'TotalAssetValue2005': np.full(l, np.nan, dtype=float),
                            'FloodDepthMean': np.full(l, np.nan, dtype=float),
                            'FloodedArea': np.full(l, np.nan, dtype=float),
                            'ImpFixExp': np.full(l, np.nan, dtype=float),
                            'Impact': np.full(l, np.nan, dtype=float),
                            'Impact_2y': np.full(l, np.nan, dtype=float),
                            'ImpFix_2y': np.full(l, np.nan, dtype=float),
                            'FloodDepthMean2y': np.full(l, np.nan, dtype=float)
                            })

if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0
excl_list = ['ANT','GIB','GLP','GUF','MAC','MCO', 'MYT', 'NRU', 'PCN','PSE', 'REU', 'SCG', 'SJP', 'TKL']

for cnt_ind in range(len(isos)):
    if isos[cnt_ind] in excl_list:
        continue
    country = [isos[cnt_ind]]
    reg = regs[cnt_ind]
    #print(conts[cnt_ind]-1)
    cont = continent_names[int(conts[cnt_ind]-1)]
    litPopFix = LitPop()
    litPopFix.set_country(country, fin_mode='pc', res_arcsec=120, reference_year=2005)
    litPopFix['if_RF'] = conts[cnt_ind]

    save_lc = line_counter
    for pro_std in range(len(PROT_STD)):
        line_counter = save_lc
        dph_path = flood_dir + 'flddph_{}_{}_{}_gev_0.1.nc'\
            .format(args.RF_model, args.CL_model, PROT_STD[pro_std])
        frc_path= flood_dir + 'fldfrc_{}_{}_{}_gev_0.1.nc'\
            .format(args.RF_model, args.CL_model, PROT_STD[pro_std])
        if not os.path.exists(dph_path):
            print('{} path not found'.format(dph_path))
            break
        if not os.path.exists(frc_path):
            print('{} path not found'.format(frc_path))
            break
        rf = RiverFlood()
        rf.set_from_nc(dph_path=dph_path, frc_path=frc_path, countries=country, years=years)
        rf2y = copy.copy(rf)
        rf2y.exclude_returnlevel(RF_PATH_FRC)
        rf.set_flooded_area()
        for year in range(len(years)):
            print('country_{}_year{}_protStd_{}'.format(country[0], str(years[year]), PROT_STD[pro_std]))
            ini_date = str(years[year]) + '-01-01'
            fin_date = str(years[year]) + '-12-31'
            imp_fix=Impact()
            imp_fix.calc(litPopFix, if_set, rf.select(date=(ini_date, fin_date)))
            affected_imp_fix = imp_fix.exp_idx
            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            dataDF.iloc[line_counter, 3] = cont
            litPop = LitPop()
            litPop.set_country(country, fin_mode='pc', res_arcsec=120, ref_year=years[year])
            litPop['if_RF'] = conts[cnt_ind]
            litPop['centr_RF']= litPopFix['centr_RF']
            imp_fl=Impact()
            imp_fl.calc(litPop, if_set, rf.select(date=(ini_date, fin_date)))
            affected_ind = imp_fl.exp_idx
            
            if pro_std < 2:
                imp2y_fl=Impact()
                imp2y_fl.calc(litPop, if_set, rf2y.select(date=(ini_date,fin_date)))
                affected_imp2y = imp2y_fl.exp_idx
                imp2y_fix=Impact()
                imp2y_fix.calc(litPopFix, if_set, rf2y.select(date=(ini_date,fin_date)))
                affected_imp2y_fix = imp2y_fix.exp_idx
                dataDF.iloc[line_counter, 10 + pro_std] = imp2y_fl.at_event[0]
                dataDF.iloc[line_counter, 11 + pro_std] = imp2y_fix.at_event[0]
                dataDF.iloc[line_counter, 12] = np.mean(rf2y.intensity[year,:].data)

            dataDF.iloc[line_counter, 4] = imp_fl.tot_value
            dataDF.iloc[line_counter, 5] = imp_fix.tot_value
            dataDF.iloc[line_counter, 6] = np.mean(rf.intensity[year,:].data)
            dataDF.iloc[line_counter, 7 + pro_std] = rf.fla_annual[year]
            dataDF.iloc[line_counter, 8 + pro_std] = imp_fix.at_event[0]
            dataDF.iloc[line_counter, 9 + pro_std] = imp_fl.at_event[0]
            line_counter+=1
    if args.RF_model == 'lpjml':
        dataDF.to_csv('output_{}_{}_fullProt_lpjml_long_2y.csv'.format(args.RF_model, args.CL_model))
    else:
        dataDF.to_csv('output_{}_{}_litPop.csv'.format(args.RF_model, args.CL_model))


