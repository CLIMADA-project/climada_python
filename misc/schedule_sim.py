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
from climada.entity.exposures.gdp_asset import GDP2Asset
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


PROT_STD = ['0', 'flopros', '100']
#for LPJ longrun

#flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
#flood_dir = '/p/projects/ebm/data/hazard/floods/benoit_input_data/'
gdp_path = '/p/projects/ebm/data/exposure/gdp/processed_data/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'
RF_PATH_FRC = '/p/projects/ebm/tobias_backup/floods/climada/isimip2a/flood_maps/fldfrc24_2.nc'
exp_path = '/p/projects/ebm/data/exposure/LitPop/'
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
                            'FloodedArea0': np.full(l, np.nan, dtype=float),
                            'FloodedAreaFlopros': np.full(l, np.nan, dtype=float),
                            'FloodedArea100': np.full(l, np.nan, dtype=float),
                            'FloodVol0': np.full(l, np.nan, dtype=float),
                            'FloodVolFlopros': np.full(l, np.nan, dtype=float),
                            'FloodVol100': np.full(l, np.nan, dtype=float),
                            'ImpFixExp0': np.full(l, np.nan, dtype=float),
                            'ImpFixExpFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFixExp100': np.full(l, np.nan, dtype=float),
                            'Impact_0': np.full(l, np.nan, dtype=float),
                            'Impact_Flopros': np.full(l, np.nan, dtype=float),
                            'Impact_100': np.full(l, np.nan, dtype=float),
                            'Impact_2y_0': np.full(l, np.nan, dtype=float),
                            'Impact_2y_Flopros': np.full(l, np.nan, dtype=float),
                            'ImpFix_2y_0': np.full(l, np.nan, dtype=float),
                            'ImpFix_2y_Flopros': np.full(l, np.nan, dtype=float)
                            })

if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0

excl_list = ['AIA','ANT', 'COK', 'FLK','GIB','GLP', 'GGY','GUF','JEY','MAC',
             'MCO','MSR', 'MTQ','MYT', 'NFK', 'PRK','NIU','NRU', 'PCN','PSE',
             'REU', 'SCG', 'SJP','SHN', 'SPM','TKL','TWN','SJM', 'VGB', 'WLF']

for cnt_ind in range(len(isos)):
    
    country = [isos[cnt_ind]]
    if country[0] in excl_list: 
        continue
    reg = regs[cnt_ind]
    #print(conts[cnt_ind]-1)
    cont = continent_names[int(conts[cnt_ind]-1)]
    gdpaFix = GDP2Asset()
    gdpaFix.read_hdf5(exp_path + country[0] + '/Litpop_{}_2005.h5'.format(country[0]))
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
        rf.set_flood_volume()
        for year in range(len(years)):
            print('country_{}_year{}_protStd_{}'.format(country[0], str(years[year]), PROT_STD[pro_std]))
            ini_date = str(years[year]) + '-01-01'
            fin_date = str(years[year]) + '-12-31'
            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            dataDF.iloc[line_counter, 3] = cont
            imp_fix=Impact()
            imp_fix.calc(gdpaFix, if_set, rf.select(date=(ini_date, fin_date)))
            gdpa = GDP2Asset()
            gdpa.read_hdf5(exp_path + country[0] + '/Litpop_{}_{}.h5'.format(country[0], str(years[year])))
            gdpa['centr_RF']= gdpaFix['centr_RF']
            imp_fl=Impact()
            imp_fl.calc(gdpa, if_set, rf.select(date=(ini_date, fin_date)))
            
            if pro_std < 2:
                imp2y_fl=Impact()
                imp2y_fl.calc(gdpa, if_set, rf2y.select(date=(ini_date,fin_date)))
                imp2y_fix=Impact()
                imp2y_fix.calc(gdpaFix, if_set, rf2y.select(date=(ini_date,fin_date)))
                dataDF.iloc[line_counter, 18 + pro_std] = imp2y_fl.at_event[0]
                dataDF.iloc[line_counter, 20 + pro_std] = imp2y_fix.at_event[0]

            dataDF.iloc[line_counter, 4] = imp_fl.tot_value
            dataDF.iloc[line_counter, 5] = imp_fix.tot_value
            dataDF.iloc[line_counter, 6 + pro_std] = rf.fla_annual[year]
            dataDF.iloc[line_counter, 9 + pro_std] = rf.fv_annual[year,0]
            dataDF.iloc[line_counter, 12 + pro_std] = imp_fix.at_event[0]
            dataDF.iloc[line_counter, 15 + pro_std] = imp_fl.at_event[0]
            line_counter+=1
    if args.RF_model == 'lpjml':
        dataDF.to_csv('output_{}_{}_fullProt_lpjml_long_2y.csv'.format(args.RF_model, args.CL_model))
    else:
        dataDF.to_csv('LitPOP_{}_{}_fullProt_All_2y.csv'.format(args.RF_model, args.CL_model))


