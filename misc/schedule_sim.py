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


PROT_STD = ['0', '100', 'flopros']
#for LPJ longrun

#flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
#flood_dir = '/p/projects/ebm/data/hazard/floods/benoit_input_data/'
gdp_path = '/p/projects/ebm/data/exposure/gdp/processed_data/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'
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
    years = years = np.arange(1971, 2002)
# 	if args.CL_model == 'watch':
#       years = np.arange(1971, 2002)
#    else:
#        years = np.arange(1971, 2011)

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
                            'FloodedArea100': np.full(l, np.nan, dtype=float),
                            'FloodedAreaFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFixExp0': np.full(l, np.nan, dtype=float),
                            'ImpFixExp100': np.full(l, np.nan, dtype=float),
                            'ImpFixExpFlopros': np.full(l, np.nan, dtype=float),
                            'ImpactAnnual0': np.full(l, np.nan, dtype=float),
                            'ImpactAnnual100': np.full(l, np.nan, dtype=float),
                            'ImpactAnnualFlopros': np.full(l, np.nan, dtype=float)
                            })

if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0

for cnt_ind in range(len(isos)):
    country = [isos[cnt_ind]]
    reg = regs[cnt_ind]
    #print(conts[cnt_ind]-1)
    cont = continent_names[int(conts[cnt_ind]-1)]
    gdpaFix = GDP2Asset()
    gdpaFix.set_countries(countries=country, ref_year=2005, path=gdp_path)

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
        rf.set_flooded_area()
        for year in range(len(years)):
            print('country_{}_year{}_protStd_{}'.format(country[0], str(years[year]), PROT_STD[pro_std]))
            ini_date = str(years[year]) + '-01-01'
            fin_date = str(years[year]) + '-12-31'
            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            dataDF.iloc[line_counter, 3] = cont
            gdpa = GDP2Asset()
            gdpa.set_countries(countries=country, ref_year=years[year], path = gdp_path)
            imp_fl=Impact()
            imp_fl.calc(gdpa, if_set, rf.select(date=(ini_date, fin_date)))
            imp_fix=Impact()
            imp_fix.calc(gdpaFix, if_set, rf.select(date=(ini_date, fin_date)))

            dataDF.iloc[line_counter, 4] = imp_fl.tot_value
            dataDF.iloc[line_counter, 5] = imp_fix.tot_value
            dataDF.iloc[line_counter, 6 + pro_std] = rf.fla_annual[year]
            dataDF.iloc[line_counter, 9 + pro_std] = imp_fix.at_event[0]
            dataDF.iloc[line_counter, 12 + pro_std] = imp_fl.at_event[0]
            line_counter+=1
    if args.RF_model == 'lpjml':
        dataDF.to_csv('output_{}_{}_fullProt_lpjml_long.csv'.format(args.RF_model, args.CL_model))
    else:
        dataDF.to_csv('output_{}_{}_fullProt_All.csv'.format(args.RF_model, args.CL_model))


