#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:29:10 2020

@author: insauer
"""

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
from climada.entity.impact_funcs.river_flood import flood_imp_func_set
from climada.hazard.river_flood import RiverFlood
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
parser.add_argument(
    '--Socmode', type=str, default='pressoc',
    help='social interaction in ghms')
parser.add_argument(
    '--SM_mode', type=str, default='smooth',
    help='social interaction in ghms')
args = parser.parse_args()

#Todo for cluster application
# set cluster true
# set output path
# set all countries
# set output dir


PROT_STD = ['0','flopros']
#for LPJ longrun

#flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
#flood_dir = '/p/projects/ebm/data/hazard/floods/benoit_input_data/'
gdp_path = '/p/projects/ebm/data/exposure/gdp/processed_data/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'
RF_PATH_FRC = '/p/projects/ebm/tobias_backup/floods/climada/isimip2a/flood_maps/fldfrc24_2.nc'


output = currentdir
#For lpj longrun
#if args.RF_model == 'lpjml':
#    flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
#    if args.CL_model == 'watch':
#        years = np.arange(1901, 2002)
#    else:
#        years = np.arange(1901, 2011)
#else:
if args.SM_mode == 'smooth':
    dis_path = '/home/insauer/projects/RiverDischarge/SmoothTrends_24_2.nc'
else:
    dis_path = '/home/insauer/projects/RiverDischarge/TrendsMedianDischarge_MK.nc'


if args.Socmode == 'nosoc':
    flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a/'.format(args.Socmode)
else:
    flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-{}/'.format(args.Socmode)

if args.CL_model == 'watch':
    years = np.arange(1971, 1972)
else:
    years = np.arange(1971, 1972)

#years = np.arange(1971, 2011)
income_groups = pd.read_csv('/home/insauer/data/CountryInfo/IncomeGroups.csv')
country_info = pd.read_csv(NAT_REG_ID)
isos = country_info['ISO'].tolist()


cont_list = country_info['if_RF'].tolist()
l = (len(years) * (len(isos)-2))
continent_names = ['Africa', 'Asia', 'Europe', 'NorthAmerica', 'Oceania', 'SouthAmerica']


dataDF = pd.DataFrame(data={'Year': np.full(l, np.nan, dtype=int),
                            'Country': np.full(l, "", dtype=str),
                            'Region': np.full(l, "", dtype=str),
                            'Continent': np.full(l, "", dtype=str),
                            'IncomeGroup': np.full(l, "", dtype=str),
                            'TotalAssetValue': np.full(l, np.nan, dtype=float),
                            'TotalAssetValue1980': np.full(l, np.nan, dtype=float),
                            'FloodedAreaPos0': np.full(l, np.nan, dtype=float),
                            'FloodedAreaPosFlopros': np.full(l, np.nan, dtype=float),
                            'FloodedAreaNeg0': np.full(l, np.nan, dtype=float),
                            'FloodedAreaNegFlopros': np.full(l, np.nan, dtype=float),
                            'FloodedArea0': np.full(l, np.nan, dtype=float),
                            'FloodedAreaFlopros': np.full(l, np.nan, dtype=float),
                            'FloodVolumePos0': np.full(l, np.nan, dtype=float),
                            'FloodVolumePosFlopros': np.full(l, np.nan, dtype=float),
                            'FloodVolumeNeg0': np.full(l, np.nan, dtype=float),
                            'FloodVolumeNegFlopros': np.full(l, np.nan, dtype=float),
                            'FloodVolume0': np.full(l, np.nan, dtype=float),
                            'FloodVolumeFlopros': np.full(l, np.nan, dtype=float),
                            'Impact_2yPos0': np.full(l, np.nan, dtype=float),
                            'Impact_2yPosFlopros': np.full(l, np.nan, dtype=float),
                            'Impact_2yNeg0': np.full(l, np.nan, dtype=float),
                            'Impact_2yNegFlopros': np.full(l, np.nan, dtype=float),
                            'Impact_2y0': np.full(l, np.nan, dtype=float),
                            'Impact_2yFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yPos0': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yPosFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yNeg0': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yNegFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFix_2y0': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yFlopros': np.full(l, np.nan, dtype=float),
                            'Imp2010_2yPos0': np.full(l, np.nan, dtype=float),
                            'Imp2010_2yPosFlopros': np.full(l, np.nan, dtype=float),
                            'Imp2010_2yNeg0': np.full(l, np.nan, dtype=float),
                            'Imp2010_2yNegFlopros': np.full(l, np.nan, dtype=float),
                            'Imp2010_2y0': np.full(l, np.nan, dtype=float),
                            'Imp2010_2yFlopros': np.full(l, np.nan, dtype=float)
                            })

if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0

for cnt_ind in range(len(isos)):
    country = [isos[cnt_ind]]
    
    if country[0] in ['GIB','MCO']:
        continue
    reg = country_info.loc[country_info['ISO']== country[0], 'Reg_name'].values[0]
    conts = country_info.loc[country_info['ISO']== country[0], 'if_RF'].values[0]
    #print(conts[cnt_ind]-1)
    cont = continent_names[int(conts-1)]
    gdpa1980 = GDP2Asset()
    gdpa1980.set_countries(countries=country, ref_year=1980, path=gdp_path)
    gdpa2010 = GDP2Asset()
    gdpa2010.set_countries(countries=country, ref_year=2010, path=gdp_path)
    #gdpaFix.correct_for_SSP(ssp_corr, country[0])
    save_lc = line_counter
    
    
    for pro_std in range(len(PROT_STD)):
        line_counter = save_lc
        dph_path = flood_dir + '{}/{}/depth-150arcsec/flddph_annual_max_gev_0.1mmpd_protection-{}.nc'\
            .format(args.CL_model, args.RF_model, PROT_STD[pro_std])
        frc_path= flood_dir + '{}/{}/area-150arcsec/fldfrc_annual_max_gev_0.1mmpd_protection-{}.nc'\
            .format(args.CL_model, args.RF_model, PROT_STD[pro_std])
            
        if not os.path.exists(dph_path):
            print('{} path not found'.format(dph_path))
            break
        if not os.path.exists(frc_path):
            print('{} path not found'.format(frc_path))
            break

        
        rf = RiverFlood()
        
        rf.set_from_nc(dph_path=dph_path, frc_path=frc_path,
                       countries=country, years = years, ISINatIDGrid=True)
        
        rf_pos = copy.copy(rf)
        rf_pos.exclude_trends(dis_path, 'pos')
        
        rf_neg = copy.copy(rf)
        rf_neg.exclude_trends(dis_path, 'neg')
        
        rf.set_flooded_area(save_centr=True)
        rf.set_flood_volume()
        rf_pos.set_flooded_area(save_centr=True)
        rf_neg.set_flooded_area(save_centr=True)
        rf_pos.set_flood_volume()
        rf_neg.set_flood_volume()
        
        
        
        rf2y = copy.copy(rf)
        
        rf2y.exclude_returnlevel(RF_PATH_FRC)

        rf2y_pos = copy.copy(rf2y)
        
        rf2y_pos.exclude_trends(dis_path, 'pos')
        
        rf2y_neg = copy.copy(rf2y)
        rf2y_neg.exclude_trends(dis_path, 'neg')
    
        for year in range(len(years)):
            print('country_{}_year{}_protStd_{}'.format(country[0], str(years[year]), PROT_STD[pro_std]))
            ini_date = str(years[year]) + '-01-01'
            fin_date = str(years[year]) + '-12-31'
            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            dataDF.iloc[line_counter, 3] = cont
            dataDF.iloc[line_counter, 4] = 0
            gdpa = GDP2Asset()
            gdpa.set_countries(countries=country, ref_year=years[year], path = gdp_path)
            #gdpa.correct_for_SSP(ssp_corr, country[0])
            
            imp2y_fl_pos=Impact()
            imp2y_fl_pos.calc(gdpa, if_set, rf2y_pos.select(date=(ini_date,fin_date)))
            imp2y_fl_neg=Impact()
            imp2y_fl_neg.calc(gdpa, if_set, rf2y_neg.select(date=(ini_date,fin_date)))
            imp2y_fl=Impact()
            imp2y_fl.calc(gdpa, if_set, rf2y.select(date=(ini_date,fin_date)))
            
            imp2y_fl_1980_pos=Impact()
            imp2y_fl_1980_pos.calc(gdpa1980, if_set, rf2y_pos.select(date=(ini_date,fin_date)))
            imp2y_fl_1980_neg=Impact()
            imp2y_fl_1980_neg.calc(gdpa1980, if_set, rf2y_neg.select(date=(ini_date,fin_date)))
            imp2y_fl_1980=Impact()
            imp2y_fl_1980.calc(gdpa1980, if_set, rf2y.select(date=(ini_date,fin_date)))
            
            imp2y_fl_2010_pos=Impact()
            imp2y_fl_2010_pos.calc(gdpa2010, if_set, rf2y_pos.select(date=(ini_date,fin_date)))
            imp2y_fl_2010_neg=Impact()
            imp2y_fl_2010_neg.calc(gdpa2010, if_set, rf2y_neg.select(date=(ini_date,fin_date)))
            imp2y_fl_2010=Impact()
            imp2y_fl_2010.calc(gdpa2010, if_set, rf2y.select(date=(ini_date,fin_date)))

            dataDF.iloc[line_counter, 5] = imp2y_fl.tot_value
            dataDF.iloc[line_counter, 6] = imp2y_fl_1980.tot_value
            
            dataDF.iloc[line_counter, 7 + pro_std] = rf_pos.fla_annual[year]
            dataDF.iloc[line_counter, 9 + pro_std] = rf_neg.fla_annual[year]
            dataDF.iloc[line_counter, 11 + pro_std] = rf.fla_annual[year]
            
            dataDF.iloc[line_counter, 13 + pro_std] = rf_pos.fv_annual[year,0]
            dataDF.iloc[line_counter, 15 + pro_std] = rf_neg.fv_annual[year,0]
            dataDF.iloc[line_counter, 17 + pro_std] = rf.fv_annual[year,0]
            
            
            dataDF.iloc[line_counter, 19 + pro_std] = imp2y_fl_pos.at_event[0]
            dataDF.iloc[line_counter, 21 + pro_std] = imp2y_fl_neg.at_event[0]
            dataDF.iloc[line_counter, 23 + pro_std] = imp2y_fl.at_event[0]
            
            dataDF.iloc[line_counter, 25 + pro_std] = imp2y_fl_1980_pos.at_event[0]
            dataDF.iloc[line_counter, 27 + pro_std] = imp2y_fl_1980_neg.at_event[0]
            dataDF.iloc[line_counter, 29 + pro_std] = imp2y_fl_1980.at_event[0]
            
            dataDF.iloc[line_counter, 31 + pro_std] = imp2y_fl_2010_pos.at_event[0]
            dataDF.iloc[line_counter, 33 + pro_std] = imp2y_fl_2010_neg.at_event[0]
            dataDF.iloc[line_counter, 35 + pro_std] = imp2y_fl_2010.at_event[0]
            
            line_counter+=1
    #if args.RF_model == 'lpjml':
        #dataDF.to_csv('output_{}_{}_fullProt_lpjml_long_2y.csv'.format(args.RF_model, args.CL_model))
    #else:
    dataDF.to_csv('Test_run_{}_Output_{}_{}_0flopros_8010_{}.csv'.format(args.SM_mode, args.RF_model, args.CL_model, args.Socmode))


