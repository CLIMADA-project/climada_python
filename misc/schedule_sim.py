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
from climada.entity.exposures.gdp_asset import GDP2Asset, get_group_GDP
from climada.entity.exposures.exp_people import ExpPop, read_group_people
from climada.entity.impact_funcs.flood import IFRiverFlood,flood_imp_func_set, assign_if_simple
from climada.hazard.flood import RiverFlood
from climada.hazard.centroids import Centroids
from climada.entity import ImpactFuncSet
from climada.util.constants import NAT_REG_ID
import copy
from climada.entity.exposures.gdp2asset_dis import GDP2AssetDis
from climada.hazard.flood_trend import FloodTrend

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
dis_path = '/home/insauer/data/DischargeTrends/Regression_CDO_trends.nc'
pop_path = '/home/insauer/Tobias/hyde_ssp2_1860-2015_0150as_yearly_zip.nc4'



output = currentdir
#For lpj longrun
#if args.RF_model == 'lpjml':
#    flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a-advanced/'
#    if args.CL_model == 'watch':
#        years = np.arange(1901, 2002)
#    else:
#        years = np.arange(1901, 2011)
#else:
flood_dir = '/p/projects/ebm/data/hazard/floods/isimip2a/'
if args.CL_model == 'watch':
    years = np.arange(1971, 2002)
else:
    years = np.arange(1971, 2011)

#years = np.arange(1971, 2011)
income_groups = pd.read_csv('/home/insauer/data/CountryInfo/IncomeGroups.csv')
country_info = pd.read_csv(NAT_REG_ID)
isos = country_info['ISO'].tolist()

cont_list = country_info['if_RF'].tolist()
l = len(years) * len(isos)
continent_names = ['Africa', 'Asia', 'Europe', 'NorthAmerica', 'Oceania', 'SouthAmerica']


dataDF = pd.DataFrame(data={'Year': np.full(l, np.nan, dtype=int),
                            'Country': np.full(l, "", dtype=str),
                            'Region': np.full(l, "", dtype=str),
                            'Continent': np.full(l, "", dtype=str),
                            'TotalAssetValue': np.full(l, np.nan, dtype=float),
                            'TotalAssetValue2005': np.full(l, np.nan, dtype=float),
                            'FloodedAreaPosFlopros': np.full(l, np.nan, dtype=float),
                            'FloodedAreaNegFlopros': np.full(l, np.nan, dtype=float),
                            'FloodVolumePosFlopros': np.full(l, np.nan, dtype=float),
                            'FloodVolumeNegFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFixPosFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFixNegFlopros': np.full(l, np.nan, dtype=float),
                            'ImpactPosFlopros': np.full(l, np.nan, dtype=float),
                            'ImpactNegFlopros': np.full(l, np.nan, dtype=float),
                            'Impact_2yPosFlopros': np.full(l, np.nan, dtype=float),
                            'Impact_2yNegFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yPosFlopros': np.full(l, np.nan, dtype=float),
                            'ImpFix_2yNegFlopros': np.full(l, np.nan, dtype=float),
                            'GDP_Pos': np.full(l, np.nan, dtype=float),
                            'GDP_Neg': np.full(l, np.nan, dtype=float),
                            'Pop_Pos': np.full(l, np.nan, dtype=float),
                            'Pop_Neg': np.full(l, np.nan, dtype=float)
                            })

if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0

for cnt_ind in range(len(isos)):
    country = [isos[cnt_ind]]
    reg = country_info.loc[country_info['ISO']== country[0], 'Reg_name'].values[0]
    conts = country_info.loc[country_info['ISO']== country[0], 'if_RF'].values[0]

    #print(conts[cnt_ind]-1)
    cont = continent_names[int(conts-1)]
    gdpaFix = GDP2Asset()
    gdpaFix.set_countries(countries=country, ref_year=2005, path=gdp_path)
    #gdpaFix.correct_for_SSP(ssp_corr, country[0])
    save_lc = line_counter
    
    dis_pos = FloodTrend()
    dis_pos.set_from_nc(dph_path=dis_path, countries=country[0])
    dis_neg = copy.copy(dis_pos)
    dis_pos.get_dis_mask(dis = 'pos')
    dis_neg.get_dis_mask(dis = 'neg')
    
    shp_exposures = np.zeros((dis_pos.centroids.lat.size,2))
    shp_exposures[:,0]= dis_pos.centroids.lat
    shp_exposures[:,1]= dis_pos.centroids.lon
    
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

        rf_pos = RiverFlood()
        rf_pos.set_from_nc(dph_path=dph_path, frc_path=frc_path, countries=country, years=years)
        rf_neg = copy.copy(rf_pos)
        rf2y_pos = copy.copy(rf_pos)
        
        rf_pos.exclude_trends(dis_pos)
        rf_neg.exclude_trends(dis_neg)

        rf2y_pos.exclude_returnlevel(RF_PATH_FRC)
        rf2y_neg = copy.copy(rf2y_pos)
        rf2y_pos.exclude_trends(dis_pos)
        rf2y_neg.exclude_trends(dis_neg)
        rf_pos.set_flooded_area()
        rf_neg.set_flooded_area()
        rf_pos.set_flood_volume()
        rf_neg.set_flood_volume()
        for year in range(len(years)):
            print('country_{}_year{}_protStd_{}'.format(country[0], str(years[year]), PROT_STD[pro_std]))
            
            

            gdp_pos = get_group_GDP(shp_exposures, ref_year=year, path = gdp_path, dis_mask = dis_pos)
            gdp_neg = get_group_GDP(shp_exposures, ref_year=year, path = gdp_path, dis_mask = dis_neg)

            pop_pos = read_group_people(shp_exposures, ref_year=year, path=pop_path, dis_mask= dis_pos)
            pop_neg = read_group_people(shp_exposures, ref_year=year, path= pop_path, dis_mask = dis_neg)

            ini_date = str(years[year]) + '-01-01'
            fin_date = str(years[year]) + '-12-31'
            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            dataDF.iloc[line_counter, 3] = cont
            gdpa = GDP2Asset()
            gdpa.set_countries(countries=country, ref_year=years[year], path = gdp_path)
            #gdpa.correct_for_SSP(ssp_corr, country[0])
            imp_fl_pos=Impact()
            imp_fl_pos.calc(gdpa, if_set, rf_pos.select(date=(ini_date, fin_date)))
            imp_fl_neg=Impact()
            imp_fl_neg.calc(gdpa, if_set, rf_neg.select(date=(ini_date, fin_date)))
            imp_fl_fix_pos=Impact()
            imp_fl_fix_pos.calc(gdpaFix, if_set, rf_pos.select(date=(ini_date, fin_date)))
            imp_fl_fix_neg=Impact()
            imp_fl_fix_neg.calc(gdpaFix, if_set, rf_neg.select(date=(ini_date, fin_date)))
            if pro_std < 2:
                imp2y_fl_pos=Impact()
                imp2y_fl_pos.calc(gdpa, if_set, rf2y_pos.select(date=(ini_date,fin_date)))
                imp2y_fl_neg=Impact()
                imp2y_fl_neg.calc(gdpa, if_set, rf2y_neg.select(date=(ini_date,fin_date)))
                imp2y_fl_fix_pos=Impact()
                imp2y_fl_fix_pos.calc(gdpaFix, if_set, rf2y_pos.select(date=(ini_date,fin_date)))
                imp2y_fl_fix_neg=Impact()
                imp2y_fl_fix_neg.calc(gdpaFix, if_set, rf2y_neg.select(date=(ini_date,fin_date)))
                
                dataDF.iloc[line_counter, 14 + pro_std] = imp2y_fl_pos.at_event[0]
                dataDF.iloc[line_counter, 15 + pro_std] = imp2y_fl_neg.at_event[0]
                
                dataDF.iloc[line_counter, 16 + pro_std] = imp2y_fl_fix_pos.at_event[0]
                dataDF.iloc[line_counter, 17 + pro_std] = imp2y_fl_fix_neg.at_event[0]

            dataDF.iloc[line_counter, 4] = imp_fl_pos.tot_value
            dataDF.iloc[line_counter, 5] = imp_fl_fix_pos.tot_value
            
            dataDF.iloc[line_counter, 6 + pro_std] = rf_pos.fla_annual[year]
            dataDF.iloc[line_counter, 7 + pro_std] = rf_neg.fla_annual[year]
            
            dataDF.iloc[line_counter, 8 + pro_std] = rf_pos.fv_annual[year,0]
            dataDF.iloc[line_counter, 9 + pro_std] = rf_neg.fv_annual[year,0]
            
            dataDF.iloc[line_counter, 10 + pro_std] = imp_fl_fix_pos.at_event[0]
            dataDF.iloc[line_counter, 11 + pro_std] = imp_fl_fix_neg.at_event[0]
            
            dataDF.iloc[line_counter, 12 + pro_std] = imp_fl_pos.at_event[0]
            dataDF.iloc[line_counter, 13 + pro_std] = imp_fl_neg.at_event[0]
            
            dataDF.iloc[line_counter, 18] = gdp_pos
            dataDF.iloc[line_counter, 19] = gdp_neg
            dataDF.iloc[line_counter, 20] = pop_pos
            dataDF.iloc[line_counter, 21] = pop_neg
            
            line_counter+=1
    #if args.RF_model == 'lpjml':
        #dataDF.to_csv('output_{}_{}_fullProt_lpjml_long_2y.csv'.format(args.RF_model, args.CL_model))
    #else:
    dataDF.to_csv('DisRiskOutput_{}_{}_flopros_newFLD_10_02.csv'.format(args.RF_model, args.CL_model))
