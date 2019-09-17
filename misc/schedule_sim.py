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
from climada.hazard import TCTracks, Centroids, TropCyclone
from climada.util.constants import GLB_CENTROIDS_MAT
from climada.entity.exposures.gdp_asset import GDP2Asset
from climada.entity.impact_funcs.trop_cyclone import IFTropCyclone 
from climada.entity import ImpactFuncSet
from climada.hazard.flood import RiverFlood
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

gdp_path = '/p/projects/ebm/data/exposure/gdp/processed_data/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'

years =  [args.RF_model]
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
                            'TotalAsset': np.full(l, np.nan, dtype=float),
                            'Impact': np.full(l, np.nan, dtype=float),
                            'FixedImpact': np.full(l, np.nan, dtype=float),
                            'NumberEvents': np.full(l, np.nan, dtype=float)
                            })

if_set = ImpactFuncSet()
if_TC = IFTropCyclone()
if_TC.set_emanuel_usa()
if_set.append(if_TC)

fail_lc = 0
line_counter = 0
basins= ['NI', 'SI', 'EP','WP', 'SP', 'NA']
test = [0, 175]
for cnt_ind in range(len(isos)):

    country = [isos[cnt_ind]]
    reg = regs[cnt_ind]
    #print(conts[cnt_ind]-1)
    cont = continent_names[int(conts[cnt_ind]-1)]

    save_lc = line_counter
    glob_centr = Centroids()
    glob_centr= RiverFlood.select_exact_area(country)
    gdpaFix = GDP2Asset()
    gdpaFix.set_countries(countries=country, ref_year=2005, path = gdp_path)
    for year in range(len(years)):
        print('country_{}_year{}'.format(country[0], str(years[year])))
        dataDF.iloc[line_counter, 0] = years[year]
        dataDF.iloc[line_counter, 1] = country[0]
        dataDF.iloc[line_counter, 2] = reg
        dataDF.iloc[line_counter, 3] = cont
        gdpa = GDP2Asset()
        gdpa.set_countries(countries=country, ref_year=years[year], path = gdp_path)
        damage = 0
        fixDamage = 0
        events = 0
        for basin in basins:
            hist_tr = TCTracks()
            hist_tr.read_ibtracs_netcdf(year_range=(years[year], years[year]+1), basin = basin)
            hist_tc = TropCyclone()
            hist_tc.set_from_tracks(hist_tr,glob_centr)
            imp_TC=Impact()
            imp_TC.calc(gdpa, if_set, hist_tc)
            imp_TCFix=Impact()
            imp_TCFix.calc(gdpaFix, if_set, hist_tc)
            if np.sum(imp_TC.at_event) > 0.01:
                events+=1
            damage += np.sum(imp_TC.at_event)
            fixDamage += np.sum(imp_TC.at_event)
        dataDF.iloc[line_counter, 4] = imp_TC.tot_value
        dataDF.iloc[line_counter, 5] = damage
        dataDF.iloc[line_counter, 6] = fixDamage
        dataDF.iloc[line_counter, 7] = events
        line_counter+=1
    dataDF.to_csv('TC_test_allCountries_{}.csv'.format(str(args.RF_model)))
    #else:
        #dataDF.to_csv('ThresExpPop_{}_{}_flopros_2yr.csv'.format(args.RF_model, args.CL_model))
