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
flood_dir = '/p/projects/ebm/data/hazard/floods/benoit_input_data/'
gdp_path = '/p/projects/ebm/data/exposure/gdp/processed_data/gdp_1850-2100_downscaled-by-nightlight_2.5arcmin_remapcon_new_yearly_shifted.nc'
output = currentdir
years = np.arange(1971, 2011)
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
                            'FloodedArea0': np.full(l, np.nan, dtype=float),
                            'FloodedArea100': np.full(l, np.nan, dtype=float),
                            'FloodedAreaFlopros': np.full(l, np.nan, dtype=float),
                            'ExpAsset0': np.full(l, np.nan, dtype=float),
                            'ExpAsset100': np.full(l, np.nan, dtype=float),
                            'ExpAssetFlopros': np.full(l, np.nan, dtype=float),
                            'ImpactAnnual0': np.full(l, np.nan, dtype=float),
                            'ImpactAnnual100': np.full(l, np.nan, dtype=float),
                            'ImpactAnnualFlopros': np.full(l, np.nan, dtype=float)
                            })
maxF = 1
failureDF = pd.DataFrame(data={'rfModel': np.full(maxF, "", dtype=str),
                               'CLData': np.full(maxF, "", dtype=str)
                              })
if_set = flood_imp_func_set()

fail_lc = 0
line_counter = 0
try:
    for cnt_ind in range(2):
        country = [isos[cnt_ind]]
        reg = regs[cnt_ind]
        #print(conts[cnt_ind]-1)
        cont = continent_names[int(conts[cnt_ind]-1)]
        gdpaFix = GDP2Asset()
        gdpaFix.set_countries(countries=country, ref_year=2005, path = gdp_path)
        for year in range(len(years)):

            dataDF.iloc[line_counter, 0] = years[year]
            dataDF.iloc[line_counter, 1] = country[0]
            dataDF.iloc[line_counter, 2] = reg
            dataDF.iloc[line_counter, 3] = cont
            gdpa = GDP2Asset()
            gdpa.set_countries(countries=country, ref_year=years[year], path = gdp_path)
            

            for pro_std in range(len(PROT_STD)):
                dph_path = flood_dir +'flddph_{}_{}_{}_gev_0.1.nc'\
                   .format(args.RF_model, args.CL_model, PROT_STD[pro_std])
                frc_path= flood_dir+'fldfrc_{}_{}_{}_gev_0.1.nc'\
                   .format(args.RF_model, args.CL_model, PROT_STD[pro_std])

                if not os.path.exists(dph_path):
                    print('{} path not found'.format(dph_path))
                    raise KeyError
                if not os.path.exists(frc_path):
                    print('{} path not found'.format(frc_path))
                    raise KeyError

                rf = RiverFlood()
                rf.set_from_nc(dph_path=dph_path, frc_path=frc_path, countries=country, years=[years[year]])
                rf.set_flooded_area()

                imp_fl=Impact()
                imp_fl.calc(gdpa, if_set, rf)
                imp_fix=Impact()
                imp_fix.calc(gdpaFix, if_set, rf)
                
                dataDF.iloc[line_counter, 4] = imp_fl.tot_value
                dataDF.iloc[line_counter, 5 + pro_std] = rf.fla_annual[0]
                dataDF.iloc[line_counter, 8 + pro_std] = imp_fix.at_event[0]
                dataDF.iloc[line_counter, 11 + pro_std] = imp_fl.at_event[0]

            line_counter+=1
    dataDF.to_csv(output + 'output_{}_{}_{}.csv'.format(args.RF_model, args.CL_model, PROT_STD[pro_std]))
except KeyError:
    print('run failed')
    failureDF.iloc[fail_lc, 0] = args.RF_model
    failureDF.iloc[fail_lc, 1] = args.CL_model
    fail_lc+=1
except AttributeError:
    print('run failed')
    failureDF.iloc[fail_lc, 0] = args.RF_model
    failureDF.iloc[fail_lc, 1] = args.CL_model
    fail_lc+=1
except NameError:
    print('run failed')
    failureDF.iloc[fail_lc, 0] = args.RF_model
    failureDF.iloc[fail_lc, 1] = args.CL_model
    fail_lc+=1
except IOError:
    print('run failed')
    failureDF.iloc[fail_lc, 0] = args.RF_model
    failureDF.iloc[fail_lc, 1] = args.CL_model
    fail_lc+=1
except IndexError:
    print('run failed')
    failureDF.iloc[fail_lc, 0] = args.RF_model
    failureDF.iloc[fail_lc, 1] = args.CL_model
    fail_lc+=1

failureDF.to_csv(output + 'failure_{}_{}.csv'.format(args.RF_model, args.CL_model))
