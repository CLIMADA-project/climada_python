#!/usr/bin/env python
# coding: utf-8

from climada.hazard.drought import Drought

from climada.entity.impact_funcs.drought import IFDrought
# from climada.entity import Entity
from climada.entity import ImpactFuncSet
from climada.engine import Impact
from climada.entity.exposures.spam_agrar import SpamAgrar
import numpy as np

"""Set Area to be analysed"""


"""Set method for defining intensity between 1 (default): min 2:sum-threshold 3:sum"""
intensity_definition = 3

"""Initialize default threshold (default: -1)"""
threshold = -1.5

#Threshold and intensity_definition to be defined only if not defalut values are used

"""To define only if no default data are used (spei06)"""
#spei_file_dir = r'C:\Users\veron\Documents\ETH\HS18\PROJECT\GIT\climada_python\data\system'
#spei_file_name = r'spei02.nc'
#spei_file_url = r'http://digital.csic.es/bitstream/10261/153475/4'

"""To define if the data are not in default path \climada_python\data\system"""
#file_path_spei = r'C:\Users\veron\Documents\ETH\HS18\PROJECT\GIT\climada_python\data\system\spei06.nc'


"""Initialize hazard Drought"""
d = Drought()

"""Set area in latitudinal longitudinal coordinates"""
#d.set_area(latmin, lonmin, latmax, lonmax)

"""Set if non default parameters are used"""
d.set_threshold(threshold)
d.set_intensity_def(intensity_definition)

"""Set link to download data if a non default data is needed (default: spei06)"""
#d.set_file_name(spei_file_name)
#d.set_file_url(spei_file_url)

"""Set path if the data are not in 'climada_python\data\system'"""
#d.set_file_path(file_path_spei)

"""Setup the hazard"""
new_haz = d.setup()

"""Plot intensity of one year event"""
# new_haz.plot_intensity_drought(event='2003')

"""Initialize Impact function"""
dr_if = ImpactFuncSet()
if_def = IFDrought()
"""set impact function: for min: set_default; for sum-thr: set_default_sumthr; for sum: set_default_sum"""
#if_def.set_default()
#if_def.set_default_sumthr()
if_def.set_default_sum()
dr_if.append(if_def)

"""Initialize Exposure"""
exposure_agrar = SpamAgrar()
exposure_agrar.init_spam_agrar(country='CHE')

"""If intensity def is not default, exposure has to be adapted"""
"""In case of sum-thr: 'if_DR_sumthr', in case of sum:'if_DR_sum'"""
#exposure_agrar['if_DR_sumthr'] = np.ones(exposure_agrar.shape[0])
exposure_agrar['if_DR_sum'] = np.ones(exposure_agrar.shape[0])

"""Initialize impact of the drought"""
imp_drought = Impact()

"""Calculate Damage for a specific event"""
imp_drought.calc(exposure_agrar, dr_if, new_haz)
index_event_start = imp_drought.event_name.index('2003')
damages_drought = np.asarray([imp_drought.at_event[index_event_start]])
print(damages_drought)
