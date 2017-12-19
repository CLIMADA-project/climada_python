"""
=====================
Config
=====================

Define configuration variables used in the climada execution.

"""
# Author: Gabriela Aznar Siguan (gabriela.aznar@usys.ethz.ch)
# Created on Fri Dec  1 16:20:31 2017

#    Copyright (C) 2017 by
#    David N. Bresch, david.bresch@gmail.com
#    Gabriela Aznar Siguan (g.aznar.siguan@gmail.com)
#    All rights reserved.

import os

working_dir = os.path.dirname(__file__) + '/../../'
data_dir = working_dir + 'data/'

hazard_default = data_dir + 'demo/Excel_hazard.xlsx'
hazard_mat = data_dir + 'demo/atl_prob.mat'
entity_default = data_dir + 'demo/demo_today.xlsx'

present_ref_year = 2016
future_ref_year = 2030