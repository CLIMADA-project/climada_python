"""
climada demo
"""

# Import classes
from climada import Entity, Hazard, Impact
from climada import HAZ_DEMO_MAT, ENT_DEMO_MAT

# Load entity
ent_demo = Entity(ENT_DEMO_MAT)

# Plot exposures values
ent_demo.exposures.plot_value()

# Plot impact functions
ent_demo.impact_funcs.plot()

# Load Hazard
haz_demo = Hazard(HAZ_DEMO_MAT, 'TC')

# Plot some events
haz_demo.plot_intensity(event=5489)
haz_demo.plot_intensity(event=-1)
haz_demo.plot_intensity(event='DONNA       ')
haz_demo.plot_intensity(centr_id=46)
haz_demo.plot_fraction(event=-1)

# Plot impact exceedence frequency curve
imp_demo = Impact()
imp_demo.calc(ent_demo.exposures, ent_demo.impact_funcs, haz_demo)
ifc_demo = imp_demo.calc_freq_curve()
ifc_demo.plot()
