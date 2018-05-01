"""
climada demo
"""

# Import classes
from climada import Entity, Hazard, Impact, Exposures
from climada import HAZ_DEMO_MAT, ENT_TEMPLATE_XLS, ENT_DEMO_MAT

def calc_plot_impact():
    """Compute impact from demo entity and hazard."""
    # Load entity
    ent_demo = Entity(ENT_TEMPLATE_XLS)
    
    # Plot exposures values
    ent_demo.exposures.plot()
    
    # Plot impact functions
    ent_demo.impact_funcs.plot()
   
    # Load Hazard
    haz_demo = Hazard('TC', HAZ_DEMO_MAT)
    
    # Plot some events
    haz_demo.plot_intensity(event=5489)
    haz_demo.plot_intensity(event=-1)
    haz_demo.plot_intensity(event='DONNA')
    haz_demo.plot_intensity(centr=46)
    haz_demo.plot_fraction(event=-1)
    
    # Plot impact exceedence frequency curve
    imp_demo = Impact()
    imp_demo.calc(ent_demo.exposures, ent_demo.impact_funcs, haz_demo)
    imp_demo.plot_at_exposure()
    ifc_demo = imp_demo.calc_freq_curve()
    ifc_demo.plot()

def load_default_entity():
    """Define entity with default values"""
    ent = Entity()
    ent.exposures = Exposures(ENT_DEMO_MAT)
    
    ent.impact_funcs.plot()

def fail_wrong_input():
    """Error is raised in different occasions when a wrong input is provided"""
    ent = Entity()
    
    try:
        ent.measures = Exposures()
    except ValueError:
        print('Successfully error raised')

if __name__ == "__main__":
    calc_plot_impact()
    load_default_entity()
    fail_wrong_input()
