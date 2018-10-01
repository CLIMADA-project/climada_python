"""
Define Impact and ImpactFreqCurve classes.
"""

__all__ = ['ImpactFreqCurve', 'Impact']

import logging
import numpy as np

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHazard
from climada.util.coordinates import GridPoints
import climada.util.plot as plot

LOGGER = logging.getLogger(__name__)

MAX_SIZE = 1.0e8
""" Maximum matrix size for impact caluculation. If the matrix is bigger,
it is chunked."""

class ImpactFreqCurve(object):
    """ Impact exceedence frequency curve.

    Attributes:
        return_per (np.array): return period
        impact (np.array): impact exceeding frequency
        unit (str): value unit used (given by exposures unit)
        label (str): string describing source data
    """
    def __init__(self):
        self.return_per = np.array([])
        self.impact = np.array([])
        self.unit = 'NA'
        self.label = ''

    def plot(self):
        """Plot impact frequency curve.

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        graph = plot.Graph2D(self.label)
        graph.add_subplot('Return period (year)', 'Impact (%s)' % self.unit)
        graph.add_curve(self.return_per, self.impact, 'y')
        return graph.get_elems()

    def plot_compare(self, ifc):
        """Plot current and input impact frequency curves in a figure.

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        if self.unit != ifc.unit:
            LOGGER.warning("Comparing between two different units: %s and %s",\
                         self.unit, ifc.unit)
        graph = plot.Graph2D('', 2)
        graph.add_subplot('Return period (year)', 'Impact (%s)' % self.unit)
        graph.add_curve(self.return_per, self.impact, 'b', label=self.label)
        graph.add_curve(ifc.return_per, ifc.impact, 'r', label=ifc.label)
        return graph.get_elems()

class Impact(object):
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes:
        exposures_tag (Tag): information about the exposures
        impact_funcs_tag (Tag): information about the impact functions
        hazard_tag (TagHazard): information about the hazard
        event_id (np.array): id (>0) of each hazard event
        event_name (list): name of each hazard event
        coord_exp (GridPoints): exposures GridPoints (in degrees)
        eai_exp (np.array): expected annual impact for each exposure
        at_event (np.array): impact for each hazard event
        frequency (np.arrray): annual frequency of event
        tot_value (float): total exposure value affected
        aai_agg (float): average annual impact (aggregated)
        unit (str): value unit used (given by exposures unit)
    """

    def __init__(self):
        """ Empty initialization."""
        self.exposures_tag = Tag()
        self.impact_funcs_tag = Tag()
        self.hazard_tag = TagHazard()
        self.event_id = np.array([], int)
        self.event_name = list()
        self.date = np.array([], int)
        self.coord_exp = GridPoints()
        self.eai_exp = np.array([])
        self.at_event = np.array([])
        self.frequency = np.array([])
        self.tot_value = 0
        self.aai_agg = 0
        self.unit = 'NA'

    def calc_freq_curve(self):
        """Compute impact exceedance frequency curve.

        Returns:
            ImpactFreqCurve
        """
        ifc = ImpactFreqCurve()
        # Sort descendingly the impacts per events
        sort_idxs = np.argsort(self.at_event)[::-1]
        # Calculate exceedence frequency
        exceed_freq = np.cumsum(self.frequency[sort_idxs])
        # Set return period and imact exceeding frequency
        ifc.return_per = 1/exceed_freq
        ifc.impact = self.at_event[sort_idxs]
        ifc.unit = self.unit
        ifc.label = 'Exceedance frequency curve'
        return ifc

    def calc(self, exposures, impact_funcs, hazard):
        """Compute impact of an hazard to exposures.

        Parameters:
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncSet): impact functions
            hazard (Hazard): hazard

        Examples:
            Use Entity class:

            >>> hazard = Hazard(HAZ_DEMO_MAT) # Set hazard
            >>> entity = Entity() # Load entity with default values
            >>> entity.exposures = Exposures(ENT_TEMPLATE_XLS) # Set exposures
            >>> tc_impact = Impact()
            >>> tc_impact.calc(entity.exposures, entity.impact_functs, hazard)

            Specify only exposures and impact functions:

            >>> hazard = Hazard(HAZ_DEMO_MAT) # Set hazard
            >>> funcs = ImpactFuncSet(ENT_TEMPLATE_XLS) # Set impact functions
            >>> exposures = Exposures(ENT_TEMPLATE_XLS) # Set exposures
            >>> tc_impact = Impact()
            >>> tc_impact.calc(exposures, funcs, hazard)
        """
        # 1. Assign centroids to each exposure if not done
        if (not exposures.assigned) or \
        (hazard.tag.haz_type not in exposures.assigned):
            exposures.assign(hazard)

        # 2. Initialize values
        self.unit = exposures.value_unit
        self.event_id = hazard.event_id
        self.event_name = hazard.event_name
        self.date = hazard.date
        self.coord_exp = exposures.coord
        self.frequency = hazard.frequency
        self.at_event = np.zeros(hazard.intensity.shape[0])
        self.eai_exp = np.zeros(len(exposures.value))
        self.tot_value = 0
        self.exposures_tag = exposures.tag
        self.impact_funcs_tag = impact_funcs.tag
        self.hazard_tag = hazard.tag
        # Select exposures with positive value and assigned centroid
        exp_idx = np.where(np.logical_and(exposures.value > 0, \
                           exposures.assigned[hazard.tag.haz_type] >= 0))[0]
        # Warning if no exposures selected
        if exp_idx.size == 0:
            LOGGER.warning("No affected exposures.")
            return
        LOGGER.info('Calculating damage for %s assets (>0) and %s events.',
                    exp_idx.size, hazard.event_id.size)

        # Get hazard type
        haz_type = hazard.tag.haz_type
        # Get damage functions for this hazard
        haz_imp = impact_funcs.get_func(haz_type)

        # Check if deductible and cover should be applied
        insure_flag = False
        if exposures.deductible.size and exposures.cover.size:
            insure_flag = True
        num_events = hazard.intensity.shape[0]
        # 3. Loop over exposures according to their impact function
        # Loop over impact functions
        for imp_fun in haz_imp:
            self.imp_fun = imp_fun
            # get indices of all the exposures with this impact function
            exp_iimp = np.where(exposures.impact_id[exp_idx] == imp_fun.id)[0]
            exp_step = int(MAX_SIZE/num_events)
            # separte in chunks if too many exposures
            i = -1
            for i in range(int(exp_iimp.size/exp_step)):
                self._exp_impact(exp_idx[exp_iimp[i*exp_step:(i+1)*exp_step]],\
                    exposures, hazard, imp_fun, insure_flag)
            self._exp_impact(exp_idx[exp_iimp[(i+1)*exp_step:]],\
                exposures, hazard, imp_fun, insure_flag)

        self.aai_agg = sum(self.at_event * hazard.frequency)

    def _exp_impact(self, exp_iimp, exposures, hazard, imp_fun, insure_flag):
        """Compute impact for inpute exposure indexes and impact function.

        Parameters:
            exp_iimp (np.array): exposures indexes
            exposures (Exposures): exposures instance
            hazard (Hazard): hazard instance
            imp_fun (ImpactFunc): impact function instance
            insure_flag (bool): consider deductible and cover of exposures
        """
        # get assigned centroids
        icens = exposures.assigned[hazard.tag.haz_type][exp_iimp]

        # get affected intensities
        inten_val = hazard.intensity[:, icens].todense()
        # get affected fractions
        fract = hazard.fraction[:, icens]
        impact = fract.multiply(imp_fun.calc_mdr(inten_val)). \
            multiply(exposures.value[exp_iimp])

        if insure_flag and impact.nonzero()[0].size:
            paa = np.interp(inten_val, imp_fun.intensity, imp_fun.paa)
            impact = np.minimum(np.maximum(impact - \
                exposures.deductible[exp_iimp] * paa, 0), \
                exposures.cover[exp_iimp])
            self.eai_exp[exp_iimp] += np.sum(np.asarray(impact) * \
                hazard.frequency.reshape(-1, 1), axis=0)
        else:
            self.eai_exp[exp_iimp] += np.squeeze(np.asarray(np.sum( \
                impact.multiply(hazard.frequency.reshape(-1, 1)), axis=0)))

        self.at_event += np.squeeze(np.asarray(np.sum(impact, axis=1)))
        self.tot_value += np.sum(exposures.value[exp_iimp])

    def plot_eai_exposure(self, ignore_zero=True, pop_name=True,
                          buffer_deg=0.0, extend='neither', var_name=None,
                          **kwargs):
        """Plot expected annual impact of each exposure.

        Parameters:
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer_deg (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            var_name (str, optional): Colorbar label
            kwargs (optional): arguments for hexbin matplotlib function

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        title = 'Expected annual impact'
        if var_name is None:
            var_name = 'Impact (' + self.unit + ')'
        if ignore_zero:
            pos_vals = self.eai_exp > 0
        else:
            pos_vals = np.ones((self.eai_exp.size,), dtype=bool)
        if 'reduce_C_function' not in kwargs:
            kwargs['reduce_C_function'] = np.sum
        return plot.geo_bin_from_array(self.eai_exp[pos_vals], \
            self.coord_exp[pos_vals], var_name, title, pop_name, buffer_deg, \
            extend, **kwargs)
