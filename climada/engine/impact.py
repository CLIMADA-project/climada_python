"""
This file is part of CLIMADA.

Copyright (C) 2017 CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Impact and ImpactFreqCurve classes.
"""

__all__ = ['ImpactFreqCurve', 'Impact']

import logging
import csv
from itertools import zip_longest
import numpy as np
import pandas as pd

from climada.entity.tag import Tag
from climada.hazard.tag import Tag as TagHaz
from climada.entity.exposures.base import INDICATOR_IF, INDICATOR_CENTR
from climada.util.coordinates import GridPoints
import climada.util.plot as u_plot
from climada.util.config import CONFIG

LOGGER = logging.getLogger(__name__)

class Impact():
    """Impact definition. Compute from an entity (exposures and impact
    functions) and hazard.

    Attributes:
        tag (dict): dictionary of tags of exposures, impact functions set and
            hazard: {'exp': Tag(), 'if_set': Tag(), 'haz': TagHazard()}
        event_id (np.array): id (>0) of each hazard event
        event_name (list): name of each hazard event
        date (np.array): date of events
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
        self.tag = dict()
        self.event_id = np.array([], int)
        self.event_name = list()
        self.date = np.array([], int)
        self.coord_exp = GridPoints()
        self.eai_exp = np.array([])
        self.at_event = np.array([])
        self.frequency = np.array([])
        self.tot_value = 0
        self.aai_agg = 0
        self.unit = ''

    def calc_freq_curve(self, return_per=None):
        """Compute impact exceedance frequency curve.

        Parameters:
            return_per (np.array, optional): return periods where to compute
                the exceedance impact. Use impact's frequencies if not provided

        Returns:
            ImpactFreqCurve
        """
        ifc = ImpactFreqCurve()
        ifc.tag = self.tag
        # Sort descendingly the impacts per events
        sort_idxs = np.argsort(self.at_event)[::-1]
        # Calculate exceedence frequency
        exceed_freq = np.cumsum(self.frequency[sort_idxs])
        # Set return period and imact exceeding frequency
        ifc.return_per = 1/exceed_freq[::-1]
        ifc.impact = self.at_event[sort_idxs][::-1]
        ifc.unit = self.unit
        ifc.label = 'Exceedance frequency curve'

        if return_per is not None:
            interp_imp = np.interp(return_per, ifc.return_per, ifc.impact)
            ifc.return_per = return_per
            ifc.impact = interp_imp

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
        assign_haz = INDICATOR_CENTR + hazard.tag.haz_type
        if assign_haz not in exposures:
            exposures.assign_centroids(hazard)
        else:
            LOGGER.info('Exposures matching centroids found in %s', assign_haz)

        # 2. Initialize values
        self.unit = exposures.value_unit
        self.event_id = hazard.event_id
        self.event_name = hazard.event_name
        self.date = hazard.date
        self.coord_exp = np.zeros((exposures.shape[0], 2))
        self.coord_exp[:, 0] = exposures.latitude.values
        self.coord_exp[:, 1] = exposures.longitude.values
        self.frequency = hazard.frequency
        self.at_event = np.zeros(hazard.intensity.shape[0])
        self.eai_exp = np.zeros(exposures.value.size)
        self.tag = {'exp': exposures.tag, 'if_set': impact_funcs.tag,
                    'haz': hazard.tag}

        # Select exposures with positive value and assigned centroid
        exp_idx = np.where(np.logical_and(exposures.value > 0, \
                           exposures[assign_haz] >= 0))[0]
        if exp_idx.size == 0:
            LOGGER.warning("No affected exposures.")

        num_events = hazard.intensity.shape[0]
        LOGGER.info('Calculating damage for %s assets (>0) and %s events.',
                    exp_idx.size, num_events)

        # Get damage functions for this hazard
        if_haz = INDICATOR_IF + hazard.tag.haz_type
        haz_imp = impact_funcs.get_func(hazard.tag.haz_type)
        if if_haz not in exposures:
            LOGGER.error('Missing exposures column %s. No exposures with impact'\
                         +' functions for peril %s.', if_haz, hazard.tag.haz_type)
            raise ValueError

        # Check if deductible and cover should be applied
        insure_flag = False
        if ('deductible' in exposures) and ('cover' in exposures) \
        and exposures.cover.max():
            insure_flag = True

        # 3. Loop over exposures according to their impact function
        tot_exp = 0
        for imp_fun in haz_imp:
            # get indices of all the exposures with this impact function
            exp_iimp = np.where(exposures[if_haz].values[exp_idx] == imp_fun.id)[0]
            tot_exp += exp_iimp.size
            exp_step = int(CONFIG['global']['max_matrix_size']/num_events)
            if not exp_step:
                LOGGER.error('Increase max_matrix_size configuration parameter'
                             ' to > %s', str(num_events))
                raise ValueError
            # separte in chunks
            chk = -1
            for chk in range(int(exp_iimp.size/exp_step)):
                self._exp_impact( \
                    exp_idx[exp_iimp[chk*exp_step:(chk+1)*exp_step]],\
                    exposures, hazard, imp_fun, insure_flag)
            self._exp_impact(exp_idx[exp_iimp[(chk+1)*exp_step:]],\
                exposures, hazard, imp_fun, insure_flag)

        if not tot_exp:
            LOGGER.warning('No impact functions match the exposures.')
        self.aai_agg = sum(self.at_event * hazard.frequency)

    def plot_eai_exposure(self, mask=None, ignore_zero=True,
                          pop_name=True, buffer_deg=0.0, extend='neither',
                          var_name=None, **kwargs):
        """Plot expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
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
        if mask is None:
            mask = np.ones((self.eai_exp.size,), dtype=bool)
        if ignore_zero:
            pos_vals = self.eai_exp[mask] > 0
        else:
            pos_vals = np.ones((self.eai_exp[mask].size,), dtype=bool)
        if 'reduce_C_function' not in kwargs:
            kwargs['reduce_C_function'] = np.sum
        return u_plot.geo_bin_from_array(self.eai_exp[mask][pos_vals], \
            self.coord_exp[mask][pos_vals], var_name, title, pop_name, \
            buffer_deg, extend, **kwargs)

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
        icens = exposures[INDICATOR_CENTR + hazard.tag.haz_type].values[exp_iimp]

        # get affected intensities
        inten_val = hazard.intensity[:, icens].todense()
        # get affected fractions
        fract = hazard.fraction[:, icens]
        # impact = fraction * mdr * value
        impact = fract.multiply(imp_fun.calc_mdr(inten_val)). \
            multiply(exposures.value.values[exp_iimp])

        if insure_flag and impact.nonzero()[0].size:
            paa = np.interp(inten_val, imp_fun.intensity, imp_fun.paa)
            impact = np.minimum(np.maximum(impact - \
                exposures.deductible.values[exp_iimp] * paa, 0), \
                exposures.cover.values[exp_iimp])
            self.eai_exp[exp_iimp] += np.sum(np.asarray(impact) * \
                hazard.frequency.reshape(-1, 1), axis=0)
        else:
            self.eai_exp[exp_iimp] += np.squeeze(np.asarray(np.sum( \
                impact.multiply(hazard.frequency.reshape(-1, 1)), axis=0)))

        self.at_event += np.squeeze(np.asarray(np.sum(impact, axis=1)))
        self.tot_value += np.sum(exposures.value.values[exp_iimp])

    def write_csv(self, file_name):
        """ Write data into csv file.

        Parameters:
            file_name (str): absolute path of the file
        """
        with open(file_name, "w") as imp_file:
            imp_wr = csv.writer(imp_file)
            imp_wr.writerow(["tag_hazard", "tag_exposure", "tag_impact_func",
                             "unit", "tot_value", "aai_agg", "event_id",
                             "event_name", "event_date", "event_frequency",
                             "at_event", "eai_exp", "exp_lat", "exp_lon"])
            csv_data = [[[self.tag['haz'].haz_type], [self.tag['haz'].file_name],
                         [self.tag['haz'].description]],
                        [[self.tag['exp'].file_name], [self.tag['exp'].description]],
                        [[self.tag['if_set'].file_name], [self.tag['if_set'].description]],
                        [self.unit], [self.tot_value], [self.aai_agg],
                        self.event_id, self.event_name, self.date,
                        self.frequency, self.at_event,
                        self.eai_exp, self.coord_exp[:, 0], self.coord_exp[:, 1]]
            for values in zip_longest(*csv_data):
                imp_wr.writerow(values)

    def read_csv(self, file_name):
        """ Read csv file containing impact data generated by write_csv.

        Parameters:
            file_name (str): absolute path of the file
        """
        imp_df = pd.read_csv(file_name)
        self.__init__()
        self.unit = imp_df.unit[0]
        self.tot_value = imp_df.tot_value[0]
        self.aai_agg = imp_df.aai_agg[0]
        self.event_id = imp_df.event_id[~np.isnan(imp_df.event_id)]
        num_ev = self.event_id.size
        self.event_name = imp_df.event_name[:num_ev]
        self.date = imp_df.event_date[:num_ev]
        self.at_event = imp_df.at_event[:num_ev]
        self.frequency = imp_df.event_frequency[:num_ev]
        self.eai_exp = imp_df.eai_exp[~np.isnan(imp_df.eai_exp)]
        num_exp = self.eai_exp.size
        self.coord_exp = np.zeros((num_exp, 2))
        self.coord_exp[:, 0] = imp_df.exp_lat[:num_exp]
        self.coord_exp[:, 1] = imp_df.exp_lon[:num_exp]
        self.tag['haz'] = TagHaz(str(imp_df.tag_hazard[0]),
                                 str(imp_df.tag_hazard[1]),
                                 str(imp_df.tag_hazard[2]))
        self.tag['exp'] = Tag(str(imp_df.tag_exposure[0]),
                              str(imp_df.tag_exposure[1]))
        self.tag['if_set'] = Tag(str(imp_df.tag_impact_func[0]),
                                 str(imp_df.tag_impact_func[1]))

    @property
    def coord_exp(self):
        """ Return coord"""
        return self._coord_exp

    @coord_exp.setter
    def coord_exp(self, value):
        """ If it is not a GridPoints instance, put it."""
        if not isinstance(value, GridPoints):
            self._coord_exp = GridPoints(value)
        else:
            self._coord_exp = value

class ImpactFreqCurve():
    """ Impact exceedence frequency curve.

    Attributes:
        tag (dict): dictionary of tags of exposures, impact functions set and
            hazard: {'exp': Tag(), 'if_set': Tag(), 'haz': TagHazard()}
        return_per (np.array): return period
        impact (np.array): impact exceeding frequency
        unit (str): value unit used (given by exposures unit)
        label (str): string describing source data
    """
    def __init__(self):
        self.tag = dict()
        self.return_per = np.array([])
        self.impact = np.array([])
        self.unit = ''
        self.label = ''

    def plot(self):
        """Plot impact frequency curve.

        Returns:
            matplotlib.figure.Figure, [matplotlib.axes._subplots.AxesSubplot]
        """
        graph = u_plot.Graph2D(self.label)
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
        graph = u_plot.Graph2D('', 2)
        graph.add_subplot('Return period (year)', 'Impact (%s)' % self.unit)
        graph.add_curve(self.return_per, self.impact, 'b', label=self.label)
        graph.add_curve(ifc.return_per, ifc.impact, 'r', label=ifc.label)
        return graph.get_elems()
