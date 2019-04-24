"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

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

import ast
import logging
import csv
from itertools import zip_longest
import numpy as np
from scipy import sparse
import pandas as pd
import xlsxwriter
import datetime as dt

from climada.entity.tag import Tag
from climada.entity.exposures.base import Exposures, DEF_CRS
from climada.hazard.tag import Tag as TagHaz
from climada.entity.exposures.base import INDICATOR_IF, INDICATOR_CENTR
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
        coord_exp (np.ndarray): exposures coordinates [lat, lon] (in degrees)
        eai_exp (np.array): expected annual impact for each exposure
        at_event (np.array): impact for each hazard event
        frequency (np.arrray): annual frequency of event
        tot_value (float): total exposure value affected
        aai_agg (float): average annual impact (aggregated)
        unit (str): value unit used (given by exposures unit)
        imp_mat (sparse.csr_matrix): matrix num_events x num_exp with impacts.
            only filled if save_mat is True in calc()
    """

    def __init__(self):
        """ Empty initialization."""
        self.tag = dict()
        self.event_id = np.array([], int)
        self.event_name = list()
        self.date = np.array([], int)
        self.coord_exp = np.ndarray([], float)
        self.crs = DEF_CRS
        self.eai_exp = np.array([])
        self.at_event = np.array([])
        self.frequency = np.array([])
        self.tot_value = 0
        self.aai_agg = 0
        self.unit = ''
        self.imp_mat = []

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

    def calc(self, exposures, impact_funcs, hazard, save_mat=False):
        """Compute impact of an hazard to exposures.

        Parameters:
            exposures (Exposures): exposures
            impact_funcs (ImpactFuncSet): impact functions
            hazard (Hazard): hazard
            self_mat (bool): self impact matrix: events x exposures

        Examples:
            Use Entity class:

            >>> haz = Hazard('TC') # Set hazard
            >>> haz.read_mat(HAZ_DEMO_MAT)
            >>> haz.check()
            >>> ent = Entity() # Load entity with default values
            >>> ent.read_excel(ENT_TEMPLATE_XLS) # Set exposures
            >>> ent.check()
            >>> imp = Impact()
            >>> imp.calc(ent.exposures, ent.impact_funcs, haz)
            >>> imp.calc_freq_curve().plot()

            Specify only exposures and impact functions:

            >>> haz = Hazard('TC') # Set hazard
            >>> haz.read_mat(HAZ_DEMO_MAT)
            >>> haz.check()
            >>> funcs = ImpactFuncSet()
            >>> funcs.read_excel(ENT_TEMPLATE_XLS) # Set impact functions
            >>> funcs.check()
            >>> exp = Exposures(pd.read_excel(ENT_TEMPLATE_XLS)) # Set exposures
            >>> exp.check()
            >>> imp = Impact()
            >>> imp.calc(exp, funcs, haz)
            >>> imp.aai_agg
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
        self.coord_exp = np.stack([exposures.latitude.values,
                                   exposures.longitude.values], axis=1)
        self.frequency = hazard.frequency
        self.at_event = np.zeros(hazard.intensity.shape[0])
        self.eai_exp = np.zeros(exposures.value.size)
        self.tag = {'exp': exposures.tag, 'if_set': impact_funcs.tag,
                    'haz': hazard.tag}
        self.crs = exposures.crs

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
        if if_haz not in exposures and INDICATOR_IF not in exposures:
            LOGGER.error('Missing exposures impact functions %s.', INDICATOR_IF)
            raise ValueError
        if if_haz not in exposures:
            LOGGER.info('Missing exposures impact functions for hazard %s. ' +\
                        'Using impact functions in %s.', if_haz, INDICATOR_IF)
            if_haz = INDICATOR_IF

        # Check if deductible and cover should be applied
        insure_flag = False
        if ('deductible' in exposures) and ('cover' in exposures) \
        and exposures.cover.max():
            insure_flag = True

        if save_mat:
            self.imp_mat = sparse.lil_matrix((self.date.size, exposures.value.size))

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

        if save_mat:
            self.imp_mat = self.imp_mat.tocsr()

    def plot_hexbin_eai_exposure(self, mask=None, ignore_zero=True,
                                 pop_name=True, buffer=0.0, extend='neither',
                                 **kwargs):
        """Plot hexbin expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            kwargs (optional): arguments for hexbin matplotlib function

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        fig, axes = eai_exp.plot_hexbin(mask, ignore_zero, pop_name, buffer,
                                        extend, **kwargs)
        axes[0, 0].set_title('Expected annual impact')
        return fig, axes

    def plot_scatter_eai_exposure(self, mask=None, ignore_zero=True,
                                  pop_name=True, buffer=0.0, extend='neither',
                                  **kwargs):
        """Plot scatter expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates.
                Default: 1.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            kwargs (optional): arguments for hexbin matplotlib function

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        fig, axes = eai_exp.plot_scatter(mask, ignore_zero, pop_name, buffer,
                                         extend, **kwargs)
        axes[0, 0].set_title('Expected annual impact')
        return fig, axes

    def plot_raster_eai_exposure(self, res=None, raster_res=None, save_tiff=None,
                                 raster_f=lambda x: np.log10((np.fmax(x+1, 1))),
                                 label='value (log10)', **kwargs):
        """Plot raster expected annual impact of each exposure.

        Parameters:
            res (float, optional): resolution of current data in units of latitude
                and longitude, approximated if not provided.
            raster_res (float, optional): desired resolution of the raster
            save_tiff (str, optional): file name to save the raster in tiff
                format, if provided
            raster_f (lambda function): transformation to use to data. Default:
                log10 adding 1.
            label (str): colorbar label
            kwargs (optional): arguments for imshow matplotlib function

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        fig, axes = eai_exp.plot_raster(res, raster_res, save_tiff, raster_f,
                                        label, **kwargs)
        axes[0, 0].set_title('Expected annual impact')
        return fig, axes

    def plot_basemap_eai_exposure(self, mask=None, ignore_zero=False, pop_name=True,
                                  buffer=0.0, extend='neither', zoom=10,
                                  url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png',
                                  **kwargs):
        """Plot basemap expected annual impact of each exposure.

        Parameters:
            mask (np.array, optional): mask to apply to eai_exp plotted.
            ignore_zero (bool, optional): flag to indicate if zero and negative
                values are ignored in plot. Default: False
            pop_name (bool, optional): add names of the populated places
            buffer (float, optional): border to add to coordinates. Default: 0.0.
            extend (str, optional): extend border colorbar with arrows.
                [ 'neither' | 'both' | 'min' | 'max' ]
            zoom (int, optional): zoom coefficient used in the satellite image
            url (str, optional): image source, e.g. ctx.sources.OSM_C
            kwargs (optional): arguments for scatter matplotlib function, e.g.
                cmap='Greys'. Default: 'Wistia'

         Returns:
            matplotlib.figure.Figure, cartopy.mpl.geoaxes.GeoAxesSubplot
        """
        eai_exp = self._build_exp()
        fig, axes = eai_exp.plot_basemap(mask, ignore_zero, pop_name, buffer,
                                         extend, zoom, url, **kwargs)
        axes[0, 0].set_title('Expected annual impact')
        return fig, axes

    def write_csv(self, file_name):
        """ Write data into csv file. imp_mat is not saved.

        Parameters:
            file_name (str): absolute path of the file
        """
        with open(file_name, "w") as imp_file:
            imp_wr = csv.writer(imp_file)
            imp_wr.writerow(["tag_hazard", "tag_exposure", "tag_impact_func",
                             "unit", "tot_value", "aai_agg", "event_id",
                             "event_name", "event_date", "event_frequency",
                             "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"])
            csv_data = [[[self.tag['haz'].haz_type], [self.tag['haz'].file_name],
                         [self.tag['haz'].description]],
                        [[self.tag['exp'].file_name], [self.tag['exp'].description]],
                        [[self.tag['if_set'].file_name], [self.tag['if_set'].description]],
                        [self.unit], [self.tot_value], [self.aai_agg],
                        self.event_id, self.event_name, self.date,
                        self.frequency, self.at_event,
                        self.eai_exp, self.coord_exp[:, 0], self.coord_exp[:, 1],
                        [str(self.crs)]]
            for values in zip_longest(*csv_data):
                imp_wr.writerow(values)

    def write_excel(self, file_name):
        """ Write data into Excel file. imp_mat is not saved.

        Parameters:
            file_name (str): absolute path of the file
        """
        def write_col(i_col, imp_ws, xls_data):
            """ Write one measure """
            row_ini = 1
            for dat_row in xls_data:
                imp_ws.write(row_ini, i_col, dat_row)
                row_ini += 1

        imp_wb = xlsxwriter.Workbook(file_name)
        imp_ws = imp_wb.add_worksheet()

        header = ["tag_hazard", "tag_exposure", "tag_impact_func",
                  "unit", "tot_value", "aai_agg", "event_id",
                  "event_name", "event_date", "event_frequency",
                  "at_event", "eai_exp", "exp_lat", "exp_lon", "exp_crs"]
        for icol, head_dat in enumerate(header):
            imp_ws.write(0, icol, head_dat)
        data = [self.tag['haz'].haz_type, str(self.tag['haz'].file_name),
                str(self.tag['haz'].description)]
        write_col(0, imp_ws, data)
        data = [str(self.tag['exp'].file_name), str(self.tag['exp'].description)]
        write_col(1, imp_ws, data)
        data = [str(self.tag['if_set'].file_name), str(self.tag['if_set'].description)]
        write_col(2, imp_ws, data)
        write_col(3, imp_ws, [self.unit])
        write_col(4, imp_ws, [self.tot_value])
        write_col(5, imp_ws, [self.aai_agg])
        write_col(6, imp_ws, self.event_id)
        write_col(7, imp_ws, self.event_name)
        write_col(8, imp_ws, self.date)
        write_col(9, imp_ws, self.frequency)
        write_col(10, imp_ws, self.at_event)
        write_col(11, imp_ws, self.eai_exp)
        write_col(12, imp_ws, self.coord_exp[:, 0])
        write_col(13, imp_ws, self.coord_exp[:, 1])
        write_col(14, imp_ws, [str(self.crs)])

        imp_wb.close()

    def write_sparse_csr(self, file_name):
        """ Write imp_mat matrix in numpy's npz format."""
        np.savez(file_name, data=self.imp_mat.data, indices=self.imp_mat.indices,
                 indptr=self.imp_mat.indptr, shape=self.imp_mat.shape)
        
    def calc_impact_year_set(self, all_years=True):
        """ Calculate yearly impact from impact data.

        Parameters:
            all_years (boolean): return values for all years between first and
            last year with event, including years without any events.
    
        Returns:
             Impact year set of type numpy.ndarray with summed impact per year.
        """
        orig_year = np.array([dt.datetime.fromordinal(date).year
                      for date in self.date])
        if all_years:
            years = np.arange(min(orig_year), max(orig_year)+1)
        else:
            years = np.array(sorted(np.unique(orig_year)))
        year_set = dict()
        for year in years:
            year_set[year] = sum(self.at_event[orig_year==year])
        return year_set

    @staticmethod
    def read_sparse_csr(file_name):
        """ Read imp_mat matrix from numpy's npz format.

        Parameters:
            file_name (str): file name

        Returns:
            sparse.csr_matrix
        """
        loader = np.load(file_name)
        return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                                 shape=loader['shape'])

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
        self.crs = ast.literal_eval(imp_df.exp_crs.values[0])
        self.tag['haz'] = TagHaz(str(imp_df.tag_hazard[0]),
                                 str(imp_df.tag_hazard[1]),
                                 str(imp_df.tag_hazard[2]))
        self.tag['exp'] = Tag(str(imp_df.tag_exposure[0]),
                              str(imp_df.tag_exposure[1]))
        self.tag['if_set'] = Tag(str(imp_df.tag_impact_func[0]),
                                 str(imp_df.tag_impact_func[1]))

    def read_excel(self, file_name):
        """ Read excel file containing impact data generated by write_excel.

        Parameters:
            file_name (str): absolute path of the file
        """
        dfr = pd.read_excel(file_name)
        self.__init__()
        self.tag['haz'] = TagHaz()
        self.tag['haz'].haz_type = dfr['tag_hazard'][0]
        self.tag['haz'].file_name = dfr['tag_hazard'][1]
        self.tag['haz'].description = dfr['tag_hazard'][2]
        self.tag['exp'] = Tag()
        self.tag['exp'].file_name = dfr['tag_exposure'][0]
        self.tag['exp'].description = dfr['tag_exposure'][1]
        self.tag['if_set'] = Tag()
        self.tag['if_set'].file_name = dfr['tag_impact_func'][0]
        self.tag['if_set'].description = dfr['tag_impact_func'][1]

        self.unit = dfr.unit[0]
        self.tot_value = dfr.tot_value[0]
        self.aai_agg = dfr.aai_agg[0]

        self.event_id = dfr.event_id[~np.isnan(dfr.event_id.values)].values
        self.event_name = dfr.event_name[:self.event_id.size].values
        self.date = dfr.event_date[:self.event_id.size].values
        self.frequency = dfr.event_frequency[:self.event_id.size].values
        self.at_event = dfr.at_event[:self.event_id.size].values

        self.eai_exp = dfr.eai_exp[~np.isnan(dfr.eai_exp.values)].values
        self.coord_exp = np.zeros((self.eai_exp.size, 2))
        self.coord_exp[:, 0] = dfr.exp_lat.values[:self.eai_exp.size]
        self.coord_exp[:, 1] = dfr.exp_lon.values[:self.eai_exp.size]
        self.crs = ast.literal_eval(dfr.exp_crs.values[0])

    def _exp_impact(self, exp_iimp, exposures, hazard, imp_fun, insure_flag):
        """Compute impact for inpute exposure indexes and impact function.

        Parameters:
            exp_iimp (np.array): exposures indexes
            exposures (Exposures): exposures instance
            hazard (Hazard): hazard instance
            imp_fun (ImpactFunc): impact function instance
            insure_flag (bool): consider deductible and cover of exposures
        """
        if not exp_iimp.size:
            return

        # get assigned centroids
        icens = exposures[INDICATOR_CENTR + hazard.tag.haz_type].values[exp_iimp]

        # get affected intensities
        inten_val = hazard.intensity[:, icens]
        # get affected fractions
        fract = hazard.fraction[:, icens]
        # impact = fraction * mdr * value
        inten_val.data = imp_fun.calc_mdr(inten_val.data)
        impact = fract.multiply(inten_val).multiply(exposures.value.values[exp_iimp])

        if insure_flag and impact.nonzero()[0].size:
            inten_val = hazard.intensity[:, icens].todense()
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
        if not isinstance(self.imp_mat, list):
            self.imp_mat[:, exp_iimp] = impact

    def _build_exp(self):
        eai_exp = Exposures()
        eai_exp['value'] = self.eai_exp
        eai_exp['latitude'] = self.coord_exp[:, 0]
        eai_exp['longitude'] = self.coord_exp[:, 1]
        eai_exp.crs = self.crs
        eai_exp.value_unit = self.unit
        eai_exp.check()
        return eai_exp

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
