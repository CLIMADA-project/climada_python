"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define Measure class.
"""

__all__ = ['Measure']

import copy
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from climada.entity.exposures.base import Exposures, INDICATOR_IMPF, INDICATOR_CENTR
from climada.hazard.base import Hazard
import climada.util.checker as u_check

LOGGER = logging.getLogger(__name__)

IMPF_ID_FACT = 1000
"""Factor internally used as id for impact functions when region selected."""

NULL_STR = 'nil'
"""String considered as no path in measures exposures_set and hazard_set or
no string in imp_fun_map"""

class Measure():
    """
    Contains the definition of one measure.

    Attributes
    ----------
    name : str
        name of the measure
    haz_type : str
        related hazard type (peril), e.g. TC
    color_rgb : np.array
        integer array of size 3. Color code of this measure in RGB
    cost : float
        discounted cost (in same units as assets)
    hazard_set : str
        file name of hazard to use (in h5 format)
    hazard_freq_cutoff : float
        hazard frequency cutoff
    exposures_set : str  or climada.entity.Exposure
        file name of exposure to use (in h5 format) or Exposure instance
    imp_fun_map : str
        change of impact function id of exposures, e.g. '1to3'
    hazard_inten_imp : tuple(float, float)
        parameter a and b of hazard intensity change
    mdd_impact : tuple(float, float)
        parameter a and b of the impact over the mean damage degree
    paa_impact : tuple(float, float)
        parameter a and b of the impact over the percentage of affected assets
    exp_region_id : int
        region id of the selected exposures to consider ALL the previous
        parameters
    risk_transf_attach : float
        risk transfer attachment
    risk_transf_cover : float
        risk transfer cover
    risk_transf_cost_factor : float
        factor to multiply to resulting insurance layer to get the total
        cost of risk transfer
    """

    def __init__(
        self,
        name: str = "",
        haz_type: str = "",
        cost: float = 0,
        hazard_set: str = NULL_STR,
        hazard_freq_cutoff: float = 0,
        exposures_set: str = NULL_STR,
        imp_fun_map: str = NULL_STR,
        hazard_inten_imp: Tuple[float, float] = (1, 0),
        mdd_impact: Tuple[float, float] = (1, 0),
        paa_impact: Tuple[float, float] = (1, 0),
        exp_region_id: Optional[list] = None,
        risk_transf_attach: float = 0,
        risk_transf_cover: float = 0,
        risk_transf_cost_factor: float = 1,
        color_rgb: Optional[np.ndarray] = None
    ):
        """Initialize a Measure object with given values.

        Parameters
        ----------
        name : str, optional
            name of the measure
        haz_type : str, optional
            related hazard type (peril), e.g. TC
        cost : float, optional
            discounted cost (in same units as assets)
        hazard_set : str, optional
            file name of hazard to use (in h5 format)
        hazard_freq_cutoff : float, optional
            hazard frequency cutoff
        exposures_set : str  or climada.entity.Exposure, optional
            file name of exposure to use (in h5 format) or Exposure instance
        imp_fun_map : str, optional
            change of impact function id of exposures, e.g. '1to3'
        hazard_inten_imp : tuple(float, float), optional
            parameter a and b of hazard intensity change
        mdd_impact : tuple(float, float), optional
            parameter a and b of the impact over the mean damage degree
        paa_impact : tuple(float, float), optional
            parameter a and b of the impact over the percentage of affected assets
        exp_region_id : int, optional
            region id of the selected exposures to consider ALL the previous
            parameters
        risk_transf_attach : float, optional
            risk transfer attachment
        risk_transf_cover : float, optional
            risk transfer cover
        risk_transf_cost_factor : float, optional
            factor to multiply to resulting insurance layer to get the total
            cost of risk transfer
        color_rgb : np.array, optional
            integer array of size 3. Color code of this measure in RGB.
            Default is None (corresponds to black).
        """
        self.name = name
        self.haz_type = haz_type
        self.color_rgb = np.array([0, 0, 0]) if color_rgb is None else color_rgb
        self.cost = cost

        # related to change in hazard
        self.hazard_set = hazard_set
        self.hazard_freq_cutoff = hazard_freq_cutoff

        # related to change in exposures
        self.exposures_set = exposures_set
        self.imp_fun_map = imp_fun_map

        # related to change in impact functions
        self.hazard_inten_imp = hazard_inten_imp
        self.mdd_impact = mdd_impact
        self.paa_impact = paa_impact

        # related to change in region
        self.exp_region_id = [] if exp_region_id is None else exp_region_id

        # risk transfer
        self.risk_transf_attach = risk_transf_attach
        self.risk_transf_cover = risk_transf_cover
        self.risk_transf_cost_factor = risk_transf_cost_factor

    def check(self):
        """
        Check consistent instance data.

        Raises
        ------
        ValueError
        """
        u_check.size([3, 4], self.color_rgb, 'Measure.color_rgb')
        u_check.size(2, self.hazard_inten_imp, 'Measure.hazard_inten_imp')
        u_check.size(2, self.mdd_impact, 'Measure.mdd_impact')
        u_check.size(2, self.paa_impact, 'Measure.paa_impact')

    def calc_impact(self, exposures, imp_fun_set, hazard, assign_centroids=True):
        """
        Apply measure and compute impact and risk transfer of measure
        implemented over inputs.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance
        imp_fun_set : climada.entity.ImpactFuncSet
            impact function set instance
        hazard : climada.hazard.Hazard
            hazard instance
        assign_centroids : bool, optional
            indicates whether centroids are assigned to the self.exposures object.
            Centroids assignment is an expensive operation; set this to ``False`` to save
            computation time if the hazards' centroids are already assigned to the exposures
            object.
            Default: True

        Returns
        -------
        climada.engine.Impact
            resulting impact and risk transfer of measure
        """

        new_exp, new_impfs, new_haz = self.apply(exposures, imp_fun_set, hazard)
        return self._calc_impact(new_exp, new_impfs, new_haz, assign_centroids)

    def apply(self, exposures, imp_fun_set, hazard):
        """
        Implement measure with all its defined parameters.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance
        imp_fun_set : climada.entity.ImpactFuncSet
            impact function set instance
        hazard : climada.hazard.Hazard
            hazard instance

        Returns
        -------
        new_exp : climada.entity.Exposure
            Exposure with implemented measure with all defined parameters
        new_ifs : climada.entity.ImpactFuncSet
            Impact function set with implemented measure with all defined parameters
        new_haz : climada.hazard.Hazard
            Hazard with implemented measure with all defined parameters
        """
        # change hazard
        new_haz = self._change_all_hazard(hazard)
        # change exposures
        new_exp = self._change_all_exposures(exposures)
        new_exp = self._change_exposures_impf(new_exp)
        # change impact functions
        new_impfs = self._change_imp_func(imp_fun_set)
        # cutoff events whose damage happen with high frequency (in region impf specified)
        new_haz = self._cutoff_hazard_damage(new_exp, new_impfs, new_haz)
        # apply all previous changes only to the selected exposures
        new_exp, new_impfs, new_haz = self._filter_exposures(
            exposures, imp_fun_set, hazard, new_exp, new_impfs, new_haz)

        return new_exp, new_impfs, new_haz

    def _calc_impact(self, new_exp, new_impfs, new_haz, assign_centroids):
        """Compute impact and risk transfer of measure implemented over inputs.

        Parameters
        ----------
        new_exp : climada.entity.Exposures
            exposures once measure applied
        new_ifs : climada.entity.ImpactFuncSet
            impact function set once measure applied
        new_haz  : climada.hazard.Hazard
            hazard once measure applied

        Returns
        -------
        climada.engine.Impact
        """
        from climada.engine.impact_calc import ImpactCalc  # pylint: disable=import-outside-toplevel
        imp = ImpactCalc(new_exp, new_impfs, new_haz)\
              .impact(save_mat=False, assign_centroids=assign_centroids)
        return imp.calc_risk_transfer(self.risk_transf_attach, self.risk_transf_cover)

    def _change_all_hazard(self, hazard):
        """
        Change hazard to provided hazard_set.

        Parameters
        ----------
        hazard : climada.hazard.Hazard
            hazard instance

        Returns
        -------
        new_haz : climada.hazard.Hazard
            Hazard
        """
        if self.hazard_set == NULL_STR:
            return hazard

        LOGGER.debug('Setting new hazard %s', self.hazard_set)
        new_haz = Hazard.from_hdf5(self.hazard_set)
        new_haz.check()
        return new_haz

    def _change_all_exposures(self, exposures):
        """
        Change exposures to provided exposures_set.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance

        Returns
        -------
        new_exp : climada.entity.Exposures()
            Exposures
        """
        if isinstance(self.exposures_set, str) and self.exposures_set == NULL_STR:
            return exposures

        if isinstance(self.exposures_set, (str, Path)):
            LOGGER.debug('Setting new exposures %s', self.exposures_set)
            new_exp = Exposures.from_hdf5(self.exposures_set)
            new_exp.check()
        elif isinstance(self.exposures_set, Exposures):
            LOGGER.debug('Setting new exposures. ')
            new_exp = self.exposures_set.copy(deep=True)
            new_exp.check()
        else:
            raise ValueError(f'{self.exposures_set} is neither a string nor an Exposures object')

        if not np.array_equal(np.unique(exposures.gdf.latitude.values),
                              np.unique(new_exp.gdf.latitude.values)) or \
        not np.array_equal(np.unique(exposures.gdf.longitude.values),
                           np.unique(new_exp.gdf.longitude.values)):
            LOGGER.warning('Exposures locations have changed.')

        return new_exp

    def _change_exposures_impf(self, exposures):
        """Change exposures impact functions ids according to imp_fun_map.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance

        Returns
        -------
        new_exp : climada.entity.Exposure
            Exposure with updated impact functions ids accordgin to
            impf_fun_map
        """
        if self.imp_fun_map == NULL_STR:
            return exposures

        LOGGER.debug('Setting new exposures impact functions%s', self.imp_fun_map)
        new_exp = exposures.copy(deep=True)
        from_id = int(self.imp_fun_map[0:self.imp_fun_map.find('to')])
        to_id = int(self.imp_fun_map[self.imp_fun_map.find('to') + 2:])
        try:
            exp_change = np.argwhere(
                new_exp.gdf[INDICATOR_IMPF + self.haz_type].values == from_id
            ).reshape(-1)
            new_exp.gdf[INDICATOR_IMPF + self.haz_type].values[exp_change] = to_id
        except KeyError:
            exp_change = np.argwhere(
                new_exp.gdf[INDICATOR_IMPF].values == from_id
            ).reshape(-1)
            new_exp.gdf[INDICATOR_IMPF].values[exp_change] = to_id
        return new_exp

    def _change_imp_func(self, imp_set):
        """
        Apply measure to impact functions of the same hazard type.

        Parameters
        ----------
        imp_set : climada.entity.ImpactFuncSet
            impact function set instance to be modified

        Returns
        -------
        new_imp_set : climada.entity.ImpactFuncSet
            ImpactFuncSet with measure applied to each impact function
            according to the defined hazard type
        """
        if self.hazard_inten_imp == (1, 0) and self.mdd_impact == (1, 0)\
        and self.paa_impact == (1, 0):
            return imp_set

        new_imp_set = copy.deepcopy(imp_set)
        for imp_fun in new_imp_set.get_func(self.haz_type):
            LOGGER.debug('Transforming impact functions.')
            imp_fun.intensity = np.maximum(
                imp_fun.intensity * self.hazard_inten_imp[0] - self.hazard_inten_imp[1], 0.0)
            imp_fun.mdd = np.maximum(
                imp_fun.mdd * self.mdd_impact[0] + self.mdd_impact[1], 0.0)
            imp_fun.paa = np.maximum(
                imp_fun.paa * self.paa_impact[0] + self.paa_impact[1], 0.0)

        if not new_imp_set.size():
            LOGGER.info('No impact function of hazard %s found.', self.haz_type)

        return new_imp_set

    def _cutoff_hazard_damage(self, exposures, impf_set, hazard):
        """Cutoff of hazard events which generate damage with a frequency higher
        than hazard_freq_cutoff.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            exposures instance
        imp_set : climada.entity.ImpactFuncSet
            impact function set instance
        hazard : climada.hazard.Hazard
            hazard instance

        Returns
        -------
        new_haz : climada.hazard.Hazard
            Hazard without events which generate damage with a frequency
            higher than hazard_freq_cutoff
        """
        if self.hazard_freq_cutoff == 0:
            return hazard

        if self.exp_region_id:
            # compute impact only in selected region
            in_reg = np.logical_or.reduce(
                [exposures.gdf.region_id.values == reg for reg in self.exp_region_id]
            )
            exp_imp = Exposures(exposures.gdf[in_reg], crs=exposures.crs)
        else:
            exp_imp = exposures

        from climada.engine.impact_calc import ImpactCalc  # pylint: disable=import-outside-toplevel
        imp = ImpactCalc(exp_imp, impf_set, hazard)\
              .impact(assign_centroids=hazard.centr_exp_col not in exp_imp.gdf)

        LOGGER.debug('Cutting events whose damage have a frequency > %s.',
                     self.hazard_freq_cutoff)
        new_haz = copy.deepcopy(hazard)
        sort_idxs = np.argsort(imp.at_event)[::-1]
        exceed_freq = np.cumsum(imp.frequency[sort_idxs])
        cutoff = exceed_freq > self.hazard_freq_cutoff
        sel_haz = sort_idxs[cutoff]
        for row in sel_haz:
            new_haz.intensity.data[new_haz.intensity.indptr[row]:
                                   new_haz.intensity.indptr[row + 1]] = 0
        new_haz.intensity.eliminate_zeros()
        return new_haz

    def _filter_exposures(self, exposures, imp_set, hazard, new_exp, new_impfs,
                          new_haz):
        """
        Incorporate changes of new elements to previous ones only for the
        selected exp_region_id. If exp_region_id is [], all new changes
        will be accepted.

        Parameters
        ----------
        exposures : climada.entity.Exposures
            old exposures instance
        imp_set :climada.entity.ImpactFuncSet
            old impact function set instance
        hazard : climada.hazard.Hazard
            old hazard instance
        new_exp : climada.entity.Exposures
            new exposures instance
        new_ifs : climada.entity.ImpactFuncSet
            new impact functions instance
        new_haz : climada.hazard.Hazard
            new hazard instance

        Returns
        -------
        new_exp,new_ifs, new_haz : climada.entity.Exposures,
                                   climada.entity.ImpactFuncSet,
                                   climada.hazard.Hazard
            Exposures, ImpactFuncSet, Hazard with incoporated elements
            for the selected exp_region_id.
        """
        if not self.exp_region_id:
            return new_exp, new_impfs, new_haz

        if exposures is new_exp:
            new_exp = exposures.copy(deep=True)

        if imp_set is not new_impfs:
            # provide new impact functions ids to changed impact functions
            fun_ids = list(new_impfs.get_func()[self.haz_type].keys())
            for key in fun_ids:
                new_impfs.get_func()[self.haz_type][key].id = key + IMPF_ID_FACT
                new_impfs.get_func()[self.haz_type][key + IMPF_ID_FACT] = \
                    new_impfs.get_func()[self.haz_type][key]
            try:
                new_exp.gdf[INDICATOR_IMPF + self.haz_type] += IMPF_ID_FACT
            except KeyError:
                new_exp.gdf[INDICATOR_IMPF] += IMPF_ID_FACT
            # collect old impact functions as well (used by exposures)
            new_impfs.get_func()[self.haz_type].update(imp_set.get_func()[self.haz_type])

        # get the indices for changing and inert regions
        chg_reg = exposures.gdf.region_id.isin(self.exp_region_id)
        no_chg_reg = ~chg_reg

        LOGGER.debug('Number of changed exposures: %s', chg_reg.sum())

        # concatenate previous and new exposures
        new_exp.set_gdf(
            GeoDataFrame(
                pd.concat([
                    exposures.gdf[no_chg_reg],  # old values for inert regions
                    new_exp.gdf[chg_reg]        # new values for changing regions
                ]).loc[exposures.gdf.index,:],  # re-establish old order
            ),
            crs=exposures.crs
        )

        # set missing values of centr_
        if INDICATOR_CENTR + self.haz_type in new_exp.gdf.columns \
            and np.isnan(new_exp.gdf[INDICATOR_CENTR + self.haz_type].values).any():
            new_exp.gdf.drop(columns=INDICATOR_CENTR + self.haz_type, inplace=True)
        elif INDICATOR_CENTR in new_exp.gdf.columns \
            and np.isnan(new_exp.gdf[INDICATOR_CENTR].values).any():
            new_exp.gdf.drop(columns=INDICATOR_CENTR, inplace=True)

        # put hazard intensities outside region to previous intensities
        if hazard is not new_haz:
            if INDICATOR_CENTR + self.haz_type in exposures.gdf.columns:
                centr = exposures.gdf[INDICATOR_CENTR + self.haz_type].values[chg_reg]
            elif INDICATOR_CENTR in exposures.gdf.columns:
                centr = exposures.gdf[INDICATOR_CENTR].values[chg_reg]
            else:
                exposures.assign_centroids(hazard)
                centr = exposures.gdf[INDICATOR_CENTR + self.haz_type].values[chg_reg]

            centr = np.delete(np.arange(hazard.intensity.shape[1]), np.unique(centr))
            new_haz_inten = new_haz.intensity.tolil()
            new_haz_inten[:, centr] = hazard.intensity[:, centr]
            new_haz.intensity = new_haz_inten.tocsr()

        return new_exp, new_impfs, new_haz
