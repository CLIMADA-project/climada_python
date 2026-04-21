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

from __future__ import annotations

import logging
from functools import reduce
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, cast

import numpy as np
import pandas as pd

from climada.entity.exposures.base import Exposures
from climada.entity.impact_funcs.base import ImpactFunc
from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
from climada.entity.measures.measure_config import (
    ExposuresModifierConfig,
    HazardModifierConfig,
    ImpfsetModifierConfig,
)
from climada.hazard.base import Hazard

if TYPE_CHECKING:
    from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
    from climada.entity.measures.types import (
        ExposuresChange,
        HazardChange,
        ImpfsetChange,
    )
    from climada.hazard.base import Hazard

    T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)

LOGGER = logging.getLogger(__name__)


def identity_function(x: T, **_kwargs: Any) -> T:
    """Return the input object unchanged.

    Parameters
    ----------
    x : T
        Object to return.
    **_kwargs : Any
        Accepted but ignored.

    Returns
    -------
    T
        The unchanged input object.
    """

    return x


def composite_fun(*funcs: Callable[..., T]) -> Callable[..., T]:
    """Compose multiple functions right-to-left into a single callable.

    Given functions ``f, g, h``, returns a function equivalent to
    ``lambda x, **kw: f(g(h(x, **kw), **kw), **kw)``.
    If no functions are provided, returns :func:`identity_function`.

    Parameters
    ----------
    *funcs : Callable[..., T]
        Functions to compose, applied from right to left.
        Each must accept an object of type ``T`` as its first positional
        argument and forward ``**kwargs``.

    Returns
    -------
    Callable[..., T]
        A single callable equivalent to the right-to-left composition of
        all provided functions.

    Examples
    --------
    >>> composed = composite_fun(f, g, h)
    >>> result = composed(x, year=2030)  # equivalent to f(g(h(x, year=2030), year=2030), year=2030)
    """

    def compose(f: Callable[..., T], g: Callable[..., T]) -> Callable[..., T]:
        def composed(x: T, **kwargs: Any) -> T:
            return f(g(x, **kwargs), **kwargs)

        return composed

    return reduce(compose, funcs, identity_function)


def replace_hazard(new_hazard: Hazard) -> HazardChange:
    """Return a change function that unconditionally replaces the hazard.

    The returned function ignores its input and always returns ``new_hazard``.
    Note that ``new_hazard`` is a shared reference; callers should ensure the
    object is not mutated after being passed here.

    Parameters
    ----------
    new_hazard : Hazard
        The hazard object to substitute in place of the original.

    Returns
    -------
    HazardChange
        A callable with signature ``(hazard: Hazard) -> Hazard``
        that discards its input and returns ``new_hazard``.
    """

    def hazard_change(_: Hazard) -> Hazard:
        return new_hazard

    return hazard_change


def impact_intensity_rp_cutoff_helper(
    cut_off_rp: float,
) -> HazardChange:
    """Return a change function that zeros out low-impact events based on a
    return period threshold.

    The returned function computes impacts on the *base* triplet
    (``base_exposures``, ``base_impfset``, ``base_hazard``) and identifies
    events whose cumulative exceedance frequency does not exceed ``1 /
    cut_off_rp``. The intensity rows of those events are set to zero in the
    hazard being transformed.

    Parameters
    ----------
    cut_off_rp : float
        Return period threshold in years. Events whose cumulative exceedance
        frequency is at or below ``1 / cut_off_rp`` are zeroed out.

    Returns
    -------
    HazardChange
        A callable with the following signature:

        .. code-block:: python

            def hazard_change(
                hazard: Hazard,
                base_exposures: Exposures,
                base_impfset: ImpactFuncSet,
                base_hazard: Hazard,
                exposures_region_id: Optional[list[int]] = None,
            ) -> Hazard

        Parameters of the returned callable:

        hazard : Hazard
            The hazard object to modify in-place (intensity rows zeroed).
        base_exposures : Exposures
            Exposures used for the reference impact computation.
        base_impfset : ImpactFuncSet
            Impact function set used for the reference impact computation.
        base_hazard : Hazard
            Original hazard used for the reference impact computation.
        exposures_region_id : list of int, optional
            If provided, the impact computation is restricted to exposure
            points whose ``region_id`` is in this list.

    Notes
    -----
    The exceedance frequency is computed on the *base* hazard, not on the
    already-modified hazard. This ensures the cutoff decision is always
    relative to the unmodified risk landscape.
    """

    from climada.engine.impact_calc import ImpactCalc

    def hazard_change(
        hazard: Hazard,
        base_exposures: Exposures,
        base_impfset: ImpactFuncSet,
        base_hazard: Hazard,
        exposures_region_id: Optional[list[int]] = None,
        **_kwargs,
    ) -> Hazard:
        exp_imp = base_exposures
        if exposures_region_id:
            # Narrowing the type for the LSP via boolean indexing
            in_reg = base_exposures.gdf["region_id"].isin(exposures_region_id)
            exp_imp = Exposures(base_exposures.gdf[in_reg], crs=base_exposures.crs)

        imp = ImpactCalc(exp_imp, base_impfset, base_hazard).impact(save_mat=False)

        # Calculate exceedance frequencies
        sort_idxs = np.argsort(imp.at_event)[::-1]
        exceed_freq = np.cumsum(imp.frequency[sort_idxs])
        events_below_cutoff = sort_idxs[exceed_freq <= 1 / cut_off_rp]

        # Modify sparse data structure
        intensity_modified = hazard.intensity.copy()
        for event in events_below_cutoff:
            start, end = (
                intensity_modified.indptr[event],
                intensity_modified.indptr[event + 1],
            )
            intensity_modified.data[start:end] = 0

        hazard.intensity = intensity_modified
        return hazard

    return hazard_change


def helper_hazard(hazard_modifier: HazardModifierConfig) -> HazardChange:
    """Return a change function that scales, shifts, and optionally
    replaces hazard intensities.

    Constructs a :class:`HazardChange` from a :class:`HazardModifierConfig`.
    The returned function optionally loads a new hazard from disk, then applies
    a linear transformation to all stored (non-zero) intensity values.

    If :attr:`~HazardModifierConfig.impact_rp_cutoff` is set, the returned
    function is further composed (via :func:`composite_fun`) with
    :func:`impact_intensity_rp_cutoff_helper` so that low-return-period events
    are zeroed out after the linear transformation.

    Parameters
    ----------
    hazard_modifier : HazardModifierConfig
        Configuration object specifying:

        - ``new_hazard_path`` : path to an HDF5 hazard file to load, or
          ``None`` to transform the input hazard in-place.
        - ``haz_int_mult`` : multiplicative factor applied to intensity data.
        - ``haz_int_add`` : additive shift applied to intensity data.
        - ``impact_rp_cutoff`` : return period threshold passed to
          :func:`impact_intensity_rp_cutoff_helper`, or ``None`` to skip.

    Returns
    -------
    HazardChange
        A callable with signature ``(hazard: Hazard, **kwargs) -> Hazard``.

    Notes
    -----
    The transformation is applied directly to the sparse matrix's ``.data``
    array, so only explicitly stored (non-zero) entries are affected.
    Structural zeros remain zero.
    """

    def hazard_change(hazard: Hazard, **_kwargs) -> Hazard:
        changed_hazard = (
            Hazard.from_hdf5(hazard_modifier.new_hazard_path)
            if hazard_modifier.new_hazard_path is not None
            else hazard
        )
        data = cast(np.ndarray, changed_hazard.intensity.data)
        data *= hazard_modifier.haz_int_mult
        data += hazard_modifier.haz_int_add
        data[data < 0] = 0
        changed_hazard.intensity.eliminate_zeros()
        return changed_hazard

    if hazard_modifier.impact_rp_cutoff is not None:
        hazard_change = composite_fun(
            impact_intensity_rp_cutoff_helper(hazard_modifier.impact_rp_cutoff),
            hazard_change,
        )

    return hazard_change


def helper_impfset(impfset_modifier: ImpfsetModifierConfig) -> ImpfsetChange:
    """Return a change function that applies linear modifications to selected
    impact functions.

    Constructs an :class:`ImpfsetChange` from an :class:`ImpfsetModifierConfig`.
    The returned function optionally loads a new :class:`ImpactFuncSet` from
    disk, then applies independent linear transformations to the ``intensity``,
    ``mdd``, and ``paa`` arrays of each targeted impact function.

    Parameters
    ----------
    impfset_modifier : ImpfsetModifierConfig
        Configuration object specifying:

        - ``new_impfset_path`` : path to an Excel file to load as the new
          :class:`ImpactFuncSet`, or ``None`` to modify the input in-place.
        - ``haz_type`` : hazard type string used to look up functions.
        - ``impf_ids`` : IDs of functions to modify. Accepts ``None`` or
          ``"all"`` to target every function of ``haz_type``, a single
          ``int`` or ``str``, or a ``list`` of IDs. Raises :class:`ValueError`
          for any other type.
        - ``impf_int_mult``, ``impf_int_add`` : scale and shift for intensity.
        - ``impf_mdd_mult``, ``impf_mdd_add`` : scale and shift for MDD.
        - ``impf_paa_mult``, ``impf_paa_add`` : scale and shift for PAA.

    Returns
    -------
    ImpfsetChange
        A callable with signature ``(impfset: ImpactFuncSet, **kwargs) -> ImpactFuncSet``.

    Raises
    ------
    ValueError
        If ``impfset_modifier.impf_ids`` is not ``None``, ``"all"``, an
        ``int``, a ``str``, or a ``list``.
    """

    def impfset_change(impfset: ImpactFuncSet, **_kwargs) -> ImpactFuncSet:
        changed_impfset = (
            impfset.from_excel(impfset_modifier.new_impfset_path)
            if impfset_modifier.new_impfset_path is not None
            else impfset
        )
        if impfset_modifier.impf_ids is None or impfset_modifier.impf_ids == "all":
            ids_to_change = impfset.get_ids(haz_type=impfset_modifier.haz_type)
        elif isinstance(impfset_modifier.impf_ids, list):
            ids_to_change = impfset_modifier.impf_ids
        elif isinstance(impfset_modifier.impf_ids, (str, int)):
            ids_to_change = [impfset_modifier.impf_ids]
        else:
            raise ValueError(
                f"Impact function ids to changes are invalid: {impfset_modifier.impf_ids}"
            )

        funcs = changed_impfset.get_func(haz_type=impfset_modifier.haz_type)
        funcs = [funcs] if isinstance(funcs, ImpactFunc) else funcs

        for impf in funcs:
            # Apply Intensity Mod
            if impf.id in ids_to_change:
                mult, add = (
                    impfset_modifier.impf_int_mult,
                    impfset_modifier.impf_int_add,
                )
                impf.intensity = impf.intensity * mult + add

                mult, add = (
                    impfset_modifier.impf_mdd_mult,
                    impfset_modifier.impf_mdd_add,
                )
                impf.mdd = impf.mdd * mult + add

                mult, add = (
                    impfset_modifier.impf_paa_mult,
                    impfset_modifier.impf_paa_add,
                )
                impf.paa = impf.paa * mult + add

        return changed_impfset

    return impfset_change


def change_impfset(new_impfsets: ImpactFuncSet) -> ImpfsetChange:
    """Return a change function that unconditionally replaces the impact function set.

    The returned function ignores its input and always returns ``new_impfsets``.
    Note that ``new_impfsets`` is a shared reference; callers should ensure the
    object is not mutated after being passed here.

    Parameters
    ----------
    new_impfsets : ImpactFuncSet
        The :class:`ImpactFuncSet` to substitute in place of the original.

    Returns
    -------
    ImpfsetChange
        A callable with signature ``(impfset: ImpactFuncSet, **kwargs) -> ImpactFuncSet``
        that discards its input and returns ``new_impfsets``.
    """

    def impfset_change(_: ImpactFuncSet) -> ImpactFuncSet:
        return new_impfsets

    return impfset_change


def helper_exposure(exposures_modifier: ExposuresModifierConfig) -> ExposuresChange:
    """Return a change function that reassigns impact function IDs and zeros
    selected exposure values.

    Constructs an :class:`ExposuresChange` from an
    :class:`ExposuresModifierConfig`. The returned function optionally loads a
    new :class:`Exposures` from disk, then applies two optional modifications
    to its underlying GeoDataFrame:

    1. **Impact function ID remapping**: replaces values in
       ``impf_<haz_type>`` columns according to a provided mapping dict.
    2. **Value zeroing**: sets ``value`` to ``0`` for rows matching a boolean
       mask or index.

    Parameters
    ----------
    exposures_modifier : ExposuresModifierConfig
        Configuration object specifying:

        - ``new_exposures_path`` : path to an HDF5 file to load as the new
          :class:`Exposures`, or ``None`` to modify the input in-place.
        - ``reassign_impf_id`` : a ``dict[haz_type, {old_id: new_id}]``
          mapping used to replace impact function IDs in the GeoDataFrame,
          or ``None`` to skip.
        - ``set_to_zero`` : a boolean array, index, or label accepted by
          ``DataFrame.loc`` identifying rows whose ``value`` should be set
          to ``0``, or ``None`` to skip.

    Returns
    -------
    ExposuresChange
        A callable with signature ``(exposures: Exposures, **kwargs) -> Exposures``.
    """

    def exposures_change(exposures: Exposures, **_kwargs) -> Exposures:
        changed_exposures = (
            exposures
            if exposures_modifier.new_exposures_path is None
            else Exposures.from_hdf5(exposures_modifier.new_exposures_path)
        )
        gdf = cast(pd.DataFrame, changed_exposures.gdf)
        if exposures_modifier.reassign_impf_id is not None:
            for haz_type, mapping in exposures_modifier.reassign_impf_id.items():
                gdf[f"impf_{haz_type}"] = gdf[f"impf_{haz_type}"].replace(mapping)

        if exposures_modifier.set_to_zero is not None:
            gdf.loc[exposures_modifier.set_to_zero, "value"] = 0

        return changed_exposures

    return exposures_change
