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

from pandas.tseries.offsets import BaseOffset

__all__ = ["Measure"]

import copy
import inspect
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional, Tuple, TypeVar

from climada.entity.measures.measure_config import MeasureConfig

from .cost_income import CostIncome

if TYPE_CHECKING:
    from climada.entity.exposures.base import Exposures
    from climada.entity.impact_funcs.impact_func_set import ImpactFuncSet
    from climada.entity.measures.types import (
        ExposuresChange,
        HazardChange,
        ImpfsetChange,
    )
    from climada.hazard.base import Hazard

    T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)

LOGGER = logging.getLogger(__name__)

# TODO: risk transfer?

T = TypeVar("T", Exposures, ImpactFuncSet, Hazard)


# Note for review:
# This function will moved in helper.py in a future PR which will
# add I/O based on MeasureConfig dataclasses
def identity_function(x: T, **_kwargs: Any) -> T:
    """Identity function

    Returns the provided parameter. Usefull to design measures without effect
    on some of the risk components (Hazard, Exposures, Impact Function)
    """
    return x


def allow_kwargs(func):
    """
    Decorator that allows a function to accept (and silently ignore) keyword arguments.

    If the wrapped function already accepts ``**kwargs``, it is called unchanged.
    Otherwise, any keyword arguments not present in the function's signature are
    filtered out before the call, preventing ``TypeError`` from unexpected keywords.

    The functions used by `Measure` objects to apply changes on ``Exposures``, ``Hazard``,
    and ``ImpactFuncSet`` always receive ``base_exposure, base_hazard, base_impfset`` as
    keyword arguments. This decorator is applied to users-defined functions and prevent
    them from not accepting these kwargs and raising a ``TypeError``.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        A wrapped version of `func` that accepts keyword arguments.

    Examples
    --------
    >>> @allow_kwargs
    ... def greet(name, greeting="Hello"):
    ...     return f"{greeting}, {name}!"
    >>> greet("Alice", greeting="Hi", unused_param="ignored")
    'Hi, Alice!'

    >>> @allow_kwargs
    ... def add(a, b):
    ...     return a + b
    >>> add(1, 2, extra=99)
    3
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the names of arguments the original function accepts
        params = inspect.signature(func).parameters

        # Filter kwargs to only include what the function can handle
        # (Unless the function already has **kwargs in its signature)
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            return func(*args, **kwargs)

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        return func(*args, **filtered_kwargs)

    return wrapper


class Measure:
    """
    Contains a measure to be applied to a set of exposures, impact functions,
    and hazard.

    A ``Measure`` represents a single adaptation or risk-reduction action. It
    holds three (optional) transformation functions, one each for
    :class:`Exposures`, :class:`ImpactFuncSet`, and :class:`Hazard`, that are
    can be applied to a triplet of ``(Exposures, ImpactFuncSet, Hazard)``, to
    reflect the effect of the measure.

    It also holds a `CostIncome` object to define the financial aspects of the
    measure (see :class:`CostIncome` and :ref:`cost-income-tutorial`).

    Finally it holds an `implementation_duration` attribute, in the form of a
    pandas ``DateOffset``, which is used when the time dimension is considered.

    Notes
    -----

    The only requirement for each function is to return an object of the same
    class (e.g. :class:`Hazard` for ``hazard_change``). Functions can accept
    keyword arguments to enable advanced effect (depending on a year of
    application for instance). These arguments can be passed when the
    :class:`Measure` is applied (see :py:meth:`~Measure.apply`). Note that for
    convenience, each functions receive the by default "base" ``(Exposures,
    ImpactFuncSet, Hazard)`` triplet as keyword arguments (`base_exposure`,
    `base_impfset`, `base_hazard`).

    If the ``Measure`` was defined from a ``MeasureConfig`` object, the
    configuration is stored and the measure can be serialized to a file.
    (see :ref:`measure-config-tutorial` and :ref:`measure-tutorial`).

    Attributes
    ----------
    name : str
        Name of the measure.
    exposures_change : ExposuresChange
        Function to change exposures.
    impfset_change : ImpfsetChange
        Function to change impact function set.
    hazard_change : HazardChange
        Function to change hazard.
    sub_measures : list of str, optional
        List of measure names that this measure is a combination of.
    cost_income : climada.entity.measures.cost_income.CostIncome
        Cost and income object associated with the measure.
    implementation_duration : pd.DateOffset, optional
        Duration of implementation before the measure is fully functional.
    """

    def __init__(
        self,
        name: str,
        *,
        exposures_changes: ExposuresChange = identity_function,
        impfset_changes: ImpfsetChange = identity_function,
        hazard_changes: HazardChange = identity_function,
        sub_measures: Optional[list[str]] = None,
        cost_income: Optional[CostIncome] = None,
        implementation_duration: Optional[BaseOffset] = None,
        color_rgb: Optional[Tuple[float, float, float]] = None,
        _config: Optional[MeasureConfig] = None,
    ):
        """
        Initialize a new Measure object.

        Parameters
        ----------
        name : str
            Name of the measure.
        exposures_change : callable, optional
            Transformation function for Exposures. Defaults to identity.
        impfset_change : callable, optional
            Transformation function for ImpactFuncSet. Defaults to identity.
        hazard_change : callable, optional
            Transformation function for Hazard. Defaults to identity.
        sub_measures : list of str, optional
            Names of component measures.
        cost_income : CostIncome, optional
            Financial data. If None, an empty CostIncome is initialized.
        implementation_duration : pd.DateOffset, optional
            Time offset for full implementation.
        """

        self.name = name
        self.exposures_changes = allow_kwargs(exposures_changes)
        self.hazard_changes = allow_kwargs(hazard_changes)
        self.impfset_changes = allow_kwargs(impfset_changes)
        self.sub_measures = sub_measures
        self.cost_income = cost_income if cost_income is not None else CostIncome()
        self.implementation_duration = implementation_duration
        self.color_rgb = (0, 0, 0) if color_rgb is None else color_rgb
        self._config = _config

    # DONE always provide exp, impfset and hazard as kwargs by default
    # Have a precedence system (if users provide their own it takes over)
    # TODO Check that it works

    @property
    def is_serializable(self) -> bool:
        """Returns True if the ``Measure`` was created from
        a ``MeasureConfig`` object and can be serialized.
        """

        return self._config is not None

    def apply_exposures_changes(
        self, exposures: Exposures, enforce_copy: bool = True, **kwargs
    ) -> Exposures:
        """Apply the changes from the measure to the given :class:`Exposures` object.

        This method applies the `exposures_changes` function of the measure to
        the provided :class:`Exposures` object. If ``enforce_copy`` is True (default), a
        deep copy of the exposures is created before modification to ensure
        immutability of the original object.

        Additional keyword arguments to the function can be passed directly.

        Parameters
        ----------
        exposures : Exposures
            The input exposures object to be transformed.
        enforce_copy : bool, optional
            If True (default), creates a deep copy of `exposures` before applying
            changes, provided the transformation function is not the identity function.
            If False, the original object may be modified in-place depending on the
            behavior of `exposures_changes`.
        **kwargs : dict, optional
            Additional keyword arguments passed directly to the `exposures_changes`
            function.

        Returns
        -------
        Exposures
            The resulting :class:`Exposures` object after the transformation has been applied.
            If `enforce_copy` was True, this is a new object.

        Notes
        -----
        The deep copy operation is skipped if `enforce_copy` is False or if
        `self.exposures_changes` is the identity function, optimizing performance
        when no actual changes are expected or when in-place modification is desired.
        """

        changed_exp = (
            copy.deepcopy(exposures)
            if enforce_copy and self.exposures_changes is not identity_function
            else exposures
        )
        try:
            return self.exposures_changes(changed_exp, **kwargs)
        except TypeError as exc:
            # Check if it's a missing argument error
            if "missing" in str(exc) and "required positional argument" in str(exc):
                raise TypeError(
                    f"The function to apply to the exposures requires additional arguments\
                    that were not provided.\n"
                    f"Please check the function signature or the helper used and provide the\
                    required arguments "
                    "via kwargs_exposures.\n"
                    f"Original error: {exc}"
                ) from exc
            raise

    def apply_impfset_changes(
        self, impfset: ImpactFuncSet, enforce_copy: bool = True, **kwargs
    ) -> ImpactFuncSet:
        """
        Apply the changes from the measure to the given :class:`ImpactFuncSet` object.

        This method applies the `impfset_changes` function of the measure to
        the provided :class:`ImpactFuncSet` object. If `enforce_copy` is True
        (default), a deep copy of the impfset is created before modification to
        ensure immutability of the original object.

        Parameters
        ----------
        impfset : ImpactFuncSet
            The input impfset object to be transformed.
        enforce_copy : bool, optional
            If True (default), creates a deep copy of `impfset` before applying
            changes, provided the transformation function is not the identity
            function. If False, the original object may be modified in-place
            depending on the behavior of `impfset_changes`.
        **kwargs : dict, optional
            Additional keyword arguments passed directly to the `impfset_changes`
            function.

        Returns
        -------
        ImpactFuncSet
            The resulting :class:`ImpactFuncSet` after the transformation has
            been applied. If `enforce_copy` was True, this is a new object.

        Notes
        -----
        The deep copy operation is skipped if `enforce_copy` is False or if
        `self.impfset_changes` is the identity function, optimizing performance
        when no actual changes are expected or when in-place modification is
        desired.
        """

        changed_impfset = (
            copy.deepcopy(impfset)
            if enforce_copy and self.impfset_changes is not identity_function
            else impfset
        )
        try:
            return self.impfset_changes(changed_impfset, **kwargs)
        except TypeError as exc:
            # Check if it's a missing argument error
            if "missing" in str(exc) and "required positional argument" in str(exc):
                raise TypeError(
                    f"The function to apply to the impact function set requires\
                    additional arguments that were not provided.\n"
                    f"Please check the function signature or the helper used\
                    and provide the required arguments via kwargs_impfset.\n"
                    f"Original error: {exc}"
                ) from exc
            raise

    def apply_hazard_changes(
        self, hazard: Hazard, enforce_copy: bool = True, **kwargs
    ) -> Hazard:
        """
        Apply the changes from the measure to the given :class:`Hazard` object.

        This method applies the `hazard_changes` function of the measure to the
        provided :class:`Hazard` object. If `enforce_copy` is True (default), a
        deep copy of the hazard is created before modification to ensure
        immutability of the original object.

        Parameters
        ----------
        hazard : Hazard
            The input hazard object to be transformed.
        enforce_copy : bool, optional
            If True (default), creates a deep copy of `hazard` before applying
            changes, provided the transformation function is not the identity
            function. If False, the original object may be modified in-place
            depending on the behavior of `hazard_changes`.
        **kwargs : dict, optional
            Additional keyword arguments passed directly to the `hazard_changes`
            function.

        Returns
        -------
        Hazard
            The resulting hazard object after the transformation has been
            applied. If `enforce_copy` was True, this is a new object.

        Notes
        -----
        The deep copy operation is skipped if `enforce_copy` is False or if
        `self.hazard_changes` is the identity function, optimizing performance
        when no actual changes are expected or when in-place modification is
        desired.
        """

        changed_hazard = (
            copy.deepcopy(hazard)
            if enforce_copy and self.hazard_changes is not identity_function
            else hazard
        )
        try:
            return self.hazard_changes(changed_hazard, **kwargs)
        except TypeError as exc:
            # Check if it's a missing argument error
            if "missing" in str(exc) and "required positional argument" in str(exc):
                raise TypeError(
                    f"The function to apply to the hazard requires\
                    additional arguments that were not provided.\n"
                    f"Please check the function signature or the helper used\
                    and provide the required arguments via kwargs_hazard.\n"
                    f"Original error: {exc}"
                ) from exc
            raise

    def apply(
        self,
        exposures: Exposures,
        impfset: ImpactFuncSet,
        hazard: Hazard,
        enforce_copy: bool = True,
        **kwargs,
    ) -> Tuple[Exposures, ImpactFuncSet, Hazard]:
        """Apply all measure transformations to the provided triplet of
        :class:`Exposures`, :class:`ImpactFuncSet`, :class:`Hazard`.

        This method applies the measure changes across all three
        risk parts: exposures, impact function set, and hazard data.

        The method implements a flexible keyword arguments merging strategy
        where the original triplet is provided as default context to each
        transformation, which can then be overridden by entity-specific kwargs
        dictionaries. This enables transformation requiring the information
        from other risk components (for instance, removing events based on impact
        threshold) or additional information (for instance, effect depending on
        year of implementation).

        Refer to :ref:`measure-tutorial` for more details.

        Parameters
        ----------
        exposures : Exposures
            The input exposures object to be transformed.
        impfset : ImpactFuncSet
            The impact function set to be transformed.
        hazard : Hazard
            The hazard data to be transformed.
        enforce_copy : bool, optional
            If True (default), creates deep copies of entities before applying
            changes, provided the transformation functions are not identity functions.
            If False, entities may be modified in-place depending on the behavior
            of the underlying transformation methods.
        **kwargs : dict, optional
            Additional keyword arguments for configuring transformations. Supports
            nested dictionaries for entity-specific customization:

            * ``kwargs_exposures``: Dict of kwargs passed to `apply_exposures_changes`
            * ``kwargs_impfset``: Dict of kwargs passed to `apply_impfset_changes`
            * ``kwargs_hazard``: Dict of kwargs passed to `apply_hazard_changes`

            Each nested dict is merged with the default triplet context (exposures,
            impfset, hazard), allowing transformations to access related entities
            while permitting entity-specific overrides.

        Returns
        -------
        Tuple[Exposures, ImpactFuncSet, Hazard]
            A tuple containing the transformed entities in the order:
            (changed_exposures, changed_impfset, changed_hazard). If `enforce_copy`
            was True, these are new objects; otherwise, they may reference the
            original inputs or modified versions thereof.

        Notes
        -----
        The kwargs merging follows this priority order:

        1. Default context: The original triplet (exposures, impfset, hazard)
        2. Entity-specific overrides: Values from ``kwargs_exposures``,
           ``kwargs_impfset``, or ``kwargs_hazard`` respectively

        This ensures that each transformation receives full context about all entities
        while allowing fine-grained control over individual transformations.

        The transformation order is: exposures → hazard → impact function set.
        Each transformation is independent, so changes to one entity do not affect
        the others during processing.
        """

        default_kwargs = {
            "base_exposures": exposures,
            "base_impfset": impfset,
            "base_hazard": hazard,
        }
        # Always provide the triplet by default, and overwrite by custom kwargs.
        kwargs_exp = default_kwargs | kwargs.get("kwargs_exposures", {})
        kwargs_impfset = default_kwargs | kwargs.get("kwargs_impfset", {})
        kwargs_hazard = default_kwargs | kwargs.get("kwargs_hazard", {})
        changed_exposures = self.apply_exposures_changes(
            exposures, enforce_copy, **kwargs_exp
        )
        changed_hazard = self.apply_hazard_changes(
            hazard, enforce_copy, **kwargs_hazard
        )
        changed_impfset = self.apply_impfset_changes(
            impfset, enforce_copy, **kwargs_impfset
        )
        return changed_exposures, changed_impfset, changed_hazard

    def calc_impact(self, exposures, impfset, hazard):
        from climada.engine.impact_calc import (
            ImpactCalc,  # pylint: disable=import-outside-toplevel
        )

        new_exp, new_impfs, new_haz = self.apply(exposures, impfset, hazard)
        if new_haz.centr_exp_col not in new_exp.gdf.columns:
            LOGGER.warning(
                "No assigned hazard centroids in exposure object after the "
                "application of the measure. The centroids will be assigned during impact "
                "calculation. This is potentiall costly. To silence this warning, make sure "
                "that centroids are assigned to all exposures."
            )
            new_exp.assign_centroids(new_haz)
        imp = ImpactCalc(new_exp, new_impfs, new_haz).impact(
            save_mat=False, assign_centroids=False
        )
        return imp.calc_risk_transfer(0, 0)
