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

Compute Tropical Cyclone windfields (see compute_windfields_sparse function).
"""

import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
from scipy import sparse

from climada.hazard import Centroids
from climada.hazard.tc_tracks import estimate_rmw
from climada.util import constants as u_const
from climada.util import coordinates as u_coord
from climada.util import ureg

LOGGER = logging.getLogger(__name__)

NM_TO_KM = (1.0 * ureg.nautical_mile).to(ureg.kilometer).magnitude
KMH_TO_MS = (1.0 * ureg.km / ureg.hour).to(ureg.meter / ureg.second).magnitude
KM_TO_M = (1.0 * ureg.kilometer).to(ureg.meter).magnitude
H_TO_S = (1.0 * ureg.hours).to(ureg.seconds).magnitude
MBAR_TO_PA = (1.0 * ureg.millibar).to(ureg.pascal).magnitude
"""Unit conversion factors for JIT functions that can't use ureg"""

DEF_MAX_DIST_EYE_KM = 300
"""Default value for the maximum distance (in km) of a centroid to the TC center at which wind
speed calculations are done."""

DEF_INTENSITY_THRES = 17.5
"""Default value for the threshold below which wind speeds (in m/s) are stored as 0."""

DEF_MAX_MEMORY_GB = 8
"""Default value of the memory limit (in GB) for windfield computations (in each thread)."""

MODEL_VANG = {"H08": 0, "H1980": 1, "H10": 2, "ER11": 3}
"""Enumerate different symmetric wind field models."""

DEF_RHO_AIR = 1.15
"""Default value for air density (in kg/m³), following Holland 1980."""

DEF_GRADIENT_TO_SURFACE_WINDS = 0.9
"""Default gradient-to-surface wind reduction factor, following the 90%-rule mentioned in:

Franklin, J.L., Black, M.L., Valde, K. (2003): GPS Dropwindsonde Wind Profiles in Hurricanes and
Their Operational Implications. Weather and Forecasting 18(1): 32–44.
https://doi.org/10.1175/1520-0434(2003)018<0032:GDWPIH>2.0.CO;2

According to Table 2, this is a reasonable factor for the 750 hPa level in the eyewall region. For
other regions and levels, values of 0.8 or even 0.75 might be justified.
"""

T_ICE_K = 273.16
"""Freezing temperatur of water (in K), for conversion between K and °C"""

V_ANG_EARTH = 7.29e-5
"""Earth angular velocity (in radians per second)"""


def _vgrad(si_track, gradient_to_surface_winds):
    """Gradient wind speeds (in m/s) without translational influence at each track node

    The track dataset is modified in place, with the "vgrad" data variable added.

    Parameters
    ----------
    si_track : xr.Dataset
        Track information as returned by `tctrack_to_si`. The data variables used by this function
        are "vmax" and "vtrans_norm". The result is stored in place as new data variable "vgrad".
    gradient_to_surface_winds : float
        The gradient-to-surface wind reduction factor to use.
    """
    si_track["vgrad"] = (
        np.fmax(0, si_track["vmax"] - si_track["vtrans_norm"])
        / gradient_to_surface_winds
    )


def compute_angular_windspeeds(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    mask_centr_close: np.ndarray,
    model: int,
    cyclostrophic: Optional[bool] = False,
    model_kwargs: Optional[dict] = None,
):
    """Compute (absolute) angular wind speeds according to a parametric wind profile

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Which data variables are used in the computation of the wind
        profile depends on the selected model.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    mask_centr_close : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    model : int
        Wind profile model selection according to MODEL_VANG.
    model_kwargs: dict, optional
        If given, forward these kwargs to the selected model. Default: None
    cyclostrophic: bool, optional, deprecated
        This argument is deprecated and will be removed in a future release.
        Include `cyclostrophic` as `model_kwargs` instead.

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
        containing the magnitude of the angular windspeed per track position per centroid location
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    if cyclostrophic is not None:
        warnings.warn(
            "The 'cyclostrophic' argument is deprecated and will be removed in a future"
            "release. Include it in 'model_kwargs' instead.",
            DeprecationWarning,
        )
        model_kwargs["cyclostrophic"] = cyclostrophic

    compute_funs = {
        MODEL_VANG["H1980"]: _compute_angular_windspeeds_h1980,
        MODEL_VANG["H08"]: _compute_angular_windspeeds_h08,
        MODEL_VANG["H10"]: _compute_angular_windspeeds_h10,
        MODEL_VANG["ER11"]: _stat_er_2011,
    }
    if model not in compute_funs:
        raise NotImplementedError(f"The specified wind model is not supported: {model}")
    result = compute_funs[model](
        si_track,
        d_centr,
        mask_centr_close,
        **model_kwargs,
    )
    result[0, :] *= 0
    return result


def _compute_angular_windspeeds_h1980(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr_msk: np.ndarray,
    cyclostrophic: bool = False,
    gradient_to_surface_winds: float = DEF_GRADIENT_TO_SURFACE_WINDS,
    rho_air_const: float = DEF_RHO_AIR,
):
    """Compute (absolute) angular wind speeds according to the Holland 1980 model

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. Which data variables are used in the computation of the wind
        profile depends on the selected model.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    close_centr_msk : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False
    gradient_to_surface_winds : float, optional
        The gradient-to-surface wind reduction factor to use. The wind profile is computed on the
        gradient level, and wind speeds are converted to the surface level using this factor.
        Default: 0.9
    rho_air_const : float or None, optional
        The constant value for air density (in kg/m³) to assume in the formulas from Holland 1980.
        By default, the constant value suggested in Holland 1980 is used. If set to None, the air
        density is computed from pressure following equation (9) in Holland et al. 2010.
        Default: 1.15

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
        containing the magnitude of the angular windspeed per track position per centroid location
    """
    _vgrad(si_track, gradient_to_surface_winds)
    _rho_air(si_track, rho_air_const)
    _B_holland_1980(si_track)
    result = _stat_holland_1980(
        si_track, d_centr, close_centr_msk, cyclostrophic=cyclostrophic
    )
    result *= gradient_to_surface_winds
    return result


def _compute_angular_windspeeds_h08(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr_msk: np.ndarray,
    cyclostrophic: bool = False,
    gradient_to_surface_winds: float = DEF_GRADIENT_TO_SURFACE_WINDS,
    rho_air_const: float = DEF_RHO_AIR,
):
    """Compute (absolute) angular wind speeds according to the Holland 2008 model

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. Which data variables are used in the computation of the wind
        profile depends on the selected model.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    close_centr_msk : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    cyclostrophic : bool, optional
        If True, don't apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False
    gradient_to_surface_winds : float, optional
        The gradient-to-surface wind reduction factor to use. The wind profile is computed on the
        surface level, but the clipping interval of the B-value depends on this factor.
        Default: 0.9
    rho_air_const : float or None, optional
        The constant value for air density (in kg/m³) to assume in the formula from Holland 1980.
        By default, the constant value suggested in Holland 1980 is used. If set to None, the air
        density is computed from pressure following equation (9) in Holland et al. 2010.
        Default: 1.15

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
        containing the magnitude of the angular windspeed per track position per centroid location
    """
    _rho_air(si_track, rho_air_const)
    _bs_holland_2008(si_track, gradient_to_surface_winds=gradient_to_surface_winds)
    return _stat_holland_1980(
        si_track, d_centr, close_centr_msk, cyclostrophic=cyclostrophic
    )


def _compute_angular_windspeeds_h10(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    close_centr_msk: np.ndarray,
    gradient_to_surface_winds: float = DEF_GRADIENT_TO_SURFACE_WINDS,
    rho_air_const: float = DEF_RHO_AIR,
    vmax_from_cen: bool = True,
    vmax_in_brackets: bool = False,
    **kwargs,
):
    """Compute (absolute) angular wind speeds according to the Holland et al. 2010 model

    Note that this model is always cyclostrophic, the parameter setting is ignored.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of `tctrack_to_si`. Which data variables are used in the computation of the wind
        profile depends on the selected model.
    d_centr : np.ndarray of shape (npositions, ncentroids)
        Distance (in m) between centroids and track positions.
    close_centr_msk : np.ndarray of shape (npositions, ncentroids)
        For each track position one row indicating which centroids are within reach.
    cyclostrophic : bool, optional
        This parameter is ignored because this model is always cyclostrophic. Default: True
    gradient_to_surface_winds : float, optional
        The gradient-to-surface wind reduction factor to use. the wind profile is computed on the
        surface level, but the clipping interval of the B-value depends on this factor.
        Default: 0.9
    rho_air_const : float or None, optional
        The constant value for air density (in kg/m³) to assume in the formula for the B-value. By
        default, the value suggested in Holland 1980 is used. If set to None, the air density is
        computed from pressure following equation (9) in Holland et al. 2010. Default: 1.15
    vmax_from_cen : boolean, optional
        If True, replace the recorded value of vmax along the track by an estimate from pressure,
        following equation (8) in Holland et al. 2010. Default: True
    vmax_in_brackets : bool, optional
        Specifies which of the two formulas in equation (6) of Holland et al. 2010 to use. If
        False, the formula with vmax outside of the brackets is used. Note that, a side-effect of
        the formula with vmax inside of the brackets is that the wind speed maximum is attained a
        bit farther away from the center than according to the recorded radius of maximum
        winds (RMW). Default: False

    Returns
    -------
    ndarray of shape (npositions, ncentroids)
        containing the magnitude of the angular windspeed per track position per centroid location
    """
    if not kwargs.get("cyclostrophic", True):
        LOGGER.warning(
            "The function _compute_angular_windspeeds_h10 was called with parameter "
            '"cyclostrophic" equal to false. Please be aware that this setting is ignored as the'
            " Holland et al. 2010 model is always cyclostrophic."
        )
    _rho_air(si_track, rho_air_const)
    if vmax_from_cen:
        _bs_holland_2008(si_track, gradient_to_surface_winds=gradient_to_surface_winds)
        _v_max_s_holland_2008(si_track)
    else:
        _B_holland_1980(si_track, gradient_to_surface_winds=gradient_to_surface_winds)
    hol_x = _x_holland_2010(
        si_track, d_centr, close_centr_msk, vmax_in_brackets=vmax_in_brackets
    )
    return _stat_holland_2010(
        si_track,
        d_centr,
        close_centr_msk,
        hol_x,
        vmax_in_brackets=vmax_in_brackets,
    )


def _rho_air(si_track: xr.Dataset, const: Optional[float]):
    """Eyewall density of air (in kg/m³) at each track node.

    The track dataset is modified in place, with the "rho_air" data variable added.

    Parameters
    ----------
    si_track : xr.Dataset
        Track information as returned by `tctrack_to_si`. The data variables used by this function
        are "lat", "cen", and "pdelta". The result is stored in place as new data
        variable "rho_air".
    const : float or None
        A constant value for air density (in kg/m³) to assume. If None, the air density is
        estimated from eyewall pressure following equation (9) in Holland et al. 2010.
    """
    if const is not None:
        si_track["rho_air"] = xr.full_like(si_track["time"], const, dtype=float)
        return

    # surface relative humidity (unitless), assumed constant following Holland et al. 2010
    surface_relative_humidity = 0.9

    # surface temperature (in °C), following equation (9) in Holland 2008
    temp_s = 28.0 - 3.0 * (si_track["lat"] - 10.0) / 20.0

    # eyewall surface pressure (in Pa), following equation (6) in Holland 2008
    pres_eyewall = si_track["cen"] + si_track["pdelta"] / np.exp(1)

    # mixing ratio (in kg/kg), estimated from temperature, using formula for saturation vapor
    # pressure in Bolton 1980 (multiplied by the ratio of molar masses of water vapor and dry air)
    # We multiply by 100, since the formula by Bolton is in hPa (mbar), and we use Pa.
    r_mix = 100 * 3.802 / pres_eyewall * np.exp(17.67 * temp_s / (243.5 + temp_s))

    # virtual surface temperature (in K)
    temp_vs = (T_ICE_K + temp_s) * (1 + 0.81 * surface_relative_humidity * r_mix)

    # specific gas constant of dry air (in J/kgK)
    r_dry_air = 286.9

    # density of air (in kg/m³); when checking the units, note that J/Pa = m³
    si_track["rho_air"] = pres_eyewall / (r_dry_air * temp_vs)


def _bs_holland_2008(
    si_track: xr.Dataset,
    gradient_to_surface_winds: float = DEF_GRADIENT_TO_SURFACE_WINDS,
):
    """Holland's 2008 b-value estimate for sustained surface winds.
    (This is also one option of how to estimate the b-value in Holland 2010,
    for the other option consult '_bs_holland_2010_v2'.)

    The result is stored in place as a new data variable "hol_b".

    Unlike the original 1980 formula (see ``_B_holland_1980``), this approach does not require any
    wind speed measurements, but is based on the more reliable pressure information.

    The parameter applies to 1-minute sustained winds at 10 meters above ground.
    It is taken from equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    b_s = -4.4 * 1e-5 * (penv - pcen)^2 + 0.01 * (penv - pcen)
          + 0.03 * (dp/dt) - 0.014 * |lat| + 0.15 * (v_trans)^hol_xx + 1.0

    where ``dp/dt`` is the time derivative of central pressure and ``hol_xx`` is Holland's x
    parameter: hol_xx = 0.6 * (1 - (penv - pcen) / 215)

    The equation for b_s has been fitted statistically using hurricane best track records for
    central pressure and maximum wind. It therefore performs best in the North Atlantic.

    Furthermore, b_s has been fitted under the assumption of a "cyclostrophic" wind field which
    means that the influence from Coriolis forces is assumed to be small. This is reasonable close
    to the radius of maximum wind where the Coriolis term (r*f/2) is small compared to the rest
    (see ``_stat_holland_1980``). More precisely: At the radius of maximum wind speeds, the typical
    order of the Coriolis term is 1 while wind speed is 50 (which changes away from the
    radius of maximum winds and as the TC moves away from the equator).

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. The data variables used by this function are "lat", "tstep",
        "vtrans_norm", "cen", and "pdelta". The result is stored in place as a new data
        variable "hol_b".
    gradient_to_surface_winds : float, optional
        The gradient-to-surface wind reduction factor to use when determining the clipping
        interval. Default: 0.9
    """
    # adjust pressure at previous track point
    prev_cen = np.zeros_like(si_track["cen"].values)
    prev_cen[1:] = si_track["cen"].values[:-1].copy()
    msk = prev_cen < 850 * MBAR_TO_PA
    prev_cen[msk] = si_track["cen"].values[msk]

    # The formula assumes that pressure values are in millibar (hPa) instead of SI units (Pa),
    # and time steps are in hours instead of seconds, but translational wind speed is still
    # expected to be in m/s.
    pdelta = si_track["pdelta"] / MBAR_TO_PA
    hol_xx = 0.6 * (1.0 - pdelta / 215)
    si_track["hol_b"] = (
        -4.4e-5 * pdelta**2
        + 0.01 * pdelta
        + 0.03
        * (si_track["cen"] - prev_cen)
        / si_track["tstep"]
        * (H_TO_S / MBAR_TO_PA)
        - 0.014 * abs(si_track["lat"])
        + 0.15 * si_track["vtrans_norm"] ** hol_xx
        + 1.0
    )
    clip_interval = _b_holland_clip_interval(gradient_to_surface_winds)
    si_track["hol_b"] = np.clip(si_track["hol_b"], *clip_interval)


def _v_max_s_holland_2008(si_track: xr.Dataset):
    """Compute maximum surface winds from pressure according to Holland 2008.

    The result is stored in place as a data variable "vmax". If a variable of that name already
    exists, its values are overwritten.

    This function implements equation (11) in the following paper:

    Holland, G. (2008). A revised hurricane pressure-wind model. Monthly
    Weather Review, 136(9), 3432–3445. https://doi.org/10.1175/2008MWR2395.1

    For reference, it reads

    v_ms = [b_s / (rho * e) * (penv - pcen)]^0.5

    where ``b_s`` is Holland b-value (see ``_bs_holland_2008``), e is Euler's number, rho is the
    density of air, ``penv`` is environmental, and ``pcen`` is central pressure.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si`` with "hol_b" (see _bs_holland_2008) and "rho_air" (see
        _rho_air) variables. The data variables used by this function are "pdelta", "hol_b",
        and "rho_air". The results are stored in place as a new data variable "vmax". If a variable
        of that name already exists, its values are overwritten.
    """
    si_track["vmax"] = np.sqrt(
        si_track["hol_b"] / (si_track["rho_air"] * np.exp(1)) * si_track["pdelta"]
    )


def _B_holland_1980(  # pylint: disable=invalid-name
    si_track: xr.Dataset,
    gradient_to_surface_winds: Optional[float] = None,
):
    """Holland's 1980 B-value computation for gradient-level winds.

    The result is stored in place as a new data variable "hol_b".

    The parameter applies to gradient-level winds (about 1000 metres above the earth's surface).
    The formula for B is derived from equations (5) and (6) in the following paper:

    Holland, G.J. (1980): An Analytic Model of the Wind and Pressure Profiles
    in Hurricanes. Monthly Weather Review 108(8): 1212–1218.
    https://doi.org/10.1175/1520-0493(1980)108<1212:AAMOTW>2.0.CO;2

    For reference, inserting (6) into (5) and solving for B at r = RMW yields:

    B = v^2 * e * rho / (penv - pcen)

    where v are maximum gradient-level winds ``gradient_winds``, e is Euler's number, rho is the
    density of air, ``penv`` is environmental, and ``pcen`` is central pressure.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si`` with "rho_air" variable (see _rho_air). The data variables
        used by this function are "vgrad" (or "vmax" if gradient_to_surface_winds is different from
        1.0), "pdelta", and "rho_air". The results are stored in place as a new data
        variable "hol_b".
    gradient_to_surface_winds : float, optional
        The gradient-to-surface wind reduction factor to use when determining the clipping
        interval. By default, the gradient level values are assumed. Default: None
    """
    windvar = "vgrad" if gradient_to_surface_winds is None else "vmax"

    si_track["hol_b"] = (
        si_track[windvar] ** 2 * np.exp(1) * si_track["rho_air"] / si_track["pdelta"]
    )

    clip_interval = _b_holland_clip_interval(gradient_to_surface_winds)
    si_track["hol_b"] = np.clip(si_track["hol_b"], *clip_interval)


def _b_holland_clip_interval(gradient_to_surface_winds):
    """The clip interval to use for the Holland B-value

    The default clip interval for gradient level B-values is taken to be (1.0, 2.5), following
    Holland 1980.

    Parameters
    ----------
    gradient_to_surface_winds : float or None
        The gradient-to-surface wind reduction factor to use when rescaling the gradient-level
        clip interval (1.0, 2.5) proposed in Holland 1980. If None, no rescaling is applied.

    Returns
    -------
    b_min, b_max : float
        Minimum and maximum value of the clip interval.
    """
    clip_interval = (1.0, 2.5)
    if gradient_to_surface_winds is not None:
        fact = gradient_to_surface_winds**2
        clip_interval = tuple(c * fact for c in clip_interval)
    return clip_interval


def _x_holland_2010(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    mask_centr_close: np.ndarray,
    v_n: Union[float, np.ndarray] = 17.0,
    r_n_km: Union[float, np.ndarray] = 300.0,
    vmax_in_brackets: bool = False,
) -> np.ndarray:
    """Compute exponent for wind model according to Holland et al. 2010.

    This function implements equation (10) from the following paper:

    Holland et al. (2010): A Revised Model for Radial Profiles of Hurricane Winds. Monthly
    Weather Review 138(12): 4393–4401. https://doi.org/10.1175/2010MWR3317.1

    For reference, it reads

    x = 0.5  [for r < r_max]
    x = 0.5 + (r - r_max) * (x_n - 0.5) / (r_n - r_max)  [for r >= r_max]

    The peripheral exponent x_n is adjusted to fit the peripheral observation of wind speeds ``v_n``
    at radius ``r_n``.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si`` with "hol_b" variable (see _bs_holland_2008). The data variables
        used by this function are "rad", "vmax", and "hol_b".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    mask_centr_close : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    v_n : np.ndarray of shape (nnodes,) or float, optional
        Peripheral wind speeds (in m/s) at radius ``r_n`` outside of radius of maximum winds
        ``r_max``. In absence of a second wind speed measurement, this value defaults to 17 m/s
        following Holland et al. 2010 (at a radius of 300 km).
    r_n_km : np.ndarray of shape (nnodes,) or float, optional
        Radius (in km) where the peripheral wind speed ``v_n`` is measured (or assumed).
        In absence of a second wind speed measurement, this value defaults to 300 km following
        Holland et al. 2010.
    vmax_in_brackets : bool, optional
        If True, use the alternative formula in equation (6) to solve for the peripheral exponent
        x_n from the second measurement. Note that, a side-effect of the formula with vmax inside
        of the brackets is that the wind speed maximum is attained a bit farther away from the
        center than according to the recorded radius of maximum winds (RMW). Default: False

    Returns
    -------
    hol_x : np.ndarray of shape (nnodes, ncentroids)
        Exponents according to Holland et al. 2010.
    """
    hol_x = np.zeros_like(d_centr)
    r_max, v_max_s, hol_b, d_centr, v_n, r_n = [
        np.broadcast_to(ar, d_centr.shape)[mask_centr_close]
        for ar in [
            si_track["rad"].values[:, None],
            si_track["vmax"].values[:, None],
            si_track["hol_b"].values[:, None],
            d_centr,
            np.atleast_1d(v_n)[:, None],
            np.atleast_1d(r_n_km)[:, None],
        ]
    ]

    # convert to SI units
    r_n *= KM_TO_M

    # compute peripheral exponent from second measurement
    # (equation (6) from Holland et al. 2010 solved for x)
    r_max_norm = (r_max / r_n) ** hol_b
    if vmax_in_brackets:
        x_n = np.log(v_n) / np.log(v_max_s**2 * r_max_norm * np.exp(1 - r_max_norm))

        # With `vmax_in_brackets`, the maximum is shifted away from the recorded RMW. We truncate
        # here to avoid an exaggerated shift. The value 1.0 has been found to be reasonable by
        # manual testing of thresholds. Note that the truncation means that the peripheral wind
        # speed v_n is not exactly attained in some cases.
        x_n = np.fmin(x_n, 1.0)
    else:
        x_n = np.log(v_n / v_max_s) / np.log(r_max_norm * np.exp(1 - r_max_norm))

    # linearly interpolate between max exponent and peripheral exponent
    x_max = 0.5
    hol_x[mask_centr_close] = x_max + np.fmax(0, d_centr - r_max) * (x_n - x_max) / (
        r_n - r_max
    )

    # Truncate to prevent wind speed from increasing again towards the peripheral radius (which is
    # unphysical). A value of 0.4 has been found to be reasonable by manual testing of thresholds.
    # Note that this means that the peripheral wind speed v_n is not exactly attained sometimes.
    hol_x[mask_centr_close] = np.fmax(hol_x[mask_centr_close], 0.4)

    return hol_x


def _stat_holland_2010(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    mask_centr_close: np.ndarray,
    hol_x: Union[float, np.ndarray],
    vmax_in_brackets: bool = False,
) -> np.ndarray:
    """Symmetric and static surface wind fields (in m/s) according to Holland et al. 2010

    This function applies the cyclostrophic surface wind model expressed in equation (6) from

    Holland et al. (2010): A Revised Model for Radial Profiles of Hurricane Winds. Monthly
    Weather Review 138(12): 4393–4401. https://doi.org/10.1175/2010MWR3317.1

    More precisely, this function implements the following equation:

    V(r) = v_max_s * [(r_max / r)^b_s * e^(1 - (r_max / r)^b_s)]^x

    In terms of this function's arguments, b_s is ``hol_b`` and r is ``d_centr``.

    If ``vmax_in_brackets`` is True, the alternative formula in (6) is used:

    .. math::

        V(r) = [v_max_s^2 * (r_max / r)^b_s * e^(1 - (r_max / r)^b_s)]^x

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si`` with "hol_b" (see _bs_holland_2008) data variables. The data
        variables used by this function are "vmax", "rad", and "hol_b".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    mask_centr_close : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    hol_x : np.ndarray of shape (nnodes, ncentroids) or float
        The exponent according to ``_x_holland_2010``.
    vmax_in_brackets : bool, optional
        If True, use the alternative formula in equation (6). Note that, a side-effect of the
        formula with vmax inside of the brackets is that the wind speed maximum is attained a bit
        farther away from the center than according to the recorded radius of maximum
        winds (RMW). Default: False

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (in m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    v_max_s, r_max, hol_b, d_centr, hol_x = [
        np.broadcast_to(ar, d_centr.shape)[mask_centr_close]
        for ar in [
            si_track["vmax"].values[:, None],
            si_track["rad"].values[:, None],
            si_track["hol_b"].values[:, None],
            d_centr,
            hol_x,
        ]
    ]

    r_max_norm = (r_max / np.fmax(1, d_centr)) ** hol_b
    if vmax_in_brackets:
        v_ang[mask_centr_close] = (
            v_max_s**2 * r_max_norm * np.exp(1 - r_max_norm)
        ) ** hol_x
    else:
        v_ang[mask_centr_close] = (
            v_max_s * (r_max_norm * np.exp(1 - r_max_norm)) ** hol_x
        )
    return v_ang


def _stat_holland_1980(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    mask_centr_close: np.ndarray,
    cyclostrophic: bool = False,
) -> np.ndarray:
    """Symmetric and static wind fields (in m/s) according to Holland 1980.

    This function applies the gradient wind model expressed in equation (4) (combined with
    equation (6)) from

    Holland, G.J. (1980): An Analytic Model of the Wind and Pressure Profiles in Hurricanes.
    Monthly Weather Review 108(8): 1212–1218.

    More precisely, this function implements the following equation:

    V(r) = [(B/rho) * (r_max/r)^B * (penv - pcen) * e^(-(r_max/r)^B) + (r*f/2)^2]^0.5 - (r*f/2)

    In terms of this function's arguments, B is ``hol_b`` and r is ``d_centr``. The air density
    rho and the Coriolis parameter f are taken from ``si_track``.

    Even though the equation has been derived originally for gradient winds (when combined with the
    output of ``_B_holland_1980``), it can be used for surface winds by adjusting the parameter
    ``hol_b`` (see function ``_bs_holland_2008``).

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si`` with "hol_b" (see, e.g., _B_holland_1980) and
        "rho_air" (see _rho_air) data variable. The data variables used by this function
        are "lat", "cp", "rad", "pdelta", "hol_b", and "rho_air".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    mask_centr_close : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    cyclostrophic : bool, optional
        If True, do not apply the influence of the Coriolis force (set the Coriolis terms to 0).
        Default: False

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    r_max, hol_b, pdelta, coriolis_p, rho_air, d_centr = [
        np.broadcast_to(ar, d_centr.shape)[mask_centr_close]
        for ar in [
            si_track["rad"].values[:, None],
            si_track["hol_b"].values[:, None],
            si_track["pdelta"].values[:, None],
            si_track["cp"].values[:, None],
            si_track["rho_air"].values[:, None],
            d_centr,
        ]
    ]

    r_coriolis = 0
    if not cyclostrophic:
        r_coriolis = 0.5 * d_centr * coriolis_p

    r_max_norm = (r_max / np.fmax(1, d_centr)) ** hol_b
    sqrt_term = (
        hol_b / rho_air * r_max_norm * pdelta * np.exp(-r_max_norm) + r_coriolis**2
    )
    v_ang[mask_centr_close] = np.sqrt(np.fmax(0, sqrt_term)) - r_coriolis
    return v_ang


def _stat_er_2011(
    si_track: xr.Dataset,
    d_centr: np.ndarray,
    mask_centr_close: np.ndarray,
    cyclostrophic: bool = False,
) -> np.ndarray:
    """Symmetric and static wind fields (in m/s) according to Emanuel and Rotunno 2011

    Emanuel, K., Rotunno, R. (2011): Self-Stratification of Tropical Cyclone Outflow. Part I:
    Implications for Storm Structure. Journal of the Atmospheric Sciences 68(10): 2236–2249.
    https://dx.doi.org/10.1175/JAS-D-10-05024.1

    The wind speeds ``v_ang`` are extracted from the momentum via the relationship M = v_ang * r,
    where r corresponds to ``d_centr``. On the other hand, the momentum is derived from the momentum
    at the peak wind position using equation (36) from Emanuel and Rotunno 2011 with Ck == Cd:

    M = M_max * [2 * (r / r_max)^2 / (1 + (r / r_max)^2)].

    The momentum at the peak wind position is

    M_max = r_max * v_max + 0.5 * f * r_max**2,

    where the Coriolis parameter f is computed from the latitude ``lat`` using the constant rotation
    rate of the earth.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. The data variables used by this function are "lat", "cp",
        "rad", and "vmax".
    d_centr : np.ndarray of shape (nnodes, ncentroids)
        Distance (in m) between centroids and track nodes.
    mask_centr_close : np.ndarray of shape (nnodes, ncentroids)
        Mask indicating for each track node which centroids are within reach of the windfield.
    cyclostrophic : bool, optional
        If True, do not apply the influence of the Coriolis force (set the Coriolis terms to 0) in
        the computation of M_max. Default: False

    Returns
    -------
    v_ang : np.ndarray (nnodes, ncentroids)
        Absolute values of wind speeds (m/s) in angular direction.
    """
    v_ang = np.zeros_like(d_centr)
    r_max, v_max, coriolis_p, d_centr = [
        np.broadcast_to(ar, d_centr.shape)[mask_centr_close]
        for ar in [
            si_track["rad"].values[:, None],
            si_track["vmax"].values[:, None],
            si_track["cp"].values[:, None],
            d_centr,
        ]
    ]

    # compute the momentum at the maximum
    momentum_max = r_max * v_max

    if not cyclostrophic:
        # add the influence of the Coriolis force
        momentum_max += 0.5 * coriolis_p * r_max**2

    # rescale the momentum using formula (36) in Emanuel and Rotunno 2011 with Ck == Cd
    r_max_norm = (d_centr / r_max) ** 2
    momentum = momentum_max * 2 * r_max_norm / (1 + r_max_norm)

    # extract the velocity from the rescaled momentum through division by r
    v_ang[mask_centr_close] = np.fmax(0, momentum / (d_centr + 1e-11))
    return v_ang


KN_TO_MS = (1.0 * ureg.knot).to(ureg.meter / ureg.second).magnitude
V_ANG_EARTH = 7.29e-5


def _vtrans(si_track: xr.Dataset, metric: str = "equirect"):
    """Translational vector and velocity (in m/s) at each track node.

    The track dataset is modified in place, with the following variables added:

    * vtrans (directional vectors of velocity, in meters per second)
    * vtrans_norm (absolute velocity in meters per second; the first velocity is always 0)

    The meridional component (v) of the vectors is listed first.

    Parameters
    ----------
    si_track : xr.Dataset
        Track information as returned by ``tctrack_to_si``. The data variables used by this function
        are "lat", "lon", and "tstep". The results are stored in place as new data
        variables "vtrans" and "vtrans_norm".
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".
    """
    npositions = si_track.sizes["time"]
    si_track["vtrans_norm"] = (["time"], np.zeros((npositions,)))
    si_track["vtrans"] = (["time", "component"], np.zeros((npositions, 2)))
    si_track["component"] = ("component", ["v", "u"])

    t_lat, t_lon = si_track["lat"].values, si_track["lon"].values
    norm, vec = u_coord.dist_approx(
        t_lat[:-1, None],
        t_lon[:-1, None],
        t_lat[1:, None],
        t_lon[1:, None],
        log=True,
        normalize=False,
        method=metric,
        units="m",
    )
    si_track["vtrans"].values[1:, :] = vec[:, 0, 0] / si_track["tstep"].values[1:, None]
    si_track["vtrans_norm"].values[1:] = norm[:, 0, 0] / si_track["tstep"].values[1:]

    # limit to 30 nautical miles per hour
    msk = si_track["vtrans_norm"].values > 30 * KN_TO_MS
    fact = 30 * KN_TO_MS / si_track["vtrans_norm"].values[msk]
    si_track["vtrans"].values[msk, :] *= fact[:, None]
    si_track["vtrans_norm"].values[msk] *= fact


def _coriolis_parameter(lat: np.ndarray) -> np.ndarray:
    """Compute the Coriolis parameter from latitude.

    Parameters
    ----------
    lat : np.ndarray
        Latitude (degrees).

    Returns
    -------
    cp : np.ndarray of same shape as input
        Coriolis parameter.
    """
    return 2 * V_ANG_EARTH * np.sin(np.radians(np.abs(lat)))


def compute_windfields_sparse(
    track: xr.Dataset,
    centroids: Centroids,
    idx_centr_filter: np.ndarray,
    model: str = "H08",
    model_kwargs: Optional[dict] = None,
    store_windfields: bool = False,
    metric: str = "equirect",
    intensity_thres: float = DEF_INTENSITY_THRES,
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Version of ``compute_windfields`` that returns sparse matrices and limits memory usage

    Parameters
    ----------
    track : xr.Dataset
        Single tropical cyclone track.
    centroids : Centroids
        Centroids instance.
    idx_centr_filter : np.ndarray
        Indices of centroids to restrict to (e.g. sufficiently close to coast).
    model : str, optional
        Parametric wind field model, one of "H1980" (the prominent Holland 1980 model),
        "H08" (Holland 1980 with b-value from Holland 2008), "H10" (Holland et al. 2010), or
        "ER11" (Emanuel and Rotunno 2011).
        Default: "H08".
    model_kwargs: dict, optional
        If given, forward these kwargs to the selected model. Default: None
    store_windfields : boolean, optional
        If True, store windfields. Default: False.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".
    intensity_thres : float, optional
        Wind speeds (in m/s) below this threshold are stored as 0. Default: 17.5
    max_dist_eye_km : float, optional
        No wind speed calculation is done for centroids with a distance (in km) to the TC
        center ("eye") larger than this parameter. Default: 300
    max_memory_gb : float, optional
        To avoid memory issues, the computation is done for chunks of the track sequentially.
        The chunk size is determined depending on the available memory (in GB). Default: 8

    Raises
    ------
    ValueError

    Returns
    -------
    intensity : csr_matrix
        Maximum wind speed in each centroid over the whole storm life time.
    windfields : csr_matrix or None
        If store_windfields is True, the full velocity vectors at each centroid and track position
        are stored in a sparse matrix of shape (npositions,  ncentroids * 2) that can be reshaped
        to a full ndarray of shape (npositions, ncentroids, 2).
        If store_windfields is False, None is returned.
    """
    try:
        mod_id = MODEL_VANG[model]
    except KeyError as err:
        raise ValueError(f"Model not implemented: {model}.") from err

    ncentroids = centroids.coord.shape[0]
    npositions = track.sizes["time"]
    windfields_shape = (npositions, ncentroids * 2)
    intensity_shape = (1, ncentroids)

    # initialise arrays for the assumption that no centroids are within reach
    windfields_sparse = (
        sparse.csr_matrix(([], ([], [])), shape=windfields_shape)
        if store_windfields
        else None
    )
    intensity_sparse = sparse.csr_matrix(([], ([], [])), shape=intensity_shape)

    # The wind field model requires at least two track positions because translational speed
    # as well as the change in pressure (in case of H08) are required.
    if npositions < 2:
        return intensity_sparse, windfields_sparse

    # convert track variables to SI units
    si_track = tctrack_to_si(track, metric=metric)

    # When done properly, finding and storing the close centroids is not a memory bottle neck and
    # can be done before chunking. Note that the longitudinal coordinates of `centroids_close` as
    # returned by `get_close_centroids` are normalized to be consistent with the coordinates in
    # `si_track`.
    centroids_close, mask_centr, mask_centr_alongtrack = get_close_centroids(
        si_track,
        centroids.coord[idx_centr_filter],
        max_dist_eye_km,
        metric=metric,
    )
    idx_centr_filter = idx_centr_filter[mask_centr]
    n_centr_close = centroids_close.shape[0]
    if n_centr_close == 0:
        return intensity_sparse, windfields_sparse

    # the total memory requirement in GB if we compute everything without chunking:
    # 8 Bytes per entry (float64), 10 arrays
    total_memory_gb = npositions * n_centr_close * 8 * 10 / 1e9
    if total_memory_gb > max_memory_gb and npositions > 2:
        # If the number of positions is down to 2 already, we cannot split any further. In that
        # case, we just take the risk and try to do the computation anyway. It might still work
        # since we have only computed an upper bound for the number of affected centroids.

        # Split the track into chunks, compute the result for each chunk, and combine:
        return _compute_windfields_sparse_chunked(
            mask_centr_alongtrack,
            track,
            centroids,
            idx_centr_filter,
            model=model,
            model_kwargs=model_kwargs,
            store_windfields=store_windfields,
            metric=metric,
            intensity_thres=intensity_thres,
            max_dist_eye_km=max_dist_eye_km,
            max_memory_gb=max_memory_gb,
        )

    windfields, idx_centr_reachable = _compute_windfields(
        si_track,
        centroids_close,
        mod_id,
        model_kwargs=model_kwargs,
        metric=metric,
        max_dist_eye_km=max_dist_eye_km,
    )
    idx_centr_filter = idx_centr_filter[idx_centr_reachable]
    npositions = windfields.shape[0]

    intensity = np.linalg.norm(windfields, axis=-1).max(axis=0)
    intensity[intensity < intensity_thres] = 0
    intensity_sparse = sparse.csr_matrix(
        (intensity, idx_centr_filter, [0, intensity.size]), shape=intensity_shape
    )
    intensity_sparse.eliminate_zeros()

    windfields_sparse = None
    if store_windfields:
        n_centr_filter = idx_centr_filter.size
        indices = np.zeros((npositions, n_centr_filter, 2), dtype=np.int64)
        indices[:, :, 0] = 2 * idx_centr_filter[None]
        indices[:, :, 1] = 2 * idx_centr_filter[None] + 1
        indices = indices.ravel()
        indptr = np.arange(npositions + 1) * n_centr_filter * 2
        windfields_sparse = sparse.csr_matrix(
            (windfields.ravel(), indices, indptr), shape=windfields_shape
        )
        windfields_sparse.eliminate_zeros()

    return intensity_sparse, windfields_sparse


def _compute_windfields_sparse_chunked(
    mask_centr_alongtrack: np.ndarray,
    track: xr.Dataset,
    *args,
    max_memory_gb: float = DEF_MAX_MEMORY_GB,
    **kwargs,
) -> Tuple[sparse.csr_matrix, Optional[sparse.csr_matrix]]:
    """Call ``compute_windfields_sparse`` for chunks of the track and re-assemble the results

    Parameters
    ----------
    mask_centr_alongtrack : np.ndarray of shape (npositions, ncentroids)
        Each row is a mask that indicates the centroids within reach for one track position.
    track : xr.Dataset
        Single tropical cyclone track.
    max_memory_gb : float, optional
        Maximum memory requirements (in GB) for the computation of a single chunk of the track.
        Default: 8
    args, kwargs :
        The remaining arguments are passed on to ``compute_windfields_sparse``.

    Returns
    -------
    intensity, windfields :
        See ``compute_windfields_sparse`` for a description of the return values.
    """
    npositions = track.sizes["time"]
    # The memory requirements for each track position are estimated for the case of 10 arrays
    # containing `nreachable` float64 (8 Byte) values each. The chunking is only relevant in
    # extreme cases with a very high temporal and/or spatial resolution.
    max_nreachable = max_memory_gb * 1e9 / (8 * 10 * npositions)
    split_pos = [0]
    chunk_size = 2
    while split_pos[-1] + chunk_size < npositions:
        chunk_size += 1
        # create overlap between consecutive chunks
        chunk_start = max(0, split_pos[-1] - 1)
        chunk_end = chunk_start + chunk_size
        nreachable = mask_centr_alongtrack[chunk_start:chunk_end].any(axis=0).sum()
        if nreachable > max_nreachable:
            split_pos.append(chunk_end - 1)
            chunk_size = 2
    split_pos.append(npositions)

    intensity = []
    windfields = []
    for prev_chunk_end, chunk_end in zip(split_pos[:-1], split_pos[1:]):
        chunk_start = max(0, prev_chunk_end - 1)
        inten, win = compute_windfields_sparse(
            track.isel(time=slice(chunk_start, chunk_end)),
            *args,
            max_memory_gb=max_memory_gb,
            **kwargs,
        )
        intensity.append(inten)
        windfields.append(win)

    intensity = sparse.csr_matrix(sparse.vstack(intensity).max(axis=0))
    if windfields[0] is not None:
        # eliminate the overlap between consecutive chunks
        windfields = [windfields[0]] + [win[1:, :] for win in windfields[1:]]
        windfields = sparse.vstack(windfields, format="csr")
    return intensity, windfields


def _compute_windfields(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    model: int,
    model_kwargs: Optional[dict] = None,
    metric: str = "equirect",
    max_dist_eye_km: float = DEF_MAX_DIST_EYE_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 1-minute sustained winds (in m/s) at 10 meters above ground

    In a first step, centroids within reach of the track are determined so that wind fields will
    only be computed and returned for those centroids. Still, since computing the distance of
    the storm center to the centroids is computationally expensive, make sure to pre-filter the
    centroids and call this function only for those centroids that are potentially affected.

    Parameters
    ----------
    si_track : xr.Dataset
        Output of ``tctrack_to_si``. Which data variables are used in the computation of the wind
        speeds depends on the selected model.
    centroids : np.ndarray with two dimensions
        Each row is a centroid [lat, lon]. Centroids that are not within reach of the track are
        ignored. Longitudinal coordinates are assumed to be normalized consistently with the
        longitudinal coordinates in ``si_track``.
    model : int
        Wind profile model selection according to MODEL_VANG.
    model_kwargs: dict, optional
        If given, forward these kwargs to the selected model. Default: None
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".
    max_dist_eye_km : float, optional
        No wind speed calculation is done for centroids with a distance (in km) to the TC center
        ("eye") larger than this parameter. Default: 300

    Returns
    -------
    windfields : np.ndarray of shape (npositions, nreachable, 2)
        Directional wind fields for each track position on those centroids within reach
        of the TC track. Note that the wind speeds at the first position are all zero because
        the discrete time derivatives involved in the process are implemented using backward
        differences. However, the first position is usually not relevant for impact calculations
        since it is far off shore.
    idx_centr_reachable : np.ndarray of shape (nreachable,)
        List of indices of input centroids within reach of the TC track.
    """
    # start with the assumption that no centroids are within reach
    npositions = si_track.sizes["time"]
    idx_centr_reachable = np.zeros((0,), dtype=np.int64)
    windfields = np.zeros((npositions, 0, 2), dtype=np.float64)

    # compute distances (in m) and vectors to all centroids
    [d_centr], [v_centr_normed] = u_coord.dist_approx(
        si_track["lat"].values[None],
        si_track["lon"].values[None],
        centroids[None, :, 0],
        centroids[None, :, 1],
        log=True,
        normalize=False,
        method=metric,
        units="m",
    )

    # exclude centroids that are too far from or too close to the eye
    mask_centr_close = (d_centr <= max_dist_eye_km * KM_TO_M) & (d_centr > 1)
    if not np.any(mask_centr_close):
        return windfields, idx_centr_reachable

    # restrict to the centroids that are within reach of any of the positions
    mask_centr_close_any = mask_centr_close.any(axis=0)
    mask_centr_close = mask_centr_close[:, mask_centr_close_any]
    d_centr = d_centr[:, mask_centr_close_any]
    v_centr_normed = v_centr_normed[:, mask_centr_close_any, :]

    # normalize the vectors pointing from the eye to the centroids
    v_centr_normed[~mask_centr_close] = 0
    v_centr_normed[mask_centr_close] /= d_centr[mask_centr_close, None]

    # derive (absolute) angular velocity from parametric wind profile
    v_ang_norm = compute_angular_windspeeds(
        si_track,
        d_centr,
        mask_centr_close,
        model,
        model_kwargs=model_kwargs,
    )

    # Influence of translational speed decreases with distance from eye.
    # The "absorbing factor" is according to the following paper (see Fig. 7):
    #
    #   Mouton, F. & Nordbeck, O. (2005). Cyclone Database Manager. A tool
    #   for converting point data from cyclone observations into tracks and
    #   wind speed profiles in a GIS. UNED/GRID-Geneva.
    #   https://unepgrid.ch/en/resource/19B7D302
    #
    t_rad_bc = np.broadcast_to(si_track["rad"].values[:, None], d_centr.shape)
    v_trans_corr = np.zeros_like(d_centr)
    v_trans_corr[mask_centr_close] = np.fmin(
        1, t_rad_bc[mask_centr_close] / d_centr[mask_centr_close]
    )

    if model in [MODEL_VANG["H08"], MODEL_VANG["H10"]]:
        # In these models, v_ang_norm already contains vtrans_norm, so subtract it first, before
        # converting to vectors and then adding (vectorial) vtrans again. Make sure to apply the
        # "absorbing factor" in both steps:
        vtrans_norm_bc = np.broadcast_to(
            si_track["vtrans_norm"].values[:, None], d_centr.shape
        )
        v_ang_norm[mask_centr_close] -= (
            vtrans_norm_bc[mask_centr_close] * v_trans_corr[mask_centr_close]
        )

    # vectorial angular velocity
    windfields = (
        si_track.attrs["latsign"]
        * np.array([1.0, -1.0])[..., :]
        * v_centr_normed[:, :, ::-1]
    )
    windfields[mask_centr_close] *= v_ang_norm[mask_centr_close, None]

    # add angular and corrected translational velocity vectors
    windfields[1:] += si_track["vtrans"].values[1:, None, :] * v_trans_corr[1:, :, None]
    windfields[np.isnan(windfields)] = 0
    windfields[0, :, :] = 0
    [idx_centr_reachable] = mask_centr_close_any.nonzero()
    return windfields, idx_centr_reachable


def tctrack_to_si(
    track: xr.Dataset,
    metric: str = "equirect",
) -> xr.Dataset:
    """Convert track variables to SI units and prepare for wind field computation

    In addition to unit conversion, the variable names are shortened, the longitudinal coordinates
    are normalized and additional variables are defined:

    * cp (coriolis parameter)
    * pdelta (difference between environmental and central pressure, always strictly positive)
    * vtrans (translational velocity vectors)
    * vtrans_norm (absolute value of translational speed)

    Furthermore, some scalar variables are stored as attributes:

    * latsign (1.0 if the track is located on the northern and -1.0 if on southern hemisphere)
    * mid_lon (the central longitude that was used to normalize the longitudinal coordinates)

    Finally, some corrections are applied to variables:

    * clip central pressure values so that environmental pressure values are never exceeded
    * extrapolate radius of max wind from pressure if missing

    Parameters
    ----------
    track : xr.Dataset
        Track information.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".

    Returns
    -------
    xr.Dataset
    """
    si_track = track[["lat", "lon", "time"]].copy()
    si_track["tstep"] = track["time_step"] * H_TO_S
    si_track["env"] = track["environmental_pressure"] * MBAR_TO_PA

    # we support some non-standard unit names
    unit_replace = {"mb": "mbar", "kn": "knots"}
    configs = [
        ("central_pressure", "cen", "Pa"),
        ("max_sustained_wind", "vmax", "m/s"),
    ]
    for long_name, var_name, si_unit in configs:
        unit = track.attrs[f"{long_name}_unit"]
        unit = unit_replace.get(unit, unit)
        try:
            conv_factor = ureg(unit).to(si_unit).magnitude
        except Exception as ex:
            raise ValueError(
                f"The {long_name}_unit '{unit}' in the provided track is not supported."
            ) from ex
        si_track[var_name] = track[long_name] * conv_factor

    # normalize longitudinal coordinates
    si_track.attrs["mid_lon"] = 0.5 * sum(u_coord.lon_bounds(si_track["lon"].values))
    u_coord.lon_normalize(si_track["lon"].values, center=si_track.attrs["mid_lon"])

    # make sure that central pressure never exceeds environmental pressure
    pres_exceed_msk = (si_track["cen"] > si_track["env"]).values
    si_track["cen"].values[pres_exceed_msk] = si_track["env"].values[pres_exceed_msk]

    # extrapolate radius of max wind from pressure if not given
    si_track["rad"] = track["radius_max_wind"].copy()
    si_track["rad"].values[:] = estimate_rmw(
        si_track["rad"].values,
        si_track["cen"].values / MBAR_TO_PA,
    )
    si_track["rad"] *= NM_TO_KM * KM_TO_M

    hemisphere = "N"
    if np.count_nonzero(si_track["lat"] < 0) > np.count_nonzero(si_track["lat"] > 0):
        hemisphere = "S"
    si_track.attrs["latsign"] = 1.0 if hemisphere == "N" else -1.0

    # add translational speed of track at every node (in m/s)
    _vtrans(si_track, metric=metric)

    # add Coriolis parameter
    si_track["cp"] = ("time", _coriolis_parameter(si_track["lat"].values))

    # add pressure drop
    si_track["pdelta"] = np.fmax(np.spacing(1), si_track["env"] - si_track["cen"])

    return si_track


def get_close_centroids(
    si_track: xr.Dataset,
    centroids: np.ndarray,
    buffer_km: float,
    metric: str = "equirect",
) -> np.ndarray:
    """Check whether centroids lay within a buffer around track positions

    Note that, hypothetically, a problem occurs when a TC track is travelling so far in longitude
    that adding a buffer exceeds 360 degrees (i.e. crosses the antimeridian), which is physically
    impossible, but might happen with synthetical or test data.

    Parameters
    ----------
    si_track : xr.Dataset with dimension "time"
        Track information as returned by ``tctrack_to_si``. Hence, longitudinal coordinates are
        normalized around the central longitude stored in the "mid_lon" attribute. This makes sure
        that the buffered bounding box around the track does not cross the antimeridian. The data
        variables used by this function are "lat", and "lon".
    centroids : np.ndarray of shape (ncentroids, 2)
        Coordinates of centroids, each row is a pair [lat, lon]. The longitudinal coordinates are
        normalized within this function to be consistent with the track coordinates.
    buffer_km : float
        Size of the buffer (in km). The buffer is converted to a lat/lon buffer, rescaled in
        longitudinal direction according to the t_lat coordinates.
    metric : str, optional
        Specify an approximation method to use for earth distances: "equirect" (faster) or
        "geosphere" (more accurate). See ``dist_approx`` function in ``climada.util.coordinates``.
        Default: "equirect".

    Returns
    -------
    centroids_close_normalized : np.ndarray of shape (nclose, 2)
        Coordinates of close centroids, each row is a pair [lat, lon]. The normalization of
        longitudinal coordinates is consistent with the track coordinates.
    mask_centr : np.ndarray of shape (ncentroids,)
        Mask that is True for close centroids and False for other centroids.
    mask_centr_alongtrack : np.ndarray of shape (npositions, nclose)
        Each row is a mask that indicates the centroids within reach for one track position. Note
        that these masks refer only to the "close centroids" to reduce memory requirements. The
        number of positions ``npositions`` corresponds to the size of the "time" dimension of
        ``si_track``.
    """
    npositions = si_track.sizes["time"]
    ncentroids = centroids.shape[0]
    t_lat, t_lon = si_track["lat"].values, si_track["lon"].values
    centr_lat, centr_lon = centroids[:, 0].copy(), centroids[:, 1].copy()

    # Normalize longitudinal coordinates of centroids.
    u_coord.lon_normalize(centr_lon, center=si_track.attrs["mid_lon"])

    # Restrict to the bounding box of the whole track first (this can already reduce the number of
    # centroids that are considered by a factor larger than 30).
    buffer_lat = buffer_km / u_const.ONE_LAT_KM
    buffer_lon = buffer_km / (
        u_const.ONE_LAT_KM
        * np.cos(np.radians(np.fmin(89.999, np.abs(centr_lat) + buffer_lat)))
    )
    [idx_close] = (
        (t_lat.min() - centr_lat <= buffer_lat)
        & (centr_lat - t_lat.max() <= buffer_lat)
        & (t_lon.min() - centr_lon <= buffer_lon)
        & (centr_lon - t_lon.max() <= buffer_lon)
    ).nonzero()
    centr_lat = centr_lat[idx_close]
    centr_lon = centr_lon[idx_close]

    # Restrict to bounding boxes of each track position.
    buffer_lat = buffer_km / u_const.ONE_LAT_KM
    buffer_lon = buffer_km / (
        u_const.ONE_LAT_KM
        * np.cos(np.radians(np.fmin(89.999, np.abs(t_lat[:, None]) + buffer_lat)))
    )
    [idx_close_sub] = (
        (
            (t_lat[:, None] - buffer_lat <= centr_lat[None])
            & (t_lat[:, None] + buffer_lat >= centr_lat[None])
            & (t_lon[:, None] - buffer_lon <= centr_lon[None])
            & (t_lon[:, None] + buffer_lon >= centr_lon[None])
        )
        .any(axis=0)
        .nonzero()
    )
    idx_close = idx_close[idx_close_sub]
    centr_lat = centr_lat[idx_close_sub]
    centr_lon = centr_lon[idx_close_sub]

    # Restrict to metric distance radius around each track position.
    #
    # We do the distance computation for chunks of the track since computing the distance requires
    # npositions*ncentroids*8*3 Bytes of memory. For example, Hurricane FAITH's life time was more
    # than 500 hours. At 0.5-hourly resolution and 1,000,000 centroids, that's 24 GB of memory for
    # FAITH. With a chunk size of 10, this figure is down to 240 MB. The final along-track mask
    # will require 1.0 GB of memory.
    chunk_size = 10
    chunks = np.split(
        np.arange(npositions), np.arange(chunk_size, npositions, chunk_size)
    )
    mask_centr_alongtrack = np.concatenate(
        [
            (
                u_coord.dist_approx(
                    t_lat[None, chunk],
                    t_lon[None, chunk],
                    centr_lat[None],
                    centr_lon[None],
                    normalize=False,
                    method=metric,
                    units="km",
                )[0]
                <= buffer_km
            )
            for chunk in chunks
        ],
        axis=0,
    )
    [idx_close_sub] = mask_centr_alongtrack.any(axis=0).nonzero()
    idx_close = idx_close[idx_close_sub]
    centr_lat = centr_lat[idx_close_sub]
    centr_lon = centr_lon[idx_close_sub]
    mask_centr_alongtrack = mask_centr_alongtrack[:, idx_close_sub]

    # Derive mask from index.
    mask_centr = np.zeros((ncentroids,), dtype=bool)
    mask_centr[idx_close] = True

    centroids_close_normalized = np.stack([centr_lat, centr_lon], axis=1)
    return centroids_close_normalized, mask_centr, mask_centr_alongtrack
