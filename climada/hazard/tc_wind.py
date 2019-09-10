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

Define wind fields for TCs.
"""

__all__ = ['windfield']

import numpy as np
from numpy import linalg as LA
from numba import jit
from pint import UnitRegistry

from climada.util.interpolation import dist_approx


CENTR_NODE_MAX_DIST_KM = 300
""" Maximum distance between centroid and TC track node in km """

@jit
def windfield(track, centroids, coastal_idx, model, inten_thresh):
    """ Compute windfields (in m/s) in centroids using Holland model 08.

    Parameters:
        track (xr.Dataset): track infomation
        centroids (2d np.array): each row is a centroid [lat, lon]
        coastal_idx (1d np.array): centroids indices that are close to coast
        model (int): Holland model selection according to MODEL_VANG

    Returns:
        np.array
    """
    np.warnings.filterwarnings('ignore')
    # Make sure that CentralPressure never exceeds EnvironmentalPressure
    up_pr = np.argwhere(track.central_pressure.values >
                        track.environmental_pressure.values)
    track.central_pressure.values[up_pr] = \
        track.environmental_pressure.values[up_pr]

    # Extrapolate RadiusMaxWind from pressure if not given
    ureg = UnitRegistry()
    track['radius_max_wind'] = ('time', _extra_rad_max_wind( \
        track.central_pressure.values, track.radius_max_wind.values, ureg))

    # Track translational speed at every node
    v_trans = _vtrans(track.lat.values, track.lon.values,
                      track.time_step.values, ureg)

    # Compute windfield
    intensity = np.zeros((centroids.shape[0], ))
    intensity[coastal_idx] = _wind_per_node(centroids[coastal_idx, :], track,
                                            v_trans, model, inten_thresh)

    return intensity

@jit
def _vtrans(t_lat, t_lon, t_tstep, ureg):
    """ Translational spped at every track node.

    Parameters:
        t_lat (np.array): track latitudes
        t_lon (np.array): track longitudes
        t_tstep (np.array): track time steps
        ureg (UnitRegistry): units handler

    Returns:
        np.array
    """
    v_trans = dist_approx(t_lat[:-1], t_lon[:-1],
                          np.cos(np.radians(t_lat[:-1])), t_lat[1:],
                          t_lon[1:]) / t_tstep[1:]
    v_trans = (v_trans * ureg.km/ureg.hour).to(ureg.meter/ureg.second).magnitude

    # nautical miles/hour, limit to 30 nmph
    v_max = (30*ureg.knot).to(ureg.meter/ureg.second).magnitude
    v_trans[v_trans > v_max] = v_max
    return v_trans

@jit
def _extra_rad_max_wind(t_cen, t_rad, ureg):
    """ Extrapolate RadiusMaxWind from pressure and change to km.

    Parameters:
        t_cen (np.array): track central pressures
        t_rad (np.array): track radius of maximum wind
        ureg (UnitRegistry): units handler

    Returns:
        np.array
    """
    # TODO: always extrapolate???!!!
    # rmax thresholds in nm
    rmax_1, rmax_2, rmax_3 = 15, 25, 50
    # pressure in mb
    pres_1, pres_2, pres_3 = 950, 980, 1020
    t_rad[t_cen <= pres_1] = rmax_1

    to_change = np.logical_and(t_cen > pres_1, t_cen <= pres_2).nonzero()[0]
    t_rad[to_change] = (t_cen[to_change] - pres_1) * \
        (rmax_2 - rmax_1)/(pres_2 - pres_1) + rmax_1

    to_change = np.argwhere(t_cen > pres_2).squeeze()
    t_rad[to_change] = (t_cen[to_change] - pres_2) * \
        (rmax_3 - rmax_2)/(pres_3 - pres_2) + rmax_2

    return (t_rad * ureg.nautical_mile).to(ureg.kilometer).magnitude

@jit(parallel=True)
def _wind_per_node(coastal_centr, track, v_trans, model, inten_thresh):
    """ Compute sustained winds at each centroid.

    Parameters:
        coastal_centr (2d np.array): centroids
        track (xr.Dataset): track latitudes
        v_trans (np.array): track translational velocity
        model (int): Holland model selection according to MODEL_VANG

    Returns:
        2d np.array
    """

    t_lat, t_lon = track.lat.values, track.lon.values
    t_rad, t_env = track.radius_max_wind.values, track.environmental_pressure.values
    t_cen, t_tstep = track.central_pressure.values, track.time_step.values

    centr_cos_lat = np.cos(np.radians(coastal_centr[:, 0]))
    intensity = np.zeros((coastal_centr.shape[0],))

    n_nodes = t_lat.size
    if 'n_nodes' in track.attrs:
        n_nodes = track.attrs['n_nodes']

    for i_node in range(1, n_nodes):
        # compute distance to all centroids
        r_arr = dist_approx(coastal_centr[:, 0], coastal_centr[:, 1], \
            centr_cos_lat, t_lat[i_node], t_lon[i_node])

        # Choose centroids that are close enough
        close_centr = np.argwhere(r_arr < CENTR_NODE_MAX_DIST_KM).reshape(-1,)
        r_arr = r_arr[close_centr]

        # translational component
        if i_node < t_lat.size-1:
            v_trans_corr = _vtrans_correct(t_lat[i_node:i_node+2], \
                t_lon[i_node:i_node+2], t_rad[i_node], \
                coastal_centr[close_centr, :], r_arr)
        else:
            v_trans_corr = np.zeros((r_arr.size,))

        # angular component
        v_ang = _vang_sym(t_env[i_node], t_cen[i_node-1:i_node+1],
                          t_lat[i_node], t_tstep[i_node], t_rad[i_node],
                          r_arr, v_trans[i_node-1], model)

        v_full = v_trans[i_node-1] * v_trans_corr + v_ang
        v_full[np.isnan(v_full)] = 0
        v_full[v_full < inten_thresh] = 0

        # keep maximum instantaneous wind
        intensity[close_centr] = np.maximum(intensity[close_centr], v_full)

    return intensity

@jit
def _vtrans_correct(t_lats, t_lons, t_rad, close_centr, r_arr):
    """ Compute Hollands translational wind corrections. Returns factor.

    Parameters:
        t_lats (tuple): current and next latitude
        t_lats (tuple): current and next longitude
        t_rad (float): current radius of maximum wind
        close_centr (np.array): centroids
        r_arr (np.array): distance from current node to all centroids

    Returns:
        np.array
    """
    # we use the scalar product of the track forward vector and the vector
    # towards each centroid to figure the angle between and hence whether
    # the translational wind needs to be added (on the right side of the
    # track for Northern hemisphere) and to which extent (100% exactly 90
    # to the right of the track, zero in front of the track)
    lon, nex_lon = t_lons
    lat, nex_lat = t_lats

    # hence, rotate track forward vector 90 degrees clockwise, i.e.
    node_dy = -nex_lon + lon
    node_dx = nex_lat - lat

    # the vector towards each centroid
    centroids_dlon = close_centr[:, 1] - lon
    centroids_dlat = close_centr[:, 0] - lat

    # scalar product, a*b=|a|*|b|*cos(phi), phi angle between vectors
    cos_phi = (centroids_dlon * node_dx + centroids_dlat * node_dy) / \
        LA.norm([centroids_dlon, centroids_dlat], axis=0) / LA.norm([node_dx, node_dy])

    # southern hemisphere
    if lat < 0:
        cos_phi = -cos_phi

    # calculate v_trans wind field array assuming that
    # - effect of v_trans decreases with distance from eye (r_arr_normed)
    # - v_trans is added 100% to the right of the track, 0% in front (cos_phi)
    r_arr_normed = t_rad / r_arr
    r_arr_normed[r_arr_normed > 1] = 1

    return np.multiply(r_arr_normed, cos_phi)

@jit(['f8(f8, f8, f8, f8, f8, f8, f8)'], nopython=True)
def _bs_hol08(v_trans, penv, pcen, prepcen, lat, hol_xx, tint):
    """ Halland's 2008 b value computation.

    Parameters:
        v_trans (float): translational wind (m/s)
        penv (float): environmental pressure (hPa)
        pcen (float): central pressure (hPa)
        prepcen (float): previous central pressure (hPa)
        lat (float): latitude (degrees)
        hol_xx (float): Holland's xx value
        tint (float): time step (h)

    Returns:
        float
    """
    return -4.4e-5 * (penv - pcen)**2 + 0.01 * (penv-pcen) + \
        0.03 * (pcen - prepcen) / tint - 0.014 * abs(lat) + \
        0.15 * v_trans**hol_xx + 1.0

@jit(nopython=True)
def _stat_holland(r_arr, r_max, hol_b, penv, pcen, ycoord):
    """ Holland symmetric and static wind field (in m/s) according to
    Holland1980 or Holland2008m depending on hol_b parameter.

    Parameters:
        r_arr (np.array): distance between coastal centroids and track node
        r_max (float): radius_max_wind
        hol_b (float): Holland's b parameter
        penv (float): environmental pressure
        pcen (float): central pressure
        ycoord (float): latitude

    Returns:
        np.array
    """
    rho = 1.15
    f_val = 2 * 0.0000729 * np.sin(np.radians(np.abs(ycoord)))
    r_arr_mult = 0.5 * 1000 * r_arr * f_val
    # units are m/s
    r_max_norm = (r_max/r_arr)**hol_b
    return np.sqrt(100 * hol_b / rho * r_max_norm * (penv - pcen) *
                   np.exp(-r_max_norm) + r_arr_mult**2) - r_arr_mult

@jit(nopython=True)
def _vang_sym(t_env, t_cens, t_lat, t_step, t_rad, r_arr, v_trans, model):
    """ Compute symmetric and static wind field (in m/s) filed (angular
    wind component.

    Parameters:
        t_env (float): environmental pressures
        t_cens (tuple): previous and current central pressures
        t_lat (float): latitude
        t_tstep (float): time steps
        t_rad (float): radius of maximum wind
        r_arr (np.array): distance from current node to all centroids
        v_trans (float): translational wind field
        model (int): Holland model to use, default 2008.

    Returns:
        np.array
    """
    # data for windfield calculation
    prev_pres, pres = t_cens
    hol_xx = 0.6 * (1. - (t_env - pres) / 215)
    if model == 0:
        # adjust pressure at previous track point
        if prev_pres < 850:
            prev_pres = pres
        hol_b = _bs_hol08(v_trans, t_env, pres, prev_pres, t_lat, hol_xx, t_step)
    else:
        # TODO H80: b=b_value(v_trans,vmax,penv,pcen,rho);
        raise NotImplementedError

    return _stat_holland(r_arr, t_rad, hol_b, t_env, pres, t_lat)
