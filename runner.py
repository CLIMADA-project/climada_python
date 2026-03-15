#!/usr/bin/env python3
"""
CLIMADA runner: Computes hurricane loss distribution for a TC scenario.

Input JSON (positional arg or stdin):
  region            : "miami" (default) or {"lat_min", "lat_max", "lon_min", "lon_max"}
  storm_category    : Saffir-Simpson category 1-5 (default 4)
  n_tracks          : ensemble size (default 50)
  total_exposed_value_usd : override total exposure (default 800e9 for Miami-Dade)

Output JSON (stdout):
  scenario, storm_category, region, n_tracks
  aai_agg_usd             : Average Annual Impact USD
  total_exposed_value_usd : total value of exposed assets
  loss_ratio              : aai / total_value
  event_loss_table        : list of {event_id, loss_usd, frequency, return_period_years}
  loss_exceedance_curve   : {return_periods_years, losses_usd}
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import xarray as xr

from climada.engine import ImpactCalc
from climada.entity import Exposures, ImpactFuncSet
from climada.entity.impact_funcs.trop_cyclone import ImpfTropCyclone
from climada.hazard.centroids import Centroids
from climada.hazard.tc_tracks import TCTracks, set_category
from climada.hazard.trop_cyclone import TropCyclone

import geopandas as gpd
from shapely.geometry import Point

# Saffir-Simpson category -> peak wind in knots
CAT_WIND_KT = {1: 75, 2: 90, 3: 110, 4: 125, 5: 155}

# Miami-Dade area default bounds (lat/lon)
REGION_BOUNDS = {
    "miami": {"lat_min": 25.1, "lat_max": 26.4, "lon_min": -80.9, "lon_max": -80.0},
}

# Historical frequency: Cat 4+ landfalling Florida ~1/25 years
# We distribute that frequency equally across the ensemble tracks
ANNUAL_FREQ = 1 / 25.0


def parse_region(region_inp):
    if isinstance(region_inp, dict):
        return region_inp
    name = str(region_inp).lower()
    if name in REGION_BOUNDS:
        return REGION_BOUNDS[name]
    raise ValueError(f"Unknown region '{region_inp}'. Use 'miami' or a dict with lat/lon bounds.")


def make_synthetic_tracks(n_tracks, category, region_bounds):
    """Build a small ensemble of synthetic TC tracks approaching the region."""
    peak_kt = CAT_WIND_KT[category]

    # Track landfall target: centre of region's western edge
    lf_lat = (region_bounds["lat_min"] + region_bounds["lat_max"]) / 2
    lf_lon = region_bounds["lon_min"] - 0.1   # just offshore

    rng = np.random.default_rng(42)
    tracks = []

    for i in range(n_tracks):
        lat_jitter = rng.uniform(-0.4, 0.4)
        lon_jitter = rng.uniform(-0.4, 0.4)
        wind_jitter = rng.uniform(0.93, 1.07)

        n_steps = 25
        # Track: starts well offshore (east), crosses landfall point, continues inland (west)
        start_lat = lf_lat - 1.5 + lat_jitter
        end_lat   = lf_lat + 1.5 + lat_jitter
        start_lon = lf_lon + 12.0 + lon_jitter
        end_lon   = lf_lon - 2.5 + lon_jitter

        lats = np.linspace(start_lat, end_lat, n_steps)
        lons = np.linspace(start_lon, end_lon, n_steps)

        # 6-hourly time steps
        times = pd.date_range("2020-09-10 00:00", periods=n_steps, freq="6h")

        # Wind profile: ramp up to peak at step ~18 (near landfall), decay inland
        lf_step = 18
        winds_raw = np.concatenate([
            np.linspace(35, peak_kt * wind_jitter, lf_step),
            np.linspace(peak_kt * wind_jitter, 40, n_steps - lf_step),
        ])
        # Clip to physical range
        winds = np.clip(winds_raw, 20, 165).astype(float)

        # Pressure (hPa) approximately inversely related to wind (empirical)
        env_pres = 1013.0
        min_pres = 935.0 if category >= 4 else 960.0
        # Linear mapping: 35 kt -> ~1005 hPa, peak_kt -> min_pres
        pressures = env_pres - (winds - 35) * (env_pres - min_pres) / (peak_kt - 35)
        pressures = np.clip(pressures, min_pres, env_pres)

        # Radius of maximum wind (nautical miles), typical Cat 4
        rmw = np.full(n_steps, 25.0)

        df = pd.DataFrame({
            "time": times,
            "lat": lats,
            "lon": lons,
            "max_sustained_wind": winds,
            "central_pressure": pressures,
            "environmental_pressure": np.full(n_steps, env_pres),
            "radius_max_wind": rmw,
            "basin": ["NA"] * n_steps,
            "time_step": np.full(n_steps, 6.0),
        })

        ds = xr.Dataset.from_dataframe(df.set_index("time"))
        # .copy() ensures writable arrays — CLIMADA normalises lon in-place
        ds.coords["lat"] = ("time", ds["lat"].values.copy())
        ds.coords["lon"] = ("time", ds["lon"].values.copy())
        ds["basin"] = ds["basin"].astype("<U2")
        ds.attrs = {
            "max_sustained_wind_unit": "kn",
            "central_pressure_unit": "mb",
            "sid": f"SYN_{category}_{i:03d}",
            "name": f"SYN_CAT{category}_{i:03d}",
            "orig_event_flag": False,
            "data_provider": "synthetic",
            "id_no": float(i),
            "category": set_category(winds, "kn"),
        }
        tracks.append(ds)

    tc_tracks = TCTracks(data=tracks)
    return tc_tracks


def make_exposure(region_bounds, total_value_usd, grid_res=0.05):
    """Create a uniform synthetic exposure grid over the region."""
    lats = np.arange(region_bounds["lat_min"], region_bounds["lat_max"], grid_res)
    lons = np.arange(region_bounds["lon_min"], region_bounds["lon_max"], grid_res)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    n_pts = len(lat_flat)
    value_per_pt = total_value_usd / n_pts

    gdf = gpd.GeoDataFrame(
        {
            "value": np.full(n_pts, value_per_pt),
            "impf_TC": np.ones(n_pts, dtype=int),
        },
        geometry=[Point(lon, lat) for lon, lat in zip(lon_flat, lat_flat)],
        crs="EPSG:4326",
    )
    exp = Exposures(gdf)
    exp.ref_year = 2020
    exp.value_unit = "USD"
    exp.check()
    return exp


def compute_ep_curve(at_event, frequency, return_periods):
    """Compute losses at given return periods from event loss table."""
    # Sort events by loss descending; cumulative frequency = exceedance probability
    order = np.argsort(at_event)[::-1]
    sorted_losses = at_event[order]
    exceedance_prob = np.cumsum(frequency[order])

    losses_at_rp = []
    for rp in return_periods:
        exc_prob_target = 1.0 / rp
        # Interpolate
        if exc_prob_target < exceedance_prob.min():
            losses_at_rp.append(float(sorted_losses[0]))
        elif exc_prob_target > exceedance_prob.max():
            losses_at_rp.append(0.0)
        else:
            loss = float(np.interp(exc_prob_target, exceedance_prob[::-1], sorted_losses[::-1]))
            losses_at_rp.append(loss)
    return losses_at_rp


def main():
    parser = argparse.ArgumentParser(description="CLIMADA TC hurricane loss runner")
    parser.add_argument("input", nargs="?", default="-", help="Input JSON file (or - for stdin)")
    args = parser.parse_args()

    if args.input == "-":
        inp = json.load(sys.stdin)
    else:
        with open(args.input) as f:
            inp = json.load(f)

    region_inp = inp.get("region", "miami")
    category = int(inp.get("storm_category", 4))
    n_tracks = int(inp.get("n_tracks", 50))
    total_value = float(inp.get("total_exposed_value_usd", 800e9))

    if category not in CAT_WIND_KT:
        raise ValueError(f"storm_category must be 1-5, got {category}")

    region_bounds = parse_region(region_inp)

    # --- Hazard ---
    tc_tracks = make_synthetic_tracks(n_tracks, category, region_bounds)

    # Centroids: cover region + buffer so wind fields extend beyond exposure
    buf = 1.5
    lats_c = np.arange(region_bounds["lat_min"] - buf, region_bounds["lat_max"] + buf, 0.05)
    lons_c = np.arange(region_bounds["lon_min"] - buf, region_bounds["lon_max"] + buf, 0.05)
    lat_c, lon_c = np.meshgrid(lats_c, lons_c, indexing="ij")
    centroids = Centroids(lat=lat_c.flatten(), lon=lon_c.flatten())

    tc_haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids)

    # Assign per-track frequency so total frequency = ANNUAL_FREQ
    tc_haz.frequency = np.full(n_tracks, ANNUAL_FREQ / n_tracks)

    # --- Exposure ---
    exp = make_exposure(region_bounds, total_value)

    # --- Impact functions ---
    # Emanuel 2011, calibrated for USA Atlantic hurricanes, impf_id=1 matches exposure column
    impf_tc = ImpfTropCyclone.from_emanuel_usa(impf_id=1)
    impf_set = ImpactFuncSet()
    impf_set.append(impf_tc)

    # --- Impact calculation ---
    imp_calc = ImpactCalc(exp, impf_set, tc_haz)
    impact = imp_calc.impact(save_mat=False)

    # --- Format output ---
    return_periods = [10, 25, 50, 100, 200, 250, 500, 1000]
    ep_losses = compute_ep_curve(impact.at_event, impact.frequency, return_periods)

    event_loss_table = []
    for idx, (ev_id, loss, freq) in enumerate(
        zip(tc_haz.event_name, impact.at_event, impact.frequency)
    ):
        rp = float(1.0 / freq) if freq > 0 else float("inf")
        event_loss_table.append({
            "event_id": str(ev_id),
            "loss_usd": float(loss),
            "frequency": float(freq),
            "return_period_years": round(rp, 1),
        })
    # Sort by loss descending
    event_loss_table.sort(key=lambda x: x["loss_usd"], reverse=True)

    result = {
        "scenario": f"Category {category} hurricane, {region_inp} area",
        "storm_category": category,
        "region": region_inp,
        "n_tracks": n_tracks,
        "aai_agg_usd": float(impact.aai_agg),
        "total_exposed_value_usd": total_value,
        "loss_ratio": float(impact.aai_agg / total_value) if total_value > 0 else 0.0,
        "event_loss_table": event_loss_table,
        "loss_exceedance_curve": {
            "return_periods_years": return_periods,
            "losses_usd": ep_losses,
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
