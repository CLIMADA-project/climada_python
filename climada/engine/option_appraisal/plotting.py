"""Plotting functions for the results of a :class:`StaticAppraiser`.

All functions take the tidy DataFrames returned by the appraiser, return the
matplotlib ``Axes`` they drew on (an array of ``Axes`` for multi-panel plots),
and accept an existing ``ax``/``axes`` to draw on.

Data preparation is kept in small private helpers so that it can be tested
independently of the plotting.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from climada.engine.option_appraisal.constants import (
    AVERTED_RISK_NAME,
    MEASURE_IMPL_COST_NAME,
)
from climada.trajectories.constants import (
    COORD_ID_COL_NAME,
    DATE_COL_NAME,
    GROUP_COL_NAME,
    MEASURE_COL_NAME,
    METRIC_COL_NAME,
    NO_MEASURE_VALUE,
    RISK_COL_NAME,
)

LOGGER = logging.getLogger(__name__)

BASELINE_COLOR = "0.5"
RATIO_COL = "benefit_cost_ratio"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clean(
    df: pd.DataFrame, group: str | None = None, drop_baseline: bool = False
) -> pd.DataFrame:
    """Copy `df`, cast measure and group columns to str, and filter rows.

    Parameters
    ----------
    df : pd.DataFrame
        A tidy metric DataFrame.
    group : str, optional
        If given, keep only the rows of this group.
    drop_baseline : bool, optional
        Whether to drop the rows without measure. Defaults to False.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If no row is left.
    """
    df = df.copy()
    for col in (MEASURE_COL_NAME, GROUP_COL_NAME):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if group is not None:
        df = df[df[GROUP_COL_NAME] == str(group)]
    if drop_baseline:
        df = df[df[MEASURE_COL_NAME] != NO_MEASURE_VALUE]
    if df.empty:
        raise ValueError("No data left to plot after filtering.")
    return df


def _date_labels(dates: Iterable) -> list[str]:
    """Years if all dates are 1 January, ISO dates otherwise."""
    dates = pd.DatetimeIndex(dates)
    if (dates.month == 1).all() and (dates.day == 1).all():
        return [str(y) for y in dates.year]
    return [d.strftime("%Y-%m-%d") for d in dates]


def _measure_colors(names: Iterable[str]) -> dict[str, tuple]:
    """One consistent color per measure name (baseline is gray)."""
    cmap = plt.get_cmap("tab10")
    colors = {
        name: cmap(i % 10)
        for i, name in enumerate(n for n in names if n != NO_MEASURE_VALUE)
    }
    colors[NO_MEASURE_VALUE] = BASELINE_COLOR
    return colors


def _get_ax(ax: Axes | None) -> Axes:
    return plt.subplots()[1] if ax is None else ax


def _get_axes(axes, n: int) -> np.ndarray:
    """Return `n` axes in a flat array, creating a row of panels if needed."""
    if axes is None:
        _, axes = plt.subplots(
            1, n, figsize=(4 * n, 3.5), sharey=True, squeeze=False, layout="constrained"
        )
    axes = np.atleast_1d(axes).ravel()
    if len(axes) < n:
        raise ValueError(f"Need {n} axes, got {len(axes)}.")
    return axes


def _grouped_bars(ax: Axes, table: pd.DataFrame, colors: dict) -> None:
    """Draw one group of bars per row of `table`, one bar per column."""
    n = table.shape[1]
    width = 0.8 / n
    x = np.arange(len(table))
    for k, col in enumerate(table.columns):
        ax.bar(
            x + (k - (n - 1) / 2) * width,
            table[col].to_numpy(),
            width,
            label=col,
            color=colors[col],
        )


def _return_periods(df: pd.DataFrame, rp_col: str | None) -> pd.Series:
    """Return period of each row, from `rp_col` or parsed from the metric name."""
    if rp_col is not None:
        return df[rp_col].astype(float)
    rp = (
        df[METRIC_COL_NAME]
        .astype(str)
        .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        .astype(float)
    )
    if rp.isna().any():
        raise ValueError(
            f"Could not infer return periods from column '{METRIC_COL_NAME}'; "
            "pass `rp_col`."
        )
    return rp


# --------------------------------------------------------------------------- #
# Static appraiser plots
# --------------------------------------------------------------------------- #
def plot_averted_risk(
    df: pd.DataFrame, *, group: str | None = None, ax: Axes | None = None
) -> Axes:
    """Grouped bars of the averted risk, one group of bars per date.

    Parameters
    ----------
    df : pd.DataFrame
        Output of e.g. `StaticAppraiser.aai_metrics()`.
    group : str, optional
        Exposure group to plot. Needed if `df` contains several groups.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes

    Raises
    ------
    ValueError
        If several rows remain for the same date and measure.
    """
    df = _clean(df, group, drop_baseline=True)
    table = df.pivot(
        index=DATE_COL_NAME, columns=MEASURE_COL_NAME, values=AVERTED_RISK_NAME
    )
    table = table[list(pd.unique(df[MEASURE_COL_NAME]))]
    ax = _get_ax(ax)
    _grouped_bars(ax, table, _measure_colors(table.columns))
    ax.set_xticks(np.arange(len(table)), _date_labels(table.index))
    ax.set_ylabel("Averted risk")
    ax.legend(title="Measure")
    return ax


def plot_risk_with_measures(
    df: pd.DataFrame, *, group: str | None = None, ax: Axes | None = None
) -> Axes:
    """Residual risk (solid) and averted risk (hatched) of each measure.

    The baseline (no measure) is shown as a gray bar, so that the top of each
    measure's stack reaches the height of the baseline bar.

    Parameters
    ----------
    df : pd.DataFrame
        Output of e.g. `StaticAppraiser.aai_metrics()`.
    group : str, optional
        Exposure group to plot. Needed if `df` contains several groups.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
    """
    df = _clean(df, group)
    measures = [m for m in pd.unique(df[MEASURE_COL_NAME]) if m != NO_MEASURE_VALUE]
    labels = [NO_MEASURE_VALUE, *measures]
    colors = _measure_colors(measures)
    dates = pd.DatetimeIndex(sorted(df[DATE_COL_NAME].unique()))
    x = np.arange(len(dates))
    width = 0.8 / len(labels)

    ax = _get_ax(ax)
    for k, lab in enumerate(labels):
        sub = df[df[MEASURE_COL_NAME] == lab].set_index(DATE_COL_NAME).reindex(dates)
        pos = x + (k - (len(labels) - 1) / 2) * width
        ax.bar(pos, sub[RISK_COL_NAME], width, color=colors[lab], label=lab)
        if lab != NO_MEASURE_VALUE:
            ax.bar(
                pos,
                sub[AVERTED_RISK_NAME],
                width,
                bottom=sub[RISK_COL_NAME],
                facecolor="none",
                edgecolor=colors[lab],
                hatch="//",
            )
    handles, names = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="none", edgecolor="k", hatch="//"))
    names.append("Averted risk")
    ax.legend(handles, names)
    ax.set_xticks(x, _date_labels(dates))
    ax.set_ylabel("Risk")
    return ax


def plot_exceedance_curves(
    df: pd.DataFrame,
    *,
    group: str | None = None,
    rp_col: str | None = None,
    shade: bool = True,
    logx: bool = True,
    axes=None,
) -> np.ndarray:
    """Impact against return period, one panel per date.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `StaticAppraiser.return_periods_metrics()`.
    group : str, optional
        Exposure group to plot. Needed if `df` contains several groups.
    rp_col : str, optional
        Column holding the return period. If None, it is parsed from the
        numbers in the metric column (e.g. "rp_100").
    shade : bool, optional
        Shade the area between the baseline and each measure (the averted
        impact). Defaults to True.
    logx : bool, optional
        Use a logarithmic return period axis. Defaults to True.
    axes : array-like of Axes, optional
        One axes per date.

    Returns
    -------
    np.ndarray
        The axes, one per date.
    """
    df = _clean(df, group)
    df = df.assign(_rp=_return_periods(df, rp_col))
    dates = pd.DatetimeIndex(sorted(df[DATE_COL_NAME].unique()))
    measures = [m for m in pd.unique(df[MEASURE_COL_NAME]) if m != NO_MEASURE_VALUE]
    colors = _measure_colors(measures)
    axes = _get_axes(axes, len(dates))

    for ax, date, label in zip(axes, dates, _date_labels(dates)):
        sub = df[df[DATE_COL_NAME] == date]
        base = (
            sub[sub[MEASURE_COL_NAME] == NO_MEASURE_VALUE]
            .set_index("_rp")[RISK_COL_NAME]
            .sort_index()
        )
        ax.plot(base.index, base, color=BASELINE_COLOR, label=NO_MEASURE_VALUE)
        for meas in measures:
            curve = (
                sub[sub[MEASURE_COL_NAME] == meas]
                .set_index("_rp")[RISK_COL_NAME]
                .sort_index()
                .reindex(base.index)
            )
            ax.plot(curve.index, curve, color=colors[meas], label=meas)
            if shade:
                ax.fill_between(base.index, curve, base, color=colors[meas], alpha=0.15)
        ax.set_title(label)
        ax.set_xlabel("Return period")
        if logx:
            ax.set_xscale("log")
            ax.set_xticks(base.index, [f"{rp:g}" for rp in base.index])
            ax.minorticks_off()
    axes[0].set_ylabel("Impact")
    axes[0].legend()
    return axes


def plot_averted_vs_cost(
    df: pd.DataFrame,
    *,
    date=None,
    group: str | None = None,
    payback_years: Iterable[float] = (),
    ax: Axes | None = None,
) -> Axes:
    """Scatter of averted annual risk against implementation cost.

    Parameters
    ----------
    df : pd.DataFrame
        Output of e.g. `StaticAppraiser.aai_metrics()`.
    date : optional
        Date to plot (anything parsable by `pandas.Timestamp`).
        Defaults to the last date.
    group : str, optional
        Exposure group to plot. Needed if `df` contains several groups.
    payback_years : iterable of float, optional
        Draw lines of constant payback time (cost / averted annual risk).
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
    """
    df = _clean(df, group, drop_baseline=True)
    date = df[DATE_COL_NAME].max() if date is None else pd.Timestamp(date)
    sub = df[df[DATE_COL_NAME] == date]
    if sub.empty:
        raise ValueError(f"No data at date {date}.")
    colors = _measure_colors(sub[MEASURE_COL_NAME])

    ax = _get_ax(ax)
    for _, row in sub.iterrows():
        meas = row[MEASURE_COL_NAME]
        ax.scatter(
            row[MEASURE_IMPL_COST_NAME], row[AVERTED_RISK_NAME], color=colors[meas]
        )
        ax.annotate(
            meas,
            (row[MEASURE_IMPL_COST_NAME], row[AVERTED_RISK_NAME]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    xmax = sub[MEASURE_IMPL_COST_NAME].max() * 1.1
    for n_years in payback_years:
        ax.plot([0, xmax], [0, xmax / n_years], "--", color="0.6", lw=1)
        ax.annotate(
            f"{n_years:g} yr payback",
            (xmax, xmax / n_years),
            ha="right",
            color="0.4",
            fontsize=8,
        )
    ax.set_xlabel("Implementation cost")
    ax.set_ylabel("Averted annual risk")
    ax.set_title(_date_labels([date])[0])
    return ax


def plot_averted_risk_per_group(
    df: pd.DataFrame,
    *,
    date=None,
    exclude_groups: Iterable[str] = (),
    ax: Axes | None = None,
) -> Axes:
    """Grouped bars of the averted risk, one group of bars per exposure group.

    Parameters
    ----------
    df : pd.DataFrame
        Output of `StaticAppraiser.aai_per_group_metrics()`.
    date : optional
        Date to plot (anything parsable by `pandas.Timestamp`).
        Defaults to the last date.
    exclude_groups : iterable of str, optional
        Groups to leave out, e.g. the "all groups" aggregate.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
    """
    df = _clean(df, drop_baseline=True)
    df = df[~df[GROUP_COL_NAME].isin([str(g) for g in exclude_groups])]
    date = df[DATE_COL_NAME].max() if date is None else pd.Timestamp(date)
    df = df[df[DATE_COL_NAME] == date]
    if df.empty:
        raise ValueError(f"No data at date {date}.")
    table = df.pivot(
        index=GROUP_COL_NAME, columns=MEASURE_COL_NAME, values=AVERTED_RISK_NAME
    )
    ax = _get_ax(ax)
    _grouped_bars(ax, table, _measure_colors(table.columns))
    ax.set_xticks(np.arange(len(table)), table.index)
    ax.set_ylabel("Averted risk")
    ax.set_title(_date_labels([date])[0])
    ax.legend(title="Measure")
    return ax


def plot_averted_risk_map(
    eai_df: pd.DataFrame,
    exposure_gdf,
    *,
    date=None,
    value_col: str = AVERTED_RISK_NAME,
    measures: Iterable[str] | None = None,
    axes=None,
    **plot_kwargs,
) -> np.ndarray:
    """Map of a per-coordinate quantity, one panel per measure.

    Parameters
    ----------
    eai_df : pd.DataFrame
        Output of `StaticAppraiser.eai_metrics()` (needs a coordinate id column).
    exposure_gdf : geopandas.GeoDataFrame
        Exposure GeoDataFrame, indexed by the coordinate id.
    date : optional
        Date to plot (anything parsable by `pandas.Timestamp`).
        Defaults to the last date.
    value_col : str, optional
        Column to map. Defaults to the averted risk. The baseline panel is
        only drawn if another column (e.g. the risk) is requested.
    measures : iterable of str, optional
        Measures to plot. Defaults to all.
    axes : array-like of Axes, optional
        One axes per panel.
    **plot_kwargs
        Passed to `GeoDataFrame.plot` (e.g. `markersize`, `cmap`).

    Returns
    -------
    np.ndarray
        The axes, one per panel.
    """
    df = _clean(eai_df, drop_baseline=value_col == AVERTED_RISK_NAME)
    date = df[DATE_COL_NAME].max() if date is None else pd.Timestamp(date)
    df = df[df[DATE_COL_NAME] == date]
    if measures is not None:
        df = df[df[MEASURE_COL_NAME].isin(list(measures))]
    if df.empty:
        raise ValueError("No data to plot for this date and measures.")

    geo = exposure_gdf.reset_index(names=COORD_ID_COL_NAME)[
        [COORD_ID_COL_NAME, exposure_gdf.geometry.name]
    ]
    panels = list(pd.unique(df[MEASURE_COL_NAME]))
    axes = _get_axes(axes, len(panels))
    plot_kwargs.setdefault("vmin", df[value_col].min())  # shared color scale
    plot_kwargs.setdefault("vmax", df[value_col].max())

    for ax, meas in zip(axes, panels):
        data = geo.merge(df[df[MEASURE_COL_NAME] == meas], on=COORD_ID_COL_NAME)
        data.plot(column=value_col, ax=ax, legend=True, **plot_kwargs)
        ax.set_title(meas)
    return axes


# --------------------------------------------------------------------------- #
# Plots for `cost_benefit_over_period`
# --------------------------------------------------------------------------- #
def plot_benefit_cost(cb: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Bars of the benefit-cost ratio of each measure.

    Parameters
    ----------
    cb : pd.DataFrame
        Output of `StaticAppraiser.cost_benefit_over_period()`.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
        A measure is worth it (on the financials alone) above the line at 1.
    """
    ratio = cb[RATIO_COL].replace([np.inf, -np.inf], np.nan)
    if ratio.isna().any():
        LOGGER.warning(
            "Non-finite ratios not plotted: %s", list(ratio.index[ratio.isna()])
        )
    colors = _measure_colors(cb.index.astype(str))
    ax = _get_ax(ax)
    ax.bar(ratio.index.astype(str), ratio, color=[colors[str(m)] for m in ratio.index])
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_ylabel("Benefit / cost")
    return ax


def plot_cost_benefit_area(cb: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Benefit-cost ratio against cumulated benefit; areas are the benefits.

    Measures are sorted by decreasing ratio. Measures with a non-finite ratio
    or a non-positive benefit cannot be drawn and are skipped.

    Parameters
    ----------
    cb : pd.DataFrame
        Output of `StaticAppraiser.cost_benefit_over_period()`.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
    """
    colors = _measure_colors(cb.index.astype(str))  # before sorting: stable colors
    keep = np.isfinite(cb[RATIO_COL]) & (cb["benefit"] > 0)
    if not keep.all():
        LOGGER.warning("Measures not plotted: %s", list(cb.index[~keep]))
    cb = cb[keep].sort_values(RATIO_COL, ascending=False)
    if cb.empty:
        raise ValueError("No measure with a positive benefit and a finite ratio.")
    left = cb["benefit"].cumsum() - cb["benefit"]

    ax = _get_ax(ax)
    ax.bar(
        left,
        cb[RATIO_COL],
        width=cb["benefit"],
        align="edge",
        color=[colors[str(m)] for m in cb.index],
        edgecolor="k",
    )
    for meas, x0, row in zip(cb.index, left, cb.itertuples()):
        ax.annotate(
            str(meas),
            (x0 + row.benefit / 2, getattr(row, RATIO_COL)),
            ha="center",
            va="bottom",
        )
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xlabel("Benefit")
    ax.set_ylabel("Benefit / cost")
    return ax


def plot_cumulative_net_benefit(cum: pd.DataFrame, *, ax: Axes | None = None) -> Axes:
    """Cumulated benefit minus cumulated net cost over time.

    The year a curve crosses zero is the payback time of the measure.

    Parameters
    ----------
    cum : pd.DataFrame
        Output of `StaticAppraiser.cumulative_net_benefit()`.
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
    """
    colors = _measure_colors(cum.columns.astype(str))
    ax = _get_ax(ax)
    for meas in cum.columns:
        ax.plot(cum.index, cum[meas], color=colors[str(meas)], label=str(meas))
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_ylabel("Cumulated net benefit")
    ax.legend(title="Measure")
    return ax


def plot_ratio_sensitivity(
    results: dict[str, pd.DataFrame], *, xlabel: str = "", ax: Axes | None = None
) -> Axes:
    """Benefit-cost ratio of each measure across scenarios.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Scenario label -> output of `StaticAppraiser.cost_benefit_over_period()`
        (e.g. one per discount rate or per period length).
    xlabel : str, optional
        Label of the x axis (what the scenarios vary).
    ax : Axes, optional
        Axes to draw on.

    Returns
    -------
    Axes
    """
    table = pd.DataFrame({k: v[RATIO_COL] for k, v in results.items()}).T
    table.columns = table.columns.astype(str)
    colors = _measure_colors(table.columns)
    ax = _get_ax(ax)
    for meas in table.columns:
        ax.plot(table.index, table[meas], marker="o", color=colors[meas], label=meas)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Benefit / cost")
    ax.legend(title="Measure")
    return ax
