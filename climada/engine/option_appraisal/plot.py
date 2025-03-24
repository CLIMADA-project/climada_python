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

Plotting functions for the option appraisal module

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from climada.util.value_representation import (
    convert_monetary_value,
    value_to_monetary_unit,
)


def plot_yearly(
    yearly_df,
    to_plot,
    with_measure,
    metric,
    y_label,
    title,
    measure_colors=None,
):
    def def_title(to_plot):
        if len(to_plot) > 1:
            return (
                "Yearly values for " + ", ".join(to_plot[:-1]) + f" and {to_plot[-1]}"
            )
        else:
            return f"Yearly values for {to_plot[0]}"

    # Calculate the Benefit/Cost Ratio
    if isinstance(to_plot, str):
        to_plot = [to_plot]

    measure = ["measure"] if with_measure else []
    y_label = "Risk" if y_label is None else y_label
    title = def_title(to_plot) if title is None else title

    yearly_df = yearly_df.loc[
        (yearly_df["metric"] == metric), ["year"] + measure + to_plot
    ]
    yearly_df = yearly_df.sort_values("year")
    yearly_df = yearly_df.melt(
        id_vars=["measure", "year"], value_name=y_label, var_name="Variable"
    )
    yearly_df.columns = yearly_df.columns.str.title()
    yearly_df["Variable"] = yearly_df["Variable"].str.title()
    yearly_df[y_label], risk_unit = value_to_monetary_unit(yearly_df[y_label])
    colors = (
        sns.color_palette("colorblind", len(yearly_df["Measure"].unique()))
        if measure_colors is None
        else measure_colors
    )

    _, ax = plt.subplots(figsize=(12, 8))
    g = sns.lineplot(
        yearly_df,
        ax=ax,
        x="Year",
        y=y_label,
        hue="Measure",
        style="Variable",
        palette=colors,
    )
    y_label = y_label + f" ({risk_unit})"
    ax.set_ylabel(y_label, fontsize=18, fontweight="bold", color="black")
    plt.title(title, fontsize=16)
    plt.tight_layout()

    # Show plot
    plt.show()
    return g


def plot_CB_summary(
    cb_df,
    measure_colors=None,
    metric="aai",
    y_label="Risk",
    title="Benefit and Benefit/Cost Ratio by Measure",
):
    # Calculate the Benefit/Cost Ratio
    cb_df = cb_df.loc[(cb_df["metric"] == metric) & (cb_df["measure"] != "no_measure")]
    cb_df = cb_df.rename(columns={"B/C ratio": "Benefit/Cost Ratio"})
    cb_df = cb_df.sort_values("averted risk")
    cb_df.columns = cb_df.columns.str.title()
    cb_df["Averted Risk"], risk_unit = value_to_monetary_unit(cb_df["Averted Risk"])
    cb_df["Base Risk"] = convert_monetary_value(cb_df["Base Risk"], risk_unit)
    cb_df["Residual Risk"] = convert_monetary_value(cb_df["Residual Risk"], risk_unit)

    colors = (
        sns.color_palette("colorblind", len(cb_df["Measure"].unique()))
        if measure_colors is None
        else measure_colors
    )

    y_label = y_label + f" ({risk_unit})"
    total_climate_risk = cb_df["Base Risk"].max()
    highest_benefit = cb_df["Averted Risk"].max()
    residual_risk = cb_df["Residual Risk"].min()

    _, ax1 = plt.subplots(figsize=(12, 8))

    bars = sns.barplot(
        cb_df,
        x="Measure",
        y="Averted Risk",
        hue="Measure",
        ax=ax1,
        palette=colors,
        dodge=False,
        legend=False,
    )
    ax1.set_ylabel(y_label, fontsize=18, fontweight="bold", color="black")
    ax1.set_xlabel("Measure", fontsize=14)

    # Placing the benefit values on top of the bars
    for bar in bars.patches:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{bar.get_height():.2f}",
            ha="center",
            color="black",
            fontsize=12,
            fontweight="bold",
        )

    # Creating a second y-axis for Benefit/Cost Ratio
    ax2 = ax1.twinx()
    line = sns.lineplot(
        x="Measure",
        y="Benefit/Cost Ratio",
        data=cb_df,
        ax=ax2,
        marker="o",
        color="red",
        linewidth=1,
    )
    ax2.set_ylabel("Benefit/Cost Ratio", color="red", fontsize=10)

    # Setting higher y-limit for the benefit/cost ratio axis
    ax2.set_ylim(0, max(cb_df["Benefit/Cost Ratio"]) * 1.5)

    # Change the color of the right y-axis line, ticks, and labels
    ax2.spines["right"].set_color("red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Change the color of the left y-axis ticks and labels, and make them bigger and bold
    ax1.tick_params(axis="y", labelcolor="black", labelsize=14, width=2)
    ax1.spines["left"].set_linewidth(2)

    # Adding a red horizontal line at Benefit/Cost Ratio = 1 without a label
    ax2.axhline(1, color="red", linestyle="--", linewidth=2)

    # Adding a black horizontal line at Total Climate Risk value
    ax1.axhline(total_climate_risk, color="black", linestyle="--", linewidth=2)
    ax1.text(
        -0.52,
        total_climate_risk * 1.02,
        f"Total climate risk: {total_climate_risk:.2f}",
        color="black",
        va="center",
        ha="left",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Adding a grey arrow to indicate the residual risk, pointing upwards and moved to the right
    arrow_x = len(cb_df) - 0.6  # Adjust x position for arrow
    arrow_y_end = total_climate_risk
    arrow_y_start = highest_benefit

    ax1.annotate(
        "",
        xy=(arrow_x, arrow_y_start),
        xycoords="data",
        xytext=(arrow_x, arrow_y_end),
        textcoords="data",
        arrowprops=dict(arrowstyle="<-", color="grey", lw=2),
    )

    # Positioning the residual risk text box so it touches the arrow on the left side
    bbox_props = dict(
        boxstyle="round,pad=0.3", edgecolor="grey", facecolor="white", alpha=0.5
    )
    ax1.text(
        arrow_x - 0.1,
        (arrow_y_end + arrow_y_start) / 2,
        f"Residual risk: {residual_risk:.2f}",
        color="grey",
        va="center",
        ha="right",
        fontsize=12,
        fontweight="bold",
        bbox=bbox_props,
    )

    # Adding title
    plt.title(title, fontsize=16)

    # Show plot
    plt.tight_layout()
    plt.show()


def _calc_waterfall_plot_df(
    annual_risk_metrics_df,
    all_annual_risk_metrics_df,
    measure=None,
    metric="aai",
    ref_year=None,
    fut_year=None,
    subtract=False,
    group=np.nan,
):
    # Check if the reference and future years are provided
    if ref_year is None:
        ref_year = annual_risk_metrics_df["year"].min()
    if fut_year is None:
        fut_year = annual_risk_metrics_df["year"].max()

    # Initialize the waterfall DataFrame
    waterfall_df = pd.DataFrame(
        columns=[
            "year",
            "metric",
            "ref_risk",
            "exp_change",
            "impfset_change",
            "haz_change",
            "risk",
            "group",
        ]
    )

    # Filter the annual_risk_metrics_df
    filt_annual_risk_metrics_df = annual_risk_metrics_df[
        (annual_risk_metrics_df["year"] >= ref_year)
        & (annual_risk_metrics_df["year"] <= fut_year)
        & (annual_risk_metrics_df["measure"] == "no_measure")
        & (annual_risk_metrics_df["metric"] == metric)
        & ((pd.isna(group)) | (annual_risk_metrics_df["group"] == group))
    ]

    # Filter the all_annual_risk_metrics_df
    filt_all_annual_risk_metrics_df = all_annual_risk_metrics_df[
        (all_annual_risk_metrics_df["year"] >= ref_year)
        & (all_annual_risk_metrics_df["year"] <= fut_year)
        & (all_annual_risk_metrics_df["measure"] == "no_measure")
        & (all_annual_risk_metrics_df["metric"] == metric)
        & ((pd.isna(group)) | (all_annual_risk_metrics_df["group"] == group))
    ]

    # Get the reference risk
    ref_risk = filt_annual_risk_metrics_df[
        filt_annual_risk_metrics_df["year"] == ref_year
    ]["result"].values[0]

    # Calculate the waterfall columns
    for year in range(ref_year, fut_year + 1):
        curr_value = filt_annual_risk_metrics_df[
            filt_annual_risk_metrics_df["year"] == year
        ]["result"].values[0]

        # Filter levels from all_annual_risk_metrics_df
        # Change in exposure
        exp_change_level = filt_all_annual_risk_metrics_df[
            (filt_all_annual_risk_metrics_df["exp_change"] == True)
            & (filt_all_annual_risk_metrics_df["impfset_change"] == False)
            & (filt_all_annual_risk_metrics_df["haz_change"] == False)
            & (filt_all_annual_risk_metrics_df["year"] == year)
        ]["result"].values[0]

        # Change in exposure and impfset
        impfset_change_level = filt_all_annual_risk_metrics_df[
            (filt_all_annual_risk_metrics_df["exp_change"] == True)
            & (filt_all_annual_risk_metrics_df["impfset_change"] == True)
            & (filt_all_annual_risk_metrics_df["haz_change"] == False)
            & (filt_all_annual_risk_metrics_df["year"] == year)
        ]["result"].values[0]

        # Calculate the changes
        if subtract:
            exp_change = exp_change_level - ref_risk
            impfset_change = impfset_change_level - exp_change_level
            climate_change = curr_value - impfset_change_level
        else:
            exp_change = exp_change_level
            impfset_change = impfset_change_level
            climate_change = curr_value

        # Append the values to the waterfall_df
        temp_df = pd.DataFrame(
            [
                [
                    year,
                    group,
                    metric,
                    ref_risk,
                    exp_change,
                    impfset_change,
                    climate_change,
                    curr_value,
                ]
            ],
            columns=[
                "year",
                "group",
                "metric",
                "ref_risk",
                "exp_change",
                "impfset_change",
                "haz_change",
                "risk",
            ],
        )

        # Check if the waterfall_df is empty
        if waterfall_df.empty:
            waterfall_df = temp_df
        else:
            waterfall_df = pd.concat([waterfall_df, temp_df], ignore_index=True)

    # Add the risk for the measure
    if measure:
        # Get the measure risk
        filt_annual_risk_metrics_df = annual_risk_metrics_df[
            (annual_risk_metrics_df["year"] >= ref_year)
            & (annual_risk_metrics_df["year"] <= fut_year)
            & (annual_risk_metrics_df["measure"] == measure)
            & (annual_risk_metrics_df["metric"] == metric)
            & ((pd.isna(group)) | (annual_risk_metrics_df["group"] == group))
        ]

        # Join the risk for the measure and add only the risk column and the measure name
        waterfall_df = waterfall_df.merge(
            filt_annual_risk_metrics_df[["year", "measure", "result"]],
            on="year",
            how="left",
        )
        waterfall_df = waterfall_df.rename(columns={"result": "measure_risk"})
        # Make an additional column called averted risk
        waterfall_df["averted_risk"] = (
            waterfall_df["risk"] - waterfall_df["measure_risk"]
        )

    return waterfall_df


# Plot the yearly waterfall plot
def _plot_yearly_waterfall(waterfall_df, metric="aai", value_unit="USD"):
    fig, ax = plt.subplots(figsize=(15, 10))

    # Get the reference and future years
    ref_year = waterfall_df["year"].min()

    colors = ["blue", "orange", "green", "red"]
    labels = [f"Risk {ref_year}", "Exp Change", "Impfset Change", "Haz Change"]

    for year in waterfall_df["year"].unique():
        year_df = waterfall_df[waterfall_df["year"] == year]
        ref_risk = year_df["ref_risk"].values[0]
        exp_change = year_df["exp_change"].values[0]
        impfset_change = year_df["impfset_change"].values[0]
        haz_change = year_df["haz_change"].values[0]

        values = [ref_risk, exp_change, impfset_change, haz_change]

        # Calculate cumulative values for positive values only
        cum_values = [0]
        for i in range(1, len(values)):
            if values[i - 1] >= 0:
                cum_values.append(cum_values[i - 1] + values[i - 1])
            else:
                cum_values.append(cum_values[i - 1])

        # Plot cumulative positive values
        for i in range(len(values)):
            if values[i] >= 0:
                ax.bar(
                    year,
                    values[i],
                    bottom=cum_values[i],
                    color=colors[i],
                    label=labels[i] if year == waterfall_df["year"].min() else "",
                )
            else:
                # Placeholder for negative values to adjust cum_values
                cum_values[i + 1] = cum_values[i]

        # Plot negative values separately
        neg_cum_values = [0]
        for i in range(1, len(values)):
            if values[i - 1] < 0:
                neg_cum_values.append(neg_cum_values[i - 1] + values[i - 1])
            else:
                neg_cum_values.append(neg_cum_values[i - 1])

        for i in range(len(values)):
            if values[i] < 0:
                ax.bar(
                    year,
                    values[i],
                    bottom=neg_cum_values[i],
                    color=colors[i],
                    label=labels[i] if year == waterfall_df["year"].min() else "",
                )
                # Adjust cumulative values
                if i < len(values) - 1:
                    neg_cum_values[i + 1] += values[i]

    ax.axhline(
        0, color="black", linewidth=0.8, linestyle="--"
    )  # Add a horizontal grid line at zero

    # Construct y-axis label and title based on parameters
    value_label = f"Risk {metric} ({value_unit})"
    title_label = f"Yearly Waterfall Plot ({metric})"

    ax.set_title(title_label)
    ax.set_xlabel("Year")
    ax.set_ylabel(value_label)

    # Reverse the legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Plot the waterfall plot for two years
def _plot_two_years_waterfall(waterfall_df, metric="aai", value_unit="USD"):
    # Get the reference and future years
    ref_year = waterfall_df["year"].min()
    fut_year = waterfall_df["year"].max()

    # Filter the DataFrame for the specified years
    ref_data = waterfall_df[waterfall_df["year"] == ref_year].iloc[0]
    fut_data = waterfall_df[waterfall_df["year"] == fut_year].iloc[0]

    # Extract values for the waterfall plot
    ref_risk = ref_data["ref_risk"]
    exp_change = fut_data["exp_change"]
    impfset_change = fut_data["impfset_change"]
    haz_change = fut_data["haz_change"]
    fut_risk = fut_data["risk"]

    # Calculate the intermediate values
    exp_change_diff = exp_change - ref_risk
    impfset_change_diff = impfset_change - exp_change
    haz_change_diff = haz_change - impfset_change

    values = [
        ref_risk,
        exp_change_diff,
        impfset_change_diff,
        haz_change_diff,
        fut_risk - (ref_risk + exp_change_diff + impfset_change_diff + haz_change_diff),
    ]
    labels = [
        f"Risk {ref_year}",
        "Exposure change",
        "Vulnerability change",
        "Climate change",
        f"Risk {fut_year}",
    ]
    colors = ["blue", "orange", "green", "red", "purple"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the initial reference risk
    ax.bar(labels[0], values[0], color=colors[0], edgecolor="black")

    # Plot cumulative changes
    cum_value = values[0]
    for i in range(1, len(values) - 1):
        cum_value += values[i]
        ax.bar(
            labels[i],
            values[i],
            bottom=cum_value - values[i],
            color=colors[i],
            edgecolor="black",
        )

    # Plot the final future risk
    ax.bar(labels[-1], fut_risk, color=colors[-1], edgecolor="black")

    # Adding labels to the bars
    cum_value = values[0]
    for i in range(len(values)):
        if i == 0:
            ax.text(
                labels[i],
                values[i],
                f"{values[i]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )
        elif i == len(values) - 1:
            ax.text(
                labels[i],
                fut_risk,
                f"{fut_risk:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )
        else:
            cum_value += values[i]
            ax.text(
                labels[i],
                cum_value,
                f"{values[i]:.0e}",
                ha="center",
                va="bottom",
                color="black",
            )

    if "measure_risk" in waterfall_df.columns:
        measure_risk = fut_data["measure_risk"]

        # Add an arrow indicating averted risk
        ax.annotate(
            "",
            xy=(labels[-1], measure_risk),
            xytext=(labels[-1], fut_risk),
            arrowprops=dict(
                facecolor="green", shrink=0.05, width=5, headwidth=20, headlength=10
            ),
        )

        # Place the text in the center of the arrow
        arrow_midpoint = measure_risk * 0.98
        ax.text(
            labels[-1],
            arrow_midpoint,
            "Averted Risk",
            ha="center",
            va="center",
            color="white",
            fontsize=14,
        )

    # Construct y-axis label and title based on parameters
    value_label = f"{metric} ({value_unit})"
    title_label = f"Risk at {ref_year} and {fut_year} ({metric}) for measure"

    ax.set_title(title_label)
    ax.set_ylabel(value_label)

    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout for better fit

    plt.show()


# Plot the risk metrics
def plot_risk_metrics(
    annual_risk_metrics_df,
    metric="aai",
    group=np.nan,
    averted=False,
    discounted=False,
    plot_type="line",
    measures=None,
    value_unit="USD",
):
    # Filter the DataFrame
    if pd.isna(group):
        filt_annual_risk_metrics_df = annual_risk_metrics_df[
            (annual_risk_metrics_df["group"].isna())
            & (annual_risk_metrics_df["metric"] == metric)
        ]
    else:
        filt_annual_risk_metrics_df = annual_risk_metrics_df[
            (annual_risk_metrics_df["group"] == group)
            & (annual_risk_metrics_df["metric"] == metric)
        ]

    # Filter the measures
    if measures:
        filt_annual_risk_metrics_df = filt_annual_risk_metrics_df[
            filt_annual_risk_metrics_df["measure"].isin(measures)
        ]

    # Ensure 'year' is treated as a categorical variable for bar plots
    filt_annual_risk_metrics_df["year"] = filt_annual_risk_metrics_df["year"].astype(
        str
    )

    # Create the plot
    plt.figure(figsize=(15, 8))

    if plot_type == "bar":
        sns.barplot(
            data=filt_annual_risk_metrics_df,
            x="year",
            y="result",
            hue="measure",
            errorbar=None,
            alpha=0.7,
        )
    else:
        sns.lineplot(
            data=filt_annual_risk_metrics_df, x="year", y="result", hue="measure"
        )

    # Make y-axis from 0 to max aai with margin
    plt.ylim(0, filt_annual_risk_metrics_df["result"].max() * 1.1)

    # Tilt the x-axis labels for better readability and display every nth year
    n = max(
        1, len(filt_annual_risk_metrics_df["year"].unique()) // 20
    )  # Display every nth year, with a maximum of 20 labels
    plt.xticks(
        ticks=np.arange(0, len(filt_annual_risk_metrics_df["year"].unique()), n),
        rotation=45,
    )

    # More compressed label construction
    label_prefix = (
        f"{'Discounted ' if discounted else ''}{'Averted ' if averted else ''}"
    )
    plt.ylabel(f"{label_prefix}{metric} ({value_unit})")
    plt.title(
        f"{label_prefix}{metric} per Year"
        + (f" for Group: {group}" if not pd.isna(group) else "")
    )

    # Move the legend to the right side, outside the plot
    plt.legend(
        title="Measure", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0
    )

    plt.tight_layout()
    plt.show()


def plot_yearly_averted_cost(
    ann_CB_df,
    measure,
    metric="aai",
    group=None,
    averted=False,
    discounted=False,
    value_unit="USD",
):
    # Filter the dataframe
    df_filtered = ann_CB_df[
        (ann_CB_df["measure"] == measure) & (ann_CB_df["metric"] == metric)
    ]

    if group is not None:
        df_filtered = df_filtered[df_filtered["group"] == group]

    # Replace None with 0 in 'cost (net)' for plotting
    df_filtered["cost (net)"] = df_filtered["cost (net)"].fillna(0)

    # Separate positive and negative values for stacking correctly
    positive_averted_risks = df_filtered["averted risk"].clip(lower=0)
    negative_averted_risks = df_filtered["averted risk"].clip(upper=0)
    positive_net_costs = df_filtered["cost (net)"].clip(lower=0)
    negative_net_costs = df_filtered["cost (net)"].clip(upper=0)

    # Plotting
    fig, ax = plt.subplots(figsize=(15, 8))

    # Construct y-axis label and title based on parameters
    label_prefix = ("Discounted " if discounted else "") + (
        "Averted " if averted else ""
    )
    value_label = f"{label_prefix} {metric} ({value_unit})"
    title_label = (
        f"Yearly {label_prefix} Risk ({metric}) vs Net Cost for measure: {measure}"
    )

    # Plot the averted risk and net cost
    years = df_filtered["year"].unique()

    # Plot positive values
    ax.bar(years, positive_averted_risks, color="green", label=label_prefix + "Risk")
    ax.bar(
        years,
        positive_net_costs,
        bottom=positive_averted_risks,
        color="red",
        label=label_prefix + "Net Cost",
    )

    # Plot negative values
    ax.bar(years, negative_averted_risks, color="green")
    ax.bar(years, negative_net_costs, bottom=negative_averted_risks, color="red")

    # Add labels and title
    ax.set_ylabel(value_label)
    ax.set_title(title_label)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2
    )  # Move legend to top
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")  # Improve x-axis readability

    # Add grid line at y=0
    ax.axhline(0, color="black", linewidth=0.5)

    # Set the y-axis limit as the maximum of the absolute values of the averted risk and net cost
    y_lim = (
        max(
            abs(df_filtered["averted risk"]).max(), abs(df_filtered["cost (net)"]).max()
        )
        * 1.1
    )
    ax.set_ylim(-y_lim, y_lim)

    plt.tight_layout()
    plt.show()
