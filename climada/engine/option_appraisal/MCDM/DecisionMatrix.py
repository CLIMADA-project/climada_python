import copy
from typing import Dict, List, Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import weighting_methods as mcda_weights

# Importing additional functions from pyrepo_mcda
from pyrepo_mcda.additions import rank_preferences
from pyrepo_mcda.compromise_rankings import (
    copeland,
    dominance_directed_graph,
    rank_position_method,
)

# Importing various methods from pyrepo_mcda
from pyrepo_mcda.mcda_methods import (
    AHP,
    ARAS,
    COCOSO,
    CODAS,
    COPRAS,
    CRADIS,
    EDAS,
    MABAC,
    MARCOS,
    MULTIMOORA,
    MULTIMOORA_RS,
    PROMETHEE_II,
    PROSA_C,
    SAW,
    SPOTIS,
    TOPSIS,
    VIKOR,
    VIKOR_SMAA,
    VMCM,
    WASPAS,
)

# Utility functions
from .MCDMoutput import RanksOutput
from .utils import filter_dataframe

"""
ToDO:

    - Make the plot methods as for the CalcRank results
        - E.g., Distribution of criteria values
    - Make the calc conditional criteria value-at-risk
    - Create a color attribute for each alternative. A dictionary of colors.

"""
# Define the MCDM ranking methods
MCDM_DEFAULT = {"Topsis": TOPSIS(), "Saw": SAW(), "Vikor": VIKOR()}  #'AHP': AHP(),

# Define the compromised ranking function of the rank matrices
COMP_DEFAULT = {
    "copeland": copeland,
}


class DecisionMatrix:
    def __init__(
        self,
        metrics_df: pd.DataFrame,
        objectives: Dict[str, int],
        alt_cols: List[str],
        crit_cols: List[str],
        weights: Optional[Dict[str, float]] = None,
        group_cols: Optional[List[str]] = [],
        group_weights: Optional[Dict[str, float]] = None,
        unc_cols: Optional[List[str]] = [],
        unc_var_prob_dist: Optional[Dict[str, object]] = None,
        crit_cats: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the DecisionMatrix object.

        Parameters:
        - metrics_df : pd.DataFrame
            DataFrame containing metrics data.
        - objectives : Dict[str, int]
            Dictionary mapping objectives to their values.
        - alt_cols : List[str]
            List of alternative columns.
        - crit_cols : List[str]
            List of criteria columns.
        - weights : Dict[str, float], optional
            Dictionary of criteria weights values. Defaults to an empty dictionary.
        - group_cols : List[str], optional
            List of group columns. Defaults to an empty list.
        - group_weights : Dict[str, float], optional
            Dictionary of weights for group columns. Defaults to an empty dictionary.
        - unc_cols : List[str], optional
            List of uncertainty columns. Defaults to an empty list.
        - crit_cats : Dict[str, List[str]], optional
            Dictionary of categorized criteria. Defaults to an empty dictionary.
        """

        # Assign input parameters to class attributes as copies
        self.metrics_df = metrics_df.copy()
        self.objectives = copy.deepcopy(objectives)
        self.alt_cols = alt_cols.copy()
        self.crit_cols = crit_cols.copy()
        self.weights = weights.copy() if weights is not None else {}
        self.group_cols = group_cols.copy() if group_cols is not None else []
        self.group_weights = group_weights.copy() if group_weights is not None else {}
        self.unc_cols = unc_cols.copy() if unc_cols is not None else []
        # self.unc_var_prob_dist = unc_var_prob_dist.copy() if unc_var_prob_dist is not None else {}
        self.crit_cats = crit_cats.copy() if crit_cats is not None else {}

        # Initialize other attributes as None
        self.dm_df = None
        self.alternatives_df = None
        self.crit_df = None
        self.cat_crit_df = None
        self.groups_df = None
        self.unc_smpls_df = None

        # Sort the dm_df based on alt_cols, group_cols, and unc_cols
        self.metrics_df = self.metrics_df.sort_values(
            by=alt_cols + group_cols + unc_cols
        )

        # Create internal group weights
        if group_cols:
            self.groups_df = (
                self.metrics_df[self.group_cols]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            self.groups_df.insert(
                0,
                "Group ID",
                ["G" + str(idx) for idx in range(1, len(self.groups_df) + 1)],
            )

            # Create internal group weights
            for group_col in group_cols:
                temp_df = pd.DataFrame(metrics_df[group_col].drop_duplicates())
                temp_df["Weight"] = (
                    np.nan
                )  # Initialize weight to NaN for each member of the group

                # Populate the group weights if given
                for idx, member in temp_df.iterrows():
                    member_name = member[group_col]
                    if (
                        member_name != "ALL"
                        and group_weights
                        and group_col in group_weights
                        and isinstance(group_weights[group_col], dict)
                        and member_name in group_weights[group_col]
                    ):
                        temp_df.at[idx, "Weight"] = group_weights[group_col][
                            member_name
                        ]

                # Exclude 'ALL' members from count
                temp_df = temp_df[temp_df[group_col] != "ALL"]

                # Calculate the sum of defined weights and count of remaining NaN values
                sum_defined_weights = temp_df["Weight"].sum()
                remaining_members = temp_df[temp_df["Weight"].isna()]
                remaining_count = len(remaining_members)

                # Distribute the remaining weight equally among the members whose weights are not defined
                if remaining_count > 0:
                    remainder = 1 - sum_defined_weights
                    equal_weight = remainder / remaining_count
                    temp_df.loc[temp_df["Weight"].isna(), "Weight"] = equal_weight
                    print(
                        f"Remaining weights distributed equally among members of group column '{group_col}'."
                    )

                self.group_weights[group_col] = temp_df

        # Get unique set of samples if unc_cols is provided and reset index
        if self.unc_cols:
            self.unc_smpls_df = (
                self.metrics_df[self.unc_cols].drop_duplicates().reset_index(drop=True)
            )
            self.unc_smpls_df.insert(
                0,
                "Sample ID",
                ["S" + str(idx) for idx in range(1, len(self.unc_smpls_df) + 1)],
            )
            # Calculate default likelihood distribution for uncertainty columns
            # for unc_col in self.unc_cols:
            #     if unc_col not in self.unc_var_prob_dist.keys():
            #         outcomes = list(self.unc_smpls_df[unc_col].drop_duplicates().values)
            #         probabilities = np.ones(len(outcomes)) / len(outcomes)
            #         self.unc_var_prob_dist[unc_col] = dict(zip(outcomes, probabilities))

        # Calculate criteria weights if not provided
        # Assumes equal weights if not provided
        if not self.weights:
            # Allow for max to decimals for weights
            def custom_round(value):
                decimal_count = (
                    len(str(value).split(".")[1]) if "." in str(value) else 0
                )
                decimals = 2 if decimal_count >= 2 else 1
                return round(value, decimals)

            self.weights = {
                crit: custom_round(1 / len(self.crit_cols)) for crit in self.crit_cols
            }
        # Check for duplicate rows
        if self.metrics_df.duplicated(
            subset=self.alt_cols + self.group_cols + self.unc_cols, keep=False
        ).any():
            raise ValueError(
                "Duplicated rows of alt_cols, group_cols, and sample. Some alternative IDs are counted more than once for some group ID and sample ID pairs."
            )

        # Create crit_df (DataFrame containing criteria data)
        data = []
        if self.crit_cols:
            for idx, criteria in enumerate(self.crit_cols):
                data.append(
                    {
                        "Criteria ID": "C" + str(idx + 1),
                        "Criteria": criteria,
                        "Weight": self.weights[criteria],
                        "Objective": self.objectives[criteria],
                    }
                )
        self.crit_df = pd.DataFrame(data)

        # Create cat_crit_df (DataFrame containing categorized criteria data)
        if not self.crit_cats:
            self.crit_cats = {crit: [crit] for crit in self.crit_cols}

        data = []
        for idx, cat_set in enumerate(self.crit_cats.items()):
            for criteria in cat_set[1]:
                data.append(
                    {
                        "Cat ID": "CAT" + str(idx + 1),
                        "Category": cat_set[0],
                        "Criteria": criteria,
                    }
                )
        self.cat_crit_df = pd.DataFrame(data)
        self.cat_crit_df = self.cat_crit_df.merge(self.crit_df, on="Criteria")

        # Create alternatives_df (DataFrame containing alternatives data)
        if self.alt_cols:
            self.alternatives_df = self.metrics_df[self.alt_cols].drop_duplicates()
            self.alternatives_df.insert(
                0,
                "Alternative ID",
                ["A" + str(idx) for idx in range(1, len(self.alternatives_df) + 1)],
            )
        else:
            raise ValueError("No alternative column given.")

        # Merge alternatives_df with groups and sample to create dm_df (Initialized DataFrame for decision-making)
        self.dm_df = self.alternatives_df.copy()
        self.dm_df["_merge"] = 1

        if isinstance(self.groups_df, pd.DataFrame):
            self.groups_df["_merge"] = 1
            self.dm_df = self.dm_df.merge(self.groups_df, on="_merge")
            self.groups_df = self.groups_df.drop("_merge", axis=1)

        if isinstance(self.unc_smpls_df, pd.DataFrame):
            self.unc_smpls_df["_merge"] = 1
            self.dm_df = self.dm_df.merge(self.unc_smpls_df, on="_merge")
            self.unc_smpls_df = self.unc_smpls_df.drop("_merge", axis=1)

        self.dm_df = self.dm_df.drop("_merge", axis=1)
        self.dm_df = pd.merge(
            self.dm_df,
            self.metrics_df[
                self.alt_cols + self.group_cols + self.unc_cols + self.crit_cols
            ],
            on=self.alt_cols + self.group_cols + self.unc_cols,
            how="left",
        )

    def pivot_and_reweight_criteria(self, piv_col):
        """
        Pivot and reweight criteria based on a specified pivot column and group weights.

        Parameters:
        - piv_col: str
            The column name to pivot the criteria data.

        Returns:
        - new_dm: DecisionMatrix
            A new instance of DecisionMatrix with pivoted criteria.
        """

        # Define pivot and index columns for pivot
        index_col = [
            col
            for col in self.alt_cols + self.unc_cols + self.group_cols + self.crit_cols
            if col not in self.crit_cols + [piv_col]
        ]

        # Filter out rows where the specified column is ALL or nan
        filt_dm_df = self.dm_df[
            self.alt_cols + self.unc_cols + self.group_cols + self.crit_cols
        ]
        filt_dm_df = filt_dm_df[
            ~filt_dm_df[piv_col].isin(["ALL"]) & filt_dm_df[piv_col].notna()
        ]

        crit_piv_df = filt_dm_df.pivot(
            index=index_col, columns=piv_col, values=self.crit_cols
        )

        crit_piv_df = crit_piv_df.reset_index()
        crit_piv_df.columns = [
            f'{"_".join(col)}' if col[1] else f"{col[0]}" for col in crit_piv_df.columns
        ]

        # Step 3: Remove duplicates and create a copy of weights
        new_weights = copy.deepcopy(self.weights)
        new_crit_cats = {key: [] for key in self.crit_cats.keys()}

        group_values = list(filt_dm_df[piv_col].dropna().drop_duplicates())
        new_objectives = copy.deepcopy(
            self.objectives
        )  # Initialize new objectives for pivoted criteria

        for crit_col in self.crit_cols:
            new_crit_cols_temp = [
                crit_col + "_" + group_value for group_value in group_values
            ]
            temp_df = crit_piv_df[new_crit_cols_temp]

            cat_crit = self.cat_crit_df[self.cat_crit_df["Criteria"].isin([crit_col])][
                "Category"
            ].values[0]

            # Step 4: Check if all columns have the same values
            if temp_df.apply(lambda col: col.equals(temp_df.iloc[:, 0])).all():
                print(
                    f"{crit_col}: All columns have the same values. Retain the original name."
                )
                crit_piv_df = crit_piv_df.rename(
                    columns={new_crit_cols_temp[0]: crit_col}
                )
                if len(new_crit_cols_temp) > 1:
                    crit_piv_df = crit_piv_df.drop(columns=new_crit_cols_temp[1:])
                # Update cat crits
                new_crit_cats[cat_crit].append(crit_col)
            else:
                print(
                    f"{crit_col}: Columns have different values. Reweight and introduce new criteria."
                )
                for group_value in group_values:
                    idx = self.group_weights[piv_col][piv_col].isin([group_value])
                    group_weight = self.group_weights[piv_col]["Weight"][idx].values[0]
                    new_weights.update(
                        {
                            crit_col
                            + "_"
                            + group_value: new_weights[crit_col] * group_weight
                        }
                    )
                    new_objectives[crit_col + "_" + group_value] = self.objectives[
                        crit_col
                    ]

                    # Update cat crits
                    new_crit_cats[cat_crit].append(crit_col + "_" + group_value)

                del new_weights[crit_col]
                del new_objectives[crit_col]

        # Step 4 (continued): Reorder the dictionary as per crit_piv_df columns
        new_weights = {
            key: new_weights[key]
            for key in crit_piv_df.columns
            if key in new_weights.keys()
        }

        # Step 4 (continued): Reorder the dictionary as per crit_piv_df columns
        new_objectives = {
            key: new_objectives[key]
            for key in crit_piv_df.columns
            if key in new_objectives.keys()
        }

        # Step 4 (continued): Reorder the dictionary as per crit_piv_df columns
        new_group_cols = [col for col in self.group_cols if col != piv_col]
        new_crit_cols = list(new_objectives.keys())

        # Step 5: Remove the group key from the group weights
        new_group_weights = {
            key: value for key, value in self.group_weights.items() if key != piv_col
        }

        # Create a new DecisionMatrix instance with modified attributes
        new_self = DecisionMatrix(
            metrics_df=crit_piv_df,
            objectives=new_objectives,
            alt_cols=self.alt_cols,
            crit_cols=new_crit_cols,
            weights=new_weights,
            group_cols=new_group_cols,
            unc_cols=self.unc_cols,
            # unc_var_prob_dist=self.unc_var_prob_dist,
            crit_cats=new_crit_cats,
            group_weights=new_group_weights,
            # Include other necessary attributes for initialization of the new instance
        )

        return new_self

    def mean_based_criteria(self, condition={}, derived_columns=None):
        """
        Apply criteria based on mean values of uncertain variables to the given data
        and generate a new instance of DecisionMatrix.

        Parameters:
        - condition (dict): Dictionary of conditions to filter the data.
        - derived_columns (dict): Dictionary of derived columns to be calculated.

        Returns:
        - new_dm (DecisionMatrix): New instance of DecisionMatrix with updated criteria.
        """

        # Create a copy of the decision matrix DataFrame
        dm_df = self.dm_df.copy()

        # Define base column
        base_cols = list(self.alternatives_df.columns) + ["Group ID"]
        if isinstance(self.groups_df, pd.DataFrame):
            base_cols += list(self.groups_df.columns)
        else:
            dm_df["Group ID"] = "G1"

        # Remove duplicates in base_cols
        base_cols = list(dict.fromkeys(base_cols))

        # Create a dataframe to store the results
        new_dm_df = pd.DataFrame(columns=base_cols + self.crit_cols)

        # Apply constraints per group and state combo
        for _, alt_group_df in (
            dm_df[["Alternative ID", "Group ID"]].drop_duplicates().iterrows()
        ):
            # Filter the group and state
            sg_df = dm_df[
                dm_df[["Alternative ID", "Group ID"]]
                .isin(alt_group_df[["Alternative ID", "Group ID"]].values)
                .all(axis=1)
            ]

            # Filter the dataframe based on the condition
            filt_sg_df, _ = filter_dataframe(
                sg_df, filter_conditions=condition, derived_columns=derived_columns
            )

            # Print the alternative and group that are all filtered out
            if filt_sg_df.empty:
                print(
                    f"The alternative {alt_group_df['Alternative ID']} in group {alt_group_df['Group ID']} did not satisfy the condition and is filtered out."
                )
                continue

            # Calculate the mean of the criteria
            mean_crits_temp_df = filt_sg_df[self.crit_cols].mean()

            # Add the mean_crits_temp_df columns and results to the base dataframe
            base_temp_df = sg_df[base_cols].drop_duplicates()
            base_temp_df = base_temp_df.assign(**mean_crits_temp_df)

            # add the base_temp_df to the new_dm_df
            if new_dm_df.empty:
                new_dm_df = base_temp_df
            else:
                new_dm_df = pd.concat([new_dm_df, base_temp_df], ignore_index=True)

        # If only one group is present, remove the group column
        if len(new_dm_df["Group ID"].unique()) == 1:
            new_dm_df = new_dm_df.drop(columns=["Group ID"])

        # Create a new DecisionMatrix instance with modified attributes
        new_self = DecisionMatrix(
            metrics_df=new_dm_df,
            objectives=self.objectives,
            alt_cols=self.alt_cols,
            crit_cols=self.crit_cols,
            weights=self.weights,
            group_cols=self.group_cols,
            crit_cats=self.crit_cats,
            group_weights=self.group_weights,
        )

        return new_self

    # def mean_based_criteria(
    #     self,
    #     unc_var_mean_based: List[str]
    # ) -> 'DecisionMatrix':
    #     """
    #     Apply criteria based on mean values of uncertain variables to the given data
    #     and generate a new instance of DecisionMatrix.

    #     Parameters:
    #     - unc_var_mean_based (list): List of uncertain variables based on their means.

    #     Returns:
    #     - new_self (DecisionMatrix): New instance of DecisionMatrix with updated criteria.
    #     """
    #     mean_dict = {}
    #     prob_dict = {}

    #     # Create containers to store mean values and probability distributions
    #     matched_rows_df = pd.DataFrame()

    #     # Iterate through each uncertain variable to determine means or probability distributions
    #     for unc_var in unc_var_mean_based:
    #         dist = self.unc_var_prob_dist[unc_var]
    #         if isinstance(dist, dict):
    #             prob_dict[unc_var] = dist  # Store probability distributions
    #         else:
    #             mean_dict[unc_var] = self.unc_var_prob_dist[unc_var].mean()  # Calculate mean values
    #             matched_rows_df[unc_var] = abs(self.dm_df[unc_var] - mean_dict[unc_var]) < 1e-6  # Find matched rows

    #     # Filter the DataFrame based on matched rows or use the entire DataFrame
    #     if len(matched_rows_df) > 0:
    #         mean_matched_dm_df = self.dm_df[matched_rows_df.all(axis=1)]  # Filtered DataFrame
    #         columns_to_drop = ['Alternative ID', 'Group ID', 'Sample ID']
    #         columns_existing = list(set(columns_to_drop) & set(mean_matched_dm_df.columns))
    #         metrics_df = mean_matched_dm_df.drop(columns_existing, axis=1)  # Drop specific columns
    #     else:
    #         metrics_df = self.dm_df.drop(['Alternative ID', 'Group ID', 'Sample ID'], axis=1)  # Use entire DataFrame

    #     # Identify and update uncertain columns and their respective probability distributions
    #     new_unc_cols = [unc_var for unc_var in self.unc_cols if unc_var not in unc_var_mean_based]
    #     new_unc_var_prob_dist = {unc_var: dist for unc_var, dist in self.unc_var_prob_dist.items() if
    #                              unc_var not in unc_var_mean_based}

    #     # Update criteria based on probability distributions
    #     if prob_dict:
    #         for var, dist in prob_dict.items():
    #             piv_col = var

    #             # Define pivot and index columns for pivot
    #             index_col = [col for col in self.alt_cols + self.unc_cols + self.group_cols + self.crit_cols if
    #                          col not in self.crit_cols + [piv_col] and col in metrics_df.columns]

    #             # Filter out rows where the specified column is ALL or nan
    #             columns = [col for col in self.alt_cols + self.unc_cols + self.group_cols + self.crit_cols if
    #                        col in metrics_df.columns]
    #             filt_dm_df = metrics_df[columns]
    #             filt_dm_df = filt_dm_df[~filt_dm_df[piv_col].isin(['ALL']) & filt_dm_df[piv_col].notna()]

    #             crit_piv_df = filt_dm_df.pivot(index=index_col,
    #                                            columns=piv_col,
    #                                            values=self.crit_cols)

    #             # Reduce the pivoted criteria columns to the expected criteria value
    #             crit_piv_df = crit_piv_df.reset_index()
    #             crit_piv_df.columns = [f'{"_".join(col)}' if col[1] else f'{col[0]}' for col in crit_piv_df.columns]

    #             for crit_col in self.crit_cols:
    #                 # Create column
    #                 crit_piv_df[crit_col] = 0
    #                 # Calculate each probability weigthed contribution
    #                 for event, prob in dist.items():
    #                     crit_piv_df[crit_col] += crit_piv_df[crit_col + '_' + event] * prob
    #                     crit_piv_df = crit_piv_df.drop(crit_col + '_' + event, axis=1)

    #             # Update the metrics df
    #             metrics_df = crit_piv_df

    #     # Initialize a new DecisionMatrix instance with updated attributes
    #     new_self = DecisionMatrix(
    #         metrics_df=metrics_df,
    #         objectives=self.objectives,
    #         alt_cols=self.alt_cols,
    #         crit_cols=self.crit_cols,
    #         weights=self.weights,
    #         group_cols=self.group_cols,
    #         unc_cols=new_unc_cols,
    #         unc_var_prob_dist=new_unc_var_prob_dist,
    #         crit_cats=self.crit_cats,
    #         group_weights=self.group_weights,
    #         # Include other necessary attributes for initialization of the new instance
    #     )

    #     return new_self

    def plot_criteria(self, group_by_category=True):
        """
        Plots the weights of criteria.

        Parameters:
        - group_by_category (bool): If True, the criteria will be grouped by category and displayed as a stacked bar plot.
                       If False, the criteria will be displayed as individual bars.

        Returns:
        None
        """

        # Get a list of unique criteria
        criteria = self.crit_df["Criteria"].unique()

        # Generate a list of unique colors
        colors = list(mcolors.CSS4_COLORS.keys())
        colors.remove("black")  # Remove 'black' from the list of colors
        colors = colors[0 : len(criteria)]

        # Make colors for each criteria and store in dictionary
        criteria_colors = dict(zip(criteria, colors))

        if group_by_category:
            # Create a bar plot from the pivoted DataFrame
            df = self.cat_crit_df.pivot(
                index="Category", columns="Criteria", values="Weight"
            )
            ax = df.plot(
                kind="bar",
                stacked=True,
                figsize=(15, 8),
                color=[criteria_colors[crit] for crit in df.columns],
            )
            # Create an array to store the cumulative height of the bars
            cumulative_height = np.zeros(len(df))
            # Iterate over each bar (patch) in the plot
            for i, p in enumerate(ax.patches):
                # Calculate the index of the current bar in its stack
                bar_index = i % len(df)
                # Update the cumulative height of the bars in the current stack
                cumulative_height[bar_index] += p.get_height()
                # Only annotate bars with a height greater than zero
                if p.get_height() > 0:
                    # Annotate the height (weights value) of each bar on the plot
                    # The coordinates given are (x, y) where x is the bar's x coordinate and y is the cumulative height of the bars in the stack minus half the bar's height
                    ax.annotate(
                        str(round(p.get_height(), 2)),
                        (
                            p.get_x() + p.get_width() / 2.0,
                            cumulative_height[bar_index] - p.get_height() / 2,
                        ),
                        ha="center",
                        va="center",
                    )

            # Set the x-axis label
            ax.set_xlabel("Criteria categories", fontsize=12)
            # Set the x-axis labels to be truncated and tilted
            ax.set_xticklabels([label[:10] for label in df.index], rotation=45)
            # Set the legend
            plt.legend(
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc="lower left",
                ncol=4,
                mode="expand",
                borderaxespad=0.0,
                edgecolor="black",
                title="Criteria",
                fontsize=12,
            )

        else:
            ax = self.crit_df.plot(
                x="Criteria",
                y="Weight",
                kind="bar",
                figsize=(15, 8),
                color=[criteria_colors[crit] for crit in self.crit_df["Criteria"]],
                title="Weight of criteria",
                legend=False,
            )
            for p in ax.patches:
                # Only annotate bars with a height greater than zero
                if p.get_height() > 0:
                    # Annotate the height (weights value) of each bar on the plot
                    # The coordinates given are (x, y) where x is the bar's x coordinate and y is half the bar's height
                    ax.annotate(
                        str(round(p.get_height(), 2)),
                        (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
                        ha="center",
                        va="center",
                    )
            ax.set_xlabel("Criteria", fontsize=12)
            # Set the x-axis labels to be truncated and tilted
            ax.set_xticklabels(
                [label[:10] for label in self.crit_df["Criteria"]], rotation=45
            )

        ax.set_ylabel("Weight", fontsize=12)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

        return

    def calc_rankings(
        self,
        mcdm_methods=MCDM_DEFAULT,
        comp_ranks=COMP_DEFAULT,
        constraints={},
        rank_filt={},
        derived_columns=None,
    ):
        """
        Calculate rankings for a DecisionMatrix instance using specified Multi-Criteria Decision Making (MCDM) methods.

        Parameters:
        - mcdm_methods: dict, optional
            Dictionary of MCDM methods to use for ranking. Defaults to the MCDM_DEFAULT dictionary.
        - comp_ranks: dict, optional
            Dictionary of compromised ranking functions to use. Defaults to the COMP_DEFAULT dictionary.
        - constraints: dict, optional
            Dictionary of constraints to filter the data. Defaults to an empty dictionary.
        - rank_filt: dict, optional
            Dictionary of filters to apply to the ranking. Defaults to an empty dictionary.
        - derived_columns: dict, optional
            Dictionary of derived columns to calculate. Defaults to an empty dictionary.

        Returns:
        - ranks_output: RanksOutput
            An instance of the RanksOutput class containing the rankings.
        """

        # provide criteria weights in array numpy.darray. All weights must sum to 1.
        weights = np.array([self.weights[crit_col] for crit_col in self.crit_cols])
        # provide criteria types in array numpy.darray. Profit criteria are represented by 1 and cost criteria by -1.
        types = np.array([self.objectives[crit_col] for crit_col in self.crit_cols])

        # Create a copy of the decision matrix DataFrame
        red_dm_df = self.dm_df.copy()
        # Check if both 'Group ID' and 'Sample ID' columns exist
        if "Group ID" not in red_dm_df.columns:
            red_dm_df["Group ID"] = "G1"
        if "Sample ID" not in red_dm_df.columns:
            red_dm_df["Sample ID"] = "S1"
        # Pre-filter which sceanrio, groups and
        red_dm_df, _ = filter_dataframe(
            red_dm_df, filter_conditions=rank_filt, derived_columns=derived_columns
        )

        ## Create data frames to store data
        # Define base column
        base_cols = list(self.alternatives_df.columns)
        if isinstance(self.groups_df, pd.DataFrame):
            base_cols += list(self.groups_df.columns)
        if isinstance(self.unc_smpls_df, pd.DataFrame):
            base_cols += list(self.unc_smpls_df.columns)
        # Alternatives not included
        alt_exc_nan_df = pd.DataFrame(
            columns=self.dm_df.columns
        )  # To store nan alternatives
        alt_exc_const_df = pd.DataFrame(
            columns=base_cols + list(constraints.keys())
        )  # To store nan alternatives
        # rank containers
        ranks_crit_df = pd.DataFrame(columns=base_cols + self.crit_cols)
        # ranks_mcdm_methods_df = pd.DataFrame(columns=base_cols + list(mcdm_methods.keys()))
        # ranks_comp_df = pd.DataFrame(columns=base_cols + list(comp_ranks.keys()))
        ranks_MCDM_df = pd.DataFrame(
            columns=base_cols + list(mcdm_methods.keys()) + list(comp_ranks.keys())
        )

        # Check if crit_cols contains any zero values
        if red_dm_df[self.crit_cols].isin([0]).any().any():
            # Iterate through MCDM methods
            for method_name, method_instance in mcdm_methods.items():
                if isinstance(method_instance, (ARAS, CODAS, CRADIS)):
                    print(
                        f"Warning: {method_name} is of type {type(method_instance)}, which may require special handling due to zero values in some criteria columns. Recmonedeation is to replace the zero values with negligaibel numbers.",
                        3 * "...\n",
                    )

        # Iterate through all pairs of 'Group ID' and 'Sample ID'
        for _, group_scen_df in (
            red_dm_df[["Group ID", "Sample ID"]].drop_duplicates().iterrows()
        ):

            # Check if both columns exist
            sg_df = red_dm_df[
                red_dm_df[["Group ID", "Sample ID"]]
                .isin(group_scen_df[["Group ID", "Sample ID"]].values)
                .all(axis=1)
            ]

            # Store all not included alternatives
            # due to NaN values
            nan_alt_rows = sg_df[self.crit_cols].isna().any(axis=1)
            if nan_alt_rows.any():
                # Check if empty or all-NA rows
                if alt_exc_nan_df.empty:
                    alt_exc_nan_df = sg_df[nan_alt_rows]
                else:
                    alt_exc_nan_df = pd.concat(
                        [alt_exc_nan_df, sg_df[nan_alt_rows]], ignore_index=True
                    )
                sg_df.reset_index(drop=True, inplace=True)
                nan_alt_rows.reset_index(drop=True, inplace=True)
                sg_df = sg_df[~nan_alt_rows]

            # Store all not included alternatives
            # due to NaN values
            # nan_alt_rows = sg_df[self.crit_cols].isna().any(axis=1)
            # if nan_alt_rows.any():
            #    alt_exc_nan_df = pd.concat([alt_exc_nan_df, sg_df[nan_alt_rows]], ignore_index=True)
            #    sg_df.reset_index(drop=True, inplace=True)
            #    nan_alt_rows.reset_index(drop=True, inplace=True)
            #    sg_df = sg_df[~nan_alt_rows]

            if constraints:
                sg_df, boolean_df = filter_dataframe(
                    sg_df,
                    filter_conditions=constraints,
                    derived_columns=derived_columns,
                    base_cols=base_cols,
                )
                alt_exc_const_df = pd.concat(
                    [
                        alt_exc_const_df,
                        boolean_df[
                            ~(boolean_df[constraints.keys()] == True).all(axis=1)
                        ],
                    ],
                    ignore_index=True,
                )

            # Find the smallest negative number in each column and add its absolute value to the column
            # matrix_df = sg_df[self.crit_cols].copy()
            # matrix_df = matrix_df.apply(lambda col: col + abs(col.min()) if col.min() < 0 else col)

            # Convert the DataFrame back to a numpy array
            # matrix = matrix_df.to_numpy()
            # add a random small positive values to each element of the matrix
            # matrix = matrix + np.random.rand(*matrix.shape) * 1e-9

            # Find the smallest number in each column and add its absolute value plus one to the column
            matrix_df = sg_df[self.crit_cols].copy()
            matrix_df = matrix_df.apply(lambda col: col + abs(col.min()) + 1)

            # Convert the DataFrame back to a numpy array
            matrix = matrix_df.to_numpy()

            # Add a random small positive value to each element of the matrix
            matrix = matrix + np.random.rand(*matrix.shape) * 1e-4

            if matrix.any():

                # Temp container
                temp_ranks_MCDM_df = sg_df[base_cols].copy()

                ## Calc ranking for each MCDM method
                for pipe in mcdm_methods.keys():

                    # Calculate the preference values of alternatives
                    if not isinstance(mcdm_methods[pipe], SPOTIS):
                        pref = mcdm_methods[pipe](matrix, weights, types)
                    else:
                        # SPOTIS preferences must be sorted in ascending order
                        bounds_min = np.amin(matrix, axis=0)
                        bounds_max = np.amax(matrix, axis=0)
                        bounds = np.vstack((bounds_min, bounds_max))
                        # Calculate the preference values of alternatives
                        pref = mcdm_methods[pipe](matrix, weights, types, bounds)

                    # Generate ranking of alternatives by sorting alternatives descendingly according to the TOPSIS algorithm (reverse = True means sorting in descending order) according to preference values
                    if isinstance(mcdm_methods[pipe], (MULTIMOORA)):
                        temp_ranks_MCDM_df.loc[~nan_alt_rows, pipe] = mcdm_methods[
                            pipe
                        ](
                            matrix, weights, types
                        )  # Mu;timoora includes ranker
                    elif isinstance(mcdm_methods[pipe], (VIKOR, SPOTIS)):
                        temp_ranks_MCDM_df.loc[~nan_alt_rows, pipe] = rank_preferences(
                            pref, reverse=False
                        )
                    else:
                        temp_ranks_MCDM_df.loc[~nan_alt_rows, pipe] = rank_preferences(
                            pref, reverse=True
                        )

                # Calc compromised ranking
                if comp_ranks:
                    for comp_rank in comp_ranks.keys():
                        temp_ranks_MCDM_df.loc[~nan_alt_rows, comp_rank] = comp_ranks[
                            comp_rank
                        ](
                            temp_ranks_MCDM_df.loc[
                                ~nan_alt_rows, mcdm_methods.keys()
                            ].to_numpy()
                        )

                # Populate the containers
                # Exclude empty or all-NA columns before concatenation
                # Check if temp_ranks_MCDM_df is empty
                temp_ranks_MCDM_df = temp_ranks_MCDM_df.dropna(how="all", axis=1)
                if temp_ranks_MCDM_df.empty:
                    pass
                elif ranks_MCDM_df.empty:
                    ranks_MCDM_df = temp_ranks_MCDM_df
                else:
                    ranks_MCDM_df = pd.concat(
                        [ranks_MCDM_df, temp_ranks_MCDM_df], ignore_index=True
                    )
                ranks_crit_df = pd.concat(
                    [
                        ranks_crit_df,
                        ranks_columns(
                            sg_df, columns=self.crit_cols, objectives=self.objectives
                        ),
                    ],
                    ignore_index=True,
                )  # Calc criteria ranking
                ranks_df = pd.merge(ranks_crit_df, ranks_MCDM_df)

        # Check if ranks_MCDM_df is empty
        if ranks_MCDM_df.empty:
            print("No alternatives to rank.")
            return
        # Store all not included alternatives with ranking zero
        # Get all columns from list(mcdm_methods.keys()) that have a value of zero in ranks_MCDM_df
        # zero = ranks_MCDM_df.columns[ranks_MCDM_df.isin([0]).any()].tolist()

        # TODO: Quick fix to add Group ID and Sample ID to the ranks_MCDM_df
        if "Group ID" not in base_cols:
            ranks_MCDM_df["Group ID"] = "G1"
        if "Sample ID" not in base_cols:
            ranks_MCDM_df["Sample ID"] = "S1"

        return RanksOutput(
            ranks_df,
            ranks_crit_df,
            ranks_MCDM_df,
            alt_exc_nan_df,
            alt_exc_const_df,
            list(mcdm_methods.keys()),
            list(comp_ranks.keys()),
            self,
        )

    def calc_imprt_sensitivity(
        self,
        mcdm_methods,
        comp_ranks={},
        crit_cols_dict={},
        cat_crit_dict={},
        imp_tot=np.linspace(0, 1, 11),
        crit_tag="Criteria",
        alt_tag="Alternative ID",
        **ranking_args,
    ):
        """
        Calculate the sensitivity of the rankings to the weights of the criteria.

        Parameters:
        - mcdm_methods (dict):
            A dictionary with MCDM method names as keys and ranking functions as values.
        - comp_ranks (dict):
            A dictionary specifying compromise ranking methods.
        - crit_cols_dict (dict):
            A dictionary specifying the criteria columns to use for the sensitivity analysis.
        - cat_crit_dict (dict):
            A dictionary specifying the category criteria to use for the sensitivity analysis.
        - imp_tot (numpy.ndarray):
            An array of total importance values to use for the sensitivity analysis.
        - crit_tag (str):
            The tag to use for the criteria column.
        - alt_tag (str):
            The tag to use for the alternative column.
        - **ranking_args:
            Additional keyword arguments for the ranking methods.

        Returns:
        - ranks_imp_df (pd.DataFrame):
            A DataFrame containing the rankings at the highest weights.
        - imp_sens_df (pd.DataFrame):
            A DataFrame containing the sensitivity of the weights values.
        """

        # Get the criteria dataframe from the decision matrix object
        cat_crit_df = self.cat_crit_df
        crit_df = self.crit_df

        # Check if a category criteria dictionary is provided
        if cat_crit_dict:
            # If so, create a group dataframe based on the category criteria
            crit_group_df = cat_crit_df[
                cat_crit_df[cat_crit_dict.keys()]
                .isin(cat_crit_dict.values())
                .all(axis=1)
            ][[crit_tag, "Weight"]]
            # Create a non-group dataframe for the remaining criteria
            crit_non_group_df = cat_crit_df[
                ~cat_crit_df[cat_crit_dict.keys()]
                .isin(cat_crit_dict.values())
                .all(axis=1)
            ][[crit_tag, "Weight"]]
            # store the items in xlabel as strings
            xlabel = list(cat_crit_dict.items())[0][1]
        else:
            # If not, create a group dataframe based on the criteria columns dictionary
            crit_group_df = crit_df[
                crit_df[crit_tag].isin(list(crit_cols_dict.items())[0][1])
            ][[crit_tag, "Weight"]]
            # Create a non-group dataframe for the remaining criteria
            crit_non_group_df = crit_df[
                ~crit_df[crit_tag].isin(list(crit_cols_dict.items())[0][1])
            ][[crit_tag, "Weight"]]
            xlabel = list(crit_cols_dict.items())[0][0]

        # Create a dataframe to store the sensitivity of the weights values
        imp_sens_df = pd.DataFrame(
            index=imp_tot,
            columns=list(crit_group_df[crit_tag].unique())
            + list(crit_non_group_df[crit_tag].unique()),
        )

        # Iterate over the total weights values
        for imp in imp_tot:
            # Create a new group weights dataframe
            new_group_weights = crit_group_df.copy()
            # Update the weights values based on the current total weights
            new_group_weights["Weight"] = (
                imp * crit_group_df["Weight"] / crit_group_df["Weight"].sum()
            )
            # Create a new non-group weights dataframe
            new_non_group_weights = crit_non_group_df.copy()
            # Update the weights values based on the current total weights
            new_non_group_weights["Weight"] = (
                (1 - imp)
                * crit_non_group_df["Weight"]
                / crit_non_group_df["Weight"].sum()
            )
            # Update the sensitivity dataframe with the new weights values
            imp_sens_df.loc[imp, new_group_weights[crit_tag]] = new_group_weights[
                "Weight"
            ].values
            imp_sens_df.loc[imp, new_non_group_weights[crit_tag]] = (
                new_non_group_weights["Weight"].values
            )

        # Check if compromise ranks are provided
        if comp_ranks:
            # If so, get the ranking method name
            rank_method_name = list(comp_ranks.keys())[0]
        elif len(mcdm_methods) != 1:
            # If not, check if only one MCDM method is provided
            print(
                "You need to specify a compromise ranking method or only one MCDM method"
            )
        else:
            # If only one MCDM method is provided, get the method name
            rank_method_name = list(mcdm_methods.keys())[0]

        # Iterate over the rows in the weights sensitivity dataframe
        for idx, row in enumerate(imp_sens_df.iterrows()):
            # Get the new weights values from the current row
            new_weights = row[1].to_dict()
            # Create a temporary decision matrix with the new weights values
            dm_temp = DecisionMatrix(
                metrics_df=self.metrics_df,
                objectives=self.objectives,
                alt_cols=self.alt_cols,
                crit_cols=list(self.objectives.keys()),
                weights=new_weights,
            )
            # Calculate the rankings with the temporary decision matrix
            rank_obj_temp = dm_temp.calc_rankings(
                mcdm_methods=mcdm_methods, comp_ranks=comp_ranks, **ranking_args
            )

            # Check if the rank object contains more than one group or sample
            if (
                len(rank_obj_temp.ranks_df["Group ID"].unique()) != 1
                or len(rank_obj_temp.ranks_df["Sample ID"].unique()) != 1
            ):
                raise ValueError(
                    "The rank object contains more than one group or sample"
                )

            # Store the rankings in a results dataframe
            if idx == 0:
                # If it's the first row, create a new dataframe
                ranks_imp_df = rank_obj_temp.ranks_df[
                    [alt_tag, rank_method_name]
                ].copy()
                ranks_imp_df["Weight"] = row[0]
            else:
                # If it's not the first row, create a temporary dataframe and append it to the results dataframe
                ranks_imp_df_temp = rank_obj_temp.ranks_df[
                    [alt_tag, rank_method_name]
                ].copy()
                ranks_imp_df_temp["Weight"] = row[0]
                ranks_imp_df = pd.concat([ranks_imp_df, ranks_imp_df_temp], axis=0)

        # Plot the rankings at the highest weights
        plot_rank_sens_weights(
            ranks_imp_df, alt_tag, rank_method_name, xlabel, order_by="highest"
        )
        # Plot the weights sensitivity dataframe
        plot_crit_weights_sensitivity(imp_sens_df, xlabel)
        # Plot the rankings at the lowest weights
        # plot_rank_sens_weights(ranks_imp_df, alt_tag, rank_method_name, xlabel, order_by='lowest')

        # Return the results dataframe and the weights sensitivity dataframe
        return ranks_imp_df, imp_sens_df


def ranks_columns(df, columns, objectives):
    """
    Rank specified columns in a DataFrame according to provided ranking objectives.

    Parameters:
        df (pandas.DataFrame):
            The DataFrame containing the data to be ranked.
        columns (list of str):
            A list of column names to be ranked.
        objectives (dict of {str: callable}):
            A dictionary with column names as keys and ranking objective functions as values.

    Returns:
        pandas.DataFrame: A new DataFrame with the specified columns ranked according to the objectives.
    """

    # Mapping for function selection
    FUNCTION_MAP = {"1": False, "-1": True}

    # Copy the input DataFrame
    ranked_df = df.copy()

    # Iterate over the columns to be ranked
    for col in columns:
        # Rank the column based on the specified objective
        ranked_df[col] = ranked_df[col].rank(
            method="min", ascending=FUNCTION_MAP[str(objectives[col])]
        )
        # Convert the ranks to integers
        ranked_df[col] = ranked_df[col].astype(int)

    return ranked_df


def plot_crit_weights_sensitivity(imp_sens_df, xlabel):
    # Plot the DataFrame as a stacked bar plot
    ax = imp_sens_df.plot(kind="bar", stacked=True, figsize=(12, 6))

    # Set the x-axis label
    ax.set_xlabel(f"Weight of {xlabel}", fontsize=14)

    # Set the y-axis label
    ax.set_ylabel("Criteria Weight", fontsize=14)

    # Set x-axis ticks to be in percentage format with no decimals
    ax.set_xticklabels(
        [f"{int(tick*100)}%" for tick in imp_sens_df.index], rotation=0, fontsize=12
    )

    # Format the y-axis labels to be in percentage format with no decimals
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    # Place the legend on the right side of the plot and set its title to "Criteria"
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Criteria", fontsize=12)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_rank_sens_weights(
    ranks_imp_df, alt_tag, rank_method_name, xlabel, order_by="highest"
):
    # Pivot the DataFrame to make each alternative a column
    plot_df = ranks_imp_df.pivot(
        index="Weight", columns=alt_tag, values=rank_method_name
    )

    # Define a color map
    color_map = cm.get_cmap("tab10", len(plot_df.columns))

    # Create a dictionary that maps each column name to a specific color
    color_dict = {col: color_map(i) for i, col in enumerate(plot_df.columns)}

    # Reorder the columns according to the rank at the highest or lowest weights
    if order_by == "highest":
        plot_df = plot_df[plot_df.iloc[-1].sort_values(ascending=False).index]
        legend_loc = (1.05, 0.5)
    elif order_by == "lowest":
        plot_df = plot_df[plot_df.iloc[0].sort_values(ascending=False).index]
        legend_loc = (-0.3, 0.5)

    # Plot the DataFrame with the color map
    ax = plot_df.plot(
        kind="line",
        grid=True,
        figsize=(12, 6),
        color=[color_dict[col] for col in plot_df.columns],
    )

    # Set the x-axis label
    ax.set_xlabel(f"Total weights of {xlabel}", fontsize=14)

    # Set the y-axis label
    ax.set_ylabel("Rank", fontsize=14)

    # Format the x-axis labels to be in percentage format with no decimals
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

    # Set the y-axis limits
    ax.set_ylim(0, plot_df.max().max() + 1)

    # set the x-axis limits
    ax.set_xlim(plot_df.index[0], plot_df.index[-1])

    # Set the y-ticks to be from 1 to the maximum rank number
    ax.yaxis.set_ticks(range(1, int(plot_df.max().max() + 2)))

    # Enable the grid for each y-tick value
    ax.yaxis.grid(True)

    # Create a custom legend for the rank at the highest or lowest weights
    lines = [
        mlines.Line2D(
            [],
            [],
            color=color_dict[col],
            label=f'{col} ({int(plot_df.iloc[-1 if order_by == "highest" else 0, i])})',
        )
        for i, col in enumerate(plot_df.columns)
    ]
    legend = plt.legend(
        handles=lines,
        bbox_to_anchor=legend_loc,
        loc="center left",
        borderaxespad=0.0,
        edgecolor="black",
        fontsize=14,
        title=f'Rank at {int(plot_df.index[-1 if order_by == "highest" else 0]*100)}%',
    )

    # Show the plot
    plt.tight_layout()
    plt.show()
