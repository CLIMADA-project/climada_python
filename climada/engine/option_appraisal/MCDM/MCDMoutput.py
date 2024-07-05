# I want to make a data frame container class that stores the following four data frames ranks_df, ranks_crit_df, ranks_MCDM_df, alt_exc_nan_df, alt_exc_const_df and has the following methods:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from .utils import filter_dataframe


# make a class
class RanksOutput:
    def __init__(self, ranks_df, ranks_crit_df, ranks_MCDM_df, alt_exc_nan_df, alt_exc_const_df, mcdm_cols, comp_rank_cols, dm):
        self.ranks_df = ranks_df
        self.ranks_crit_df = ranks_crit_df
        self.ranks_MCDM_df = ranks_MCDM_df
        self.alt_exc_nan_df = alt_exc_nan_df
        self.alt_exc_const_df = alt_exc_const_df
        
        self.mcdm_cols = mcdm_cols
        self.comp_rank_cols = comp_rank_cols

        self.dm = dm

        # Check if self.dm.unc_smpls_df is not None 
        if isinstance(self.dm.unc_smpls_df, pd.DataFrame):
            self.counts_rank_df, self.rel_counts_rank_df = calculate_counts(self.dm.crit_cols, mcdm_cols, comp_rank_cols, self.dm.unc_smpls_df, ranks_df)
        else:
            self.counts_rank_df = None
            self.rel_counts_rank_df = None


        


    # make a method to plot the ranks
    def plot_ranks(self, rank_type = 'MCDM', alt_name_col = 'Alternative ID', disp_rnk_cols = [], sort_by_col = None, transpose = False, group_id = 'G1', state_id = 'S1'):

        # Get the disp_rnk_cols
        if disp_rnk_cols:
            legend_title = 'Rank columns'
            df = self.ranks_df
        elif rank_type == 'criteria':
            legend_title = 'Criteria'
            df = self.ranks_crit_df
            disp_rnk_cols = self.dm.crit_cols
        elif rank_type == 'MCDM':
            legend_title = 'MCDM method'
            df = self.ranks_MCDM_df
            disp_rnk_cols = self.mcdm_cols + self.comp_rank_cols
        
        # Filter out based on group_id and state_id
        df = df[df['Group ID'] == group_id]
        df = df[df['Sample ID'] == state_id]

        # Filter out the columns
        df = df[[alt_name_col] + disp_rnk_cols ]

        # Store number of ranks
        step = 1
        list_rank = np.arange(1, len(df) + 1, step)

        # Sort the columns
        if sort_by_col:
            df = df.sort_values(by=sort_by_col, ascending=True)

        # Check if transpose
        if not transpose:
            df = df.set_index(alt_name_col)    
        else:
            df = df.set_index(alt_name_col).transpose()
            # Rename the index
            df.index.name = 'Rank columns'
            # Rename the legend title
            legend_title = alt_name_col

       # Plot the dataframe
        ax = df.plot(kind='bar', width = 0.8, stacked=False, edgecolor = 'black', figsize = (15,8))
        ax.set_xlabel(df.index.name, fontsize = 12)
        ax.set_ylabel('Rank', fontsize = 12)
        ax.set_yticks(list_rank)

        # Make rotation of the labels tilted 45 degrees and truncate to the first 10 characters
        ax.set_xticklabels([label[:10] for label in df.index], rotation = 45)
        ax.tick_params(axis = 'both', labelsize = 12)
        y_ticks = ax.yaxis.get_major_ticks()
        ax.set_ylim(0, len(list_rank) + 1)

        # Legend
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=4, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 12, title = legend_title)

        ax.grid(True, linestyle = ':')
        ax.set_axisbelow(True)
        plt.tight_layout()
        plt.show()


    # make a print function
    def print_rankings(self, disp_filt={}, disp_rnk_cols = [], rank_type='MCDM', sort_by_col=None):
        
        # Filter the rank columns
        filt_rank_df  = filter_dataframe(self.ranks_df, disp_filt)[0]

        # Get the disp_rnk_cols
        if disp_rnk_cols:
            pass
        elif rank_type == 'criteria':
            disp_rnk_cols = self.dm.crit_cols
        elif rank_type == 'MCDM':
            disp_rnk_cols = self.mcdm_cols + self.comp_rank_cols

        # Define base column
        base_cols = list( self.dm.alternatives_df.columns) + ['Group ID', 'Sample ID']
        if isinstance(self.dm.groups_df, pd.DataFrame):
            base_cols += list(self.dm.groups_df.columns)
        if isinstance(self.dm.unc_smpls_df, pd.DataFrame):
            base_cols += list(self.dm.unc_smpls_df.columns)

        # Remove duplicates in base_cols
        base_cols = list(dict.fromkeys(base_cols))

        # Ranking columns  to print
        filt_rank_df = filt_rank_df[base_cols +disp_rnk_cols]

        # Print the rankings per group and state combo
        for _, group_scen_df in filt_rank_df[['Group ID', 'Sample ID']].drop_duplicates().iterrows():
            # Print if there are more than one group and state
            if len(filt_rank_df[['Group ID']].drop_duplicates()) > 1 and len(filt_rank_df[['Sample ID']].drop_duplicates()) > 1:
                group_id = group_scen_df['Group ID']
                scen_id = group_scen_df['Sample ID']
                print(f'Group: {group_id}, State: {scen_id}')
                print('-----------------------------------')
            elif len(filt_rank_df[['Group ID']].drop_duplicates()) > 1:
                group_id = group_scen_df['Group ID']
                print(f'Group: {group_id}')
                print('-----------------------------------')
            elif len(filt_rank_df[['Sample ID']].drop_duplicates()) > 1:
                scen_id = group_scen_df['Sample ID']
                print(f'State: {scen_id}')
                print('-----------------------------------')


            # Filter the group and state
            sg_df = filt_rank_df[filt_rank_df[['Group ID', 'Sample ID']].isin(group_scen_df[['Group ID', 'Sample ID']].values).all(axis=1)]

            # For the print exclude the group and state columns and the  index column and sort by  sort_by_col
            if sort_by_col:
                print_df = sg_df.drop(['Group ID', 'Sample ID'], axis=1).sort_values(by=sort_by_col, ascending=True)
            else:
                print_df = sg_df.drop(['Group ID', 'Sample ID'], axis=1)
            print_df = print_df.set_index('Alternative ID')
            print(tabulate(print_df, headers='keys', tablefmt='psql'))
            print('\n')
            
    def plot_rank_distribution(self,  disp_rnk_col, alt_name_col = 'Alternative ID', sort_by_perf=True):
        # Assuming df is your DataFrame and it's already been prepared as needed
        pivot_df = self.rel_counts_rank_df.pivot(index=alt_name_col, columns='Rank_Count', values=disp_rnk_col)

        # Move column with 0 rank to the end
        pivot_df = pivot_df[[col for col in pivot_df.columns if col != 0] + [0]]
        # Rename the column to null
        pivot_df.rename(columns={0: 'null'}, inplace=True)

        # Normalize the data to get percentages and multiply by 100
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        if sort_by_perf:
            # Calculate the mean for each alternative based on multplying the column value with the cell calue for each row
            # exclude the last column which is the 0 rank
            pivot_df['mean'] = pivot_df.apply(lambda row: np.mean(row[:-1] * pivot_df.columns[:-1]), axis=1)
            # Sort the pivoted data frame based on the sorted_cum_sum_df
            sorted_pivot_df = pivot_df.sort_values(by='mean', ascending=True).drop('mean', axis=1)

        # Create a colormap
        cmap = plt.get_cmap('plasma')  # Changed to a more contrasting colormap
        colors = cmap(np.linspace(0, 1, len(pivot_df.columns)))

        # Plot the DataFrame
        ax = sorted_pivot_df.plot(kind='bar', stacked=True, figsize=(15,10), color=colors)  # Increased figure size

        plt.title('Distribution of Ranking Results', fontsize=20)
        plt.xlabel(alt_name_col, fontsize=16)
        plt.ylabel('Percentage of Total Samples', fontsize=16)

        # Move legend to the left side and increase its size
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 14}, title='Rank')

        # Loop through the bars to annotate each segment with Rank_Count
        for bar in ax.containers:
            for rect in bar:
                # Calculate height and width for the annotation position
                height = rect.get_height()
                width = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                
                # The label is the Rank_Count, which corresponds to the column names in pivot_df
                # We identify the Rank_Count based on the rectangle's position and size
                label = bar.get_label()
                
                # Only annotate if there's enough space (height) in the bar segment
                if height > 0:
                    ax.text(x + width/2, y + height/2, str(label), ha='center', va='center', color='white', fontsize=12)  # Changed text color to white for better visibility

        # Tilt the x-axis labels
        plt.xticks(rotation=45)

        plt.show()
        


def calculate_counts(crit_cols, mcdm_cols, comp_rank_cols, unc_smpls_df, ranks_df):
    # Calculate max rank value and create base rank count DataFrame
    max_rank_value = ranks_df[crit_cols + mcdm_cols + comp_rank_cols].max().max()
    base_rank_count_df = pd.DataFrame({'Rank_Count': range(max_rank_value+1), 'merge_': 1})

    # Calculate max rank value and create base rank count DataFrame
    max_rank_value = ranks_df[crit_cols + mcdm_cols + comp_rank_cols].max().max()
    base_rank_count_df = pd.DataFrame({'Rank_Count': range(max_rank_value+1), 'merge_': 1})

    # Define columns
    rank_cols = crit_cols + mcdm_cols + comp_rank_cols
    base_cols = [col for col in ranks_df.columns if col not in rank_cols + list(unc_smpls_df.columns)]

    # Initialize result DataFrames
    all_count_ranks_df, all_rel_counts_df = pd.DataFrame(), pd.DataFrame()

    # Iterate through all unique 'Group ID's
    for _, group_df in ranks_df[['Group ID']].drop_duplicates().iterrows():
        sg_df = ranks_df[ranks_df[['Group ID']].isin(group_df[['Group ID']].values).all(axis=1)]
        
        # Prepare counts DataFrame
        counts_df = sg_df[base_cols].copy().drop_duplicates()
        counts_df = pd.merge(base_rank_count_df, counts_df.assign(merge_=1), on='merge_').drop('merge_', axis=1)
        counts_df[crit_cols + mcdm_cols + comp_rank_cols] = 0

        # Count the relative number of ranks for each alternative and store in ranks_count_df
        for rank_count in range(max_rank_value+1):
            for alt in sg_df['Alternative ID'].unique():
                for col in rank_cols:
                    count = ((sg_df[col] == rank_count) & (sg_df['Alternative ID'] == alt)).sum()
                    row_idx = ((counts_df['Rank_Count'] == rank_count) & (counts_df['Alternative ID'] == alt))
                    counts_df.loc[row_idx, col]  = count

        # Append counts_df to count_ranks_df
        all_count_ranks_df = pd.concat([all_count_ranks_df, counts_df])
        
        # Calculate the relative counts
        rel_counts_df = counts_df.copy()
        rel_counts_df[rank_cols] = rel_counts_df[rank_cols]/len(sg_df['Sample ID'].unique())

        # Append rel_counts_df to all_rel_counts_df
        all_rel_counts_df = pd.concat([all_rel_counts_df, rel_counts_df])

    return all_count_ranks_df, all_rel_counts_df


