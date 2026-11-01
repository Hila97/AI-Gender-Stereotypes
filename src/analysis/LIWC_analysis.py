"""
    original RQ3, not included in the final project.
    LIWC analysis for RQ2 on the comments presented at liwc_v3_analysis.py
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from pathlib import Path
import numpy as np
import os
import six

# --- Configuration & Paths ---
POSTS_PATH = Path("/content/gdrive/MyDrive/Data mining/text mining/data/processed/final_posts_dataset.csv")
RESULTS_DIR = Path("/content/gdrive/MyDrive/Data mining/text mining/visualization")

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# --- LIWC Categories (Raw Names) ---
LIWC_COLS_RAW = [
    "LIWC_power_achievement",
    "LIWC_cognitive_processes",
    "LIWC_negative_emotions",
    "LIWC_certainty",
    "LIWC_affiliation_social",
    "LIWC_perceptual_processes"
]

# --- Helper Function for Table Rendering ---
def render_mpl_table(data, col_width=4.0, row_height=0.625, font_size=14,
                     header_color='#e8eaf6', row_colors=['#ffffff', '#ffffff'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    """
    Renders a pandas DataFrame as a matplotlib image.
    header_columns: Number of columns (from left) to be colored like the header.
    """
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        
        # --- Center Alignment for ALL cells ---
        cell.set_text_props(ha='center', va='center')
        
        # Apply Header Colors (Top row OR first columns)
        if k[0] == 0 or k[1] < header_columns:
            # Re-apply bold for headers, keeping center alignment
            cell.set_text_props(weight='bold', color='black', ha='center', va='center')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
            
    return ax

def analyze_rq3_liwc():
    print("Loading posts dataset...")
    df = pd.read_csv(POSTS_PATH)
    
    # 1. Filter by gender and self-talk
    df_self = df[
        (df['gender_context'].isin(['female', 'male'])) & 
        (df['is_self_talk'] == True)
    ].copy()
    
    print(f"Total posts analyzing (Self-Talk only): {len(df_self)}")
    print(f"Female posts: {len(df_self[df_self['gender_context']=='female'])}")
    print(f"Male posts:   {len(df_self[df_self['gender_context']=='male'])}")

    # 2. OPTIMIZATION: Rename columns globally here
    # Create a mapping: { 'LIWC_power_achievement': 'Power Achievement', ... }
    rename_map = {col: col.replace("LIWC_", "").replace("_", " ").title() for col in LIWC_COLS_RAW}
    
    # Apply renaming to the DataFrame
    df_self.rename(columns=rename_map, inplace=True)
    
    # Create a list of the NEW clean names to use from now on
    clean_cols = list(rename_map.values())

    # 3. Prepare data for Graph (Melt)
    df_long = df_self.melt(
        id_vars=['gender_context'], 
        value_vars=clean_cols,      # Use clean names directly
        var_name='LIWC_Category', 
        value_name='Score'
    )
    
    # --- Visualization 1: Barplot ---
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    sns.barplot(
        data=df_long, 
        x='LIWC_Category', 
        y='Score', 
        hue='gender_context', 
        palette={'female':'#d5a6bd', 'male': '#a4c2f4'}, 
        ci=95, 
        capsize=0.1
    )
    plt.title('Psychological Analysis (LIWC) of Self-Talk Posts: Female vs. Male', fontsize=16)
    plt.ylabel('Percentage of Words used (%)', fontsize=12)
    plt.xlabel('Psychological Category', fontsize=12)
    plt.legend(title='Gender')
    plt.grid(axis='y', alpha=0.3)
    
    # Save Barplot
    save_path_bar = RESULTS_DIR / "RQ3_LIWC_Barplot.png"
    plt.savefig(save_path_bar, bbox_inches='tight', dpi=300)
    print(f"Saved Barplot to: {save_path_bar}")
    plt.show()

    # --- Statistical Analysis & Table ---
    print("\n" + "="*60)
    print("Calculating Statistics...")
    print("="*60)
    
    results_list = []
    
    # Loop over the CLEAN names directly
    for col in clean_cols:
        group_f = df_self[df_self['gender_context'] == 'female'][col]
        group_m = df_self[df_self['gender_context'] == 'male'][col]
        
        # T-Test
        t_stat, p_val = ttest_ind(group_f, group_m, equal_var=False)
        
        res_str = ""
        if p_val < 0.05:
            if group_f.mean() > group_m.mean():
                res_str = "Female > Male (SIG)"
            else:
                res_str = "Male > Female (SIG)"
        else:
            res_str = "No Diff"
            
        # Format P-Value
        if p_val < 0.0001:
            p_val_display = f"{p_val:.2e}"
        else:
            p_val_display = f"{p_val:.4f}"
            
        results_list.append([
            col,  # 'col' is already the clean name (e.g., "Power Achievement")
            f"{group_f.mean():.4f}", 
            f"{group_m.mean():.4f}", 
            p_val_display,
            res_str
        ])

    df_results = pd.DataFrame(results_list, columns=['Category', 'Female Mean', 'Male Mean', 'P-Value', 'Result'])

    # Print to console
    print(df_results.to_string(index=False))
    
    # --- Visualization 2: Save Table as Image ---
    ax = render_mpl_table(
        df_results, 
        header_columns=1, 
        col_width=4.5,  
        header_color='#a2c4c9' 
    )
    plt.title("Table 2: LIWC Analysis Results (Self-Talk)", fontsize=16, y=1.1)
    
    save_path_table = RESULTS_DIR / "RQ3_LIWC_Stats_Table.png"
    plt.savefig(save_path_table, bbox_inches='tight', dpi=300)
    print(f"Saved Stats Table to: {save_path_table}")
    plt.show()

def main():
    print("=" * 60)
    print("STARTING LIWC analysis")
    print("=" * 60)
    analyze_rq3_liwc()

if __name__ == "__main__":
    main()