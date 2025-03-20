import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# Define file paths
tt = "../models/MPI3d/ablative/mpi3d_selection_0_enc{i}_tr{j}/results_enc{i}_tr{j}.csv"
file_paths = {
    1: [tt.format(i=1, j=1), tt.format(i=1, j=2), tt.format(i=1, j=3)],
    2: [tt.format(i=2, j=1), tt.format(i=2, j=2), tt.format(i=2, j=3)],
    3: [tt.format(i=3, j=1), tt.format(i=3, j=2), tt.format(i=3, j=3)],
    4: [tt.format(i=4, j=1), tt.format(i=4, j=2), tt.format(i=4, j=3)],
}

# Factor name mappings with a strict order
factor_name_map = OrderedDict([
    ("OBJ_COLOR", r"$\mathbf{\ color}$"),
    ("OBJ_SHAPE", r"${\ shape}$"),
    ("OBJ_SIZE",  r"${\ size}$"),
    ("CAMERA",    r"${\ cam.}$"),
    ("BACKGROUND", r"${\ back.}$"),
    ("H_AXIS", r"${\ hori.}$"),
    ("V_AXIS", r"${\ vert.}$")
])

# Load and aggregate results
aggregated_results = {}

for enc_size, files in file_paths.items():
    data_list = []
    for file in files:
        df = pd.read_csv(file)
        data_list.append(df)

    # Compute mean and standard deviation across different seeds
    combined_df = pd.concat(data_list)
    mean_results = combined_df.groupby("semantic_name").mean(numeric_only=True)
    std_results = combined_df.groupby("semantic_name").std(numeric_only=True)

    aggregated_results[enc_size] = (mean_results, std_results)

# Extract unique factors from the dataset
all_factors = set()
for enc_size, (mean_results, std_results) in aggregated_results.items():
    all_factors.update(mean_results.index.tolist())

# Generate separate plots with adjusted font sizes for x and y ticks
fig, axes = plt.subplots(1, 2, figsize=(22, 6), gridspec_kw={'wspace': 0.3})  # Added wspace for better spacing

# Titles for the plots
titles = ["Linear Model (LogReg)", "Nonlinear Model (MLP)"]
metrics = ["mcc_logreg", "mcc_mlp"]

for ax, title, metric in zip(axes, titles, metrics):
    plotted_handles = {}  # Store handles for ordered legend

    for factor in factor_name_map.keys():  # Enforce strict order in plotting
        if factor in all_factors:  # Only process existing factors
            mean_values = []
            std_values = []
            x_ticks = []

            for enc_size in sorted(aggregated_results.keys()):
                mean_results, std_results = aggregated_results[enc_size]

                if factor in mean_results.index:
                    mean_values.append(mean_results.loc[factor, metric])
                    std_values.append(std_results.loc[factor, metric])
                    x_ticks.append(enc_size)

            if mean_values:
                line = ax.errorbar(
                    x_ticks, mean_values, yerr=std_values,
                    fmt='-o', capsize=5, linewidth=2, label=factor_name_map[factor]
                )
                plotted_handles[factor_name_map[factor]] = line  # Store handle

    ax.set_xlabel("Encoding Size", fontsize=23)
    if metric == "mcc_logreg":
        ax.set_ylabel("MCC", fontsize=23)
    else:
        ax.set_ylabel("")

    ax.set_title(f"{title}", fontsize=23)
    ax.set_xticks(sorted(aggregated_results.keys()))
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Ensure the legend order strictly follows factor_name_map
    ordered_handles = [plotted_handles[factor_name_map[factor]] for factor in factor_name_map.keys() if factor_name_map[factor] in plotted_handles]

    ax.legend(ordered_handles, [factor_name_map[f] for f in factor_name_map.keys() if factor_name_map[f] in plotted_handles], 
              title="", fontsize=18)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("../models/MPI3d/mpi_ablation_encodingsize.pdf", dpi=300, bbox_inches="tight")
plt.show()
