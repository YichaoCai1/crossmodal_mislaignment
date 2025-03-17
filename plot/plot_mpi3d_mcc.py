import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
from decimal import Decimal, ROUND_HALF_UP

# Define mode: "selection" or "perturbation"
mode = "perturbation"  # Change to "selection" if needed

# Define root directory
root_path = f"../models/MPI3d/"
file_paths = glob.glob(os.path.join(root_path, f"mpi3d_{mode}_*_tr*/results.csv"))  

data_frames = []

# Fixed factor order
fixed_factor_order = ["OBJ_COLOR", "OBJ_SHAPE", "OBJ_SIZE", "CAMERA", "BACKGROUND", "H_AXIS", "V_AXIS"]

# Regex pattern to extract perturbation number and seed number
pattern = re.compile(f"mpi3d_{mode}_(\d+)_tr(\d+)")

for file_path in file_paths:
    match = pattern.search(file_path)
    if match:
        mode_num = int(match.group(1))
        seed_num = int(match.group(2))

        df = pd.read_csv(file_path, index_col=0)  # Remove extra index column
        df[f"{mode}"] = f"{mode}_{mode_num}"
        df["seed"] = seed_num

        data_frames.append(df)

# Concatenate all loaded CSVs
df_all = pd.concat(data_frames, ignore_index=True)

# Keep relevant columns
df_all = df_all[["modality", "semantic_name", "mcc_logreg", "mcc_mlp", f"{mode}", "seed"]]

# Filter to include only fixed order factors
df_all = df_all[df_all["semantic_name"].isin(fixed_factor_order)]

# Compute mean and standard deviation across seeds
df_grouped = df_all.groupby(["modality", "semantic_name", f"{mode}"]).agg(
    mcc_logreg_mean=("mcc_logreg", "mean"),
    mcc_mlp_mean=("mcc_mlp", "mean"),  # Nonlinear MLP mean
    mcc_logreg_std=("mcc_logreg", "std"),
    mcc_mlp_std=("mcc_mlp", "std")  # Nonlinear MLP std
).reset_index()

# Ensure consistent ordering based on fixed order
df_grouped["semantic_name"] = pd.Categorical(df_grouped["semantic_name"], categories=fixed_factor_order, ordered=True)

# Define perturbation configurations
perturb_configs = sorted(df_grouped[f"{mode}"].unique(), key=lambda x: int(x.split("_")[-1]))

# Generate LaTeX-compatible semantic names
factor_name_map = {
    "OBJ_COLOR": r"$\mathbf{\ color}$",
    "OBJ_SHAPE": r"$\mathbf{\ shape}$",
    "OBJ_SIZE":  r"$\mathbf{\ size}$",
    "CAMERA":    r"$\mathbf{\ cam.}$",
    "BACKGROUND": r"$\mathbf{\ back.}$",
    "H_AXIS": r"$\mathbf{\ hori.}$",
    "V_AXIS": r"$\mathbf{\ vert.}$"
}
semantic_name = [factor_name_map[factor] for factor in fixed_factor_order]

# Pivot tables for visualization
df_image_linear = df_grouped[df_grouped["modality"] == "image"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_logreg_mean").reindex(index=fixed_factor_order)
df_text_linear = df_grouped[df_grouped["modality"] == "text"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_logreg_mean").reindex(index=fixed_factor_order)

df_image_nonlinear = df_grouped[df_grouped["modality"] == "image"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_mlp_mean").reindex(index=fixed_factor_order)
df_text_nonlinear = df_grouped[df_grouped["modality"] == "text"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_mlp_mean").reindex(index=fixed_factor_order)

df_image_linear_std = df_grouped[df_grouped["modality"] == "image"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_logreg_std").reindex(index=fixed_factor_order)
df_text_linear_std = df_grouped[df_grouped["modality"] == "text"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_logreg_std").reindex(index=fixed_factor_order)

df_image_nonlinear_std = df_grouped[df_grouped["modality"] == "image"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_mlp_std").reindex(index=fixed_factor_order)
df_text_nonlinear_std = df_grouped[df_grouped["modality"] == "text"].pivot(index="semantic_name", columns=f"{mode}", values="mcc_mlp_std").reindex(index=fixed_factor_order)

# Ensure correct column ordering
df_image_linear = df_image_linear.reindex(columns=perturb_configs)
df_text_linear = df_text_linear.reindex(columns=perturb_configs)
df_image_nonlinear = df_image_nonlinear.reindex(columns=perturb_configs)
df_text_nonlinear = df_text_nonlinear.reindex(columns=perturb_configs)

# Function to format heatmap labels
def format_labels(mean_df, std_df):
    formatted = mean_df.copy().astype(str)
    for i in range(formatted.shape[0]):
        for j in range(formatted.shape[1]):
            mean_val = mean_df.iloc[i, j]
            std_val = std_df.iloc[i, j] if not np.isnan(std_df.iloc[i, j]) else 0.0
            rounded_value = Decimal(mean_val).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            formatted.iloc[i, j] = f"{rounded_value}"
    return formatted

# Generate formatted labels for heatmaps
labels_image_linear = format_labels(df_image_linear, df_image_linear_std)
labels_text_linear = format_labels(df_text_linear, df_text_linear_std)
labels_image_nonlinear = format_labels(df_image_nonlinear, df_image_nonlinear_std)
labels_text_nonlinear = format_labels(df_text_nonlinear, df_text_nonlinear_std)

# Unicode circled numbers for x-axis labels
circled_numbers = ["①", "②", "③", "④", "⑤"][:len(perturb_configs)]

# Plot heatmaps
heatmap_data = [
    (df_image_linear, labels_image_linear, f"linear_image_{mode}.pdf"),
    (df_text_linear, labels_text_linear, f"linear_text_{mode}.pdf"),
    (df_image_nonlinear, labels_image_nonlinear, f"nonlinear_image_{mode}.pdf"),
    (df_text_nonlinear, labels_text_nonlinear, f"nonlinear_text_{mode}.pdf")
]

for matrix, labels, title in heatmap_data:
    plt.figure(figsize=(5, 3))

    ax = sns.heatmap(matrix, annot=labels, fmt="", cmap="BuGn", vmin=None, vmax=None, cbar=False,
                     annot_kws={"fontsize": 18})

    
    # Set x-axis labels with circled numbers
    plt.xticks(ticks=np.arange(len(perturb_configs)) + 0.5, labels=circled_numbers, fontsize=20, rotation=0)
    
    if mode == "perturbation":
        plt.gca().invert_xaxis()  # Reverse x-axis when dropping semantics

    if "image" in title:
        plt.xlabel(f"{mode} biases"+r", $\hat{\mathbf{z}}_x$", fontsize=20)
    else:
        plt.xlabel(f"{mode} biases"+r", $\hat{\mathbf{z}}_t$", fontsize=20)

    # Set y-axis labels
    if mode == "selection" and "image" in title:
        plt.ylabel("MCC", fontsize=20)
        plt.yticks(ticks=np.arange(len(semantic_name)) + 0.5, labels=semantic_name, fontsize=20, rotation=0)
        
    else:
        plt.ylabel("")
        plt.yticks([])  # Hide y-axis tick

        # Add an overall bounding box
    ax.add_patch(plt.Rectangle((0, 0), matrix.shape[1], matrix.shape[0],
                               linewidth=1, edgecolor='#137e6d', facecolor='none', clip_on=False))

    # Save figure
    plt.savefig(os.path.join(root_path, title), format="pdf", dpi=600, bbox_inches="tight")
    plt.close()
