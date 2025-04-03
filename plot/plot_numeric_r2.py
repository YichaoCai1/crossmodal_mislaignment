import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.colors as mcolors

# Set font to DejaVu Sans
# plt.rcParams["font.family"] = "DejaVu Sans"

# Load all files matching the pattern
mode = "drop"   # drop, perturb
stats = "dep"   # ind, dep

root_path = f"../models/Numeric/{stats}_{mode}/"
file_paths = glob.glob(os.path.join(root_path, f"{stats}_{mode}*_tr*/results.csv"))  # Adjust with actual path
data_frames = []

# Extract perturb configuration and seed number from file names
pattern = re.compile(f"{stats}_{mode}"+r"(\d+)_tr(\d+)")  # Extracts perturb number and trial number

for file_path in file_paths:
    match = pattern.search(file_path)
    if match:
        mode_num = int(match.group(1))  # Extracts perturb number
        seed_num = int(match.group(2))  # Extracts trial number
        df = pd.read_csv(file_path)
        df[f"{mode}"] = f"{mode}{mode_num}"  # Create perturb identifier
        df["seed"] = seed_num  # Add seed number
        data_frames.append(df)

# Concatenate all data
df_all = pd.concat(data_frames, ignore_index=True)

# Extract relevant columns
df_all = df_all[["encoding", "predicted_factors", "r2_linear", "r2_nonlinear", f"{mode}", "seed"]]

# Modify the filter to include only specific predicted factors
included_factors = [f"s_{i}" for i in range(10)] + ["m_x", "m_t"]
df_all = df_all[df_all["predicted_factors"].isin(included_factors)]

# Pivot to compute mean and std across seeds
df_pivot = df_all.pivot_table(index=["encoding", "predicted_factors", f"{mode}"], 
                              columns="seed", values=["r2_linear", "r2_nonlinear"])

# Compute mean and standard deviation across seeds
df_pivot["r2_linear_mean"] = df_pivot["r2_linear"].mean(axis=1)
df_pivot["r2_nonlinear_mean"] = df_pivot["r2_nonlinear"].mean(axis=1)
df_pivot["r2_linear_std"] = df_pivot["r2_linear"].std(axis=1)
df_pivot["r2_nonlinear_std"] = df_pivot["r2_nonlinear"].std(axis=1)

# Clamp negative values to 0
df_pivot["r2_linear_mean"] = df_pivot["r2_linear_mean"].clip(lower=0)
df_pivot["r2_nonlinear_mean"] = df_pivot["r2_nonlinear_mean"].clip(lower=0)

# Reset index for processing
df_pivot = df_pivot.reset_index()

# Define perturb configurations and predicted factors
perturb_configs = [f"{mode}{i}" for i in range(10)]
# Unicode subscripts
subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
predicted_factors = [r"$\mathbf{m}_x$", r"$\mathbf{m}_t$"] + [f"s{i}".translate(subscript_map) for i in range(1,11)]

# Separate data for hz_x and hz_t
df_hz_x_linear = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_linear_mean")
df_hz_x_nonlinear = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_nonlinear_mean")
df_hz_t_linear = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_linear_mean")
df_hz_t_nonlinear = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_nonlinear_mean")

df_hz_x_linear_std = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_linear_std")
df_hz_x_nonlinear_std = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_nonlinear_std")
df_hz_t_linear_std = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_linear_std")
df_hz_t_nonlinear_std = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns=f"{mode}", values="r2_nonlinear_std")

# Ensure column ordering
df_hz_x_linear = df_hz_x_linear[perturb_configs]
df_hz_x_nonlinear = df_hz_x_nonlinear[perturb_configs]
df_hz_t_linear = df_hz_t_linear[perturb_configs]
df_hz_t_nonlinear = df_hz_t_nonlinear[perturb_configs]

df_hz_x_linear_std = df_hz_x_linear_std[perturb_configs]
df_hz_x_nonlinear_std = df_hz_x_nonlinear_std[perturb_configs]
df_hz_t_linear_std = df_hz_t_linear_std[perturb_configs]
df_hz_t_nonlinear_std = df_hz_t_nonlinear_std[perturb_configs]


# Format labels with mean and std (std in smaller font)
def format_labels(mean_df, std_df):
    formatted = mean_df.copy().astype(str)
    for i in range(formatted.shape[0]):
        for j in range(formatted.shape[1]):
            mean_val = mean_df.iloc[i, j]
            # std_val = std_df.iloc[i, j]
            # formatted.iloc[i, j] = f"${{{mean_val:.2f}}}$\n$_{{{std_val:.2f}}}$"
            rounded_value = Decimal(mean_val).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            formatted.iloc[i, j] = f"{rounded_value}"
    return formatted

# Generate formatted text labels
labels_hz_x_linear = format_labels(df_hz_x_linear, df_hz_x_linear_std)
labels_hz_x_nonlinear = format_labels(df_hz_x_nonlinear, df_hz_x_nonlinear_std)
labels_hz_t_linear = format_labels(df_hz_t_linear, df_hz_t_linear_std)
labels_hz_t_nonlinear = format_labels(df_hz_t_nonlinear, df_hz_t_nonlinear_std)

# Plot heatmaps
# fig, axes = plt.subplots(2, 2, figsize=(18, 14))

heatmap_data = [
    (df_hz_x_linear, labels_hz_x_linear, "linear_image_feature.pdf"),
    (df_hz_x_nonlinear, labels_hz_x_nonlinear, "nonlinear_image_feature.pdf"),
    (df_hz_t_linear, labels_hz_t_linear, "linear_text_feature.pdf"),
    (df_hz_t_nonlinear, labels_hz_t_nonlinear, "nonlinear_text_feature.pdf")
]


for matrix, labels, title in heatmap_data:
    if "image" in title and mode == "drop":
        plt.figure(figsize=(6.5, 5))
    else:
        plt.figure(figsize=(6.5, 5))

    ax = sns.heatmap(matrix, annot=labels, fmt="", cmap=sns.color_palette("GnBu", as_cmap=True), cbar=False, norm=mcolors.Normalize(vmin=0, vmax=1),
                     annot_kws={"fontsize": 17})  # Adjust font size of annotations

    
    # Set axis labels
    if mode == "perturb":
        # if stats == "dep" or "enlarged" in title or "nonlinear" not in title:
        if "image" in title:
            plt.xlabel(r"$\mathrm{\mathbb{I}}_{\rho},\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}},\ \hat{\mathbf{z}}_x$", fontsize=23)
        else:
            plt.xlabel(r"$\mathrm{\mathbb{I}}_{\rho},\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}},\ \hat{\mathbf{z}}_t$", fontsize=23)
        
        plt.xticks(ticks=np.arange(10) + 0.5, labels=reversed([r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$",
                                                            r"${[5]}$", r"${[4]}$", r"${[3]}$", r"${[2]}$",
                                                            r"${\{1\}}$", r"${\emptyset}$"]), fontsize=20, rotation=0)
        # else:
        #     plt.xlabel("")
        #     plt.xticks(ticks=[])
            
    else:
        # if stats == "dep" or "enlarged" in title or "nonlinear" not in title:
        if "image" in title:
            plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta},\ \ \ \ \mathrm{\mathbb{I}}_{\rho}=\emptyset,\ \hat{\mathbf{z}}_x$", fontsize=23)
        else:
            plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta},\ \ \ \ \mathrm{\mathbb{I}}_{\rho}=\emptyset,\ \hat{\mathbf{z}}_t$", fontsize=23)
        plt.xticks(ticks=np.arange(10) + 0.5, labels=[r"${[10]}$", r"${[9]}$", r"${[8]}$", r"${[7]}$",
                                            r"${[6]}$", r"${[5]}$", r"${[4]}$", r"${[3]}$",
                                            r"${[2]}$", r"${\{1\}}$"], fontsize=20, rotation=0)
        # else:
        #     plt.xlabel("")
        #     plt.xticks(ticks=[])

    plt.gca().invert_xaxis()  # Reverse x-axis when dropping semantics

    if ("image" in title and stats == "ind"):
        plt.ylabel(r"$R²$", fontsize=23)
        plt.yticks(ticks=np.arange(len(predicted_factors)) + 0.5, labels=predicted_factors, fontsize=23, rotation=0)
    else:
        plt.ylabel("")
        plt.yticks(ticks=[])

    # # Add an overall bounding box
    # ax.add_patch(plt.Rectangle((0, 0), matrix.shape[1], matrix.shape[0],
    #                            linewidth=1, edgecolor='#137e6d', facecolor='none', clip_on=False))

    # Save figure
    # plt.show()
    plt.savefig(os.path.join(root_path, title), format="pdf", dpi=600, bbox_inches="tight")
    plt.close()  # Close the figure to prevent overlapping plots
