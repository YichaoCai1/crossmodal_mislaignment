import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import os

# Set font to DejaVu Sans
plt.rcParams["font.family"] = "DejaVu Sans"

# Load all files matching the pattern
root_path = "models/2_dep_drop/"
file_paths = glob.glob(os.path.join(root_path, "dep_drop*_tr*/results.csv"))  # Adjust with actual path
data_frames = []

# Extract drop configuration and seed number from file names
pattern = re.compile(r"dep_drop(\d+)_tr(\d+)")  # Extracts drop number and trial number

for file_path in file_paths:
    match = pattern.search(file_path)
    if match:
        drop_num = int(match.group(1))  # Extracts drop number
        seed_num = int(match.group(2))  # Extracts trial number
        df = pd.read_csv(file_path)
        df["drop"] = f"drop{drop_num}"  # Create drop identifier
        df["seed"] = seed_num  # Add seed number
        data_frames.append(df)

# Concatenate all data
df_all = pd.concat(data_frames, ignore_index=True)

# Extract relevant columns
df_all = df_all[["encoding", "predicted_factors", "r2_linear", "r2_nonlinear", "drop", "seed"]]

# Modify the filter to include only specific predicted factors
included_factors = [f"s_{i}" for i in range(10)] + ["m_x", "m_t"]
df_all = df_all[df_all["predicted_factors"].isin(included_factors)]

# Pivot to compute mean and std across seeds
df_pivot = df_all.pivot_table(index=["encoding", "predicted_factors", "drop"], 
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

# Define drop configurations and predicted factors
drop_configs = [f"drop{i}" for i in range(10)]
# Unicode subscripts
subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
predicted_factors = [r"$\mathbf{m}_x$", r"$\mathbf{m}_t$"] + [f"s{i}".translate(subscript_map) for i in range(10)]

# Separate data for hz_x and hz_t
df_hz_x_linear = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns="drop", values="r2_linear_mean")
df_hz_x_nonlinear = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns="drop", values="r2_nonlinear_mean")
df_hz_t_linear = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns="drop", values="r2_linear_mean")
df_hz_t_nonlinear = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns="drop", values="r2_nonlinear_mean")

df_hz_x_linear_std = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns="drop", values="r2_linear_std")
df_hz_x_nonlinear_std = df_pivot[df_pivot["encoding"] == "hz_x"].pivot(index="predicted_factors", columns="drop", values="r2_nonlinear_std")
df_hz_t_linear_std = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns="drop", values="r2_linear_std")
df_hz_t_nonlinear_std = df_pivot[df_pivot["encoding"] == "hz_t"].pivot(index="predicted_factors", columns="drop", values="r2_nonlinear_std")

# Ensure column ordering
df_hz_x_linear = df_hz_x_linear[drop_configs]
df_hz_x_nonlinear = df_hz_x_nonlinear[drop_configs]
df_hz_t_linear = df_hz_t_linear[drop_configs]
df_hz_t_nonlinear = df_hz_t_nonlinear[drop_configs]

df_hz_x_linear_std = df_hz_x_linear_std[drop_configs]
df_hz_x_nonlinear_std = df_hz_x_nonlinear_std[drop_configs]
df_hz_t_linear_std = df_hz_t_linear_std[drop_configs]
df_hz_t_nonlinear_std = df_hz_t_nonlinear_std[drop_configs]


# Format labels with mean and std (std in smaller font)
def format_labels(mean_df, std_df):
    formatted = mean_df.copy().astype(str)
    for i in range(formatted.shape[0]):
        for j in range(formatted.shape[1]):
            mean_val = mean_df.iloc[i, j]
            std_val = std_df.iloc[i, j]
            formatted.iloc[i, j] = f"${{{mean_val:.2f}}}$\n$_{{{std_val:.2f}}}$"
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
    (df_hz_t_linear, labels_hz_t_linear, "linear_text_features.pdf"),
    (df_hz_t_nonlinear, labels_hz_t_nonlinear, "nonlinear_text_feature.pdf")
]

for matrix, labels, title in heatmap_data:
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=labels, fmt="", cmap="BuGn", vmin=0, vmax=1, cbar=True,
                annot_kws={"fontsize": 12})  # Adjust font size of annotations
    plt.xlabel("dropation Settings", fontsize=16)
    plt.ylabel("Predicted Factors (R²)", fontsize=16)
    plt.xticks(ticks=np.arange(10) + 0.5, labels=["①","②","③","④","⑤","⑥","⑦","⑧","⑨", "⑩"], fontsize=16)
    plt.yticks(ticks=np.arange(len(predicted_factors)) + 0.5, labels=predicted_factors, fontsize=16, rotation=0)

    # Save figure
    plt.savefig(os.path.join(root_path, title), format="pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()  # Close the figure to prevent overlapping plots
