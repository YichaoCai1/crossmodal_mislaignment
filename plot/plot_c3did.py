import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmaps
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------------------
# PDF-inspired custom colormap (5 stops from light mint → deep teal)
# -----------------------------------------------------------------------------
pdf_colors = ["#f7fcf5", "#c7e9c0", "#66c2a5", "#137e6d", "#004c6d"]
pdf_cmap    = LinearSegmentedColormap.from_list("pdf_theme", pdf_colors)

# -----------------------------------------------------------------------------
# Define the base directory where the results are stored
# -----------------------------------------------------------------------------
mode     = "selections"     # perturbations, selections
base_dir = f"../models/C3DID/{mode}"

# Get a list of all folders
folders = sorted(f for f in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, f)))

# Dictionary to store data for each selection setting
data = []

# Loop through folders and process results.csv
for folder in folders:
    selection = int(folder.split("_")[2])  # Extract selection number
    file_path = os.path.join(base_dir, folder, "results.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df[f"{mode}"] = selection     # Assign selection number
        df["seed"]      = folder.split("_")[-1]
        data.append(df)

# Merge all data
df_all = pd.concat(data, ignore_index=True)

# Rename factor names for better readability
rename_dict = {
    "object_shape": "shape", "object_xpos": "x_pos", "object_ypos": "y_pos", "object_zpos": "obj. z",
    "object_alpharot": "alpha", "object_betarot": "beta", "object_gammarot": "gamma",
    "object_color": "color", "spotlight_pos": "s_pos", "spotlight_color": "s_color",
    "background_color": "b_color", "object_color_index": "color",
    "splotlight_color_index": "s_color", "background_color_index": "b_color",
    "text_phrasing": "phrase"
}
df_all["factor_name"] = df_all["factor_name"].replace(rename_dict)

# Assign correct metric based on factor type
df_all["metric_linear"] = df_all.apply(
    lambda r: r["mcc_logreg"] if r["factor_type"] == "discrete" else r["r2_linreg"],
    axis=1
)
df_all["metric_nonlinear"] = df_all.apply(
    lambda r: r["mcc_mlp"]    if r["factor_type"] == "discrete" else r["r2_krreg"],
    axis=1
)

# Clip to [0, 1]
df_all["metric_linear"]    = df_all["metric_linear"].clip(0, 1)
df_all["metric_nonlinear"] = df_all["metric_nonlinear"].clip(0, 1)

# Aggregate by selection, modality, factor
df_grouped = (
    df_all
      .groupby([mode, "modality", "factor_name"])
      .agg(metric_linear    = ("metric_linear",    "mean"),
           metric_nonlinear = ("metric_nonlinear", "mean"))
      .reset_index()
)

# Fixed y-ordering
factor_order_image = ["alpha","beta","gamma","shape","x_pos","y_pos","s_pos","color","s_color","b_color"]
factor_order_text  = ["phrase","shape","x_pos","y_pos","s_pos","color","s_color","b_color"]

# Pivot into heatmap-ready tables
df_linear_image    = (df_grouped[df_grouped["modality"]=="image"]
                      .pivot(index="factor_name", columns=mode, values="metric_linear")
                      .reindex(factor_order_image))

df_nonlinear_image = (df_grouped[df_grouped["modality"]=="image"]
                      .pivot(index="factor_name", columns=mode, values="metric_nonlinear")
                      .reindex(factor_order_image))

df_linear_text     = (df_grouped[df_grouped["modality"]=="text"]
                      .pivot(index="factor_name", columns=mode, values="metric_linear")
                      .reindex(factor_order_text))

df_nonlinear_text  = (df_grouped[df_grouped["modality"]=="text"]
                      .pivot(index="factor_name", columns=mode, values="metric_nonlinear")
                      .reindex(factor_order_text))

# -----------------------------------------------------------------------------
# Plotting function (default cmap is now pdf_cmap)
# -----------------------------------------------------------------------------
def plot_heatmap(data, title, save_path, cmap="Blues"):
    plt.figure(figsize=(10, 5))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        mask=data.isnull(),  # gray-out missing
        cbar=True,
        annot_kws={"fontsize": 20}
    )

    # Tweak colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    # Convert x-ticks to circled numbers
    circled = ["①","②","③","④","⑤","⑥","⑦"]
    xtks     = data.columns.tolist()
    ax.set_xticklabels(
        [circled[i] if 0 <= i < len(circled) else str(i) for i in xtks],
        fontsize=25
    )

    plt.xlabel(f"{mode} settings", fontsize=25)
    if mode == "selections":
        plt.ylabel(r"$R^2$ / MCC", fontsize=25)
    else:
        plt.ylabel("")
        plt.gca().invert_xaxis()

    plt.yticks(rotation=0, fontsize=20, fontweight="bold")
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------------------------
# Generate & save
# -----------------------------------------------------------------------------
plot_heatmap(df_linear_image,    "Image - Linear",    os.path.join(base_dir, "image_linear.pdf"))
plot_heatmap(df_nonlinear_image, "Image - Nonlinear", os.path.join(base_dir, "image_nonlinear.pdf"))
plot_heatmap(df_linear_text,     "Text - Linear",     os.path.join(base_dir, "text_linear.pdf"))
plot_heatmap(df_nonlinear_text,  "Text - Nonlinear",  os.path.join(base_dir, "text_nonlinear.pdf"))
