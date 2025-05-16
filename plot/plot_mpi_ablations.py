import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns

# -----------------------------------------------------------------------------
# Modern style setup (no data changes)
# -----------------------------------------------------------------------------
sns.set_style("ticks")
plt.rc("font", family="DejaVu Sans")

# high-contrast, colorblind-friendly palette for up to 7 factors
palette = sns.color_palette("tab10", n_colors=7)

# -----------------------------------------------------------------------------
# Define file paths & factor ordering
# -----------------------------------------------------------------------------
tt = "../models/MPI3d/ablative/mpi3d_selection_0_enc{i}_tr{j}/results_enc{i}_tr{j}.csv"
file_paths = {
    1: [tt.format(i=1, j=1), tt.format(i=1, j=2), tt.format(i=1, j=3)],
    2: [tt.format(i=2, j=1), tt.format(i=2, j=2), tt.format(i=2, j=3)],
    3: [tt.format(i=3, j=1), tt.format(i=3, j=2), tt.format(i=3, j=3)],
    4: [tt.format(i=4, j=1), tt.format(i=4, j=2), tt.format(i=4, j=3)],
}
factor_name_map = OrderedDict([
    ("OBJ_COLOR",  r"$\mathbf{\ color}$"),
    ("OBJ_SHAPE",  r"${\ shape}$"),
    ("OBJ_SIZE",   r"${\ size}$"),
    ("CAMERA",     r"${\ cam.}$"),
    ("BACKGROUND", r"${\ back.}$"),
    ("H_AXIS",     r"${\ hori.}$"),
    ("V_AXIS",     r"${\ vert.}$")
])

# -----------------------------------------------------------------------------
# Load & aggregate results
# -----------------------------------------------------------------------------
aggregated_results = {}
for enc_size, files in file_paths.items():
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    mean_res = combined.groupby("semantic_name").mean(numeric_only=True)
    std_res  = combined.groupby("semantic_name").std(numeric_only=True)
    aggregated_results[enc_size] = (mean_res, std_res)

# -----------------------------------------------------------------------------
# Prepare figure
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(22, 6), gridspec_kw={"wspace": 0.3})
titles  = ["Linear Model (LogReg)", "Nonlinear Model (MLP)"]
metrics = ["mcc_logreg", "mcc_mlp"]

for ax, title, metric in zip(axes, titles, metrics):
    for idx, (factor, label) in enumerate(factor_name_map.items()):
        x_vals, y_vals, y_errs = [], [], []
        for enc_size in sorted(aggregated_results.keys()):
            mean_res, std_res = aggregated_results[enc_size]
            if factor in mean_res.index:
                x_vals.append(enc_size)
                y_vals.append(mean_res.loc[factor, metric])
                y_errs.append(std_res.loc[factor, metric])
        if not x_vals:
            continue

        ax.errorbar(
            x_vals, y_vals, yerr=y_errs,
            fmt="-o",
            color=palette[idx],
            markerfacecolor=palette[idx],
            markeredgecolor="k",
            markeredgewidth=2.0,
            markersize=10,
            capsize=5,
            linewidth=3,
            label=label
        )

    ax.set_title(title, fontsize=23, pad=12)
    ax.set_xlabel("Encoding Size", fontsize=20)
    if metric == "mcc_logreg":
        ax.set_ylabel("MCC", fontsize=20)
    ax.set_xticks(sorted(aggregated_results.keys()))
    ax.tick_params(axis="both", labelsize=18, width=2.0)
    ax.grid(True, linestyle="--", linewidth=1.2)
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.5)
    # sns.despine(ax=ax, trim=False)

# -----------------------------------------------------------------------------
# Legend & save
# -----------------------------------------------------------------------------
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(
    handles, labels,
    fontsize=18,
    ncol=2,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0
)
axes[1].legend(
    handles, labels,
    fontsize=18,
    ncol=2,
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0
)

plt.tight_layout()
plt.savefig(
    "../models/MPI3d/mpi_ablation_encodingsize.pdf",
    dpi=300, bbox_inches="tight"
)
plt.show()