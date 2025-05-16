import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import seaborn as sns
from matplotlib.patches import Patch

# -----------------------------------------------------------------------------
# Style setup
# -----------------------------------------------------------------------------
# sns.set_style("ticks")
plt.style.use('fast')
plt.rc("font", family="DejaVu Sans")
plt.rcParams["axes.xmargin"] = 0

# high-contrast, colorblind-friendly palette
palette     = sns.color_palette("tab10")
id_color    = palette[0]    # deep blue
ood_color   = palette[1]    # orange
inv_color   = "#ffffff"     # light gray for shift span
shift_color = "#e0e0e0"     # darker gray for inv. span

# -----------------------------------------------------------------------------
# Define mode and directory
# -----------------------------------------------------------------------------
mode    = "perturb"    # drop or "perturb"
setting = "ind"     # dep or "ind"
base_dir = os.path.join("../models/Numeric/", f"{setting}_{mode}")

# -----------------------------------------------------------------------------
# Build x-axis labels
# -----------------------------------------------------------------------------
if mode == "drop":
    x_labels = [
        r"${[10]}$", r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$",
        r"${[5]}$",  r"${[4]}$", r"${[3]}$", r"${[2]}$", r"${\{1\}}$"
    ]
else:
    x_labels = list(reversed([
        r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$", r"${[5]}$",
        r"${[4]}$", r"${[3]}$", r"${[2]}$", r"${\{1\}}$", r"${\emptyset}$"
    ]))

# -----------------------------------------------------------------------------
# Collect ID / OOD metrics across biases & trials
# -----------------------------------------------------------------------------
id_means, id_mins, id_maxs = [], [], []
ood_means, ood_mins, ood_maxs = [], [], []

for bias in range(10):
    id_vals, ood_vals = [], []
    for trial in range(1, 4):
        fp = os.path.join(base_dir, f"{setting}_{mode}{bias}_tr{trial}", "results_classify.csv")
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            id_vals.append(df["iid"].iat[0])
            ood_vals.append(df["ood"].iat[0])
    if id_vals:
        id_means.append(np.mean(id_vals))
        id_mins .append(np.min(id_vals))
        id_maxs .append(np.max(id_vals))
    else:
        id_means.append(np.nan); id_mins.append(np.nan); id_maxs.append(np.nan)
    if ood_vals:
        ood_means.append(np.mean(ood_vals))
        ood_mins .append(np.min(ood_vals))
        ood_maxs .append(np.max(ood_vals))
    else:
        ood_means.append(np.nan); ood_mins.append(np.nan); ood_maxs.append(np.nan)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
plt.figure(figsize=(5.2, 3.5))

# Draw background spans (no legend entries)
if mode == "drop":
    plt.axvspan(1.5, 9.5,  color=inv_color,   alpha=0.5, label="_nolegend_")
    plt.axvspan(-0.5, 1.5, color=shift_color, alpha=0.5, label="_nolegend_")
    plt.axvline(4.5, color=palette[2], linestyle="--", alpha=0.6)
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta},\ \mathrm{\mathbb{I}}_{\rho}=\emptyset$", fontsize=20)
else:
    plt.axvspan(-0.5, 7.5, color=inv_color,   alpha=0.5, label="_nolegend_")
    plt.axvspan(7.5, 9.5,  color=shift_color, alpha=0.5, label="_nolegend_")
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\rho},\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}}$", fontsize=20)

# Plot ID line & fill
ln1, = plt.plot(
    range(10), id_means, "-o",
    color=id_color,
    markerfacecolor=id_color,
    markeredgecolor=id_color,
    markeredgewidth=0.1,
    markersize=7,
    label="ID"
)
plt.fill_between(range(10), id_mins, id_maxs, color=id_color, alpha=0.2)

# Plot OOD line & fill
ln2, = plt.plot(
    range(10), ood_means, "--s",
    color=ood_color,
    markerfacecolor=ood_color,
    markeredgecolor=ood_color,
    markeredgewidth=0.1,
    markersize=7,
    label="OOD"
)
plt.fill_between(range(10), ood_mins, ood_maxs, color=ood_color, alpha=0.2)

# Axes formatting
plt.gca().invert_xaxis()
plt.xticks(np.arange(10), x_labels, fontsize=16)
plt.yticks(np.arange(0, 1.01, 0.2), fontsize=16)
if setting == "ind":
    plt.ylabel("MCC", fontsize=20)
    
plt.title(
    "Classification, Independent" if setting == "ind" else "Classification, Dependent",
    fontsize=20
)

# Custom legend with high-contrast boundaries
patch_shift = Patch(facecolor=shift_color, edgecolor="k", alpha=0.5, label="shift")
patch_inv   = Patch(facecolor=inv_color,   edgecolor="k", alpha=0.5, label="inv.")
plt.legend(
    handles=[ln1, ln2, patch_shift, patch_inv],
    fontsize=14,
    ncol=2,
    frameon=False,
    handletextpad=0.3,
    handlelength=2
)

# sns.despine(trim=True, left=False, bottom=False)  # keep top/right

# Save and close
plt.savefig(
    os.path.join(base_dir, "predict_class.pdf"),
    format="pdf",
    dpi=600,
    bbox_inches="tight"
)
plt.close()
