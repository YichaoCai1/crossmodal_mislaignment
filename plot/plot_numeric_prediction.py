import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import os

import seaborn as sns

# -----------------------------------------------------------------------------
# Style setup
# -----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rc("font", family="DejaVu Sans")
plt.rcParams["axes.xmargin"] = 0

# high-contrast, colorblind-friendly palette for tasks
palette    = sns.color_palette("tab10")
inv_color  = "#ffffff"

# -----------------------------------------------------------------------------
# Load & prepare data
# -----------------------------------------------------------------------------
mode     = "drop"   # drop or "perturb"
stats    = "ind"    # dep or "ind"
root_path = f"../models/Numeric/{stats}_{mode}/"
file_paths = glob.glob(os.path.join(root_path, f"{stats}_{mode}*_tr*/results_predict.csv"))

data_frames = []
pattern     = re.compile(f"{stats}_{mode}" + r"(\d+)_tr(\d+)")

for fp in file_paths:
    m = pattern.search(fp)
    if m:
        cfg  = int(m.group(1))
        seed = int(m.group(2))
        df   = pd.read_csv(fp)
        df[mode]   = f"{mode}{cfg}"
        df["seed"] = seed
        data_frames.append(df)

if data_frames:
    df_all = pd.concat(data_frames, ignore_index=True)
    df_all["metric_scores"] = df_all["metric_scores"].clip(lower=0)

    df_grouped = (
        df_all
          .groupby(["task", mode])
          .agg(
              mean_metric=("metric_scores", "mean"),
              min_metric =("metric_scores", "min"),
              max_metric =("metric_scores", "max")
          )
          .reset_index()
    )

    mode_configs = sorted(
        df_grouped[mode].unique(),
        key=lambda x: int(x.replace(mode, ""))
    )

    # x-axis labels
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

    legend_labels = {
        "y1": r"$y_1$",
        "y2": r"$y_2$",
        "y3": r"$y_3$",
        "y4": r"$y_4$"
    }

    # choose a distinct marker for each task
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    # -----------------------------------------------------------------------------
    # Plot performance curves
    # -----------------------------------------------------------------------------
    plt.figure(figsize=(5.2, 3.5))

    # background span
    plt.axvspan(-0.5, len(mode_configs)-0.5, color=inv_color, alpha=0.5, label="_nolegend_")

    for idx, task in enumerate(df_grouped["task"].unique()):
        task_data = df_grouped[df_grouped["task"] == task]
        x_vals    = [int(s.replace(mode, "")) for s in task_data[mode]]
        color     = palette[idx % len(palette)]
        marker    = markers[idx % len(markers)]

        # mean curve with distinct marker shape
        plt.plot(
            x_vals,
            task_data["mean_metric"],
            f"-{marker}",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.1,
            markersize=7,
            label=legend_labels.get(task, task)
        )
        # shaded band
        plt.fill_between(
            x_vals,
            task_data["min_metric"],
            task_data["max_metric"],
            color=color,
            alpha=0.2
        )

    # axes labels & ticks
    xlabel = (
        r"selection bias, $\hat{\mathbf{z}}_x$"
        if mode == "drop"
        else r"perturbation bias, $\hat{\mathbf{z}}_x$"
    )
    plt.xlabel(xlabel, fontsize=20)

    if stats == "ind":
        plt.ylabel(r"$R^2$", fontsize=20)

    plt.xticks(np.arange(len(mode_configs)), x_labels, fontsize=16, rotation=0)
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=16)
    ax = plt.gca()
    ax.invert_xaxis()
    # ax.grid(True)

    # legend with bold marker borders
    plt.legend(
        fontsize=14,
        ncol=2,
        frameon=False,
        handletextpad=0.4,
        handlelength=2
    )

    plt.title(
        "Regression, Independent" if stats == "ind" else "Regression, Dependent",
        fontsize=20
    )

    # sns.despine(trim=True)

    plt.savefig(
        os.path.join(root_path, "predict_recog.pdf"),
        format="pdf",
        dpi=600,
        bbox_inches="tight"
    )
    plt.close()

else:
    print("No valid data files found. Please check the file paths and directory structure.")
