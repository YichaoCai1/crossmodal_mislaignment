import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# Load all result_predict.csv files from the specified directory pattern
mode = "perturb"  # or "perturb", drop
stats = "dep"  # or "ind", dep
root_path = f"../models/Numeric/{stats}_{mode}/"
file_paths = glob.glob(os.path.join(root_path, f"{stats}_{mode}*_tr*/results_predict.csv"))
data_frames = []

# Extract configuration and seed number from file names
pattern = re.compile(f"{stats}_{mode}" + r"(\d+)_tr(\d+)")  # Extracts mode number and trial number

for file_path in file_paths:
    match = pattern.search(file_path)
    if match:
        mode_num = int(match.group(1))  # Extract mode number
        seed_num = int(match.group(2))  # Extract trial number
        df = pd.read_csv(file_path)
        df[mode] = f"{mode}{mode_num}"  # Create mode identifier
        df["seed"] = seed_num  # Assign seed
        data_frames.append(df)

# Concatenate all data if any files were found
if data_frames:
    df_all = pd.concat(data_frames, ignore_index=True)
    
    # Ensure metric values are >= 0 before computing statistics
    df_all["metric_scores"] = df_all["metric_scores"].clip(lower=0)

    # Group by task and mode setting, compute mean, min, and max
    df_grouped = df_all.groupby(["task", mode]).agg(
        mean_metric=("metric_scores", "mean"),
        min_metric=("metric_scores", "min"),
        max_metric=("metric_scores", "max")
    ).reset_index()

    # Define plot settings
    mode_configs = sorted(df_grouped[mode].unique(), key=lambda x: int(x.replace(mode, "")))  # Sort mode configs
    
    if mode == "drop":
        x_labels = [r"${[10]}$", r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$",
                    r"${[5]}$", r"${[4]}$", r"${[3]}$", r"${[2]}$", r"${\{1\}}$"]
    else:
        x_labels = list(reversed([r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$",
                                  r"${[5]}$", r"${[4]}$", r"${[3]}$", r"${[2]}$", r"${\{1\}}$", r"${\emptyset}$"]))

    legend_labels = {
        "y1": r"$y_1$",
        "y2": r"$y_2$",
        "y3": r"$y_3$",
        "y4": r"$y_4$"
    }
    
    # Plot Performance
    plt.figure(figsize=(5.2, 3.5)) # if stats == "ind" else (5, 4))
    
    # Define the x positions where vertical lines should be drawn
    x_positions = [7, 5, 3, 1]
    
    task_colors = {}  # Dictionary to store assigned colors per task

    for task_idx, task in enumerate(df_grouped["task"].unique()):
        task_data = df_grouped[df_grouped["task"] == task]
        x_values = [int(d.replace(mode, "")) for d in task_data[mode]]  # Convert mode names to numerical indices
        
        # Assign unique color to each task
        color = f"C{task_idx}"  # Matplotlib color cycle
        task_colors[task] = color

        # Plot mean curve
        plt.plot(x_values, task_data["mean_metric"], label=legend_labels.get(task, task), marker="o", linestyle="-", color=color)
        
        # Fill area between min and max values
        plt.fill_between(x_values,
                         task_data["min_metric"],
                         task_data["max_metric"],
                         alpha=0.2,
                         color=color)

    # Draw vertical dashed lines only if mode == "drop"
    # if mode == "drop":
    #     for idx, x in enumerate(x_positions):
    #         if idx < len(df_grouped["task"].unique()):  # Ensure index is within available tasks
    #             task = list(df_grouped["task"].unique())[idx]
    #             color = task_colors.get(task, "black")  # Get assigned color, default to black
    #             plt.axvline(x, color=color, linestyle="dashed", linewidth=1, alpha=0.3)

    # Formatting the plot
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta},\ \mathrm{\mathbb{I}}_{\rho}=\emptyset$" if mode == "drop" else r"$\mathrm{\mathbb{I}}_{\rho},\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}}$", fontsize=20)
    if stats == "ind":
        plt.ylabel(r"$R^2$", fontsize=20)
    plt.xticks(ticks=np.arange(10), labels=x_labels, fontsize=16, rotation=0)
    plt.yticks(fontsize=16)
    
    # Reverse x-axis
    plt.gca().invert_xaxis()
    plt.legend(fontsize=15, ncol=2, loc="best", handletextpad=0.4, columnspacing=0.5)

    if stats == "ind":
        plt.title(r"Regression, Independent", fontsize=20)
    else:
        plt.title(r"Regression, Dependent", fontsize=20)
    plt.savefig(os.path.join(root_path, "predict_recog.pdf"), format="pdf", dpi=600, bbox_inches="tight")
    # plt.show()
    plt.close()

else:
    print("No valid data files found. Please check the file paths and directory structure.")
