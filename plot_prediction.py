import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os

# Load all result_predict.csv files from the specified directory pattern
# Load all files matching the pattern
mode = "drop" 
# mode = "perturb" 

# stats = "dep"
stats = "ind"
root_path = f"models/{stats}_{mode}/"
file_paths = glob.glob(os.path.join(root_path, f"{stats}_{mode}*_tr*/results_predict.csv"))
data_frames = []

# Extract drop configuration and seed number from file names
pattern = re.compile(f"{stats}_{mode}"+r"(\d+)_tr(\d+)")  # Extracts drop number and trial number

for file_path in file_paths:
    match = pattern.search(file_path)
    if match:
        mode_num = int(match.group(1))  # Extract drop number
        seed_num = int(match.group(2))  # Extract trial number
        df = pd.read_csv(file_path)
        df[f"{mode}"] = f"{mode}{mode_num}"  # Create perturb identifier
        df["seed"] = seed_num  # Assign seed
        data_frames.append(df)

# Concatenate all data if any files were found
if data_frames:
    df_all = pd.concat(data_frames, ignore_index=True)
    
    # Ensure metric values are >= 0 before computing mean/std
    df_all["metric_scores"] = df_all["metric_scores"].clip(lower=0)

    # Group by task and drop setting, compute mean and std
    df_grouped = df_all.groupby(["task", f"{mode}"]).agg(
        mean_metric=("metric_scores", "mean"),
        std_metric=("metric_scores", "std")
    ).reset_index()

    # Define plot settings
    drop_configs = sorted(df_grouped[f"{mode}"].unique(), key=lambda x: int(x.replace(f"{mode}", "")))  # Sort drop configs
    
    if mode == "drop":
        x_labels = [r"${[1:10]}$", r"${[1:9]}$", r"${[1:8]}$",r"${[1:7]}$",r"${[1:6]}$",
                        r"${[1:5]}$",r"${[1:4]}$",r"${[1:3]}$",r"${[1:2]}$",r"${\{1\}}$"]      # drop
    else:
        x_labels = list(reversed([r"${[1:9]}$", r"${[1:8]}$",r"${[1:7]}$",r"${[1:6]}$",
                            r"${[1:5]}$",r"${[1:4]}$",r"${[1:3]}$",r"${[1:2]}$",r"${\{1\}}$",r"${\emptyset}$"]))    # perturb

    
    legend_labels = {
    "y1": r"$y_1$",
    "y2": r"$y_2$",
    "y3": r"$y_3$",
    "y4": r"$y_4$"
    }
    
    # Plot Nonlinear IID Performance
    if stats == "ind":
        plt.figure(figsize=(5.2, 4))
    else:
        plt.figure(figsize=(5, 4))
        
    markers=["o", ".", ""]
    for task in df_grouped["task"].unique():
        task_data = df_grouped[df_grouped["task"] == task]
        x_positions = [int(d.replace(f"{mode}", "")) for d in task_data[f"{mode}"]]  # Convert drop names to numerical indices
        
        # Plot mean curve
        plt.plot(x_positions, task_data["mean_metric"], label=legend_labels.get(task, task), marker="o", linestyle="-")
        
        # Fill standard deviation as shadow
        plt.fill_between(x_positions,
                        task_data["mean_metric"] - task_data["std_metric"],
                        task_data["mean_metric"] + task_data["std_metric"],
                        alpha=0.2)

    # Formatting the plot
    if mode == "drop":
        plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta}$", fontsize=20)
    else:
        plt.xlabel(r"$\mathrm{\mathbb{I}}_{\beta}\ |\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}}$", fontsize=20)
    
    if stats == "ind":
        plt.ylabel(r"$R^2$", fontsize=20)
    plt.xticks(ticks=np.arange(10), labels=x_labels, fontsize=16, rotation=45)
    plt.yticks(fontsize=16)
    
    # Reverse x-axis
    plt.gca().invert_xaxis()

    
    if stats == "ind":
        plt.legend(fontsize=20)
        plt.title(r"Regression, Independent", fontsize=20)
    else:
        plt.title(r"Regression, Dependent", fontsize=20)
    plt.savefig(os.path.join(root_path, "predict_recog.pdf"), format="pdf", dpi=600, bbox_inches="tight")
    plt.show()
    plt.close()

else:
    print("No valid data files found. Please check the file paths and directory structure.")

