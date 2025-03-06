import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# Define mode and setting variables
mode = "perturb"  # or "perturb"
setting = "ind"  # or "dep"

base_dir = os.path.join("models/", f"{setting}_{mode}")

# Define x-axis labels based on mode
if mode == "drop":
    x_labels = [r"${[1:10]}$", r"${[1:9]}$", r"${[1:8]}$", r"${[1:7]}$", r"${[1:6]}$",
                r"${[1:5]}$", r"${[1:4]}$", r"${[1:3]}$", r"${[1:2]}$", r"${\{1\}}$"]
else:
    x_labels = list(reversed([r"${[1:9]}$", r"${[1:8]}$", r"${[1:7]}$", r"${[1:6]}$",
                              r"${[1:5]}$", r"${[1:4]}$", r"${[1:3]}$", r"${[1:2]}$", r"${\{1\}}$", r"${\emptyset}$"]))

# Initialize lists to store results
iid_scores, ood_scores = [], []

# Loop through drop settings (0-9) and collect results from all trials (tr1-tr3)
for bias in range(10):  # Assuming 10 drop settings (0-9)
    iid_vals, ood_vals = [], []

    for trial in range(1, 4):  # Assuming trials tr1, tr2, tr3
        file_path = os.path.join(base_dir, f"{setting}_{mode}{bias}_tr{trial}", "results_classify.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            iid_vals.append(df["iid"].values[0])
            ood_vals.append(df["ood"].values[0])
    
    # Compute mean and std for each drop setting
    iid_scores.append((np.mean(iid_vals), np.std(iid_vals)))
    ood_scores.append((np.mean(ood_vals), np.std(ood_vals)))

# Extract means and standard deviations
iid_means, iid_stds = zip(*iid_scores)
ood_means, ood_stds = zip(*ood_scores)

# Plot IID and OOD MCC curves with shaded regions for standard deviation
if setting == "ind":
    plt.figure(figsize=(5.2, 4))
else:
    plt.figure(figsize=(5, 4))

# Plot IID MCC with shaded region
plt.plot(range(len(x_labels)), iid_means, '-o', label="IID", color='orange')
plt.fill_between(range(len(x_labels)), 
                 np.array(iid_means) - np.array(iid_stds), 
                 np.array(iid_means) + np.array(iid_stds), 
                 color='orange', alpha=0.2)

# Plot OOD MCC with shaded region
plt.plot(range(len(x_labels)), ood_means, '-s', label="OOD", color='red')
plt.fill_between(range(len(x_labels)), 
                 np.array(ood_means) - np.array(ood_stds), 
                 np.array(ood_means) + np.array(ood_stds), 
                 color='red', alpha=0.2)

# Formatting the plot
if mode == "drop":
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta}$", fontsize=20)
else:
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\beta}\ |\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}}$", fontsize=20)

# Reverse x-axis
plt.gca().invert_xaxis()

if setting == "ind":
    plt.ylabel(r"MCC", fontsize=20)
plt.xticks(ticks=np.arange(10), labels=x_labels, fontsize=16, rotation=45)
plt.yticks(fontsize=16)

if setting == "ind":
    plt.legend(fontsize=20)
    plt.title(r"Classification, Independent", fontsize=20)
else:
    plt.title(r"Classification, Dependent", fontsize=20)
plt.xticks(ticks=np.arange(10), labels=x_labels, fontsize=16, rotation=45)
plt.savefig(os.path.join(base_dir, "predict_class.pdf"), format="pdf", dpi=600, bbox_inches="tight")
plt.show()
plt.close()