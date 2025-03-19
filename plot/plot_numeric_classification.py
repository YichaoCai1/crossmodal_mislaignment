import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define mode and setting variables
mode = "drop"  # or "perturb", drop
setting = "dep"  # or "ind", dep

base_dir = os.path.join("../models/Numeric/", f"{setting}_{mode}")

# Define x-axis labels based on mode
if mode == "drop":
    x_labels = [r"${[10]}$", r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$",
                r"${[5]}$", r"${[4]}$", r"${[3]}$", r"${[2]}$", r"${\{1\}}$"]
else:
    x_labels = list(reversed([r"${[9]}$", r"${[8]}$", r"${[7]}$", r"${[6]}$",
                                r"${[5]}$", r"${[4]}$", r"${[3]}$", r"${[2]}$", r"${\{1\}}$", r"${\emptyset}$"]))

# Initialize lists to store results
id_means, id_mins, id_maxs = [], [], []
ood_means, ood_mins, ood_maxs = [], [], []

# Loop through drop settings (0-9) and collect results from all trials (tr1-tr3)
for bias in range(10):  # Assuming 10 drop settings (0-9)
    id_vals, ood_vals = [], []

    for trial in range(1, 4):  # Assuming trials tr1, tr2, tr3
        file_path = os.path.join(base_dir, f"{setting}_{mode}{bias}_tr{trial}", "results_classify.csv")

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            id_vals.append(df["iid"].values[0])
            ood_vals.append(df["ood"].values[0])
    
    # Compute mean, min, and max for each drop setting
    id_means.append(np.mean(id_vals) if id_vals else np.nan)
    id_mins.append(np.min(id_vals) if id_vals else np.nan)
    id_maxs.append(np.max(id_vals) if id_vals else np.nan)

    ood_means.append(np.mean(ood_vals) if ood_vals else np.nan)
    ood_mins.append(np.min(ood_vals) if ood_vals else np.nan)
    ood_maxs.append(np.max(ood_vals) if ood_vals else np.nan)

# Plot ID and OOD MCC curves with shaded regions for min-max bounds
plt.figure(figsize=(5.2, 3.5)) # if setting == "ind" else (5, 4))

# Plot ID MCC with shaded region (bounded by min-max)
plt.plot(range(len(x_labels)), id_means, '-o', label="ID", color='orange')
plt.fill_between(range(len(x_labels)), id_mins, id_maxs, color='orange', alpha=0.2)

# Plot OOD MCC with shaded region (bounded by min-max)
plt.plot(range(len(x_labels)), ood_means, '-s', label="OOD", color='red')
plt.fill_between(range(len(x_labels)), ood_mins, ood_maxs, color='red', alpha=0.2)

# Formatting the plot
if mode == "drop":
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\theta},\ \mathrm{\mathbb{I}}_{\rho}=\emptyset$", fontsize=20)
else:
    plt.xlabel(r"$\mathrm{\mathbb{I}}_{\rho},\ \mathrm{\mathbb{I}}_{\theta}=\mathrm{\mathbb{I}}_{\mathbf{s}}$", fontsize=20)

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

plt.savefig(os.path.join(base_dir, "predict_class.pdf"), format="pdf", dpi=600, bbox_inches="tight")
# plt.show()
plt.close()
