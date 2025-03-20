import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # For heatmaps

# Define the base directory where the results are stored
mode = "selections"     # perturbations, selections
base_dir = f"../models/C3DID/{mode}"

# Get a list of all folders
folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])

# Dictionary to store data for each selection setting
data = []

# Loop through folders and process results.csv
for folder in folders:
    selection = int(folder.split("_")[2])  # Extract selection number from folder name
    file_path = os.path.join(base_dir, folder, "results.csv")
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df[f"{mode}"] = selection  # Assign selection number
        df["seed"] = folder.split("_")[-1]  # Assign seed number
        data.append(df)

# Merge all data
df_all = pd.concat(data, ignore_index=True)

# Rename factor names for better readability
rename_dict = {
    "object_shape": "obj. shape", "object_xpos": "obj. x_pos", "object_ypos": "obj. y_pos", "object_zpos": "obj. z",
    "object_alpharot": "obj. alpha", "object_betarot": "obj. beta", "object_gammarot": "obj. gamma",
    "object_color": "obj. color", "spotlight_pos": "spot. pos", "spotlight_color": "spot. color",
    "background_color": "back. color", "object_color_index": "obj. color",
    "splotlight_color_index": "spot. color", "background_color_index": "back. color",
    "text_phrasing": "phrasing"
}
df_all["factor_name"] = df_all["factor_name"].replace(rename_dict)

# Assign correct metric based on factor type
df_all["metric_linear"] = df_all.apply(
    lambda row: row["mcc_logreg"] if row["factor_type"] == "discrete" else row["r2_linreg"], axis=1
)
df_all["metric_nonlinear"] = df_all.apply(
    lambda row: row["mcc_mlp"] if row["factor_type"] == "discrete" else row["r2_krreg"], axis=1
)


# Clip metric values into the range [0, 1] before aggregation
df_all["metric_linear"] = df_all["metric_linear"].clip(0, 1)
df_all["metric_nonlinear"] = df_all["metric_nonlinear"].clip(0, 1)


# Aggregate results by selection setting, modality, and factor name
df_grouped = df_all.groupby([f"{mode}", "modality", "factor_name"]).agg(
    metric_linear=("metric_linear", "mean"),
    metric_nonlinear=("metric_nonlinear", "mean"),
).reset_index()

# Define the fixed y-axis ordering for each modality
factor_order_image = [
    "obj. alpha", "obj. beta", "obj. gamma", "obj. shape", "obj. x_pos", "obj. y_pos", "obj. color", "spot. pos", "spot. color", "back. color"
]
factor_order_text = [
    "phrasing", "obj. shape", "obj. x_pos", "obj. y_pos", "spot. pos",
    "obj. color", "spot. color", "back. color"
]

# Pivot the dataframe for heatmap format (Separate Image and Text)
df_linear_image = df_grouped[df_grouped["modality"] == "image"].pivot(index="factor_name", columns=f"{mode}", values="metric_linear").reindex(factor_order_image)
df_nonlinear_image = df_grouped[df_grouped["modality"] == "image"].pivot(index="factor_name", columns=f"{mode}", values="metric_nonlinear").reindex(factor_order_image)

df_linear_text = df_grouped[df_grouped["modality"] == "text"].pivot(index="factor_name", columns=f"{mode}", values="metric_linear").reindex(factor_order_text)
df_nonlinear_text = df_grouped[df_grouped["modality"] == "text"].pivot(index="factor_name", columns=f"{mode}", values="metric_nonlinear").reindex(factor_order_text)


# Define function to plot and save heatmaps
def plot_heatmap(data, title, save_path, cmap="GnBu"):
    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap, 
                     mask=data.isnull(),  # Black out missing values
                     cbar=False,
                     annot_kws={"fontsize": 19})

    # Convert numeric x-tick labels to circled numbers (1-7)
    circled_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦"]
    xticks = data.columns.tolist()  # Get column labels (selection settings)

    # Map selection numbers to circled numbers
    xtick_labels = [circled_numbers[i] if 0 <= i < len(circled_numbers) else str(i) for i in xticks]

    ax.set_xticklabels(xtick_labels, fontsize=25)

    plt.xlabel(f"{mode} settings", fontsize=25)
    
    if mode == "selections":
        plt.ylabel(r"$R^2$ / MCC", fontsize=25)  # Make ylabel bold
        plt.yticks(rotation=0, fontsize=20, fontweight="bold")  # Make yticks bold
    else:
        plt.ylabel("")
        plt.yticks([])
        plt.gca().invert_xaxis()  
    
    # Save the figure as a PDF
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to avoid overlapping plots

# **Generate and save heatmaps for Image Modality**
plot_heatmap(df_linear_image, "Image - Linear Evaluation (MCC for Discrete, R² for Continuous)", os.path.join(base_dir, "image_linear.pdf"))
plot_heatmap(df_nonlinear_image, "Image - Nonlinear Evaluation (MCC for Discrete, R² for Continuous)", os.path.join(base_dir, "image_nonlinear.pdf"))

# **Generate and save heatmaps for Text Modality**
plot_heatmap(df_linear_text, "Text - Linear Evaluation (MCC for Discrete, R² for Continuous)", os.path.join(base_dir, "text_linear.pdf"))
plot_heatmap(df_nonlinear_text, "Text - Nonlinear Evaluation (MCC for Discrete, R² for Continuous)", os.path.join(base_dir, "text_nonlinear.pdf"))
