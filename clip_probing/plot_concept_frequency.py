import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# === Load CSV and normalize ===
csv_path = "clip_probing/concept_frequencies.csv"
df = pd.read_csv(csv_path)
df['Concept'] = df['Concept'].str.strip().str.lower()
percentage_map = dict(zip(df['Concept'], df['Percentage']))

# === Your concept dictionary ===
concept_dict = {
    'common concepts': {
        'Animal': ['dog', 'cat', 'horse', 'bird', 'elephant', 'giraffe', 'cow', 'zebra', 'rabbit', 'duck'],
        'Clothing': ['shirt', 'pants', 'dress', 'shoes', 'hat', 'jacket', 'skirt', 'tie', 'hoodie', 'socks'],
        'Color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'pink', 'gray', 'brown'],
        'Food': ['pizza', 'burger', 'sandwich', 'salad', 'cake', 'coffee', 'tea', 'beer', 'ice cream', 'noodles'],
        'Object': ['chair', 'table', 'phone', 'laptop', 'car', 'bottle', 'bag', 'cup', 'backpack', 'television'],
        'Role': ['chef', 'teacher', 'athlete', 'doctor', 'engineer', 'artist', 'pilot', 'firefighter', 'police officer', 'lawyer'],
        'Scene': ['beach', 'kitchen', 'forest', 'street', 'park', 'office', 'bedroom', 'classroom', 'stadium', 'playground'],
        'Vehicle': ['bus', 'truck', 'airplane', 'train', 'motorcycle', 'bicycle', 'boat', 'van', 'taxi', 'scooter'],
        'Weather': ['sandstorm', 'drought', 'rain', 'snow', 'fog', 'rainbow', 'storm', 'moonlight', 'overcast', 'frost']
    },
    'valuable under-captioned concepts': {
        'Texture': ['glossy', 'matte', 'rough', 'smooth', 'fuzzy', 'silky', 'grainy', 'wrinkled', 'slippery', 'furry'],
        'POV': ['close-up', 'wide shot', 'top-down', 'low angle', 'high angle', 'aerial', 'shallow-depth', 'long exposure'],
        'Emot.': ['tired', 'focused', 'surprised', 'proud', 'shy', 'bored', 'confused', 'excited', 'thoughtful', 'nervous']
    },
    'nuisance cues': {
        'Postproc.': ['hdr', 'desaturated', 'color-filtered', 'over-sharpened', 'saturated', 'sepia', 'heavily edited', 'bokeh', 'tilt-shift'],
    },
    'sensitive concepts': {
        'Trait': ['easy-going', 'cold-hearted', 'evil', 'criminal', 'immoral', 'pure', 'corrupt', 'hero', 'cheater', 'deserving'],
        'Stere.': ['exotic look', 'ethnic look', 'slacker', 'aggressive look', 'gangster look', 'terrorist look', 'illegal look', 'privileged', 'ghetto look']
    }
}

# === Directory to save per-group plots ===
output_dir = "clip_probing/plot_figures"
concepts_dir = os.path.join(output_dir, "concepts")
os.makedirs(concepts_dir, exist_ok=True)

# === Store per-group stats and values for violin plot ===
group_stat_records = []
violin_data = []
group_labels = []

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

# === Plot per group and collect stats ===
for main_cat, subcats in concept_dict.items():
    for group, concept_list in subcats.items():
        concept_data = {}
        for concept in concept_list:
            concept_norm = concept.strip().lower()
            percentage = float(percentage_map.get(concept_norm, 0.0))
            concept_data[concept] = percentage

        # Calculate statistics
        values = list(concept_data.values())
        mean_percentage = np.mean(values)
        median_percentage = float(np.median(values))
        max_percentage = max(values)
        min_percentage = min(values)

        # Save stats for CSV and plotting
        group_stat_records.append({
            "main_category": main_cat,
            "group": group,
            "mean": mean_percentage,
            "median": median_percentage,
            "max": max_percentage,
            "min": min_percentage
        })

        violin_data.append(values)
        group_labels.append(group)

        # Plot this group as a bar plot (for individual inspection, optional)
        sorted_items = sorted(concept_data.items(), key=lambda x: x[1], reverse=True)
        concepts, percentages = zip(*sorted_items)
        plt.figure(figsize=(0.5 * len(concepts), 4))
        plt.bar(concepts, percentages)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Percentage (%)")
        plt.title(f"{group.capitalize()} (mean: {mean_percentage:.3f}%)")
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(concepts_dir, f"{group.replace(' ', '_')}.pdf"))
        plt.close()

# === Save group stats to CSV ===
stat_df = pd.DataFrame(group_stat_records)
stat_df = stat_df[['main_category', 'group', 'mean', 'median', 'max', 'min']]
stat_csv_path = os.path.join(output_dir, "summary_group_statistics.csv")
stat_df.to_csv(stat_csv_path, index=False)
print(f"Saved group statistics to {stat_csv_path}")

# === Violin plot for statistics by group, with median dots, value labels, sorted by median ===
means = [np.mean(data) for data in violin_data]
sort_idx = np.argsort(means)[::-1]  # descending
# Filter out 'ACT' (actions) from the sorted lists
sorted_group_labels = []
sorted_violin_data = []
sorted_means = []
for i in sort_idx:
    group = group_labels[i]
    if group != 'ACT':  # Exclude 'actions'
        sorted_group_labels.append(group)
        sorted_violin_data.append(violin_data[i])
        sorted_means.append(means[i])

fig, ax = plt.subplots(figsize=(0.7 * len(sorted_group_labels), 3))

parts = ax.violinplot(sorted_violin_data, showmeans=False, showmedians=True, showextrema=True)

# Style violins
for pc in parts['bodies']:
    pc.set_facecolor('#7EA6E0')
    pc.set_alpha(0.5)
    pc.set_edgecolor('#6C8EBF')
    pc.set_linewidth(1)

# Extract actual violin color for legend
violin_body_color = parts['bodies'][0].get_facecolor().flatten()
violin_legend_patch = mpl.patches.Patch(color=violin_body_color, label='Distribution')

# Style median line (solid gold)
median_line_color = '#FFA500'
if 'cmedians' in parts:
    parts['cmedians'].set_color(median_line_color)
    parts['cmedians'].set_linewidth(1.5)
    parts['cmedians'].set_linestyle('-')

median_line_legend = mpl.lines.Line2D([], [], color=median_line_color, linestyle='-', linewidth=2, label='Median value')

# Overlay mean dots and annotate
mean_dot_color = 'black'
positions = np.arange(1, len(sorted_group_labels) + 1)
ax.scatter(positions, sorted_means, color=mean_dot_color, marker='.', zorder=3, label='Mean value', s=35)
for pos, mean in zip(positions, sorted_means):
    ax.text(pos, mean + 0.2 * max(sorted_means), f"{mean:.4f}", color=mean_dot_color, fontsize=9, 
            ha='center', va='bottom', fontweight='bold')

mean_dot_legend = mpl.lines.Line2D([], [], color=mean_dot_color, marker='o', linestyle='', label='Mean value', markersize=8)

# Insert a vertical line between the last 7 and last 6 concepts
split_index = len(sorted_group_labels) - 6 + 0.5  # +0.5 to place line between categories
ax.axvline(split_index, color='#B22222', linestyle='--', linewidth=1, alpha=0.7)

ax.legend(handles=[violin_legend_patch, median_line_legend, mean_dot_legend], loc='upper right')

ax.set_ylabel("Percentage (%)")
# ax.set_title("Concept Coverage Distribution by Group (sorted by mean)")
ax.set_xticks(positions)
ax.set_xticklabels(sorted_group_labels, rotation=0)# 45, ha='right')
# ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "group_coverage_violinplot.pdf"))
plt.show()
