import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
import seaborn as sns
import open_clip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

# ------------ CONFIG -----------------
ROOT_DIR = r'clip_probing/flickr_images'  # change to your dataset root
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# MODELS TO TEST
MODELS = [
    {"name": "ViT-B-32", "pretrained": "laion400m_e32"},
    {"name": "ViT-L-14", "pretrained": "laion400m_e32"}
]
# -------------------------------------

def find_groups(root):
    groups = {}
    for note in os.listdir(root):
        note_path = os.path.join(root, note)
        if not os.path.isdir(note_path):
            continue
        for group in os.listdir(note_path):
            group_path = os.path.join(note_path, group)
            if not os.path.isdir(group_path):
                continue
            for concept in os.listdir(group_path):
                concept_path = os.path.join(group_path, concept)
                if not os.path.isdir(concept_path):
                    continue
                imgs = []
                for ext in IMAGE_EXTS:
                    imgs.extend(glob(os.path.join(concept_path, f'*{ext}')))
                    imgs.extend(glob(os.path.join(concept_path, f'*{ext.upper()}')))
                if imgs:
                    groups.setdefault(group, {}).setdefault(concept, []).extend(imgs)
    return groups

def zero_shot_predict(img_paths, concept_names, model, tokenizer, preprocess, prompt_template=None):
    if prompt_template is None:
        prompts = concept_names
    else:
        prompts = [prompt_template.format(c) for c in concept_names]
    with torch.no_grad():
        text = tokenizer(prompts).to(DEVICE)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        preds = []
        labels = []
        for idx in tqdm(range(0, len(img_paths), BATCH_SIZE), desc="Batching images"):
            batch_paths = img_paths[idx:idx+BATCH_SIZE]
            images = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert('RGB')
                    img = preprocess(img).unsqueeze(0)
                    images.append(img)
                except Exception as e:
                    print(f"Error loading {p}: {e}")
            if not images:
                continue
            image_input = torch.cat(images).to(DEVICE)
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            top1 = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(top1)
            batch_labels = [concept_names.index(os.path.basename(os.path.dirname(p))) for p in batch_paths]
            labels.extend(batch_labels)
    return labels, preds


import seaborn as sns

def plot_and_report(group, concept_names, labels, preds, model_name):
    # Compute macro F1 score
    f1_macro = f1_score(labels, preds, average='macro')
    print(f"[{group}][{model_name}] Macro F1 Score: {f1_macro*100:.2f}%")

    # Compute confusion matrix
    cm = confusion_matrix(labels, preds, labels=range(len(concept_names)))

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=False,
        fmt='d',
        cmap="Blues",
        cbar=False,
        xticklabels=concept_names,
        yticklabels=concept_names,
        ax=ax
    )

    # Axis labels and title
    ax.set_title(f"Confusion Matrix for {group} ({model_name})\nMacro F1: {f1_macro*100:.2f}%", fontsize=20, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=20)
    ax.set_ylabel('True Label', fontsize=20)
    ax.tick_params(axis='x', labelsize=20, rotation=45)
    ax.tick_params(axis='y', labelsize=20, rotation=0)

    # Annotate cells with values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            ax.text(
                j + 0.5, i + 0.5, str(value),
                ha='center', va='center',
                fontsize=20,
                color='black' if value < cm.max() / 2 else 'white'
            )

    plt.tight_layout()

    # Save figure
    fname = os.path.join(
        RESULTS_DIR, f"confmat_{group.replace(' ', '_')}_{model_name.replace('-', '_')}.pdf"
    )
    plt.savefig(fname)
    print(f"Confusion matrix saved to {fname}")
    plt.close(fig)

    return f1_macro


def save_concept_image_counts(groups, output_csv_path):
    records = []
    for group_name, concept_dict in groups.items():
        for concept_name, img_paths in concept_dict.items():
            count = len(img_paths)
            records.append({
                "group": group_name,
                "concept": concept_name,
                "count": count
            })
    
    df = pd.DataFrame(records)
    df = df.sort_values(by=["group", "count"], ascending=[True, False])
    df.to_csv(output_csv_path, index=False)
    print(f"Saved concept image counts to {output_csv_path}")

import pandas as pd
def main():
    
    groups = find_groups(ROOT_DIR)
    print("Found groups:", list(groups.keys()))

    
    csv_path = os.path.join(RESULTS_DIR, "summary_group_f1_scores.csv")
    # Try loading from CSV
    if os.path.isfile(csv_path):
        print(f"Found cached F1 CSV at {csv_path}, loading instead of re-evaluating...")
        df = pd.read_csv(csv_path, index_col=0)
        all_results = {model: {group: df.loc[group, model] for group in df.index if not pd.isna(df.loc[group, model])}
                       for model in df.columns}
    else:
        # {model_name: {group: f1}}
        all_results = {m["name"]: {} for m in MODELS}

        for model_cfg in MODELS:
            print(f"\n==== Evaluating model: {model_cfg['name']} ({model_cfg['pretrained']}) ====")
            model, _, preprocess = open_clip.create_model_and_transforms(model_cfg["name"], pretrained=model_cfg["pretrained"])
            tokenizer = open_clip.get_tokenizer(model_cfg["name"])
            model = model.to(DEVICE)
            model.eval()

            for group, concept_dict in groups.items():
                print(f"\n--- Probing group: {group} ---")
                concept_names = sorted(list(concept_dict.keys()))
                img_paths = []
                for concept in concept_names:
                    img_paths.extend(concept_dict[concept])
                if len(concept_names) < 2 or len(img_paths) == 0:
                    print(f"Skipping {group}: not enough data.")
                    continue
                labels, preds = zero_shot_predict(img_paths, concept_names, model, tokenizer, preprocess)
                f1_macro = plot_and_report(group, concept_names, labels, preds, model_cfg["name"])
                all_results[model_cfg["name"]][group] = f1_macro

        # Save results to CSV
        # Convert all_results to DataFrame
        df = pd.DataFrame(all_results).sort_index()
        df.to_csv(csv_path)
        print(f"Saved F1 scores to {csv_path}")

    # --- Visualization: Bar/Line Plot, sorted by overall mean ---
    all_groups = list(df.index)
    n_models = len(MODELS)

    # Compute group-wise mean F1
    mean_f1s = []
    for g in all_groups:
        vals = [df.loc[g, m["name"]] if m["name"] in df.columns else 0 for m in MODELS]
        mean_f1s.append(np.mean(vals))
    # Sort indices by mean F1 (descending)
    sorted_idx = np.argsort(mean_f1s)[::-1]
    sorted_groups = [all_groups[i] for i in sorted_idx]

    width = 0.8 / n_models
    x = np.arange(len(sorted_groups))
    colors = ['#2269AC', '#C0D2EB']

    # Bar Chart
    fig, ax = plt.subplots(figsize=(0.7*len(sorted_groups), 3))
    for i, model_cfg in enumerate(MODELS):
        model_name = model_cfg["name"]
        f1s = [df.loc[g, model_name]*100 if model_name in df.columns and not pd.isna(df.loc[g, model_name]) else 0 for g in sorted_groups]
        ax.bar(x + i*width, f1s, width=width, label=model_name, color=colors[i % len(colors)])
        for xi, f1 in zip(x, f1s):
            ax.text(xi + i*width, f1, f"{f1:.1f}", ha='center', va='bottom', fontsize=7, fontweight='bold')

    # Insert a vertical line between the last 7 and last 6 concepts
    split_index = len(sorted_groups) - 6.3  # +0.5 to place line between categories
    ax.axvline(split_index, color='#B22222', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_ylabel("Macro F1 Score (%)")
    # ax.set_title("Group-wise Macro F1 Score by Model (Sorted)")
    ax.set_xticks(x + width*(n_models-1)/2)
    ax.set_xticklabels(sorted_groups, rotation=0) # 45, ha='right')
    ax.legend()
    # ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "summary_group_f1_scores_bar.pdf"))
    plt.show()
    
    count_csv_path = os.path.join(RESULTS_DIR, "concept_image_counts.csv")
    save_concept_image_counts(groups, count_csv_path)

if __name__ == "__main__":
    main()
