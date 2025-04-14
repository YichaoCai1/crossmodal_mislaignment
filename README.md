# Crossmodal Misalignment

Official code for the paper:  **"Negate or Embrace: On How Misalignment Shapes Multimodal Representation Learning"**

---

## 1. Installation

```bash
# Install dependencies (preferably inside a conda/virtual environment)
pip install -r requirements.txt

# Verify CUDA support; this should not raise an error
python -c "import torch; assert torch.cuda.device_count() > 0, 'No CUDA support detected'"
```

## 2 Numerical Experiments

### 2.1 Training Representations with MMCL
```bash
# Training artifacts will be saved to the path specified by --model-id

# --theta-value: controls the selected subset (𝕀_θ)
#   - Default: 1022 (full semantics)
#   - Alternatives: 0, 10, 55, 175, 385, 637, 847, 967, 1012 
#     (corresponding to progressive inclusion of latent indices)

# --beta-value: controls the perturbable subset (𝕀_ρ)
#   - Default: -1 (empty set)
#   - Alternatives: same as above

# --causal-dependence: enables statistical dependencies among latent semantics.
#   If not set, assumes independence.

# --encoding-size: dimensionality of the learned representation.
#   If not specified, uses the true unbiased dimensionality.

# Additional arguments can be found in `main_numeric.py`. Default values were used in our experiments.

python main_numeric.py \
    --model-id $OUTPUT_DIR$ \
    --theta-value 1022 \
    --beta-value -1 \
    --causal-dependence \
    --mlp-eval \
    --encoding-size 10
```

### 2.2 Evaluating Downstream Performance

Use the following script to evaluate downstream tasks. Results will be saved in each model directory as ```results_predict.csv``` and ```results_classify.csv```.
```bash
BASE_DIR="models/{YOUR_ARTIFACTS_DIRS}"

# Evaluate all model directories under BASE_DIR
for MODEL_ID in $(ls -d "$BASE_DIR"/*/ 2>/dev/null); do
    echo "Running evaluation for: $MODEL_ID"
    python eval_numeric_predict.py --model-id "$MODEL_ID"
    wait
done

echo "All evaluations completed!"
```

## 3 MPI3D-Complex Dataset
### 3.1 Preparing the Data 

1. Download the image data from [Google Drive Link](https://drive.google.com/file/d/1Tp8eTdHxgUMtsZv5uAoYAbJR1BOa_OQm/view?usp=sharing)
(Originally from [rr-learning/disentanglement_dataset](https://github.com/rr-learning/disentanglement_dataset))

2. Prepare the data:
```bash
mkdir -p data/MPI_real_complex  # Create directory structure

# Move 'real3d_complicated_shapes_ordered.npz' into data/MPI_real_complex

# Generate multimodal text data for all configurations
python rendering_data/MPI3D/renfer_multimodal_mpi3d.py
```

### 3.2 Running Experiments

```bash
# --model-id: output directory
# --bias-type: "selection" (default) or "perturbation"
# --bias-id:
#   - For selection: 0–4, where 4 means all semantic factors
#   - For perturbation: 0–4, increasing number of perturbed dimensions

python main_mpi3dreal.py \
    --datapath "./data/MPI3d_real_complex/" \
    --model-id $OUTPUT_DIR$ \
    --bias-type "perturbation" \
    --bias-id 2
```

## 4 Causal3DIdent Dataset
### 4.1 Preparing the Data 
We build upon [Multimodal3DIdent](https://github.com/imantdaunhawer/Multimodal3DIdent), extending it to support selection/perturbation biases and causal dependencies.

1. Clone the repo and install dependencies following its instructions.

2. Replace relevant files with those in rendering_data/Causal3DIdent/.

3. Render the data as per their pipeline.

4. Organize the outputs as:

```bash
data/Causal3DIDent/
├── images/
│   └── *****.jpg
├── text/
│   ├── perturbations_0/
|   ├── ...
│   ├── perturbations_6/
│   ├── selections_0/
|   ├── ...
│   └── selections_6/
├── latents_image.csv
├── latents_text_perturbations_*.csv
└── latents_text_selections_*.csv
```

### 4.2 Running Experiments

```bash
# --model-id: output directory
# --bias-type: "selections" (default) or "perturbations"
# --bias-id:
#   - Selections: 0–6 (more semantics included)
#   - Perturbations: 0–6 (more perturbations applied)

python main_causal3dident.py \
    --datapath "data/Causal3DIDent" \
    --model-id $OUTPUT_DIR$ \
    --bias-type "selections" \
    --bias-id 1
```


## Acknowledgements

The code is based on:
- https://github.com/ysharma1126/ssl_identifiability
- https://github.com/brendel-group/cl-ica
- https://github.com/imantdaunhawer/multimodal-contrastive-learning