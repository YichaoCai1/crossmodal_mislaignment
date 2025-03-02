#!/bin/bash

BASE_DIR="models/2_dep_drops"

# Loop through only directories inside BASE_DIR
for MODEL_ID in $(ls -d "$BASE_DIR"/*/ 2>/dev/null); do
    echo "Running evaluation for: $MODEL_ID"
    python eval_numeric_predict.py --model-id "$MODEL_ID"
    wait
done

echo "All evaluations completed!"
