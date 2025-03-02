echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_drop2_tr1" --theta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_drop1_tr2" --theta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_drop1_tr3" --theta-value 1012 --causal-dependence --mlp-eval
wait

echo "training end."
