echo "run numerical simulaton..."
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "dep_perturb1_tr1" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "dep_perturb1_tr2" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "dep_perturb1_tr3" --beta-value 0 --causal-dependence --mlp-eval

echo "training end."