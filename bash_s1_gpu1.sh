echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb2_tr1" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb2_tr2" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb2_tr3" --beta-value 10 --causal-dependence --mlp-eval
wait

echo "training end."