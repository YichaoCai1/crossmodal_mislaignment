echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb3_tr1" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb3_tr2" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb3_tr3" --beta-value 55 --causal-dependence --mlp-eval
wait

echo "training end."