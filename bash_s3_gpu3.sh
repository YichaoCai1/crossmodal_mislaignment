echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb1_tr1" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb1_tr2" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb1_tr3" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb2_tr1" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb2_tr2" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb2_tr3" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb3_tr1" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb3_tr2" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_perturb3_tr3" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_drop2_tr1" --theta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_drop1_tr2" --theta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=3 python main_numeric.py --model-id "dep_drop1_tr3" --theta-value 1012 --causal-dependence --mlp-eval
wait

echo "training end."