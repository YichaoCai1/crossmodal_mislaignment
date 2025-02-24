echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb9_tr1" --beta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb9_tr2" --beta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb9_tr3" --beta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb8_tr1" --beta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb8_tr2" --beta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb8_tr3" --beta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb7_tr1" --beta-value 847 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb7_tr2" --beta-value 847 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb7_tr3" --beta-value 847 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb6_tr1" --beta-value 637 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb6_tr2" --beta-value 637 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb6_tr3" --beta-value 637 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb5_tr1" --beta-value 385 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb5_tr2" --beta-value 385 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb5_tr3" --beta-value 385 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb4_tr1" --beta-value 175 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb4_tr2" --beta-value 175 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb4_tr3" --beta-value 175 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb3_tr1" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb3_tr2" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb3_tr3" --beta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb2_tr1" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb2_tr2" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb2_tr3" --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb1_tr1" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb1_tr2" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb1_tr3" --beta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb0_tr1"  --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb0_tr2"  --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "dep_perturb0_tr3"  --causal-dependence --mlp-eval
wait

echo "training end."