echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb9_tr1" --beta-value 1012 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb9_tr2" --beta-value 1012 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb9_tr3" --beta-value 1012 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb8_tr1" --beta-value 967 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb8_tr2" --beta-value 967 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb8_tr3" --beta-value 967 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb7_tr1" --beta-value 847 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb7_tr2" --beta-value 847 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb7_tr3" --beta-value 847 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb6_tr1" --beta-value 637 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb6_tr2" --beta-value 637 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb6_tr3" --beta-value 637 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb5_tr1" --beta-value 385 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb5_tr2" --beta-value 385 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb5_tr3" --beta-value 385 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb4_tr1" --beta-value 175 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb4_tr2" --beta-value 175 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb4_tr3" --beta-value 175 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb3_tr1" --beta-value 55 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb3_tr2" --beta-value 55 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb3_tr3" --beta-value 55 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb2_tr1" --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb2_tr2" --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb2_tr3" --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb1_tr1" --beta-value 0 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb1_tr2" --beta-value 0 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb1_tr3" --beta-value 0 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb0_tr1"  --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb0_tr2"  --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb0_tr3"  --mlp-eval
wait

echo "training end."