echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "dep_drop2_perturb2_tr1" --theta-value 967 --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "dep_drop2_perturb2_tr2" --theta-value 967 --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "dep_drop2_perturb2_tr3" --theta-value 967 --beta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "ind_drop2_perturb2_tr1" --theta-value 967 --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "ind_drop2_perturb2_tr2" --theta-value 967 --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "ind_drop2_perturb2_tr3" --theta-value 967 --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "dep_drop7_encode5_tr1" --theta-value 55 --encoding-size 5 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "dep_drop7_encode5_tr2" --theta-value 55 --encoding-size 5 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "dep_drop7_encode5_tr3" --theta-value 55 --encoding-size 5 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "ind_drop7_encode5_tr1" --theta-value 55 --encoding-size 5 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "ind_drop7_encode5_tr2" --theta-value 55 --encoding-size 5 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=2 python main_numeric.py --model-id "ind_drop7_encode5_tr3" --theta-value 55 --encoding-size 5 --mlp-eval
wait

echo "training end."