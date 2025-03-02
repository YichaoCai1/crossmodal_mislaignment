echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb2_tr1" --beta-value 10 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "ind_perturb1_tr2" --beta-value 0 --mlp-eval
wait

echo "training end."