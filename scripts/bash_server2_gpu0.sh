echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "ind_drop3_tr1" --theta-value 847 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "ind_drop3_tr2" --theta-value 847 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "ind_drop1_tr1" --theta-value 1012 --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "ind_drop1_tr3" --theta-value 1012 --mlp-eval
wait

echo "training end."