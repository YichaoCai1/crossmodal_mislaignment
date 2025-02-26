echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "multidist_ind_drop3_tr1" --theta-value 847 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "multidist_ind_drop3_tr2" --theta-value 847 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "multidist_ind_drop3_tr3" --theta-value 847 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "multidist_ind_drop1_tr1" --theta-value 1012 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "multidist_ind_drop1_tr2" --theta-value 1012 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=1 python main_numeric.py --model-id "multidist_ind_drop1_tr3" --theta-value 1012 --multi-distribution --mlp-eval
wait


echo "training end."