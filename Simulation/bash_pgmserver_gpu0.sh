echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop9_tr1" --theta-value 0 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop9_tr2" --theta-value 0 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop9_tr3" --theta-value 0 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop7_tr1" --theta-value 55 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop7_tr2" --theta-value 55 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop7_tr3" --theta-value 55 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop5_tr1" --theta-value 385 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop5_tr2" --theta-value 385 --multi-distribution --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "multidist_ind_drop5_tr3" --theta-value 385 --multi-distribution --mlp-eval
wait

echo "training end."