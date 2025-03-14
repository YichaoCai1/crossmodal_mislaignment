echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop9_tr1" --theta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop9_tr2" --theta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop9_tr3" --theta-value 0 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop8_tr1" --theta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop8_tr2" --theta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop8_tr3" --theta-value 10 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop7_tr1" --theta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop7_tr2" --theta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop7_tr3" --theta-value 55 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop6_tr1" --theta-value 175 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop6_tr2" --theta-value 175 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop6_tr3" --theta-value 175 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop5_tr1" --theta-value 385 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop5_tr2" --theta-value 385 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop5_tr3" --theta-value 385 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop4_tr1" --theta-value 637 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop4_tr2" --theta-value 637 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop4_tr3" --theta-value 637 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop3_tr1" --theta-value 847 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop3_tr2" --theta-value 847 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop3_tr3" --theta-value 847 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop2_tr1" --theta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop2_tr2" --theta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop2_tr3" --theta-value 967 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop1_tr1" --theta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop1_tr2" --theta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop1_tr3" --theta-value 1012 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop0_tr1" --theta-value 1022 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop0_tr2" --theta-value 1022 --causal-dependence --mlp-eval
wait
CUDA_VISIBLE_DEVICES=0 python main_numeric.py --model-id "Numeric/dep_drop0_tr3" --theta-value 1022 --causal-dependence --mlp-eval
wait

echo "training end."