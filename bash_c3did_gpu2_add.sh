echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_3_tr1" --bias-type "perturbations" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_3_tr2" --bias-type "perturbations" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_3_tr3" --bias-type "perturbations" --bias-id 3
wait

echo "training end."