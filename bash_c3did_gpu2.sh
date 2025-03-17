echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_0_tr1" --bias-type "perturbations" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_0_tr2" --bias-type "perturbations" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_0_tr3" --bias-type "perturbations" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_1_tr1" --bias-type "perturbations" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_1_tr2" --bias-type "perturbations" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_1_tr3" --bias-type "perturbations" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_2_tr1" --bias-type "perturbations" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_2_tr2" --bias-type "perturbations" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_2_tr3" --bias-type "perturbations" --bias-id 2
wait

echo "training end."