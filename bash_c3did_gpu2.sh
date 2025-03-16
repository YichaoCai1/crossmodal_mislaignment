echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_0" --bias-type "perturbations" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_0" --bias-type "perturbations" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_0" --bias-type "perturbations" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_1" --bias-type "perturbations" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_1" --bias-type "perturbations" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_1" --bias-type "perturbations" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_2" --bias-type "perturbations" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_2" --bias-type "perturbations" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=2 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_2" --bias-type "perturbations" --bias-id 2
wait

echo "training end."