echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_3" --bias-type "perturbations" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_3" --bias-type "perturbations" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_3" --bias-type "perturbations" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_4" --bias-type "perturbations" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_4" --bias-type "perturbations" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_4" --bias-type "perturbations" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_5" --bias-type "perturbations" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_5" --bias-type "perturbations" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_5" --bias-type "perturbations" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_6" --bias-type "perturbations" --bias-id 6
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_6" --bias-type "perturbations" --bias-id 6
wait
CUDA_VISIBLE_DEVICES=3 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_perturbations_6" --bias-type "perturbations" --bias-id 6
wait

echo "training end."