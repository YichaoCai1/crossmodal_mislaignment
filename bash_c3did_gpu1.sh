echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_4" --bias-type "selections" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_4" --bias-type "selections" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_4" --bias-type "selections" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_5" --bias-type "selections" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_5" --bias-type "selections" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_5" --bias-type "selections" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_6" --bias-type "selections" --bias-id 6
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_6" --bias-type "selections" --bias-id 6
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_6" --bias-type "selections" --bias-id 6
wait

echo "training end."