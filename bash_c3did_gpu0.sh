echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_0_tr1" --bias-type "selections" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_0_tr2" --bias-type "selections" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_0_tr3" --bias-type "selections" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_1_tr1" --bias-type "selections" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_1_tr2" --bias-type "selections" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_1_tr3" --bias-type "selections" --bias-id 1
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_2_tr1" --bias-type "selections" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_2_tr2" --bias-type "selections" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_2_tr3" --bias-type "selections" --bias-id 2
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_3_tr1" --bias-type "selections" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_3_tr2" --bias-type "selections" --bias-id 3
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_3_tr3" --bias-type "selections" --bias-id 3
wait

echo "training end."