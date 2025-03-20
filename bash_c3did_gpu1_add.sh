echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_enc2_0_tr1" --bias-type "selections" --bias-id 0 --encoding-size 1
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_enc2_0_tr2" --bias-type "selections" --bias-id 0 --encoding-size 1
wait
CUDA_VISIBLE_DEVICES=1 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_enc2_0_tr3" --bias-type "selections" --bias-id 0 --encoding-size 1
wait

echo "training end."