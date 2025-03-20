echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_0_enc2_tr1" --bias-type "selections" --bias-id 0 --encoding-size 2
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_0_enc2_tr2" --bias-type "selections" --bias-id 0 --encoding-size 2
wait
CUDA_VISIBLE_DEVICES=0 python main_causal3dident.py --datapath "data/Causal3DIDent" --model-id "C3dId/c3did_selections_0_enc2_tr3" --bias-type "selections" --bias-id 0 --encoding-size 2
wait

echo "training end."