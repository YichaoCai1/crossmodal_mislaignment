echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc1_tr1" --bias-type "selection" --bias-id 0 --encoding-size 1
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc1_tr2" --bias-type "selection" --bias-id 0 --encoding-size 1
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc1_tr3" --bias-type "selection" --bias-id 0 --encoding-size 1
wait

echo "training end."