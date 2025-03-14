echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=1 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc2_tr1" --bias-type "selection" --bias-id 1 --encoding-size 2
wait
CUDA_VISIBLE_DEVICES=1 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc2_tr2" --bias-type "selection" --bias-id 1 --encoding-size 2
wait
CUDA_VISIBLE_DEVICES=1 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc2_tr3" --bias-type "selection" --bias-id 1 --encoding-size 2
wait


echo "training end."