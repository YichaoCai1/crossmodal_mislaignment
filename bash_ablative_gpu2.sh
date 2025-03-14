echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=2 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc4_tr1" --bias-type "selection" --bias-id 1 --encoding-size 4
wait
CUDA_VISIBLE_DEVICES=2 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc4_tr2" --bias-type "selection" --bias-id 1 --encoding-size 4
wait
CUDA_VISIBLE_DEVICES=2 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_enc4_tr3" --bias-type "selection" --bias-id 1 --encoding-size 4
wait


echo "training end."