echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_tr1" --bias-type "selection" --bias-id 0 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_tr2" --bias-type "selection" --bias-id 0 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_0_tr3" --bias-type "selection" --bias-id 0 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_1_tr1" --bias-type "selection" --bias-id 1 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_1_tr2" --bias-type "selection" --bias-id 1 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_selection_1_tr3" --bias-type "selection" --bias-id 1 --encoding-size 3
wait

echo "training end."