echo "run numerical simulaton..."


CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_0_tr1" --bias-type "perturbation" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_0_tr2" --bias-type "perturbation" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_0_tr3" --bias-type "perturbation" --bias-id 0
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_4_tr1" --bias-type "perturbation" --bias-id 4 --encoding-size 3
wait

echo "training end."