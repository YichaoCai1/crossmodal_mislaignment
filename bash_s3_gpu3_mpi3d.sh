echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_4_tr1" --bias-type "perturbation" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_4_tr2" --bias-type "perturbation" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_4_tr3" --bias-type "perturbation" --bias-id 4
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_5_tr1" --bias-type "perturbation" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_5_tr2" --bias-type "perturbation" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_5_tr3" --bias-type "perturbation" --bias-id 5
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_6_tr1" --bias-type "perturbation" --bias-id 6
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_6_tr2" --bias-type "perturbation" --bias-id 6
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "mpi3d_perturbation_6_tr3" --bias-type "perturbation" --bias-id 6
wait

echo "training end."