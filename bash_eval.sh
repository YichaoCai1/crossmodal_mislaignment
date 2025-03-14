echo "run numerical simulaton..."

CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_1_tr3" --bias-type "perturbation" --bias-id 1 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_2_tr1" --bias-type "perturbation" --bias-id 2 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_2_tr2" --bias-type "perturbation" --bias-id 2 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_2_tr3" --bias-type "perturbation" --bias-id 2 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_3_tr1" --bias-type "perturbation" --bias-id 3 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_0_tr1" --bias-type "perturbation" --bias-id 0 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_0_tr2" --bias-type "perturbation" --bias-id 0 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_0_tr3" --bias-type "perturbation" --bias-id 0 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_1_tr1" --bias-type "perturbation" --bias-id 1 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_1_tr2" --bias-type "perturbation" --bias-id 1 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_3_tr2" --bias-type "perturbation" --bias-id 3 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_3_tr3" --bias-type "perturbation" --bias-id 3 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_4_tr1" --bias-type "perturbation" --bias-id 4 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_4_tr2" --bias-type "perturbation" --bias-id 4 --evaluate
wait
CUDA_VISIBLE_DEVICES=0 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_4_tr3" --bias-type "perturbation" --bias-id 4 --evaluate
wait

echo "training end."