echo "run numerical simulaton..."


CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_3_tr1" --bias-type "perturbation" --bias-id 3 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_3_tr2" --bias-type "perturbation" --bias-id 3 --encoding-size 3
wait
CUDA_VISIBLE_DEVICES=3 python main_mpi3dreal.py --datapath "./data/MPI3d_real_complex/" --model-id "MPI3d/mpi3d_perturbation_3_tr3" --bias-type "perturbation" --bias-id 3 --encoding-size 3
wait


echo "training end."