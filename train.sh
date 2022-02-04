#!/bin/bash
#SBATCH --account def-lemc2220
#SBATCH --gres=gpu:p100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=20000M        # memory per node
#SBATCH --time=7-00:00      # time (DD-HH:MM)
#SBATCH --mail-user carl.lemaire@calculquebec.ca
#SBATCH --mail-type ALL


module load gcc/8.4.0 cuda/10.2
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64
export TORCH_CUDA_ARCH_LIST='6.0'
export FORCE_CUDA=1

cd /home/lemc2220/projects/def-rodrigu1/tmp/program/dMaSIF

source env/bin/activate

tensorboard --logdir=runs --host 0.0.0.0 --load_fast false &

python -W ignore -u main_training.py --experiment_name dMaSIF_search_1layer_12A --batch_size 64 --embedding_layer dMaSIF --search True --device cuda:0 --random_rotation True --radius 12.0 --n_layers 3 --emb_dims 16 --n_epochs 100

