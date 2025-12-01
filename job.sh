#!/bin/bash
#SBATCH -A m4474
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:20:00
#SBATCH -N 1
#SBATCH -J vqvae
#SBATCH --mail-user=rmastand@berkeley.edu
#SBATCH --mail-type=ALL # Options: BEGIN, END, FAIL, ALL

source /pscratch/sd/r/rmastand/particlemind_env/bin/activate
module load pytorch

srun --ntasks-per-node=4 -c 32 -G 4 --cpu-bind=cores --gpu-bind=none python -m src.train-radha --train_embedder