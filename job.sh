#!/bin/bash
#SBATCH -A m4474
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 24:0:00
#SBATCH -N 4
#SBATCH --gpus-per-node=1
#SBATCH -J vqvae
#SBATCH --mail-user=rmastand@berkeley.edu
#SBATCH --mail-type=ALL   # Options: BEGIN, END, FAIL, ALL


source /pscratch/sd/r/rmastand/particlemind_env/bin/activate
module load pytorch


srun python -m src.train-radha --train_embedder
#srun python -m src.train-radha --train_tokenizer
