#!/bin/bash
#SBATCH -A m4474
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:20:00
#SBATCH -N 1
#SBATCH -J vqvae
#SBATCH --array=0-3
#SBATCH --mail-user=rmastand@berkeley.edu
#SBATCH --mail-type=ALL

# Activate environment
source /pscratch/sd/r/rmastand/particlemind_env/bin/activate
module load pytorch

# Select config based on array index
CONFIG="vqvae_${SLURM_ARRAY_TASK_ID}"

echo "Running config: ${CONFIG}"

srun --ntasks-per-node=4 \
     -c 32 \
     -G 4 \
     --cpu-bind=cores \
     --gpu-bind=none \
     python -m src.train-radha \
       --train_embedder \
       --config_embedder ${CONFIG}
