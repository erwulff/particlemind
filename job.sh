#!/bin/bash
#SBATCH -A m4474
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -t 24:0:00
#SBATCH -N 1
#SBATCH -J vqvae_big
#SBATCH --array=1,2,3,4,5,7,10,11
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=rmastand@berkeley.edu
#SBATCH --mail-type=ALL

# -----------------------------
# Environment
# -----------------------------
source /pscratch/sd/r/rmastand/particlemind_env/bin/activate
module load pytorch

# -----------------------------
# Select config from array
# -----------------------------
CONFIG="vqvae_${SLURM_ARRAY_TASK_ID}"
echo "Running config: ${CONFIG}"

# -----------------------------
# Launch with torchrun
# -----------------------------
torchrun \
  --nproc_per_node=4 \
  -m src.train-radha \
  --train_embedder \
  --config_embedder ${CONFIG} \
  --name ${CONFIG}
