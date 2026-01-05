#!/bin/bash
#SBATCH -A m4474
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 0:30:00
#SBATCH -N 1
#SBATCH -J vqvae
#SBATCH --array=0-3
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
  --config_embedder ${CONFIG}
