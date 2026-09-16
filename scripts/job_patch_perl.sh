#!/bin/bash
#SBATCH -A m4474
#SBATCH -q shared
#SBATCH -t 24:00:00
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -J patch
#SBATCH --array=1
#SBATCH --mail-user=rmastand@uchicago.edu
#SBATCH --mail-type=ALL


# -----------------------------
# Environment
# -----------------------------
module load pytorch
export HF_TOKEN=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
cd /global/u1/r/rmastand/mlpf-ssl/particlemind
export PYTHONPATH=/global/homes/r/rmastand/mlpf-ssl/particlemind:$PYTHONPATH

commands=(
  "python -u -m src.train-radha   --name vae_patch   --config_data data_patch    --config_embedder vae_patch  --train_embedder"


  "python -u -m src.train-radha   --name vqvae_patch   --config_data data_patch    --config_embedder vqvae_patch  --train_embedder"

MPICH_GPU_SUPPORT_ENABLED=0 PYTHONPATH=. python -u src/train-radha.py --name vae_patch --config_data data_patch --config_embedder vae_patch --train_embedder



  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}

