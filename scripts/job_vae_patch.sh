#!/bin/bash
#SBATCH --account=pi-dfreedman 
#SBATCH --job-name=patch
#SBATCH --partition=gpu   
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

# -----------------------------
# Environment
# -----------------------------
source /scratch/midway3/rmastand/particlemind_env/bin/activate 
cd /home/rmastand/particlemind
export HF_TOKEN=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
export PYTHONPATH=/home/rmastand/particlemind:$PYTHONPATH

commands=(
  "python -u src/train-radha.py   --name vae_patch   --config_data data_patch    --config_embedder vae_patch  --train_embedder"


  "python -u src/train-radha.py   --name vqvae_patch   --config_data data_patch    --config_embedder vqvae_patch  --train_embedder"





  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}

