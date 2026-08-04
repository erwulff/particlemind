#!/bin/bash
#SBATCH --account=pi-dfreedman 
#SBATCH --job-name=vae-mlp
#SBATCH --partition=gpu
#SBATCH --constraint=a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=36:00:00
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
  "python -u src/train-radha.py \
    --name mlp1_sharedweight0 \
    --config_data data_hit_mixed \
    --config_embedder vae_mlp_hit-Copy1 \
    --train_embedder"

  "python -u src/train-radha.py \
    --name mlp2_sharedweight0 \
    --config_data data_hit_mixed \
    --config_embedder vae_mlp_hit-Copy2 \
    --train_embedder"

  "python -u src/train-radha.py \
    --name mlp3_sharedweight0 \
    --config_data data_hit_mixed \
    --config_embedder vae_mlp_hit-Copy3 \
    --train_embedder"





  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}

