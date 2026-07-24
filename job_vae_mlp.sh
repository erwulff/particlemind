#!/bin/bash
#SBATCH --account=pi-rmastand  
#SBATCH --job-name=vae-mlp
#SBATCH --partition=gpu   
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:15:00
#SBATCH --array=0-2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rmastand@rcc.uchicago.edu  # Where to send email

# -----------------------------
# Environment
# -----------------------------
cd
pmind_env


commands=(
  "python -u -m src.train-radha \
    --name mlp1_weight0
    --config_data data_hit_mixed \
    --config_embedder vae_mlp_hit-Copy1 \
    --train_embedder"

  "python -u -m src.train-radha \
    --name mlp2_weight0
    --config_data data_hit_mixed \
    --config_embedder vae_mlp_hit-Copy2 \
    --train_embedder"

  "python -u -m src.train-radha \
    --name mlp3_weight0
    --config_data data_hit_mixed \
    --config_embedder vae_mlp_hit-Copy3 \
    --train_embedder"





  )

exec ${commands[$SLURM_ARRAY_TASK_ID]}

