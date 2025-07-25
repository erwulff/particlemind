#!/bin/bash

#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --constraint=a100-80gb&sxm4
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

# --constraint=a100-80gb&sxm4
# --constraint=ib-h100p

# Job name
#SBATCH -J train_4a100

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err

# Add jobscript to job output
echo "#################### Job submission script. #############################"
cat $0
echo "################# End of job submission script. #########################"

set -x

module --force purge; module load modules/2.2-20230808
module load slurm gcc cmake cuda/12.1.1 cudnn/8.9.2.26-12.x nccl openmpi apptainer

nvidia-smi
export PYTHONPATH=`pwd`
source ~/miniforge3/bin/activate mind
which python3
python3 --version

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1


srun python src/train.py \
    --data_dir /mnt/ceph/users/ewulff/data/cld/processed/parquet \
    --input_size 4 \
    --output_size 4 \
    --embed_dim 64 \
    --num_layers 4 \
    --num_heads 4 \
    --ff_dim 64 \
    --dropout 0.01 \
    --batch_size 1 \
    --lr 0.0001 \
    --num_epochs 10 \
    --gpus 4 \
    --ntrain 1000 \
    --nval 1000
