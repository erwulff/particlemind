#!/bin/sh

# Walltime limit
#SBATCH -t 168:00:00
#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH -p genx
#SBATCH --cpus-per-task=64

# Job name
#SBATCH -J process_root2parquet

# Output and error logs
#SBATCH -o logs_slurm/log_%x_%j.out
#SBATCH -e logs_slurm/log_%x_%j.err


module --force purge; module load modules/2.2-20230808
module load slurm gcc cmake cuda/12.1.1 cudnn/8.9.2.26-12.x nccl openmpi apptainer

nvidia-smi
export PYTHONPATH=`pwd`
source ~/miniforge3/bin/activate mlpf
which python3
python3 --version

# Run the Python script to process ROOT files to Parquet
python data_processing/cld_root2parquet_parallel.py -i /mnt/ceph/users/ewulff/data/cld/ -o /mnt/ceph/users/ewulff/data/cld/processed/parquet
