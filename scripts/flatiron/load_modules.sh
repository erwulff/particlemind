#!/bin/bash

module --force purge; module load modules/2.2-20230808
module load slurm gcc cmake cuda/12.1.1 cudnn/8.9.2.26-12.x nccl openmpi apptainer

nvidia-smi
export PYTHONPATH=`pwd`
source ~/miniforge3/bin/activate mind
which python3
python3 --version
