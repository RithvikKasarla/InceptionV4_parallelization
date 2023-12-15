#!/bin/bash

#SBATCH --job-name=pytorch_training
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem=64GB 
#SBATCH --ntasks-per-node=2

module load gcc/6.3.0
module load python/3.8.6
module load cuda/8.0


cd /home/rxk654/Documents/InceptionV4_parallelization

source ./venv/bin/activate

python PytorchLightningBaseline.py

deactivate