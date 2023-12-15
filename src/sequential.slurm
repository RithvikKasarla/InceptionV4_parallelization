#!/bin/bash

#SBATCH --job-name=sequential_job        # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --gpus=1                         # Number of GPUs
#SBATCH --ntasks=2                       # Number of tasks (processes)
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH -A sxg125_csds438

# Load any modules or environment setup required

module load gcc/6.3.0
module load python/3.8.6
module load cuda/8.0

# Activate virtual environment if you have one
source /home/rxk654/Documents/InceptionV4_parallelization/venv/bin/activate

# Run the Python script
srun python /home/rxk654/Documents/InceptionV4_parallelization/sequential.py

# Deactivate virtual environment
deactivate
