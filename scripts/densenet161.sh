#!/bin/bash

#SBATCH --job-name=densenet161_training
#SBATCH --partition=Nvidia2060          
#SBATCH --output=logs/densenet161.%j.out
#SBATCH --error=logs/densenet161.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Execute the model training
python cars196-classifier/src/train.py model="densenet161" task_name="densenet161"

# Deactivate the virtual environment
deactivate
