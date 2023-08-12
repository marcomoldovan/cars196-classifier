#!/bin/bash

#SBATCH --job-name=coatnet_training
#SBATCH --partition=Nvidia2060          
#SBATCH --output=logs/coatnet.%j.out
#SBATCH --error=logs/coatnet.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Execute the model training
python cars196-classifier/src/train.py model="coatnet" task_name="coatnet"

# Deactivate the virtual environment
deactivate
