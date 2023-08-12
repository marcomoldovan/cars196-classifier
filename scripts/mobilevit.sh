#!/bin/bash

#SBATCH --job-name=mobilevit_training
#SBATCH --partition=Nvidia2060          
#SBATCH --output=logs/mobilevit.%j.out
#SBATCH --error=logs/mobilevit.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Execute the model training
python cars196-classifier/src/train.py model="mobilevit" task_name="mobilevit"

# Deactivate the virtual environment
deactivate
