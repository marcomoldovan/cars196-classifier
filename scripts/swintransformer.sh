#!/bin/bash

#SBATCH --job-name=swintransformer_training
#SBATCH --partition=Nvidia2060          
#SBATCH --output=logs/swintransformer.%j.out
#SBATCH --error=logs/swintransformer.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Execute the model training
python cars196-classifier/src/train.py model="swintransformer" task_name="swintransformer"

# Deactivate the virtual environment
deactivate
