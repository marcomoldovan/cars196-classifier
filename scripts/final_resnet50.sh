#!/bin/bash

#SBATCH --job-name=resnet50_training
#SBATCH --partition=Nvidia2060          
#SBATCH --output=logs/resnet50.%j.out
#SBATCH --error=logs/resnet50.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Execute the model training
python cars196-classifier/src/train.py model="resnet50" task_name="final_resnet50" data.batch_size=32 trainer.max_epochs=100

# Deactivate the virtual environment
deactivate
