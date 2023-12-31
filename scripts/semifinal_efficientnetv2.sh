#!/bin/bash

#SBATCH --job-name=efficientnetv2_training
#SBATCH --partition=Nvidia2060          
#SBATCH --output=logs/semifinal_efficientnetv2.%j.out
#SBATCH --error=logs/semifinal_efficientnetv2.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Execute the model training
python cars196-classifier/src/train.py model="efficientnetv2" task_name="semifinal_efficientnetv2" data.batch_size=32 trainer.max_epochs=10

# Deactivate the virtual environment
deactivate
