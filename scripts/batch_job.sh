#!/bin/bash

#SBATCH --job-name=stanford_cars_training
#SBATCH --partition=Nvidia2060
#SBATCH --nodes=6              # Requesting 6 nodes (and thus 6 GPUs).
#SBATCH --ntasks-per-node=1    # One task for each node's GPU.
#SBATCH --mem=64G              
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err

# Activate the virtual environment
source cars196-classifier/.venv/bin/activate

# Define the models and task names
declare -A commands=(
    ["coatnet"]="coatnet"
    ["densenet161"]="densenet161"
    ["efficientnetv2"]="efficientnetv2"
    ["mobilevit"]="mobilevit"
    ["resnet50"]="resnet50"
    ["swintransformer"]="swintransformer"
)

# Loop through and execute each command
for model in "${!commands[@]}"; do
    task_name="${commands[$model]}"
    echo "Executing for model: $model and task_name: $task_name"

    # Use srun for parallel execution across nodes and GPUs
    srun -l python cars196-classifier/src/train.py model="$model" task_name="$task_name" &

    wait # Wait for the above command to complete before moving to the next model
    echo -e "\n-----------------------------------\n"
done

# Optionally, deactivate the virtual environment after the job.
deactivate
