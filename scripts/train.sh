#!/bin/bash

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
    python src/train.py model="$model" task_name="$task_name"
    echo -e "\n-----------------------------------\n"
done

