#!/bin/bash
# Schedule execution of all models

python src/train.py model=coatnet task_name=coatnet

python src/train.py model=densenet161 task_name=densenet161

python src/train.py model=efficientnetv2 task_name=efficientnetv2

python src/train.py model=mobilevit task_name=mobilevit

python src/train.py model=resnet50 task_name=resnet50

python src/train.py model=swintransformer task_name=swintransformer



