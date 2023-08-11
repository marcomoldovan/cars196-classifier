#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

python src/train.py model=coatnet

python src/train.py model=densenet

python src/train.py model=efficientnetv2

python src/train.py model=mobilevit

python src/train.py model=resnet50

python src/train.py model=swintransformer



