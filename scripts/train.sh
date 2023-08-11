#!/bin/bash
# Schedule execution of all models

python src/train.py model=coatnet

python src/train.py model=densenet

python src/train.py model=efficientnetv2

python src/train.py model=mobilevit

python src/train.py model=resnet50

python src/train.py model=swintransformer



