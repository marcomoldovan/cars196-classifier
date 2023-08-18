______________________________________________________________________

<div align="center">

# Stanford Cars Classification

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
<!-- [![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020) -->

</div>

## Description

Image Classification of Car Models using the Stanford Cars Dataset

## Installation

#### Poetry
  
```bash
# install poetry and add poetry to path
curl -sSL https://install.python-poetry.org | python3 -
~/.local/share/pypoetry/venv/bin/poetry

# clone project
git clone https://github.com/marcomoldovan/cars196-classifier
cd cars196-classifier

# install dependencies
poetry install

# activate environment
source $(poetry env info --path)/bin/activate
```

#### Pip

```bash
# clone project
git clone https://github.com/marcomoldovan/cars196-classifier
cd cars196-classifier

# create virtual environment and activate it
python -m venv .venv
source .venv/bin/activate

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## How to run

Download datasets:

https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset

https://www.kaggle.com/datasets/abdelrahmant11/standford-cars-dataset-meta

**Folder structure:**
```
data:

--> stanford-cars-dataset
 
--> stanford-cars-dataset-meta
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
