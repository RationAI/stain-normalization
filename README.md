# Stain Normalization
export MLFLOW_TRACKING_USERNAME=<YOUR_USERNAME>



[![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)](https://github.com/Lightning-AI/lightning)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/patch-camelyon/-/blob/master/LICENSE)


The main goal ...


## Getting Started

### Installation

Install [PDM](https://pdm.fming.dev/) package manager and install all the dependencies using the following command:
```bash
pdm install
```

### Preprocessing

```bash
export MLFLOW_TRACKING_USERNAME=<YOUR_USERNAME>
pdm run preprocessing/generate_tissue_mask.py
pdm run preprocessing/tiling.py
```

### Training

```bash
export MLFLOW_TRACKING_USERNAME=<YOUR_USERNAME>
pdm fit model/backbone=(vgg16|resnet18)
```

### Testing

```bash
export MLFLOW_TRACKING_USERNAME=<YOUR_USERNAME>
pdm test model/backbone=(vgg16|resnet18) 'checkpoint="<CHECKPOINT_PATH>"'
```
