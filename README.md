# MLPMixer

This repository contains an unofficial implementation of the MLP-Mixer architecture using PyTorch. The goal of this project is to explore the MLP-Mixer architecture, rather than focusing on achieving SOTA performance.

MLP-Mixer is a vision model introduced by Tolstikhin et al. in the paper [MLPMixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf). It replaces convolutional layers and self-attention with a simple MLP-based approach, demonstrating competitive performance on image classification tasks.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vtzouras/MLP-Mixer
cd mlp-mixer-pytorch
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To train the model on CIFAR-10, run the following command:
```bash
python train.py
```

To change the configuration, modify the `configs/default.yaml` file.

## Example Results

The model achieves an accuracy of $86.72 \%$ on the CIFAR-10 dataset.



