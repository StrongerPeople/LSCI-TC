# LSCI-TC: Semantic Consistence Interaction with Calibration Loss for Remote Sensing Image-Text Retrieval

[![Paper](https://img.shields.io/badge/Paper-ArXiv%20Link-red)](https://ieeexplore.ieee.org/document/11316677) [![GitHub Stars](https://img.shields.io/github/stars/StrongerPeople/LSCI-TC)](https://github.com/StrongerPeople/LSCI-TC) [![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Project Introduction

This is the source code implementation of the paper "Semantic Consistency Interaction With Calibration Loss for Remote Sensing Image–Text Retrieval". The project focuses on remote sensing image-text retrieval tasks, improving retrieval performance through Local Semantic Consistency Interaction (LSCI) and Task-specific Calibration Loss (TC).

## Key Features

- **Local Semantic Consistency Interaction (LSCI)**: Enhances semantic consistency between image and text modalities
- **Task-specific Calibration Loss (TC)**: Optimizes the calibration mechanism for retrieval tasks
- **Multi-dataset Support**: Supports RSICD, RSITMD, and UCM datasets
- **Based on GeoCLIP**: Uses pre-trained GeoCLIP models for feature extraction

## Environment Requirements

- Python 3.8+
- PyTorch 2.0.1+
- CUDA 11.8+ (recommended)
- Ubuntu 22.04 or WSL2.0

## Installation

1. Clone the repository:

```bash
git clone https://github.com/StrongerPeople/LSCI-TC.git
# the RS5M pretrain checkpoint
git clone https://huggingface.co/Tom1long/LSCI_TC
cd LSCI-TC
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

The project uses three remote sensing image-text retrieval datasets:

### RSICD Dataset

- Download the RSICD dataset and place it in the `data/rsicd_images/` directory
- Data files: `data/finetune/rsicd_train.json`, `rsicd_val.json`, `rsicd_test.json`

### RSITMD Dataset

- Download the RSITMD dataset and place it in the `data/rsitmd_images/` directory
- Data files: `data/finetune/rsitmd_train.json`, `rsitmd_val.json`, `rsitmd_test.json`

### UCM Dataset

- Download the UCM dataset and place it in the `data/UCM/` directory
- Data files: `data/finetune/ucm_train_data.json`, `ucm_val_data.json`, `ucm_test_data.json`

## Usage

### Training

Use the provided training script:

```bash
# Training on RSICD dataset
sh train.sh

```

### Evaluation

```bash
# Evaluation script
bash eval.sh

```

## Configuration Files

Configuration files are located in the `configs/` directory:

- `Retrieval_rsicd_vit.yaml`: Configuration for RSICD dataset
- `Retrieval_rsitmd_vit.yaml`: Configuration for RSITMD dataset

## Pre-trained Models

The project provides pre-trained weights `RS5M_Pretrain.pth` in the huggingface repository ([https://huggingface.co/Tom1long/LSCI_TC](https://huggingface.co/Tom1long/LSCI_TC)), which can be used for model initialization or fine-tuning., which can be used for model initialization or fine-tuning.

## Project Structure

```
LSCI-TC/
├── configs/                 # Configuration files
├── data/                    # Datasets
│   ├── finetune/           # Fine-tuning data
│   ├── rsicd_images/       # RSICD images
│   ├── rsitmd_images/      # RSITMD images
│   └── UCM/                # UCM dataset
├── dataset/                # Dataset processing code
├── models/                 # Model definitions
│   └── open_clip/          # OpenCLIP related code
├── utils/                  # Utility functions
├── README.md               # Project documentation
├── requirements.txt        # Dependencies list
├── run.py                  # Main training script
├── evaluate_retrieval.py   # Evaluation script
└── Retrieval.py            # Inference script
```

## Experimental Results

Experimental results on different datasets:


| Dataset | Recall@1 | Recall@5 | Recall@10 |
| ------- | -------- | -------- | --------- |
| RSICD   | -        | -        | -         |
| RSITMD  | -        | -        | -         |
| UCM     | -        | -        | -         |

*Note: Please refer to the paper for detailed experimental results*

## Citation

If this project helps your research, please cite our paper:

```
@ARTICLE{11316677,
  author={Xu, Jinlong and Ge, Yun and Zeng, Yan and Liu, Huyang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Semantic Consistency Interaction With Calibration Loss for Remote Sensing Image–Text Retrieval}, 
  year={2026},
  volume={64},
  number={},
  pages={1-17},
  keywords={Calibration;Remote sensing;Semantics;Visualization;Adaptation models;Training;Accuracy;Text to image;Linguistics;Image retrieval;Local semantic consistency interaction (LSCI);modality interaction;remote sensing image–text retrieval (RSITR);task-specific calibration loss (TC)},
  doi={10.1109/TGRS.2025.3649046}}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or suggestions, please contact us through GitHub Issues.
