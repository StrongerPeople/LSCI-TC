#!/bin/bash
nvidia-smi
hostname
python run.py --precheckpoint './RS5M_Pretrain.pth' --task 'itr_rsitmd_vit' --dist "gpu0" --config 'configs/Retrieval_rsitmd_vit.yaml' --output_dir './checkpoints/rsitmd/'