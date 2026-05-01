***

# 🛡️ Artifact for Anonymous Review

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-orange.svg)](https://pytorch.org/)

## 📖 Overview

This repository provides the implementation and experimental scripts for the submitted paper. The goal of this artifact is to help reviewers inspect the implementation and verify the main experimental pipeline. We provide the results for Setting No.1 reported in Table 1 of the paper.

In this setting, the model is first trained on the original domain and then evaluated under downstream fine-tuning on the suppression domain.

```text
Original domain: CIFAR-10
Suppression domain: TinyImageNet-10
Backbone: ResNet-18
Metric: Classification accuracy
```

## 🛠️ Installation

We recommend using Python 3.8+ and PyTorch 1.13+.

```bash
pip install torch torchvision peft matplotlib pillow numpy
```

## 📦 Data Preparation


| Dataset | Role | Source & Handling |
| :--- | :--- | :--- |
| **CIFAR-10** | Benign Task  | Automatically downloaded via `torchvision.datasets.CIFAR10` to `./data/`. |
| **TinyImageNet-200** | Suppression Task | Automatically downloaded from `http://cs231n.stanford.edu/tiny-imagenet-200.zip` and extracted. We randomly sample 10 classes to create a subset. |


## 🚀 Quick Start

The workflow consists of three steps: **Pretraining the Backbone**, **Training PnP-LoRA**, and **Evaluation**.

### 1️⃣ Pretrain Backbone Model
First, train a standard ResNet-18 on the source domain (CIFAR-10).

```bash
python pretrain_backbone.py \
  --dataset cifar10 \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.001 \
  --weight_decay 5e-4 \
  --save_path ./checkpoints/resnet18_cifar10_best.pth
```
*   **Output**: `./checkpoints/resnet18_cifar10_best.pth`

### 2️⃣ Training Stage
Run the main training script. It automatically loads the backbone, trains the Suppressor and Booster, and merges them to produce a protected model.

```bash
python train_pnp_lora.py
```
*   **Mechanism**:
    *   **Suppressor**: The Suppressor is trained on the suppression domain with randomized supervision. Its goal is to weaken task-relevant representations for the suppression domain, making downstream fine-tuning less effective.
    *   **Booster**: The Booster is trained on the original domain to preserve the model's original utility. It helps maintain performance on benign/original data while reducing interference with the Suppressor.
    *   **Merging**: After training, the Suppressor and Booster updates are merged into the backbone model through parameter-based merging. The released model keeps the same architecture and inference interface as the original backbone.
*   **Output**:
    *   `params_S.pth`, `params_B.pth` (weights of the Suppressor and Booster.)
    *   `backbone_with_lora_S_merged_resnet.pth` (merging the Suppressor into the backbone.)
    *   `backbone_with_lora_SB_merged_resnet.pth` (protected model with merging both the Suppressor and the Booster into the backbone.)

### 3️⃣ Evaluate Robustness
Evaluate the model's resistance to fine-tuning (S-domain) and compare it with baselines.

```bash
python evaluate_and_curve.py \
  --merged_sb_otr_path ./backbone_with_lora_SB_merged_resnet.pth \
  --original_path ./checkpoints/resnet18_cifar10_best.pth \
  --epochs 20 \
  --lr 0.56e-5 \
  --out_curve otr_compare.pt
```
*   **Output**:
    *   `otr_compare.pt`: Training logs.
    *   `otr_compare.png`: Fine-tuning accuracy comparison curve (PnP-LoRA vs. Original vs. Random).

## 🔬 Comparison with NTLBench (SOPHON & tNTL)

To benchmark against state-of-the-art NTL methods like **SOPHON** and **tNTL**, this project integrates with the unified benchmark [NTLBench](https://github.com/tmllab/NTLBench).

Since SOPHON and tNTL require specific training pipelines not included in this repository, follow these steps to generate their weights for comparison:

### 1. Clone NTLBench
```bash
git clone https://github.com/tmllab/NTLBench.git
cd NTLBench
pip install -r requirements.txt
```

### 2. Train SOPHON and tNTL Models
Run the NTLBench training script for the `cifar_tinyimagenet` domain pair:
```bash
# Train SOPHON
python NTL_pretrain.py --config config/cifar_tinyimagenet/pretrain.yml --task_name tSOPHON

# Train tNTL
python NTL_pretrain.py --config config/cifar_tinyimagenet/pretrain.yml --task_name tNTL
```

### 3. Link Checkpoints to This Project
Move the generated weights to this project's `./saved_models/` directory:
```bash
cp ./saved_models/SOPHON_cifar_tinyimagenet_resnet18_std.pth ../saved_models/
cp ./saved_models/tNTL_cifar_tinyimagenet_resnet18_std.pth ../saved_models/
```

### 4. Run Unified Evaluation
Update the `evaluate_otr.py` arguments to include these paths:
```bash
python evaluate_otr.py \
  --merged_sb_otr_path ./backbone_with_lora_SB_merged_resnet223.pth \
  --original_path ./checkpoints/resnet18_cifar10_best.pth \
  --sophon_path ./saved_models/SOPHON_cifar_tinyimagenet_resnet18_std.pth \
  --ntl_path ./saved_models/tNTL_cifar_tinyimagenet_resnet18_std.pth \
  --epochs 20 --lr 7e-6
```

## 📂 Directory Structure

```text
.
├── pretrain_backbone.py          # Script to pretrain the ResNet-18 backbone
├── train_pnp_lora.py             # Script to train PnP-LoRA adapters and merge
├── evaluate_otr.py               # Script to evaluate robustness and compare with SOTA
├── checkpoints/                  # Directory for backbone weights
├── saved_models/                 # Directory for SOPHON/tNTL comparison weights
├── data/                         # Auto-downloaded CIFAR-10 and TinyImageNet data
├── tiny10_wnids.txt              # List of selected TinyImageNet-10 classes
└── otr_compare.png               # Generated evaluation plots
```

