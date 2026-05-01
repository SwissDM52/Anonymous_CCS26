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
    *   `otr_compare.png`: Fine-tuning accuracy comparison curve.

## 📝 Note for Reviewers

This repository is prepared solely for double-blind review purposes. It contains all necessary scripts, configuration files, and instructions to reproduce Setting No.1 from the paper. After the paper is accepted, this repository will be updated and released as a public GitHub project for publication.


