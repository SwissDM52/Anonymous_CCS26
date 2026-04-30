***

# 🛡️ PnP-LoRA: Plug-and-Play Low-Rank Adaptation for Non-Transferable Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.13-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview

This repository provides an implementation of **Plug-and-Play Low-Rank Adaptation (PnP-LoRA)** for **Non-Transferable Learning (NTL)**.

To prevent unauthorized transfer learning while maintaining utility on the original task, we employ a dual-adapter architecture:
1.  **Adapter-S (Suppression)**: Trained on out-of-distribution data (S-domain) to disrupt transferability.
2.  **Adapter-B (Benign)**: Trained on the original task data (B-domain) to preserve performance.

An **Orthogonality Constraint** is introduced to minimize interference between the two adapters. The final model can be merged into the backbone and distributed as a standard PyTorch model, requiring no architectural changes for the end-user.

## ✨ Key Features

-   **Decoupled Dual-Adapter Architecture**: Independent optimization for suppression and benign tasks.
-   **LoRA Orthogonality Regularization**: Ensures suppression operations do not degrade original task representations.
-   **Plug-and-Play Compatibility**: Supports `merge_and_unload`, outputting standard models compatible with any inference framework.
-   **Automated Data Pipeline**: Built-in logic for downloading CIFAR-10 and generating a deterministic TinyImageNet-10 subset (reproducible across environments).
-   **Unified Evaluation**: Native support for comparing against SOTA NTL methods like **SOPHON** and **tNTL**.

## 🛠️ Installation

We recommend using Python 3.8+ and PyTorch 1.13+.

```bash
pip install torch torchvision peft matplotlib pillow numpy
```

## 📦 Data Preparation

This project automates the entire data preparation process. **No manual dataset preparation is required.**

| Dataset | Role | Source & Handling |
| :--- | :--- | :--- |
| **CIFAR-10** | Benign Task (B-Domain) | Automatically downloaded via `torchvision.datasets.CIFAR10` to `./data/`. Used to train Adapter-B. |
| **TinyImageNet-200** | Suppression Task (S-Domain) | Automatically downloaded from `http://cs231n.stanford.edu/tiny-imagenet-200.zip` and extracted. The script **randomly samples 10 classes** (fixed seed `seed=0`) to create a subset. |
| **`tiny10_wnids.txt`** | Reproducibility | The list of sampled 10 classes is saved to this file upon first run. Subsequent runs load this file to ensure the exact same subset is used, guaranteeing experimental reproducibility. |

## 🚀 Quick Start

The workflow consists of three steps: **Pretraining the Backbone**, **Training PnP-LoRA**, and **Evaluation**.

### 1️⃣ Pretrain Backbone Model
First, train a standard ResNet-18 on the source domain (CIFAR-10).

```bash
python pretrain_backbone.py \
  --dataset cifar10 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.001 \
  --weight_decay 5e-4 \
  --save_path ./checkpoints/resnet18_cifar10_best.pth
```
*   **Output**: `./checkpoints/resnet18_cifar10_best.pth`

### 2️⃣ Train PnP-LoRA Adapters
Run the main training script. It automatically loads the backbone, trains Adapter-S (via KL-Divergence) and Adapter-B (via Cross-Entropy + Orthogonality Loss), and merges them.

```bash
python train_pnp_lora.py
```
*   **Mechanism**:
    *   **Adapter-S**: Optimized on TinyImageNet-10 to flatten output distribution.
    *   **Adapter-B**: Optimized on CIFAR-10 with an orthogonality penalty (`0.5 * loss_orth`) to ensure it doesn't overlap with Adapter-S.
*   **Output**:
    *   `params_S.pth`, `params_B.pth` (LoRA weights)
    *   `backbone_with_lora_S_merged_resnet.pth` (Suppression-only model)
    *   `backbone_with_lora_SB_merged_resnet.pth` (Final PnP-LoRA model)

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

## 📝 Citation

If you use this code in your research, please consider citing:
```bibtex
@misc{pnp_lora_ntl_2025,
  author = {Your Name},
  title = {PnP-LoRA: Plug-and-Play Low-Rank Adaptation for Non-Transferable Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/your-repo/pnp-lora-ntl}
}
```

If you use the NTLBench comparison results, please cite their paper:
```bibtex
@article{hong2025toward,
  title={Toward Robust Non-Transferable Learning: A Survey and Benchmark},
  author={Hong, Ziming and Xiang, Yongli and Liu, Tongliang},
  journal={arXiv preprint arXiv:2502.13593},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
