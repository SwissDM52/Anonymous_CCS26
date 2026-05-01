import os
import random
import argparse
import math
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18


def build_transform(img_size: int = 32):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class TinyImageNetSubset(torch.utils.data.Dataset):
    def __init__(self, root, wnids, transform=None):
        self.root = root
        self.wnids = wnids
        self.transform = transform
        self.samples = []

        train_root = os.path.join(root, "train")
        for idx, wnid in enumerate(wnids):
            img_dir = os.path.join(train_root, wnid, "images")
            if not os.path.isdir(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(img_dir, fname), idx))

        print(f"TinyImageNetSubset: {len(self.samples)} images from {len(self.wnids)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


_selected_tiny_wnids = None


def ensure_tinyimagenet_downloaded():
    data_dir = "./data/tiny-imagenet-200"
    if not os.path.exists(data_dir):
        import urllib.request
        import zipfile

        url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
        print("Downloading TinyImageNet from:", url)
        urllib.request.urlretrieve(url, "tiny.zip")
        with zipfile.ZipFile("tiny.zip", "r") as zf:
            zf.extractall(".")
        print("Extracted TinyImageNet to ./tiny-imagenet-200")
    return "./tiny-imagenet-200"


def load_tinyimagenet10(transform_train, transform_test=None,
                        seed_for_wnids: int = 0,
                        train_ratio: float = 0.8,
                        split_seed: int = 42):
    global _selected_tiny_wnids

    if transform_test is None:
        transform_test = transform_train

    data_dir = ensure_tinyimagenet_downloaded()
    train_root = os.path.join(data_dir, "train")
    wnids_txt = "tiny10_wnids.txt"

    if _selected_tiny_wnids is None:
        if os.path.exists(wnids_txt):
            with open(wnids_txt, "r") as f:
                _selected_tiny_wnids = [line.strip() for line in f if line.strip()]
            print("Loaded TinyImageNet-10 wnids:", _selected_tiny_wnids)
        else:
            all_wnids = sorted(
                d for d in os.listdir(train_root)
                if os.path.isdir(os.path.join(train_root, d))
            )
            rnd = random.Random(seed_for_wnids)
            _selected_tiny_wnids = rnd.sample(all_wnids, 10)
            with open(wnids_txt, "w") as f:
                for wnid in _selected_tiny_wnids:
                    f.write(wnid + "\n")
            print("Sampled wnids:", _selected_tiny_wnids)

    full_ds = TinyImageNetSubset(data_dir, _selected_tiny_wnids, transform=None)

    from collections import defaultdict
    class_to_indices = defaultdict(list)
    for i in range(len(full_ds)):
        _, label = full_ds.samples[i]
        class_to_indices[label].append(i)

    rng = random.Random(split_seed)
    train_indices = []
    test_indices = []
    for label in sorted(class_to_indices.keys()):
        indices = class_to_indices[label]
        rng.shuffle(indices)
        n_train = int(len(indices) * train_ratio)
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])

    print(f"Train/Test split: {len(train_indices)} train, {len(test_indices)} test "
          f"(ratio={train_ratio}, seed={split_seed})")

    train_ds = _SubsetWithTransform(full_ds, train_indices, transform_train)
    test_ds = _SubsetWithTransform(full_ds, test_indices, transform_test)

    return train_ds, test_ds, len(_selected_tiny_wnids)


class _SubsetWithTransform(torch.utils.data.Dataset):
    """Like torch.utils.data.Subset, but allows overriding the transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[self.indices[idx]]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def load_cifar10_test(transform, data_root: str = "./data"):
    from torchvision.datasets import CIFAR10
    test_ds = CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )
    print(f"Loaded CIFAR-10 test set: {len(test_ds)} images, 10 classes")
    return test_ds

def print_trainable_parameters_MB(model, name: str = ""):
    trainable_params = 0
    trainable_bytes = 0
    all_params = 0
    all_bytes = 0
    for _, p in model.named_parameters():
        n = p.numel()
        s = n * p.element_size()
        all_params += n
        all_bytes += s
        if p.requires_grad:
            trainable_params += n
            trainable_bytes += s
    trainable_mb = trainable_bytes / (1024 ** 2)
    all_mb = all_bytes / (1024 ** 2)
    trainable_pct = 100.0 * trainable_params / all_params if all_params > 0 else 0
    prefix = f"[{name}] " if name else ""
    print(
        f"{prefix}trainable params: {trainable_mb:.4f} MB "
        f"|| all params: {all_mb:.4f} MB "
        f"|| trainable%: {trainable_pct:.4f}"
    )
    return trainable_mb, all_mb


def load_model_maybe_state_dict(path: str, name: str, device: torch.device, num_classes: int,
                                img_size: int) -> nn.Module:
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        m = obj
        print(f"[OK] Loaded {name}: nn.Module {type(m)}")
    elif isinstance(obj, dict):
        print(f"[WARN] Loaded {name}: state_dict; rebuilding resnet18 then load_state_dict(strict=False)")
        m = resnet18(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        missing, unexpected = m.load_state_dict(obj, strict=False)
        if missing:
            print(f"[{name}] missing keys head:", missing[:5])
        if unexpected:
            print(f"[{name}] unexpected keys head:", unexpected[:5])
    else:
        raise TypeError(f"{name}: unknown checkpoint type: {type(obj)}")

    m = m.to(device)
    for p in m.parameters():
        p.requires_grad = True
    return m


def load_resnet18_backbone(path: str, num_classes: int = 10) -> nn.Module:
    backbone = resnet18(pretrained=False)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    checkpoint = torch.load(path, weights_only=True)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    backbone.eval()
    return backbone


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device, scheduler=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


def build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    def lr_lambda(step: int):
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    parser = argparse.ArgumentParser(
        description="Compare 6 models on S-domain (TinyImageNet-10) for ResNet18"
    )
    parser.add_argument("--merged_sb_path", type=str, default="./backbone_with_lora_SB_merged_resnet.pth")
    parser.add_argument("--sophon_path", type=str, default="./saved_models/SOPHON_cifar_tinyimagenet_resnet18_std.pth")
    parser.add_argument("--ntl_path", type=str, default="./saved_models/tNTL_cifar_tinyimagenet_resnet18_std.pth")
    parser.add_argument("--original_path", type=str, default="./checkpoints/resnet18_cifar10_best.pth")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--lr", type=float, default=0.56e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_curve", type=str, default="pnp_compare.pt")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_ratio", type=float, default=0.8, help="fraction of data for training")
    parser.add_argument("--split_seed", type=int, default=42, help="seed for train/test split")

    parser.add_argument("--reset_head", action="store_true")
    parser.add_argument("--use_cosine", action="store_true")
    parser.add_argument("--warmup_epochs", type=float, default=0.5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    transform_train = build_transform(args.img_size)
    transform_test = build_transform(args.img_size)

    suppression_train_ds, suppression_test_ds, num_classes = load_tinyimagenet10(
        transform_train=transform_train,
        transform_test=transform_test,
        seed_for_wnids=0,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )

    suppression_train_loader = DataLoader(
        suppression_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    suppression_test_loader = DataLoader(
        suppression_test_ds, batch_size=32, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    cifar10_transform = build_transform(args.img_size)
    benign_test_ds = load_cifar10_test(transform=cifar10_transform, data_root="./data")
    benign_test_loader = DataLoader(
        benign_test_ds, batch_size=32, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Loaded S-domain train dataset: {len(suppression_train_ds)} images, classes={num_classes}")
    print(f"Loaded S-domain test  dataset: {len(suppression_test_ds)} images, classes={num_classes}")

    print(f"Loading merged S+B model otr_yes: {args.merged_sb_path}")
    model_merged_sb_otr_yes = load_model_maybe_state_dict(args.merged_sb_path, "merged_SB", device, num_classes, args.img_size)
    for p in model_merged_sb_otr_yes.parameters():
        p.requires_grad = True

    print(f"Loading Original Model: {args.original_path}")
    model_original = load_resnet18_backbone(args.original_path, num_classes=num_classes)
    for p in model_original.parameters():
        p.requires_grad = True
    model_original.to(device)

    print("Creating random model...")
    model_random = resnet18(pretrained=False)
    model_random.fc = nn.Linear(model_random.fc.in_features, num_classes)
    for p in model_random.parameters():
        p.requires_grad = True
    model_random.to(device)

    models_dict = {
        "Random": model_random,
        "Original": model_original,
        "PnP-LoRA": model_merged_sb_otr_yes,
    }

    optimizers = {}
    schedulers = {}

    steps_per_epoch = len(suppression_train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    for name, m in models_dict.items():
        opt = torch.optim.AdamW(
            m.parameters(),
            lr=args.lr,
            betas=tuple(args.betas),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        optimizers[name] = opt

        if args.use_cosine:
            schedulers[name] = build_warmup_cosine_scheduler(
                opt, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=args.min_lr_ratio
            )
        else:
            schedulers[name] = None

    criterion = nn.CrossEntropyLoss()

    curves = {
        name: {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
        for name in models_dict
    }

    curves2 = {
        name: {"train_acc": [], "test_acc": [], "train_loss": [], "test_loss": []}
        for name in models_dict
    }

    def record_epoch0_initial2():
        print("===== Epoch 0 (Before Training) Evaluation on CIFAR-10 =====")
        for name, model in models_dict.items():
            cifar_loss, cifar_acc = evaluate(model, benign_test_loader, criterion, device)
            tr_loss, tr_acc = evaluate(model, suppression_train_loader, criterion, device)

            curves2[name]["train_loss"].append(tr_loss)
            curves2[name]["train_acc"].append(tr_acc)
            curves2[name]["test_loss"].append(cifar_loss)
            curves2[name]["test_acc"].append(cifar_acc)
            print(f"[{name}] CIFAR-10_test_acc={cifar_acc:.4f}, TinyIN10_train_acc={tr_acc:.4f}")
        print("=============================================================\n")

    record_epoch0_initial2()

    def record_epoch0_initial():
        print("===== Epoch 0 (Before Training) Evaluation (S-domain only) =====")
        for name, model in models_dict.items():
            tr_loss, tr_acc = evaluate(model, suppression_train_loader, criterion, device)
            te_loss, te_acc = evaluate(model, suppression_test_loader, criterion, device)
            curves[name]["train_loss"].append(tr_loss)
            curves[name]["train_acc"].append(tr_acc)
            curves[name]["test_loss"].append(te_loss)
            curves[name]["test_acc"].append(te_acc)
            print(f"[{name}] test_acc={te_acc:.4f}, train_acc={tr_acc:.4f}")
        print("==============================================================\n")

    record_epoch0_initial()

    print("\nStart training ALL MODELS on S-domain (TinyImageNet-10)...\n")

    for epoch in range(1, args.epochs + 1):
        for name, model in models_dict.items():
            tr_loss, tr_acc = train_one_epoch(
                model, suppression_train_loader, optimizers[name], criterion, device, scheduler=schedulers[name]
            )
            te_loss, te_acc = evaluate(model, suppression_test_loader, criterion, device)

            curves[name]["train_loss"].append(tr_loss)
            curves[name]["train_acc"].append(tr_acc)
            curves[name]["test_loss"].append(te_loss)
            curves[name]["test_acc"].append(te_acc)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"PnP-LoRA={curves['PnP-LoRA']['test_acc'][-1]:.3f}, "
            f"Original ModelL={curves['Original']['test_acc'][-1]:.3f}, "
            f"Random={curves['Random']['test_acc'][-1]:.3f}"
        )

    torch.save(curves, args.out_curve)
    print(f"\nSaved curves to: {args.out_curve}")

    plt.figure(figsize=(10, 6))
    xs = list(range(0, args.epochs + 1))

    BLUE = '#6BAED6'
    RED = '#FC9272'
    ORANGE = '#FDCC8A'
    PURPLE = '#C6A1D7'
    PINK = '#F4A7B9'
    GREEN = '#A1D99B'

    plt.plot(xs, curves["Random"]["test_acc"],
             color=ORANGE, linestyle='-', linewidth=3, label="Random")
    plt.plot(xs, curves["Original"]["test_acc"],
             color=GREEN, linestyle='-', linewidth=3, label="Original")
    plt.plot(xs, curves["PnP-LoRA"]["test_acc"],
             color=PURPLE, linestyle='-', linewidth=3, label="PnP-LoRA")

    plt.title("")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("otr_compare.png", dpi=300)
    plt.show()
    print("\nSaved figure: otr_compare.png")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()
