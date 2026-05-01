from torchvision.models import resnet18
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from peft import LoraConfig, PeftMixedModel

import os
import copy
import random
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image


class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)



class TinyImageNetSubset(torch.utils.data.Dataset):
    """
    A subset of TinyImageNet for 10 classes.
    """
    def __init__(self, root, wnids, transform=None, split="train"):
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

        print(f"TinyImageNetSubset({split}): {len(self.samples)} images from {len(self.wnids)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


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

_selected_tiny_wnids = None

def load_dataset(name: str, train: bool, transform):
    global _selected_tiny_wnids
    name = name.lower()
    root = "."

    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=train, download=True, transform=transform), 10

    if name == "tinyimagenet":
        data_dir = ensure_tinyimagenet_downloaded()
        train_root = os.path.join(data_dir, "train")
        wnids_txt = "tiny10_wnids.txt"

        if _selected_tiny_wnids is None:
            if os.path.exists(wnids_txt):
                with open(wnids_txt, "r") as f:
                    _selected_tiny_wnids = [line.strip() for line in f if line.strip()]
            else:
                all_wnids = sorted(
                    d for d in os.listdir(train_root)
                    if os.path.isdir(os.path.join(train_root, d))
                )
                random.seed(0)
                _selected_tiny_wnids = random.sample(all_wnids, 10)
                with open(wnids_txt, "w") as f:
                    for wnid in _selected_tiny_wnids:
                        f.write(wnid + "\n")

        split = "train" if train else "test"
        return TinyImageNetSubset(data_dir, _selected_tiny_wnids, transform=transform, split=split), len(_selected_tiny_wnids)

    raise ValueError(f"Unsupported dataset: {name}")

def load_backbone(path: str, num_classes: int = 10) -> nn.Module:
    """
        Compatible with two saving methods:
          1) torch.save(model) -> directly load to get nn.Module
          2) torch.save(model.state_dict()) -> load to get dict, need to re-instantiate then load_state_dict
    """
    obj = torch.load(path, map_location=device)
    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = ResNet18Backbone(num_classes=num_classes)
        missing, unexpected = model.load_state_dict(obj, strict=False)
        if missing:
            print("[load_backbone] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print("[load_backbone] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    else:
        raise TypeError(f"Unknown checkpoint type: {type(obj)}")
    return model.to(device)

def validate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Accuracy: {acc:.2f}%')
    return acc

def kl_to_uniform(logits, num_classes: int):
    p = F.log_softmax(logits, dim=1)
    p_prob = p.exp()
    return (p_prob * p).sum(dim=1).mean() + math.log(num_classes)

transform_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]),    ])

def resnet_feature_activation_loss(model, x):
    m = model.resnet if hasattr(model, "resnet") else model

    h = m.conv1(x)
    h = m.bn1(h)
    h = m.relu(h)
    h = m.maxpool(h)

    h = m.layer1(h)
    h = m.layer2(h)
    h = m.layer3(h)
    h = m.layer4(h)

    h = m.avgpool(h)
    h = torch.flatten(h, 1)

    return (h ** 2).mean()

def calculate_lora_orthogonality_loss(model):
    ortho_loss = 0.0
    count = 0

    target_device = next(model.parameters()).device

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and "adapter_S" in module.lora_A and "adapter_B" in module.lora_A:
            AS = module.lora_A["adapter_S"].weight  # [r, in_features]
            BS = module.lora_B["adapter_S"].weight  # [out_features, r]

            AB = module.lora_A["adapter_B"].weight
            BB = module.lora_B["adapter_B"].weight

            WS = BS.view(BS.size(0), -1) @ AS.view(AS.size(0), -1)
            WB = BB.view(BB.size(0), -1) @ AB.view(AB.size(0), -1)

            mag_S = torch.norm(WS, p='fro')
            mag_B = torch.norm(WB, p='fro')

            if mag_S > 1e-6 and mag_B > 1e-6:
                inner_product = torch.sum(WS * WB)
                ortho_loss += torch.abs(inner_product) / (mag_S * mag_B)
                count += 1

    return ortho_loss / count if count > 0 else torch.tensor(0.0).to(target_device)


def print_update_status(model, target_adapter):
    print(f"\n>>> Switching to stage: training {target_adapter} <<<")

    active_params = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"    [Status] Updating parameter prefix: {set([n.split('.')[-1] for n in active_params])}")
    print(f"    [Status] Total number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


def print_trainable_parameters_MB(model):
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
    print(
        f"trainable params: {trainable_mb:.4f} MB "
        f"|| all params: {all_mb:.4f} MB "
        f"|| trainable%: {trainable_pct:.4f}"
    )


def forward_manual_joint(model, x, mode="S+B", verbose=False):
    with model.disable_adapter():
        base_logits = model(x)

    model.set_adapter("adapter_S")
    yS = model(x)
    delta_s = yS - base_logits

    model.set_adapter("adapter_B")
    yB = model(x)
    delta_b = yB - base_logits

    if verbose:
        strength_s = torch.norm(delta_s).item()
        strength_b = torch.norm(delta_b).item()
        print(f"[forward] mode: {mode} | S strength: {strength_s:.4f} | B strength: {strength_b:.4f}")

    if mode == "S+B":
        return base_logits + delta_s + delta_b
    elif mode == "B-S":
        return base_logits - delta_s + delta_b
    elif mode == "S":
        return yS
    elif mode == "B":
        return yB


    return base_logits

def evaluate_plain(model, loader):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total
    return acc


def prepare_alternating_training(model, train_target):
    """
    train_target: "adapter_S" or "adapter_B"
    """
    for name, param in model.named_parameters():
        if "lora_" in name:
            if train_target in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False


def load_resnet18_backbone(path: str, num_classes: int = 10) -> nn.Module:
    backbone = resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    checkpoint = torch.load(path, map_location=device)
    backbone.load_state_dict(checkpoint['model_state_dict'])
    backbone.eval()
    return backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
        print("Loading pretrained backbone...")
        base_model = load_resnet18_backbone(
            "./checkpoints/resnet18_cifar10_best.pth", num_classes=10
        )
        base_model.to(device)

        benign_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        benign_testloader = torch.utils.data.DataLoader(benign_testset, batch_size=128, shuffle=False, num_workers=2)

        suppression_dataset_name = "tinyimagenet"
        suppression_ds, num_classes_suppression = load_dataset(suppression_dataset_name, train=True, transform=transform_test)
        suppression_loader = DataLoader(suppression_ds, batch_size=64, shuffle=True)

        config_S = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["conv1", "conv2", "fc", "downsample.0"],
            lora_dropout=0,
            bias="none",
        )

        lora_model = PeftMixedModel(base_model, config_S, adapter_name="adapter_S")

        config_B = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["conv1", "conv2", "fc", "downsample.0"],
            lora_dropout=0,
            bias="none"
        )
        lora_model.add_adapter("adapter_B", config_B)

        lora_model.base_model.set_adapter(["adapter_S", "adapter_B"])

        criterion = nn.CrossEntropyLoss()
        params_S = [p for n, p in lora_model.named_parameters() if "adapter_S" in n]
        params_B = [p for n, p in lora_model.named_parameters() if "adapter_B" in n]

        lr_S = 1.89e-4

        optimizer_S = torch.optim.Adam(params_S, lr=lr_S)

        optimizer_B = torch.optim.Adam(params_B, lr=2e-4)

        trainset_cifar = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=transform_test)
        trainloader_cifar = torch.utils.data.DataLoader(trainset_cifar, batch_size=64, shuffle=True, num_workers=0)

        epochs = 20
        print("epochs:", epochs)
        lambda_act= 0.0
        for epoch in range(epochs):
            prepare_alternating_training(lora_model, "adapter_S")

            epoch_loss_s = 0.0
            num_batches_s = 0
            for imgs, _ in suppression_loader:
                optimizer_S.zero_grad()
                logits = lora_model(imgs.to(device))
                loss_act = resnet_feature_activation_loss(lora_model, imgs.to(device))
                loss = kl_to_uniform(logits, 10) + lambda_act * loss_act
                loss.backward()
                optimizer_S.step()
                epoch_loss_s += loss.item()
                num_batches_s += 1

            avg_loss_s = epoch_loss_s / num_batches_s if num_batches_s > 0 else 0.0

            prepare_alternating_training(lora_model, "adapter_B")

            epoch_loss_c = 0.0
            epoch_loss_orth = 0.0
            epoch_loss_b = 0.0
            num_batches_b = 0
            for imgs, labels in trainloader_cifar:
                optimizer_B.zero_grad()
                logits = lora_model(imgs.to(device))
                loss_c = criterion(logits, labels.to(device))
                loss_orth = calculate_lora_orthogonality_loss(lora_model)
                lossB = loss_c  + 0.5 * loss_orth
                lossB.backward()
                optimizer_B.step()
                epoch_loss_c += loss_c.item()
                epoch_loss_orth += loss_orth.item() if isinstance(loss_orth, torch.Tensor) else loss_orth
                epoch_loss_b += lossB.item()
                num_batches_b += 1

            avg_loss_c = epoch_loss_c / num_batches_b if num_batches_b > 0 else 0.0
            avg_loss_orth = epoch_loss_orth / num_batches_b if num_batches_b > 0 else 0.0
            avg_loss_b = epoch_loss_b / num_batches_b if num_batches_b > 0 else 0.0

            print(f"\n--- Epoch {epoch} Loss  ---")
            print(f"  [adapter_S]  KL-to-Uniform Loss:  {avg_loss_s:.6f}  (over {num_batches_s} batches)")
            print(f"  [adapter_B]  CE Loss:             {avg_loss_c:.6f}")
            print(f"  [adapter_B]  Orthogonality Loss:  {avg_loss_orth:.6f}")
            print(f"  [adapter_B]  Total Loss (lossB):  {avg_loss_b:.6f}  (over {num_batches_b} batches)")


        lora_model.eval()

        params_S = {n: p for n, p in lora_model.named_parameters() if "adapter_S" in n}
        params_B = {n: p for n, p in lora_model.named_parameters() if "adapter_B" in n}

        full_state = lora_model.state_dict()
        lora_S = {k: v for k, v in full_state.items() if "adapter_S" in k}
        lora_B = {k: v for k, v in full_state.items() if "adapter_B" in k}


        torch.save(params_S, "params_S.pth")
        torch.save(params_B, "params_B.pth")
        torch.save(lora_S, "lora_S.pth")
        torch.save(lora_B, "lora_B.pth")
        print("  saved: params_S.pth, params_B.pth, lora_S.pth, lora_B.pth")

        lora_model.base_model.set_adapter(["adapter_S"])

        model_S_only = copy.deepcopy(lora_model).base_model.merge_and_unload(["adapter_S"])


        torch.save(model_S_only, "backbone_with_lora_S_merged_resnet.pth")
        print(" saved (Base + S) -> backbone_with_lora_S_merged_resnet.pth")

        print("evaluated S model (CIFAR-10): ", end="")
        s_only_acc = evaluate_plain(model_S_only, benign_testloader)
        print(f" s_only Accuracy: {s_only_acc * 100:.2f}%")

        lora_model.base_model.set_adapter(["adapter_S", "adapter_B"])
        model_final = lora_model.base_model.merge_and_unload(["adapter_S","adapter_B"])

        s_B_acc = evaluate_plain(model_final, benign_testloader)
        print(f" PnP (Base+S+B) CIFAR10 Accuracy: {s_B_acc * 100:.2f}%")

        s_B_acc_t = evaluate_plain(model_final, suppression_loader)
        print(f" PnP (Base+S+B)  suppression_loader Accuracy: {s_B_acc_t * 100:.2f}%")


        torch.save(model_final, f"backbone_with_lora_SB_merged_resnet.pth")
        print(f" saved after merge backbone_with_lora_SB_merged_resnet.pth")
        print("lr_S == ",lr_S)

if __name__ == "__main__":
    main()
