import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import resnet18


def get_cifar10_loaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 16
) -> Tuple[DataLoader, DataLoader]:
    image_size = 64
    transform_train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomCrop(image_size, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])

    transform_test = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

    train_set = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def get_mnist_loaders(
    data_root: str,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            _, preds = logits.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain backbone model for Plug-and-Play LoRA experiments"
    )
    parser.add_argument("--data_root", type=str, default="./data",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "mnist"],
                        help="which dataset to pretrain on")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--save_path", type=str,
                        default="./checkpoints/resnet18_cifar10_best.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.dataset == "cifar10":
        in_channels = 3
        num_classes = 10
        train_loader, test_loader = get_cifar10_loaders(
            args.data_root, args.batch_size
        )
    else:  # mnist
        in_channels = 1
        num_classes = 10
        train_loader, test_loader = get_mnist_loaders(
            args.data_root, args.batch_size
        )

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 80, 90],
        gamma=0.2,
    )

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch,
            }, args.save_path)
            print(f"  ==> New best model saved to {args.save_path} "
                  f"(Acc: {best_acc*100:.2f}%)")

    print(f"Training finished. Best test accuracy: {best_acc*100:.2f}%")
    print(f"Pretrained backbone saved at: {args.save_path}")


if __name__ == "__main__":
    main()
