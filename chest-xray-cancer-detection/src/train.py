"""
Training script for Chest X-Ray Classification.

Usage:
    python src/train.py
    python src/train.py --epochs 30 --batch_size 32 --lr 0.001
    python src/train.py --data_dir data/chest_xray --model_path models/best_model.pth
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dotenv import load_dotenv
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloaders
from src.model import ChestXRayModel, save_model

load_dotenv()


# ── Config ─────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_dir":   os.getenv("DATA_DIR", "data/chest_xray"),
    "model_path": os.getenv("MODEL_SAVE_PATH", "models/best_model.pth"),
    "epochs":     20,
    "batch_size": 32,
    "lr":         1e-3,
    "weight_decay": 1e-4,
    "warmup_epochs": 3,    # freeze backbone for first N epochs
    "num_workers": 4,
    "seed": 42,
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU — training will be slow")
    return device


# ── Train / Validate ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = total_correct = total_samples = 0

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss     += loss.item() * len(labels)
        total_correct  += (preds == labels).sum().item()
        total_samples  += len(labels)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        total_loss    += loss.item() * len(labels)
        total_correct += (preds == labels).sum().item()
        total_samples += len(labels)

    return total_loss / total_samples, total_correct / total_samples


# ── Main Training Loop ─────────────────────────────────────────────────────────
def train(config: dict):
    set_seed(config["seed"])
    device = get_device()

    print("\nLoading dataset...")
    loaders, datasets = get_dataloaders(
        config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    print("\nBuilding model...")
    model = ChestXRayModel(pretrained=True).to(device)
    params = model.count_parameters()
    print(f"   Total params: {params['total']:,} | Trainable: {params['trainable']:,}")

    # Class-weighted loss to handle imbalance
    class_weights = datasets["train"].get_class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\n  Starting training for {config['epochs']} epochs")
    print(f"   Warmup (head only): {config['warmup_epochs']} epochs")
    print("=" * 60)

    for epoch in range(1, config["epochs"] + 1):
        # Warmup phase: only train classifier head
        if epoch == 1:
            model.freeze_backbone()
        elif epoch == config["warmup_epochs"] + 1:
            model.unfreeze_backbone()
            print(f"   Epoch {epoch}: Full fine-tuning started")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device, epoch)
        val_loss,   val_acc   = validate(model, loaders["val"], criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Save best model
        star = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, config["model_path"], extra_info={"val_acc": val_acc, "epoch": epoch})
            star = " BEST"

        print(
            f"Epoch {epoch:>3}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s{star}"
        )

    print("=" * 60)
    print(f"\n Training complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"   Model saved to: {config['model_path']}")
    print("\nNext step: python src/evaluate.py")
    return history


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Chest X-Ray classifier")
    parser.add_argument("--data_dir",      default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--model_path",    default=DEFAULT_CONFIG["model_path"])
    parser.add_argument("--epochs",        type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",    type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",            type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--weight_decay",  type=float, default=DEFAULT_CONFIG["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int,   default=DEFAULT_CONFIG["warmup_epochs"])
    parser.add_argument("--num_workers",   type=int,   default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--seed",          type=int,   default=DEFAULT_CONFIG["seed"])
    return vars(parser.parse_args())


if __name__ == "__main__":
    config = parse_args()
    print("Chest X-Ray Cancer Detection — Training")
    print("Config:", config)
    train(config)
