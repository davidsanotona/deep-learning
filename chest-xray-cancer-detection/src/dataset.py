"""
Dataset loader for Chest X-Ray classification.
Handles loading, augmentation, and class balancing.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ── Class configuration ────────────────────────────────────────────────────────
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_SIZE = 224  # EfficientNetB3 default


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split: str = "train") -> transforms.Compose:
    """Return augmentation pipeline for given split."""
    if split == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ── Dataset ────────────────────────────────────────────────────────────────────
class ChestXRayDataset(Dataset):
    """PyTorch Dataset for Chest X-Ray images."""

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        """
        Args:
            root_dir: Path to chest_xray/ folder (contains train/, val/, test/)
            split: One of 'train', 'val', 'test'
            transform: Optional torchvision transforms
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform or get_transforms(split)
        self.samples = []  # list of (image_path, label)
        self.class_counts = {}

        self._load_samples()

    def _load_samples(self):
        """Scan directory and collect all image paths with labels."""
        valid_extensions = {".jpg", ".jpeg", ".png"}

        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} not found, skipping.")
                continue

            label = CLASS_TO_IDX[class_name]
            files = [
                f for f in class_dir.iterdir()
                if f.suffix.lower() in valid_extensions
            ]
            self.class_counts[class_name] = len(files)
            for f in files:
                self.samples.append((str(f), label))

        print(f"[{self.split.upper()}] Loaded {len(self.samples)} samples: {self.class_counts}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss function."""
        counts = np.array([self.class_counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float32)
        weights = 1.0 / counts
        weights = weights / weights.sum() * NUM_CLASSES  # normalize
        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler_weights(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler to handle class imbalance."""
        counts = np.array([self.class_counts.get(c, 1) for c in CLASS_NAMES], dtype=np.float32)
        class_weights = 1.0 / counts
        sample_weights = [class_weights[label] for _, label in self.samples]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


# ── DataLoader factory ─────────────────────────────────────────────────────────
def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    use_sampler: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build DataLoaders for train, val, and test splits.

    Args:
        data_dir:    Path to chest_xray/ directory
        batch_size:  Batch size
        num_workers: Workers for DataLoader
        use_sampler: Use WeightedRandomSampler to balance classes

    Returns:
        Dict with keys 'train', 'val', 'test'
    """
    datasets = {
        split: ChestXRayDataset(data_dir, split=split)
        for split in ["train", "val", "test"]
    }

    train_sampler = datasets["train"].get_sampler_weights() if use_sampler else None

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders, datasets


if __name__ == "__main__":
    # Quick smoke test
    loaders, datasets = get_dataloaders("data/chest_xray", batch_size=8, num_workers=0)
    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape: {images.shape}, Labels: {labels}")
    print(f"Class weights: {datasets['train'].get_class_weights()}")
