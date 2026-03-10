"""
EfficientNetB3 Transfer Learning model for Chest X-Ray classification.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import timm

from src.dataset import NUM_CLASSES, CLASS_NAMES


# ── Model Definition ───────────────────────────────────────────────────────────
class ChestXRayModel(nn.Module):
    """
    EfficientNetB3 with a custom classification head.
    Uses timm for pretrained weights.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained EfficientNetB3 backbone
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=pretrained,
            num_classes=0,       # remove original classifier
            global_pool="avg",   # global average pooling
        )
        feature_dim = self.backbone.num_features  # 1536 for B3

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings (before classifier)."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze backbone — only train the head."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen. Training classifier head only.")

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. Fine-tuning all layers.")

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── Model I/O ──────────────────────────────────────────────────────────────────
def save_model(model: ChestXRayModel, path: str, extra_info: Optional[dict] = None):
    """Save model weights and metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "class_names": CLASS_NAMES,
    }
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: Optional[torch.device] = None) -> ChestXRayModel:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(path, map_location=device)
    model = ChestXRayModel(
        num_classes=checkpoint.get("num_classes", NUM_CLASSES),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {path} | Device: {device}")
    return model


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChestXRayModel(pretrained=True).to(device)
    params = model.count_parameters()
    print(f"Model: EfficientNetB3")
    print(f"  Total params:     {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")

    # Forward pass test
    dummy = torch.randn(2, 3, 224, 224).to(device)
    out = model(dummy)
    print(f"  Output shape: {out.shape}")  # [2, 2]
