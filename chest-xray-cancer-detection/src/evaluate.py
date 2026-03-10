"""
Evaluation script: metrics, confusion matrix, ROC curve, Grad-CAM visualization.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model_path models/best_model.pth --data_dir data/chest_xray
"""

import sys
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, accuracy_score,
)
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloaders, CLASS_NAMES
from src.model import load_model

load_dotenv()


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    """Run model on entire loader, return predictions and true labels."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs   = F.softmax(outputs, dim=1)
        preds   = probs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Metrics ────────────────────────────────────────────────────────────────────
def print_metrics(labels, preds, probs):
    """Print classification report and key metrics."""
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average="weighted")
    auc  = roc_auc_score(labels, probs[:, 1])  # binary AUC

    print("\n Evaluation Metrics")
    print("=" * 50)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))
    return {"accuracy": acc, "f1": f1, "auc": auc}


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_dir: str = "models"):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Confusion Matrix — Chest X-Ray Classifier", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw Counts", "Normalized"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    out_path = Path(save_dir) / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Confusion matrix saved: {out_path}")
    plt.close()


def plot_roc_curve(labels, probs, save_dir: str = "models"):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    auc = roc_auc_score(labels, probs[:, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Chest X-Ray Classifier")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out_path = Path(save_dir) / "roc_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  ROC curve saved: {out_path}")
    plt.close()


def plot_prediction_samples(model, loader, device, save_dir: str = "models", n: int = 12):
    """Plot sample predictions with confidence scores."""
    model.eval()
    images_shown, labels_shown, preds_shown, confs_shown = [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs   = F.softmax(outputs, dim=1)
            preds   = probs.argmax(dim=1).cpu()
            confs   = probs.max(dim=1).values.cpu()

            for i in range(len(images)):
                images_shown.append(images[i])
                labels_shown.append(labels[i].item())
                preds_shown.append(preds[i].item())
                confs_shown.append(confs[i].item())
                if len(images_shown) >= n:
                    break
            if len(images_shown) >= n:
                break

    # Denormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx >= n:
            ax.axis("off")
            continue
        img = images_shown[idx] * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img, cmap="gray")

        correct = labels_shown[idx] == preds_shown[idx]
        color = "green" if correct else "red"
        ax.set_title(
            f"True: {CLASS_NAMES[labels_shown[idx]]}\n"
            f"Pred: {CLASS_NAMES[preds_shown[idx]]} ({confs_shown[idx]:.2%})",
            fontsize=8, color=color,
        )
        ax.axis("off")

    plt.tight_layout()
    out_path = Path(save_dir) / "sample_predictions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  💾 Sample predictions saved: {out_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
def evaluate(model_path: str, data_dir: str, save_dir: str = "models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    print("\n Loading test data...")
    loaders, _ = get_dataloaders(data_dir, batch_size=64, num_workers=4)

    print("Loading model...")
    model = load_model(model_path, device)

    print("\nRunning inference on test set...")
    labels, preds, probs = run_inference(model, loaders["test"], device)

    metrics = print_metrics(labels, preds, probs)

    print("\nGenerating plots...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, save_dir)
    plot_roc_curve(labels, probs, save_dir)
    plot_prediction_samples(model, loaders["test"], device, save_dir)

    print(f"\nEvaluation complete. Plots saved to: {save_dir}/")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.getenv("MODEL_SAVE_PATH", "models/best_model.pth"))
    parser.add_argument("--data_dir",   default=os.getenv("DATA_DIR", "data/chest_xray"))
    parser.add_argument("--save_dir",   default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.model_path, args.data_dir, args.save_dir)
