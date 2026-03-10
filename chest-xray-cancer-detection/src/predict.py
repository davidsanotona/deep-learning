"""
Single-image inference with optional Grad-CAM visualization.

Usage:
    python src/predict.py --image path/to/xray.jpg
    python src/predict.py --image path/to/xray.jpg --gradcam
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, Dict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from src.model import load_model

DEFAULT_MODEL_PATH = "models/best_model.pth"


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_image(image_path: str) -> Tuple[torch.Tensor, Image.Image]:
    """Load and preprocess a single chest X-ray image."""
    pil_image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(pil_image).unsqueeze(0)  # add batch dim
    return tensor, pil_image


# ── Grad-CAM ───────────────────────────────────────────────────────────────────
class GradCAM:
    """Simple Grad-CAM for EfficientNet."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def generate_gradcam_overlay(model, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Generate Grad-CAM heatmap overlay."""
    # Target the last conv block of EfficientNet
    target_layer = model.backbone.blocks[-1]
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(tensor, class_idx)
    return heatmap


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: torch.device = None,
) -> Dict:
    """
    Run inference on a single image.

    Returns:
        dict with keys: class_name, class_idx, confidence, probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    tensor, pil_image = preprocess_image(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)
        probs  = F.softmax(output, dim=1)[0]
        pred_idx  = probs.argmax().item()
        confidence = probs[pred_idx].item()

    result = {
        "class_name":    CLASS_NAMES[pred_idx],
        "class_idx":     pred_idx,
        "confidence":    confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        },
        "image_path": str(image_path),
    }

    return result


def predict_with_gradcam(
    image_path: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: torch.device = None,
    save_path: str = None,
) -> Tuple[Dict, np.ndarray]:
    """Run inference and generate Grad-CAM heatmap."""
    import matplotlib.pyplot as plt
    import matplotlib

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)
    model.eval()
    tensor, pil_image = preprocess_image(image_path)
    tensor = tensor.to(device).requires_grad_(True)

    # Forward pass
    output = model(tensor)
    probs  = F.softmax(output, dim=1)[0]
    pred_idx  = probs.argmax().item()
    confidence = probs[pred_idx].item()

    heatmap = generate_gradcam_overlay(model, tensor, pred_idx)

    # Overlay heatmap on original image
    img_array = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]
    overlay = 0.6 * img_array + 0.4 * colored_heatmap

    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_array, cmap="gray")
        axes[0].set_title("Original X-Ray")
        axes[1].imshow(colored_heatmap)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay — {CLASS_NAMES[pred_idx]} ({confidence:.1%})")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Grad-CAM saved: {save_path}")

    result = {
        "class_name":    CLASS_NAMES[pred_idx],
        "class_idx":     pred_idx,
        "confidence":    confidence,
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
        "image_path":    str(image_path),
    }
    return result, overlay


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True, help="Path to X-ray image")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gradcam",    action="store_true", help="Generate Grad-CAM")
    args = parser.parse_args()

    print(f"Predicting: {args.image}")

    if args.gradcam:
        save_path = Path(args.image).stem + "_gradcam.png"
        result, _ = predict_with_gradcam(args.image, args.model_path, save_path=save_path)
    else:
        result = predict(args.image, args.model_path)

    print(f"\n  Prediction:  {result['class_name']}")
    print(f"  Confidence:  {result['confidence']:.2%}")
    print(f"  Probabilities:")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:12s}: {prob:.4f}  {bar}")
