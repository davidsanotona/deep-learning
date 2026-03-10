"""
Unit tests for the Chest X-Ray Detection project.

Run: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Model Tests ────────────────────────────────────────────────────────────────
class TestChestXRayModel:
    def test_model_instantiation(self):
        from src.model import ChestXRayModel
        model = ChestXRayModel(pretrained=False)
        assert model is not None

    def test_model_output_shape(self):
        from src.model import ChestXRayModel, NUM_CLASSES
        from src.dataset import NUM_CLASSES
        model = ChestXRayModel(pretrained=False)
        model.eval()
        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (4, NUM_CLASSES), f"Expected (4, {NUM_CLASSES}), got {out.shape}"

    def test_freeze_unfreeze(self):
        from src.model import ChestXRayModel
        model = ChestXRayModel(pretrained=False)
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad

        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_parameter_count(self):
        from src.model import ChestXRayModel
        model = ChestXRayModel(pretrained=False)
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["trainable"] <= params["total"]

    def test_feature_extraction(self):
        from src.model import ChestXRayModel
        model = ChestXRayModel(pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            features = model.get_features(dummy)
        assert features.ndim == 2  # [batch, feature_dim]
        assert features.shape[0] == 2


# ── Dataset Tests ──────────────────────────────────────────────────────────────
class TestTransforms:
    def test_train_transform_output_shape(self):
        from src.dataset import get_transforms
        transform = get_transforms("train")
        img = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        from src.dataset import get_transforms
        transform = get_transforms("val")
        img = Image.fromarray(np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_transforms_normalize(self):
        from src.dataset import get_transforms
        transform = get_transforms("val")
        img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
        tensor = transform(img)
        # After ImageNet normalization, values should be roughly centered around 0
        assert tensor.mean().abs() < 2.0


# ── Prediction Tests ───────────────────────────────────────────────────────────
class TestPreprocessing:
    def test_preprocess_image(self, tmp_path):
        from src.predict import preprocess_image
        # Create a dummy image
        img = Image.fromarray(np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8))
        img_path = tmp_path / "test_xray.jpg"
        img.save(str(img_path))

        tensor, pil = preprocess_image(str(img_path))
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(pil, Image.Image)


# ── Class config tests ─────────────────────────────────────────────────────────
class TestConfig:
    def test_class_names(self):
        from src.dataset import CLASS_NAMES, CLASS_TO_IDX, NUM_CLASSES
        assert len(CLASS_NAMES) == NUM_CLASSES
        assert all(name in CLASS_TO_IDX for name in CLASS_NAMES)

    def test_class_idx_mapping(self):
        from src.dataset import CLASS_NAMES, CLASS_TO_IDX
        for idx, name in enumerate(CLASS_NAMES):
            assert CLASS_TO_IDX[name] == idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
