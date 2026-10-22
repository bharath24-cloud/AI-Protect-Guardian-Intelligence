import math
import numpy as np
import pytest
import torch
from PIL import Image

from src.evaluation_metrics import calculate_image_quality_metrics, to_numpy_rgb
from src.classifier_eval import (
    get_style_labels,
    load_evaluation_model,
    get_model_predictions,
    compare_original_versus_cloaked,
    evaluate_cross_model_transferability,
)
from src.robustness_eval import ImageTransformations, evaluate_asset_robustness


class TestEvaluationMetrics:
    """Rigorous tests for image quality and mathematical perturbation metrics."""

    def test_identical_images_perfect_similarity(self):
        """Identical images must yield SSIM=1.0, PSNR=100.0, Linf=0.0, L2=0.0, passed=True."""
        img = np.random.uniform(0.1, 0.9, (128, 128, 3)).astype(np.float32)
        res = calculate_image_quality_metrics(img, img.copy(), epsilon_bound=8.0/255.0)

        assert res["ssim"] == 1.0, f"Expected SSIM 1.0, got {res['ssim']}"
        assert res["psnr_db"] == 100.0, f"Expected PSNR 100.0 dB, got {res['psnr_db']}"
        assert res["linf_norm"] == 0.0, f"Expected Linf 0.0, got {res['linf_norm']}"
        assert res["l2_norm"] == 0.0, f"Expected L2 0.0, got {res['l2_norm']}"
        assert res["mae"] == 0.0, f"Expected MAE 0.0, got {res['mae']}"
        assert res["epsilon_passed"] is True
        if res["lpips"] is not None:
            assert res["lpips"] < 0.001, f"Expected LPIPS near 0.0, got {res['lpips']}"

    def test_bounded_perturbation_epsilon_check(self):
        """Perturbation within epsilon must pass strict bound verification."""
        eps = 8.0 / 255.0
        img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        # Perturbation within epsilon bound
        img2 = img1 + (eps * 0.9)
        res = calculate_image_quality_metrics(img1, img2, epsilon_bound=eps)

        assert res["epsilon_passed"] is True
        assert res["linf_norm"] <= (eps + 1e-5)
        assert res["ssim"] > 0.95

    def test_exceeded_perturbation_fails_check(self):
        """Perturbation exceeding epsilon + tolerance must fail verification."""
        eps = 8.0 / 255.0
        img1 = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        img2 = img1 + (eps * 1.5)  # 50% above bound
        res = calculate_image_quality_metrics(img1, img2, epsilon_bound=eps)

        assert res["epsilon_passed"] is False
        assert res["linf_norm"] > eps

    def test_tensor_conversion_and_range(self):
        """Tensors in various formats should convert properly to [0, 1] RGB numpy."""
        t = torch.rand(1, 3, 32, 32)
        arr = to_numpy_rgb(t)
        assert arr.shape == (32, 32, 3)
        assert arr.min() >= 0.0 and arr.max() <= 1.0


class TestClassifierEvaluation:
    """Tests for model loading, inference, top-1/top-5 prediction, and transferability."""

    def test_style_labels_integrity(self):
        """Style label mapping must contain 38 unique valid WikiArt style classes."""
        labels = get_style_labels()
        assert len(labels) == 38, f"Expected 38 style classes, got {len(labels)}"
        assert isinstance(labels[0], str)
        assert len(set(labels.values())) == 38, "Duplicate labels found in mapping"

    def test_top5_probabilities_monotonic_and_sum(self):
        """Top-5 predictions must be monotonically decreasing in confidence and sum <= 1.0."""
        model, msg = load_evaluation_model("resnet50")
        assert model is not None, f"ResNet-50 should load: {msg}"

        dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
        preds = get_model_predictions(model, dummy_img)

        top5 = preds["top5_predictions"]
        assert len(top5) == 5
        for i in range(len(top5) - 1):
            assert top5[i]["confidence"] >= top5[i+1]["confidence"], "Top 5 must be sorted by confidence"

        total_conf = sum(p["confidence"] for p in top5)
        assert total_conf <= 1.0001, f"Top 5 confidence sum exceeds 1: {total_conf}"

    def test_unavailable_model_handled_gracefully(self):
        """Non-existent model must return status 'unavailable' without crash."""
        results = evaluate_cross_model_transferability(
            orig_img=np.zeros((64, 64, 3), dtype=np.float32),
            adv_img=np.zeros((64, 64, 3), dtype=np.float32),
            candidate_models=["non_existent_architecture_xyz"]
        )
        assert len(results) == 1
        assert results[0]["status"] == "unavailable"
        assert results[0]["original_top1_class"] is None


class TestRobustnessSuite:
    """Tests for digital image transformations and empirical survival checks."""

    def test_transformations_preserve_format_and_range(self):
        """Transformations must return valid RGB PIL images with consistent dimensions."""
        img = Image.new("RGB", (100, 100), (200, 100, 50))
        
        t_jpeg = ImageTransformations.jpeg_compression(img, 75)
        assert t_jpeg.size == (100, 100)
        assert t_jpeg.mode == "RGB"

        t_resize = ImageTransformations.resize_restore(img, 0.8)
        assert t_resize.size == (100, 100)

        t_crop = ImageTransformations.center_crop(img, 0.9)
        assert t_crop.size == (100, 100)

        t_blur = ImageTransformations.gaussian_blur(img, 1.0)
        assert t_blur.size == (100, 100)

        t_bright = ImageTransformations.brightness(img, 1.15)
        assert t_bright.size == (100, 100)

        t_contrast = ImageTransformations.contrast(img, 1.15)
        assert t_contrast.size == (100, 100)
