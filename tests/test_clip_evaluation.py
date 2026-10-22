import math
import numpy as np
import pytest
import torch
from PIL import Image

from src.classifier_eval import (
    CLIPZeroShotEvaluator,
    get_style_labels,
    load_evaluation_model,
    compare_original_versus_cloaked,
    evaluate_cross_model_transferability,
)


class TestCLIPZeroShotEvaluation:
    """Verifies that CLIP-ViT-B/32 operates as a zero-shot multimodal style evaluator."""

    def test_clip_evaluator_initialization_and_prompts(self):
        """CLIP evaluator must precompute 38 normalized text prompt embeddings."""
        labels = get_style_labels()
        assert len(labels) == 38, "WikiArt style labels must have 38 classes"

        evaluator = CLIPZeroShotEvaluator()
        assert evaluator.model is not None
        assert evaluator.processor is not None
        assert evaluator._text_features is not None
        assert evaluator._text_features.shape[0] == 38
        assert evaluator._text_features.shape[1] == 512  # CLIP ViT-B/32 embedding dimension

        # Verify embeddings are L2 normalized
        norms = evaluator._text_features.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), "Text features must be L2-normalized"

    def test_clip_inference_on_sample_image(self):
        """CLIP inference must return top-1, top-5, and 512-dim image embedding."""
        evaluator = CLIPZeroShotEvaluator()
        # Create a sample PIL test image (warm tones)
        test_img = Image.new("RGB", (224, 224), (180, 120, 80))

        pred = evaluator.get_predictions(test_img)
        assert "top1_style" in pred
        assert "top1_confidence" in pred
        assert "top5_predictions" in pred
        assert len(pred["top5_predictions"]) == 5
        assert pred["embedding"] is not None
        assert pred["embedding"].shape[-1] == 512

        # Probabilities must be valid and sum to ~1.0
        all_probs = pred["all_probabilities"]
        assert len(all_probs) == 38
        assert abs(float(all_probs.sum()) - 1.0) < 1e-4

        # Monotonicity of top-5
        top5 = pred["top5_predictions"]
        for i in range(len(top5) - 1):
            assert top5[i]["confidence"] >= top5[i+1]["confidence"]

    def test_clip_compare_original_versus_cloaked(self):
        """compare_original_versus_cloaked must compute cosine distance and confidence drop."""
        evaluator = CLIPZeroShotEvaluator()
        orig_img = Image.new("RGB", (224, 224), (200, 150, 100))
        # Create slightly perturbed image
        adv_arr = (np.array(orig_img).astype(np.float32) + 8.0).clip(0, 255).astype(np.uint8)
        adv_img = Image.fromarray(adv_arr)

        cmp_res = evaluator.compare_original_versus_cloaked(orig_img, adv_img)
        assert "original_prediction" in cmp_res
        assert "cloaked_prediction" in cmp_res
        assert "confidence_drop" in cmp_res
        assert "embedding_cosine_distance" in cmp_res
        assert "evaluation_type" in cmp_res
        assert cmp_res["evaluation_type"] == "zero_shot_multimodal"
        assert cmp_res["prompt_template"] == "a painting in {style} style"
        assert cmp_res["embedding_cosine_distance"] >= 0.0

    def test_load_evaluation_model_clip(self):
        """load_evaluation_model must load CLIPZeroShotEvaluator with 'clip_vit_b32'."""
        model, msg = load_evaluation_model("clip_vit_b32")
        assert model is not None, f"CLIP should load successfully, got msg: {msg}"
        assert isinstance(model, CLIPZeroShotEvaluator)
        assert "loaded" in msg

    def test_evaluate_cross_model_transferability_with_clip(self):
        """evaluate_cross_model_transferability must evaluate CLIP and report zero_shot_multimodal."""
        orig_img = Image.new("RGB", (64, 64), (100, 100, 100))
        adv_img = Image.new("RGB", (64, 64), (108, 108, 108))

        results = evaluate_cross_model_transferability(
            orig_img=orig_img,
            adv_img=adv_img,
            candidate_models=["clip_vit_b32"]
        )

        assert len(results) == 1
        clip_res = results[0]
        assert clip_res["model_name"] == "clip_vit_b32"
        assert clip_res["status"] == "evaluated", f"Expected evaluated, got {clip_res['status']}: {clip_res.get('status_detail')}"
        assert clip_res["evaluation_type"] == "zero_shot_multimodal"
        assert clip_res["prompt_template"] == "a painting in {style} style"
        assert clip_res["original_top1_class"] is not None
        assert clip_res["original_top1_confidence"] is not None
        assert clip_res["embedding_cosine_distance"] is not None
