import io
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models import SurrogateVisionModel
from src.classifier_eval import get_model_predictions, get_style_labels
from src.evaluation_metrics import to_numpy_rgb, calculate_image_quality_metrics


class ImageTransformations:
    """Standard transformations encountered by digital assets online."""

    @staticmethod
    def jpeg_compression(img: Image.Image, quality: int = 75) -> Image.Image:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def resize_restore(img: Image.Image, scale: float = 0.8) -> Image.Image:
        w, h = img.size
        down_w, down_h = max(16, int(w * scale)), max(16, int(h * scale))
        down = img.resize((down_w, down_h), Image.Resampling.BILINEAR)
        return down.resize((w, h), Image.Resampling.BILINEAR)

    @staticmethod
    def center_crop(img: Image.Image, crop_ratio: float = 0.9) -> Image.Image:
        w, h = img.size
        crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.Resampling.BILINEAR)

    @staticmethod
    def gaussian_blur(img: Image.Image, radius: float = 1.0) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def brightness(img: Image.Image, factor: float = 1.15) -> Image.Image:
        return ImageEnhance.Brightness(img).enhance(factor)

    @staticmethod
    def contrast(img: Image.Image, factor: float = 1.15) -> Image.Image:
        return ImageEnhance.Contrast(img).enhance(factor)


def evaluate_asset_robustness(
    orig_img: Union[Image.Image, np.ndarray, torch.Tensor],
    adv_img: Union[Image.Image, np.ndarray, torch.Tensor],
    model: SurrogateVisionModel,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates empirical perturbation survival across standard transformations.
    Runs on original unperturbed image and cloaked image.
    """
    if isinstance(orig_img, Image.Image):
        pil_orig = orig_img.convert("RGB")
    else:
        orig_np = (to_numpy_rgb(orig_img) * 255.0).astype(np.uint8)
        pil_orig = Image.fromarray(orig_np)

    if isinstance(adv_img, Image.Image):
        pil_adv = adv_img.convert("RGB")
    else:
        adv_np = (to_numpy_rgb(adv_img) * 255.0).astype(np.uint8)
        pil_adv = Image.fromarray(adv_np)

    # 1. Baseline prediction on original image
    orig_pred = get_model_predictions(model, pil_orig, device=device)
    orig_target_idx = orig_pred["top1_class_idx"]
    orig_target_style = orig_pred["top1_style"]

    # 2. Cloaked prediction
    adv_pred = get_model_predictions(model, pil_adv, device=device)
    adv_initial_idx = adv_pred["top1_class_idx"]

    transform_suite = [
        {
            "name": "JPEG Compression (Q=75)",
            "params": {"format": "JPEG", "quality": 75},
            "fn": lambda im: ImageTransformations.jpeg_compression(im, 75)
        },
        {
            "name": "JPEG Compression (Q=50)",
            "params": {"format": "JPEG", "quality": 50},
            "fn": lambda im: ImageTransformations.jpeg_compression(im, 50)
        },
        {
            "name": "Resizing (0.8x)",
            "params": {"scale": 0.8, "interpolation": "bilinear"},
            "fn": lambda im: ImageTransformations.resize_restore(im, 0.8)
        },
        {
            "name": "Mild Center Crop (90%)",
            "params": {"crop_ratio": 0.9},
            "fn": lambda im: ImageTransformations.center_crop(im, 0.9)
        },
        {
            "name": "Gaussian Spatial Blur",
            "params": {"radius": 1.0},
            "fn": lambda im: ImageTransformations.gaussian_blur(im, 1.0)
        },
        {
            "name": "Brightness Adjustment (+15%)",
            "params": {"factor": 1.15},
            "fn": lambda im: ImageTransformations.brightness(im, 1.15)
        },
        {
            "name": "Contrast Adjustment (+15%)",
            "params": {"factor": 1.15},
            "fn": lambda im: ImageTransformations.contrast(im, 1.15)
        },
    ]

    results = []

    for item in transform_suite:
        t_name = item["name"]
        t_params = item["params"]
        t_fn = item["fn"]

        try:
            transformed_adv = t_fn(pil_adv)
            trans_pred = get_model_predictions(model, transformed_adv, device=device)

            trans_idx = trans_pred["top1_class_idx"]
            trans_style = trans_pred["top1_style"]
            trans_conf = trans_pred["top1_confidence"]

            # Protection survived if the model does NOT revert back to original true class!
            is_disrupted = (trans_idx != orig_target_idx)
            retained_rate = 100.0 if is_disrupted else 0.0

            # Measure perceptual difference of transformed image relative to original
            ssim_score = None
            try:
                np_orig = np.array(pil_orig).astype(np.float32) / 255.0
                np_trans = np.array(transformed_adv).astype(np.float32) / 255.0
                from skimage.metrics import structural_similarity as ssim_fn
                ssim_score = round(float(ssim_fn(np_orig, np_trans, channel_axis=2, data_range=1.0)), 4)
            except Exception:
                pass

            results.append({
                "transformation_name": t_name,
                "transformation_params": t_params,
                "top1_class_after_transform": trans_style,
                "top1_confidence_after_transform": trans_conf,
                "prediction_disrupted": is_disrupted,
                "retained_protection_rate": retained_rate,
                "status": "Robust" if is_disrupted else "Reverted to Baseline",
                "ssim_after_transform": ssim_score
            })

        except Exception as e:
            print(f"[Robustness] Error testing {t_name}: {e}")
            results.append({
                "transformation_name": t_name,
                "transformation_params": t_params,
                "top1_class_after_transform": "Error",
                "top1_confidence_after_transform": 0.0,
                "prediction_disrupted": False,
                "retained_protection_rate": 0.0,
                "status": f"Evaluation Error: {str(e)}",
                "ssim_after_transform": None
            })

    return results
