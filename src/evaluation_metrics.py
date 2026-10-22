import math
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# Optional LPIPS dependency
_lpips_model = None
_lpips_initialized = False

def get_lpips_model(device: Optional[torch.device] = None):
    """Lazily loads and caches the AlexNet-based LPIPS model."""
    global _lpips_model, _lpips_initialized
    if _lpips_initialized:
        return _lpips_model
    try:
        import lpips
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(dev)
        _lpips_model.eval()
        _lpips_initialized = True
        return _lpips_model
    except Exception as e:
        print(f"[Metrics] LPIPS unavailable ({e}); metric will be reported as unavailable.")
        _lpips_initialized = True
        _lpips_model = None
        return None


def to_numpy_rgb(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Converts any supported image format to a float32 NumPy array with shape (H, W, 3) in [0, 1].
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        return np.clip(arr, 0.0, 1.0)
    elif isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        arr = t.numpy().astype(np.float32)
        return np.clip(arr, 0.0, 1.0)
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.01:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def calculate_image_quality_metrics(
    orig_img: Union[Image.Image, np.ndarray, torch.Tensor],
    adv_img: Union[Image.Image, np.ndarray, torch.Tensor],
    epsilon_bound: float = 8.0 / 255.0,
    device: Optional[torch.device] = None,
    tolerance: float = 1.0 / 255.0 + 1e-6
) -> Dict[str, Any]:
    """
    Computes rigorous mathematical and perceptual image quality metrics
    between original and cloaked assets at identical dimensions and coordinate alignment.

    Args:
        orig_img: Original image in [0, 1] range
        adv_img: Cloaked image in [0, 1] range
        epsilon_bound: The configured L-infinity perturbation budget in [0, 1]
        device: PyTorch device for LPIPS calculation
        tolerance: Quantization / rounding tolerance for 8-bit image formats

    Returns:
        Dictionary of computed metric values and verification statuses.
    """
    orig_np = to_numpy_rgb(orig_img)
    adv_np = to_numpy_rgb(adv_img)

    if orig_np.shape != adv_np.shape:
        raise ValueError(f"Image shapes do not match: {orig_np.shape} vs {adv_np.shape}")

    h, w, c = orig_np.shape
    diff = np.abs(adv_np - orig_np)

    # 1. Structural Similarity Index (SSIM)
    try:
        ssim_val = float(ssim_fn(orig_np, adv_np, channel_axis=2, data_range=1.0))
        ssim_val = max(-1.0, min(1.0, ssim_val))
    except Exception as e:
        ssim_val = None

    # 2. Peak Signal-to-Noise Ratio (PSNR)
    mse = float(np.mean(diff ** 2))
    if mse == 0.0:
        psnr_val = 100.0  # Perfect reconstruction benchmark convention
    else:
        try:
            psnr_val = float(psnr_fn(orig_np, adv_np, data_range=1.0))
        except Exception:
            psnr_val = float(10.0 * np.log10(1.0 / mse)) if mse > 0 else 100.0

    # 3. L-infinity Norm (raw pixel space [0, 1])
    linf_norm = float(np.max(diff))

    # 4. L2 Norm (Euclidean perturbation norm and RMSE)
    l2_norm = float(np.linalg.norm(diff))
    rmse = float(np.sqrt(mse))

    # 5. Mean Absolute Error (MAE)
    mae = float(np.mean(diff))

    # 6. Perturbation Magnitude Percentage
    perturbation_pct = float((linf_norm / 1.0) * 100.0)

    # 7. Epsilon Constraint Verification
    epsilon_passed = bool(linf_norm <= (epsilon_bound + tolerance))

    # 8. Learned Perceptual Image Patch Similarity (LPIPS)
    lpips_val = None
    lpips_status = "unavailable"
    lpips_model = get_lpips_model(device=device)
    if lpips_model is not None:
        try:
            dev = next(lpips_model.parameters()).device
            # LPIPS expects tensors in range [-1, 1], shape [1, 3, H, W]
            t_orig = torch.from_numpy(orig_np).permute(2, 0, 1).unsqueeze(0).to(dev) * 2.0 - 1.0
            t_adv = torch.from_numpy(adv_np).permute(2, 0, 1).unsqueeze(0).to(dev) * 2.0 - 1.0

            with torch.no_grad():
                dist = lpips_model(t_orig, t_adv)
                lpips_val = float(dist.squeeze().item())
                lpips_status = "computed"
        except Exception as e:
            print(f"[Metrics] Error computing LPIPS: {e}")
            lpips_val = None
            lpips_status = f"error: {str(e)}"

    return {
        "dimensions": {"width": w, "height": h, "channels": c},
        "ssim": round(ssim_val, 4) if ssim_val is not None else None,
        "psnr_db": round(psnr_val, 2) if psnr_val is not None else None,
        "lpips": round(lpips_val, 4) if lpips_val is not None else None,
        "lpips_status": lpips_status,
        "linf_norm": round(linf_norm, 5),
        "l2_norm": round(l2_norm, 4),
        "rmse": round(rmse, 5),
        "mae": round(mae, 5),
        "perturbation_percent": round(perturbation_pct, 2),
        "epsilon_configured": float(epsilon_bound),
        "epsilon_tolerance": float(tolerance),
        "epsilon_passed": epsilon_passed,
        "color_space": "sRGB",
        "input_range": "[0.0, 1.0]",
    }
