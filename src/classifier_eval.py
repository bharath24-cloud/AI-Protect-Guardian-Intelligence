import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models import get_model, SurrogateVisionModel
from src.preprocessing import IMAGENET_MEAN, IMAGENET_STD

# Global cache for evaluated models to avoid re-instantiating on every request
_MODEL_CACHE: Dict[str, Any] = {}
_LABEL_MAPPING_CACHE: Optional[Dict[int, str]] = None


def get_style_labels() -> Dict[int, str]:
    """Loads canonical 38 WikiArt style label mapping."""
    global _LABEL_MAPPING_CACHE
    if _LABEL_MAPPING_CACHE is not None:
        return _LABEL_MAPPING_CACHE

    # First attempt: load from training_meta.json
    meta_path = Path("models/checkpoints/training_meta.json")
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                classes = meta.get("classes", [])
                if classes:
                    _LABEL_MAPPING_CACHE = {i: c for i, c in enumerate(classes)}
                    return _LABEL_MAPPING_CACHE
        except Exception as e:
            print(f"[Warning] Failed loading training_meta.json: {e}")

    # Fallback to checkpoint dict
    ckpt_path = Path("models/checkpoints/best_model.pt")
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "idx_to_label" in ckpt and isinstance(ckpt["idx_to_label"], dict):
                _LABEL_MAPPING_CACHE = {int(k): v for k, v in ckpt["idx_to_label"].items()}
                return _LABEL_MAPPING_CACHE
            elif "label_mapping" in ckpt and isinstance(ckpt["label_mapping"], dict):
                # reverse mapping if needed
                mapping = ckpt["label_mapping"]
                _LABEL_MAPPING_CACHE = {int(v): k for k, v in mapping.items()}
                return _LABEL_MAPPING_CACHE
        except Exception as e:
            print(f"[Warning] Failed extracting labels from checkpoint: {e}")

    # Generic fallback
    _LABEL_MAPPING_CACHE = {i: f"Style_{i}" for i in range(38)}
    return _LABEL_MAPPING_CACHE


class CLIPZeroShotEvaluator:
    """
    Zero-shot multimodal style evaluator for OpenAI CLIP ViT-B/32.
    Evaluates image-text alignment against canonical 38 WikiArt style prompts
    and produces L2-normalized image embeddings for representation shift analysis.
    """
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[torch.device] = None,
        prompt_template: str = "a painting in {style} style"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.labels = get_style_labels()
        self._indices = sorted(self.labels.keys())
        self._clean_labels = [self.labels[i].replace("_", " ") for i in self._indices]
        self._text_prompts = [self.prompt_template.format(style=s) for s in self._clean_labels]

        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Precompute and cache normalized text embeddings
        self._text_features: Optional[torch.Tensor] = None
        self._encode_text_prompts()

    @staticmethod
    def _extract_tensor(out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "text_embeds") and out.text_embeds is not None:
            return out.text_embeds
        if hasattr(out, "image_embeds") and out.image_embeds is not None:
            return out.image_embeds
        raise ValueError(f"Could not extract tensor from {type(out)}")

    def _encode_text_prompts(self):
        with torch.no_grad():
            text_inputs = self.processor(
                text=self._text_prompts,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            raw_text = self._extract_tensor(self.model.get_text_features(**text_inputs))
            self._text_features = raw_text / raw_text.norm(dim=-1, keepdim=True)

    def _to_pil(self, image_input: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            arr = image_input
            if arr.max() <= 1.01:
                arr = (arr * 255.0).astype(np.uint8)
            return Image.fromarray(arr.astype(np.uint8))
        elif isinstance(image_input, torch.Tensor):
            t = image_input.detach().cpu()
            if t.ndim == 4:
                t = t.squeeze(0)
            if t.shape[0] == 3:
                t = t.permute(1, 2, 0)
            arr = (t.numpy() * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def get_predictions(
        self,
        image_input: Union[Image.Image, np.ndarray, torch.Tensor],
        top_k: int = 5
    ) -> Dict[str, Any]:
        pil_img = self._to_pil(image_input)

        with torch.no_grad():
            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            raw_img = self._extract_tensor(self.model.get_image_features(**img_inputs))
            norm_img = raw_img / raw_img.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            logits = (logit_scale * (norm_img @ self._text_features.T)).squeeze(0)
            probs = logits.softmax(dim=-1)

            top1_pos = int(torch.argmax(probs).item())
            top1_conf = float(probs[top1_pos].item())
            top1_style = self._clean_labels[top1_pos]
            top1_orig_idx = self._indices[top1_pos]

            k = min(top_k, probs.shape[0])
            top_k_confs, top_k_pos = torch.topk(probs, k=k)

            top_k_preds = []
            for conf, pos in zip(top_k_confs.tolist(), top_k_pos.tolist()):
                top_k_preds.append({
                    "class_idx": int(self._indices[pos]),
                    "style_name": self._clean_labels[pos],
                    "confidence": round(float(conf), 4),
                    "confidence_percent": round(float(conf) * 100.0, 2)
                })

        return {
            "top1_class_idx": top1_orig_idx,
            "top1_style": top1_style,
            "top1_confidence": round(top1_conf, 4),
            "top1_confidence_percent": round(top1_conf * 100.0, 2),
            "top5_predictions": top_k_preds,
            "all_probabilities": probs.cpu().numpy(),
            "embedding": norm_img.detach().cpu()
        }

    def compare_original_versus_cloaked(
        self,
        orig_img: Union[Image.Image, np.ndarray, torch.Tensor],
        adv_img: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        orig_pred = self.get_predictions(orig_img)
        adv_pred = self.get_predictions(adv_img)

        orig_class_idx = orig_pred["top1_class_idx"]
        adv_class_idx = adv_pred["top1_class_idx"]

        orig_pos = self._indices.index(orig_class_idx) if orig_class_idx in self._indices else 0
        adv_orig_class_conf = float(adv_pred["all_probabilities"][orig_pos])
        conf_drop = float(orig_pred["top1_confidence"] - adv_orig_class_conf)

        e_orig = orig_pred["embedding"]
        e_adv = adv_pred["embedding"]
        cos_dist = float(1.0 - F.cosine_similarity(e_adv, e_orig, dim=-1).item())
        l2_dist = float(torch.norm(e_adv - e_orig, p=2, dim=-1).item())

        prediction_changed = bool(orig_class_idx != adv_class_idx)

        return {
            "original_prediction": {
                "style_name": orig_pred["top1_style"],
                "class_idx": orig_pred["top1_class_idx"],
                "confidence": orig_pred["top1_confidence"],
                "confidence_percent": orig_pred["top1_confidence_percent"],
                "top5": orig_pred["top5_predictions"]
            },
            "cloaked_prediction": {
                "style_name": adv_pred["top1_style"],
                "class_idx": adv_pred["top1_class_idx"],
                "confidence": adv_pred["top1_confidence"],
                "confidence_percent": adv_pred["top1_confidence_percent"],
                "cloaked_confidence_in_orig_class": round(adv_orig_class_conf, 4),
                "top5": adv_pred["top5_predictions"]
            },
            "top1_prediction_changed": prediction_changed,
            "classification_disrupted": prediction_changed,
            "confidence_drop": round(conf_drop, 4),
            "confidence_drop_percent": round(conf_drop * 100.0, 2),
            "embedding_cosine_distance": round(cos_dist, 4),
            "embedding_l2_distance": round(l2_dist, 4),
            "evaluation_type": "zero_shot_multimodal",
            "prompt_template": self.prompt_template,
            "scientific_disclaimer": (
                "CLIP zero-shot evaluation measures cosine similarity between the image embedding and 38 style prompt templates ('a painting in {style} style'). "
                "Disruption reflects how effectively the adversarial perturbation depresses alignment with the original artistic style."
            )
        }


def load_evaluation_model(
    model_name: str,
    device: Optional[torch.device] = None
) -> Tuple[Optional[Union[SurrogateVisionModel, CLIPZeroShotEvaluator]], str]:
    """
    Loads model checkpoint into memory in eval mode.
    Returns (model, status_message).
    """
    global _MODEL_CACHE
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = f"{model_name}_{device.type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key], "loaded (cached)"

    try:
        labels = get_style_labels()
        num_classes = len(labels)

        if model_name.lower() in ["resnet50", "resnet_50"]:
            ckpt_path = Path("models/checkpoints/best_model.pt")
            if not ckpt_path.exists():
                ckpt_path = Path("models/checkpoints/accuracy_improvement/Exp_B_Finetune_Layer4_best.pth")

            model = get_model("resnet50", num_classes=num_classes, pretrained=False, device=device)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state)
                msg = f"loaded checkpoint {ckpt_path.name}"
            else:
                model = get_model("resnet50", num_classes=num_classes, pretrained=True, device=device)
                msg = "loaded pretrained ImageNet weights (checkpoint not found)"

        elif model_name.lower() in ["convnext_tiny", "convnext"]:
            ckpt_path = Path("models/checkpoints/accuracy_improvement/Exp_F_ConvNeXt_Tiny_best.pth")
            model = get_model("convnext_tiny", num_classes=num_classes, pretrained=False, device=device)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device)
                state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state)
                msg = f"loaded checkpoint {ckpt_path.name}"
            else:
                model = get_model("convnext_tiny", num_classes=num_classes, pretrained=True, device=device)
                msg = "loaded pretrained ImageNet weights"

        elif model_name.lower() in ["vit_b_16", "vit"]:
            # Evaluate with standard ViT
            model = get_model("vit_b_16", num_classes=num_classes, pretrained=True, device=device)
            msg = "loaded pretrained ViT-B/16 architecture"

        elif model_name.lower() in [
            "clip_vit_b32", "clip", "clip_vit_b_32", "clip-vit-b32", "clip_vit_base_patch32", "clip-vit-b-32"
        ]:
            try:
                import transformers
            except ImportError:
                return None, "unavailable: transformers library is not installed (run 'pip install transformers')"

            try:
                evaluator = CLIPZeroShotEvaluator(device=device)
                model = evaluator
                msg = "loaded pretrained CLIP-ViT-B/32 (zero-shot multimodal style evaluator)"
            except torch.cuda.OutOfMemoryError as oom:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None, f"failed: Insufficient GPU VRAM to load CLIP ({oom})"
            except Exception as e:
                return None, f"failed: Could not load pretrained CLIP checkpoint ({e})"

        else:
            return None, f"unavailable: Model architecture '{model_name}' not supported."

        if hasattr(model, "eval"):
            model.eval()
        _MODEL_CACHE[cache_key] = model
        return model, msg

    except Exception as e:
        print(f"[ClassifierEval] Error loading {model_name}: {e}")
        return None, f"failed to load: {str(e)}"



def get_model_predictions(
    model: SurrogateVisionModel,
    image_input: Union[Image.Image, np.ndarray, torch.Tensor],
    device: Optional[torch.device] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Passes an image through a vision model using standard 224x224 ImageNet preprocessing
    and extracts top-1 and top-5 predictions with confidence probabilities and embeddings.
    """
    device = device or next(model.parameters()).device
    labels = get_style_labels()

    # Preprocess image to [1, 3, 224, 224] tensor
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        arr = image_input
        if arr.max() <= 1.01:
            arr = (arr * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(arr.astype(np.uint8))
    elif isinstance(image_input, torch.Tensor):
        t = image_input.detach().cpu()
        if t.ndim == 4:
            t = t.squeeze(0)
        if t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        arr = (t.numpy() * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    x_tensor = preprocess(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, emb = model(x_tensor, return_embedding=True)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        # Top-1
        top1_idx = int(torch.argmax(probs).item())
        top1_conf = float(probs[top1_idx].item())
        top1_label = labels.get(top1_idx, f"Class_{top1_idx}")

        # Top-k
        k = min(top_k, probs.shape[0])
        top_k_confs, top_k_indices = torch.topk(probs, k=k)

        top_k_preds = []
        for conf, idx in zip(top_k_confs.tolist(), top_k_indices.tolist()):
            top_k_preds.append({
                "class_idx": int(idx),
                "style_name": labels.get(idx, f"Class_{idx}"),
                "confidence": round(float(conf), 4),
                "confidence_percent": round(float(conf) * 100.0, 2)
            })

    return {
        "top1_class_idx": top1_idx,
        "top1_style": top1_label,
        "top1_confidence": round(top1_conf, 4),
        "top1_confidence_percent": round(top1_conf * 100.0, 2),
        "top5_predictions": top_k_preds,
        "all_probabilities": probs.cpu().numpy(),
        "embedding": emb.detach().cpu()
    }


def compare_original_versus_cloaked(
    orig_img: Union[Image.Image, np.ndarray, torch.Tensor],
    adv_img: Union[Image.Image, np.ndarray, torch.Tensor],
    model: Union[SurrogateVisionModel, CLIPZeroShotEvaluator],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Evaluates both original and cloaked assets through identical model forward passes.
    Supports standard SurrogateVisionModel classifiers as well as CLIPZeroShotEvaluator.
    """
    if isinstance(model, CLIPZeroShotEvaluator):
        return model.compare_original_versus_cloaked(orig_img, adv_img)

    orig_pred = get_model_predictions(model, orig_img, device=device)
    adv_pred = get_model_predictions(model, adv_img, device=device)

    orig_class_idx = orig_pred["top1_class_idx"]
    adv_class_idx = adv_pred["top1_class_idx"]

    # Probability assigned to the ORIGINAL class by the cloaked prediction
    adv_orig_class_conf = float(adv_pred["all_probabilities"][orig_class_idx])
    conf_drop = float(orig_pred["top1_confidence"] - adv_orig_class_conf)

    # Embedding distance
    e_orig = orig_pred["embedding"]
    e_adv = adv_pred["embedding"]
    cos_dist = float(1.0 - F.cosine_similarity(e_adv, e_orig, dim=-1).item())
    l2_dist = float(torch.norm(e_adv - e_orig, p=2, dim=-1).item())

    prediction_changed = bool(orig_class_idx != adv_class_idx)

    return {
        "original_prediction": {
            "style_name": orig_pred["top1_style"],
            "class_idx": orig_pred["top1_class_idx"],
            "confidence": orig_pred["top1_confidence"],
            "confidence_percent": orig_pred["top1_confidence_percent"],
            "top5": orig_pred["top5_predictions"]
        },
        "cloaked_prediction": {
            "style_name": adv_pred["top1_style"],
            "class_idx": adv_pred["top1_class_idx"],
            "confidence": adv_pred["top1_confidence"],
            "confidence_percent": adv_pred["top1_confidence_percent"],
            "cloaked_confidence_in_orig_class": round(adv_orig_class_conf, 4),
            "top5": adv_pred["top5_predictions"]
        },
        "top1_prediction_changed": prediction_changed,
        "classification_disrupted": prediction_changed,
        "confidence_drop": round(conf_drop, 4),
        "confidence_drop_percent": round(conf_drop * 100.0, 2),
        "embedding_cosine_distance": round(cos_dist, 4),
        "embedding_l2_distance": round(l2_dist, 4),
        "evaluation_type": "classifier",
        "prompt_template": None,
        "scientific_disclaimer": (
            "A change in top-1 predicted style provides empirical evidence of neural disruption for this specific model; "
            "it is not mathematical proof against all unseen AI architectures or downstream tasks."
        )
    }


def evaluate_cross_model_transferability(
    orig_img: Union[Image.Image, np.ndarray, torch.Tensor],
    adv_img: Union[Image.Image, np.ndarray, torch.Tensor],
    surrogate_name: str = "resnet50",
    candidate_models: Optional[List[str]] = None,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates original vs cloaked asset across all available candidate models,
    distinguishing the surrogate model from black-box target architectures.
    Runs sequentially in inference mode to preserve GPU memory on 4GB VRAM.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = candidate_models or ["resnet50", "convnext_tiny", "vit_b_16", "clip_vit_b32"]

    results = []

    for name in candidates:
        is_surrogate = (name.lower() == surrogate_name.lower())
        is_clip = "clip" in name.lower()
        default_eval_type = "zero_shot_multimodal" if is_clip else "classifier"
        default_prompt = "a painting in {style} style" if is_clip else None

        model, load_msg = load_evaluation_model(name, device=device)

        if model is None:
            # Parse granular status from load_msg
            status = "unavailable"
            if load_msg.startswith("failed:"):
                status = "failed"
            elif load_msg.startswith("not_configured:"):
                status = "not_configured"
            elif load_msg.startswith("unavailable:"):
                status = "unavailable"

            results.append({
                "model_name": name,
                "is_surrogate": is_surrogate,
                "status": status,
                "status_detail": load_msg,
                "evaluation_type": default_eval_type,
                "prompt_template": default_prompt,
                "original_top1_class": None,
                "original_top1_confidence": None,
                "protected_top1_class": None,
                "protected_top1_confidence": None,
                "confidence_drop": None,
                "classification_disrupted": None,
                "embedding_cosine_distance": None,
                "embedding_l2_distance": None
            })
            continue

        try:
            cmp_res = compare_original_versus_cloaked(orig_img, adv_img, model, device=device)
            eval_type = cmp_res.get("evaluation_type", default_eval_type)
            prompt_tpl = cmp_res.get("prompt_template", default_prompt)

            results.append({
                "model_name": name,
                "is_surrogate": is_surrogate,
                "status": "evaluated",
                "status_detail": load_msg,
                "evaluation_type": eval_type,
                "prompt_template": prompt_tpl,
                "original_top1_class": cmp_res["original_prediction"]["style_name"],
                "original_top1_confidence": cmp_res["original_prediction"]["confidence"],
                "protected_top1_class": cmp_res["cloaked_prediction"]["style_name"],
                "protected_top1_confidence": cmp_res["cloaked_prediction"]["confidence"],
                "confidence_drop": cmp_res["confidence_drop"],
                "classification_disrupted": cmp_res["classification_disrupted"],
                "embedding_cosine_distance": cmp_res["embedding_cosine_distance"],
                "embedding_l2_distance": cmp_res["embedding_l2_distance"]
            })
        except Exception as e:
            print(f"[Transferability] Error evaluating {name}: {e}")
            results.append({
                "model_name": name,
                "is_surrogate": is_surrogate,
                "status": "failed",
                "status_detail": str(e),
                "evaluation_type": default_eval_type,
                "prompt_template": default_prompt,
                "original_top1_class": None,
                "original_top1_confidence": None,
                "protected_top1_class": None,
                "protected_top1_confidence": None,
                "confidence_drop": None,
                "classification_disrupted": None,
                "embedding_cosine_distance": None,
                "embedding_l2_distance": None
            })

        # Clear GPU cache after each model to ensure 4GB VRAM safety
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results
