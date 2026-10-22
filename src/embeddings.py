import argparse
import io
import sys
from pathlib import Path
from typing import Union, Optional, Tuple, Dict, Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import WikiArtDataset
from src.preprocessing import get_transforms, create_artist_aware_splits
from src.models import get_model, SurrogateVisionModel


def get_embedding(
    image: Union[Image.Image, torch.Tensor, str, Path],
    model: Optional[SurrogateVisionModel] = None,
    device: Optional[torch.device] = None,
    model_name: str = "resnet50",
    image_size: int = 224,
    normalize: bool = True
) -> torch.Tensor:
    """
    Standard interface to extract a deep feature embedding vector from an image.
    
    Accepts:
      - PIL Image
      - Image File Path
      - Preprocessed torch.Tensor [C, H, W] or [B, C, H, W]
    
    Returns:
      - 1D or 2D torch.Tensor feature vector (L2 normalized by default).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = get_model(model_name=model_name, num_classes=0, pretrained=True, device=device)
    model.eval()

    # Preprocess if image is PIL or file path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        transform = get_transforms(image_size=image_size, is_training=False, model_name=model_name)
        tensor_img = transform(image).unsqueeze(0).to(device)
    elif isinstance(image, torch.Tensor):
        if image.ndim == 3:
            tensor_img = image.unsqueeze(0).to(device)
        else:
            tensor_img = image.to(device)
    else:
        raise TypeError(f"Unsupported image input type: {type(image)}")

    with torch.no_grad():
        emb = model.get_embedding(tensor_img, normalize=normalize)

    if emb.size(0) == 1:
        return emb.squeeze(0)
    return emb


def compare_embeddings(
    emb_orig: torch.Tensor,
    emb_cloaked: torch.Tensor
) -> Dict[str, float]:
    """
    Calculates metric distances between original and cloaked feature embeddings.
    
    Returns:
      - cosine_distance: 1 - cosine_similarity (Range: [0, 2], higher = more disrupted)
      - l2_distance: Euclidean distance
      - dot_product: Dot product similarity
    """
    if emb_orig.ndim == 1:
        emb_orig = emb_orig.unsqueeze(0)
    if emb_cloaked.ndim == 1:
        emb_cloaked = emb_cloaked.unsqueeze(0)

    # Ensure L2 normalized
    e1 = F.normalize(emb_orig, p=2, dim=-1)
    e2 = F.normalize(emb_cloaked, p=2, dim=-1)

    cos_sim = F.cosine_similarity(e1, e2, dim=-1).item()
    cos_dist = 1.0 - cos_sim
    l2_dist = torch.norm(e1 - e2, p=2, dim=-1).item()
    dot_prod = torch.sum(e1 * e2, dim=-1).item()

    return {
        "cosine_similarity": float(cos_sim),
        "cosine_distance": float(cos_dist),
        "l2_distance": float(l2_dist),
        "dot_product": float(dot_prod),
    }


def extract_and_save_test_embeddings(
    model: SurrogateVisionModel,
    loader: DataLoader,
    device: torch.device,
    save_path: Path = Path("models/embeddings/test_embeddings.pt")
) -> Dict[str, Any]:
    """
    Extracts and persists all test set embeddings for downstream research.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    all_embs = []
    all_targets = []
    all_ids = []
    all_artists = []

    print(f"[Embeddings] Extracting deep feature representations...")
    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            embs = model.get_embedding(images, normalize=True)
            all_embs.append(embs.cpu())
            all_targets.extend(targets.numpy())
            all_ids.extend(meta.get("id", []))
            all_artists.extend(meta.get("artistName", []))

    cat_embs = torch.cat(all_embs, dim=0)
    data_dict = {
        "embeddings": cat_embs,
        "targets": np.array(all_targets),
        "ids": all_ids,
        "artists": all_artists,
        "feature_dim": cat_embs.size(-1),
        "total_samples": len(cat_embs),
    }

    torch.save(data_dict, save_path)
    print(f"[Embeddings] Saved {len(cat_embs)} embeddings (dim: {cat_embs.size(-1)}) to: {save_path}")
    return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and Compare Deep Feature Embeddings")
    parser.add_argument("--data_dir", type=str, default="wikiart-all")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--fast_dev_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model = get_model(ckpt.get("model_name", "resnet50"), num_classes=ckpt.get("num_classes", 20), pretrained=False, device=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = get_model("resnet50", num_classes=10, pretrained=True, device=device)

    dataset = WikiArtDataset(data_dir=args.data_dir, fast_dev_mode=True, fast_dev_samples=100)
    _, _, test_ds = create_artist_aware_splits(dataset, artist_aware=True)
    loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    extract_and_save_test_embeddings(model, loader, device)

    # Test single image embedding
    sample_img, _, _ = dataset[0]
    emb = get_embedding(sample_img, model=model, device=device)
    print(f"[Verification] Single image embedding extracted. Shape: {emb.shape}")
