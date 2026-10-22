import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from src.dataset import WikiArtDataset
from src.preprocessing import create_artist_aware_splits
from src.models import get_model
from src.embeddings import get_embedding, compare_embeddings
from src.adversarial_cloak import AdversarialCloakingEngine
from src.metrics import PerceptualQualityEvaluator

def run_verification():
    print("=" * 80)
    print("         AI-PROTECT COMPLETE PIPELINE VERIFICATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Hardware Check: Using {device.type.upper()}")

    # 1. Dataset Streaming & Indexing
    print("[2/6] Testing WikiArtDataset (FAST_DEV_MODE: 100 samples)...")
    ds = WikiArtDataset(data_dir="wikiart-all", fast_dev_mode=True, fast_dev_samples=100)
    train_ds, val_ds, test_ds = create_artist_aware_splits(ds, artist_aware=True)
    print(f"      Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 2. Model & Embedding Hooks
    print("[3/6] Testing Model Architectures & Feature Embeddings...")
    model = get_model("resnet50", num_classes=ds.num_classes, pretrained=False, device=device)
    sample_img, true_label, meta = test_ds[0]
    emb_orig = get_embedding(sample_img, model=model, device=device)
    print(f"      Feature embedding extracted. Shape: {list(emb_orig.shape)}")

    # 3. Adversarial Cloaking Engine
    print("[4/6] Generating Bounded Adversarial Perturbation (PGD, eps=8/255)...")
    engine = AdversarialCloakingEngine(model=model, device=device)
    x_adv, delta, stats = engine.generate_cloak(
        image=sample_img,
        epsilon=8.0 / 255.0,
        iterations=10,
        step_size=2.0 / 255.0,
        objective="classification_disruption"
    )

    eps_bound = stats["epsilon_configured"]
    linf_measured = stats["linf_norm_measured"]
    assert linf_measured <= eps_bound + 1e-5, f"L-inf {linf_measured} exceeded bound {eps_bound}!"
    print(f"      L-infinity constraint strictly satisfied: {linf_measured:.5f} <= {eps_bound:.5f}")
    print(f"      Original Class: {stats['original_class_idx']}, Cloaked Class: {stats['cloaked_pred_class_idx']}")

    # 4. Embedding Shift Measurement
    print("[5/6] Measuring Deep Feature Representation Shift...")
    emb_cloaked = get_embedding(x_adv, model=model, device=device)
    dist_metrics = compare_embeddings(emb_orig, emb_cloaked)
    print(f"      Cosine Distance Shift : {dist_metrics['cosine_distance']:.4f}")
    print(f"      Euclidean L2 Distance : {dist_metrics['l2_distance']:.4f}")

    # 5. Perceptual Quality
    print("[6/6] Computing Perceptual Fidelity Metrics (SSIM & PSNR)...")
    evaluator = PerceptualQualityEvaluator(device=device)
    q_metrics = evaluator.evaluate_pair(sample_img, x_adv)
    print(f"      PSNR : {q_metrics['psnr_db']} dB")
    print(f"      SSIM : {q_metrics['ssim']}")

    print("=" * 80)
    print("[PASS] ALL MODULES AND VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_verification()
