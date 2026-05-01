"""
Evaluation: Recall@K and Precision@K against the annotated dataset.

The annotated data ships with a 462/10 train/val split that is too small
for meaningful evaluation. This script re-splits all 472 annotated images
80/20 (stratified by primary category) into its own eval set, keeping the
original files untouched.

Metrics computed:
  - Recall@K    fraction of queries where ≥1 correct category appears in top-K
  - Precision@K fraction of top-K results that match a correct category
  - Both reported globally AND broken down per category

Usage:
    python scripts/evaluate.py [--k 5 10 20] [--val-ratio 0.2] [--seed 42]
"""

import argparse
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Set before any native library loads to avoid OpenMP duplicate-init segfault
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch          # must load before faiss to avoid OpenMP segfault on macOS
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings as _settings

_ANNOTATED_ROOT  = Path(_settings.ANNOTATED_DATA_ROOT) if _settings.ANNOTATED_DATA_ROOT else None

ANNOTATED_IMAGES = _ANNOTATED_ROOT / "images/train" if _ANNOTATED_ROOT else Path(".")
ANNOTATED_LABELS = _ANNOTATED_ROOT / "labels/train" if _ANNOTATED_ROOT else Path(".")
EXTRA_VAL_IMAGES = _ANNOTATED_ROOT / "images/val"   if _ANNOTATED_ROOT else Path(".")
EXTRA_VAL_LABELS = _ANNOTATED_ROOT / "labels/val"   if _ANNOTATED_ROOT else Path(".")

API_ROOT     = Path(__file__).resolve().parent.parent
FEATURES_PKL = API_ROOT / "data" / "features_normalized.pkl"

CLASS_NAMES  = ["topwear", "bottomwear", "footwear", "eyewear", "handbag"]


# ── feature extraction ────────────────────────────────────────────────────────

import torchvision.models as models
import torchvision.transforms as T

_device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_transform = T.Compose([
    T.ToPILImage(), T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def _build_model():
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()
    m.eval()
    return m.to(_device)

_model = _build_model()


def extract_features(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(2048, dtype=np.float32)
    rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = _transform(rgb).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _model(tensor)
    vec  = feat.squeeze().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── label parsing ─────────────────────────────────────────────────────────────

def parse_label(label_path: Path) -> List[str]:
    """Return unique category names present in a YOLO label file."""
    cats = set()
    try:
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                cid = int(parts[0])
                if cid < len(CLASS_NAMES):
                    cats.add(CLASS_NAMES[cid])
    except Exception:
        pass
    return list(cats)


def primary_category(label_path: Path) -> str:
    """Most common class in the label file (for stratification)."""
    from collections import Counter
    counts = Counter()
    try:
        for line in label_path.read_text().strip().splitlines():
            parts = line.split()
            if parts:
                cid = int(parts[0])
                if cid < len(CLASS_NAMES):
                    counts[CLASS_NAMES[cid]] += 1
    except Exception:
        pass
    return counts.most_common(1)[0][0] if counts else "topwear"


# ── dataset building ──────────────────────────────────────────────────────────

def collect_annotated_pairs() -> List[Tuple[Path, Path]]:
    """Return (image_path, label_path) for every annotated image."""
    pairs = []

    for img_path in sorted(ANNOTATED_IMAGES.glob("*.jpg")) + sorted(ANNOTATED_IMAGES.glob("*.jpg.jpg")):
        stem = img_path.name
        label = ANNOTATED_LABELS / (stem + ".txt")
        if not label.exists():
            label = ANNOTATED_LABELS / (stem.replace(".jpg.jpg", ".jpg") + ".txt")
        if label.exists():
            pairs.append((img_path, label))

    # include the original val set
    for img_path in sorted(EXTRA_VAL_IMAGES.glob("*.jpg")) + sorted(EXTRA_VAL_IMAGES.glob("*.jpg.jpg")):
        stem = img_path.name
        label = EXTRA_VAL_LABELS / (stem + ".txt")
        if not label.exists():
            label = EXTRA_VAL_LABELS / (stem.replace(".jpg.jpg", ".jpg") + ".txt")
        if label.exists():
            pairs.append((img_path, label))

    return pairs


def stratified_val_split(
    pairs: List[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> List[Tuple[Path, Path]]:
    """
    Stratified split by the SET of categories present in each image.
    Groups images by their category combination (e.g. topwear+bottomwear+footwear),
    then takes val_ratio from each group so every combination is represented.
    Falls back to a simple random split for rare combinations.
    """
    by_combo: Dict[str, list] = defaultdict(list)
    for pair in pairs:
        cats  = sorted(parse_label(pair[1]))
        combo = "+".join(cats) if cats else "unknown"
        by_combo[combo].append(pair)

    rng = random.Random(seed)
    val_pairs = []
    for combo, combo_pairs in sorted(by_combo.items()):
        rng.shuffle(combo_pairs)
        n_val = max(1, round(len(combo_pairs) * val_ratio))
        val_pairs.extend(combo_pairs[:n_val])
        logger.info(f"  [{combo}] total={len(combo_pairs)} val={n_val}")

    return val_pairs


# ── FAISS index ───────────────────────────────────────────────────────────────

def load_catalog() -> Tuple[np.ndarray, List[str]]:
    with open(FEATURES_PKL, "rb") as f:
        data = pickle.load(f)
    return data["features"].astype(np.float32), data["categories"]  # image_paths unused in eval


def build_faiss_index(features: np.ndarray):
    import faiss  # imported here so env var is set first
    idx = faiss.IndexFlatIP(features.shape[1])
    idx.add(features)
    return idx


# ── core evaluation ───────────────────────────────────────────────────────────

def evaluate(k_values: List[int], val_ratio: float = 0.2, seed: int = 42) -> Dict:

    features, cat_labels = load_catalog()
    index = build_faiss_index(features)
    logger.info(f"FAISS index: {index.ntotal} vectors")

    all_pairs  = collect_annotated_pairs()
    logger.info(f"Total annotated images: {len(all_pairs)}")
    val_pairs  = stratified_val_split(all_pairs, val_ratio, seed)
    logger.info(f"Val set after {int(val_ratio*100)}/{ 100 - int(val_ratio*100)} split: {len(val_pairs)} images")

    # accumulators: global + per-category
    recall_hits    = {k: 0 for k in k_values}
    precision_sum  = {k: 0.0 for k in k_values}
    total          = 0

    cat_recall_hits   = {cat: {k: 0 for k in k_values} for cat in CLASS_NAMES}
    cat_precision_sum = {cat: {k: 0.0 for k in k_values} for cat in CLASS_NAMES}
    cat_total         = {cat: 0 for cat in CLASS_NAMES}

    max_k = max(k_values)

    for img_path, label_path in val_pairs:
        gt_cats = parse_label(label_path)
        if not gt_cats:
            continue

        query_vec       = extract_features(str(img_path)).reshape(1, -1)
        _, idxs = index.search(query_vec, min(max_k, index.ntotal))
        result_cats = [cat_labels[i] for i in idxs[0] if i < len(cat_labels)]

        primary = primary_category(label_path)
        total += 1
        cat_total[primary] += 1

        for k in k_values:
            top_k = result_cats[:k]

            # Recall@K — did any correct category appear?
            hit = int(any(c in gt_cats for c in top_k))
            recall_hits[k]        += hit
            cat_recall_hits[primary][k] += hit

            # Precision@K — what fraction of top-K is correct?
            prec = sum(1 for c in top_k if c in gt_cats) / k
            precision_sum[k]           += prec
            cat_precision_sum[primary][k] += prec

    # ── assemble metrics dict ─────────────────────────────────────────────────
    metrics = {"total_queries": total}

    logger.info("\n── GLOBAL ──────────────────────────────────")
    for k in k_values:
        recall = recall_hits[k] / total if total else 0.0
        prec   = precision_sum[k] / total if total else 0.0
        metrics[f"recall_at_{k}"]    = round(recall, 4)
        metrics[f"precision_at_{k}"] = round(prec, 4)
        logger.info(f"  Recall@{k}:    {recall:.4f}  ({recall_hits[k]}/{total})")
        logger.info(f"  Precision@{k}: {prec:.4f}")

    logger.info("\n── PER CATEGORY ────────────────────────────")
    for cat in CLASS_NAMES:
        n = cat_total[cat]
        if n == 0:
            continue
        logger.info(f"  [{cat}]  n={n}")
        for k in k_values:
            r = cat_recall_hits[cat][k] / n
            p = cat_precision_sum[cat][k] / n
            metrics[f"{cat}_recall_at_{k}"]    = round(r, 4)
            metrics[f"{cat}_precision_at_{k}"] = round(p, 4)
            logger.info(f"    Recall@{k}={r:.4f}  Precision@{k}={p:.4f}")

    return metrics


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",         type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    evaluate(args.k, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
