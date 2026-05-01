"""
Build catalog: converts the raw metadata.json from the product library into
the normalized format the API expects, and recomputes features.pkl over all
2306 product images using color histogram extraction.

Usage:
    python scripts/build_catalog.py
"""

import json
import logging
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings as _settings

API_ROOT    = Path(__file__).resolve().parent.parent
DATA_DIR    = API_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

PRODUCT_DIR = Path(_settings.CATALOG_IMAGE_ROOT) if _settings.CATALOG_IMAGE_ROOT else None
RAW_META    = PRODUCT_DIR / "metadata.json" if PRODUCT_DIR else None

if not PRODUCT_DIR or not PRODUCT_DIR.exists():
    logger.error(
        "CATALOG_IMAGE_ROOT not set or does not exist. "
        "Set it in your .env file: CATALOG_IMAGE_ROOT=/path/to/Product"
    )
    raise SystemExit(1)

OUT_META       = DATA_DIR / "metadata.json"
OUT_FEATURES   = DATA_DIR / "features_normalized.pkl"

CATEGORIES     = ["topwear", "bottomwear", "footwear", "eyewear", "handbag"]
INPUT_SIZE     = (224, 224)
FEATURE_DIM    = 2048


# ── feature extraction (ResNet-50 via PyTorch, histogram fallback) ────────────

_torch_model   = None
_torch_transform = None
_torch_device  = None


def _get_torch_model():
    global _torch_model, _torch_transform, _torch_device
    if _torch_model is not None:
        return _torch_model, _torch_transform, _torch_device

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _torch_model, _torch_transform, _torch_device = model, transform, device
    logger.info(f"ResNet-50 loaded on {device}")
    return model, transform, device


def extract_features(img_path: str) -> np.ndarray:
    """ResNet-50 features with histogram fallback."""
    try:
        import torch
        model, transform, device = _get_torch_model()
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor  = transform(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor)
        vec = feat.squeeze().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception as e:
        logger.warning(f"ResNet fallback to histogram for {img_path}: {e}")
        return _histogram_features(img_path)


def _histogram_features(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path)
    if img is None:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    img = cv2.resize(img, INPUT_SIZE)
    features = []
    for i in range(3):
        features.extend(cv2.calcHist([img], [i], None, [64], [0, 256]).flatten())
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for i in range(3):
        features.extend(cv2.calcHist([hsv], [i], None, [64], [0, 256]).flatten())
    features.extend(cv2.resize(img, (32, 32)).flatten()[:1024])
    arr = np.array(features, dtype=np.float32)
    arr = arr[:FEATURE_DIM] if len(arr) >= FEATURE_DIM else np.pad(arr, (0, FEATURE_DIM - len(arr)))
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


# ── metadata rebuild ───────────────────────────────────────────────────────────

def build_metadata() -> list:
    with open(RAW_META) as f:
        raw = json.load(f)

    n = len(raw["link"])
    records = []

    for i in range(n):
        category = raw["category"][i].lower()
        name     = raw["name"][i].replace("&amp;", "&").replace("&amp_", "&")

        # filenames on disk use "&amp_" (underscore), not "&amp;" (semicolon)
        img_filename = raw["name"][i].replace("&amp;", "&amp_") + ".jpg"
        img_path     = str(PRODUCT_DIR / category / img_filename)

        records.append({
            "id":          i,
            "name":        name,
            "brand":       raw["title"][i],
            "price":       raw["price"][i],
            "category":    category,
            "gender":      raw["sex"][i].lower(),
            "product_url": raw["link"][i],
            "image_path":  img_path,
        })

    logger.info(f"Built metadata for {len(records)} products")
    return records


# ── feature recompute ─────────────────────────────────────────────────────────

def build_features(metadata: list) -> dict:
    features    = []
    image_paths = []
    categories  = []
    missing     = 0

    for item in tqdm(metadata, desc="Extracting features"):
        img_path = item["image_path"]
        if not os.path.exists(img_path):
            missing += 1
            features.append(np.zeros(FEATURE_DIM, dtype=np.float32))
        else:
            features.append(extract_features(img_path))

        image_paths.append(img_path)
        categories.append(item["category"])

    features_array = np.array(features, dtype=np.float32)

    # L2-normalise so cosine ≈ euclidean distance
    norms = np.linalg.norm(features_array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    features_normalized = features_array / norms

    logger.info(f"Extracted features for {len(features)} items ({missing} missing images)")

    return {
        "features":    features_normalized,
        "image_paths": image_paths,
        "categories":  categories,
        "num_images":  len(features),
        "feature_dim": FEATURE_DIM,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("Step 1/3 — Building metadata")
    metadata = build_metadata()
    with open(OUT_META, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata → {OUT_META}")

    logger.info("Step 2/3 — Extracting features")
    features_data = build_features(metadata)
    with open(OUT_FEATURES, "wb") as f:
        pickle.dump(features_data, f)
    logger.info(f"Saved features → {OUT_FEATURES}")

    logger.info("Step 3/3 — Summary")
    by_category = {}
    for item in metadata:
        by_category.setdefault(item["category"], 0)
        by_category[item["category"]] += 1
    for cat, count in sorted(by_category.items()):
        logger.info(f"  {cat}: {count} items")
    logger.info(f"Feature matrix shape: {features_data['features'].shape}")
    logger.info("Done. Run `python main.py` to start the API.")


if __name__ == "__main__":
    main()
