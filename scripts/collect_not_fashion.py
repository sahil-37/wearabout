"""
Downloads not-fashion images from Google Open Images v7 for training
the binary fashion/not-fashion classifier.

Uses the Open Images CSV index — no extra tools needed, just requests.

Usage:
    python scripts/collect_not_fashion.py
    python scripts/collect_not_fashion.py --per-class 200 --out data/not_fashion
"""

import argparse
import csv
import io
import logging
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Open Images v7 validation image list (small, ~40k URLs, no login needed)
OPEN_IMAGES_CSV = "https://storage.googleapis.com/openimages/v7/oidv7-val-images-with-labels.csv"

# Open Images class names we want → local folder name
# These are all clearly non-fashion
TARGET_CLASSES = {
    "Food":          "food",
    "Fast food":     "food",
    "Fruit":         "food",
    "Dog":           "animals",
    "Cat":           "animals",
    "Bird":          "animals",
    "Car":           "vehicles",
    "Truck":         "vehicles",
    "Building":      "buildings",
    "Tree":          "nature",
    "Flower":        "nature",
    "Chair":         "household",
    "Table":         "household",
    "Laptop":        "electronics",
    "Mobile phone":  "electronics",
}

# Fallback: curated list of direct image URLs if CSV download fails
FALLBACK_URLS = {
    "food": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
    ],
}


def download_open_images_csv(url: str) -> list[dict]:
    """Download and parse the Open Images image list CSV."""
    logger.info(f"Downloading Open Images index from {url} …")
    try:
        r = requests.get(url, timeout=60, stream=True)
        r.raise_for_status()
        content = r.content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        logger.info(f"Downloaded {len(rows)} image records")
        return rows
    except Exception as e:
        logger.error(f"Failed to download Open Images CSV: {e}")
        return []


def download_image(url: str, dest: Path) -> bool:
    """Download a single image to dest path."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "image" not in content_type and not url.lower().endswith((".jpg", ".jpeg", ".png")):
            return False
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def collect_from_open_images(out_dir: Path, per_class: int) -> dict[str, int]:
    """Download not-fashion images by class from Open Images."""
    rows = download_open_images_csv(OPEN_IMAGES_CSV)
    if not rows:
        return {}

    # group image URLs by label
    by_class: dict[str, list[str]] = {v: [] for v in set(TARGET_CLASSES.values())}
    for row in rows:
        label = row.get("LabelName", row.get("label_name", ""))
        if label in TARGET_CLASSES:
            folder = TARGET_CLASSES[label]
            img_url = row.get("ImageURL", row.get("image_url", ""))
            if img_url:
                by_class[folder].append(img_url)

    counts = {}
    for folder, urls in by_class.items():
        if not urls:
            continue
        dest_dir = out_dir / folder
        dest_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(dest_dir.glob("*.jpg")))
        needed = per_class - existing
        if needed <= 0:
            logger.info(f"  {folder}: already has {existing} images, skipping")
            counts[folder] = existing
            continue

        random.shuffle(urls)
        downloaded = 0
        for url in urls:
            if downloaded >= needed:
                break
            fname = dest_dir / f"{folder}_{existing + downloaded:04d}.jpg"
            if download_image(url, fname):
                downloaded += 1
                if downloaded % 20 == 0:
                    logger.info(f"  {folder}: {existing + downloaded}/{per_class}")
            time.sleep(0.05)  # polite delay

        counts[folder] = existing + downloaded
        logger.info(f"  {folder}: {existing + downloaded} total")

    return counts


def collect_from_imagenet_urls(out_dir: Path, per_class: int) -> dict[str, int]:
    """
    Downloads not-fashion images using the COCO 2017 val dataset —
    80 everyday object categories, freely available, no login needed.
    128 images (~25MB zip).
    """
    import zipfile
    import urllib.request

    logger.info("Downloading COCO 2017 val sample (non-fashion objects) …")

    coco_url  = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path  = out_dir.parent / "coco_val_sample.zip"
    coco_dir  = out_dir.parent / "coco_val_sample"

    # download zip if not already present
    if not coco_dir.exists():
        logger.info(f"Downloading from {coco_url} (this may take a minute) …")
        try:
            urllib.request.urlretrieve(coco_url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(out_dir.parent)
            zip_path.unlink(missing_ok=True)
            logger.info("COCO val downloaded and extracted")
        except Exception as e:
            logger.error(f"COCO download failed: {e}")
            return {}

    # copy a random sample of COCO images as "not_fashion"
    # handle both extraction layouts
    val_dir  = coco_dir / "val2017"
    all_imgs = list(val_dir.glob("*.jpg")) if val_dir.exists() else list(coco_dir.glob("*.jpg"))
    # also check if it extracted directly into the parent
    if not all_imgs:
        all_imgs = list((out_dir.parent / "val2017").glob("*.jpg"))

    if not all_imgs:
        logger.error("No COCO images found after extraction")
        return {}

    dest_dir = out_dir / "coco_objects"
    dest_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(all_imgs)
    target = min(per_class * 4, len(all_imgs))  # take up to 4x per_class total
    copied = 0
    for i, src in enumerate(all_imgs[:target]):
        dst = dest_dir / f"coco_{i:04d}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    logger.info(f"  coco_objects: {copied} images")
    return {"coco_objects": copied}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=150,
                        help="Target images per category (default 150)")
    parser.add_argument("--out", type=str,
                        default="data/classifier/not_fashion",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting not-fashion images → {out_dir}")
    logger.info(f"Target: {args.per_class} images per category")

    # try Open Images first, fallback to Wikimedia
    counts = collect_from_open_images(out_dir, args.per_class)
    total = sum(counts.values())

    if total < 50:
        logger.warning("Open Images download yielded too few images, using fallback")
        counts = collect_from_imagenet_urls(out_dir, args.per_class)
        total = sum(counts.values())

    logger.info(f"\nDone — {total} not-fashion images collected")
    for folder, count in sorted(counts.items()):
        logger.info(f"  {folder}: {count}")
    logger.info(f"\nNext: python scripts/train_classifier.py")


if __name__ == "__main__":
    main()
