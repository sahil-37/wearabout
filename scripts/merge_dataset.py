"""
Merge the Roboflow dataset into our Annotated_Data directory.

Remaps 12 Roboflow classes → our 5-class schema:
  topwear=0, bottomwear=1, footwear=2, eyewear=3, handbag=4

Classes with no mapping (e.g. dresses that are ambiguous) are kept
as topwear since they cover the upper body in the image.

Usage:
    python scripts/merge_dataset.py [--limit N]   # N images per split
    python scripts/merge_dataset.py               # all 2071 images
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_ROOT      = Path(__file__).resolve().parent.parent
ROBOFLOW_DIR  = API_ROOT / "roboflow_dataset"
ANNOTATED_DIR = Path("/Users/sahil/Desktop/AI Ebooks/Buy Me That Look/Annotated_Data")

# Roboflow class_id → our class_id
# 0=Tshirt, 1=jacket, 2=long-dress, 3=long-skirt, 4=midi-dress,
# 5=midi-skirt, 6=pants, 7=shirt, 8=short, 9=short-dress,
# 10=short-skirt, 11=sweater
REMAP = {
    0:  0,   # Tshirt      → topwear
    1:  0,   # jacket      → topwear
    2:  0,   # long-dress  → topwear (full-length, treat as top)
    3:  1,   # long-skirt  → bottomwear
    4:  0,   # midi-dress  → topwear
    5:  1,   # midi-skirt  → bottomwear
    6:  1,   # pants       → bottomwear
    7:  0,   # shirt       → topwear
    8:  1,   # short       → bottomwear
    9:  0,   # short-dress → topwear
    10: 1,   # short-skirt → bottomwear
    11: 0,   # sweater     → topwear
}


def remap_label_file(src: Path, dst: Path) -> bool:
    """Read a label file, remap class IDs, write to dst. Returns False if empty."""
    lines_out = []
    for line in src.read_text().strip().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        new_cls = REMAP.get(old_cls)
        if new_cls is None:
            continue                        # skip unmapped classes
        lines_out.append(f"{new_cls} {' '.join(parts[1:])}")

    if not lines_out:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out))
    return True


def merge(limit: int | None, seed: int = 42):
    src_img_dir   = ROBOFLOW_DIR / "train" / "images"
    src_label_dir = ROBOFLOW_DIR / "train" / "labels"

    dst_img_dir   = ANNOTATED_DIR / "images" / "train"
    dst_label_dir = ANNOTATED_DIR / "labels" / "train"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    all_images = sorted(src_img_dir.glob("*.jpg"))
    logger.info(f"Roboflow train images available: {len(all_images)}")

    if limit:
        rng = random.Random(seed)
        all_images = rng.sample(all_images, min(limit, len(all_images)))
        logger.info(f"Using subset of {len(all_images)} images (--limit {limit})")

    copied = skipped = 0
    class_counts = {0: 0, 1: 0}

    for img_path in all_images:
        label_src = src_label_dir / (img_path.stem + ".txt")
        if not label_src.exists():
            skipped += 1
            continue

        label_dst = dst_label_dir / (img_path.stem + "_rf.txt")
        img_dst   = dst_img_dir   / (img_path.stem + "_rf.jpg")

        ok = remap_label_file(label_src, label_dst)
        if not ok:
            skipped += 1
            continue

        shutil.copy2(img_path, img_dst)

        # count classes added
        for line in label_dst.read_text().strip().splitlines():
            cls = int(line.split()[0])
            class_counts[cls] = class_counts.get(cls, 0) + 1

        copied += 1

    logger.info(f"Merged:  {copied} images")
    logger.info(f"Skipped: {skipped} (no label or empty after remap)")
    logger.info(f"Class annotations added:")
    names = {0: "topwear", 1: "bottomwear", 2: "footwear", 3: "eyewear", 4: "handbag"}
    for cid, count in sorted(class_counts.items()):
        logger.info(f"  {names[cid]}: {count} boxes")

    # count total annotated images now
    total = len(list((ANNOTATED_DIR / "images" / "train").glob("*.jpg")))
    logger.info(f"Total train images in Annotated_Data: {total}")
    logger.info("Done. Run `make retrain` to rebuild and evaluate.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max images to merge (default: all 2071)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    merge(args.limit, args.seed)


if __name__ == "__main__":
    main()
