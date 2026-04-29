"""
Retraining pipeline — runs the full catalog rebuild, evaluates quality,
logs everything to MLflow, and promotes the new catalog if metrics improve.

Usage:
    python scripts/retrain.py                    # full run
    python scripts/retrain.py --dry-run          # evaluate only, no promotion
    python scripts/retrain.py --force-promote    # promote even if metrics regress

MLflow UI:
    mlflow ui --backend-store-uri ./mlruns
    open http://localhost:5000
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch          # must load before faiss to avoid OpenMP segfault on macOS
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

API_ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR       = API_ROOT / "data"
MLRUNS_DIR     = API_ROOT / "mlruns"
VERSIONS_DIR   = DATA_DIR / "versions"
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_PATH  = DATA_DIR / "features_normalized.pkl"
METADATA_PATH  = DATA_DIR / "metadata.json"
BASELINE_FILE  = DATA_DIR / "baseline_metrics.json"

EXPERIMENT_NAME = "wearabout"
K_VALUES        = [5, 10, 20]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        with open(BASELINE_FILE) as f:
            return json.load(f)
    return {}


def save_baseline(metrics: dict):
    with open(BASELINE_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Baseline updated → {BASELINE_FILE}")


def version_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def archive_current(tag: str):
    """Copy current catalog to versions/ before overwriting."""
    for src in [FEATURES_PATH, METADATA_PATH]:
        if src.exists():
            dst = VERSIONS_DIR / f"{src.stem}_{tag}{src.suffix}"
            shutil.copy2(src, dst)
            logger.info(f"Archived {src.name} → {dst.name}")


def promote(tmp_features: Path, tmp_metadata: Path):
    shutil.move(str(tmp_features), str(FEATURES_PATH))
    shutil.move(str(tmp_metadata), str(METADATA_PATH))
    logger.info("New catalog promoted to production")


# ── step 1: build catalog ─────────────────────────────────────────────────────

def run_build() -> dict:
    """Run build_catalog and return summary stats."""
    import importlib.util, sys

    build_script = API_ROOT / "scripts" / "build_catalog.py"
    spec = importlib.util.spec_from_file_location("build_catalog", build_script)
    bc   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)

    t0 = time.time()

    # write to tmp files so we don't overwrite production until eval passes
    tmp_meta     = DATA_DIR / "metadata_tmp.json"
    tmp_features = DATA_DIR / "features_tmp.pkl"

    bc.OUT_META     = tmp_meta
    bc.OUT_FEATURES = tmp_features

    metadata      = bc.build_metadata()
    features_data = bc.build_features(metadata)

    with open(tmp_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(tmp_features, "wb") as f:
        pickle.dump(features_data, f)

    elapsed = time.time() - t0

    by_category = {}
    for item in metadata:
        by_category.setdefault(item["category"], 0)
        by_category[item["category"]] += 1

    return {
        "num_products":      len(metadata),
        "feature_dim":       int(features_data["features"].shape[1]),
        "build_time_sec":    round(elapsed, 2),
        "by_category":       by_category,
        "tmp_features_path": tmp_features,
        "tmp_metadata_path": tmp_meta,
    }


# ── step 2: evaluate ──────────────────────────────────────────────────────────

def run_evaluate(tmp_features_path: Path) -> dict:
    """Evaluate the tmp feature store against the annotated val set."""
    import importlib.util

    # temporarily swap features path for evaluate.py
    eval_script = API_ROOT / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate", eval_script)
    ev   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    # monkey-patch path so evaluator uses tmp file
    ev.FEATURES_PKL = tmp_features_path

    return ev.evaluate(K_VALUES)


# ── step 3: compare & decide ──────────────────────────────────────────────────

def should_promote(new_metrics: dict, baseline: dict, force: bool) -> bool:
    if force:
        logger.info("--force-promote set, skipping metric comparison")
        return True
    if not baseline:
        logger.info("No baseline found, promoting automatically")
        return True

    primary_key = f"recall_at_{K_VALUES[1]}"   # recall@10
    new_val  = new_metrics.get(primary_key, 0)
    base_val = baseline.get(primary_key, 0)

    logger.info(f"Recall@10 — new: {new_val:.4f}  baseline: {base_val:.4f}")

    if new_val >= base_val:
        logger.info("New catalog meets or beats baseline → promoting")
        return True
    else:
        logger.warning(f"New catalog regresses recall@10 by {base_val - new_val:.4f} → NOT promoting")
        return False


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",       action="store_true", help="Evaluate only, do not promote")
    parser.add_argument("--force-promote", action="store_true", help="Promote even if metrics regress")
    args = parser.parse_args()

    import mlflow
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(EXPERIMENT_NAME)

    tag      = version_tag()
    baseline = load_baseline()

    with mlflow.start_run(run_name=f"retrain_{tag}"):

        # ── params ────────────────────────────────────────────────────────────
        mlflow.log_params({
            "feature_extractor": "resnet50_imagenet",
            "similarity_metric": "cosine_faiss_ip",
            "k_values":          str(K_VALUES),
            "dry_run":           args.dry_run,
            "force_promote":     args.force_promote,
            "run_tag":           tag,
        })

        # ── step 1: build ─────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1/3 — Building catalog")
        logger.info("=" * 60)
        build_stats = run_build()

        mlflow.log_metrics({
            "num_products":   build_stats["num_products"],
            "build_time_sec": build_stats["build_time_sec"],
        })
        for cat, count in build_stats["by_category"].items():
            mlflow.log_metric(f"count_{cat}", count)

        logger.info(f"Built {build_stats['num_products']} products in {build_stats['build_time_sec']}s")

        # ── step 2: evaluate ──────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2/3 — Evaluating")
        logger.info("=" * 60)
        metrics = run_evaluate(build_stats["tmp_features_path"])
        mlflow.log_metrics(metrics)

        # log baseline comparison
        for k in K_VALUES:
            key = f"recall_at_{k}"
            if key in baseline:
                delta = metrics.get(key, 0) - baseline[key]
                mlflow.log_metric(f"delta_{key}", round(delta, 4))

        # ── step 3: promote or discard ────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3/3 — Promotion decision")
        logger.info("=" * 60)

        promoted = False
        if not args.dry_run and should_promote(metrics, baseline, args.force_promote):
            archive_current(tag)
            promote(
                build_stats["tmp_features_path"],
                build_stats["tmp_metadata_path"],
            )
            save_baseline(metrics)
            promoted = True
        else:
            # clean up tmp files
            Path(build_stats["tmp_features_path"]).unlink(missing_ok=True)
            Path(build_stats["tmp_metadata_path"]).unlink(missing_ok=True)
            if args.dry_run:
                logger.info("Dry run — tmp files discarded")

        mlflow.log_param("promoted", promoted)

        # ── log artifacts ─────────────────────────────────────────────────────
        if promoted and FEATURES_PATH.exists():
            mlflow.log_artifact(str(FEATURES_PATH), artifact_path="catalog")
            mlflow.log_artifact(str(METADATA_PATH), artifact_path="catalog")

        # ── summary ───────────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        for k in K_VALUES:
            key = f"recall_at_{k}"
            logger.info(f"  {key}: {metrics.get(key, 'n/a')}")
        logger.info(f"  Promoted: {promoted}")
        logger.info(f"  MLflow run: {mlflow.active_run().info.run_id}")
        logger.info("=" * 60)
        logger.info("View results:  mlflow ui --backend-store-uri ./mlruns")


if __name__ == "__main__":
    main()
