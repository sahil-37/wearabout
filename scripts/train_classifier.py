"""
Fine-tunes MobileNetV3-Small as a binary fashion/not-fashion classifier.

Fashion data:  data/classifier/fashion/     (copied from your product catalog)
Not-fashion:   data/classifier/not_fashion/ (downloaded by collect_not_fashion.py)

Usage:
    python scripts/train_classifier.py
    python scripts/train_classifier.py --epochs 15 --batch-size 32

Output:
    models/fashion_gate.pth   — best checkpoint (used by FashionGate at runtime)
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR      = API_ROOT / "data" / "classifier"
FASHION_DIR   = DATA_DIR / "fashion"
NOT_FASHION_DIR = DATA_DIR / "not_fashion"
MODEL_OUT     = API_ROOT / "models" / "fashion_gate.pth"
MLRUNS_DIR    = API_ROOT / "mlruns"


# ── data prep ─────────────────────────────────────────────────────────────────

def prepare_fashion_data(max_images: int = 2000):
    """Symlink/copy product catalog images into data/classifier/fashion/."""
    FASHION_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(FASHION_DIR.glob("*.jpg")))
    if existing >= 500:
        logger.info(f"Fashion data already prepared ({existing} images)")
        return existing

    sources = [
        API_ROOT / "data" / "metadata.json",
    ]

    # collect from product catalog via metadata.json
    import json
    meta_path = API_ROOT / "data" / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        copied = 0
        for item in metadata[:max_images]:
            src = Path(item.get("image_path", ""))
            if src.exists():
                dst = FASHION_DIR / f"fashion_{copied:04d}.jpg"
                if not dst.exists():
                    shutil.copy2(src, dst)
                copied += 1
        logger.info(f"Copied {copied} fashion images from catalog")
        return copied

    logger.warning("No metadata.json found — add product images to data/classifier/fashion/ manually")
    return 0


def build_datasets(val_ratio: float = 0.2, img_size: int = 224):
    """Build train/val datasets from fashion + not_fashion folders."""

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ImageFolder expects: data/classifier/{fashion, not_fashion}/
    full_ds = datasets.ImageFolder(str(DATA_DIR), transform=train_tf)
    n = len(full_ds)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # val uses val transforms
    val_ds.dataset = datasets.ImageFolder(str(DATA_DIR), transform=val_tf)

    logger.info(f"Dataset: {n} total — {n_train} train / {n_val} val")
    logger.info(f"Classes: {full_ds.classes}")

    # weighted sampler to handle class imbalance
    targets = [full_ds.targets[i] for i in train_ds.indices]
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]
    sampler = WeightedRandomSampler(weights, len(weights))

    return train_ds, val_ds, sampler, full_ds.classes


# ── model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


# ── training loop ─────────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, epochs: int, lr: float, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(labels)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += len(labels)

        scheduler.step()

        # ── val ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += len(labels)

        train_acc = train_correct / train_total
        val_acc   = val_correct / val_total
        avg_loss  = train_loss / train_total

        logger.info(
            f"Epoch {epoch:2d}/{epochs} — "
            f"loss: {avg_loss:.4f}  train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(f"  ✓ New best val_acc: {val_acc:.4f}")

    return best_state, best_val_acc


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--no-mlflow",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── prepare data ──────────────────────────────────────────────────────────
    logger.info("Preparing fashion data …")
    n_fashion = prepare_fashion_data()

    n_not_fashion = len(list(NOT_FASHION_DIR.rglob("*.jpg"))) if NOT_FASHION_DIR.exists() else 0
    if n_not_fashion < 50:
        logger.error(
            f"Not enough not-fashion images ({n_not_fashion}). "
            f"Run: python scripts/collect_not_fashion.py"
        )
        return

    logger.info(f"Fashion: {n_fashion}  Not-fashion: {n_not_fashion}")

    train_ds, val_ds, sampler, classes = build_datasets()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── build model ───────────────────────────────────────────────────────────
    model = build_model(num_classes=len(classes)).to(device)
    logger.info(f"MobileNetV3-Small loaded — classes: {classes}")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(str(MLRUNS_DIR))
            mlflow.set_experiment("wearabout-classifier")
            mlflow_run = mlflow.start_run(run_name="mobilenetv3_finetune")
            mlflow.log_params({
                "model":      "mobilenet_v3_small",
                "epochs":     args.epochs,
                "batch_size": args.batch_size,
                "lr":         args.lr,
                "n_fashion":  n_fashion,
                "n_not_fashion": n_not_fashion,
                "classes":    str(classes),
            })
        except Exception as e:
            logger.warning(f"MLflow unavailable: {e}")

    # ── train ─────────────────────────────────────────────────────────────────
    logger.info("Training …")
    best_state, best_val_acc = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, device=device
    )

    # ── save ──────────────────────────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "classes":    classes,
        "val_acc":    best_val_acc,
        "img_size":   224,
    }, MODEL_OUT)
    logger.info(f"Saved → {MODEL_OUT}  (val_acc={best_val_acc:.4f})")

    if mlflow_run:
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_artifact(str(MODEL_OUT), artifact_path="model")
        mlflow.end_run()

    logger.info("Done. Next: python main.py  (FashionGate loads automatically)")


if __name__ == "__main__":
    main()
