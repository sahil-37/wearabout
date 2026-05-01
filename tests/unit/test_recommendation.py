"""
Unit tests for RecommendationEngine
"""

import json
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.engines.recommendation import RecommendationEngine


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def catalog(tmp_path):
    """Write a minimal feature store + metadata to tmp_path."""
    n = 30
    np.random.seed(0)
    feats = np.random.randn(n, 2048).astype(np.float32)
    # L2 normalise
    feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    cats = (["topwear"] * 10 + ["bottomwear"] * 10 + ["footwear"] * 10)
    img_paths = [f"img_{i}.jpg" for i in range(n)]

    features_path = tmp_path / "features_normalized.pkl"
    with open(features_path, "wb") as f:
        pickle.dump({"features": feats, "image_paths": img_paths, "categories": cats}, f)

    metadata = [
        {
            "id": i,
            "name": f"Product {i}",
            "brand": "TestBrand",
            "category": cats[i],
            "gender": ["men", "women", "unisex"][i % 3],
            "price": 500.0 + i,
            "product_url": f"https://example.com/{i}",
            "image_path": img_paths[i],
        }
        for i in range(n)
    ]
    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return {"features_path": str(features_path), "metadata_path": str(metadata_path), "n": n}


@pytest.fixture
def mock_gate():
    m = MagicMock()
    m.check.return_value = (True, 1.0)
    return m


@pytest.fixture
def mock_pose():
    m = MagicMock()
    m.pose_estimation.return_value = (True, None, None)
    m.cleanup.return_value = None
    return m


@pytest.fixture
def mock_extractor():
    m = MagicMock()
    np.random.seed(1)
    m.extract_features.return_value = np.random.randn(2048).astype(np.float32)
    m.cleanup.return_value = None
    return m


@pytest.fixture
def mock_detector():
    m = MagicMock()
    m.detect.return_value = {
        "success": True,
        "detections": [{"class": "Topwear", "confidence": 0.9,
                         "bbox": {"x1": 0, "y1": 0, "x2": 100, "y2": 100,
                                  "width": 100, "height": 100}}],
    }
    m.model = {"type": "onnx"}
    m.cleanup.return_value = None
    return m


@pytest.fixture
def engine(catalog, mock_gate, mock_pose, mock_extractor, mock_detector, monkeypatch):
    monkeypatch.setattr("app.config.settings.FEATURES_PATH",   catalog["features_path"])
    monkeypatch.setattr("app.config.settings.METADATA_PATH",   catalog["metadata_path"])
    monkeypatch.setattr("app.config.settings.MAX_RECOMMENDATIONS", 10)
    monkeypatch.setattr("app.config.settings.MIN_RECOMMENDATIONS", 2)
    monkeypatch.setattr("app.config.settings.DEFAULT_K_NEIGHBORS", 10)
    monkeypatch.setattr("app.config.settings.YOLO_WEIGHTS_PATH", "")
    monkeypatch.setattr("app.config.settings.YOLO_CONFIDENCE_THRESHOLD", 0.4)
    monkeypatch.setattr("app.config.settings.NMS_IOU_THRESHOLD", 0.45)
    monkeypatch.setattr("app.config.settings.POSE_CONFIDENCE_THRESHOLD", 0.5)

    with patch("app.engines.recommendation.FashionGate", return_value=mock_gate), \
         patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose), \
         patch("app.engines.recommendation.FeatureExtractor", return_value=mock_extractor), \
         patch("app.engines.recommendation.ObjectDetector", return_value=mock_detector):
        e = RecommendationEngine()
        e._ensure_loaded()
        return e


# ── initialisation ────────────────────────────────────────────────────────────

def test_engine_loads(engine, catalog):
    assert engine.faiss_index is not None
    assert engine.faiss_index.ntotal == catalog["n"]
    assert len(engine.metadata) == catalog["n"]


def test_per_category_indices_built(engine):
    assert "topwear" in engine.faiss_by_category
    assert "bottomwear" in engine.faiss_by_category
    assert "footwear" in engine.faiss_by_category


# ── find_similar ──────────────────────────────────────────────────────────────

def test_find_similar_success(engine, tmp_path):
    import cv2
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    p = str(tmp_path / "q.jpg")
    cv2.imwrite(p, img)

    result = engine.find_similar_images(p, top_k=5)
    assert result["success"] is True
    assert "similar_items" in result
    assert len(result["similar_items"]) <= 5


def test_find_similar_result_fields(engine, tmp_path):
    import cv2
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    p = str(tmp_path / "q2.jpg")
    cv2.imwrite(p, img)

    result = engine.find_similar_images(p, top_k=3)
    if result["success"] and result["similar_items"]:
        item = result["similar_items"][0]
        for field in ("id", "name", "category", "price", "similarity_score"):
            assert field in item


def test_find_similar_blocked_by_gate(engine, tmp_path, mock_gate):
    mock_gate.check.return_value = (False, 0.1)
    import cv2
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    p = str(tmp_path / "garbage.jpg")
    cv2.imwrite(p, img)

    result = engine.find_similar_images(p)
    assert result["success"] is False
    assert "clothing" in result["error"].lower()


# ── recommend ─────────────────────────────────────────────────────────────────

def test_recommend_success(engine, tmp_path):
    import cv2
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    p = str(tmp_path / "outfit.jpg")
    cv2.imwrite(p, img)

    result = engine.recommend(p, gender="unisex", top_k=5)
    assert result["success"] is True
    assert "recommendations" in result
    assert "detections" in result


def test_recommend_gender_filter(engine, tmp_path):
    import cv2
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    p = str(tmp_path / "outfit2.jpg")
    cv2.imwrite(p, img)

    result = engine.recommend(p, gender="men")
    assert result["success"] is True
    for item in result["recommendations"]:
        assert item["gender"] in ("men", "unisex", "n.a.")


def test_recommend_blocked_by_gate(engine, tmp_path, mock_gate):
    mock_gate.check.return_value = (False, 0.05)
    import cv2
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    p = str(tmp_path / "garbage2.jpg")
    cv2.imwrite(p, img)

    result = engine.recommend(p)
    assert result["success"] is False


def test_cleanup_does_not_raise(engine):
    engine.cleanup()
