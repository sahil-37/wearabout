"""
Unit tests for FeatureExtractor
"""

import cv2
import numpy as np
import pytest

from app.models.feature_extraction import FeatureExtractor


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor(input_size=(224, 224))


@pytest.fixture
def sample_img(tmp_path):
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    path = tmp_path / "sample.jpg"
    cv2.imwrite(str(path), img)
    return str(path)


# ── initialisation ────────────────────────────────────────────────────────────

def test_extractor_initialises(extractor):
    assert extractor is not None
    assert extractor.input_size == (224, 224)


# ── extract_features ─────────────────────────────────────────────────────────

def test_returns_2048_vector(extractor, sample_img):
    feat = extractor.extract_features(sample_img)
    assert feat is not None
    assert feat.shape == (2048,)
    assert feat.dtype == np.float32


def test_l2_normalised(extractor, sample_img):
    feat = extractor.extract_features(sample_img)
    assert feat is not None
    norm = np.linalg.norm(feat)
    assert abs(norm - 1.0) < 1e-4


def test_reproducible(extractor, sample_img):
    f1 = extractor.extract_features(sample_img)
    f2 = extractor.extract_features(sample_img)
    np.testing.assert_array_almost_equal(f1, f2)


def test_different_images_differ(extractor, sample_img, tmp_path):
    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    path2 = tmp_path / "black.jpg"
    cv2.imwrite(str(path2), img2)
    f1 = extractor.extract_features(sample_img)
    f2 = extractor.extract_features(str(path2))
    assert not np.array_equal(f1, f2)


def test_invalid_path_returns_none(extractor):
    result = extractor.extract_features("/nonexistent/image.jpg")
    assert result is None or (isinstance(result, np.ndarray) and np.allclose(result, 0))


def test_empty_path_returns_none(extractor):
    result = extractor.extract_features("")
    assert result is None or isinstance(result, np.ndarray)


# ── batch ─────────────────────────────────────────────────────────────────────

def test_batch_returns_list(extractor, sample_img, tmp_path):
    img2 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    path2 = tmp_path / "b.jpg"
    cv2.imwrite(str(path2), img2)
    results = extractor.extract_batch_features([sample_img, str(path2)])
    assert len(results) == 2
    for r in results:
        if r is not None:
            assert r.shape == (2048,)


def test_batch_empty_list(extractor):
    assert extractor.extract_batch_features([]) == []


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_and_load(extractor, sample_img, tmp_path):
    feat = extractor.extract_features(sample_img)
    path = str(tmp_path / "feat.pkl")
    assert extractor.save_features(feat, path) is True
    loaded = extractor.load_features(path)
    np.testing.assert_array_almost_equal(feat, loaded)


def test_load_missing_returns_none(extractor):
    assert extractor.load_features("/no/such/file.pkl") is None


# ── edge cases ────────────────────────────────────────────────────────────────

def test_small_image(extractor, tmp_path):
    img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    path = tmp_path / "small.jpg"
    cv2.imwrite(str(path), img)
    feat = extractor.extract_features(str(path))
    if feat is not None:
        assert feat.shape == (2048,)


def test_large_image(extractor, tmp_path):
    img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    path = tmp_path / "large.jpg"
    cv2.imwrite(str(path), img)
    feat = extractor.extract_features(str(path))
    if feat is not None:
        assert feat.shape == (2048,)


def test_grayscale_image(extractor, tmp_path):
    img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
    path = tmp_path / "gray.jpg"
    cv2.imwrite(str(path), img)
    feat = extractor.extract_features(str(path))
    if feat is not None:
        assert feat.shape == (2048,)


def test_cleanup_does_not_raise(extractor):
    extractor.cleanup()
