"""
Shared pytest fixtures for Buy Me That Look API tests
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import app


# ============================================================================
# API Client Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """
    FastAPI test client fixture.
    Uses module scope to reuse client across tests in same module.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="function")
def fresh_client() -> Generator[TestClient, None, None]:
    """
    Fresh FastAPI test client for tests that need isolation.
    """
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# Image Fixtures
# ============================================================================

@pytest.fixture
def sample_image_path(tmp_path: Path) -> str:
    """
    Create a sample test image and return its path.
    Creates a simple 224x224 RGB image.
    """
    import cv2

    # Create a simple test image with some color variation
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    # Add some color regions for feature extraction tests
    img[0:112, 0:112] = [255, 0, 0]    # Red quadrant (BGR)
    img[0:112, 112:224] = [0, 255, 0]  # Green quadrant
    img[112:224, 0:112] = [0, 0, 255]  # Blue quadrant
    img[112:224, 112:224] = [128, 128, 128]  # Gray quadrant

    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), img)

    return str(image_path)


@pytest.fixture
def sample_person_image(tmp_path: Path) -> str:
    """
    Create a sample image that mimics a person silhouette.
    """
    import cv2

    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light gray background

    # Draw a simple person-like shape
    cv2.rectangle(img, (270, 50), (370, 150), (100, 100, 100), -1)  # Head
    cv2.rectangle(img, (250, 150), (390, 350), (80, 80, 80), -1)    # Body
    cv2.rectangle(img, (260, 350), (310, 480), (60, 60, 60), -1)    # Left leg
    cv2.rectangle(img, (330, 350), (380, 480), (60, 60, 60), -1)    # Right leg

    image_path = tmp_path / "person_image.jpg"
    cv2.imwrite(str(image_path), img)

    return str(image_path)


@pytest.fixture
def invalid_image_path(tmp_path: Path) -> str:
    """
    Create an invalid/corrupted image file.
    """
    invalid_path = tmp_path / "invalid.jpg"
    invalid_path.write_text("not an image")
    return str(invalid_path)


@pytest.fixture
def sample_image_bytes(sample_image_path: str) -> bytes:
    """
    Return sample image as bytes for upload tests.
    """
    with open(sample_image_path, "rb") as f:
        return f.read()


# ============================================================================
# Feature Extraction Fixtures
# ============================================================================

@pytest.fixture
def mock_feature_extractor():
    """
    Mock feature extractor that returns consistent features.
    """
    mock = MagicMock()
    mock.extract_features.return_value = np.random.randn(2048).astype(np.float32)
    mock.extract_batch_features.return_value = [
        np.random.randn(2048).astype(np.float32) for _ in range(3)
    ]
    return mock


@pytest.fixture
def sample_features() -> np.ndarray:
    """
    Generate sample feature vector for testing.
    """
    np.random.seed(42)  # For reproducibility
    return np.random.randn(2048).astype(np.float32)


@pytest.fixture
def sample_features_batch() -> list:
    """
    Generate batch of sample feature vectors.
    """
    np.random.seed(42)
    return [np.random.randn(2048).astype(np.float32) for _ in range(5)]


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
def mock_features_database(tmp_path: Path) -> dict:
    """
    Create mock features database for testing.
    """
    import pickle

    np.random.seed(42)

    # Create mock features
    features = {
        f"product_{i}.jpg": np.random.randn(2048).astype(np.float32)
        for i in range(50)
    }

    # Save to pickle file
    features_path = tmp_path / "features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(features, f)

    return {
        "features": features,
        "path": str(features_path)
    }


@pytest.fixture
def mock_metadata(tmp_path: Path) -> dict:
    """
    Create mock metadata for testing.
    """
    import json

    metadata = {
        "categories": ["topwear", "bottomwear", "footwear", "eyewear", "handbag"],
        "total_products": 50,
        "feature_dim": 2048,
        "products": {
            f"product_{i}.jpg": {
                "category": ["topwear", "bottomwear", "footwear", "eyewear", "handbag"][i % 5],
                "id": i
            }
            for i in range(50)
        }
    }

    metadata_path = tmp_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return {
        "metadata": metadata,
        "path": str(metadata_path)
    }


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Set mock environment variables for testing.
    """
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DEBUG", "True")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "8000")


@pytest.fixture
def temp_upload_dir(tmp_path: Path) -> Path:
    """
    Create temporary upload directory.
    """
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(exist_ok=True)
    return upload_dir


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """
    Automatically clean up temporary files after each test.
    """
    yield
    # Cleanup happens automatically with tmp_path fixture


# ============================================================================
# Markers
# ============================================================================

def pytest_configure(config):
    """
    Register custom markers.
    """
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
