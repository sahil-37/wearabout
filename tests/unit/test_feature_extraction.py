"""
Unit tests for Feature Extraction module
"""

import numpy as np
import pytest
import cv2
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.feature_extraction import FeatureExtractor


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class"""

    @pytest.fixture
    def extractor(self):
        """Create FeatureExtractor instance for testing"""
        return FeatureExtractor(input_size=(224, 224))

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    @pytest.mark.unit
    def test_extractor_initialization(self, extractor):
        """Test that FeatureExtractor initializes correctly"""
        assert extractor is not None
        assert extractor.input_size == (224, 224)
        assert extractor.scaler is not None

    @pytest.mark.unit
    def test_extractor_custom_input_size(self):
        """Test FeatureExtractor with custom input size"""
        custom_extractor = FeatureExtractor(input_size=(128, 128))
        assert custom_extractor.input_size == (128, 128)

    # ========================================================================
    # Feature Extraction Tests
    # ========================================================================

    @pytest.mark.unit
    def test_extract_features_returns_correct_shape(self, extractor, sample_image_path):
        """Test that extracted features have correct dimensionality (2048)"""
        features = extractor.extract_features(sample_image_path)

        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape == (2048,)

    @pytest.mark.unit
    def test_extract_features_dtype(self, extractor, sample_image_path):
        """Test that features are float32"""
        features = extractor.extract_features(sample_image_path)

        assert features.dtype == np.float32

    @pytest.mark.unit
    def test_extract_features_not_all_zeros(self, extractor, sample_image_path):
        """Test that extracted features are not all zeros"""
        features = extractor.extract_features(sample_image_path)

        assert not np.allclose(features, 0)

    @pytest.mark.unit
    def test_extract_features_reproducible(self, extractor, sample_image_path):
        """Test that same image produces same features"""
        features1 = extractor.extract_features(sample_image_path)
        features2 = extractor.extract_features(sample_image_path)

        np.testing.assert_array_almost_equal(features1, features2)

    @pytest.mark.unit
    def test_extract_features_different_images(self, extractor, sample_image_path, tmp_path):
        """Test that different images produce different features"""
        # Create a different image
        different_img = np.ones((224, 224, 3), dtype=np.uint8) * 127
        different_path = tmp_path / "different.jpg"
        cv2.imwrite(str(different_path), different_img)

        features1 = extractor.extract_features(sample_image_path)
        features2 = extractor.extract_features(str(different_path))

        # Features should be different (not exactly equal)
        assert not np.array_equal(features1, features2)

    # ========================================================================
    # Invalid Input Tests
    # ========================================================================

    @pytest.mark.unit
    def test_extract_features_invalid_path(self, extractor):
        """Test handling of non-existent file path"""
        features = extractor.extract_features("/nonexistent/path/image.jpg")

        # Should return None or zeros on error
        assert features is None or np.allclose(features, 0)

    @pytest.mark.unit
    def test_extract_features_corrupted_file(self, extractor, invalid_image_path):
        """Test handling of corrupted image file"""
        features = extractor.extract_features(invalid_image_path)

        # Should handle gracefully
        assert features is None or isinstance(features, np.ndarray)

    @pytest.mark.unit
    def test_extract_features_empty_path(self, extractor):
        """Test handling of empty path string"""
        features = extractor.extract_features("")

        assert features is None or np.allclose(features, 0)

    # ========================================================================
    # Histogram Fallback Tests
    # ========================================================================

    @pytest.mark.unit
    def test_histogram_features_shape(self, extractor, sample_image_path):
        """Test histogram fallback produces correct shape"""
        features = extractor._extract_histogram_features(sample_image_path)

        assert features.shape == (2048,)
        assert features.dtype == np.float32

    @pytest.mark.unit
    def test_histogram_features_nonzero(self, extractor, sample_image_path):
        """Test histogram features are not all zeros for valid image"""
        features = extractor._extract_histogram_features(sample_image_path)

        assert np.sum(np.abs(features)) > 0

    @pytest.mark.unit
    def test_histogram_features_invalid_returns_zeros(self, extractor):
        """Test histogram fallback returns zeros for invalid path"""
        features = extractor._extract_histogram_features("/invalid/path.jpg")

        np.testing.assert_array_equal(features, np.zeros(2048))

    # ========================================================================
    # Batch Processing Tests
    # ========================================================================

    @pytest.mark.unit
    def test_extract_batch_features(self, extractor, sample_image_path, tmp_path):
        """Test batch feature extraction"""
        # Create multiple test images
        paths = [sample_image_path]
        for i in range(2):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            path = tmp_path / f"batch_img_{i}.jpg"
            cv2.imwrite(str(path), img)
            paths.append(str(path))

        features = extractor.extract_batch_features(paths)

        assert len(features) == 3
        for feat in features:
            if feat is not None:
                assert feat.shape == (2048,)

    @pytest.mark.unit
    def test_extract_batch_empty_list(self, extractor):
        """Test batch extraction with empty list"""
        features = extractor.extract_batch_features([])

        assert features == []

    @pytest.mark.unit
    def test_extract_batch_with_invalid(self, extractor, sample_image_path):
        """Test batch extraction handles invalid paths gracefully"""
        paths = [sample_image_path, "/invalid/path.jpg"]
        features = extractor.extract_batch_features(paths)

        assert len(features) == 2
        assert features[0] is not None  # Valid image
        # Second should be None or zeros

    # ========================================================================
    # Scaling Tests
    # ========================================================================

    @pytest.mark.unit
    def test_scale_features_1d(self, extractor, sample_features):
        """Test feature scaling with 1D array"""
        scaled = extractor.scale_features(sample_features)

        assert scaled is not None
        assert isinstance(scaled, np.ndarray)

    @pytest.mark.unit
    def test_scale_features_2d(self, extractor, sample_features_batch):
        """Test feature scaling with 2D array"""
        features_2d = np.array(sample_features_batch)
        scaled = extractor.scale_features(features_2d)

        assert scaled is not None
        assert scaled.shape == features_2d.shape

    # ========================================================================
    # Save/Load Tests
    # ========================================================================

    @pytest.mark.unit
    def test_save_and_load_features(self, extractor, sample_features, tmp_path):
        """Test saving and loading features"""
        filepath = tmp_path / "test_features.pkl"

        # Save
        success = extractor.save_features(sample_features, str(filepath))
        assert success is True
        assert filepath.exists()

        # Load
        loaded = extractor.load_features(str(filepath))
        np.testing.assert_array_almost_equal(loaded, sample_features)

    @pytest.mark.unit
    def test_load_features_invalid_path(self, extractor):
        """Test loading from invalid path returns None"""
        loaded = extractor.load_features("/nonexistent/features.pkl")

        assert loaded is None

    # ========================================================================
    # Cleanup Tests
    # ========================================================================

    @pytest.mark.unit
    def test_cleanup(self, extractor):
        """Test cleanup method doesn't raise errors"""
        # Should not raise any exception
        extractor.cleanup()


class TestFeatureExtractorEdgeCases:
    """Edge case tests for FeatureExtractor"""

    @pytest.fixture
    def extractor(self):
        return FeatureExtractor()

    @pytest.mark.unit
    def test_very_small_image(self, extractor, tmp_path):
        """Test handling of very small images"""
        small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        path = tmp_path / "small.jpg"
        cv2.imwrite(str(path), small_img)

        features = extractor.extract_features(str(path))

        # Should still produce 2048 features
        if features is not None:
            assert features.shape == (2048,)

    @pytest.mark.unit
    def test_grayscale_image(self, extractor, tmp_path):
        """Test handling of grayscale images"""
        gray_img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        path = tmp_path / "gray.jpg"
        cv2.imwrite(str(path), gray_img)

        features = extractor.extract_features(str(path))

        # Should handle grayscale
        if features is not None:
            assert features.shape == (2048,)

    @pytest.mark.unit
    def test_large_image(self, extractor, tmp_path):
        """Test handling of large images"""
        large_img = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        path = tmp_path / "large.jpg"
        cv2.imwrite(str(path), large_img)

        features = extractor.extract_features(str(path))

        # Should resize and process
        if features is not None:
            assert features.shape == (2048,)

    @pytest.mark.unit
    def test_png_format(self, extractor, tmp_path):
        """Test handling of PNG format"""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)

        features = extractor.extract_features(str(path))

        if features is not None:
            assert features.shape == (2048,)
