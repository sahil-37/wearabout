"""
Unit tests for Recommendation Engine module
"""

import numpy as np
import pytest
import pickle
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestRecommendationEngine:
    """Test suite for RecommendationEngine class"""

    @pytest.fixture
    def mock_settings(self, tmp_path, monkeypatch):
        """Mock settings for testing"""
        features_path = tmp_path / "features.pkl"
        metadata_path = tmp_path / "metadata.json"
        normalized_path = tmp_path / "features_normalized.pkl"

        # Create mock features
        np.random.seed(42)
        num_products = 50
        features_dict = {
            f"product_{i}.jpg": np.random.randn(2048).astype(np.float32)
            for i in range(num_products)
        }

        with open(features_path, "wb") as f:
            pickle.dump(features_dict, f)

        # Create normalized features
        image_paths = list(features_dict.keys())
        features_array = np.array([features_dict[p] for p in image_paths])
        normalized_data = {
            "features": features_array,
            "image_paths": image_paths,
            "categories": ["topwear"] * num_products
        }
        with open(normalized_path, "wb") as f:
            pickle.dump(normalized_data, f)

        # Create mock metadata
        metadata = [
            {
                "name": f"Product {i}",
                "category": ["topwear", "bottomwear", "footwear"][i % 3],
                "gender": ["men", "women", "unisex"][i % 3],
                "price": 29.99 + i,
                "image_url": f"https://example.com/product_{i}.jpg",
                "product_url": f"https://example.com/products/{i}"
            }
            for i in range(num_products)
        ]
        metadata_dict = {
            "products": metadata,
            "num_images": num_products,
            "categories": ["topwear", "bottomwear", "footwear"]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        # Patch settings
        monkeypatch.setattr("app.config.settings.FEATURES_PATH", str(features_path))
        monkeypatch.setattr("app.config.settings.METADATA_PATH", str(metadata_path))
        monkeypatch.setattr("app.config.settings.DEFAULT_K_NEIGHBORS", 10)
        monkeypatch.setattr("app.config.settings.MAX_RECOMMENDATIONS", 10)
        monkeypatch.setattr("app.config.settings.MIN_RECOMMENDATIONS", 3)
        monkeypatch.setattr("app.config.settings.POSE_CONFIDENCE_THRESHOLD", 0.5)
        monkeypatch.setattr("app.config.settings.YOLO_CONFIDENCE_THRESHOLD", 0.4)
        monkeypatch.setattr("app.config.settings.NMS_IOU_THRESHOLD", 0.45)
        monkeypatch.setattr("app.config.settings.YOLO_WEIGHTS_PATH", "")

        return {
            "features_path": str(features_path),
            "metadata_path": str(metadata_path),
            "normalized_path": str(normalized_path),
            "num_products": num_products
        }

    @pytest.fixture
    def mock_pose_estimator(self):
        """Mock pose estimator"""
        mock = MagicMock()
        mock.pose_estimation.return_value = (True, np.zeros((224, 224, 3)), np.zeros((224, 224, 3)))
        mock.cleanup.return_value = None
        return mock

    @pytest.fixture
    def mock_feature_extractor(self):
        """Mock feature extractor"""
        mock = MagicMock()
        mock.extract_features.return_value = np.random.randn(2048).astype(np.float32)
        mock.scale_features.return_value = np.random.randn(1, 2048).astype(np.float32)
        mock.load_features.return_value = {
            f"product_{i}.jpg": np.random.randn(2048).astype(np.float32)
            for i in range(50)
        }
        mock.cleanup.return_value = None
        return mock

    @pytest.fixture
    def mock_object_detector(self):
        """Mock object detector"""
        mock = MagicMock()
        mock.detect.return_value = {
            "success": True,
            "detections": [
                {"class": "topwear", "confidence": 0.9, "bbox": [100, 100, 200, 300]}
            ]
        }
        mock.cleanup.return_value = None
        return mock

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    @pytest.mark.unit
    def test_engine_initialization(self, mock_settings, mock_pose_estimator,
                                   mock_feature_extractor, mock_object_detector):
        """Test that RecommendationEngine initializes correctly"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            assert engine is not None
            assert engine.knn_initialized is False  # Lazy loading

    @pytest.mark.unit
    def test_lazy_loading_not_triggered_at_init(self, mock_settings, mock_pose_estimator,
                                                 mock_feature_extractor, mock_object_detector):
        """Test that features are not loaded at initialization"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            # Features should not be loaded yet
            assert engine.features_array is None
            assert engine.knn_model is None

    # ========================================================================
    # Lazy Loading Tests
    # ========================================================================

    @pytest.mark.unit
    def test_lazy_load_features(self, mock_settings, mock_pose_estimator,
                                mock_feature_extractor, mock_object_detector):
        """Test that features are loaded on first use"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            # Trigger lazy loading
            engine._lazy_load_features()

            assert engine.features_array is not None
            assert len(engine.features_image_paths) == mock_settings["num_products"]

    @pytest.mark.unit
    def test_knn_initialization_after_lazy_load(self, mock_settings, mock_pose_estimator,
                                                 mock_feature_extractor, mock_object_detector):
        """Test KNN model is initialized after lazy loading"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            engine._lazy_load_features()
            engine._initialize_knn()

            assert engine.knn_model is not None

    # ========================================================================
    # Find Similar Tests
    # ========================================================================

    @pytest.mark.unit
    def test_find_similar_images_returns_results(self, mock_settings, mock_pose_estimator,
                                                  mock_feature_extractor, mock_object_detector,
                                                  sample_image_path):
        """Test find_similar_images returns similar items"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.find_similar_images(sample_image_path, top_k=5)

            assert result["success"] is True
            assert "similar_items" in result
            assert len(result["similar_items"]) <= 5

    @pytest.mark.unit
    def test_find_similar_images_contains_expected_fields(self, mock_settings, mock_pose_estimator,
                                                           mock_feature_extractor, mock_object_detector,
                                                           sample_image_path):
        """Test that similar items contain expected fields"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.find_similar_images(sample_image_path, top_k=5)

            if result["success"] and result["similar_items"]:
                item = result["similar_items"][0]
                assert "id" in item
                assert "similarity_score" in item
                assert "distance" in item

    @pytest.mark.unit
    def test_find_similar_returns_sorted_by_distance(self, mock_settings, mock_pose_estimator,
                                                      mock_feature_extractor, mock_object_detector,
                                                      sample_image_path):
        """Test that results are sorted by distance (ascending)"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.find_similar_images(sample_image_path, top_k=10)

            if result["success"] and len(result["similar_items"]) > 1:
                distances = [item["distance"] for item in result["similar_items"]]
                assert distances == sorted(distances)

    # ========================================================================
    # Recommendation Tests
    # ========================================================================

    @pytest.mark.unit
    def test_recommend_full_body_required(self, mock_settings, mock_pose_estimator,
                                          mock_feature_extractor, mock_object_detector,
                                          sample_image_path):
        """Test that recommendations require full-body shot"""
        # Mock pose estimator to return False (not full body)
        mock_pose_estimator.pose_estimation.return_value = (False, None, None)

        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend(sample_image_path)

            assert result["success"] is False
            assert "full-body" in result["error"].lower()

    @pytest.mark.unit
    def test_recommend_with_full_body(self, mock_settings, mock_pose_estimator,
                                       mock_feature_extractor, mock_object_detector,
                                       sample_image_path):
        """Test recommendations with valid full-body image"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend(sample_image_path)

            assert result["success"] is True
            assert "recommendations" in result
            assert "full_shot_detected" in result

    @pytest.mark.unit
    def test_recommend_includes_detections(self, mock_settings, mock_pose_estimator,
                                            mock_feature_extractor, mock_object_detector,
                                            sample_image_path):
        """Test that recommendations include detection results"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend(sample_image_path)

            assert "detections" in result

    # ========================================================================
    # Gender Filter Tests
    # ========================================================================

    @pytest.mark.unit
    def test_recommend_gender_filter_men(self, mock_settings, mock_pose_estimator,
                                          mock_feature_extractor, mock_object_detector,
                                          sample_image_path):
        """Test gender filter for men's products"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend(sample_image_path, gender="men")

            # Should succeed
            assert result["success"] is True

    @pytest.mark.unit
    def test_recommend_gender_filter_women(self, mock_settings, mock_pose_estimator,
                                            mock_feature_extractor, mock_object_detector,
                                            sample_image_path):
        """Test gender filter for women's products"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend(sample_image_path, gender="women")

            assert result["success"] is True

    @pytest.mark.unit
    def test_recommend_unisex_returns_all(self, mock_settings, mock_pose_estimator,
                                           mock_feature_extractor, mock_object_detector,
                                           sample_image_path):
        """Test unisex filter returns all products"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend(sample_image_path, gender="unisex")

            assert result["success"] is True

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    @pytest.mark.unit
    def test_recommend_invalid_image(self, mock_settings, mock_pose_estimator,
                                      mock_feature_extractor, mock_object_detector):
        """Test handling of invalid image path"""
        # Mock feature extractor to return None
        mock_feature_extractor.extract_features.return_value = None
        mock_pose_estimator.pose_estimation.return_value = (True, None, None)

        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.recommend("/invalid/path.jpg")

            assert result["success"] is False

    @pytest.mark.unit
    def test_find_similar_invalid_image(self, mock_settings, mock_pose_estimator,
                                         mock_feature_extractor, mock_object_detector):
        """Test find_similar_images with invalid image"""
        mock_feature_extractor.extract_features.return_value = None

        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            result = engine.find_similar_images("/invalid/path.jpg")

            assert result["success"] is False

    # ========================================================================
    # Cleanup Tests
    # ========================================================================

    @pytest.mark.unit
    def test_cleanup(self, mock_settings, mock_pose_estimator,
                     mock_feature_extractor, mock_object_detector):
        """Test cleanup releases resources"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()

            # Should not raise
            engine.cleanup()

            mock_pose_estimator.cleanup.assert_called_once()
            mock_feature_extractor.cleanup.assert_called_once()


class TestBuildRecommendations:
    """Test suite for _build_recommendations helper method"""

    @pytest.fixture
    def engine_with_metadata(self, mock_settings, mock_pose_estimator,
                             mock_feature_extractor, mock_object_detector):
        """Create engine with loaded metadata"""
        with patch("app.engines.recommendation.PoseEstimation", return_value=mock_pose_estimator), \
             patch("app.engines.recommendation.FeatureExtractor", return_value=mock_feature_extractor), \
             patch("app.engines.recommendation.ObjectDetector", return_value=mock_object_detector):

            from app.engines.recommendation import RecommendationEngine
            engine = RecommendationEngine()
            engine._lazy_load_features()
            return engine

    @pytest.fixture
    def mock_settings(self, tmp_path, monkeypatch):
        """Mock settings for build recommendations tests"""
        features_path = tmp_path / "features.pkl"
        metadata_path = tmp_path / "metadata.json"
        normalized_path = tmp_path / "features_normalized.pkl"

        np.random.seed(42)
        num_products = 20

        features_dict = {
            f"product_{i}.jpg": np.random.randn(2048).astype(np.float32)
            for i in range(num_products)
        }
        with open(features_path, "wb") as f:
            pickle.dump(features_dict, f)

        image_paths = list(features_dict.keys())
        features_array = np.array([features_dict[p] for p in image_paths])
        with open(normalized_path, "wb") as f:
            pickle.dump({
                "features": features_array,
                "image_paths": image_paths,
                "categories": ["topwear"] * num_products
            }, f)

        metadata = [
            {
                "name": f"Product {i}",
                "category": "topwear",
                "gender": ["men", "women", "unisex"][i % 3],
                "price": 29.99
            }
            for i in range(num_products)
        ]
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        monkeypatch.setattr("app.config.settings.FEATURES_PATH", str(features_path))
        monkeypatch.setattr("app.config.settings.METADATA_PATH", str(metadata_path))
        monkeypatch.setattr("app.config.settings.DEFAULT_K_NEIGHBORS", 10)
        monkeypatch.setattr("app.config.settings.MAX_RECOMMENDATIONS", 10)
        monkeypatch.setattr("app.config.settings.MIN_RECOMMENDATIONS", 3)
        monkeypatch.setattr("app.config.settings.POSE_CONFIDENCE_THRESHOLD", 0.5)
        monkeypatch.setattr("app.config.settings.YOLO_CONFIDENCE_THRESHOLD", 0.4)
        monkeypatch.setattr("app.config.settings.NMS_IOU_THRESHOLD", 0.45)
        monkeypatch.setattr("app.config.settings.YOLO_WEIGHTS_PATH", "")

        return {"num_products": num_products}

    @pytest.fixture
    def mock_pose_estimator(self):
        mock = MagicMock()
        mock.pose_estimation.return_value = (True, None, None)
        mock.cleanup.return_value = None
        return mock

    @pytest.fixture
    def mock_feature_extractor(self):
        mock = MagicMock()
        mock.extract_features.return_value = np.random.randn(2048).astype(np.float32)
        mock.scale_features.return_value = np.random.randn(1, 2048).astype(np.float32)
        mock.load_features.return_value = {}
        mock.cleanup.return_value = None
        return mock

    @pytest.fixture
    def mock_object_detector(self):
        mock = MagicMock()
        mock.detect.return_value = {"success": True, "detections": []}
        mock.cleanup.return_value = None
        return mock

    @pytest.mark.unit
    def test_build_recommendations_similarity_score(self, engine_with_metadata):
        """Test similarity score calculation"""
        indices = np.array([0, 1, 2])
        distances = np.array([0.0, 1.0, 2.0])

        recommendations = engine_with_metadata._build_recommendations(
            indices, distances, "unisex"
        )

        if recommendations:
            # First item with distance 0 should have highest similarity
            assert recommendations[0]["similarity_score"] == 1.0
            # Similarity should decrease as distance increases
            for i in range(1, len(recommendations)):
                assert recommendations[i]["similarity_score"] <= recommendations[i-1]["similarity_score"]

    @pytest.mark.unit
    def test_build_recommendations_respects_max(self, engine_with_metadata, monkeypatch):
        """Test that MAX_RECOMMENDATIONS is respected"""
        monkeypatch.setattr("app.config.settings.MAX_RECOMMENDATIONS", 3)

        indices = np.array([0, 1, 2, 3, 4, 5])
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

        recommendations = engine_with_metadata._build_recommendations(
            indices, distances, "unisex"
        )

        assert len(recommendations) <= 3
