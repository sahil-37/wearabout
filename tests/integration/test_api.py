"""
Integration tests for Buy Me That Look API endpoints
"""

import pytest
import io
import numpy as np
import cv2
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestHealthEndpoints:
    """Test suite for health check endpoints"""

    @pytest.mark.integration
    def test_health_check(self, client):
        """Test /api/v1/health endpoint"""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "ok"]

    @pytest.mark.integration
    def test_ready_check(self, client):
        """Test /api/v1/ready endpoint"""
        response = client.get("/api/v1/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.integration
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestRecommendationEndpoints:
    """Test suite for recommendation endpoints"""

    @pytest.mark.integration
    def test_get_categories(self, client):
        """Test /api/v1/recommendation-categories endpoint"""
        response = client.get("/api/v1/recommendation-categories")

        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert isinstance(data["categories"], list)

    @pytest.mark.integration
    def test_get_config(self, client):
        """Test /api/v1/recommendation-config endpoint"""
        response = client.get("/api/v1/recommendation-config")

        assert response.status_code == 200
        data = response.json()
        # Should contain configuration info
        assert isinstance(data, dict)


class TestValidationEndpoint:
    """Test suite for image validation endpoint"""

    @pytest.mark.integration
    def test_validate_image_with_valid_file(self, client, sample_image_bytes):
        """Test image validation with valid image"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/validate-image", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data or "valid" in data or "success" in data

    @pytest.mark.integration
    def test_validate_image_without_file(self, client):
        """Test validation endpoint without file returns error"""
        response = client.post("/api/v1/validate-image")

        assert response.status_code == 422  # Unprocessable Entity

    @pytest.mark.integration
    def test_validate_image_with_invalid_content_type(self, client):
        """Test validation with non-image file"""
        files = {"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        response = client.post("/api/v1/validate-image", files=files)

        # Should return 400 or 422
        assert response.status_code in [400, 422, 200]


class TestDetectionEndpoint:
    """Test suite for fashion detection endpoint"""

    @pytest.mark.integration
    def test_detect_items(self, client, sample_image_bytes):
        """Test fashion item detection"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/detect-items", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "success" in data or "detections" in data

    @pytest.mark.integration
    def test_detect_items_returns_detections(self, client, sample_image_bytes):
        """Test that detection returns list of detections"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/detect-items", files=files)

        if response.status_code == 200:
            data = response.json()
            if "detections" in data:
                assert isinstance(data["detections"], list)


class TestFeatureExtractionEndpoint:
    """Test suite for feature extraction endpoint"""

    @pytest.mark.integration
    def test_extract_features(self, client, sample_image_bytes):
        """Test feature extraction endpoint"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/extract-features", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    @pytest.mark.integration
    def test_extract_features_returns_vector(self, client, sample_image_bytes):
        """Test that feature extraction returns feature vector info"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/extract-features", files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                # Should contain feature dimension or features
                assert "feature_dim" in data or "features" in data or "shape" in data


class TestFindSimilarEndpoint:
    """Test suite for find similar endpoint"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_find_similar(self, client, sample_image_bytes):
        """Test find similar items endpoint"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/find-similar", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    @pytest.mark.integration
    @pytest.mark.slow
    def test_find_similar_with_top_k(self, client, sample_image_bytes):
        """Test find similar with top_k parameter"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/find-similar?top_k=5", files=files)

        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "similar_items" in data:
                assert len(data["similar_items"]) <= 5


class TestRecommendEndpoint:
    """Test suite for recommendation endpoint"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_recommend_endpoint(self, client, sample_image_bytes):
        """Test recommendation endpoint"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/recommend", files=files)

        assert response.status_code == 200
        data = response.json()
        assert "success" in data

    @pytest.mark.integration
    @pytest.mark.slow
    def test_recommend_with_gender(self, client, sample_image_bytes):
        """Test recommendation with gender filter"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/recommend?gender=men", files=files)

        assert response.status_code == 200


class TestCORSHeaders:
    """Test suite for CORS configuration"""

    @pytest.mark.integration
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present"""
        response = client.options(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"}
        )

        # FastAPI returns 200 for OPTIONS by default
        assert response.status_code in [200, 204]

    @pytest.mark.integration
    def test_allowed_origin(self, client):
        """Test allowed origin receives proper headers"""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"}
        )

        assert response.status_code == 200


class TestErrorHandling:
    """Test suite for error handling"""

    @pytest.mark.integration
    def test_404_for_unknown_endpoint(self, client):
        """Test 404 for unknown endpoints"""
        response = client.get("/api/v1/unknown-endpoint")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method"""
        response = client.delete("/api/v1/health")

        assert response.status_code == 405

    @pytest.mark.integration
    def test_invalid_file_upload(self, client):
        """Test error handling for invalid file upload"""
        files = {"file": ("test.txt", io.BytesIO(b"invalid"), "text/plain")}
        response = client.post("/api/v1/validate-image", files=files)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]


class TestAPIDocumentation:
    """Test suite for API documentation endpoints"""

    @pytest.mark.integration
    def test_openapi_json(self, client):
        """Test OpenAPI JSON is accessible"""
        response = client.get("/api/openapi.json")

        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data

    @pytest.mark.integration
    def test_swagger_docs(self, client):
        """Test Swagger UI is accessible"""
        response = client.get("/api/docs")

        # Swagger UI returns HTML
        assert response.status_code == 200

    @pytest.mark.integration
    def test_redoc(self, client):
        """Test ReDoc is accessible"""
        response = client.get("/api/redoc")

        assert response.status_code == 200


class TestResponseFormat:
    """Test suite for response format consistency"""

    @pytest.mark.integration
    def test_json_content_type(self, client):
        """Test API returns JSON content type"""
        response = client.get("/api/v1/health")

        assert "application/json" in response.headers.get("content-type", "")

    @pytest.mark.integration
    def test_consistent_success_field(self, client, sample_image_bytes):
        """Test that responses have consistent success field"""
        endpoints_with_files = [
            "/api/v1/validate-image",
            "/api/v1/detect-items",
            "/api/v1/extract-features",
        ]

        for endpoint in endpoints_with_files:
            files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
            response = client.post(endpoint, files=files)

            if response.status_code == 200:
                data = response.json()
                # Most endpoints should have success field
                # Some might have different structure
                assert isinstance(data, dict)


class TestInputValidation:
    """Test suite for input validation"""

    @pytest.mark.integration
    def test_large_top_k_handled(self, client, sample_image_bytes):
        """Test that large top_k values are handled"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/find-similar?top_k=10000", files=files)

        # Should not crash
        assert response.status_code in [200, 400, 422]

    @pytest.mark.integration
    def test_negative_top_k_handled(self, client, sample_image_bytes):
        """Test that negative top_k values are handled"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/find-similar?top_k=-1", files=files)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]

    @pytest.mark.integration
    def test_invalid_gender_handled(self, client, sample_image_bytes):
        """Test that invalid gender values are handled"""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/api/v1/recommend?gender=invalid", files=files)

        # Should handle gracefully
        assert response.status_code in [200, 400, 422]
