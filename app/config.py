"""
Configuration settings for the API
"""

import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # API Configuration
    API_TITLE: str = "Fashion Recommendation API"
    API_VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
    ]

    # Model Configuration
    MODEL_DEVICE: str = "cpu"  # "cpu" or "cuda"
    POSE_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_CONFIDENCE_THRESHOLD: float = 0.3
    NMS_IOU_THRESHOLD: float = 0.45

    # Feature Extraction
    FEATURE_DIM: int = 2048
    RESNET_INPUT_SIZE: int = 224

    # Similarity Search
    DEFAULT_K_NEIGHBORS: int = 70
    MAX_RECOMMENDATIONS: int = 30
    MIN_RECOMMENDATIONS: int = 5

    # File Configuration
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png"]
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    # Model Paths
    YOLO_WEIGHTS_PATH: str = "./models/best.onnx"
    FEATURES_PATH: str = "./data/features_normalized.pkl"
    METADATA_PATH: str = "./data/metadata.json"

    # Catalog image root — used by /product-image endpoint and build_catalog.py
    CATALOG_IMAGE_ROOT: str = ""

    # Annotated data root — used by evaluate.py
    ANNOTATED_DATA_ROOT: str = ""

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/api.log"

    # JWT Authentication
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "your-secret-key-change-in-production-min-32-chars"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Categories
    FASHION_CATEGORIES: List[str] = [
        "topwear",
        "bottomwear",
        "footwear",
        "eyewear",
        "handbag"
    ]

    GENDER_OPTIONS: List[str] = [
        "men",
        "women",
        "unisex",
        "n.a."
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
